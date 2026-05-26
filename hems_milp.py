"""
CQ-DSM HEMS MILP v11
Speed changes vs v10:
  SPEED: _new_model() sets model.cuts=0 directly (no str_param; reliable on all platforms)
  SPEED: HEMS_TIME_LIMIT 1.2→0.8 s (from config)
  SPEED: DA_TIME_LIMIT   5.0→3.0 s (from config)
  SPEED: MPC_TIME_LIMIT  2.0 s (from config)
  SPEED: model.threads=1 preserved (avoids thread-spawn overhead for small models)
  FIX:   Fallback thermostat unchanged
  FIX:   All battery / QoS / comfort logic identical to v10
"""
import numpy as np
import mip
from config import *


def _thermostat_fallback(T_indoor_t, T_outdoor_h, alpha, beta, E_hvac, H):
    """Rule-based thermostat fallback with tighter comfort band."""
    yh = np.zeros(H, int)
    Ti = T_indoor_t
    lo = T_LOW  + FALLBACK_MARGIN
    hi = T_HIGH - FALLBACK_MARGIN
    for h in range(H):
        yh[h] = 1 if Ti > hi else 0
        Ti = (1 - alpha)*Ti + alpha*float(T_outdoor_h[h]) - beta*E_hvac*yh[h]*DT
    return yh


def _new_model(time_limit):
    """
    Create a CBC model.
    model.cuts = 0  → disables all cut generation:
      * prevents the CbcCountRowCut hash assertion crash on PMA scenarios
      * removes cut-generation overhead (worthwhile for small sub-problems)
    model.threads = 1 → no thread-spawn cost for tiny MILPs
    """
    mdl = mip.Model(solver_name=mip.CBC)
    mdl.verbose     = 0
    mdl.max_seconds = time_limit
    mdl.threads     = 1
    mdl.cuts        = 0   # ← replaces fragile str_param approach; always works
    return mdl


def solve_hems_rolling(t0, H, prosumer, T_indoor_t, lam_h, p_atk_h,
                       w2_adj, T_outdoor_h, include_qos, delta_h, pi_loss_h,
                       dw_done, wm_done, soc_t,
                       fallback=False, time_limit=None):
    """
    Rolling H-slot MILP for a single prosumer.
    fallback=True applies tighter comfort bounds (CQ QoS fallback mode).
    Returns (yh, ydw, ywm, p_ch_h, p_dis_h, feasible).
    """
    if time_limit is None:
        time_limit = HEMS_TIME_LIMIT

    alpha   = float(prosumer["alpha"])
    beta    = float(prosumer["beta"])
    E_h     = float(prosumer["E_hvac"])
    L_b     = prosumer["L_base"]
    G_s     = prosumer.get("G_solar_forecast", prosumer["G_solar"])
    t_dw    = int(prosumer["t_dw_ideal"])
    t_wm    = int(prosumer["t_wm_ideal"])
    has_bat = bool(prosumer.get("has_battery", False))

    mdl = _new_model(time_limit)

    y_hvac = [mdl.add_var(var_type=mip.BINARY) for _ in range(H)]
    y_dw   = [mdl.add_var(var_type=mip.BINARY) for _ in range(H)]
    y_wm   = [mdl.add_var(var_type=mip.BINARY) for _ in range(H)]
    s_temp = [mdl.add_var(lb=0)                for _ in range(H)]
    T_in   = [mdl.add_var(lb=T_LOW-10, ub=T_HIGH+10) for _ in range(H+1)]

    if has_bat:
        P_ch  = [mdl.add_var(lb=0, ub=BAT_P_MAX) for _ in range(H)]
        P_dis = [mdl.add_var(lb=0, ub=BAT_P_MAX) for _ in range(H)]
        SOC   = [mdl.add_var(lb=BAT_SOC_MIN*BAT_CAP,
                              ub=BAT_SOC_MAX*BAT_CAP) for _ in range(H+1)]
        u_ch  = [mdl.add_var(var_type=mip.BINARY) for _ in range(H)]
        mdl  += SOC[0] == soc_t
        for h in range(H):
            mdl += SOC[h+1] == (SOC[h]
                                 + BAT_ETA_CH * P_ch[h] * DT
                                 - P_dis[h] / BAT_ETA_DIS * DT)
            mdl += P_ch[h]  <= BAT_P_MAX * u_ch[h]
            mdl += P_dis[h] <= BAT_P_MAX * (1 - u_ch[h])
    else:
        P_ch = P_dis = [0.0] * H

    mdl += T_in[0] == T_indoor_t
    for h in range(H):
        mdl += (T_in[h+1] == (1-alpha)*T_in[h]
                + alpha*float(T_outdoor_h[h])
                - beta*E_h*DT*y_hvac[h])

    lo = T_LOW  + (FALLBACK_MARGIN if fallback else 0)
    hi = T_HIGH - (FALLBACK_MARGIN if fallback else 0)
    for h in range(H):
        mdl += s_temp[h] >= T_LOW  - T_in[h]
        mdl += s_temp[h] >= T_in[h] - T_HIGH
        if fallback:
            mdl += T_in[h] >= lo
            mdl += T_in[h] <= hi

    for h in range(H):
        if dw_done or (t0+h) < DW_WIN[0] or (t0+h) >= DW_WIN[1]:
            mdl += y_dw[h] == 0
        if wm_done or (t0+h) < WM_WIN[0] or (t0+h) >= WM_WIN[1]:
            mdl += y_wm[h] == 0
    if not dw_done:
        mdl += mip.xsum(y_dw[h] for h in range(H)) <= DUR_DW
    if not wm_done:
        mdl += mip.xsum(y_wm[h] for h in range(H)) <= DUR_WM

    def net_load(h):
        t  = t0 + h
        Lb = float(L_b[t]) if t < len(L_b) else 0.0
        Gs = float(G_s[t]) if t < len(G_s) else 0.0
        ch  = P_ch[h]  if has_bat else 0.0
        dis = P_dis[h] if has_bat else 0.0
        return Lb + E_h*y_hvac[h] + E_DW*y_dw[h] + E_WM*y_wm[h] + ch - dis - Gs

    obj_energy  = mip.xsum(float(lam_h[h]) * net_load(h) * DT for h in range(H))
    obj_comfort = mip.xsum(
        MU_P * s_temp[h]
        + NU_P * (abs(t0+h - t_dw)*y_dw[h] + abs(t0+h - t_wm)*y_wm[h])
        for h in range(H))
    obj_cyber = mip.xsum(
        float(p_atk_h[h]) * float(lam_h[h]) * net_load(h) * DT for h in range(H))
    obj_qos = (mip.xsum(
        (KAPPA_DELTA*float(delta_h[h]) + KAPPA_PI*float(pi_loss_h[h]))
        * net_load(h) * DT for h in range(H)) if include_qos else 0)
    obj_bat_deg = 0
    if has_bat:
        obj_bat_deg = BAT_COST_DEG * mip.xsum(
            (P_ch[h] + P_dis[h]) * DT for h in range(H))

    mdl.objective = mip.minimize(
        W1 * obj_energy
        + w2_adj * obj_comfort
        + W4 * obj_cyber
        + (W3 * obj_qos if include_qos else 0)
        + obj_bat_deg)

    mdl.optimize()

    if mdl.num_solutions > 0:
        yh   = np.round([y_hvac[h].x for h in range(H)]).astype(int)
        ydw  = np.round([y_dw[h].x   for h in range(H)]).astype(int)
        ywm  = np.round([y_wm[h].x   for h in range(H)]).astype(int)
        if has_bat:
            pch  = np.array([float(P_ch[h].x)  for h in range(H)])
            pdis = np.array([float(P_dis[h].x) for h in range(H)])
        else:
            pch = pdis = np.zeros(H)
        feas = True
    else:
        yh   = _thermostat_fallback(T_indoor_t, T_outdoor_h, alpha, beta, E_h, H)
        ydw  = np.zeros(H, int)
        ywm  = np.zeros(H, int)
        pch  = pdis = np.zeros(H)
        feas = False

    return yh, ydw, ywm, pch, pdis, feas


def solve_appliance_dayahead(prosumer, lam_reported, p_atk, w2_adj,
                              T_outdoor, time_limit=None):
    """Day-ahead appliance scheduling (dishwasher + washer, full T horizon)."""
    if time_limit is None:
        time_limit = DA_TIME_LIMIT

    E_h  = float(prosumer["E_hvac"])
    L_b  = prosumer["L_base"]
    G_s  = prosumer.get("G_solar_forecast", prosumer["G_solar"])
    t_dw = int(prosumer["t_dw_ideal"])
    t_wm = int(prosumer["t_wm_ideal"])

    mdl = _new_model(time_limit)

    y_dw = [mdl.add_var(var_type=mip.BINARY) for _ in range(T)]
    y_wm = [mdl.add_var(var_type=mip.BINARY) for _ in range(T)]

    def _add_consec(m2, y_app, win_lo, win_hi, dur):
        m2 += mip.xsum(y_app[t] for t in range(T)) == dur
        for t in range(T):
            if t < win_lo or t >= win_hi:
                m2 += y_app[t] == 0
        z = [m2.add_var(var_type=mip.BINARY) for _ in range(T)]
        m2 += z[0] == y_app[0]
        for t in range(1, T):
            m2 += z[t] >= y_app[t] - y_app[t-1]
            m2 += z[t] <= y_app[t]
            m2 += z[t] <= 1 - y_app[t-1]
        m2 += mip.xsum(z[t] for t in range(T)) == 1

    _add_consec(mdl, y_dw, DW_WIN[0], DW_WIN[1], DUR_DW)
    _add_consec(mdl, y_wm, WM_WIN[0], WM_WIN[1], DUR_WM)

    obj = mip.xsum(
        float(lam_reported[t]) * (E_DW*y_dw[t] + E_WM*y_wm[t]) * DT
        + w2_adj * NU_P * (abs(t-t_dw)*y_dw[t] + abs(t-t_wm)*y_wm[t])
        + W4 * float(p_atk[t]) * float(lam_reported[t])
          * (E_DW*y_dw[t] + E_WM*y_wm[t]) * DT
        for t in range(T))
    mdl.objective = mip.minimize(obj)
    mdl.optimize()

    if mdl.num_solutions > 0:
        ydw = np.round([y_dw[t].x for t in range(T)]).astype(int)
        ywm = np.round([y_wm[t].x for t in range(T)]).astype(int)
    else:
        ydw = np.zeros(T, int); ydw[t_dw:t_dw+DUR_DW] = 1
        ywm = np.zeros(T, int); ywm[t_wm:t_wm+DUR_WM] = 1

    return ydw, ywm
