"""
CQ-DSM — Data Generation v10
Changes vs v9:
  FIX: make_qos degraded state uses new LAT_MU_DEG=5.0, P01=0.04, PI_MAX=0.25
  FIX: apply_fdi uses one-sided upward injection only (consistent with paper)
  FIX: apply_fdi_stealthy added for the stealthy FDI experiment
  PRESERVED: all other logic from v9 (multiday, PV noise, battery fields)
"""
import numpy as np
import pandas as pd
from config import *


def load_nyiso_price(csv_path=None):
    if csv_path is None:
        import os
        csv_path = os.path.join(os.path.dirname(__file__), NYISO_CSV)
    df = pd.read_csv(csv_path)
    lbmp_hourly = df["LBMP ($/MWHr)"].values.astype(float)
    slots  = np.linspace(0, 23, T)
    lbmp5m = np.interp(slots, np.arange(24), lbmp_hourly)
    return lbmp5m / 1000.0   # convert to $/kWh


def make_nyiso_multiday(csv_path=None):
    """7 representative day types scaled from the Aug-15 baseline."""
    base = load_nyiso_price(csv_path)
    rng  = np.random.default_rng(SEED + 99)
    out  = {}
    for name, scale in DAY_SCALES.items():
        noise = rng.normal(0, 0.03 * scale, T)
        out[name] = np.clip(base * scale + noise / 1000.0, 0.010, 0.300)
    return out


def make_outdoor_temp(day_name="peak_summer"):
    """NOAA JFK climatology by day type."""
    t_hourly = np.array([
        27.2, 26.8, 26.4, 26.1, 25.9, 25.8,
        26.3, 27.4, 28.8, 30.1, 31.2, 32.0,
        32.6, 33.1, 33.3, 33.2, 32.7, 31.9,
        30.9, 30.0, 29.3, 28.7, 28.2, 27.8,
    ])
    if day_name == "winter_peak":
        t_hourly -= 25.0
    elif day_name in ("shoulder", "low_price"):
        t_hourly -= 12.0
    return np.interp(np.linspace(0, 23, T), np.arange(24), t_hourly)


def make_pv_profile(seed=SEED, noise=False):
    """PV profile with optional forecast uncertainty."""
    rng   = np.random.default_rng(seed + 1)
    solar = np.zeros(T)
    dawn, dusk = 72, 234
    mask  = np.arange(T)
    mask  = (mask >= dawn) & (mask <= dusk)
    t     = np.arange(T)
    solar[mask] = np.sin(np.pi * (t[mask] - dawn) / (dusk - dawn))
    solar = np.clip(solar + rng.normal(0, 0.02, T), 0, 1)
    pv_actual = PV_PEAK * solar
    if noise:
        err = rng.normal(0, PV_NOISE_STD, T) * pv_actual
        pv_forecast = np.clip(pv_actual + err, 0, PV_PEAK)
    else:
        pv_forecast = pv_actual.copy()
    return pv_actual, pv_forecast


def make_prosumers(seed=SEED, pv_noise=False):
    """Generate P prosumer parameter sets."""
    rng     = np.random.default_rng(seed + 2)
    has_pv  = rng.random(P) < PV_FRAC
    has_bat = rng.random(P) < BAT_FRAC
    E_hvac  = rng.uniform(E_HVAC_LO, E_HVAC_HI, P)
    alpha   = rng.uniform(ALPHA_LO,  ALPHA_HI,  P)
    beta    = rng.uniform(BETA_LO,   BETA_HI,   P)
    L_base  = np.clip(rng.normal(BASE_MU, BASE_STD, (P, T)), 0.4, 3.5)
    pv_a, pv_f = make_pv_profile(seed, noise=pv_noise)
    G_solar_actual   = has_pv[:, None] * pv_a[None, :]
    G_solar_forecast = has_pv[:, None] * pv_f[None, :]
    t_dw    = rng.integers(DW_WIN[0] + DUR_DW, DW_WIN[1] - DUR_DW, P)
    t_wm    = rng.integers(WM_WIN[0] + DUR_WM, WM_WIN[1] - DUR_WM, P)
    soc_init = np.where(has_bat, BAT_SOC_INIT, 0.0)
    return dict(
        has_pv=has_pv, has_battery=has_bat,
        E_hvac=E_hvac, alpha=alpha, beta=beta,
        L_base=L_base,
        G_solar=G_solar_actual,
        G_solar_forecast=G_solar_forecast,
        t_dw_ideal=t_dw, t_wm_ideal=t_wm,
        soc_init=soc_init,
    )


def make_qos(degraded=False, seed=SEED):
    """
    Generate per-slot latency δ(t) and packet loss π_loss(t).
    Degraded state uses a two-state Markov chain with recalibrated parameters:
      - LAT_MU_DEG = 5.0 s (was 2.0 s) → more severe latency
      - P01 = 0.04 (was 0.10) → longer good-state runs
      - PI_MAX = 0.25 (threshold) → fallback triggers more deliberately
    This ensures sustained bad bursts that actually disrupt unaware methods.
    """
    rng     = np.random.default_rng(seed + 3)
    delta   = np.full(T, LAT_NOM, float)
    pi_loss = np.full(T, PI_NOM,  float)
    if degraded:
        state = 0  # 0 = good, 1 = bad
        for t in range(T):
            if state == 0:
                pi_loss[t] = rng.uniform(0.01, 0.06)
                delta[t]   = rng.lognormal(np.log(0.2), 0.3)
                if rng.random() < P01:
                    state = 1
            else:
                pi_loss[t] = rng.uniform(0.28, 0.50)   # above PI_MAX=0.25
                delta[t]   = rng.lognormal(np.log(LAT_MU_DEG), 0.4)
                if rng.random() < P10:
                    state = 0
    return delta, pi_loss


def apply_fdi(L_true, seed=SEED):
    """
    Standard FDI: one-sided upward injection.
    L_reported = L_true + e_FDI, e_FDI ~ U(FDI_MAG/2, FDI_MAG) * L_true
    in [FDI_START, FDI_END). Upward only: AU/RS procure more than needed.
    """
    rng = np.random.default_rng(seed + 5)
    e   = np.zeros(T)
    n   = FDI_END - FDI_START
    e[FDI_START:FDI_END] = (rng.uniform(FDI_MAG / 2, FDI_MAG, n)
                             * L_true[FDI_START:FDI_END])
    return L_true + e


def apply_fdi_stealthy(L_true, seed=SEED):
    """
    Stealthy FDI: low-magnitude injection designed to stay below detector threshold.
    Amplitude = FDI_MAG_STEALTHY = 0.06 (30% of standard).
    """
    rng = np.random.default_rng(seed + 55)
    e   = np.zeros(T)
    n   = FDI_END - FDI_START
    e[FDI_START:FDI_END] = (rng.uniform(FDI_MAG_STEALTHY / 2, FDI_MAG_STEALTHY, n)
                             * L_true[FDI_START:FDI_END])
    return L_true + e


def apply_pma(lam, direction="up", seed=SEED):
    """PMA: multiplicative price distortion in [PMA_START, PMA_END)."""
    rng = np.random.default_rng(seed + 6)
    eps = np.zeros(T)
    n   = PMA_END - PMA_START
    if direction == "up":
        eps[PMA_START:PMA_END] =  rng.uniform(0, PMA_UP_MAX, n)
    else:
        eps[PMA_START:PMA_END] = -rng.uniform(0, PMA_DN_MAX, n)
    return lam * (1 + eps)
