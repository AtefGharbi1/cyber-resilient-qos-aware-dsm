"""
CQ-DSM v11 — Experiment Runner

Speed changes vs v10:
  SPEED: 7 methods per scenario run in parallel (ProcessPoolExecutor).
  SPEED: MPC re-solves every H_MPC=8 steps (true receding-horizon).
  SPEED: Tighter per-solve time limits (0.8 s HEMS, 2.0 s MPC, 3.0 s DA).
  SPEED: model.cuts=0 via attribute (no str_param; crash-proof on all platforms).
  SAFE:  n_workers=1 falls back to sequential (set via --workers 1 for debugging).
  SAFE:  Each worker creates its own Model object — no shared state.
  FIX:   All scenarios complete without assertion crash.

Usage:
  python run_experiments.py --quick              # seed 42, ~20-40 min
  python run_experiments.py --quick --workers 1  # seed 42, sequential (debug)
  python run_experiments.py                      # 5 seeds, ~2-4 h
"""
import json, os, time, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import *
from data_gen import (load_nyiso_price, make_nyiso_multiday,
                      make_outdoor_temp, make_prosumers, make_qos,
                      apply_fdi, apply_fdi_stealthy, apply_pma)
from simulator import simulate
from lstm_detector import SlidingWindowMLPDetector

OUTFILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_v11.json")
METHODS  = ["AU", "RS", "QA", "RC", "CC", "MPC", "CQ"]


# ── Per-method worker (top-level for pickling) ────────────────────────────────

def _method_worker(args):
    """Run one method for one scenario. Returns (method, result_dict)."""
    (method, prosumers, lam_true, lam_rep, L_true, L_rep,
     delta, pi_loss, p_atk, T_outdoor, seed, n_workers) = args
    r = simulate(method, prosumers, lam_true, lam_rep,
                 L_true, L_rep, delta, pi_loss, p_atk,
                 T_outdoor, seed=seed, n_workers=n_workers)
    return method, r


def run_scenario(tag, prosumers, lam_true, lam_rep, L_true, L_rep,
                 delta, pi_loss, p_atk, T_outdoor, seed,
                 n_method_workers, n_prosumer_workers):
    """
    Run all METHODS for one scenario.

    n_method_workers:   number of methods to run in parallel (up to 7).
    n_prosumer_workers: number of prosumers to solve in parallel per slot.

    Setting n_method_workers>1 AND n_prosumer_workers>1 can over-subscribe
    the CPU — use n_method_workers * n_prosumer_workers <= cpu_count().
    Recommended: n_method_workers=7, n_prosumer_workers=1 (method-level only).
    """
    results = {}
    t0_all  = time.time()

    if n_method_workers > 1:
        args_list = [
            (m, prosumers, lam_true, lam_rep, L_true, L_rep,
             delta, pi_loss, p_atk, T_outdoor, seed, n_prosumer_workers)
            for m in METHODS
        ]
        # Use ProcessPoolExecutor: each method gets its own process+CBC instance
        with ProcessPoolExecutor(max_workers=n_method_workers) as ex:
            futures = {ex.submit(_method_worker, a): a[0] for a in args_list}
            for fut in as_completed(futures):
                m, r = fut.result()
                r["elapsed_s"] = round(time.time() - t0_all, 1)
                results[m] = r
                print(f"    {m:5s}: E=${r['cost_energy']:.4f}  "
                      f"imb=${r['cost_imbalance']:.4f}  "
                      f"tot=${r['cost_total']:.4f}  "
                      f"comf={r['comfort_pen']:.2f}  "
                      f"[wall {r['elapsed_s']:.0f}s]")
    else:
        # Sequential fallback (for debugging or single-core machines)
        for m in METHODS:
            t0 = time.time()
            r  = simulate(m, prosumers, lam_true, lam_rep,
                          L_true, L_rep, delta, pi_loss, p_atk,
                          T_outdoor, seed=seed, n_workers=n_prosumer_workers)
            r["elapsed_s"] = round(time.time() - t0, 1)
            results[m] = r
            print(f"    {m:5s}: E=${r['cost_energy']:.4f}  "
                  f"imb=${r['cost_imbalance']:.4f}  "
                  f"tot=${r['cost_total']:.4f}  "
                  f"comf={r['comfort_pen']:.2f}  "
                  f"[{r['elapsed_s']:.0f}s]")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",   action="store_true",
                        help="1 seed only (fast validation)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--seeds",   type=int, nargs="+",
                        default=[42, 7, 13, 99, 2024])
    parser.add_argument("--workers", type=int, default=7,
                        help="Number of parallel method workers (default=7). "
                             "Set to 1 for sequential/debug mode.")
    args   = parser.parse_args()
    seeds  = [args.seed] if args.quick else args.seeds

    # n_method_workers: run up to 7 methods in parallel
    # n_prosumer_workers: 1 (methods already parallel; avoid over-subscription)
    n_method_workers   = max(1, min(args.workers, len(METHODS)))
    n_prosumer_workers = 1   # change to >1 only if running sequential methods

    print("=" * 68)
    print("  CQ-DSM v11 — Experiment Runner")
    print(f"  Seeds: {seeds}  |  Methods: {METHODS}")
    print(f"  Quick: {args.quick}  |  Output: {OUTFILE}")
    print(f"  Key config: T_LOW={T_LOW} T_HIGH={T_HIGH}  "
          f"PI_MAX={PI_MAX}  TAU_MAX={TAU_MAX:.1f}s")
    print(f"  LAT_MU_DEG={LAT_MU_DEG}s  H_MPC={H_MPC}  "
          f"FALLBACK_MARGIN={FALLBACK_MARGIN}")
    print(f"  Parallelism: {n_method_workers} method workers × "
          f"{n_prosumer_workers} prosumer workers")
    print(f"  Time limits: HEMS={HEMS_TIME_LIMIT}s  "
          f"MPC={MPC_TIME_LIMIT}s  DA={DA_TIME_LIMIT}s")
    print("=" * 68)

    t_start = time.time()

    # ── Train detector (once, shared) ─────────────────────────────────────────
    print("\n[0] Training anomaly detector …")
    det = SlidingWindowMLPDetector(seed=SEED)
    det.train(n_days=60, seed=SEED)
    lstm_stats = {
        "model":      "SlidingWindowMLPDetector v11",
        "AUC":        round(det.auc,  4),
        "FPR":        round(det.fpr,  3),
        "FNR":        round(det.fnr,  3),
        "ECE_before": round(det.ece_before, 4),
        "ECE_after":  round(det.ece_after,  4),
        "T_scale":    round(det.ts.T_scale, 4),
    }
    print(f"  AUC={det.auc:.4f}  FPR={det.fpr:.3f}  FNR={det.fnr:.3f}")

    # ── Multi-day price traces ────────────────────────────────────────────────
    print("\n[1] Loading multi-day price traces …")
    multiday_lam = make_nyiso_multiday()
    lam_baseline = load_nyiso_price()
    print(f"  {len(multiday_lam)} day types: {list(multiday_lam.keys())}")

    all_results = {}

    # ── Per-seed simulation ───────────────────────────────────────────────────
    for seed in seeds:
        print(f"\n{'─' * 60}")
        print(f"[Seed {seed}]")

        lam_true  = load_nyiso_price()
        T_outdoor = make_outdoor_temp("peak_summer")
        prosumers = make_prosumers(seed=seed, pv_noise=True)

        L_base = prosumers["L_base"].sum(0)
        G_sol  = prosumers["G_solar"].sum(0)
        L_true = np.maximum(L_base - G_sol, 0)

        seed_res = {}

        # ── Normal ───────────────────────────────────────────────────────────
        print("\n  [normal]")
        delta_n, pi_n = make_qos(degraded=False, seed=seed)
        p_atk_n = det.predict(L_true, lam_true, delta_n, pi_n)
        seed_res["normal"] = run_scenario(
            "normal", prosumers, lam_true, lam_true,
            L_true, L_true, delta_n, pi_n, p_atk_n, T_outdoor, seed,
            n_method_workers, n_prosumer_workers)

        # ── FDI ──────────────────────────────────────────────────────────────
        print("\n  [fdi]")
        L_rep_fdi = apply_fdi(L_true, seed=seed)
        p_atk_fdi = det.predict(L_rep_fdi, lam_true, delta_n, pi_n)
        p_atk_fdi[FDI_START:FDI_END] = np.maximum(
            p_atk_fdi[FDI_START:FDI_END], P_ATK_HIGH)
        seed_res["fdi"] = run_scenario(
            "fdi", prosumers, lam_true, lam_true,
            L_true, L_rep_fdi, delta_n, pi_n, p_atk_fdi, T_outdoor, seed,
            n_method_workers, n_prosumer_workers)

        # ── Stealthy FDI ──────────────────────────────────────────────────────
        print("\n  [fdi_stealthy]")
        L_rep_stealthy = apply_fdi_stealthy(L_true, seed=seed)
        p_atk_stealthy = det.predict(L_rep_stealthy, lam_true, delta_n, pi_n)
        seed_res["fdi_stealthy"] = run_scenario(
            "fdi_stealthy", prosumers, lam_true, lam_true,
            L_true, L_rep_stealthy, delta_n, pi_n, p_atk_stealthy,
            T_outdoor, seed,
            n_method_workers, n_prosumer_workers)

        # ── PMA-up / PMA-dn ──────────────────────────────────────────────────
        for direction in ("up", "dn"):
            tag = f"pma_{direction}"
            print(f"\n  [{tag}]")
            lam_pma = apply_pma(lam_true,
                                direction="up" if direction == "up" else "down",
                                seed=seed)
            p_atk_pma = det.predict(L_true, lam_pma, delta_n, pi_n)
            p_atk_pma[PMA_START:PMA_END] = np.maximum(
                p_atk_pma[PMA_START:PMA_END], P_ATK_HIGH)
            seed_res[tag] = run_scenario(
                tag, prosumers, lam_true, lam_pma,
                L_true, L_true, delta_n, pi_n, p_atk_pma, T_outdoor, seed,
                n_method_workers, n_prosumer_workers)

        # ── QoS degraded ─────────────────────────────────────────────────────
        print("\n  [qos_deg]")
        delta_d, pi_d = make_qos(degraded=True, seed=seed)
        p_atk_qos = det.predict(L_true, lam_true, delta_d, pi_d)
        n_bad = int((pi_d > PI_MAX).sum())
        print(f"    QoS bad slots: {n_bad}/{T} "
              f"({100*n_bad/T:.1f}%)  "
              f"pi_mean={pi_d.mean():.3f}  "
              f"delta_mean={delta_d.mean():.2f}s")
        seed_res["qos_deg"] = run_scenario(
            "qos_deg", prosumers, lam_true, lam_true,
            L_true, L_true, delta_d, pi_d, p_atk_qos, T_outdoor, seed,
            n_method_workers, n_prosumer_workers)

        all_results[f"seed_{seed}"] = seed_res

    # ── Multi-day FDI evaluation ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("[Multi-day] Evaluating FDI across 7 representative day types …")
    pros42  = make_prosumers(seed=42, pv_noise=True)
    L42     = np.maximum(pros42["L_base"].sum(0) - pros42["G_solar"].sum(0), 0)
    d42, pi42 = make_qos(degraded=False, seed=42)
    L_rep_fdi42 = apply_fdi(L42, seed=42)

    multiday_results = {}
    for day_name, lam_day in multiday_lam.items():
        print(f"  Day: {day_name} (scale={DAY_SCALES[day_name]:.2f}x)")
        T_out_d  = make_outdoor_temp(day_name)
        p_atk_d  = det.predict(L_rep_fdi42, lam_day, d42, pi42)
        p_atk_d[FDI_START:FDI_END] = np.maximum(
            p_atk_d[FDI_START:FDI_END], P_ATK_HIGH)
        day_res = run_scenario(
            day_name, pros42, lam_day, lam_day,
            L42, L_rep_fdi42, d42, pi42, p_atk_d, T_out_d, 42,
            n_method_workers, n_prosumer_workers)
        multiday_results[day_name] = {
            m: {k: round(v, 4) for k, v in day_res[m].items()
                if isinstance(v, float)}
            for m in METHODS
        }

    # Multi-day stats
    multiday_stats = {}
    for m in METHODS:
        tots  = [multiday_results[d][m]["cost_total"]  for d in multiday_results]
        comfs = [multiday_results[d][m]["comfort_pen"] for d in multiday_results]
        multiday_stats[m] = {
            "cost_total_mean": round(float(np.mean(tots)),  4),
            "cost_total_std":  round(float(np.std(tots)),   4),
            "comfort_mean":    round(float(np.mean(comfs)), 2),
            "comfort_std":     round(float(np.std(comfs)),  2),
        }

    print("\n  Multi-day FDI summary (cost_total mean ± std):")
    for m in METHODS:
        s = multiday_stats[m]
        print(f"    {m:5s}: tot={s['cost_total_mean']:.4f}"
              f"±{s['cost_total_std']:.4f}  "
              f"comf={s['comfort_mean']:.2f}±{s['comfort_std']:.2f}")

    # ── Aggregate across seeds ────────────────────────────────────────────────
    scenarios  = ["normal", "fdi", "fdi_stealthy", "pma_up", "pma_dn", "qos_deg"]
    aggregated = {sc: {} for sc in scenarios}
    for sc in scenarios:
        for m in METHODS:
            vals = [all_results[f"seed_{s}"][sc][m]
                    for s in seeds
                    if (f"seed_{s}" in all_results
                        and sc in all_results[f"seed_{s}"])]
            if not vals:
                continue
            agg_m = dict(vals[0])
            for key in ["cost_energy", "cost_imbalance", "cost_total",
                        "comfort_pen", "cost_qos", "cost_cyber", "PAR"]:
                if key in vals[0]:
                    arr = [v[key] for v in vals]
                    agg_m[f"{key}_mean"] = round(float(np.mean(arr)), 4)
                    agg_m[f"{key}_std"]  = round(float(np.std(arr)),  4)
            aggregated[sc][m] = agg_m

    # ── Honest results summary ────────────────────────────────────────────────
    total_s = round(time.time() - t_start, 1)
    print(f"\n{'=' * 68}")
    print("  HONEST RESULTS SUMMARY (seed 42)")
    print(f"{'=' * 68}")
    s42 = all_results["seed_42"]
    au_n_tot = s42["normal"]["AU"]["cost_total"]
    for sc in scenarios:
        print(f"\n  [{sc}]")
        print(f"  {'M':4}  {'tot/P':>9}  {'Δtot%':>9}  {'comf':>8}")
        for m in METHODS:
            if m not in s42[sc]:
                continue
            tot  = s42[sc][m]["cost_total"]
            comf = s42[sc][m]["comfort_pen"]
            dpct = (tot - au_n_tot) / au_n_tot * 100
            mark = " ◀" if m == "CQ" else ""
            print(f"  {m:4}  {tot/P:>9.4f}  {dpct:>+8.3f}%  {comf:>8.2f}{mark}")

    print(f"\n  Detector: AUC={det.auc:.4f}  FPR={det.fpr:.3f}  FNR={det.fnr:.3f}")
    print(f"  Total runtime: {total_s:.1f}s = {total_s/60:.1f} min")

    # ── Save ─────────────────────────────────────────────────────────────────
    output = {
        "format_version": 11,
        "generated_by":   "run_experiments.py v11",
        "seeds":          seeds,
        "methods":        METHODS,
        "config_summary": {
            "T_LOW": T_LOW, "T_HIGH": T_HIGH,
            "PI_MAX": PI_MAX, "TAU_MAX": TAU_MAX,
            "LAT_MU_DEG": LAT_MU_DEG, "P01": P01,
            "FALLBACK_MARGIN": FALLBACK_MARGIN,
            "H_MPC": H_MPC, "K_IMBALANCE": K_IMBALANCE,
            "HEMS_TIME_LIMIT": HEMS_TIME_LIMIT,
            "MPC_TIME_LIMIT":  MPC_TIME_LIMIT,
            "DA_TIME_LIMIT":   DA_TIME_LIMIT,
            "n_method_workers": n_method_workers,
        },
        "lstm_stats":     lstm_stats,
        "run_metadata":   {"runtime_seconds": total_s, "n_seeds": len(seeds)},
        "multiday_stats": multiday_stats,
        "multiday_raw":   multiday_results,
        "aggregated":     aggregated,
    }
    for s in seeds:
        output[f"seed_{s}"] = all_results[f"seed_{s}"]

    if os.path.exists(OUTFILE):
        os.remove(OUTFILE)
    with open(OUTFILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Saved {OUTFILE}")

    with open(OUTFILE) as f:
        chk = json.load(f)
    print(f"  format_version={chk['format_version']}  "
          f"scenarios={list(chk['aggregated'].keys())}  "
          f"methods={chk['methods']}")


if __name__ == "__main__":
    main()
