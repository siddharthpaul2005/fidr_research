"""
Remaining analyses for the paper:

 1. LEAVE-ONE-OUT ABLATION - remove one detector at a time from the fused
    system (the remaining two must then BOTH agree), remove the arc detector,
    and remove the vote (any detector fires = alarm).
 2. DATASET STATISTICS - what the generated telemetry dataset contains:
    sizes, per-channel healthy statistics, recycle rates, fault parameters.

Output: extra_analyses2.json + printed tables.
"""
import json, time
import numpy as np
import thruster_sim as ts
import fdir as fd
import run_experiments as rx
from extra_analyses import gen_raw, summarise

N = rx.N
BIG = 10**9


def t(x):
    return np.where(x < 0, BIG, x)


def back(x):
    return np.where(x >= BIG, -1, x)


def main():
    t_start = time.time()
    R = json.load(open("results.json"))
    thr = R["thresholds"]
    out = {}

    # ---------------- 1. leave-one-out ablation
    print("1) leave-one-out ablation: running detectors on the test set")
    test = rx.run_set(rx.N_TEST, seed=2)
    def variant(a, name):
        C, K, B, A = t(a["CUSUM"]), t(a["KF"]), t(a["BOCPD"]), t(a["ARC"])
        if name == "full system (2-of-3 + arc)":
            return back(np.minimum(np.sort(np.stack([C, K, B]), 0)[1], A))
        if name == "without CUSUM (Kalman AND BOCPD)":
            return back(np.minimum(np.maximum(K, B), A))
        if name == "without Kalman (CUSUM AND BOCPD)":
            return back(np.minimum(np.maximum(C, B), A))
        if name == "without BOCPD (CUSUM AND Kalman)":
            return back(np.minimum(np.maximum(C, K), A))
        if name == "without arc detector":
            return back(np.sort(np.stack([C, K, B]), 0)[1])
        if name == "without vote (any one fires)":
            return back(np.min(np.stack([C, K, B, A]), 0))
    names = ["full system (2-of-3 + arc)", "without CUSUM (Kalman AND BOCPD)",
             "without Kalman (CUSUM AND BOCPD)", "without BOCPD (CUSUM AND Kalman)",
             "without arc detector", "without vote (any one fires)"]
    abl = {}
    for name in names:
        per, fa, expo = {}, 0, 0.0
        for d in test:
            al = variant(rx.alarms(d, thr), name)
            t0 = d["p"]["t0"]
            fa += int(((al >= 0) & (al < t0)).sum())
            end = np.minimum(np.where(np.isinf(t0), N, t0), np.where(al >= 0, al, N))
            expo += float(np.clip(end - fd.CAL_N, 0, None).sum() / 60)
            if d["fault"] != "NOM":
                per[d["fault"]] = summarise(al, t0, d["fault"])
        abl[name] = dict(per_fault=per, false_alarms=fa, exposure_h=expo)
    out["leave_one_out"] = abl
    del test

    # ---------------- 2. dataset statistics
    print("2) dataset statistics")
    stats = dict(frame_period_s=60, run_length_h=N / 60, frames_per_run=N,
                 channels=["Ib_mA", "Ia_mA", "Prf_W", "mdot_ugs", "Vn_V", "T_C", "n_recycle"])
    sets = {"training": (rx.N_TRAIN, 1), "test": (rx.N_TEST, 2)}
    total_runs = 0
    for sname, (counts, seed) in sets.items():
        n = sum(counts.values()); total_runs += n
        stats[f"{sname}_runs"] = counts
    stats["total_runs"] = total_runs
    stats["total_hours"] = total_runs * N / 60
    stats["total_frames"] = total_runs * N
    ch = {}
    fault_par = {}
    rec_nom, hours_nom = 0, 0.0
    for fault, p, y, narc in gen_raw(rx.N_TEST, seed=2):
        if fault == "NOM":
            ok = narc == 0
            for c, nm in enumerate(["Ib_mA", "Ia_mA", "Prf_W", "mdot_ugs", "Vn_V"]):
                v = y[:, c][ok]
                ch[nm] = dict(mean=float(v.mean()), std=float(v.std()),
                              min=float(v.min()), max=float(v.max()),
                              noise_sigma_model=float(ts.SIGMA[c]))
            rec_nom = int(narc.sum()); hours_nom = narc.size / 60
        else:
            fault_par[fault] = dict(onset_h_range=[float(p["t0"].min() / 60), float(p["t0"].max() / 60)],
                                    growth_h_range=[float(p["tau"].min() / 60), float(p["tau"].max() / 60)],
                                    severity_range=[float(p["sev"].min()), float(p["sev"].max())],
                                    recycles_total=int(narc.sum()))
        print(f"   {fault} done", flush=True)
    rng = np.random.default_rng(5)
    Tp = ts.temperature(100, N, rng)
    ch["T_C"] = dict(mean=float(Tp.mean()), std=float(Tp.std()), min=float(Tp.min()), max=float(Tp.max()))
    stats["healthy_channel_stats_test_set"] = ch
    stats["healthy_recycles_per_1000h"] = 1000 * rec_nom / hours_nom
    stats["fault_parameters_test_set"] = fault_par
    out["dataset"] = stats
    out["runtime_s"] = time.time() - t_start
    json.dump(out, open("extra_analyses2.json", "w"), indent=1)

    # ---------------- print
    flts = fd.CLASSES
    print("\nLEAVE-ONE-OUT ABLATION: detection rate | median delay   (FA = false alarms)")
    print(f"{'variant':36s}" + "".join(f"{f:>14s}" for f in flts) + f"{'FA':>6s}")
    for name, r in abl.items():
        cells = []
        for f in flts:
            x = r["per_fault"][f]
            dl = "-" if x["delay_med_h"] is None else f"{x['delay_med_h']:.1f}"
            cells.append(f"{100*x['pd']:.0f}%|{dl}h")
        print(f"{name:36s}" + "".join(f"{c:>14s}" for c in cells) + f"{r['false_alarms']:>6d}")
    print("\nDATASET")
    print(f"  runs: {stats['total_runs']}  hours: {stats['total_hours']:,.0f}  frames: {stats['total_frames']:,}")
    print("  healthy telemetry (test set, recycle frames excluded):")
    for nm, s in ch.items():
        print(f"    {nm:9s} mean {s['mean']:8.3f}  std {s['std']:7.3f}  min {s['min']:8.3f}  max {s['max']:8.3f}")
    print(f"  healthy recycle rate: {stats['healthy_recycles_per_1000h']:.1f} per 1000 h")
    for f, fp in fault_par.items():
        print(f"  {f:5s} onset {fp['onset_h_range'][0]:.0f}-{fp['onset_h_range'][1]:.0f} h, "
              f"growth {fp['growth_h_range'][0]:.0f}-{fp['growth_h_range'][1]:.0f} h, "
              f"severity {fp['severity_range'][0]:.3g}-{fp['severity_range'][1]:.3g}")
    print(f"\nruntime {out['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()