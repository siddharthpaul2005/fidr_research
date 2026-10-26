"""
Ground-in-the-loop vs onboard FDIR comparison.

Same 500 test fault runs as run_experiments.py (seed 2). For every run we
simulate the thruster three ways and measure the damage done after onset:

  no action  : nobody fixes anything
  onboard    : our FDIR acts autonomously 60 min after its own alarm,
               using the int8 classifier's diagnosis
  ground     : the same detectors run on the ground (generous assumption),
               but the satellite is only in contact during passes:
                 1. the alarm becomes visible at the first pass after it,
                 2. operators need D hours to analyse it (and are always right),
                 3. the command is uplinked at the first pass after that.

Ground scenarios (pass interval T, analysis delay D) are in SCENARIOS.
Output: ground_vs_onboard.json + a printed summary table.
Run with argument "fast" for faults that grow 10x faster -> ground_vs_onboard_fast.json
"""
import json, time, sys
import numpy as np
import thruster_sim as ts
import fdir as fd
import run_experiments as rx

SCENARIOS = {                       # name: (pass interval [h], analysis delay [h])
    "ground_fast":   (6.0, 2.0),    # 4 passes/day, quick response
    "ground_typical": (12.0, 8.0),  # 2 passes/day, same-day analysis
    "ground_slow":   (24.0, 24.0),  # 1 useful pass/day, next-day response
}
CLASSES = fd.CLASSES                 # GRID, CATH, ARC, FEED, RF
N, H = rx.N, rx.N // 60


def next_pass(t_min, phase_min, T_min):
    """first pass time >= t (minutes)."""
    k = np.ceil((t_min - phase_min) / T_min)
    return phase_min + np.maximum(k, 0) * T_min


T_NOM_mN = ts.IB_SET * 1e-3 * np.sqrt(2 * ts.M_I * ts.VS / ts.E) * 1e3   # nominal thrust


def impulse_lost(truth, t0):
    """total impulse lost after onset [N s] vs nominal thrust (thrust drop from
    backstreaming, recycles not included, and from any throttle-down)."""
    after_m = np.arange(N)[None, :] >= t0[:, None]
    return ((T_NOM_mN - truth["thrust_mN"]) * 1e-3 * 60.0 * after_m).sum(1)


def damage(fault, y, narc, truth, t0):
    """fault-specific 'out of spec' measure after onset (per run)."""
    hours = np.arange(H)[None, :]
    after_h = hours >= np.floor(t0[:, None] / 60)
    after_m = np.arange(N)[None, :] >= t0[:, None]
    hm = y.reshape(y.shape[0], 5, H, 60).mean(-1)          # hourly means
    if fault == "GRID":   # hours with electron backstreaming
        ebs = truth["i_ebs"].reshape(-1, H, 60).mean(-1) > 0.01
        return (ebs & after_h).sum(1).astype(float), "hours with electron backstreaming"
    if fault == "CATH":   # hours with neutraliser voltage > 1 V above nominal
        bad = truth["vn_d"].reshape(-1, H, 60).mean(-1) > 1.0
        return (bad & after_h).sum(1).astype(float), "hours with V_n > 1 V above nominal"
    if fault == "ARC":    # recycles after onset
        return (narc * after_m).sum(1).astype(float), "recycles (arcs) after onset"
    if fault == "FEED":   # hours with flow deficit > 5 %
        bad = hm[:, 3] < 0.95 * ts.MDOT_SET
        return (bad & after_h).sum(1).astype(float), "hours with flow deficit > 5 %"
    if fault == "RF":     # hours above 40 W RF power
        bad = hm[:, 2] > 40.0
        return (bad & after_h).sum(1).astype(float), "hours above 40 W RF power"


def main():
    t_start = time.time()
    R = json.load(open("results.json"))
    thr = R["thresholds"]
    m = R["model"]
    qm = dict(Wq=np.array(m["Wq"]), bq=np.array(m["bq"]), mu=np.array(m["mu"]),
              sd=np.array(m["sd"]), in_scale=16.0)
    fast = len(sys.argv) > 1 and sys.argv[1] == "fast"
    tag = "_fast" if fast else ""
    if fast:   # faults develop 10x faster (hours instead of days)
        orig = ts.make_scenarios
        def ms(n_runs, fault, n, rng, onset_range=(50 * 60, 150 * 60)):
            p = orig(n_runs, fault, n, rng, onset_range)
            p["tau"] = p["tau"] * 0.1
            return p
        ts.make_scenarios = ms
        print("FAST-FAULT variant: growth time constants x 0.1")
    print("regenerating the test set (same seed as run_experiments.py)")
    test = rx.run_set(rx.N_TEST, seed=2)
    rng_ph = np.random.default_rng(123)
    out = {"scenarios": {k: dict(pass_interval_h=v[0], analysis_delay_h=v[1])
                         for k, v in SCENARIOS.items()}, "faults": {}}
    for d in test:
        f = d["fault"]
        if f == "NOM":
            continue
        a = rx.alarms(d, thr)["VOTE"]
        ok = a >= d["p"]["t0"]                       # detected runs (all, in practice)
        runs = np.where(ok)[0]
        a = a[runs]
        t0 = d["p"]["t0"][runs]
        t_iso = np.minimum(a + rx.ISO_WIN, N - 1)
        X = fd.features(d["kb"][runs], d["z"][runs], d["ac"][runs], t_iso)
        pred = fd.predict_q(qm, X)
        true_cls = CLASSES.index(f)
        sub = {k: v[runs] for k, v in d["p"].items()}
        Tp = d["Tp"][runs]
        phase = rng_ph.uniform(0, 1, len(runs))      # pass phase per run (fraction of interval)

        def sim(t_r, action):
            rng = np.random.default_rng(99)           # identical noise for every variant
            rec = None if t_r is None else dict(t_r=t_r.astype(float), action=action)
            y, narc, _, truth = ts.simulate(sub, N, rng, recovery=rec, Tp=Tp)
            dmg, lab = damage(f, y, narc, truth, t0)
            return dmg, lab, impulse_lost(truth, t0)

        res = {}
        dmg, label, imp = sim(None, None)
        res["no_action"] = dict(damage_mean=float(dmg.mean()), damage_median=float(np.median(dmg)),
                                impulse_lost_Ns=float(imp.mean()))
        dmg, _, imp = sim(t_iso, pred + 1)
        res["onboard"] = dict(damage_mean=float(dmg.mean()), damage_median=float(np.median(dmg)),
                              impulse_lost_Ns=float(imp.mean()), passes_needed=0, operator_hours=0.0,
                              fix_delay_med_h=float(np.median((t_iso - t0) / 60)),
                              fix_delay_p90_h=float(np.percentile((t_iso - t0) / 60, 90)),
                              correct_action=float((pred == true_cls).mean()))
        for name, (T, D) in SCENARIOS.items():
            Tm, ph = T * 60, phase * T * 60
            seen = next_pass(a, ph, Tm)                  # alarm visible on ground
            t_cmd = next_pass(seen + D * 60, ph, Tm)     # command uplinked
            t_cmd = np.minimum(t_cmd, N - 1)
            dmg, _, imp = sim(t_cmd, np.full(len(runs), true_cls + 1))
            res[name] = dict(damage_mean=float(dmg.mean()), damage_median=float(np.median(dmg)),
                             impulse_lost_Ns=float(imp.mean()), operator_hours=D,
                             fix_delay_med_h=float(np.median((t_cmd - t0) / 60)),
                             fix_delay_p90_h=float(np.percentile((t_cmd - t0) / 60, 90)),
                             passes_needed=2)   # 1 pass to see the alarm, 1 to uplink the command
        res["metric"] = label
        res["n"] = int(len(runs))
        out["faults"][f] = res
        print(f"  {f}: done", flush=True)
    out["runtime_s"] = time.time() - t_start
    json.dump(out, open(f"ground_vs_onboard{tag}.json", "w"), indent=1)

    # ---- summary table
    cols = ["no_action", "onboard"] + list(SCENARIOS)
    print("\nTIME FROM FAULT ONSET TO CORRECTIVE COMMAND (median hours)")
    print(f"{'fault':6s}" + "".join(f"{c:>16s}" for c in cols[1:]))
    for f, r in out["faults"].items():
        print(f"{f:6s}" + "".join(f"{r[c]['fix_delay_med_h']:16.1f}" for c in cols[1:]))
    print("\nDAMAGE AFTER ONSET (mean per run; lower is better)")
    print(f"{'fault':6s}" + "".join(f"{c:>16s}" for c in cols) + "   metric")
    for f, r in out["faults"].items():
        print(f"{f:6s}" + "".join(f"{r[c]['damage_mean']:16.1f}" for c in cols) + "   " + r["metric"])
    print("\nTOTAL IMPULSE LOST AFTER ONSET (mean N s per run; nominal thrust %.3f mN)" % T_NOM_mN)
    print(f"{'fault':6s}" + "".join(f"{c:>16s}" for c in cols))
    for f, r in out["faults"].items():
        print(f"{f:6s}" + "".join(f"{r[c]['impulse_lost_Ns']:16.1f}" for c in cols))
    print("\nGROUND EFFORT PER FAULT: onboard 0 passes / 0 operator-hours; " +
          "; ".join(f"{k}: 2 passes / {v[1]:.0f} operator-hours" for k, v in SCENARIOS.items()))
    print(f"\nonboard action correct: " +
          ", ".join(f"{f} {100*r['onboard']['correct_action']:.0f}%" for f, r in out["faults"].items()))
    print(f"runtime {out['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()