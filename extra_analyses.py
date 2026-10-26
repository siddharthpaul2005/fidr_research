"""
Extra analyses for the paper (uses the thresholds and classifier in results.json):

 1. LIMIT-CHECK BASELINE - what most satellites do today: fixed upper/lower
    limits on each raw telemetry channel (set as tight as possible from the
    healthy TRAINING runs) plus a recycle-count limit over 24 h.
 2. ABLATION - each detector alone and in combinations, vs our 2-of-3 vote.
 3. CLASSIFIER EXPLAINABILITY - the learned weights: which feature pushes
    the decision towards which fault.

Output: extra_analyses.json + printed tables.
"""
import json, time
import numpy as np
import thruster_sim as ts
import fdir as fd
import run_experiments as rx

N = rx.N
CH = ["Ib", "Ia", "Prf", "mdot", "Vn"]


def gen_raw(counts, seed):
    """regenerate exactly the same raw telemetry as rx.run_set(counts, seed)."""
    rng = np.random.default_rng(seed)
    for fault, n_runs in counts.items():
        p = ts.make_scenarios(n_runs, fault, N, rng)
        y, narc, Tp, _ = ts.simulate(p, N, rng)
        yield fault, p, y, narc


def trailing_sum(x, w):
    c = np.cumsum(x, axis=1)
    out = c.copy()
    out[:, w:] = c[:, w:] - c[:, :-w]
    return out


def limit_alarm(y, narc, lim, arc_lim, persist=3):
    ok = narc == 0
    viol = np.zeros(narc.shape, bool)
    for c in range(5):
        v = ((y[:, c] < lim[c][0]) | (y[:, c] > lim[c][1])) & ok
        viol |= v
    run = np.zeros(viol.shape[0], int); first = np.full(viol.shape[0], -1)
    for t in range(fd.CAL_N, viol.shape[1]):
        run = np.where(viol[:, t], run + 1, 0)
        newly = (run >= persist) & (first < 0)
        first[newly] = t
    arc24 = trailing_sum(narc, 1440)
    a_arc = np.where((arc24[:, fd.CAL_N:] > arc_lim).any(1),
                     (arc24[:, fd.CAL_N:] > arc_lim).argmax(1) + fd.CAL_N, -1)
    big = 10**9
    f = np.minimum(np.where(first < 0, big, first), np.where(a_arc < 0, big, a_arc))
    return np.where(f >= big, -1, f)


def summarise(alarm, t0, fault):
    pre = (alarm >= 0) & (alarm < t0)
    det = alarm >= t0
    d = (alarm[det] - t0[det]) / 60
    return dict(pd=float(det.mean()), pre_onset_alarms=int(pre.sum()),
                delay_med_h=float(np.median(d)) if det.any() else None)


def main():
    t_start = time.time()
    R = json.load(open("results.json"))
    thr = R["thresholds"]
    out = {}

    # ---------------- 1. limit-check baseline
    print("1) limit-check baseline: setting limits from healthy TRAINING runs")
    _, p, y, narc = next(gen_raw(dict(NOM=100), seed=1))
    ok = (narc == 0)[:, None, :] & (np.arange(N)[None, None, :] >= fd.CAL_N)
    lim = []
    for c in range(5):
        v = y[:, c][ok[:, 0]]
        lo, hi = v.min(), v.max(); m = 0.10 * (hi - lo)
        lim.append((float(lo - m), float(hi + m)))
    arc_lim = int(trailing_sum(narc, 1440)[:, fd.CAL_N:].max())
    out["limits"] = {CH[c]: lim[c] for c in range(5)}
    out["limits"]["recycles_per_24h_max"] = arc_lim
    print("   limits:", {k: (round(v[0], 3), round(v[1], 3)) if isinstance(v, tuple) else v
                         for k, v in out["limits"].items()})
    del y, narc

    lim_res, fa, expo = {}, 0, 0.0
    for fault, p, y, narc in gen_raw(rx.N_TEST, seed=2):
        a = limit_alarm(y, narc, lim, arc_lim)
        t0 = p["t0"]
        fa += int(((a >= 0) & (a < t0)).sum())
        end = np.minimum(np.where(np.isinf(t0), N, t0), np.where(a >= 0, a, N))
        expo += float(np.clip(end - fd.CAL_N, 0, None).sum() / 60)
        if fault != "NOM":
            lim_res[fault] = summarise(a, t0, fault)
        print(f"   {fault} done", flush=True)
    out["limit_check"] = dict(per_fault=lim_res, false_alarms=fa, exposure_h=expo)

    # ---------------- 2. ablation (detector statistics from the same test set)
    print("2) ablation: re-running detectors on the test set")
    test = rx.run_set(rx.N_TEST, seed=2)
    variants = {"CUSUM only": ["CUSUM"], "Kalman only": ["KF"], "BOCPD only": ["BOCPD"],
                "CUSUM or Kalman": ["CUSUM", "KF"], "any of 3 (OR)": ["CUSUM", "KF", "BOCPD"],
                "2-of-3 vote (ours)": "VOTE"}
    abl = {}
    big = 10**9
    for name, v in variants.items():
        per, fa, expo = {}, 0, 0.0
        for d in test:
            a = rx.alarms(d, thr)
            if v == "VOTE":
                al = a["VOTE"]
            else:
                al = np.min([np.where(a[k] < 0, big, a[k]) for k in v] +
                            [np.where(a["ARC"] < 0, big, a["ARC"])], axis=0)
                al = np.where(al >= big, -1, al)
            t0 = d["p"]["t0"]
            fa += int(((al >= 0) & (al < t0)).sum())
            end = np.minimum(np.where(np.isinf(t0), N, t0), np.where(al >= 0, al, N))
            expo += float(np.clip(end - fd.CAL_N, 0, None).sum() / 60)
            if d["fault"] != "NOM":
                per[d["fault"]] = summarise(al, t0, d["fault"])
        abl[name] = dict(per_fault=per, false_alarms=fa, exposure_h=expo)
    out["ablation"] = abl

    # ---------------- 3. explainability
    W = np.array(R["model"]["W"])
    feats = ["bias Ib", "bias Ia", "bias Prf", "bias mdot", "bias Vn",
             "arc score", "Vn noise", "total drift"]
    out["explainability"] = dict(features=feats, classes=fd.CLASSES, W=W.tolist())
    out["runtime_s"] = time.time() - t_start
    json.dump(out, open("extra_analyses.json", "w"), indent=1)

    # ---------------- print tables
    flts = fd.CLASSES
    def row(name, r):
        cells = []
        for f in flts:
            x = r["per_fault"][f]
            dl = "-" if x["delay_med_h"] is None else f"{x['delay_med_h']:.1f}"
            cells.append(f"{100*x['pd']:.0f}%|{dl}h")
        return f"{name:22s}" + "".join(f"{c:>14s}" for c in cells) + f"{r['false_alarms']:>8d}"
    hdr = f"{'method':22s}" + "".join(f"{f:>14s}" for f in flts) + f"{'FA':>8s}"
    print("\nDETECTION: detection rate | median delay   (FA = false alarms in healthy hours)")
    print(hdr)
    print(row("limit check (today)", out["limit_check"]))
    for name, r in abl.items():
        print(row(name, r))
    print("\nCLASSIFIER WEIGHTS (positive = pushes towards that fault)")
    print(f"{'feature':12s}" + "".join(f"{c:>8s}" for c in flts))
    for i, fn in enumerate(feats):
        print(f"{fn:12s}" + "".join(f"{W[i, k]:8.2f}" for k in range(5)))
    print(f"\nruntime {out['runtime_s']:.0f} s")


if __name__ == "__main__":
    main()