import json, time, sys
import numpy as np
import thruster_sim as ts
import fdir as fd

N = 300 * 60                      # 300 h per run, 1 sample/min
N_TRAIN = dict(NOM=100, GRID=60, CATH=60, ARC=60, FEED=60, RF=60)
N_TEST = dict(NOM=100, GRID=100, CATH=100, ARC=100, FEED=100, RF=100)
PERSIST_BOCPD = 15
ISO_WIN = 60                      # minutes of evidence after alarm before isolating


def run_set(counts, seed, fixed=False):
    rng = np.random.default_rng(seed)
    out = []
    for fault, n_runs in counts.items():
        t0 = time.time()
        p = ts.make_scenarios(n_runs, fault, N, rng)
        y, narc, Tp, truth = ts.simulate(p, N, rng)
        g, _ = ts.simulate(p, N, rng, model=True, Tp=Tp)
        z = fd.normalised_residuals(y, g, narc, Tp).astype(np.float32)
        del g
        cs = fd.cusum(z)
        kt, kb, _ = fd.kalman_bias(z)
        bo = fd.bocpd(z)
        ac = fd.poisson_cusum(narc)
        d = dict(fault=fault, p=p, narc=narc, Tp=Tp, z=z, kb=kb, ac=ac,
                 cs_max=cs.max(1), kt_max=kt.max(1), bo_max=bo.max(1).astype(np.float32),
                 ex=dict(y=y[0].copy(), cs=cs[0].copy(), kt=kt[0].copy(), bo=bo[0].copy()))
        if fixed:
            kt_q, kb_q, _ = fd.kalman_q(z)
            cs_q = fd.cusum_q(z)
            d.update(kb_q=kb_q, kt_q_max=kt_q.max(1), cs_q_max=cs_q.max(1),
                     kb_err=float(np.abs(kb_q[:, :, fd.CAL_N:] - kb[:, :, fd.CAL_N:]).max()),
                     cs_err=float(np.abs(cs_q[:, :, fd.CAL_N:] - cs[:, :, fd.CAL_N:]).max()))
            if fault == "GRID":
                bo32 = fd.bocpd(z[:20], dtype=np.float32)
                d["bo32_max"] = bo32.max(1); d["bo32_err"] = float(np.abs(bo32[:, :, fd.CAL_N:] - bo[:20, :, fd.CAL_N:]).max())
                del bo32
            del kt_q, cs_q
        del y, cs, kt, bo
        out.append(d)
        print(f"  {fault}: {n_runs} runs in {time.time()-t0:.1f}s", flush=True)
    return out


def rolling_min(x, w):
    # x: (B, N) ; min over trailing window w
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.zeros((x.shape[0], w - 1)), x], axis=1)
    return sliding_window_view(pad, w, axis=1).min(axis=2)


def system_stats(d):
    return dict(CUSUM=d["cs_max"], KF=d["kt_max"],
                BOCPD=rolling_min(d["bo_max"], PERSIST_BOCPD), ARC=d["ac"])


def calibrate(train):
    nom = [d for d in train if d["fault"] == "NOM"][0]
    st = system_stats(nom)
    thr = {}
    for k, v in st.items():
        m = v[:, fd.CAL_N:].max()
        thr[k] = float(min(m * 1.05, 0.9999)) if k == "BOCPD" else float(m * 1.10)
    return thr


def alarms(d, thr):
    st = system_stats(d)
    a = {k: fd.first_crossing(st[k], thr[k]) for k in st}
    B = len(a["CUSUM"])
    big = 10**9
    def t(x): return np.where(x < 0, big, x)
    a["OR"] = np.minimum.reduce([t(a["CUSUM"]), t(a["KF"]), t(a["BOCPD"]), t(a["ARC"])])
    # 2-of-3 vote among continuous detectors (state = latched after first crossing), OR arc
    s3 = np.sort(np.stack([t(a["CUSUM"]), t(a["KF"]), t(a["BOCPD"])]), axis=0)
    a["VOTE"] = np.minimum(s3[1], t(a["ARC"]))
    for k in ("OR", "VOTE"):
        a[k] = np.where(a[k] >= big, -1, a[k])
    return a


def detection_metrics(data, thr):
    res = {}
    det_names = ["CUSUM", "KF", "BOCPD", "OR", "VOTE"]
    exposure_h = {k: 0.0 for k in det_names}
    false_alarms = {k: 0 for k in det_names}
    for d in data:
        a = alarms(d, thr)
        t0 = d["p"]["t0"]
        for k in det_names:
            ak = a[k]
            fa = (ak >= 0) & (ak < t0)
            false_alarms[k] += int(fa.sum())
            # nominal exposure: from end of calibration to onset (or alarm)
            end = np.minimum(np.where(np.isinf(t0), N, t0), np.where(ak >= 0, ak, N))
            exposure_h[k] += float(np.clip(end - fd.CAL_N, 0, None).sum() / 60)
            if d["fault"] == "NOM":
                continue
            det = (ak >= t0)
            delay_h = (ak[det] - t0[det]) / 60
            res.setdefault(d["fault"], {})[k] = dict(
                pd=float(det.mean()), n=int(len(det)), pre_onset_alarms=int(fa.sum()),
                delay_med_h=float(np.median(delay_h)) if det.any() else None,
                delay_p25_h=float(np.percentile(delay_h, 25)) if det.any() else None,
                delay_p75_h=float(np.percentile(delay_h, 75)) if det.any() else None,
                delays=delay_h.tolist())
    far = {k: dict(false_alarms=false_alarms[k], exposure_h=exposure_h[k],
                   per_1000h=1000 * false_alarms[k] / exposure_h[k]) for k in det_names}
    return res, far


def iso_dataset(data, thr, det="VOTE", bias_key="kb"):
    X, y, idx = [], [], []
    for d in data:
        if d["fault"] == "NOM":
            continue
        a = alarms(d, thr)[det]
        ok = (a >= d["p"]["t0"])
        t_iso = np.minimum(np.where(ok, a + ISO_WIN, 0), N - 1)
        f = fd.features(d[bias_key], d["z"], d["ac"], t_iso)
        X.append(f[ok]); y += [fd.CLASSES.index(d["fault"])] * int(ok.sum())
        idx.append((d["fault"], np.where(ok)[0], t_iso[ok]))
    return np.vstack(X), np.array(y), idx


def main():
    t_start = time.time()
    print("training set"); train = run_set(N_TRAIN, seed=1)
    thr = calibrate(train)
    print("thresholds", thr)
    print("test set"); test = run_set(N_TEST, seed=2, fixed=True)

    det, far = detection_metrics(test, thr)
    Xtr, ytr, _ = iso_dataset(train, thr)
    Xte, yte, idx = iso_dataset(test, thr)
    model = fd.train_logreg(Xtr, ytr, 5)
    qm = fd.quantise_model(model)
    pf = fd.predict(model, Xte); pq = fd.predict_q(qm, Xte)
    cm_f = np.zeros((5, 5), int); cm_q = np.zeros((5, 5), int)
    for a, b in zip(yte, pf): cm_f[a, b] += 1
    for a, b in zip(yte, pq): cm_q[a, b] += 1

    # ------------- closed-loop recovery: re-simulate with actions applied
    rec = {}
    pred_by_fault = {}
    k = 0
    for (fault, runs, t_iso) in idx:
        pred_by_fault[fault] = (runs, t_iso, pq[k:k + len(runs)]); k += len(runs)
    for d in test:
        f = d["fault"]
        if f == "NOM": continue
        runs, t_iso, pcls = pred_by_fault[f]
        sub = {kk: v[runs] for kk, v in d["p"].items()}
        rng = np.random.default_rng(99)
        recovery = dict(t_r=t_iso.astype(float), action=pcls + 1)
        y1, n1, _, tr1 = ts.simulate(sub, N, rng, recovery=recovery, Tp=d["Tp"][runs])
        rng = np.random.default_rng(99)
        y0, n0, _, tr0 = ts.simulate(sub, N, rng, Tp=d["Tp"][runs])
        last = slice(N - 600, N)                       # final 10 h of the run
        if f == "GRID":
            m0 = (tr0["i_ebs"][:, last].mean(1) > 0.01); m1 = (tr1["i_ebs"][:, last].mean(1) > 0.01)
            rec[f] = dict(metric="runs with electron backstreaming in final 10 h",
                          no_action=float(m0.mean()), with_fdir=float(m1.mean()))
        elif f == "CATH":
            rec[f] = dict(metric="mean Vn rise in final 10 h [V]",
                          no_action=float(tr0["vn_d"][:, last].mean()),
                          with_fdir=float(tr1["vn_d"][:, last].mean()))
        elif f == "ARC":
            post = np.arange(N)[None, :] >= t_iso[:, None]
            rec[f] = dict(metric="recycles per 24 h after action",
                          no_action=float((n0 * post).sum(1).mean() / ((N - t_iso).mean() / 1440)),
                          with_fdir=float((n1 * post).sum(1).mean() / ((N - t_iso).mean() / 1440)))
        elif f == "FEED":
            rec[f] = dict(metric="mean flow deficit in final 10 h [%]",
                          no_action=float(100 * (1 - y0[:, 3, last].mean(1) / ts.MDOT_SET).mean()),
                          with_fdir=float(100 * (1 - y1[:, 3, last].mean(1) /
                                    (ts.MDOT_SET * np.where(recovery['action'] == 5, 0.9, 1.0))).mean()))
        elif f == "RF":
            lim = 40.0
            rec[f] = dict(metric="runs exceeding 40 W RF-power envelope in final 10 h",
                          no_action=float((y0[:, 2, last].mean(1) > lim).mean()),
                          with_fdir=float((y1[:, 2, last].mean(1) > lim).mean()))
        rec[f]["n"] = int(len(runs))

    # ------------- fixed-point vs float on test set
    fx = {}
    test_q = []
    for d in test:
        dq = dict(d); dq.update(kt_max=d["kt_q_max"], kb=d["kb_q"], cs_max=d["cs_q_max"])
        test_q.append(dq)
        aq = alarms(dq, thr); af = alarms(d, thr)
        for kk in ("CUSUM", "KF", "VOTE", "OR"):
            same = (aq[kk] == af[kk])
            dd = np.abs(aq[kk] - af[kk])[(aq[kk] >= 0) & (af[kk] >= 0)]
            e = fx.setdefault(kk, dict(same=0, n=0, max_shift_min=0))
            e["same"] += int(same.sum()); e["n"] += len(same)
            e["max_shift_min"] = int(max(e["max_shift_min"], dd.max() if len(dd) else 0))
        fx["max_abs_bias_err"] = max(fx.get("max_abs_bias_err", 0), d["kb_err"])
        fx["max_abs_cusum_err"] = max(fx.get("max_abs_cusum_err", 0), d["cs_err"])
    fx["kf_gain"] = fd.kf_steady_gain()[0].tolist()
    detq, farq = detection_metrics(test_q, thr)
    fx["detection_fixed"] = {f: {k: dict(pd=v["pd"], delay_med_h=v["delay_med_h"]) for k, v in dv.items()}
                             for f, dv in detq.items()}
    fx["far_fixed"] = farq
    Xq, yq, _ = iso_dataset(test_q, thr)
    fx["iso_acc_full_fixed_chain"] = float((fd.predict_q(qm, Xq) == yq).mean())
    fx["iso_n_full_fixed_chain"] = int(len(yq))
    g = [d for d in test if d["fault"] == "GRID"][0]
    s64 = rolling_min(g["bo_max"][:20], PERSIST_BOCPD)
    s32 = rolling_min(g["bo32_max"].astype(np.float64), PERSIST_BOCPD)
    a64 = fd.first_crossing(s64, thr["BOCPD"]); a32 = fd.first_crossing(s32, thr["BOCPD"])
    fx["bocpd_f32_same_alarm"] = int((a64 == a32).sum()); fx["bocpd_f32_n"] = 20
    fx["bocpd_f32_max_shift_min"] = int(np.abs(a64 - a32).max())
    fx["bocpd_f32_max_abs_err"] = g["bo32_err"]

    results = dict(thresholds=thr, detection=det, far=far,
                   iso=dict(acc_float=float((pf == yte).mean()), acc_int8=float((pq == yte).mean()),
                            n=int(len(yte)), cm_float=cm_f.tolist(), cm_int8=cm_q.tolist(),
                            classes=fd.CLASSES,
                            agree=float((pf == pq).mean())),
                   recovery=rec, fixed_point=fx,
                   model=dict(W=model["W"].tolist(), b=model["b"].tolist(),
                              mu=model["mu"].tolist(), sd=model["sd"].tolist(),
                              Wq=qm["Wq"].tolist(), bq=qm["bq"].tolist(), w_scale=qm["w_scale"]),
                   config=dict(N_h=N / 60, n_train=N_TRAIN, n_test=N_TEST, cal_h=fd.CAL_N / 60,
                               persist_bocpd=PERSIST_BOCPD, iso_win_min=ISO_WIN,
                               sigma=ts.SIGMA.tolist()),
                   runtime_s=time.time() - t_start)
    json.dump(results, open("results.json", "w"), indent=1)
    ex = {}
    for d in test:
        a = alarms(d, thr)
        ex[d["fault"]] = dict(z=d["z"][0], narc=d["narc"][0], t0=d["p"]["t0"][0],
                              alarm=a["VOTE"][0], ac=d["ac"][0], **d["ex"])
    np.save("examples.npy", ex, allow_pickle=True)
    print(json.dumps({k: v for k, v in results.items() if k not in ("detection", "model")}, indent=1))
    for f, v in det.items():
        print(f, {k: (round(x["pd"], 3), x["delay_med_h"] and round(x["delay_med_h"], 2)) for k, x in v.items()})


if __name__ == "__main__":
    main()
