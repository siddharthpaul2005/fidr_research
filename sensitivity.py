import json, numpy as np
import thruster_sim as ts, fdir as fd, run_experiments as rx

thr = json.load(open("results.json"))["thresholds"]
out = dict(severity={}, mismatch={})
orig = ts.make_scenarios
for scale in (0.1, 0.25, 0.5):
    def ms(n_runs, fault, n, rng, onset_range=(50*60, 150*60), _s=scale):
        p = orig(n_runs, fault, n, rng, onset_range)
        if fault == "ARC":
            p["sev"] = ts.LAMBDA0 + _s * (p["sev"] - ts.LAMBDA0)
        else:
            p["sev"] = _s * p["sev"]
        return p
    ts.make_scenarios = ms
    data = rx.run_set(dict(GRID=50, CATH=50, ARC=50, FEED=50, RF=50), seed=int(100 * scale) + 7)
    det, far = rx.detection_metrics(data, thr)
    out["severity"][scale] = {f: dict(pd=v["VOTE"]["pd"], delay_med_h=v["VOTE"]["delay_med_h"],
                                      pd_or=v["OR"]["pd"], delay_or=v["OR"]["delay_med_h"])
                              for f, v in det.items()}
    out["severity"][scale]["far_vote_per_1000h"] = far["VOTE"]["per_1000h"]
    print(scale, out["severity"][scale], flush=True)
    del data
ts.make_scenarios = orig
for k, sens in (("x1.5", [0.0054, 0.0027, 0.0675]), ("x2.0", [0.0072, 0.0036, 0.090])):
    ts.TRUE_SENS = sens
    data = rx.run_set(dict(NOM=100), seed=55)
    det, far = rx.detection_metrics(data, thr)
    out["mismatch"][k] = {d: dict(fa=v["false_alarms"], h=v["exposure_h"], per_1000h=v["per_1000h"]) for d, v in far.items()}
    print(k, out["mismatch"][k], flush=True)
ts.TRUE_SENS = [0.0040, 0.0020, 0.050]
json.dump(out, open("sensitivity.json", "w"), indent=1)
