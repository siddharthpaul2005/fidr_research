import sys, numpy as np
sys.path.insert(0, "..")
import thruster_sim as ts
fault = sys.argv[1] if len(sys.argv) > 1 else "GRID"
rng = np.random.default_rng(1)
N = 300 * 60
p = ts.make_scenarios(1, fault, N, rng)
y, narc, Tp, _ = ts.simulate(p, N, rng)
fr = np.concatenate([y[0], Tp, narc.astype(float)], 0).T.astype(np.float32)
fr.tofile("frames.bin")
np.savetxt(f"telemetry_{fault}.csv", fr, delimiter=",", comments="",
           header="Ib_mA,Ia_mA,Prf_W,mdot_ugs,Vn_V,T_C,n_recycle")
print("fault", fault, "starts at minute", int(p["t0"][0]) if fault != "NOM" else "none")