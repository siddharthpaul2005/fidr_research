"""
Reduced-order, physics-informed telemetry simulator for a CubeSat-class
iodine RF gridded ion thruster (~BIT-3 class operating point).

All parameter values here are ASSUMPTIONS chosen to be physically plausible
for a 2.5 cm-grid iodine RF thruster; they are not vendor data.  The
simulator exists to exercise the FDIR pipeline, not to predict a specific
thruster's behaviour.

Telemetry (one sample per minute = on-board 60 s average):
  0 Ib   beam (screen-supply) current      [mA]
  1 Ia   accelerator-grid impingement curr  [mA]
  2 Prf  RF forward power                   [W]
  3 mdot propellant mass flow               [ug/s]
  4 Vn   neutraliser coupling voltage       [V]
  5 narc recycle (arc) events in the minute [count]
  6 Tp   propulsion-module temperature      [degC]  (exogenous input)
"""
import numpy as np

# ---- physical constants & nominal operating point --------------------------
E = 1.602176634e-19
M_I = 126.904 * 1.66053906660e-27        # iodine atomic mass [kg]
MA_TO_UGS = 1e-3 * M_I / E * 1e9          # 1 mA singly-charged I+ -> ug/s
IB_SET = 15.0      # mA   (BIT-3 datasheet range 9-17 mA)
VS = 1500.0        # V    screen voltage (assumed)
VA0 = -200.0       # V    accel voltage (assumed)
MDOT_SET = 30.0    # ug/s main flow (assumed)
EPS_INF = 1500.0   # W/A  asymptotic beam-ion cost (assumed)
C0 = 3.0           # primary-utilisation factor in Brophy-type cost model
CHI0 = 0.006       # Ia/Ib at nominal utilisation (assumed)
VN0 = 18.0         # V nominal neutraliser coupling voltage (assumed)
LAMBDA0 = 0.001    # nominal recycle rate [1/min]
DT_MIN = 1.0       # sample period [min]
ORBIT_MIN = 95.0   # orbital period [min]

SIGMA = np.array([0.03, 0.004, 0.25, 0.30, 0.30])   # sensor/regulation noise
CH_NAMES = ["Ib", "Ia", "Prf", "mdot", "Vn"]
FAULTS = ["NOM", "GRID", "CATH", "ARC", "FEED", "RF"]
TRUE_SENS = [0.0040, 0.0020, 0.050]   # true thermal sensitivities (model uses 0.0036/0.0018/0.045)


def beam_ion_cost(eta_m, c0):
    """Brophy-type beam-ion production cost [W/A] (Goebel & Katz Ch. 4 form)."""
    return EPS_INF / (1.0 - np.exp(-c0 * (1.0 - eta_m)))


def temperature(n_runs, n, rng):
    t = np.arange(n) * DT_MIN
    phase = rng.uniform(0, 2 * np.pi, (n_runs, 1))
    amp = rng.uniform(6, 10, (n_runs, 1))
    return 20.0 + amp * np.sin(2 * np.pi * t / ORBIT_MIN + phase) \
        + 0.2 * rng.standard_normal((n_runs, n))


def ramp(t, t0, tau, shape=1.0):
    return np.clip((t - t0) / tau, 0.0, 1.0) ** shape


def make_scenarios(n_runs, fault, n, rng, onset_range=(50 * 60, 150 * 60)):
    """Return per-run fault parameters (dict of arrays)."""
    p = dict(fault=np.full(n_runs, FAULTS.index(fault)),
             t0=rng.uniform(*onset_range, n_runs) if fault != "NOM"
             else np.full(n_runs, np.inf))
    if fault == "GRID":
        p["tau"] = rng.uniform(100, 300, n_runs) * 60
        p["sev"] = rng.uniform(0.8, 1.2, n_runs)
    elif fault == "CATH":
        p["tau"] = rng.uniform(50, 200, n_runs) * 60
        p["sev"] = rng.uniform(2.0, 5.0, n_runs)       # V of coupling-voltage rise
    elif fault == "ARC":
        p["tau"] = rng.uniform(10, 50, n_runs) * 60
        p["sev"] = rng.uniform(0.02, 0.06, n_runs)     # final recycle rate [1/min]
    elif fault in ("FEED", "RF"):
        p["tau"] = rng.uniform(20, 150, n_runs) * 60
        p["sev"] = rng.uniform(0.08, 0.20, n_runs)     # fractional loss
    else:
        p["tau"] = np.ones(n_runs)
        p["sev"] = np.zeros(n_runs)
    return p


def simulate(p, n, rng, recovery=None, model=False, Tp=None):
    """
    Simulate telemetry.  If model=True, returns the on-board NOMINAL model
    prediction g(u) (no faults, no noise, slightly mismatched sensitivities).
    recovery: optional dict with per-run 't_r' (min) and 'action' codes.
    """
    n_runs = len(p["fault"])
    t = np.arange(n)[None, :] * DT_MIN
    if Tp is None:
        Tp = temperature(n_runs, n, rng)
    dT = Tp - 20.0
    f = p["fault"][:, None]
    t0, tau, sev = p["t0"][:, None], p["tau"][:, None], p["sev"][:, None]

    a_T, b_T, c_T, c0 = (0.0036, 0.0018, 0.045, C0 * 1.03) if model else \
                        tuple(TRUE_SENS) + (C0,)

    zero = np.zeros((n_runs, n))
    e = np.where(f == 1, sev * ramp(t, t0, tau, 1.5), zero) if not model else zero
    vn_d = np.where(f == 2, sev * ramp(t, t0, tau), zero) if not model else zero
    lam = LAMBDA0 + (np.where(f == 3, (sev - LAMBDA0) * ramp(t, t0, tau), zero)
                     if not model else zero)
    feed = np.where(f == 4, sev * ramp(t, t0, tau), zero) if not model else zero
    rfl = np.where(f == 5, sev * ramp(t, t0, tau), zero) if not model else zero

    # ---- recovery actions (closed loop) -----------------------------------
    ebs_margin = np.full((n_runs, n), 0.70)
    throttle = np.ones((n_runs, n))
    cath_reset = np.zeros((n_runs, n), bool)
    feed_corr = np.zeros((n_runs, n))
    if recovery is not None and not model:
        tr = recovery["t_r"][:, None]
        act = recovery["action"][:, None]
        after = t >= tr
        ebs_margin = np.where(after & (act == 1), 0.85, ebs_margin)  # Va -25 V
        cath_reset = after & (act == 2)                              # redundant cathode
        throttle = np.where(after & ((act == 3) | (act == 5)), 0.9, throttle)
        feed_corr = np.where(after & (act == 4), 1.0, feed_corr)     # heater setpoint
        lam = np.where(after & (act == 3), LAMBDA0 + 0.3 * (lam - LAMBDA0), lam)
        vn_d = np.where(cath_reset, 0.0, vn_d)

    ib_set = IB_SET * throttle
    mdot = MDOT_SET * throttle * (1 + a_T * dT) * (1 - feed * (1 - 0.8 * feed_corr))
    i_ebs = 3.0 * np.clip(e - ebs_margin, 0, None)          # backstreaming electrons [mA]
    i_ion = ib_set - i_ebs                                   # loop regulates measured Ib
    eta_m = i_ion * MA_TO_UGS / mdot
    c0_eff = c0 * (1 - 0.25 * e)                             # enlarged apertures -> neutral loss
    prf = i_ion * 1e-3 * beam_ion_cost(eta_m, c0_eff) / ((1 - b_T * dT) * (1 - rfl))
    eta_m0 = IB_SET * MA_TO_UGS / MDOT_SET
    ia = i_ion * CHI0 * (1 - eta_m) / (1 - eta_m0) * (1 + 0.6 * e)
    vn = VN0 + c_T * dT + vn_d
    ib = ib_set.copy()

    if model:
        return np.stack([ib, ia, prf, mdot, vn], axis=1), Tp

    narc = rng.poisson(lam * DT_MIN)
    # a recycle blanks the beam for ~3 s of the 60 s average and gives an Ia spike
    blank = np.clip(narc * 3.0 / 60.0, 0, 1)
    ib = ib * (1 - blank)
    prf = prf * (1 - blank)
    ia = ia + 0.02 * narc
    noise = rng.standard_normal((n_runs, 5, n)) * SIGMA[None, :, None]
    noise[:, 4, :] *= (1 + 1.5 * vn_d / 5.0)                 # worn cathode: noisier Vn
    y = np.stack([ib, ia, prf, mdot, vn], axis=1) + noise
    truth = dict(e=e, i_ebs=i_ebs, vn_d=vn_d, lam=lam, feed=feed, rfl=rfl,
                 thrust_mN=i_ion * 1e-3 * np.sqrt(2 * M_I * VS / E) * 1e3)
    return y, narc, Tp, truth


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = make_scenarios(2, "NOM", 600, rng)
    y, narc, Tp, tr = simulate(p, 600, rng)
    print("nominal means:", y.mean(axis=(0, 2)).round(3))
    print("thrust mN:", tr["thrust_mN"].mean().round(3),
          " eta_m:", round(IB_SET * MA_TO_UGS / MDOT_SET, 3))
    eta = IB_SET * MA_TO_UGS / MDOT_SET
    print("Isp(ion only) s:", (tr["thrust_mN"].mean() * 1e-3 /
                               (MDOT_SET * 1e-9) / 9.80665).round(0))
