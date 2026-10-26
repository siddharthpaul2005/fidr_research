"""
On-board FDIR pipeline (reference, vectorised over Monte-Carlo runs).

Detection  : per-channel two-sided CUSUM, truncated BOCPD (Adams & MacKay),
             Kalman bias/drift filter on model residuals; Poisson CUSUM on
             recycle counts.  Recycle-flagged samples are masked.
Isolation  : 8-feature multinomial logistic classifier evaluated 60 min
             after the alarm (float64 and int8-quantised).
Recovery   : rule table (fault class -> corrective command).
Fixed point: KF + CUSUM re-implemented in Q16.16 integer arithmetic.
"""
import numpy as np
from scipy.special import logsumexp

CAL_N = 24 * 60          # self-calibration window [samples]
NCH = 5


# ---------------------------------------------------------------- residuals
def normalised_residuals(y, g, narc, Tp=None):
    """
    Residual r = y - g(u).  During the 24 h self-calibration window an
    offset and (if Tp is given) a thermal gain are fitted per channel by
    least squares on non-recycle samples; z is the standardised remainder.
    """
    r = y - g
    mask = narc[:, None, :] > 0
    if Tp is None:
        X = None
        rc = np.where(mask, np.nan, r)[:, :, :CAL_N]
        mu = np.nanmean(rc, axis=2, keepdims=True)
        e = r - mu
    else:
        dT = (Tp - 20.0)[:, None, :]
        w = (~mask[:, :, :CAL_N]).astype(float)
        x = np.broadcast_to(dT[:, :, :CAL_N], w.shape); rr = r[:, :, :CAL_N]
        sw = w.sum(2, keepdims=True); sx = (w * x).sum(2, keepdims=True)
        sy = (w * rr).sum(2, keepdims=True); sxx = (w * x * x).sum(2, keepdims=True)
        sxy = (w * x * rr).sum(2, keepdims=True)
        beta = (sw * sxy - sx * sy) / (sw * sxx - sx * sx)
        mu = (sy - beta * sx) / sw
        e = r - mu - beta * dT
    ec = np.where(mask, np.nan, e)[:, :, :CAL_N]
    sd = np.sqrt(np.nanmean(ec ** 2, axis=2, keepdims=True))
    z = e / sd
    z = np.where(mask, 0.0, z)            # recycle-aware masking
    return z


# ---------------------------------------------------------------- CUSUM
def cusum(z, k=0.5):
    B, C, N = z.shape
    sp = np.zeros((B, C)); sn = np.zeros((B, C))
    out = np.empty((B, C, N), np.float32)
    for t in range(N):
        sp = np.maximum(0, sp + z[:, :, t] - k)
        sn = np.maximum(0, sn - z[:, :, t] - k)
        out[:, :, t] = np.maximum(sp, sn)
    return out


def poisson_cusum(narc, lam0=0.001, ratio=10.0):
    lam1 = lam0 * ratio
    llr = narc * np.log(ratio) - (lam1 - lam0)
    B, N = narc.shape
    s = np.zeros(B); out = np.empty((B, N))
    for t in range(N):
        s = np.maximum(0, s + llr[:, t]); out[:, t] = s
    return out


# ---------------------------------------------------------------- Kalman
KF_Q = np.diag([1e-5, 1e-10])


def kf_steady_gain(q=KF_Q, r=1.0):
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    P = np.array([[1.0, 0.0], [0.0, 1e-6]])
    for _ in range(20000):
        P = F @ P @ F.T + q
        K = P[:, 0] / (P[0, 0] + r)
        P = P - np.outer(K, P[0, :])
    return K, P


def kalman_bias(z, q=KF_Q, r=1.0):
    """
    2-state (bias, drift) steady-state Kalman filter on the normalised
    residual.  The gain is computed offline (the Riccati recursion converges
    well inside the calibration window), which is also what is deployed.
    Returns |bias|/sigma_b test statistic, bias and drift estimates.
    """
    K, P = kf_steady_gain(q, r)
    sig = np.sqrt(P[0, 0])
    B, C, N = z.shape
    b = np.zeros((B, C)); d = np.zeros((B, C))
    tstat = np.empty((B, C, N), np.float32); bias = np.empty((B, C, N), np.float32)
    for t in range(N):
        bp = b + d
        nu = z[:, :, t] - bp
        b = bp + K[0] * nu
        d = d + K[1] * nu
        bias[:, :, t] = b
        tstat[:, :, t] = np.abs(b) / sig
    return tstat, bias, None


# ---------------------------------------------------------------- BOCPD
def bocpd(z, R=64, hazard=1 / 1000.0, kappa0=0.1, dtype=np.float64):
    """
    Truncated Bayesian online change-point detection (Gaussian, known unit
    variance, unknown mean with N(0, 1/kappa0) prior).  Bin R-1 is the
    'no change since reset' baseline hypothesis.  Returns
    p_recent = P(change within last R-1 samples).
    """
    B, C, N = z.shape
    lp = np.full((B, C, R), -np.inf, dtype=dtype); lp[:, :, 0] = 0.0
    n = np.zeros((B, C, R), dtype=dtype); s = np.zeros((B, C, R), dtype=dtype)
    lH, l1H = dtype(np.log(hazard)), dtype(np.log1p(-hazard))
    out = np.empty((B, C, N), dtype=dtype)
    half_l2pi = dtype(0.5 * np.log(2 * np.pi))
    for t in range(N):
        x = z[:, :, t].astype(dtype)[:, :, None]
        lam = kappa0 + n
        m = s / lam
        var = 1.0 + 1.0 / lam
        pred = -0.5 * (x - m) ** 2 / var - 0.5 * np.log(var) - half_l2pi
        joint = lp + pred
        new = np.empty_like(lp)
        new[:, :, 0] = logsumexp(joint, axis=2) + lH
        new[:, :, 1:R - 1] = joint[:, :, 0:R - 2] + l1H
        new[:, :, R - 1] = np.logaddexp(joint[:, :, R - 2], joint[:, :, R - 1]) + l1H
        new -= logsumexp(new, axis=2, keepdims=True)
        # sufficient statistics
        n_new = np.empty_like(n); s_new = np.empty_like(s)
        n_new[:, :, 0] = 0; s_new[:, :, 0] = 0
        n_new[:, :, 1:R - 1] = n[:, :, 0:R - 2] + 1
        s_new[:, :, 1:R - 1] = s[:, :, 0:R - 2] + x
        n_new[:, :, R - 1] = n[:, :, R - 1] + 1          # baseline: all data since reset
        s_new[:, :, R - 1] = s[:, :, R - 1] + x[:, :, 0]
        lp, n, s = new.astype(dtype), n_new, s_new
        out[:, :, t] = -np.expm1(lp[:, :, R - 1]) if t >= R else 0.0
    return out


# ---------------------------------------------------------------- alarms
def first_crossing(stat, thr, persist=1, start=CAL_N):
    """stat: (B, N) -> index of first sample where stat>thr for `persist` samples."""
    above = stat > thr
    above[:, :start] = False
    if persist > 1:
        c = np.zeros_like(above, dtype=np.int32)
        run = np.zeros(above.shape[0], np.int32)
        for t in range(above.shape[1]):
            run = np.where(above[:, t], run + 1, 0); c[:, t] = run
        above = c >= persist
    idx = np.where(above.any(axis=1), above.argmax(axis=1), -1)
    return idx


# ---------------------------------------------------------------- isolation
CLASSES = ["GRID", "CATH", "ARC", "FEED", "RF"]
ACTIONS = {1: "Accel-grid voltage correction (Va -25 V; throttle if at limit)",
           2: "Switch to redundant neutraliser/cathode",
           3: "Throttle down 10 % + high-voltage recycle/conditioning",
           4: "Feed-system setpoint correction (sublimator heater), then throttle",
           5: "Throttle down 10 % to stay inside PPU power envelope"}


def features(bias, z, arcstat, t_iso):
    """8 features at isolation time: 5 KF biases, arc stat, Vn variance ratio, |bias| norm."""
    B = bias.shape[0]
    f = np.zeros((B, 8))
    for b in range(B):
        t = t_iso[b]
        f[b, :5] = bias[b, :, t]
        f[b, 5] = np.log1p(arcstat[b, t])
        w = z[b, 4, max(0, t - 60):t + 1]
        f[b, 6] = np.log(np.var(w - w.mean()) + 1e-3)
        f[b, 7] = np.log1p(np.abs(bias[b, :, t]).sum())
    return f


def train_logreg(X, y, ncls, lr=0.2, epochs=3000, l2=1e-3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    W = np.zeros((X.shape[1], ncls)); b = np.zeros(ncls)
    Y = np.eye(ncls)[y]
    for _ in range(epochs):
        logits = Xs @ W + b
        P = np.exp(logits - logits.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        G = P - Y
        W -= lr * (Xs.T @ G / len(y) + l2 * W); b -= lr * G.mean(0)
    return dict(W=W, b=b, mu=mu, sd=sd)


def predict(model, X):
    return np.argmax(((X - model["mu"]) / model["sd"]) @ model["W"] + model["b"], 1)


def quantise_model(model, in_scale=16.0):
    """int8 weights (per-tensor symmetric), int8 inputs (Q4.4 after standardisation)."""
    W, b = model["W"], model["b"]
    w_scale = 127.0 / np.abs(W).max()
    Wq = np.clip(np.round(W * w_scale), -127, 127).astype(np.int8)
    bq = np.round(b * w_scale * in_scale).astype(np.int32)
    return dict(Wq=Wq, bq=bq, mu=model["mu"], sd=model["sd"], in_scale=in_scale,
                w_scale=w_scale)


def predict_q(qm, X):
    xs = (X - qm["mu"]) / qm["sd"]
    xq = np.clip(np.round(xs * qm["in_scale"]), -127, 127).astype(np.int32)
    acc = xq @ qm["Wq"].astype(np.int32) + qm["bq"]
    return np.argmax(acc, 1)


# ---------------------------------------------------------------- fixed point
Q = 16
ONE = 1 << Q


def to_q(x):
    return np.clip(np.round(x * ONE), -2**31, 2**31 - 1).astype(np.int64)


def sat32(x):
    return np.clip(x, -2**31, 2**31 - 1)


def cusum_q(z, k=0.5):
    """Q16.16 CUSUM; inputs quantised to int16 Q8.8 first (telemetry word)."""
    zq = np.clip(np.round(z * 256), -32768, 32767).astype(np.int64) << (Q - 8)
    kq = to_q(k)
    B, C, N = z.shape
    sp = np.zeros((B, C), np.int64); sn = np.zeros((B, C), np.int64)
    out = np.empty((B, C, N), np.float32)
    for t in range(N):
        sp = sat32(np.maximum(0, sp + zq[:, :, t] - kq))
        sn = sat32(np.maximum(0, sn - zq[:, :, t] - kq))
        out[:, :, t] = np.maximum(sp, sn) / ONE
    return out


def kalman_q(z, q=KF_Q, r=1.0):
    """
    Fixed-point steady-state KF.  Gain is pre-computed offline and stored as
    Q0.31 (k0) and Q-9.40 (k1); states (bias, drift) are int32 Q8.24; the
    input telemetry residual is an int16 Q8.8 word.  Integer-only update with
    round-to-nearest shifts and saturation.
    """
    K, P = kf_steady_gain(q, r)
    k0 = int(round(K[0] * 2**31)); k1 = int(round(K[1] * 2**40))
    sig = np.sqrt(P[0, 0])
    QS = 24
    zq = np.clip(np.round(z * 256), -32768, 32767).astype(np.int64) << (QS - 8)
    B, C, N = z.shape
    b = np.zeros((B, C), np.int64); d = np.zeros((B, C), np.int64)
    tstat = np.empty((B, C, N), np.float32); bias = np.empty((B, C, N), np.float32)
    for t in range(N):
        bp = sat32(b + d)
        nu = sat32(zq[:, :, t] - bp)
        b = sat32(bp + ((nu * k0 + (1 << 30)) >> 31))
        d = sat32(d + ((nu * k1 + (1 << 39)) >> 40))
        bias[:, :, t] = b / float(1 << QS)
        tstat[:, :, t] = np.abs(b / float(1 << QS)) / sig
    return tstat, bias, dict(K=K, P=P, sig=sig)
