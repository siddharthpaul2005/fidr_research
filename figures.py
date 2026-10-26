import json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import thruster_sim as ts, fdir as fd

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.4,
                     "savefig.dpi": 300, "figure.facecolor": "white"})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2 = "#0b0b0b", "#52514e"
R = json.load(open("results.json"))
ex = np.load("examples.npy", allow_pickle=True).item()
CLS = ["GRID", "CATH", "ARC", "FEED", "RF"]
LBL = {"GRID": "Grid erosion", "CATH": "Cathode/neutraliser wear", "ARC": "Arcing",
       "FEED": "Feed degradation", "RF": "RF-circuit degradation"}

# ---------------- Fig 1: architecture
fig, ax = plt.subplots(figsize=(6.8, 2.8)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-6, 40)
boxes = [(1, 14, 15, 14, "PPU telemetry\n(1 frame / 60 s)\nIb, Ia, P$_{RF}$, ṁ,\nV$_n$, recycles, T"),
         (20, 14, 15, 14, "Nominal\nmodel g(u)\n+ 24 h self-\ncalibration\n→ residuals z"),
         (39, 22, 19, 14, "Detection\nCUSUM · Kalman\nbias/drift · BOCPD\n(2-of-3 vote)"),
         (39, 2, 19, 13, "Poisson CUSUM\non recycle count\n(arcing)"),
         (62, 14, 16, 14, "Isolation\nint8 linear\nclassifier\n(5 classes)"),
         (82, 14, 17, 14, "Recovery\nrule table →\nVa correction,\ncathode switch,\nthrottle, feed")]
for x, y, w, h, t in boxes:
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc="#f3f6fb", ec=CAT[0], lw=1.0))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=6.6, color=INK)
arr = dict(arrowstyle="-|>", color=INK2, lw=0.9)
for a, b in [((16.4, 21), (19.6, 21)), ((35.4, 23), (38.6, 28)), ((35.4, 19), (38.6, 9)),
             ((58.4, 28), (61.6, 23)), ((58.4, 9), (61.6, 19)), ((78.4, 21), (81.6, 21))]:
    ax.annotate("", xy=b, xytext=a, arrowprops=arr)
ax.plot([90.5, 90.5, 27.5], [13.4, -1.5, -1.5], color=CAT[1], lw=0.9)
ax.annotate("", xy=(27.5, 13.4), xytext=(27.5, -1.5), arrowprops=dict(arrowstyle="-|>", color=CAT[1], lw=0.9))
ax.text(70, -4.6, "re-arm / re-calibrate after action", ha="center", fontsize=6.8, color=CAT[1])
fig.savefig("fig1_architecture.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 2: example residuals
fig, axes = plt.subplots(5, 1, figsize=(6.6, 7.2), sharex=False)
for ax, f in zip(axes, CLS):
    e = ex[f]; th = np.arange(e["z"].shape[1]) / 60
    if f == "ARC":
        ax.plot(th, np.cumsum(e["narc"]), color=CAT[0], label="cumulative recycles")
        ax.set_ylabel("count")
    else:
        k = 60
        for c in range(5):
            zs = np.convolve(e["z"][c], np.ones(k) / k, mode="same"); zs[-k:] = np.nan; zs[:k] = np.nan
            ax.plot(th, zs, color=CAT[c], lw=1.1, label=ts.CH_NAMES[c].replace("Prf", "P$_{RF}$").replace("mdot", "ṁ").replace("Vn", "V$_n$"))
        ax.set_ylabel("z (60-min mean)")
    ax.axvline(e["t0"] / 60, color=INK2, ls="--", lw=0.9)
    if e["alarm"] >= 0:
        ax.axvline(e["alarm"] / 60, color="#e34948", lw=1.1)
    ax.set_title(f"{LBL[f]}: onset (dashed) at {e['t0']/60:.0f} h, alarm (solid) at {e['alarm']/60:.1f} h",
                 fontsize=8.5, loc="left", color=INK)
    ax.set_xlim(0, 300)
axes[0].legend(ncol=5, fontsize=7, loc="upper left", frameon=False)
axes[2].legend(fontsize=7, loc="upper left", frameon=False)
axes[-1].set_xlabel("time [h]")
fig.tight_layout(); fig.savefig("fig2_examples.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 3: detection delays
fig, ax = plt.subplots(figsize=(6.6, 2.8))
dets = ["CUSUM", "KF", "BOCPD", "VOTE"]
names = {"CUSUM": "CUSUM", "KF": "Kalman", "BOCPD": "BOCPD", "VOTE": "Fused (2-of-3)"}
w = 0.19
for i, d in enumerate(dets):
    data, pos = [], []
    for j, f in enumerate(CLS):
        if f == "ARC":
            if d == "VOTE":
                data.append(R["detection"][f]["VOTE"]["delays"]); pos.append(j + (i - 1.5) * w)
            continue
        data.append(R["detection"][f][d]["delays"]); pos.append(j + (i - 1.5) * w)
    bp = ax.boxplot(data, positions=pos, widths=w * 0.85, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=INK2, lw=0.8),
                    capprops=dict(color=INK2, lw=0.8))
    for b in bp["boxes"]:
        b.set(facecolor=CAT[i], edgecolor="white", lw=1)
    ax.plot([], [], "s", color=CAT[i], label=names[d])
ax.set_xticks(range(5)); ax.set_xticklabels(["Grid\nerosion", "Cathode/neutr.\nwear", "Arcing", "Feed\ndegradation", "RF-circuit\ndegradation"], fontsize=7.5)
ax.set_yscale("log"); ax.set_ylabel("detection delay [h]")
ax.text(2, 1.3, "Poisson CUSUM\n(recycle count)", ha="center", fontsize=7, color=INK2)
ax.legend(ncol=4, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig("fig3_delays.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 4: confusion matrix (int8)
cm = np.array(R["iso"]["cm_int8"])
fig, ax = plt.subplots(figsize=(3.6, 3.1)); ax.grid(False)
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
for i in range(5):
    for j in range(5):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > 50 else INK)
short = ["Grid", "Cathode", "Arc", "Feed", "RF"]
ax.set_xticks(range(5)); ax.set_xticklabels(short, fontsize=7.5)
ax.set_yticks(range(5)); ax.set_yticklabels(short, fontsize=7.5)
ax.set_xlabel("isolated class"); ax.set_ylabel("true class")
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig("fig4_confusion.png", bbox_inches="tight"); plt.close(fig)

# ---------------- Fig 5: closed-loop examples
thr = R["thresholds"]
import run_experiments as rx
def closed_loop(fault, t0h, tauh, sev, seed):
    rng = np.random.default_rng(seed)
    p = ts.make_scenarios(1, fault, rx.N, rng)
    p["t0"][:] = t0h * 60; p["tau"][:] = tauh * 60; p["sev"][:] = sev
    rng = np.random.default_rng(seed + 1)
    y, narc, Tp, tr0 = ts.simulate(p, rx.N, rng)
    g, _ = ts.simulate(p, rx.N, rng, model=True, Tp=Tp)
    z = fd.normalised_residuals(y, g, narc, Tp).astype(np.float32)
    cs = fd.cusum(z); kt, kb, _ = fd.kalman_bias(z); bo = fd.bocpd(z); ac = fd.poisson_cusum(narc)
    d = dict(cs_max=cs.max(1), kt_max=kt.max(1), bo_max=bo.max(1), ac=ac)
    a = rx.alarms(d, thr)["VOTE"][0]
    tiso = a + rx.ISO_WIN
    Xf = fd.features(kb, z, ac, np.array([tiso]))
    m = R["model"]; qm = dict(Wq=np.array(m["Wq"]), bq=np.array(m["bq"]), mu=np.array(m["mu"]),
                              sd=np.array(m["sd"]), in_scale=16.0)
    cls = int(fd.predict_q(qm, Xf)[0])
    rng = np.random.default_rng(seed + 1)
    y1, _, _, tr1 = ts.simulate(p, rx.N, rng, recovery=dict(t_r=np.array([float(tiso)]),
                                                             action=np.array([cls + 1])), Tp=Tp)
    return y, y1, tr0, tr1, a, tiso, cls
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
y0, y1, tr0, tr1, a, tiso, cls = closed_loop("GRID", 50, 150, 1.2, 11)
th = np.arange(rx.N) / 60
ax = axes[0]
ax.plot(th, tr0["i_ebs"][0], color=CAT[1], label="no action")
ax.plot(th, tr1["i_ebs"][0], color=CAT[0], label="FDIR: V$_a$ correction")
ax.axvline(tiso / 60, color="#e34948", lw=1)
ax.set_title(f"(a) Grid erosion, isolated as {fd.CLASSES[cls]}", fontsize=8.5, loc="left")
ax.set_xlabel("time [h]"); ax.set_ylabel("backstreaming current [mA]"); ax.legend(fontsize=7, frameon=False)
G = closed_loop("FEED", 60, 60, 0.18, 21)
ax = axes[1]
k = 380
def sm(v):
    o = np.convolve(v, np.ones(k) / k, "same"); o[:k] = np.nan; o[-k:] = np.nan; return o
ax.plot(th, sm(G[0][0, 3]), color=CAT[1], label="no action")
ax.plot(th, sm(G[1][0, 3]), color=CAT[0], label="FDIR: feed setpoint")
ax.axvline(G[5] / 60, color="#e34948", lw=1)
ax.set_title(f"(b) Feed degradation, isolated as {fd.CLASSES[G[6]]}", fontsize=8.5, loc="left")
ax.set_xlabel("time [h]"); ax.set_ylabel("ṁ, 4-orbit mean [µg/s]"); ax.set_xlim(30, 300); ax.set_ylim(24, 32)
ax.legend(fontsize=7, frameon=False, loc="center right")
fig.tight_layout(); fig.savefig("fig5_closedloop.png", bbox_inches="tight"); plt.close(fig)
print("GRID example alarm h", a / 60, "iso", tiso / 60, "FEED alarm", G[4] / 60)
