"""Three-way overlay closing the smoothness question:
  640c  production window (24,32)   — delivered run
  1000c production window (24,32)   — ISDF-convergence A/B
  1000c clean window   (22,34)      — the mechanism test
Prediction: the iQ 6/9/16-17 dips (htransform window-cache artifacts of
the (24,32) window's 5.9 meV 31|32 Kramers-adjacent boundary) VANISH in
the clean-window curve.  Second panel: the clean-window free-pair floor
D_min(Q) (same window/a as the clean driver caches) — where E_S(Q)
tracks D_min structure, it is single-particle kinematics."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C640, C1000, CCLEAN, C_MARK, C_DMIN = ("0.55", "#2a78d6", "#eb6834",
                                       "#eda100", "#4a3aa7")
RUNS = [
    ("../01_lorrax_exciton_bands/exciton_bands_12x12_GMKG.dat",
     C640, "-", 0.9, "640c, production window (24,32)"),
    ("../03_lorrax_exciton_bands_1000c/exciton_bands_1000c.dat",
     C1000, "--", 1.1, "1000c, production window (24,32)"),
    ("exciton_bands_1000c_cleanwin.dat",
     CCLEAN, "-", 1.5, "1000c, CLEAN window (22,34)"),
]
ARTIFACT_IQ = [6, 9, 16, 17]
OUT = "exciton_bands_threeway_GMKG.png"


def load(path):
    rows, nodes = [], None
    with open(path, encoding="utf8") as fh:
        for ln in fh:
            if ln.startswith("# nodes:"):
                nodes = [(int(t.split(":")[0]), t.split(":")[1])
                         for t in ln.split()[2:]]
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            if t[5] == "interp":
                rows.append((int(t[0]), float(t[1]),
                             [float(x) for x in t[6:]]))
    rows.sort()
    return (np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]), nodes)


data = [(load(p), c, ls, lw, lab) for p, c, ls, lw, lab in RUNS]
(s0, E0, nodes) = data[0][0]
s_c, E_c = data[2][0][0], data[2][0][1]
E_p1000 = data[1][0][1]

# dip metrics: local depth of E_1 at the artifact rows vs neighbor mean
print("E_1 local dip depth (meV) [neighbors mean minus row]:")
for i in ARTIFACT_IQ:
    for (s, E, _n), _c, _ls, _lw, lab in data:
        nb = 0.5 * (E[i - 1, 0] + E[i + 1, 0])
        print(f"  iQ {i:2d}  {lab:34s} {1e3 * (nb - E[i, 0]):8.1f}")
d = np.abs(E_c - E_p1000) * 1e3
mask = np.ones(len(s_c), bool)
mask[ARTIFACT_IQ] = False
print(f"|dE(clean-win − prod-win)| @1000c: overall median "
      f"{np.median(d):.1f} meV; artifact rows median {np.median(d[~mask]):.1f} "
      f"max {d[~mask].max():.1f}; elsewhere max {d[mask].max():.1f}")

dm = np.loadtxt("dmin_2234_GMKG.dat")

fig, (ax, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(7.6, 7.6),
    gridspec_kw={"height_ratios": [2.1, 1.0], "hspace": 0.07})
for i in ARTIFACT_IQ:
    lo = 0.5 * (s0[i - 1] + s0[i])
    hi = 0.5 * (s0[i] + s0[i + 1])
    for a in (ax, ax2):
        a.axvspan(lo, hi, color=C_MARK, alpha=0.12,
                  label=("(24,32) window-artifact rows (iQ 6/9/16-17)"
                         if (i == ARTIFACT_IQ[0] and a is ax) else None))
for (s, E, _n), c, ls, lw, lab in data:
    for b in range(E.shape[1]):
        ax.plot(s, E[:, b], lw=lw, ls=ls, color=c,
                label=lab if b == 0 else None)
node_x = [s0[i] for i, _ in nodes]
for a in (ax, ax2):
    for xv in node_x:
        a.axvline(xv, color="k", lw=0.6, alpha=0.25)
ax.set_ylabel("$E_S(Q)$ (eV)")
ax.set_title("MoS$_2$ exciton bands (TDA, 12$\\times$12): the htransform-"
             "window mechanism test", fontsize=11)
ax.legend(loc="lower right", fontsize="small", framealpha=0.9)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.3)

ax2.plot(dm[:, 1], dm[:, 5], lw=1.6, color=C_DMIN,
         label=r"$D_\min(Q)$, clean window (22,34)@1000c (driver's caches)")
ax2.set_ylabel("energy (eV)")
ax2.set_xticks(node_x, [l for _, l in nodes])
ax2.set_xlim(s0[0], s0[-1])
ax2.legend(loc="lower right", fontsize="small", framealpha=0.9)
ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.3)

fig.text(0.5, -0.03,
         "Same restart physics and path everywhere.  The two production-"
         "window curves share the shaded dips (640c vs 1000c: basis does "
         "not remove them);\nthe clean-window curve tests the mechanism — "
         "window boundary cutting Kramers-degenerate pairs in the "
         "htransform caches.\nPanel 2: the clean free-pair floor; "
         "remaining $E_S(Q)$ structure that tracks it is single-particle "
         "kinematics.",
         ha="center", fontsize=8)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Wrote {OUT}")
