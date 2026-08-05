"""Overlay: delivered 3x3-coarse-grid exciton bands vs this run's 12x12.

Reads both .dat files (exciton_bands driver format: iQ s_path Qx Qy Qz mode
E_1..E_neig), separates interp rows from refit spot checks, and plots the two
runs on SHARED axes with the s_path of EACH run rescaled to its own node
positions (the two paths have different point counts; nodes G/M/K/G align by
construction).  Caption states the interpretation split: band SHIFTS between
the two curves are coarse-grid (k-convergence) physics; smoothness and
refit-marker agreement within a run are interpolation quality.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/"
       "B_exciton_bands_2026-07-17/exciton_bands_GMKG.dat")
NEW = "exciton_bands_12x12_GMKG.dat"
OUT = "exciton_bands_12x12_vs_3x3_GMKG.png"


def load(path):
    interp, refit, nodes = [], [], None
    with open(path, encoding="utf8") as fh:
        for ln in fh:
            if ln.startswith("# nodes:"):
                nodes = [(int(tok.split(":")[0]), tok.split(":")[1])
                         for tok in ln.split()[2:]]
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            row = (int(t[0]), float(t[1]), [float(x) for x in t[6:]])
            (interp if t[5] == "interp" else refit).append(row)
    interp.sort(key=lambda r: r[0])
    s = np.array([r[1] for r in interp])
    E = np.array([r[2] for r in interp])
    return s, E, refit, nodes


s_o, E_o, refit_o, nodes_o = load(OLD)
s_n, E_n, refit_n, nodes_n = load(NEW)
# common axis: rescale each run's s_path piecewise so its nodes land on the
# NEW run's node positions (paths are geometrically identical G-M-K-G)
node_s_n = [s_n[i] for i, _ in nodes_n]
node_s_o = [s_o[i] for i, _ in nodes_o]
s_o_r = np.interp(s_o, node_s_o, node_s_n)


def s_of(run_s, node_from, iq):
    return np.interp(run_s[iq], node_from, node_s_n)


fig, ax = plt.subplots(figsize=(7.0, 4.8))
for b in range(E_o.shape[1]):
    ax.plot(s_o_r, E_o[:, b], lw=1.0, color="0.55", ls="--",
            label="3x3 coarse grid (delivered)" if b == 0 else None)
for b in range(E_n.shape[1]):
    ax.plot(s_n, E_n[:, b], lw=1.3, color="C0",
            label="12x12 coarse grid (this run)" if b == 0 else None)
for j, (iq, _s, ev) in enumerate(refit_n):
    ax.scatter(np.full(len(ev), s_n[iq]), ev, s=24, facecolors="none",
               edgecolors="C3", zorder=5,
               label="12x12 refit ground truth" if j == 0 else None)
for xpos in node_s_n:
    ax.axvline(xpos, color="k", lw=0.6, alpha=0.3)
ax.set_xticks(node_s_n, [l for _, l in nodes_n])
ax.set_xlim(s_n[0], s_n[-1])
ax.set_ylabel("$E_S(Q)$ (eV)")
ax.set_title("MoS2 exciton bandstructure (TDA): 12x12 vs 3x3 coarse grid")
ax.legend(loc="best", fontsize="small")
fig.text(0.5, -0.04,
         "Runs differ ONLY in the coarse k-grid (3x3 vs 12x12; same 640 centroids, window, path).\n"
         "Band SHIFTS between curves = coarse-grid (k-convergence) physics; smoothness and\n"
         "refit-marker agreement within the 12x12 run = V_Q interpolation quality.",
         ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Wrote {OUT}")
