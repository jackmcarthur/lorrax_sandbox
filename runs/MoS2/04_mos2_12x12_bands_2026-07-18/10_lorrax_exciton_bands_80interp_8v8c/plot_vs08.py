"""Overlay: run-08 (4v4c BSE window / nband=40 htransform interp) vs run-10
(8v8c BSE window / nband=80 htransform interp) exciton bands, SAME 12x12 grid,
SAME 40-pt Gamma-M-K-Gamma path, SAME 640 centroids, SAME restart/V_Q/W.  The
differences are (a) BSE window 4v4c -> 8v8c and (b) htransform interp basis
40 -> 80 bands (+ retuned a_band).  Does 8v8c/80-interp change the picture?"""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
R08 = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/04_mos2_12x12_bands_2026-07-18/"
       "08_lorrax_exciton_bands_fullbasis/exciton_bands_fullbasis.dat")
# The deliverable interp basis is nband=40 (nband=80 is broken; see sp_reconcile).
R10 = os.environ.get("R10", "exciton_bands_40interp_8v8c.dat")
OUT = "exciton_vs08_overlay.png"

def load(path):
    interp, nodes = [], None
    with open(path, encoding="utf8") as fh:
        for ln in fh:
            if ln.startswith("# nodes:"):
                nodes = [(int(t.split(":")[0]), t.split(":")[1]) for t in ln.split()[2:]]
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            if t[5] != "interp":
                continue
            interp.append((int(t[0]), float(t[1]), [float(x) for x in t[6:]]))
    interp.sort(key=lambda r: r[0])
    s = np.array([r[1] for r in interp]); E = np.array([r[2] for r in interp])
    return s, E, nodes

s8, E8, nodes = load(R08)
s10, E10, nodes10 = load(R10)
nb = min(E8.shape[1], E10.shape[1])
node_s = [s10[i] for i, _ in nodes10]

fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=160)
for b in range(nb):
    ax.plot(s8, E8[:, b], lw=1.0, color="0.6", ls="--",
            label="run 08: 4v4c / 40-interp" if b == 0 else None)
for b in range(nb):
    ax.plot(s10, E10[:, b], lw=1.5, color="#b5432c",
            label="run 10: 8v8c / 40-interp" if b == 0 else None)
for xs in node_s:
    ax.axvline(xs, color="k", lw=0.6, alpha=0.3)
ax.set_xticks(node_s); ax.set_xticklabels(["Γ", "M", "K", "Γ"])
ax.set_xlim(s10[0], s10[-1])
ax.set_ylabel("$E_S(Q)$ (eV)")
ax.set_title("MoS₂ exciton bands: 8v8c/40-interp (run 10) vs 4v4c/40-interp (run 08)")
ax.legend(loc="best", fontsize=9)

# quick numeric delta of E_1 at shared on-grid nodes
d1 = (E10[:, 0] - E8[:, 0]) * 1e3   # meV, E_1
txt = (f"ΔE₁ (10−08): median {np.median(d1):+.1f}, "
       f"mean {np.mean(d1):+.1f}, max|Δ| {np.max(np.abs(d1)):.1f} meV")
ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=8,
        va="bottom", ha="left", bbox=dict(boxstyle="round", fc="w", ec="0.7", alpha=0.8))
fig.tight_layout(); fig.savefig(OUT)
print(f"SAVED {OUT}")
print(txt)
print(f"E_1 run08 Gamma={E8[0,0]:.4f}  run10 Gamma={E10[0,0]:.4f} eV")
print(f"E_1 run08 K={E8[nodes[2][0],0]:.4f}  run10 K={E10[nodes10[2][0],0]:.4f} eV")
