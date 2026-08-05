#!/usr/bin/env python3
"""Plot DFT and GW bandstructures from htransform output. Best-effort / tolerant
to the exact bandstructure.dat column layout (col0 = k-distance, cols1.. = band
energies). Usage: python3 plot_bands.py <system_label> [dft.dat] [gw.dat]
Run post-maintenance once htransform has produced the .dat files."""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

label = sys.argv[1] if len(sys.argv) > 1 else "system"
dft = sys.argv[2] if len(sys.argv) > 2 else "bandstructure.dat"
gw  = sys.argv[3] if len(sys.argv) > 3 else "bandstructure_gw.dat"

def load(path):
    try:
        a = np.loadtxt(path, comments=["#", "!"])
    except Exception as e:
        print(f"[plot_bands] could not read {path}: {e}"); return None
    if a.ndim == 1: a = a[None, :]
    return a

fig, ax = plt.subplots(figsize=(6.0, 5.0))
d = load(dft)
if d is not None:
    k, E = d[:, 0], d[:, 1:]
    for j in range(E.shape[1]):
        ax.plot(k, E[:, j], color="0.55", lw=0.9, zorder=1,
                label="DFT" if j == 0 else None)
g = load(gw)
if g is not None:
    k, E = g[:, 0], g[:, 1:]
    for j in range(E.shape[1]):
        ax.plot(k, E[:, j], color="crimson", lw=1.1, zorder=2,
                label="GW (GN-PPM)" if j == 0 else None)

# Gamma-M-K-Gamma ticks (40+40+40+1 path)
if d is not None:
    kk = d[:, 0]
    # segment boundaries at indices 0, 40, 80, 120 (approx by fraction)
    n = len(kk)
    idx = [0, int(n*40/121), int(n*80/121), n-1]
    for i in idx[1:-1]:
        ax.axvline(kk[i], color="0.8", lw=0.7)
    ax.set_xticks([kk[i] for i in idx]); ax.set_xticklabels([r"$\Gamma$","M","K",r"$\Gamma$"])
ax.set_ylabel("E (eV)"); ax.set_title(f"{label}: DFT vs GW (GN-PPM) bandstructure")
ax.legend(loc="best", fontsize=9)
fig.tight_layout(); out = f"bands_{label}.png"; fig.savefig(out, dpi=160)
print(f"[plot_bands] wrote {out}")
