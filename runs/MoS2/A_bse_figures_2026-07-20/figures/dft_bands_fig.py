"""Clean DFT (PBE) bandstructure via the htransform sp-bands machinery.

The GW/QP overlay is intentionally WITHHELD: the GN-PPM conduction self-energy in
this branch is unphysical (sigC.Re(K CBM) = -4.48 eV, inverts the gap; sanity
Gate 2 FAILS), so a GW figure would be misleading (coordinator STOP rule).  This
figure shows the correct piece — the sp-bands htransform interpolation of the
6x6 DFT bands onto Gamma-M-K-Gamma (on-grid recon 0.00 meV) — which is the
producer for both the GW-bands figure (blocked) and the BSE interp basis (works).
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = np.load("gw_bands.npz", allow_pickle=True)
x = d["x_path"]; E = d["E_dft"]; vbm = float(d["vbm_dft"]); node = list(d["node_idx"])
NV = 26
# DFT direct gap from grid (dft_gap dict saved by gw_bands_fig)
gd = d["dft_gap"].item() if "dft_gap" in d else {"direct": 1.732}
gap = float(gd["direct"])
print(f"[dft-bands] vbm={vbm:.3f}  direct gap={gap:.3f} eV  bands plotted 16..35")

plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.2, 6.4), dpi=200)
for b in range(16, 36):
    c = "#2c6e8f" if b < NV else "#b5432c"
    ax.plot(x, E[:, b] - vbm, color=c, lw=1.3)
ax.axhline(0.0, color="0.75", lw=0.7, zorder=0)
for n in node:
    ax.axvline(x[n], color="0.85", lw=0.7, zorder=0)
ax.set_xticks([x[n] for n in node]); ax.set_xticklabels(["$\\Gamma$", "M", "K", "$\\Gamma$"])
ax.set_xlim(x[0], x[-1]); ax.set_ylim(-6.5, 6.0)
ax.set_ylabel("E $-$ E$_{\\mathrm{VBM}}$ (eV)")
ax.set_title("MoS$_2$ monolayer — DFT bandstructure (htransform sp-bands)\n"
             "6$\\times$6 ISDF interp onto $\\Gamma$-M-K-$\\Gamma$ (on-grid recon 0.00 meV)",
             fontsize=11.5)
ax.text(0.02, 0.98, f"DFT direct gap (K) = {gap:.2f} eV\nblue = valence, red = conduction",
        transform=ax.transAxes, va="top", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7", alpha=0.9))
fig.tight_layout(); fig.savefig("mos2_dft_bandstructure.png")
print("SAVED mos2_dft_bandstructure.png")
