"""ON (mini-BZ head average) vs OFF (point-value head) exciton bands overlay,
both from the 16-GPU cusolverMp .dat files.  Deliverable: the A/B overlay PNG
+ the per-Q head shift.  Host-only (matplotlib Agg)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_dat import load_dat

OFF = sys.argv[1] if len(sys.argv) > 1 else "exciton_bands_16gpu_off.dat"
ON = sys.argv[2] if len(sys.argv) > 2 else "exciton_bands_16gpu_on.dat"
OUT = sys.argv[3] if len(sys.argv) > 3 else "exciton_bands_16gpu_on_vs_off.png"

iO, sO, QO, EO = load_dat(OFF)
iN, sN, QN, EN = load_dat(ON)
assert np.array_equal(iO, iN) and EO.shape == EN.shape

dE_meV = (EN - EO) * 1e3
per_q_max = np.max(np.abs(dE_meV), axis=1)
gmax = float(np.max(np.abs(dE_meV)))
worst = int(np.argmax(per_q_max))
print(f"# ON(mini-BZ) vs OFF(point) head shift: GLOBAL max|dE| = {gmax:.4f} meV "
      f"at iQ={iO[worst]} s={sO[worst]:.4f} |Q|={np.linalg.norm(QO[worst]):.5f}")
print(f"# mean|dE| = {float(np.mean(np.abs(dE_meV))):.4f} meV; "
      f"per-Q max (meV): min {per_q_max.min():.4f}, max {per_q_max.max():.4f}")
print(f"# near-Gamma per-Q max|dE| (first 4 Q past Γ): "
      + " ".join(f"{per_q_max[j]:.4f}" for j in range(1, 5)))

neig = EO.shape[1]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.0, 6.4),
                              gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
for b in range(neig):
    ax.plot(sO, EO[:, b], color="C0", lw=1.1, alpha=0.85,
            label="OFF (point-value head)" if b == 0 else None)
    ax.plot(sN, EN[:, b], color="C3", lw=1.0, ls="--", alpha=0.85,
            label="ON (mini-BZ head average)" if b == 0 else None)
ax.set_ylabel("$E_S(Q)$ (eV)")
ax.set_title("MoS$_2$ 12×12 exciton bands (TDA), 16-GPU cusolverMp — head A/B")
ax.legend(loc="best", fontsize="small")
ax2.plot(sO, per_q_max, color="k", lw=1.2)
ax2.set_ylabel("max$_b$|ΔE|\n(meV)")
ax2.set_xlabel("path distance")
ax2.axhline(0, color="gray", lw=0.5)
for a in (ax, ax2):
    a.set_xlim(sO[0], sO[-1])
fig.tight_layout()
fig.savefig(OUT, dpi=180)
print(f"# wrote {OUT}")
np.savez(OUT + ".npz", iQ=iO, s_path=sO, Q=QO, E_off=EO, E_on=EN,
         dE_meV=dE_meV, per_q_max_meV=per_q_max, gmax_meV=gmax)
