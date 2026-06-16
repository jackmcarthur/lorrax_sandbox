"""Band-convergence plot from a dumped npz (colA_z/colB_z).  Usage: plot_band_conv_N.py <npz> <out.png>"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RY2EV = 13.605693122994; PREF = 0.5
npz = sys.argv[1] if len(sys.argv) > 1 else "orbmag_FM_6x6_2000.npz"
png = sys.argv[2] if len(sys.argv) > 2 else "band_convergence_6x6_2000.png"
d = np.load(npz)
colA, colB = d["colA_z"], d["colB_z"]; E = d["E"]; nocc = int(d["nocc"])
m_spin_z = float(d["m_spin_z"]); nb = colA.shape[0]
frame = 1.0 if m_spin_z >= 0 else -1.0
VBM = float(E[:, nocc - 1].max()); CBM = float(E[:, nocc].min())
mids = {"mu = VBM": VBM, "mu = midgap": 0.5 * (VBM + CBM), "mu = CBM": CBM}
N = np.arange(1, nb + 1)

fig, ax = plt.subplots(figsize=(7.8, 4.8))
final_mid = None; cum_mid = None
for label, mu in mids.items():
    c = frame * (-PREF) * np.cumsum(colA - 2.0 * mu * colB).imag
    ax.plot(N, c, lw=1.2, label=f"{label} ({mu*RY2EV:.2f} eV)")
    if "midgap" in label:
        final_mid = c[-1]; cum_mid = c
# tail extrapolation: cum(N) = cum_inf + a*N^-1.9  (since per-band inc ~ band^-2.9)
sel = N >= max(400, nb // 3)
A = np.vstack([np.ones(sel.sum()), N[sel].astype(float) ** -1.9]).T
coef, *_ = np.linalg.lstsq(A, cum_mid[sel], rcond=None)
cum_inf = coef[0]
ax.axhline(cum_inf, color="crimson", ls="-", lw=1.1,
           label=f"extrapolated N→∞: {cum_inf:+.4f} μ$_B$")
ax.axvline(nocc, color="0.6", ls=":", lw=1, label=f"occupied = {nocc}")
for nn in (180, 400):
    if nn < nb:
        ax.axvline(nn, color="0.8", ls="--", lw=0.8)
ax.axhline(0, color="0.7", lw=0.7)
ax.set_xlim(nocc - 2, nb); ax.set_ylim(-0.12, 0.06)
ax.set_xlabel("number of bands in sum-over-states (inner-m ceiling)")
ax.set_ylabel("orbital moment ∥ spin  m$_z$  (μ$_B$/cell)")
ax.set_title(f"CrI$_3$ (FM, 6×6 IBZ, p+vNL): orbital-moment band convergence to {nb}")
ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(png, dpi=150)
print(f"{png}: final m_z @ {nb} = {final_mid:+.5f};  extrapolated N->inf = {cum_inf:+.5f} mu_B")
for n in (180, 400, 800, 1200, 1600, 2000):
    if n <= nb:
        print(f"  N={n:4d}: m_z = {cum_mid[n-1]:+.5f}")
print(f"  tail mean [{nb-200},{nb}] = {cum_mid[nb-200:].mean():+.5f} +/- {cum_mid[nb-200:].std():.5f}")
