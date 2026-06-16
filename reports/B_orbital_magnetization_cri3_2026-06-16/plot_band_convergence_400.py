"""Band-convergence plot to 400 bands (6x6 IBZ), from the dumped colA_z/colB_z."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RY2EV = 13.605693122994
PREF = 0.5
d = np.load("orbmag_FM_6x6_400.npz")
colA, colB = d["colA_z"], d["colB_z"]          # (nb,) z-columns (occ-summed, BZ-wt)
E = d["E"]; nocc = int(d["nocc"]); m_spin_z = float(d["m_spin_z"])
nb = colA.shape[0]
frame = 1.0 if m_spin_z >= 0 else -1.0
VBM = float(E[:, nocc - 1].max()); CBM = float(E[:, nocc].min())
mids = {"mu = VBM": VBM, "mu = midgap": 0.5 * (VBM + CBM), "mu = CBM": CBM}

N = np.arange(1, nb + 1)
fig, ax = plt.subplots(figsize=(7.6, 4.7))
final_mid = None
for label, mu in mids.items():
    m_of_N = frame * (-PREF) * np.cumsum(colA - 2.0 * mu * colB).imag
    ax.plot(N, m_of_N, lw=1.4, label=f"{label} ({mu*RY2EV:.2f} eV)")
    if "midgap" in label:
        final_mid = m_of_N[-1]
ax.axvline(nocc, color="0.5", ls=":", lw=1, label=f"occupied = {nocc}")
ax.axvline(180, color="0.7", ls="--", lw=1, label="180 (previous)")
ax.axhline(0, color="0.7", lw=0.8)
ax.annotate(f"{final_mid:+.4f} μ$_B$ @ {nb} bands", xy=(nb, final_mid),
            xytext=(nb - 150, final_mid + 0.018), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set_xlim(nocc - 2, nb)
ax.set_ylim(-0.08, 0.08)
ax.text(0.99, 0.02, "(near-degeneracy transients clipped; cancel in full sum)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.5")
ax.set_xlabel("number of bands in sum-over-states (inner-m ceiling)")
ax.set_ylabel("orbital moment ∥ spin  m$_z$  (μ$_B$/cell)")
ax.set_title("CrI$_3$ monolayer (FM, 6×6 IBZ, p+vNL): band convergence to 400")
ax.legend(fontsize=8, loc="upper left", ncol=2); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("band_convergence_6x6_400.png", dpi=150)
print(f"final midgap m_z @ {nb} bands = {final_mid:+.5f} mu_B")
mu0 = mids["mu = midgap"]; c = frame * (-PREF) * np.cumsum(colA - 2.0 * mu0 * colB).imag
for n in (70, 120, 180, 240, 300, 360, 400):
    if n <= nb:
        print(f"  N={n:3d}:  m_z = {c[n-1]:+.5f} mu_B")
# rolling-mean tail estimate (last 60 bands) as a convergence indicator
print(f"  mean over N in [{nb-60},{nb}] = {c[nb-60:].mean():+.5f} +/- {c[nb-60:].std():.5f}")
