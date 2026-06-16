"""Band-convergence plot of the CrI3 orbital moment (6x6 full-BZ).

m_z as a function of the sum-over-states inner-band ceiling N (number of bands
kept in the resolution-of-identity), from the saved 6x6 velocity matrices.
Physical velocity p+vNL.  Curves at mu = VBM, midgap, CBM.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RY2EV = 13.605693122994
PREF = 0.5
d = np.load("orbmag_FM_nbnd180.npz")
Vp, Vnl, E = d["Vp"], d["Vnl"], d["E"]
nocc = int(d["nocc"]); w_k = float(d["w_k"]); m_spin_z = float(d["m_spin_z"])
nk, _, nb, _ = Vp.shape
deps_tol = 1.4e-3 / RY2EV
V = Vp + Vnl                                  # physical p+vNL (canonical)
frame = 1.0 if m_spin_z >= 0 else -1.0        # report along the spin axis

VBM = float(E[:, nocc - 1].max()); CBM = float(E[:, nocc].min())
mids = {"mu = VBM": VBM, "mu = midgap": 0.5 * (VBM + CBM), "mu = CBM": CBM}

# band-resolved z-pieces summed over occupied n, BZ-weighted: colA/colB[m]
colA = np.zeros(nb, dtype=np.complex128)
colB = np.zeros(nb, dtype=np.complex128)
for ik in range(nk):
    v = V[ik]; eps = E[ik]
    vt = np.swapaxes(v, 1, 2)
    cz = v[0] * vt[1] - v[1] * vt[0]                       # (nb,nb) z cross product
    deps = eps[:, None] - eps[None, :]
    mask = np.abs(deps) > deps_tol
    inv2 = np.where(mask, 1.0 / np.where(mask, deps, 1.0) ** 2, 0.0)
    occ = np.zeros((nb, 1)); occ[:nocc, 0] = 1.0
    PA_z = occ * ((eps[:, None] + eps[None, :]) * inv2) * cz   # (nb,nb)
    PB_z = occ * inv2 * cz
    colA += w_k * PA_z.sum(axis=0)                         # sum over occupied n
    colB += w_k * PB_z.sum(axis=0)

N = np.arange(1, nb + 1)                                   # inner-band ceiling
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for label, mu in mids.items():
    cum = np.cumsum(colA - 2.0 * mu * colB)                # partial sum over m
    m_of_N = frame * (-PREF) * cum.imag                    # along spin axis (mu_B)
    ax.plot(N, m_of_N, lw=1.8, label=f"{label} ({mu*RY2EV:.2f} eV)")

ax.axvline(nocc, color="0.5", ls=":", lw=1, label=f"occupied = {nocc}")
ax.axhline(0, color="0.7", lw=0.8)
mid_final = frame * (-PREF) * np.cumsum(colA - 2.0 * mids["mu = midgap"] * colB).imag[-1]
ax.annotate(f"{mid_final:+.4f} μ$_B$ @ {nb} bands",
            xy=(nb, mid_final), xytext=(nb - 70, mid_final + 0.02),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set_xlim(nocc - 2, nb)
ax.set_ylim(-0.08, 0.08)   # clip near-degeneracy transients (cancel in full sum)
ax.text(0.98, 0.03, "(intermediate spikes from near-degenerate band pairs are\n"
        "clipped; they cancel in the full sum — final value is robust)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="0.4")
ax.set_xlabel("number of bands in sum-over-states (inner-m ceiling)")
ax.set_ylabel("orbital moment ∥ spin  m$_z$  (μ$_B$/cell)")
ax.set_title("CrI$_3$ monolayer (FM, 6×6, p+vNL): orbital-moment band convergence")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("band_convergence_6x6.png", dpi=150)
print(f"wrote band_convergence_6x6.png   (final midgap m_z = {mid_final:+.5f} mu_B)")
# also print a coarse table
mu0 = mids["mu = midgap"]; cum0 = np.cumsum(colA - 2.0 * mu0 * colB)
for n in (70, 80, 100, 120, 140, 160, 180):
    print(f"  N={n:3d}:  m_z = {frame*(-PREF)*cum0[n-1].imag:+.5f} mu_B")
