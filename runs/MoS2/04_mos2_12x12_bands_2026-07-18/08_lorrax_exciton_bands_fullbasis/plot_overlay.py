"""Overlay: OLD 640c narrow-window (24,32) exciton bands vs NEW full-band
(nband=40) exciton bands, SAME 12x12 grid + SAME 40-pt Gamma-M-K-Gamma path +
SAME 640 centroids + SAME restart/V_Q/W.  The ONLY difference is the htransform
fH window that produced the conduction caches eps_c(k+Q)/psi_c(k+Q): a sliver
(24,32) whose top boundary cuts a 5.9 meV near-degenerate pair (rings off-grid)
vs the full-band window with guard bands (interior selection, smooth).

Prediction: on-grid rows (iQ 0,5,10,15=M,19,23=K,...) coincide (htransform is
on-grid exact both ways); the off-grid dips at iQ 6/9/16-17 present in the
narrow-window curve VANISH in the full-band curve.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OLD = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/04_mos2_12x12_bands_2026-07-18/"
       "01_lorrax_exciton_bands/exciton_bands_12x12_GMKG.dat")
NEW = "exciton_bands_fullbasis.dat"
OUT = "exciton_bands_fullbasis_vs_640c_GMKG.png"
DIP_IQ = [6, 9, 16, 17]


def load(path):
    interp, nodes = [], None
    with open(path, encoding="utf8") as fh:
        for ln in fh:
            if ln.startswith("# nodes:"):
                nodes = [(int(tok.split(":")[0]), tok.split(":")[1])
                         for tok in ln.split()[2:]]
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            if t[5] != "interp":
                continue
            interp.append((int(t[0]), float(t[1]), [float(x) for x in t[6:]]))
    interp.sort(key=lambda r: r[0])
    iq = np.array([r[0] for r in interp])
    s = np.array([r[1] for r in interp])
    E = np.array([r[2] for r in interp])
    return iq, s, E, nodes


iq_o, s_o, E_o, nodes = load(OLD)
iq_n, s_n, E_n, nodes_n = load(NEW)
node_s = [s_n[i] for i, _ in nodes_n]

fig, ax = plt.subplots(figsize=(7.4, 5.0))
for b in range(E_o.shape[1]):
    ax.plot(s_o, E_o[:, b], lw=1.0, color="0.6", ls="--",
            label="640c narrow window (24,32) — has dips" if b == 0 else None)
for b in range(E_n.shape[1]):
    ax.plot(s_n, E_n[:, b], lw=1.4, color="C0",
            label="640c full band (nband=40) — this fix" if b == 0 else None)
# annotate the artifact points
for q in DIP_IQ:
    io = np.where(iq_o == q)[0]
    if io.size:
        ax.scatter([s_o[io[0]]], [E_o[io[0], 0]], s=40, marker="v",
                   color="C3", zorder=6,
                   label="narrow-window dip (iQ 6/9/16/17)" if q == DIP_IQ[0] else None)
for xpos in node_s:
    ax.axvline(xpos, color="k", lw=0.6, alpha=0.3)
ax.set_xticks(node_s, [l for _, l in nodes_n])
ax.set_xlim(s_n[0], s_n[-1])
ax.set_ylabel("$E_S(Q)$ (eV)")
ax.set_title("MoS2 exciton bands (TDA, 12x12): full-band vs narrow-window htransform caches")
ax.legend(loc="best", fontsize="small")
fig.text(0.5, -0.03,
         "Identical grid/path/centroids/restart; only the htransform fH window differs.\n"
         "The iQ 6/9/16-17 dips are off-grid conduction-cache window artifacts; the full-band\n"
         "basis (all 26v+14c, guard bands above the BSE conduction selection) removes them.",
         ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT, dpi=180, bbox_inches="tight")

# quantitative dip verdict
print(f"Wrote {OUT}")
print("\n iQ   Q            E_1 narrow   E_1 full    dE1(meV)   verdict")
for q in DIP_IQ:
    io = np.where(iq_o == q)[0]
    inn = np.where(iq_n == q)[0]
    if io.size and inn.size:
        e_o = E_o[io[0], 0]
        e_n = E_n[inn[0], 0]
        d = (e_n - e_o) * 1e3
        print(f" {q:3d}  {'':2}         {e_o:8.4f}    {e_n:8.4f}   {d:+8.1f}   "
              f"{'DIP LIFTED' if d > 50 else 'unchanged'}")
# on-grid coincidence check (iQ 0,5,10,15,19,23)
print("\n on-grid rows (should coincide within Lanczos/interp floor):")
for q in [0, 5, 10, 15, 19, 23]:
    io = np.where(iq_o == q)[0]
    inn = np.where(iq_n == q)[0]
    if io.size and inn.size:
        d = np.abs(E_o[io[0]] - E_n[inn[0]]).max() * 1e3
        print(f" iQ {q:3d}: max|dE| over 8 states = {d:8.3f} meV")
