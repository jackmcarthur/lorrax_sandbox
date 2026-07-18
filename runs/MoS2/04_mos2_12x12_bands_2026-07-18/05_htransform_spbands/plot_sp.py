"""Render sp_bands_12x12_GMKG.png from the two .dat files (no GPU).

Panel 1: htransform single-particle bands (valence 22-25 + conduction
26-31), VBM-referenced.
Panel 2: free-pair floor D_min(Q) from BOTH windows — the exciton
driver's production (24,32) caches vs the structurally clean (26,34)
window — plus the delivered 640c exciton E_1(Q).  The iQ 6/9/16/17 dips
live ONLY in the driver-window curve (and the exciton bands track THEM):
htransform window-cache artifacts, not kinematics.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_VAL, C_COND, C_DMIN, C_DMIN2, C_EXC = ("#2a78d6", "#eb6834", "#4a3aa7",
                                         "#1baf7a", "#e34948")

sp = np.loadtxt("sp_bands_12x12_GMKG.dat")          # iQ s kx ky kz E22..E31
dm = np.loadtxt("dmin_12x12_GMKG.dat")              # iQ s Qx Qy Qz D D2634 kx ky
x_path = sp[:, 1]
bands_ev = sp[:, 5:15]
D_ref, D_clean = dm[:, 5], dm[:, 6]

nodes = [(0, "Γ"), (15, "M"), (23, "K"), (39, "Γ")]
rows = []
with open("../01_lorrax_exciton_bands/exciton_bands_12x12_GMKG.dat",
          encoding="utf8") as fh:
    for ln in fh:
        if ln.startswith("#") or not ln.strip():
            continue
        t = ln.split()
        if t[5] == "interp":
            rows.append((int(t[0]), float(t[1]), float(t[6])))
rows.sort()
exc_s = np.array([r[1] for r in rows])
exc_E1 = np.array([r[2] for r in rows])

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(7.2, 7.4),
    gridspec_kw={"height_ratios": [1.8, 1.3], "hspace": 0.07})
for b in range(4):
    ax1.plot(x_path, bands_ev[:, b], lw=1.4, color=C_VAL,
             label="valence 22-25" if b == 0 else None)
for b in range(4, 10):
    ax1.plot(x_path, bands_ev[:, b], lw=1.4, color=C_COND,
             label="conduction 26-31" if b == 4 else None)
ax1.axhline(0.0, color="0.6", lw=0.7, ls=":")
ax1.set_ylabel(r"$\varepsilon_n(k) - E_\mathrm{VBM}$ (eV)")
ax1.set_title("MoS$_2$ 12$\\times$12 — htransform single-particle bands "
              "(640 centroids)", fontsize=11)
ax1.legend(loc="center left", fontsize="small", framealpha=0.9)
ax1.grid(axis="y", ls="--", lw=0.4, alpha=0.3)

ax2.plot(x_path, D_ref, lw=1.6, color=C_DMIN, ls="--",
         label=r"$D_\min(Q)$, driver-window (24,32) caches")
ax2.plot(x_path, D_clean, lw=1.8, color=C_DMIN2,
         label=r"$D_\min(Q)$, clean-window (26,34) floor")
ax2.plot(exc_s, exc_E1, lw=1.2, color=C_EXC, alpha=0.85,
         label=r"$E_1(Q)$ exciton (640c, interp)")
iq = 9
ax2.annotate("iQ 9 'dip': only in the driver-window\ncurve — cache artifact, "
             "not kinematics\n(exciton tracks it)",
             xy=(x_path[iq], D_ref[iq]),
             xytext=(x_path[iq] + 0.75, D_ref[iq] - 0.28),
             fontsize=8, ha="left",
             arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"))
ax2.set_ylabel("energy (eV)")
ax2.legend(loc="lower right", fontsize="small", framealpha=0.9)
ax2.grid(axis="y", ls="--", lw=0.4, alpha=0.3)

node_x = [x_path[i] for i, _ in nodes]
for ax in (ax1, ax2):
    for xv in node_x:
        ax.axvline(xv, color="k", lw=0.6, alpha=0.25)
ax2.set_xticks(node_x, [l for _, l in nodes])
ax2.set_xlim(x_path[0], x_path[-1])

fig.text(0.5, -0.03,
         "Panel 2: where the two $D_\\min$ curves agree (on-grid Q: exact; "
         "smooth trend), exciton structure is single-particle kinematics.\n"
         "Where they split (iQ 6, 9, 16-17 — off-grid), the exciton bands "
         "follow the (24,32) curve: htransform window-cache artifacts\n"
         "(31|32 boundary min-gap 5.9 meV), NOT ISDF-basis error and NOT "
         "physics.  Windows chosen per gap_scan.py Kramers-pair analysis.",
         ha="center", fontsize=8)
fig.savefig("sp_bands_12x12_GMKG.png", dpi=180, bbox_inches="tight")
print("Wrote sp_bands_12x12_GMKG.png")
