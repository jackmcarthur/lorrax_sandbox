"""Plot MoS2 k-means centroids inside the hex unit cell."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")     # force headless before pyplot

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5"
CENTROIDS = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/centroids_new_500.txt"
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/mos2_centroids_500.png"
BOHR_TO_ANG = 0.529177210544

with h5py.File(WFN, "r") as f:
    avec = f["/mf_header/crystal/avec"][...]            # (3, 3) rows = lat vecs, alat units
    alat = float(f["/mf_header/crystal/alat"][...])      # Bohr
    apos = f["/mf_header/crystal/apos"][...]             # cartesian, alat units
avec_ang = avec * alat * BOHR_TO_ANG                     # (3, 3) Å
cent_frac = np.loadtxt(CENTROIDS, comments="#")          # (N, 3) fractional [0,1)
cent_cart = cent_frac @ avec_ang                         # (N, 3) Å
atom_cart = apos * alat * BOHR_TO_ANG                    # (nat, 3) Å

# Atom labels: the BGW WFN doesn't always store readable species names; for MoS2
# in this run the 3 atoms are Mo (heavier) at z≈0 and 2 S above/below. Just
# colour by z.
atom_z = atom_cart[:, 2]
mo_mask = np.abs(atom_z) < 0.1                           # central layer
s_mask = ~mo_mask

fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection="3d")

# Draw cell edges
verts = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
verts_cart = verts @ avec_ang
edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
         (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
for i, j in edges:
    ax.plot(*zip(verts_cart[i], verts_cart[j]), "k-", lw=0.6, alpha=0.4)

# Centroids
ax.scatter(cent_cart[:, 0], cent_cart[:, 1], cent_cart[:, 2],
           c="tab:red", s=22, alpha=0.85, depthshade=True,
           label=f"centroids (N={cent_cart.shape[0]})")

# Atoms — bigger spheres for visibility
ax.scatter(atom_cart[mo_mask, 0], atom_cart[mo_mask, 1], atom_cart[mo_mask, 2],
           c="tab:purple", s=200, marker="o", edgecolors="black",
           label=f"Mo (z≈0, n={int(mo_mask.sum())})")
ax.scatter(atom_cart[s_mask, 0], atom_cart[s_mask, 1], atom_cart[s_mask, 2],
           c="goldenrod", s=130, marker="o", edgecolors="black",
           label=f"S (n={int(s_mask.sum())})")

ax.set_xlabel("x (Å)")
ax.set_ylabel("y (Å)")
ax.set_zlabel("z (Å)")
lat_lengths = np.linalg.norm(avec_ang, axis=1)
ax.set_title(f"MoS2 hex unit cell · k-means centroids\n"
             f"a={lat_lengths[0]:.3f} Å, c={lat_lengths[2]:.3f} Å, "
             f"γ=120° · 24×24×80 FFT grid")
ax.legend(loc="upper left", fontsize=9)

# Reasonable view angle showing the hex shape
ax.view_init(elev=15, azim=-65)
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")

# Also save a top-down view (looking along c) so the hex symmetry is visible
fig2, ax2 = plt.subplots(figsize=(8, 7))
ax2.set_aspect("equal")
# Plot 2x2 supercell so the hex tiling is visible
shifts = np.array([(0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)])
for sx, sy in shifts:
    sh = sx * avec_ang[0, :2] + sy * avec_ang[1, :2]
    ax2.scatter(cent_cart[:, 0] + sh[0], cent_cart[:, 1] + sh[1],
                c="tab:red", s=12, alpha=0.5)
    ax2.scatter(atom_cart[mo_mask, 0] + sh[0], atom_cart[mo_mask, 1] + sh[1],
                c="tab:purple", s=140, marker="o", edgecolors="black", alpha=0.9)
    ax2.scatter(atom_cart[s_mask, 0] + sh[0], atom_cart[s_mask, 1] + sh[1],
                c="goldenrod", s=80, marker="o", edgecolors="black", alpha=0.7)

# Highlight primary cell
prim_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ avec_ang[:2, :2]
ax2.plot(prim_corners[:, 0], prim_corners[:, 1], "k-", lw=1.5)
ax2.set_xlabel("x (Å)")
ax2.set_ylabel("y (Å)")
ax2.set_title("MoS2 — top-down view (looking along c) · 3×3 tiling")
plt.tight_layout()
out2 = OUT.replace(".png", "_top.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"saved {out2}")
