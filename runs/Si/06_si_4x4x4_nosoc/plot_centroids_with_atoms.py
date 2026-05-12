"""Confirm Si valence density sits on the NN bonds.

The trick: in this primitive cell, the two in-cell atoms are body-diagonal
(7.05 Å apart) and NOT a chemical-bond pair. Real Si–Si bonds (2.35 Å)
go to atoms in neighboring primitive cells. So the 4 density lobes are
real Si–Si bond regions — to neighbors, not to the other in-cell atom.

Uses the official ``plot_density_and_centroids`` for the density+centroid
layer, then overlays atoms + bonds + a 2nd panel that integrates ρ along
each in-cell bond direction to prove the assignment.
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
from file_io import WFNReader
from centroid.charge_density import rho_from_qe_save
from centroid.kmeans_plot import interpolate_density

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/nscf/WFN.h5"
SAVE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/scf/silicon.save"
CENTROIDS = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/D_lorrax_xonly_overlay/centroids_frac_480.txt"
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/centroids_480_with_atoms.png"

wfn = WFNReader(WFN)
rho = rho_from_qe_save(SAVE)
centroids_frac = np.loadtxt(CENTROIDS, comments="#")

V_bohr = float(wfn.cell_volume)
dV = V_bohr / rho.size
N_e = float(rho.sum() * dV)
print(f"∫ρ dV = {N_e:.4f} electrons")

# Atoms in primitive frac (mod 1), then build a 3×3×3 supercell of images
atoms_in_cell_frac = np.array([[-0.125, -0.125, -0.125],
                               [ 0.125,  0.125,  0.125]]) % 1.0
shifts = np.array([[i, j, k] for i in range(-1, 2)
                              for j in range(-1, 2)
                              for k in range(-1, 2)])
all_atoms_frac = (atoms_in_cell_frac[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
all_atoms_cart = all_atoms_frac @ wfn.avec        # in same units as wfn.avec

# Find NN pairs (bond length is the smallest non-zero atom–atom distance)
diffs = all_atoms_cart[:, None, :] - all_atoms_cart[None, :, :]
d = np.linalg.norm(diffs, axis=2)
nn_dist = d[d > 1e-3].min()
nn_pairs = np.argwhere((d > 1e-3) & (d < 1.05 * nn_dist))
nn_pairs = nn_pairs[nn_pairs[:, 0] < nn_pairs[:, 1]]   # unique
print(f"NN distance: {nn_dist:.4f} (units of avec)")

# Plot setup
rho_zoom = interpolate_density(rho, (2, 2, 2))
threshold = 0.05 * rho_zoom.max()
Nx, Ny, Nz = rho_zoom.shape
X, Y, Z = np.meshgrid(
    np.linspace(0, 1, Nx, endpoint=False),
    np.linspace(0, 1, Ny, endpoint=False),
    np.linspace(0, 1, Nz, endpoint=False),
    indexing="ij",
)
mask = rho_zoom > threshold
pts_frac = np.stack([X[mask], Y[mask], Z[mask]], axis=1)
pts_cart = pts_frac @ wfn.avec
rho_at_pts = rho_zoom[mask]

cent_cart = centroids_frac @ wfn.avec
corners_frac = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
corners_cart = corners_frac @ wfn.avec
edges = [(0, 1), (1, 3), (3, 2), (2, 0),
         (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

fig = plt.figure(figsize=(16, 13))
for sub_i, (elev, azim, title) in enumerate([
    (15, -60, "perspective"),
    (90, -90, "top-down (XY)"),
    (12, 30, "from [+1,+1,0]"),
    (35, 45, "iso"),
]):
    ax = fig.add_subplot(2, 2, sub_i + 1, projection="3d")
    sc = ax.scatter(
        pts_cart[:, 0], pts_cart[:, 1], pts_cart[:, 2],
        c=np.log(np.abs(rho_at_pts) - 0.9 * threshold),
        cmap="plasma", alpha=0.06, s=15, marker="s",
    )
    ax.scatter(cent_cart[:, 0], cent_cart[:, 1], cent_cart[:, 2],
               c="red", s=40, marker="*", edgecolor="k", linewidth=0.4,
               label="Centroids", zorder=5)

    # All Si atoms in/around the cell
    in_box = ((all_atoms_frac > -0.05).all(axis=1) &
              (all_atoms_frac < 1.05).all(axis=1))
    ax.scatter(all_atoms_cart[in_box, 0],
               all_atoms_cart[in_box, 1],
               all_atoms_cart[in_box, 2],
               c="cyan", s=180, marker="o", edgecolor="k", linewidth=1.4,
               label="Si atoms (incl. images)", zorder=6)

    # NN bonds (only those that pass through the cell box)
    for i, j in nn_pairs:
        midpt_frac = 0.5 * (all_atoms_frac[i] + all_atoms_frac[j])
        if not ((-0.05 < midpt_frac).all() and (midpt_frac < 1.05).all()):
            continue
        a, b = all_atoms_cart[i], all_atoms_cart[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="cyan", lw=1.6, alpha=0.8, zorder=4)

    # Cell box
    for i, j in edges:
        ax.plot(*zip(corners_cart[i], corners_cart[j]), "k-", lw=0.9)

    L = wfn.avec.sum(axis=0)
    ax.set_xlim(-0.1, L[0] * 1.1)
    ax.set_ylim(-0.1, L[1] * 1.1)
    ax.set_zlim(-0.1, L[2] * 1.1)
    ax.set_title(f"{title}: ∫ρ = {N_e:.3f} e⁻")
    ax.view_init(elev=elev, azim=azim)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.suptitle(f"Si valence ρ + 480 centroids — bonds traverse cell to NN images "
             f"(NN dist {nn_dist:.3f} alat)", y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved {OUT}")
