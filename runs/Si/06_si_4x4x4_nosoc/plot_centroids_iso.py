"""Solid isosurface (single iso value) of Si valence ρ + centroids + atoms.
Uses matplotlib voxels with explicit Cartesian corner arrays so the iso
is rendered correctly in the parallelepiped primitive cell."""
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
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/centroids_480_iso.png"

wfn = WFNReader(WFN)
rho = rho_from_qe_save(SAVE)
centroids_frac = np.loadtxt(CENTROIDS, comments="#")

dV = float(wfn.cell_volume) / rho.size
N_e = float(rho.sum() * dV)
print(f"∫ρ dV = {N_e:.4f} electrons    ρ.max() = {rho.max():.4f}")

# Native 24³ grid for fast voxel render (mpl voxels is O(N³))
rho_zoom = rho
Nx, Ny, Nz = rho_zoom.shape

# Corner-grid (Nx+1) of FRACTIONAL coords, then map → Cartesian via avec
fx = np.linspace(0, 1, Nx + 1, endpoint=True)
fy = np.linspace(0, 1, Ny + 1, endpoint=True)
fz = np.linspace(0, 1, Nz + 1, endpoint=True)
FX, FY, FZ = np.meshgrid(fx, fy, fz, indexing="ij")
frac = np.stack([FX, FY, FZ], axis=-1)            # (Nx+1, Ny+1, Nz+1, 3)
cart = frac @ wfn.avec                            # cell-skewed corner coords
CX, CY, CZ = cart[..., 0], cart[..., 1], cart[..., 2]

iso_level = 0.85 * rho_zoom.max()
print(f"iso = {iso_level:.5f}   "
      f"voxels above iso = {int((rho_zoom >= iso_level).sum())} / {rho_zoom.size}")
colors = [(0.85, 0.30, 0.85, 0.95)]

# Atoms in/near cell
atoms_in_cell_frac = np.array([[-0.125]*3, [0.125]*3]) % 1.0
shifts = np.array([[i, j, k] for i in range(-1, 2)
                              for j in range(-1, 2)
                              for k in range(-1, 2)])
all_atoms_frac = (atoms_in_cell_frac[:, None, :] + shifts[None, :, :]).reshape(-1, 3)
all_atoms_cart = all_atoms_frac @ wfn.avec

# NN bond pairs within/across cell box
diffs = all_atoms_cart[:, None, :] - all_atoms_cart[None, :, :]
d = np.linalg.norm(diffs, axis=2)
nn_dist = d[d > 1e-3].min()
nn_pairs = np.argwhere((d > 1e-3) & (d < 1.05 * nn_dist))
nn_pairs = nn_pairs[nn_pairs[:, 0] < nn_pairs[:, 1]]

cent_cart = centroids_frac @ wfn.avec
in_box_atoms = ((all_atoms_frac > -0.05).all(axis=1) &
                (all_atoms_frac < 1.05).all(axis=1))

corners_frac = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
corners_cart = corners_frac @ wfn.avec
edges = [(0, 1), (1, 3), (3, 2), (2, 0),
         (4, 5), (5, 7), (7, 6), (6, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]

fig = plt.figure(figsize=(15, 13))
for sub_i, (elev, azim, title) in enumerate([
    (15, -60, "perspective"),
    (90, -90, "top-down (XY)"),
    (12, 30, "from [+1,+1,0]"),
    (35, 45, "iso view"),
]):
    ax = fig.add_subplot(2, 2, sub_i + 1, projection="3d")

    filled = rho_zoom >= iso_level
    ax.voxels(CX, CY, CZ, filled,
              facecolors=colors[0], edgecolor=(0, 0, 0, 0.3),
              linewidth=0.2)

    ax.scatter(cent_cart[:, 0], cent_cart[:, 1], cent_cart[:, 2],
               c="red", s=35, marker="*", edgecolor="k", linewidth=0.4,
               label="Centroids", zorder=5)
    ax.scatter(all_atoms_cart[in_box_atoms, 0],
               all_atoms_cart[in_box_atoms, 1],
               all_atoms_cart[in_box_atoms, 2],
               c="cyan", s=180, marker="o", edgecolor="k", linewidth=1.4,
               label="Si atoms", zorder=6)

    for i, j in nn_pairs:
        midpt_frac = 0.5 * (all_atoms_frac[i] + all_atoms_frac[j])
        if not ((-0.05 < midpt_frac).all() and (midpt_frac < 1.05).all()):
            continue
        a, b = all_atoms_cart[i], all_atoms_cart[j]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color="cyan", lw=1.6, alpha=0.85, zorder=4)

    for i, j in edges:
        ax.plot(*zip(corners_cart[i], corners_cart[j]), "k-", lw=0.9)

    L = wfn.avec.sum(axis=0)
    ax.set_xlim(-0.1, L[0] * 1.1)
    ax.set_ylim(-0.1, L[1] * 1.1)
    ax.set_zlim(-0.1, L[2] * 1.1)
    ax.set_title(f"{title}: iso = {iso_level:.4f} (= 0.85 ρ_max)")
    ax.view_init(elev=elev, azim=azim)

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.suptitle(f"Si valence ρ isosurfaces  +  480 centroids   "
             f"(∫ρ = {N_e:.3f} e⁻, NN = {nn_dist:.3f} alat)", y=1.0)
plt.tight_layout()
plt.savefig(OUT, dpi=140, bbox_inches="tight")
print(f"saved {OUT}")
