"""Plot MoS2 charge density alongside k-means centroids for both
the old (avec.T@avec, identity-only offsets) and new (avec@avec.T, hex
offset table) implementations. Question: do centroids cluster where ρ peaks?
"""
import os
os.environ.setdefault("MPLBACKEND", "Agg")

import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_A/src")
from runtime import set_default_env
set_default_env()
import jax  # noqa: F401
from file_io import WFNReader
from common import symmetry_maps
from centroid.charge_density import get_charge_density

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5"
SAVE_DIR = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/scf/MoS2.save"
CENT_OLD = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/centroids_old_500.txt"
CENT_NEW = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/centroids_new_500.txt"
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/density_vs_centroids_500.png"
BOHR_TO_ANG = 0.529177210544

with h5py.File(WFN, "r") as f:
    avec = f["/mf_header/crystal/avec"][...]
    alat = float(f["/mf_header/crystal/alat"][...])
    apos = f["/mf_header/crystal/apos"][...]
avec_ang = avec * alat * BOHR_TO_ANG
atom_cart = apos * alat * BOHR_TO_ANG

wfn = WFNReader(WFN)
sym = symmetry_maps.SymMaps(wfn)
rho_r = np.asarray(get_charge_density(wfn=wfn, sym=sym, source="qe_save",
                                      save_dir=SAVE_DIR))
Nx, Ny, Nz = rho_r.shape
print(f"rho shape: {rho_r.shape}, min/mean/max: "
      f"{rho_r.min():.3e} / {rho_r.mean():.3e} / {rho_r.max():.3e}")

cent_old = np.loadtxt(CENT_OLD, comments="#") @ avec_ang
cent_new = np.loadtxt(CENT_NEW, comments="#") @ avec_ang

# Top-down: integrate ρ along c
rho_xy = rho_r.sum(axis=2)
xs = np.linspace(0, 1, Nx, endpoint=False)
ys = np.linspace(0, 1, Ny, endpoint=False)
Xf, Yf = np.meshgrid(xs, ys, indexing="ij")
P = np.stack([Xf, Yf], axis=-1).reshape(-1, 2) @ avec_ang[:2, :2]
tri = Triangulation(P[:, 0], P[:, 1])

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), sharex=True, sharey=True)
for ax, cent_cart, title in [
    (axes[0], cent_old, "OLD: avec.T@avec, no offset table (origin/main)"),
    (axes[1], cent_new, "NEW: avec@avec.T, 5-offset hex table"),
]:
    ax.set_aspect("equal")
    cf = ax.tricontourf(tri, rho_xy.flatten(), levels=18, cmap="viridis", alpha=0.9)
    ax.scatter(cent_cart[:, 0], cent_cart[:, 1],
               c="red", s=14, alpha=0.95, edgecolors="white", linewidths=0.4,
               label=f"centroids (xy proj, N={cent_cart.shape[0]})")
    ax.scatter(atom_cart[:, 0], atom_cart[:, 1],
               c=["purple" if abs(atom_cart[i, 2]) < 0.1 else "gold"
                  for i in range(atom_cart.shape[0])],
               s=180, marker="o", edgecolors="black", linewidths=1.2,
               label="atoms")
    box = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]) @ avec_ang[:2, :2]
    ax.plot(box[:, 0], box[:, 1], "k-", lw=1.2)
    ax.set_xlabel("x (Å)"); ax.set_ylabel("y (Å)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
plt.colorbar(cf, ax=axes, shrink=0.7, label="∫ρ(r) dz")
plt.suptitle("MoS2: centroid placement vs charge density (top-down)")
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved {OUT}")
