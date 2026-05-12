"""Two things:
  (1) Plot ρ slices with a low-end-suppressing cmap (low ρ → background,
      not 'bright purple').
  (2) Compute the fraction of centroids that sit at a grid point in the
      ≥ 90th-percentile of ρ (and a few related percentiles, for context)."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
from file_io import WFNReader
from centroid.charge_density import rho_from_qe_save

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib.colors import LinearSegmentedColormap

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/nscf/WFN.h5"
SAVE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/scf/silicon.save"
CENTROIDS = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/D_lorrax_xonly_overlay/centroids_frac_480.txt"

wfn = WFNReader(WFN)
rho = rho_from_qe_save(SAVE)
centroids_frac = np.loadtxt(CENTROIDS, comments="#")

# === (2) Centroid-density statistics ===
# Snap centroids to FFT-grid indices the same way file_io/centroids.py does
fft = np.array(rho.shape)
idx = np.round(centroids_frac * fft).astype(int) % fft
ρ_at_centroids = rho[idx[:, 0], idx[:, 1], idx[:, 2]]

# Percentiles of the full ρ field (over all 24^3 grid points)
percentiles = [50, 75, 80, 85, 90, 95, 99]
print("Density percentiles over the full FFT grid (24^3 = 13824 cells):")
for p in percentiles:
    pv = np.percentile(rho.ravel(), p)
    n_at = int((ρ_at_centroids >= pv).sum())
    print(f"  p{p:>2d} = ρ ≥ {pv:.5f}    "
          f"centroids above = {n_at:>4d}/{len(ρ_at_centroids)} = {n_at/len(ρ_at_centroids):6.2%}")

print(f"\n  centroid ρ:  min={ρ_at_centroids.min():.5f}  "
      f"med={np.median(ρ_at_centroids):.5f}  "
      f"max={ρ_at_centroids.max():.5f}  mean={ρ_at_centroids.mean():.5f}")
print(f"  full grid ρ: min={rho.min():.5f}  "
      f"med={np.median(rho):.5f}  "
      f"max={rho.max():.5f}  mean={rho.mean():.5f}")

# === (1) Slice plot with a cmap that suppresses the low end ===
# Use 'inferno' but clip vmin so anything below 30% of max is plotted at
# the lowest cmap end (~black/blank). High density → bright yellow.
vmin = 0.30 * rho.max()
vmax = rho.max()

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
slices = [
    (rho[:, :, 0],            "ρ at z=0 (XY through bond midpoint)"),
    (rho[:, 0, :],            "ρ at y=0"),
    (rho[0, :, :],            "ρ at x=0"),
    (rho[:, :, 12],           "ρ at z=N/2 (XY mid-cell)"),
]
for ax, (sl, t) in zip(axes, slices):
    im = ax.imshow(sl, origin='lower', cmap='inferno',
                   vmin=vmin, vmax=vmax, interpolation='bilinear')
    ax.set_title(t)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(f"Si valence ρ  (vmin = {vmin:.4f} = 0.30 ρ_max  →  black; "
             f"vmax = {vmax:.4f} = ρ_max  →  bright yellow)")
plt.tight_layout()
plt.savefig("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/rho_slices.png", dpi=120)
print("\nsaved rho_slices.png")

# Also: histogram of ρ at centroids vs full grid
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
bins = np.linspace(0, rho.max() * 1.02, 40)
ax.hist(rho.ravel(), bins=bins, alpha=0.55, density=True,
        label=f"all {rho.size} grid points", color='gray')
ax.hist(ρ_at_centroids, bins=bins, alpha=0.65, density=True,
        label=f"{len(ρ_at_centroids)} centroids", color='red')
ax.axvline(np.percentile(rho.ravel(), 90), color='blue', ls='--',
           label='90th percentile of grid ρ')
ax.set_xlabel("ρ"); ax.set_ylabel("density of points")
ax.set_title("Distribution of ρ at centroids vs full FFT grid")
ax.legend()
plt.tight_layout()
plt.savefig("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/centroid_rho_hist.png", dpi=120)
print("saved centroid_rho_hist.png")
