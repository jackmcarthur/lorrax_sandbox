"""Quantify whether centroids are concentrated at high-ρ regions:
mean(ρ at centroid) vs mean(ρ over whole cell).
"""
import os; os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_A/src")
from runtime import set_default_env; set_default_env()
import jax  # noqa
from file_io import WFNReader
from common import symmetry_maps
from centroid.charge_density import get_charge_density

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5"
SAVE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/scf/MoS2.save"

wfn = WFNReader(WFN); sym = symmetry_maps.SymMaps(wfn)
rho = np.asarray(get_charge_density(wfn=wfn, sym=sym, source="qe_save", save_dir=SAVE))
Nx, Ny, Nz = rho.shape
fft_grid = np.array([Nx, Ny, Nz])

mean_rho = rho.mean()
print(f"\nρ over the whole {Nx}×{Ny}×{Nz} grid: mean = {mean_rho:.4e}, "
      f"max = {rho.max():.4e}, max/mean = {rho.max()/mean_rho:.1f}")

for name, fname in [("OLD (origin/main)", "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/centroids_old_500.txt"),
                    ("NEW (this branch)", "/pscratch/sd/j/jackm/lorrax_sandbox/tmp/centroids_new_500.txt")]:
    cent_frac = np.loadtxt(fname, comments="#")
    # snap to grid
    idx = (np.round(cent_frac * fft_grid) % fft_grid).astype(int)
    rho_at_cent = rho[idx[:, 0], idx[:, 1], idx[:, 2]]
    print(f"\n{name}: N_c={cent_frac.shape[0]}")
    print(f"  mean ρ at centroid : {rho_at_cent.mean():.4e}")
    print(f"  vs grid mean        : {mean_rho:.4e}")
    print(f"  enrichment factor   : {rho_at_cent.mean() / mean_rho:.2f}×")
    print(f"  fraction of centroids at top-10% ρ : {np.mean(rho_at_cent > np.quantile(rho.flatten(), 0.9)):.2%}")
    print(f"  fraction of centroids at bottom-50% ρ: {np.mean(rho_at_cent < np.quantile(rho.flatten(), 0.5)):.2%}")
