"""Use centroid.kmeans_plot.plot_density_and_centroids directly.

Also independently verify the density by reading it both from the QE
.save and from the WFN.h5 occupied-band sum, and reporting integral +
peak location for each."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
from file_io import WFNReader
from common.symmetry_maps import SymMaps
from centroid.charge_density import (
    rho_from_qe_save, rho_from_wfn_ibz, get_charge_density,
)
from centroid.kmeans_plot import plot_density_and_centroids, interpolate_density

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/nscf/WFN.h5"
SAVE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/scf/silicon.save"
CENTROIDS = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/D_lorrax_xonly_overlay/centroids_frac_480.txt"

wfn = WFNReader(WFN)
fft_grid = tuple(int(v) for v in wfn.fft_grid)
V_bohr = float(getattr(wfn, "cell_volume", abs(np.linalg.det(wfn.avec))/0.529177**3))
dV = V_bohr / np.prod(fft_grid)

print(f"FFT grid: {fft_grid}   V_cell = {V_bohr:.4f} bohr^3   dV = {dV:.5f}")
print(f"avec [Å]:\n{wfn.avec}")

# Atomic positions (crystal frac of the primitive cell)
print("Atoms (frac, Cartesian Å):")
atom_frac = np.array([[-0.125, -0.125, -0.125], [0.125, 0.125, 0.125]]) % 1.0
for af in atom_frac:
    ac = af @ wfn.avec
    print(f"  frac={af}  cart={ac}")

# 1) QE-save density (point-group symmetric)
rho_qe = rho_from_qe_save(SAVE)
N_qe = float(rho_qe.sum()) * dV
i_max = np.unravel_index(int(np.argmax(rho_qe)), rho_qe.shape)
peak_frac = np.array(i_max) / np.array(fft_grid)
peak_cart = peak_frac @ wfn.avec
print(f"\n[QE  ] sum*dV = {N_qe:.4f} electrons  (expect 8)")
print(f"       peak at idx={i_max}  frac={peak_frac}  cart={peak_cart}")

# 2) WFN occupied-band sum  (no symmetrization)
sym = SymMaps(wfn)
rho_wfn = rho_from_wfn_ibz(wfn, sym, n_val=8 // 2)  # 4 spinor pairs = 8 electrons
N_wfn = float(rho_wfn.sum()) * dV
i_max2 = np.unravel_index(int(np.argmax(rho_wfn)), rho_wfn.shape)
peak_frac2 = np.array(i_max2) / np.array(fft_grid)
print(f"\n[WFN ] sum*dV = {N_wfn:.4f} electrons  (expect 8)")
print(f"       peak at idx={i_max2}  frac={peak_frac2}")

# 3) Cross-check
diff = np.abs(rho_qe - rho_wfn).max() / rho_qe.max()
print(f"\nmax|ρ_qe - ρ_wfn| / max(ρ_qe) = {diff:.3e}  (small = consistent)")

# Plot using the official function
centroids_frac = np.loadtxt(CENTROIDS, comments="#")
rho_plot = interpolate_density(rho_qe, (2, 2, 2))
plot_density_and_centroids(wfn, rho_plot, centroids_frac,
    out="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/centroids_480_official.png",
    threshold_frac=0.05)
