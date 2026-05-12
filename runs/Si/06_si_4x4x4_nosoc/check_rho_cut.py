"""Sanity-check Si valence ρ by plotting a 1D cut along the NN bond."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import sys
import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
from file_io import WFNReader
from centroid.charge_density import rho_from_qe_save

import matplotlib.pyplot as plt

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/nscf/WFN.h5"
SAVE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/qe/scf/silicon.save"

wfn = WFNReader(WFN)
rho = rho_from_qe_save(SAVE)
print(f"ρ.shape = {rho.shape}, ρ.max = {rho.max():.4f}, ρ.min = {rho.min():.4f}")

# Bond: from Si2 (0.125,0.125,0.125) to Si1 image at (-0.125,-0.125,-0.125) ≡ (.875,.875,.875)-(1,1,1)
# Midpoint: (0,0,0). Bond direction: along (1,1,1) primitive frac.
# Sample 21 points along the bond from atom 1 to atom 2.
A = np.array([-0.125, -0.125, -0.125])    # Si1 image (cell -1,-1,-1)
B = np.array([ 0.125,  0.125,  0.125])    # Si2 in cell

ts = np.linspace(0, 1, 41)
ρ_bond = []
for t in ts:
    pos_frac = (1 - t) * A + t * B          # in primitive frac
    pos_frac_wrap = pos_frac % 1.0
    idx = (pos_frac_wrap * np.array(rho.shape)).astype(int) % np.array(rho.shape)
    ρ_bond.append(float(rho[tuple(idx)]))
ρ_bond = np.array(ρ_bond)

print(f"\nρ along Si-Si bond (Si1_image @ t=0 → midpoint @ t=0.5 → Si2 @ t=1):")
for t, ρ in zip(ts, ρ_bond):
    bar = "#" * int(60 * ρ / rho.max())
    marker = ""
    if abs(t - 0.0) < 1e-6:  marker = "  ← Si1 image"
    if abs(t - 0.5) < 1e-6:  marker = "  ← bond midpoint"
    if abs(t - 1.0) < 1e-6:  marker = "  ← Si2"
    print(f"  t={t:.3f}  ρ={ρ:.5f}  |{bar}{marker}")

# Slice plots: 3 orthogonal cuts through the bond midpoint at frac (0,0,0) i.e. idx (0,0,0)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
slice_z = rho[:, :, 0]
axes[0].imshow(slice_z, origin='lower', cmap='plasma')
axes[0].set_title("ρ at z=0 (XY plane through bond midpoint)")
slice_y = rho[:, 0, :]
axes[1].imshow(slice_y, origin='lower', cmap='plasma')
axes[1].set_title("ρ at y=0")
slice_x = rho[0, :, :]
axes[2].imshow(slice_x, origin='lower', cmap='plasma')
axes[2].set_title("ρ at x=0")

# 1D bond cut
ax = axes[3]
ax.plot(ts, ρ_bond, "o-")
ax.axvline(0.5, color="g", ls="--", label="bond midpoint")
ax.axvline(0.0, color="cyan", ls="--", label="Si1 image (atom)")
ax.axvline(1.0, color="cyan", ls="--")
ax.set_xlabel("t along Si–Si bond")
ax.set_ylabel("ρ")
ax.set_title("1D ρ along NN bond")
ax.legend(); ax.grid(True)

plt.tight_layout()
plt.savefig("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/06_si_4x4x4_nosoc/rho_diag.png", dpi=120)
print("saved rho_diag.png")
