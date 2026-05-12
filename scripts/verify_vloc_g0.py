"""Verify V_loc(G=0) alpha-Z for both Si and MoS2."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core

for name, save_dir in [
    ("Si", "runs/Si/02_si_4x4x4_nosym/qe/nscf/silicon.save"),
    ("MoS2", "runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save"),
]:
    crystal = CrystalData.from_qe_save(save_dir)
    pseudos = load_pseudopotentials(save_dir)
    fft_grid = crystal.fft_grid
    nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
    N = nx * ny * nz

    V_loc_r, _, _ = build_ionic_and_core(crystal, pseudos, fft_grid, truncation_2d=False)
    V_loc_np = np.asarray(V_loc_r)

    # V_loc(G=0) from the ortho-IFFT convention:
    # V_loc(r) = (1/sqrt(N)) * sum_G V_loc_G * exp(iGr)
    # mean(V_loc) = V_loc(r)_avg = (1/sqrt(N)) * V_loc_G(0)
    # So V_loc_G(0) = sqrt(N) * mean(V_loc)
    V_loc_G0 = np.sqrt(N) * np.mean(V_loc_np)
    
    # Eigenvalue contribution from V_loc(G=0):
    # <n|V_loc|n>_G0 = V_loc_G(0) / sqrt(N) = mean(V_loc)
    vloc_g0_eig = np.mean(V_loc_np)

    print(f"\n=== {name} ===")
    print(f"  vol={crystal.cell_volume:.2f} bohr^3, N={N}")
    print(f"  V_loc(r) mean = {vloc_g0_eig:.6f} Ry = {vloc_g0_eig*1000:.3f} mRy")
    print(f"  V_loc(r) min = {V_loc_np.min():.4f}, max = {V_loc_np.max():.4f}")

    # Also compute without alpha-Z override (set truncation_2d=True to skip it)
    V_loc_noaz, _, _ = build_ionic_and_core(crystal, pseudos, fft_grid, truncation_2d=True)
    V_loc_noaz_np = np.asarray(V_loc_noaz)
    vloc_noaz_g0 = np.mean(V_loc_noaz_np)
    
    print(f"  V_loc(r) mean (no alpha-Z) = {vloc_noaz_g0:.6f} Ry = {vloc_noaz_g0*1000:.3f} mRy")
    print(f"  Alpha-Z shift = {(vloc_g0_eig - vloc_noaz_g0)*1000:.3f} mRy")
    
    # Check V_loc at G!=0
    diff = V_loc_np - V_loc_noaz_np
    print(f"  V_loc - V_loc_noaz: mean={np.mean(diff)*1000:.3f} mRy, "
          f"max_nonG0={np.max(np.abs(diff - np.mean(diff)))*1000:.3f} mRy")

