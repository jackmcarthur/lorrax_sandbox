"""Diagnostic: what units is wfn.bvec actually in?

We need to pin this down because the kinetic-balance lift
ψ_S = (α_FS/2) σ·(k+G) ψ_L only gives the correct physical magnitude
when (k+G) is in atomic units (Bohr⁻¹).  If wfn.bvec is in 2π/alat
units (BGW HDF5 convention) and we treat the matmul output as Bohr⁻¹,
the lift is wrong by a factor of 2π/alat.

For MoS2 alat ≈ 5.93 Bohr → 2π/alat ≈ 1.06 (factor near 1, easy to miss).
For Si alat = 10.26 Bohr → 2π/alat ≈ 0.61 (factor noticeable).
For CrI3 alat ~ 13 Bohr → 2π/alat ≈ 0.48 (factor ~2, very noticeable).
"""

from __future__ import annotations
import os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from runtime import set_default_env

set_default_env()

from file_io import WFNReader

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_lorrax_cohsex/WFN.h5"

wfn = WFNReader(WFN)
print(f"alat = {float(wfn.alat):.6f}")
print(f"blat = {float(wfn.blat):.6f}  (BGW stores blat = 2π/alat)")
print(f"avec (rows = a_i):\n{np.asarray(wfn.avec)}")
print(f"bvec (rows = b_i):\n{np.asarray(wfn.bvec)}")

avec = np.asarray(wfn.avec)
bvec = np.asarray(wfn.bvec)

# avec_cart = alat * avec  (Bohr)  ← BGW: avec rows are in alat units
avec_cart = float(wfn.alat) * avec

# bvec_cart_in_per_bohr = (2π / alat) * bvec  (Bohr⁻¹)  ← BGW: bvec rows are in 2π/alat units
bvec_cart = (2.0 * np.pi / float(wfn.alat)) * bvec

print(f"\nIf bvec is in 2π/alat units (BGW H5 convention):")
print(f"  bvec_cart in Bohr⁻¹:\n{bvec_cart}")
print(f"  |b_1| = {np.linalg.norm(bvec_cart[0]):.4f} Bohr⁻¹")
print(f"  |b_3| = {np.linalg.norm(bvec_cart[2]):.4f} Bohr⁻¹")

print(f"\nIf bvec is taken as-is (used as if Bohr⁻¹):")
print(f"  |b_1_raw| = {np.linalg.norm(bvec[0]):.4f}")
print(f"  |b_3_raw| = {np.linalg.norm(bvec[2]):.4f}")
print(f"  ratio (true / as-is) = 2π/alat = {2.0*np.pi/float(wfn.alat):.4f}")

# Check orthogonality: a_i · b_j = 2π δ_ij in physical units
print(f"\nOrthogonality test: avec_cart @ bvec_cart.T (should be 2π·I):")
print(avec_cart @ bvec_cart.T)
print(f"  → these should be diag(2π, 2π, 2π) = diag({2*np.pi:.4f},…)")
print(f"  if they are, BGW convention is confirmed (bvec stored in 2π/alat units)")

# Check via raw bvec: avec @ bvec.T = ?
print(f"\nRaw test: avec @ bvec.T (BGW convention says this should equal I, dimensionally):")
print(avec @ bvec.T)

# bdot — should be the metric of bvec_cart
bdot = np.asarray(wfn.bdot)
print(f"\nbdot (BGW: metric tensor of bvec, in (2π/alat)² units):\n{bdot}")
print(f"bvec @ bvec.T (should equal bdot if bvec.bdot are consistent):\n{bvec @ bvec.T}")
