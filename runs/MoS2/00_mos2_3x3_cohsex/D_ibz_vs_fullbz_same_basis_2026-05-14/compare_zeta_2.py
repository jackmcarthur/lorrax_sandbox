"""Find IBZ -> full-BZ q mapping and compare ζ at matching q.

The hypothesis: at an IBZ q (which is a member of the full BZ set), Run A's
ζ for that q and Run B's ζ for that same q should be bit-identical, because
both runs use the same centroids and the same wavefunctions.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
jax.config.update("jax_enable_x64", True)

import h5py
import numpy as np
import sys

sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_B/src")

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(HERE, "run_A_ibz", "tmp", "zeta_q.h5")
B = os.path.join(HERE, "run_B_fullbz", "tmp", "zeta_q.h5")
WFN = os.path.join(HERE, "WFN.h5")

# Find IBZ q's from sym + kgrid
from common.symmetry_maps import SymMaps
from file_io import WfnLoader as WFNReader

wfn = WFNReader(WFN)
sym = SymMaps(wfn)
q_irr_int, full_to_irr_idx, full_to_irr_sym, q_irr_full_idx = sym.find_irreducible_qpoints()
print(f"IBZ q-list (kgrid int) shape: {q_irr_int.shape}")
print(f"  q_irr_int = {q_irr_int}")
print(f"  full_to_irr_idx = {full_to_irr_idx}")
print(f"  full_to_irr_sym = {full_to_irr_sym}")
print(f"  q_irr_full_idx (IBZ index in full-BZ list) = {q_irr_full_idx}")

# Build the full BZ q list
nkx, nky, nkz = (int(x) for x in wfn.kgrid)
q_full = np.array(
    [(qx, qy, qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)],
    dtype=np.int32)
print(f"\nFull-BZ q-list (kgrid int):\n{q_full}")

print(f"\nFull-BZ indices that are also IBZ: {q_irr_full_idx}")
print(f"  i.e. full-BZ q at these indices == IBZ q[i] for i = 0, 1, ...")

# Now open both ζ files and compare
with h5py.File(A, "r") as fA, h5py.File(B, "r") as fB:
    zA = fA["zeta_q_G"][:]
    zB = fB["zeta_q_G"][:]
    gvA = fA["isdf_header/gvec_components"][:]
    gvB = fB["isdf_header/gvec_components"][:]
    ngkA = fA["isdf_header/ngk"][:]
    ngkB = fB["isdf_header/ngk"][:]

print(f"\nzA shape: {zA.shape}, zB shape: {zB.shape}")
print(f"ngkA: {ngkA}")
print(f"ngkB: {ngkB}")

# For each IBZ q (Run A index i), find the corresponding full-BZ index j
# in Run B.  Then compare ζ_A[i] vs ζ_B[j] over the first ngkA[i] G's.
print("\nIBZ → full-BZ ζ comparison:")
print(f"  {'i':>3} {'j_full':>6} {'q_irr_int':>14} {'q_full_int':>14} "
      f"{'ngk_A':>6} {'ngk_B':>6} {'gvec_match':>10} {'max|ΔζA-ζB|':>15} {'rel':>12}")
for i in range(q_irr_int.shape[0]):
    j_full = int(q_irr_full_idx[i])
    qi = q_irr_int[i]
    qj = q_full[j_full]
    ngk_a = int(ngkA[i])
    ngk_b = int(ngkB[j_full])
    gv_match = np.array_equal(gvA[i, :, :ngk_a], gvB[j_full, :, :ngk_b])
    # Compare over min(ngk_a, ngk_b) G-vectors and over all centroids
    n = min(ngk_a, ngk_b)
    diff = zA[i, :, :n] - zB[j_full, :, :n]
    max_abs = float(np.abs(diff).max())
    norm_B = float(np.abs(zB[j_full, :, :n]).max())
    rel = max_abs / norm_B if norm_B > 0 else 0.0
    print(f"  {i:3d} {j_full:6d} {str(tuple(qi)):>14} {str(tuple(qj)):>14} "
          f"{ngk_a:6d} {ngk_b:6d} {str(gv_match):>10} {max_abs:15.6e} {rel:12.3e}")
