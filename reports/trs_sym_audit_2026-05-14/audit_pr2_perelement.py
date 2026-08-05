"""Audit PR2 — per-element correctness of unfold_v_q under TRS rows.

Builds a synthetic Hermitian V_ibz at random q's, calls unfold_v_q with
mixed spatial + TRS sym_idx, compares against the numpy hand reference
copied verbatim from tests/test_trs_unfold_centroid_perm.py:_hand_unfold_v_q.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src')
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from centroid.orbit_syms import compute_centroid_sym_perm
from common.symmetry_maps import unfold_v_q


def _hand_unfold_v_q(V_ibz, *, irr_idx, sym_idx, sym_perm, ntran):
    inv_perm = np.argsort(sym_perm, axis=-1)
    n_q_full = irr_idx.shape[0]
    n_rmu = V_ibz.shape[-1]
    V_full = np.zeros((n_q_full, n_rmu, n_rmu), dtype=V_ibz.dtype)
    for iq in range(n_q_full):
        parent = int(irr_idx[iq])
        s = int(sym_idx[iq])
        perm_inv = inv_perm[s]
        is_trs = s >= ntran
        V_perm = V_ibz[parent][np.ix_(perm_inv, perm_inv)]
        V_full[iq] = np.conj(V_perm) if is_trs else V_perm
    return V_full


# Geometry: {I, σ_y, σ_x, C2z}-like 4-element group on 4x4x1 grid
fft_grid = np.array([4, 4, 1], dtype=np.int64)
I3 = np.eye(3, dtype=np.int64)
sigma_y = np.diag([1, -1, 1]).astype(np.int64)
sigma_x = np.diag([-1, 1, 1]).astype(np.int64)
C2z = np.diag([-1, -1, 1]).astype(np.int64)
sym_matrices = np.stack([I3, sigma_y, sigma_x, C2z], axis=0)
translations = np.zeros((4, 3), dtype=np.float64)
ntran = 4

# Orbit-closed centroid set
seeds = np.array([[0,0,0],[1,0,0],[0,1,0],[2,2,0],[1,1,0]], dtype=np.int64)
Rinv = np.rint(np.linalg.inv(sym_matrices)).astype(np.int64)
imgs = set()
for r in seeds:
    for s in range(ntran):
        imgs.add(tuple(((Rinv[s] @ r) % fft_grid).tolist()))
cent_idx = np.array(sorted(imgs), dtype=np.int32)
n_rmu = cent_idx.shape[0]
print(f"n_rmu = {n_rmu}, ntran = {ntran}")

sym_perm = compute_centroid_sym_perm(
    cent_idx, sym_matrices, 2.0 * np.pi * translations, fft_grid,
    validate=True, extend_trs=True)
print(f"sym_perm.shape = {sym_perm.shape}")
assert sym_perm.shape == (2 * ntran, n_rmu)

# Build a varied IBZ→full table: 7 q's covering spatial 0..3 and TRS 4..7
rng = np.random.default_rng(seed=42)
n_q_ibz = 4
n_q_full = 9
full_to_irr_idx = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.int32)
full_to_irr_sym = np.array([0, 1, 2, 3, 4, 5, 6, 7, 0], dtype=np.int32)  # 4 spatial + 4 TRS + 1 redundant identity

# Hermitian random V_ibz
A = (rng.standard_normal((n_q_ibz, n_rmu, n_rmu))
     + 1j * rng.standard_normal((n_q_ibz, n_rmu, n_rmu)))
V_ibz = 0.5 * (A + np.swapaxes(A.conj(), -1, -2))

# Reference
V_ref = _hand_unfold_v_q(V_ibz, irr_idx=full_to_irr_idx,
                        sym_idx=full_to_irr_sym, sym_perm=sym_perm,
                        ntran=ntran)

# Codebase
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
V_sh = NamedSharding(mesh, P(None, 'x', 'y'))
V_ibz_j = jax.device_put(V_ibz.astype(np.complex128), V_sh)
V_full = np.asarray(jax.device_get(unfold_v_q(
    V_ibz_j, irr_idx=full_to_irr_idx, sym_idx=full_to_irr_sym,
    sym_perm=sym_perm, mesh_xy=mesh, n_sym_spatial=ntran)))

# Per-q max diff
per_q = np.max(np.abs(V_full - V_ref).reshape(n_q_full, -1), axis=-1)
print("Per-q max abs diff:")
for iq in range(n_q_full):
    s = int(full_to_irr_sym[iq])
    print(f"  q={iq}: s={s} ({'TRS' if s >= ntran else 'spatial'}), max|Δ|={per_q[iq]:.3e}")

max_diff = float(per_q.max())
ref_norm = float(np.linalg.norm(V_ref))
print(f"\nGLOBAL max abs diff = {max_diff:.3e}")
print(f"V_ref Frobenius norm = {ref_norm:.3e}")
print(f"relative diff = {max_diff/max(ref_norm,1e-30):.3e}")
if max_diff < 1e-12:
    print("PASS: per-element correctness within 1e-12 tolerance")
else:
    print("FAIL: per-element discrepancy exceeds 1e-12")
