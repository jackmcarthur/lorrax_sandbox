#!/usr/bin/env python3
"""Test the production unfold_v_q (with L-phase) against V_q dumps.

Loads V_ibz from the sym dump, calls production unfold_v_q, compares to
V_full from nosym dump. Pass gate: max |ΔV| < 1e-3 relative (ISDF noise).
"""
import os
os.environ['JAX_ENABLE_X64'] = '1'
import numpy as np
import h5py
import sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src')
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm
from common.symmetry_maps import unfold_v_q

# Load dumps
DUMP = '/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/v_q_dumps'
with h5py.File(f'{DUMP}/Vq_ibz_sym.h5') as f:
    V_q_ibz = f['V_q_ibz'][:]                 # (8, 300, 300)
    f2i_idx = f['full_to_irr_idx'][:]
    f2i_sym = f['full_to_irr_sym'][:]
with h5py.File(f'{DUMP}/Vqmunu_nosym.h5') as f:
    V_full_nosym = f['V_qmunu'][:]            # (36, 300, 300)

# Rebuild sym + L from WFN
wfn = WfnLoader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5')
ntran = int(wfn.ntran)
fft_grid = np.array([45, 45, 120])
r_mu_frac = np.loadtxt('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/centroids_frac_300.txt')
r_mu_idx = np.rint(r_mu_frac * fft_grid).astype(np.int32)

sym_perm, L_table = compute_centroid_sym_perm(
    r_mu_idx,
    sym_matrices=np.asarray(wfn.sym_matrices[:ntran]),
    translations=np.asarray(wfn.translations[:ntran]),
    fft_grid=fft_grid,
    extend_trs=True,
)
print(f'sym_perm.shape={sym_perm.shape}, L_table.shape={L_table.shape}')

# Build q_irr_frac (BGW-wrapped) — match _resolve_ibz_q_list convention
kgrid = np.asarray(wfn.kgrid, dtype=np.float64)
# Reconstruct q_irr_kgrid_int from f2i_idx
_, first_occ = np.unique(f2i_idx, return_index=True)
n_q_full = int(np.prod(kgrid))
all_q = np.array(
    [(qx, qy, qz) for qx in range(int(kgrid[0]))
     for qy in range(int(kgrid[1]))
     for qz in range(int(kgrid[2]))], dtype=np.int64)
q_irr_kg = all_q[np.sort(first_occ)]
q_irr_wrap = np.where(q_irr_kg > kgrid / 2, q_irr_kg - kgrid, q_irr_kg).astype(np.float64)
q_irr_frac = q_irr_wrap / kgrid

# Single-device mesh
mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))

# Put V_q_ibz on device with required sharding
V_sh = NamedSharding(mesh, P(None, 'x', 'y'))
V_ibz_dev = jax.device_put(V_q_ibz, V_sh)

# Call production unfold_v_q
V_full_dev = unfold_v_q(
    V_ibz_dev,
    irr_idx=f2i_idx.astype(np.int32),
    sym_idx=f2i_sym.astype(np.int32),
    sym_perm=sym_perm,
    L_table=L_table,
    q_irr_frac=q_irr_frac,
    mesh_xy=mesh,
    n_sym_spatial=ntran,
)
V_full_host = np.asarray(V_full_dev)
print(f'Got V_full of shape {V_full_host.shape}')

# Compare
err = np.abs(V_full_host - V_full_nosym)
print(f'\n|V_nosym|max = {np.abs(V_full_nosym).max():.4e}')
print(f'max |ΔV| ALL 36 q: {err.max():.4e}  (rel {err.max()/np.abs(V_full_nosym).max():.3e})')
max_per_q = err.reshape(36, -1).max(axis=1)
print(f'\nper-q max err:')
for q in range(36):
    parent = int(f2i_idx[q]); s = int(f2i_sym[q])
    print(f'  q={q:2d} parent={parent:2d} sym={s:2d}: {max_per_q[q]:.3e}')

if err.max() / np.abs(V_full_nosym).max() < 1e-3:
    print('\n✓ PASS: production unfold_v_q + L-phase closes V_q to ISDF noise floor on all 36 q\'s')
else:
    print('\n✗ FAIL: residual exceeds 1e-3 relative tolerance')
    sys.exit(1)
