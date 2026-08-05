#!/usr/bin/env python3
"""Numerically verify the umklapp gauge phase formula on the CrI3 V_q dumps.

Derivation (from ISDF identity + Bloch q-periodicity, NOT from "ζ localized
at r_μ"):

    ζ_{q+G, μ}(r) = exp(+i G·(r - r_μ)) ζ_{q, μ}(r)         (Step 1)

    Under sym op S and umklapp q_full = S·q_irr − kg0 (kg0 brings BZ-back):

    V_{q_full}[μ', ν'] = exp(+i kg0·(r_{ν'} − r_{μ'}))
                       · V_{q_irr}[π^{-1}(μ'), π^{-1}(ν')]

This script computes the predicted V_{q_full} from V_{q_irr} via the formula
and compares to the directly-computed nosym V_{q_full}. If the formula is
right, residual should be ~ULP (or ISDF basis noise floor ~1e-5). If wrong,
residual stays at ~unity relative.
"""
from __future__ import annotations
import numpy as np
import h5py
import sys

DUMP = '/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/v_q_dumps'

with h5py.File(f'{DUMP}/Vq_ibz_sym.h5', 'r') as f:
    V_q_ibz   = f['V_q_ibz'][:]          # (8, 300, 300)
    f2i_idx   = f['full_to_irr_idx'][:]  # (36,)
    f2i_sym   = f['full_to_irr_sym'][:]  # (36,)
    sym_perm  = f['sym_perm'][:]         # (12, 300) — extended-trs

with h5py.File(f'{DUMP}/Vqmunu_nosym.h5', 'r') as f:
    V_full_nosym = f['V_qmunu'][:]       # (36, 300, 300) ground truth

with h5py.File(f'{DUMP}/Vqmunu_sym.h5', 'r') as f:
    V_full_sym_unfold = f['V_qmunu'][:]  # (36, 300, 300) what LORRAX produced

# Load centroid coords and kgrid info from the sym WFN
import sys as _sys
_sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src')
from file_io import WfnLoader

wfn = WfnLoader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5')
kgrid = np.asarray(wfn.kgrid, dtype=np.int64)
ntran = int(wfn.ntran)
mtrx = np.asarray(wfn.sym_matrices[:ntran], dtype=np.int64)
sym_mats_k_spatial = mtrx.transpose(0, 2, 1)
sym_mats_k = np.concatenate([sym_mats_k_spatial, -sym_mats_k_spatial], axis=0)

# Load centroid fractional coords
centroids_frac_path = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/centroids_frac_300.txt'
r_mu_frac = np.loadtxt(centroids_frac_path)  # (300, 3) fractional crystal coords
print(f'centroid r_μ shape: {r_mu_frac.shape}, range: '
      f'x∈[{r_mu_frac[:,0].min():.3f},{r_mu_frac[:,0].max():.3f}], '
      f'y∈[{r_mu_frac[:,1].min():.3f},{r_mu_frac[:,1].max():.3f}], '
      f'z∈[{r_mu_frac[:,2].min():.3f},{r_mu_frac[:,2].max():.3f}]')

# Build kgrid-int q_full list (same canonical order LORRAX uses)
kx, ky, kz = np.meshgrid(np.arange(kgrid[0]), np.arange(kgrid[1]),
                          np.arange(kgrid[2]), indexing='ij')
q_full_kgrid = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1).astype(np.int64)
assert q_full_kgrid.shape == (36, 3)

# IBZ q-list (first occurrence in canonical order — agent dumped 8 IBZ q's)
# Reconstruct from f2i_idx: each unique value is an IBZ q; its first occurrence
# in q_full_kgrid is the IBZ representative.
_, first_occ = np.unique(f2i_idx, return_index=True)
q_irr_kgrid = q_full_kgrid[np.sort(first_occ)]
print(f'IBZ q-list ({q_irr_kgrid.shape[0]} q-points):')
for i, q in enumerate(q_irr_kgrid):
    print(f'  q_irr[{i}] = {q.tolist()}')

# Verify the agent's dumped f2i_idx is in our canonical ordering
# (it should be — both built from the same kgrid)

# Now test the umklapp phase formula on EVERY full-BZ q.
# Predicted: V_full_predicted[q, μ', ν'] = exp(2π i kg0·(r_{ν'} - r_{μ'})) · V_q_ibz[parent(q), π^-1(μ'), π^-1(ν')]
# where π_s = sym_perm[s] (forward, after our flip)

inv_perm = np.argsort(sym_perm, axis=-1)   # (12, 300); inv_perm[s] = π_s^{-1}
n_q_full = 36
n_mu = 300
n_sym_spatial = sym_perm.shape[0] // 2

print(f'\n=== Verifying umklapp phase formula on every q_full ===')
print(f'{"q":>3} {"parent":>6} {"sym":>4}  kg0(kgrid-int)  max|ΔV_predicted| (vs nosym)  max|ΔV_no_phase|')

max_err_with_phase = 0.0
max_err_no_phase = 0.0
for q_full in range(n_q_full):
    parent = int(f2i_idx[q_full])
    s = int(f2i_sym[q_full])
    is_trs = s >= n_sym_spatial
    s_spatial = s - n_sym_spatial if is_trs else s

    # Compute kg0_kgrid = sym_mats_k[s] @ q_irr - q_full (in kgrid-int units)
    q_irr_kg = q_irr_kgrid[parent]
    S_kgrid = sym_mats_k[s]
    Sq_kgrid = S_kgrid @ q_irr_kg
    kg0_kgrid = Sq_kgrid - q_full_kgrid[q_full]   # integer triple in kgrid units

    # Convert kg0 to fractional reciprocal (kg0_frac = kg0_kgrid / kgrid).
    # Then phase = exp(2π i (kg0_kgrid / kgrid) · r_frac · kgrid) = exp(2π i kg0_kgrid·r_frac)
    # Wait: kg0 is a reciprocal lattice vector. In kgrid-integer coords with q_int = q_frac·N,
    # kg0_int_in_kgrid corresponds to fractional reciprocal kg0_frac = kg0_int / N.
    # But the umklapp G we want is in INTEGER reciprocal-lattice units (multiples of b).
    # When q_int wraps around at N (=kgrid axis size), it's a shift by 1 reciprocal lattice vector.
    # So kg0_int_RECIP = kg0_int_kgrid / N (must be integer; else not actually an umklapp).
    # For CrI3 6x6x1: an umklapp in x means kg0_kgrid_x = ±6 → kg0_recip_x = ±1. Phase factor:
    #   exp(2π i G_recip · r_frac) = exp(2π i (kg0_kgrid/kgrid_axis) · r_frac · kgrid_axis)
    #   = exp(2π i kg0_kgrid · r_frac) — but ONLY if r_frac is in units of "kgrid spacing"
    # Hmm, let me redo. r_μ in crystal fractional ∈ [0, 1). G in integer reciprocal units.
    # The phase is exp(2π i G_int · r_frac). And G_int = kg0_kgrid / kgrid (if integer).

    # For CrI3 6x6x1: kg0_kgrid components can be ±6 (umklapp by 1 recip vector along x or y),
    # or ±0 (no umklapp). kg0_recip = kg0_kgrid / kgrid → integer.
    kg0_recip = kg0_kgrid / kgrid.astype(np.float64)
    # Sanity: should be integer triple
    kg0_recip_int = np.rint(kg0_recip).astype(np.int64)
    assert np.allclose(kg0_recip, kg0_recip_int, atol=1e-9), \
        f"kg0_recip not integer at q={q_full}: {kg0_recip}"

    # Phase per centroid pair: exp(2π i kg0_recip · (r_ν - r_μ))
    # For each (μ', ν') in the unfolded V_q_full:
    phase_per_mu = np.exp(2j * np.pi * (r_mu_frac @ kg0_recip_int.astype(np.float64)))   # (n_mu,)
    # D = diag(phase_per_mu^{-1})  applied to row μ'; D^* to column ν'
    # V_predict[μ', ν'] = (1/phase_per_mu[μ']) · (phase_per_mu[ν']) · V_q_ibz[parent, π^-1(μ'), π^-1(ν')]
    # (per derivation: V_{q_full}[μ', ν'] = exp(+i kg0·(r_{ν'}-r_{μ'})) · V_ibz[parent, π^-1(μ'), π^-1(ν')])

    # Apply centroid permutation
    pi_inv = inv_perm[s]   # (n_mu,)
    V_parent = V_q_ibz[parent]              # (n_mu, n_mu)
    V_perm = V_parent[np.ix_(pi_inv, pi_inv)]   # (n_mu, n_mu)

    # Apply phase: V[μ', ν'] *= exp(2π i kg0_recip · r_{ν'}) · exp(-2π i kg0_recip · r_{μ'})
    V_predicted = V_perm * (phase_per_mu[None, :] / phase_per_mu[:, None])

    # Compare to nosym
    V_nosym_q = V_full_nosym[q_full]
    err_with_phase = float(np.max(np.abs(V_predicted - V_nosym_q)))

    # Also try without phase (just permutation)
    err_no_phase = float(np.max(np.abs(V_perm - V_nosym_q)))

    max_err_with_phase = max(max_err_with_phase, err_with_phase)
    max_err_no_phase = max(max_err_no_phase, err_no_phase)

    print(f'{q_full:>3} {parent:>6} {s:>4}  {kg0_recip_int.tolist()!s:>14}  '
          f'{err_with_phase:.3e}            {err_no_phase:.3e}')

print(f'\n=== SUMMARY ===')
print(f'  max |V_predicted - V_nosym|        = {max_err_with_phase:.6e}')
print(f'  max |V_perm_only - V_nosym|        = {max_err_no_phase:.6e}')
print(f'  max |V_nosym|                       = {np.abs(V_full_nosym).max():.6e}')
print(f'  rel err with phase    = {max_err_with_phase / np.abs(V_full_nosym).max():.3e}')
print(f'  rel err without phase = {max_err_no_phase / np.abs(V_full_nosym).max():.3e}')
print()
if max_err_with_phase < 1e-2:
    print('  VERDICT: phase formula CORRECT (closes to ISDF-floor)')
elif max_err_with_phase < max_err_no_phase * 0.1:
    print('  VERDICT: phase formula PARTIALLY correct (large improvement but not closed)')
else:
    print('  VERDICT: phase formula INCORRECT (no improvement over centroid permutation alone)')
