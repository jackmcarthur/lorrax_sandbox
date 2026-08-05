#!/usr/bin/env python3
"""Try all sign/index combinations of the umklapp phase formula to find which
one (if any) closes V_predicted-vs-V_nosym to ISDF-floor.

Possible formulas:
  A:  exp(+2π i kg0·(r_{ν'} - r_{μ'})) · V_ibz[parent, π^-1(μ'), π^-1(ν')]
  B:  exp(-2π i kg0·(r_{ν'} - r_{μ'})) · V_ibz[parent, π^-1(μ'), π^-1(ν')]
  C:  exp(+2π i kg0·(r_{μ'} - r_{ν'})) · V_ibz[parent, π^-1(μ'), π^-1(ν')]
  D:  exp(+2π i kg0·(r_{ν'} - r_{μ'})) · V_ibz[parent, π(μ'), π(ν')]      (forward perm)
  E:  No phase, π^-1 perm
  F:  No phase, π forward perm

Also test whether the centroid permutation direction we have (forward S, from
my flip commit 80edbe8) is what V_q_ibz dump expects. Try both.
"""
from __future__ import annotations
import numpy as np
import h5py
import sys as _sys
_sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src')
from file_io import WfnLoader

DUMP = '/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/v_q_dumps'

with h5py.File(f'{DUMP}/Vq_ibz_sym.h5', 'r') as f:
    V_q_ibz  = f['V_q_ibz'][:]
    f2i_idx  = f['full_to_irr_idx'][:]
    f2i_sym  = f['full_to_irr_sym'][:]
    sym_perm = f['sym_perm'][:]

with h5py.File(f'{DUMP}/Vqmunu_nosym.h5', 'r') as f:
    V_full_nosym = f['V_qmunu'][:]

wfn = WfnLoader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5')
kgrid = np.asarray(wfn.kgrid, dtype=np.int64)
ntran = int(wfn.ntran)
mtrx = np.asarray(wfn.sym_matrices[:ntran], dtype=np.int64)
sym_mats_k_spatial = mtrx.transpose(0, 2, 1)
sym_mats_k = np.concatenate([sym_mats_k_spatial, -sym_mats_k_spatial], axis=0)

r_mu_frac = np.loadtxt('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/centroids_frac_300.txt')

kx, ky, kz = np.meshgrid(np.arange(kgrid[0]), np.arange(kgrid[1]),
                          np.arange(kgrid[2]), indexing='ij')
q_full_kgrid = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1).astype(np.int64)
_, first_occ = np.unique(f2i_idx, return_index=True)
q_irr_kgrid = q_full_kgrid[np.sort(first_occ)]

inv_perm = np.argsort(sym_perm, axis=-1)   # π^-1 if sym_perm = π
fwd_perm = sym_perm                          # π
n_sym_spatial = sym_perm.shape[0] // 2

# Pick a specific q_full with non-zero kg0 to scan formulas in detail
q_test_list = [4, 10, 13, 28]   # q's with sym_idx > 0 and various kg0 patterns

def get_kg0(q_full):
    parent = int(f2i_idx[q_full])
    s = int(f2i_sym[q_full])
    q_irr_kg = q_irr_kgrid[parent]
    Sq = sym_mats_k[s] @ q_irr_kg
    kg0_kg = Sq - q_full_kgrid[q_full]
    kg0_recip = np.rint(kg0_kg / kgrid.astype(np.float64)).astype(np.int64)
    return parent, s, kg0_recip

print(f"{'q':>3} {'p':>3} {'s':>3}  kg0_recip   " +
      "  ".join([f'{x:>12}' for x in ['no-phase π^-1', 'no-phase π', 'A: +ν-μ π^-1', 'B: -ν+μ π^-1',
                                       'C: +μ-ν π^-1', 'D: +ν-μ π', 'E: -ν+μ π', 'F: +μ-ν π']]))
for q_full in range(36):
    parent, s, kg0 = get_kg0(q_full)
    V_target = V_full_nosym[q_full]
    V_parent = V_q_ibz[parent]
    pi_inv = inv_perm[s]
    pi_fwd = fwd_perm[s]

    phase_mu = np.exp(2j * np.pi * (r_mu_frac @ kg0.astype(np.float64)))  # (n_mu,)
    # phase_mu[μ] = exp(2π i kg0 · r_μ)

    # phase ratio (ν - μ): phase_mu[ν] / phase_mu[μ] = exp(2π i kg0·(r_ν - r_μ))
    PH_pos = phase_mu[None, :] / phase_mu[:, None]   # +ν − μ
    PH_neg = phase_mu[:, None] / phase_mu[None, :]   # +μ − ν = −(ν − μ)

    # Build all candidates
    V_no_pi_inv = V_parent[np.ix_(pi_inv, pi_inv)]
    V_no_pi_fwd = V_parent[np.ix_(pi_fwd, pi_fwd)]

    candidates = {
        'no-phase π^-1': V_no_pi_inv,
        'no-phase π':    V_no_pi_fwd,
        'A: +ν-μ π^-1':  PH_pos * V_no_pi_inv,
        'B: -ν+μ π^-1':  PH_neg * V_no_pi_inv,
        'C: +μ-ν π^-1':  PH_neg * V_no_pi_inv,    # same as B; keep for label
        'D: +ν-μ π':     PH_pos * V_no_pi_fwd,
        'E: -ν+μ π':     PH_neg * V_no_pi_fwd,
        'F: +μ-ν π':     PH_neg * V_no_pi_fwd,
    }
    # (C == B and F == E; keeping them for label consistency in the output.)
    errs = {k: float(np.max(np.abs(v - V_target))) for k, v in candidates.items()}
    err_str = "  ".join([f'{errs[k]:.2e}' for k in
                          ['no-phase π^-1', 'no-phase π', 'A: +ν-μ π^-1', 'B: -ν+μ π^-1',
                           'C: +μ-ν π^-1', 'D: +ν-μ π', 'E: -ν+μ π', 'F: +μ-ν π']])
    print(f'{q_full:>3} {parent:>3} {s:>3}  {kg0.tolist()!s:>10}   {err_str}')

# Summary: for each formula, max err across all q
print(f'\n=== Summary: max |ΔV| across all 36 q for each formula ===')
formulas = ['no-phase π^-1', 'no-phase π', 'A: +ν-μ π^-1', 'B: -ν+μ π^-1',
            'D: +ν-μ π', 'E: -ν+μ π']
max_errs = {k: 0.0 for k in formulas}
for q_full in range(36):
    parent, s, kg0 = get_kg0(q_full)
    V_target = V_full_nosym[q_full]
    V_parent = V_q_ibz[parent]
    pi_inv = inv_perm[s]
    pi_fwd = fwd_perm[s]
    phase_mu = np.exp(2j * np.pi * (r_mu_frac @ kg0.astype(np.float64)))
    PH_pos = phase_mu[None, :] / phase_mu[:, None]
    PH_neg = phase_mu[:, None] / phase_mu[None, :]
    V_no_pi_inv = V_parent[np.ix_(pi_inv, pi_inv)]
    V_no_pi_fwd = V_parent[np.ix_(pi_fwd, pi_fwd)]
    candidates = {
        'no-phase π^-1': V_no_pi_inv,
        'no-phase π':    V_no_pi_fwd,
        'A: +ν-μ π^-1':  PH_pos * V_no_pi_inv,
        'B: -ν+μ π^-1':  PH_neg * V_no_pi_inv,
        'D: +ν-μ π':     PH_pos * V_no_pi_fwd,
        'E: -ν+μ π':     PH_neg * V_no_pi_fwd,
    }
    for k, v in candidates.items():
        max_errs[k] = max(max_errs[k], float(np.max(np.abs(v - V_target))))

print()
print(f"|V_nosym| max = {np.abs(V_full_nosym).max():.4e}")
for k in formulas:
    print(f"  {k:18s} : max |ΔV| = {max_errs[k]:.4e}  (rel {max_errs[k] / np.abs(V_full_nosym).max():.3e})")
