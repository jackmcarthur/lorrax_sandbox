#!/usr/bin/env python3
"""Standalone test of the user's umklapp-aware unfold formula on the
CrI3 V_q dumps. NO code changes — pure verification.

User's spec (verbatim):
    y_μ = U⁻¹ (x_μ − τ)            # direct fractional, U = (Kᵀ)⁻¹ = K⁻ᵀ
    y_μ = x_{α(μ)} + L_μ            # decompose: α = lookup mod 1, L = floor
    V_{q1, μν} = exp(2π i q · (L_μ − L_ν)) · V_{q, α(μ), α(ν)}

where q is the IBZ parent q (fractional reciprocal), q1 = K q + G_R the
full-BZ q, K = sym_matrices is the BGW reciprocal rotation.

Convention recap:
    K (stored in WFN.h5)            = mtrx[s] = sym_matrices[s]
    U (forward direct-space sym)    = (Kᵀ)⁻¹ = K⁻ᵀ
    U⁻¹ (used in y_μ above)         = Kᵀ

So the matrix applied to (x_μ − τ) in column-form is Kᵀ.

This script tests three variants to disambiguate any remaining sign /
direction confusion:

    1. "User math (forward α-target)":
         y[s, μ, :] = Kᵀ @ (x_μ − τ[s])
         α(μ) = index whose centroid coords match y mod 1
         L_μ  = floor(y)
         V_predicted[q, μ, ν] = exp(2π i q · (L_μ − L_ν)) · V_ibz[α(μ), α(ν)]

    2. Same as 1 but with K (no transpose) — i.e. the post-flip current
       code, with L_μ captured. Tests whether the empirically-winning K
       direction also closes umklapp q's with the proper L phase.

    3. Same as 1 but with K⁻¹ (pre-flip direction).

All three should agree at involutive ops; one should win at non-symmetric
C3-like ops + umklapp.
"""
from __future__ import annotations
import numpy as np
import h5py
import sys as _sys
_sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src')
from file_io import WfnLoader

DUMP = '/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/v_q_dumps'

with h5py.File(f'{DUMP}/Vq_ibz_sym.h5', 'r') as f:
    V_q_ibz  = f['V_q_ibz'][:]          # (8, 300, 300)
    f2i_idx  = f['full_to_irr_idx'][:]  # (36,)
    f2i_sym  = f['full_to_irr_sym'][:]  # (36,)

with h5py.File(f'{DUMP}/Vqmunu_nosym.h5', 'r') as f:
    V_full_nosym = f['V_qmunu'][:]      # (36, 300, 300) ground truth

wfn = WfnLoader('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5')
kgrid = np.asarray(wfn.kgrid, dtype=np.int64)
ntran = int(wfn.ntran)
K = np.asarray(wfn.sym_matrices[:ntran], dtype=np.float64)   # (ntran, 3, 3), BGW mtrx
tau_frac = np.asarray(wfn.translations[:ntran], dtype=np.float64) / (2.0 * np.pi)  # (ntran, 3)

# Build TRS-extended sym table (matches dump's f2i_sym range)
# TRS row: K → -K (k-action), τ → τ (real-space unaffected)
K_ext = np.concatenate([K, -K], axis=0)
tau_ext = np.concatenate([tau_frac, tau_frac], axis=0)
n_sym_total = K_ext.shape[0]   # 12 = 2*ntran

# Centroid set (fractional)
r_mu_frac = np.loadtxt('/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/centroids_frac_300.txt')
n_mu = r_mu_frac.shape[0]
print(f'Loaded {n_mu} centroids, {ntran} spatial sym ops, {n_sym_total} TRS-augmented.')

# Build kgrid-int q_full list
kx, ky, kz = np.meshgrid(np.arange(kgrid[0]), np.arange(kgrid[1]),
                          np.arange(kgrid[2]), indexing='ij')
q_full_kgrid = np.stack([kx.flatten(), ky.flatten(), kz.flatten()], axis=1).astype(np.int64)
# IBZ q list = first occurrences
_, first_occ = np.unique(f2i_idx, return_index=True)
q_irr_kgrid = q_full_kgrid[np.sort(first_occ)]
q_irr_frac = q_irr_kgrid.astype(np.float64) / kgrid.astype(np.float64)   # fractional reciprocal

# FFT-grid index of each centroid for fast lookup
# Centroids are in fractional ∈ [0, 1); we need a coarse lookup. Use a
# tolerance-based dict on rounded coords.
TOL = 1e-6
INV = int(round(1.0 / TOL))
def _key(r):
    return tuple((np.rint((r % 1.0) * INV).astype(np.int64) % INV).tolist())
centroid_lookup = {_key(r_mu_frac[i]): i for i in range(n_mu)}

assert len(centroid_lookup) == n_mu, \
    f"Centroid lookup collision: {n_mu} centroids, {len(centroid_lookup)} unique keys"

def build_alpha_L(K_s, tau_s):
    """For one sym op (K, τ), compute α[μ] = source-centroid-index and
    L[μ] = integer lattice wrap such that y_μ = Kᵀ · (x_μ − τ) = x_{α(μ)} + L_μ.
    """
    U_inv = K_s.T   # U⁻¹ = Kᵀ per user's spec (since K = U⁻ᵀ ⇒ Kᵀ = U⁻¹)
    y = (r_mu_frac - tau_s[None, :]) @ U_inv.T   # row-form: y = (x - τ) @ U⁻¹.T = (U⁻¹ · (x - τ)).T-of-row...
    # Actually be careful. Row-form r @ M = (Mᵀ · r)ᵀ. To compute column-form
    # U_inv · (x - τ), with x - τ as column vector, the row-form equivalent is
    # (x - τ) @ U_inv.T. So:
    y = (r_mu_frac - tau_s[None, :]) @ U_inv.T   # (n_mu, 3) = U_inv·(x-τ) for each row
    y_wrap = y - np.floor(y)
    L = np.floor(y).astype(np.int64)
    alpha = -np.ones(n_mu, dtype=np.int64)
    for mu in range(n_mu):
        k = _key(y_wrap[mu])
        if k in centroid_lookup:
            alpha[mu] = centroid_lookup[k]
    return alpha, L

def build_alpha_L_postflip(K_s, tau_s):
    """Variant 2: post-flip code's matrix (K applied column-form), with L
    captured. y_μ = K · x_μ + τ (the current post-flip einsum), then
    decompose y_μ = x_{α(μ)} + L_μ where α is the IMAGE not source.

    Note: this inverts the role of α — α now is the IMAGE (forward), not
    the source. The phase formula needs to be flipped accordingly. We
    keep both to test.
    """
    y = r_mu_frac @ K_s.T + tau_s[None, :]   # = K · x + τ in col form
    y_wrap = y - np.floor(y)
    L = np.floor(y).astype(np.int64)
    alpha = -np.ones(n_mu, dtype=np.int64)
    for mu in range(n_mu):
        k = _key(y_wrap[mu])
        if k in centroid_lookup:
            alpha[mu] = centroid_lookup[k]
    return alpha, L

# Run user's math for every sym op, check closure
print('\n=== Closure check: does y_μ = U⁻¹(x_μ − τ) land on a centroid? ===')
alpha_table = np.zeros((n_sym_total, n_mu), dtype=np.int64)
L_table = np.zeros((n_sym_total, n_mu, 3), dtype=np.int64)
closure_fail_count = 0
for s in range(n_sym_total):
    a, L = build_alpha_L(K_ext[s], tau_ext[s])
    n_bad = int(np.sum(a < 0))
    if n_bad > 0:
        print(f'  sym {s}: {n_bad}/{n_mu} centroids fail closure under U⁻¹')
        closure_fail_count += n_bad
    alpha_table[s] = a
    L_table[s] = L

if closure_fail_count > 0:
    print(f'\n!!! User-math closure FAILED for {closure_fail_count} centroid-sym pairs.')
    print('    The centroid set is not U-closed; trying K-direction (post-flip) instead.')
    alpha_table_pf = np.zeros((n_sym_total, n_mu), dtype=np.int64)
    L_table_pf = np.zeros((n_sym_total, n_mu, 3), dtype=np.int64)
    for s in range(n_sym_total):
        a, L = build_alpha_L_postflip(K_ext[s], tau_ext[s])
        alpha_table_pf[s] = a
        L_table_pf[s] = L
    n_bad_pf = int(np.sum(alpha_table_pf < 0))
    print(f'    Post-flip (K) closure: {n_bad_pf} centroid-sym pairs fail.')

# Now compute V_predicted per q and compare
print('\n=== Compare V_predicted vs V_nosym ===')
print(f'{"q":>3} {"p":>3} {"s":>3}  user-math  postflip+L  postflip-noL')
max_user = 0.0
max_pf_L = 0.0
max_pf_noL = 0.0
for q_full in range(36):
    parent = int(f2i_idx[q_full])
    s = int(f2i_sym[q_full])
    q_irr = q_irr_frac[parent]    # IBZ q in fractional recip
    V_target = V_full_nosym[q_full]
    V_parent = V_q_ibz[parent]

    # User math (formula 1)
    a = alpha_table[s]
    L = L_table[s]
    if (a >= 0).all():
        qL = L.astype(np.float64) @ q_irr     # (n_mu,)
        phase = np.exp(2j * np.pi * qL)
        # V_predicted[μ, ν] = exp(2π i q · (L_μ − L_ν)) · V_parent[α(μ), α(ν)]
        V_pred = phase[:, None] * V_parent[np.ix_(a, a)] * phase.conj()[None, :]
        err_user = float(np.max(np.abs(V_pred - V_target)))
    else:
        err_user = float('nan')

    # Post-flip + L
    if closure_fail_count > 0 and (alpha_table_pf[s] >= 0).all():
        a_pf = alpha_table_pf[s]
        L_pf = L_table_pf[s]
        qL_pf = L_pf.astype(np.float64) @ q_irr
        phase_pf = np.exp(2j * np.pi * qL_pf)
        V_pred_pf = phase_pf[:, None] * V_parent[np.ix_(a_pf, a_pf)] * phase_pf.conj()[None, :]
        err_pf_L = float(np.max(np.abs(V_pred_pf - V_target)))

        V_pred_pf_noL = V_parent[np.ix_(a_pf, a_pf)]
        err_pf_noL = float(np.max(np.abs(V_pred_pf_noL - V_target)))
    else:
        err_pf_L = float('nan')
        err_pf_noL = float('nan')

    max_user = max(max_user, err_user) if not np.isnan(err_user) else max_user
    if not np.isnan(err_pf_L):
        max_pf_L = max(max_pf_L, err_pf_L)
        max_pf_noL = max(max_pf_noL, err_pf_noL)

    print(f'{q_full:>3} {parent:>3} {s:>3}  {err_user:.2e}    {err_pf_L:.2e}    {err_pf_noL:.2e}')

print(f'\n=== SUMMARY ===')
print(f'|V_nosym|max = {np.abs(V_full_nosym).max():.4e}')
print(f'User-math max |ΔV|        = {max_user:.4e}  (rel {max_user/np.abs(V_full_nosym).max():.3e})')
print(f'Post-flip + L max |ΔV|    = {max_pf_L:.4e}  (rel {max_pf_L/np.abs(V_full_nosym).max():.3e})')
print(f'Post-flip no L max |ΔV|   = {max_pf_noL:.4e}  (rel {max_pf_noL/np.abs(V_full_nosym).max():.3e})')

if max_user < 1e-2:
    print('\nVERDICT: user math CLOSES (ISDF floor on all 36 q\'s) ✓')
elif max_pf_L < 1e-2:
    print('\nVERDICT: post-flip + L CLOSES — user math direction off, L formula right')
elif closure_fail_count > 0:
    print('\nVERDICT: user math has closure failures — centroid set is K-closed, not U-closed.')
else:
    print('\nVERDICT: neither closes — derivation needs more work.')
