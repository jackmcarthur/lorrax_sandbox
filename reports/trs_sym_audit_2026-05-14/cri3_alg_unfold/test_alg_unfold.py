"""Algebraic cross-check of LORRAX's sym unfold ops on CrI3 6×6 30Ry.

Three tests:
  1. ψ unfold algebraic: LORRAX unfold vs hand-rolled vs nosym (gauge).
  2. ζ unfold algebraic: hand-rolled IBZ→full ζ vs nosym ζ; AND
                          V_q hand-rolled unfold vs unfold_v_q vs nosym V_q.
  3. G-vector list consistency: WfnLoader.gvecs(full_bz) vs nosym gvecs.

Hand-rolled unfold derived from first principles (BGW convention), NOT from
LORRAX's unfold_psi / unfold_v_q.  Conventions:
    BGW mtrx[s] acts on G-vectors as G' = mtrx[s] @ G  (column).
    SymMaps.sym_mats_k[s] = mtrx[s].T  (row form — applies to k as a row).
        Equivalently, sym_mats_k acts on k/G in row form as
        v' = sym_mats_k @ v_row  ≡  v_col @ sym_mats_k^T  ≡  mtrx @ v_col.
    So in column form, the G rotation under SymMaps row index s is identical
    to mtrx[s] @ G.
    (Verified by reading symmetry_maps.py:478:
        sym_mats_k = sym_matrices[:ntran].transpose(0,2,1))

For ζ in G-space (charge channel, BGW v_q bilinear):

    ψ_{Sk}(G_rot) = U_s(σ) · ψ_k(S^{-1} G_rot)         (spatial sym, τ=0 here)

    The pair-density ⟨n_μ|ψ_n*ψ_m⟩ becomes, at a centroid r_μ that maps to
    r_{π_s(μ)} = S r_μ:

        ζ_{Sq}(r_{π_s(μ)}, G') ?

    For ζ in G-space at q, ζ_q(μ, G) = ∫ dr e^{-i(q+G)·r} φ_μ(r) χ_q(r)
    where φ_μ is the centroid indicator at r_μ.  After the sym op:
        ζ_{Sq}(π_s(μ), G') = ζ_q(μ, S^{-1} G')                 (τ=0 case)
                           = ζ_q(μ, S^T G')                    (since S in O(3): S^{-1}=S^T but here in cryst coords S^{-1}=Rinv)

    For symmorphic P-3 (τ=0):
        ζ_full[q', π_s(μ), G'] = ζ_ibz[i(q), μ, S^{-1} G']
    where S = sym_mats_k[sym_idx_q[q']], i(q) = irr_idx_q[q'].

    But the G-vector list on disk is per-q: ζ_ibz[i(q)] has its own gvec_components.
    The expected nosym G-list at q' is also per-q.  So we need to map G'
    in the nosym basis to S^{-1} G' in the ibz basis.
"""
from __future__ import annotations

import os
import sys
import json
import numpy as np
import h5py

sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_B/src')

# Mute jax import warnings
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')

from common.symmetry_maps import SymMaps, unfold_psi as lorrax_unfold_psi
from file_io.wfn_loader import WfnLoader


SYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/WFN.h5'
NOSYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/WFN.h5'
SYM_ZETA = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/tmp/zeta_q.h5'
NOSYM_ZETA = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/tmp/zeta_q.h5'
CENTROIDS = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/centroids_frac_300.txt'

OUT = '/pscratch/sd/j/jackm/lorrax_sandbox/reports/trs_sym_audit_2026-05-14/cri3_alg_unfold'


# ---------------------------------------------------------------------------
# Setup: load WFN headers, SymMaps, ζ files
# ---------------------------------------------------------------------------
print("=" * 78)
print("CrI3 6×6 30Ry — algebraic unfold cross-check")
print("=" * 78)

w_sym = WfnLoader(SYM_WFN, backend='eager')
w_nosym = WfnLoader(NOSYM_WFN, backend='eager')

print(f"\n[setup] sym WFN:    ntran={w_sym.ntran}, nkpts(IBZ)={w_sym.nkpts}, "
      f"nspinor={w_sym.nspinor}, kgrid={tuple(w_sym.kgrid)}")
print(f"[setup] nosym WFN:  ntran={w_nosym.ntran}, nkpts(full)={w_nosym.nkpts}, "
      f"nspinor={w_nosym.nspinor}, kgrid={tuple(w_nosym.kgrid)}")

sym = SymMaps(w_sym._sym_wfn_stub())
sym_nosym = SymMaps(w_nosym._sym_wfn_stub())

print(f"\n[setup] sym.nk_tot (unfolded) = {sym.nk_tot}")
print(f"[setup] sym.irr_idx_k = {sym.irr_idx_k.tolist()}")
print(f"[setup] sym.sym_idx_k = {sym.sym_idx_k.tolist()}")
print(f"[setup] sym.irr_idx_q = {sym.irr_idx_q.tolist()}")
print(f"[setup] sym.sym_idx_q = {sym.sym_idx_q.tolist()}")
print(f"[setup] # TRS-tagged k = {int((sym.sym_idx_k >= w_sym.ntran).sum())}")
print(f"[setup] # TRS-tagged q = {int((sym.sym_idx_q >= w_sym.ntran).sum())}")
print(f"[setup] sym.q_irr_kgrid_int = {sym.q_irr_kgrid_int.tolist()}")
print(f"[setup] sym.q_irr_full_idx = {sym.q_irr_full_idx.tolist()}")

# Sanity: ntran=6, no TRS rows used (CrI3 has inversion).
ntran = int(w_sym.ntran)
assert int((sym.sym_idx_k >= ntran).sum()) == 0, "Expected no TRS k-rows for CrI3 P-3"
assert int((sym.sym_idx_q >= ntran).sum()) == 0, "Expected no TRS q-rows for CrI3 P-3"
print("\n[setup] OK: no TRS-augmented rows fire — CrI3 P-3 has inversion in mtrx.")


# ---------------------------------------------------------------------------
# Test 3 — G-vector full-BZ k-list consistency
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TEST 3 — G-vector full-BZ k-list consistency")
print("=" * 78)

# Build full-BZ k-coords for both runs and a matching index map.
kg = np.asarray(w_sym.kgrid)
unfolded = np.asarray(sym.unfolded_kpts)               # (nk_tot, 3) in frac
nosym_ks = np.asarray(w_nosym.kpoints)                 # (nk_full, 3) in frac
print(f"\n[T3] sym.unfolded_kpts shape = {unfolded.shape}")
print(f"[T3] nosym kpoints shape    = {nosym_ks.shape}")
assert unfolded.shape == nosym_ks.shape, (
    f"unfolded vs nosym k-counts differ: {unfolded.shape} vs {nosym_ks.shape}")

# Build kgrid-integer match between the two k-lists.
def as_int_kgrid(kfrac, kgrid):
    kw = np.mod(kfrac, 1.0)
    return np.mod(np.rint(kw * kgrid[None, :]).astype(np.int64),
                  kgrid[None, :].astype(np.int64))

ki_sym = as_int_kgrid(unfolded, kg)
ki_nos = as_int_kgrid(nosym_ks, kg)
# Map: nosym_idx[i] = full-BZ k index in sym ordering for nosym k[i].
hash_sym = ki_sym[:, 0] * (kg[1] * kg[2]) + ki_sym[:, 1] * kg[2] + ki_sym[:, 2]
hash_nos = ki_nos[:, 0] * (kg[1] * kg[2]) + ki_nos[:, 1] * kg[2] + ki_nos[:, 2]
assert set(hash_sym.tolist()) == set(hash_nos.tolist()), (
    "sym unfolded vs nosym k-sets are not the same set!")
# Build sym→nosym permutation: for each sym full-BZ index, the nosym index.
sym2nos = np.zeros(unfolded.shape[0], dtype=np.int64)
nos_pos = {int(h): i for i, h in enumerate(hash_nos.tolist())}
for i, h in enumerate(hash_sym.tolist()):
    sym2nos[i] = nos_pos[int(h)]
print(f"[T3] sym↔nosym k-mapping permutation built (max idx={sym2nos.max()}).")

gvecs_sym_full = w_sym.gvecs(k='full_bz')              # (nk_tot, ngkmax, 3)
ngk_sym_full = w_sym.ngk_valid(k='full_bz')            # (nk_tot,)
gvecs_nos = w_nosym.gvecs(k='ibz')                     # nosym 'ibz' == full BZ
ngk_nos = w_nosym.ngk_valid(k='ibz')
print(f"[T3] gvecs_sym_full shape={gvecs_sym_full.shape}, ngkmax={w_sym.ngkmax}")

t3_results = []
worst_t3 = (-1, -1, 0)  # (kf, n_diff, n_total)
for k_full in range(unfolded.shape[0]):
    ngk_a = int(ngk_sym_full[k_full])
    k_nos = int(sym2nos[k_full])
    ngk_b = int(ngk_nos[k_nos])
    Ga = gvecs_sym_full[k_full, :ngk_a]
    Gb = gvecs_nos[k_nos, :ngk_b]
    Ga_set = {tuple(int(x) for x in row) for row in Ga}
    Gb_set = {tuple(int(x) for x in row) for row in Gb}
    same = (Ga_set == Gb_set)
    n_in_a_not_b = len(Ga_set - Gb_set)
    n_in_b_not_a = len(Gb_set - Ga_set)
    t3_results.append((k_full, k_nos, ngk_a, ngk_b, n_in_a_not_b, n_in_b_not_a, same))
    if (n_in_a_not_b + n_in_b_not_a) > worst_t3[1]:
        worst_t3 = (k_full, n_in_a_not_b + n_in_b_not_a, max(ngk_a, ngk_b))

n_pass = sum(1 for r in t3_results if r[6])
print(f"[T3] {n_pass}/{len(t3_results)} k-points have identical G-sets")
print(f"[T3] worst case: k_full={worst_t3[0]}, {worst_t3[1]} mismatched G's out of {worst_t3[2]}")
if n_pass != len(t3_results):
    for r in t3_results:
        if not r[6]:
            print(f"      k_full={r[0]} (sym_idx={int(sym.sym_idx_k[r[0]])}, "
                  f"k_irr={int(sym.irr_idx_k[r[0]])}): "
                  f"ngk_sym={r[2]}, ngk_nos={r[3]}, |Δ|={r[4]+r[5]}")

# Save T3 summary
with open(os.path.join(OUT, 'test3_gvec_results.json'), 'w') as fh:
    json.dump({
        'n_k_total': len(t3_results),
        'n_k_pass': n_pass,
        'worst_k_full': int(worst_t3[0]),
        'worst_n_diff': int(worst_t3[1]),
        'per_k': [
            {'k_full': int(r[0]), 'k_nos': int(r[1]),
             'sym_idx': int(sym.sym_idx_k[r[0]]),
             'k_irr': int(sym.irr_idx_k[r[0]]),
             'ngk_sym': int(r[2]), 'ngk_nos': int(r[3]),
             'in_sym_not_nos': int(r[4]), 'in_nos_not_sym': int(r[5]),
             'match': bool(r[6])}
            for r in t3_results
        ]
    }, fh, indent=2)


# ---------------------------------------------------------------------------
# Test 1 — ψ unfold algebraic check
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TEST 1 — ψ unfold algebraic check")
print("=" * 78)


def hand_unfold_psi(cnk_ibz, *, sym_idx, n_sym_spatial, sym_mats_k,
                    translations, U_spatial, g_ibz):
    """Hand-rolled ψ unfold (BGW convention, symmorphic-or-not).
    Returns ψ on the IBZ G-axis (i.e. cnk[b, σ, g] corresponds to
    G' = sym_mats_k[sym_idx] @ g_ibz[g] in the full-k basis).

    Spatial branch (sym_idx < ntran):
        cnk_full[b, σ', g] = sum_σ U[σ', σ] · cnk_ibz[b, σ, g] · exp(-i (S G_ibz[g])·τ)
    TRS branch (sym_idx ≥ ntran), s_sp = sym_idx - ntran, S_full = -S_spatial:
        cnk_full[b, σ', g] = (iσ_y · conj(U_sp))[σ', σ] · conj(cnk_ibz[b, σ, g])
                           · exp(-i (-S_sp G_ibz[g])·τ)
                           = (iσ_y · conj(U_sp))[σ', σ] · conj(cnk_ibz[b, σ, g])
                           · exp(+i (S_sp G_ibz[g])·τ)
    For CrI3 here τ=0 ⇒ phase=1; no TRS rows fire.
    """
    sym_idx = int(sym_idx)
    is_trs = sym_idx >= n_sym_spatial
    s_sp = sym_idx - n_sym_spatial if is_trs else sym_idx
    S_full = np.asarray(sym_mats_k[sym_idx], dtype=np.int64)
    tau = np.asarray(translations[s_sp], dtype=np.float64)
    g = np.asarray(g_ibz, dtype=np.int64)
    rotated = (S_full @ g.T).T.astype(np.float64)   # (ngk, 3)
    phase = np.exp(-1j * (rotated @ tau))           # (ngk,)
    cnk = np.asarray(cnk_ibz, dtype=np.complex128)
    if is_trs:
        # iσ_y · conj(U_sp) is the TRS spinor factor.
        ISY = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
        Ueff = ISY @ np.conj(np.asarray(U_spatial[s_sp]))
        cnk = np.conj(cnk) * phase[None, None, :]
    else:
        Ueff = np.asarray(U_spatial[s_sp])
        cnk = cnk * phase[None, None, :]
    return np.einsum('jk,nkl->njl', Ueff, cnk)


def raw_ibz_psi(wfn, k_irr):
    """Load ψ_kbar from sym WFN at one IBZ k, all bands, both spinor comps.
    Returns (nb, nspinor, ngk_irr) complex128."""
    start = int(wfn._kpt_starts[k_irr])
    ngk = int(wfn.ngk[k_irr])
    raw = wfn._coeffs_raw[:, :, start:start + ngk, :]  # (nb, ns, ngk, 2)
    return raw[..., 0] + 1j * raw[..., 1]


def raw_full_psi(wfn_nosym, k_full):
    """Load ψ from nosym WFN at one full-BZ k."""
    return raw_ibz_psi(wfn_nosym, k_full)


def cosine_similarity_matrix(A, B):
    """A, B: (nb, ngk_eff).  Returns A @ B^† (nb x nb).  Use for gauge check."""
    # both vectors flattened over spinor+G if needed by caller
    return A @ np.conj(B).T


# We need to test a few (k_full, sym_idx) pairs that cover:
#   sym_idx=0 (identity) — sanity
#   sym_idx=1 or 2 (C3 rotation, det=+1, axis z)
#   sym_idx=3 (-I, inversion, det=-1)
#   sym_idx=4 or 5 (S6, improper, det=-1)
# Pick first occurrence of each.
pick = {}
for kf in range(unfolded.shape[0]):
    s = int(sym.sym_idx_k[kf])
    if s not in pick:
        pick[s] = kf
test_pairs = [(pick[s], s) for s in sorted(pick.keys())]
print(f"[T1] sym_idx representatives (kf, sym_idx): {test_pairs}")

t1_results = []
ISY = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)
for k_full, s_idx in test_pairs:
    k_irr = int(sym.irr_idx_k[k_full])
    ngk_irr = int(w_sym.ngk[k_irr])
    cnk_ibz = raw_ibz_psi(w_sym, k_irr)
    g_ibz = w_sym._gvecs_raw[w_sym._kpt_starts[k_irr]:
                              w_sym._kpt_starts[k_irr] + ngk_irr].copy()

    # (b) Hand-rolled unfold of cnk_ibz at sym idx s_idx
    cnk_hr = hand_unfold_psi(
        cnk_ibz, sym_idx=s_idx, n_sym_spatial=ntran,
        sym_mats_k=sym.sym_mats_k,
        translations=sym.translations,
        U_spatial=sym.U_spinor,
        g_ibz=g_ibz)

    # (a) LORRAX unfold via the same helper from symmetry_maps
    cnk_lx = lorrax_unfold_psi(
        cnk_ibz,
        sym_idx=s_idx, n_sym_spatial=ntran,
        g_kbar=g_ibz,
        sym_mats_k=sym.sym_mats_k,
        translations=sym.translations,
        U_spinor_spatial=sym.U_spinor)

    # Compare (a) vs (b)
    diff_ab = np.max(np.abs(cnk_lx - cnk_hr))
    rel_ab = diff_ab / (np.max(np.abs(cnk_hr)) + 1e-30)

    # (c) Nosym ψ at same k_full
    k_nos = int(sym2nos[k_full])
    ngk_nos_k = int(w_nosym.ngk[k_nos])
    cnk_nos = raw_full_psi(w_nosym, k_nos)             # (nb, ns, ngk_nos_k)
    g_nos_k = w_nosym._gvecs_raw[w_nosym._kpt_starts[k_nos]:
                                  w_nosym._kpt_starts[k_nos] + ngk_nos_k]

    # cnk_hr is on the rotated G-list S @ g_ibz.  Need to map nosym G-vector
    # rows to the hand-rolled-unfold's G index ordering.
    S = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int64)
    g_rot = (S @ g_ibz.T).T  # (ngk_irr, 3); these are the G-vectors on which cnk_hr is defined
    # Build a map from G → row index in g_rot (hashable tuple).
    g_rot_to_row = {tuple(int(x) for x in row): i for i, row in enumerate(g_rot)}
    # For each nosym G-vector at this k, find row in g_rot.
    nos_to_hr = -np.ones(ngk_nos_k, dtype=np.int64)
    for gi, gv in enumerate(g_nos_k):
        key = tuple(int(x) for x in gv)
        nos_to_hr[gi] = g_rot_to_row.get(key, -1)
    missing = int((nos_to_hr < 0).sum())
    if missing:
        # Try with the umklapp from LORRAX's sym._get_umklapp_vector
        sym_krep = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int32)
        kg0 = sym._get_umklapp_vector(w_sym, k_full, s_idx, k_irr, sym_krep)
        g_rot_shifted = g_rot - kg0[None, :]
        g_rot_to_row2 = {tuple(int(x) for x in row): i for i, row in enumerate(g_rot_shifted)}
        nos_to_hr = -np.ones(ngk_nos_k, dtype=np.int64)
        for gi, gv in enumerate(g_nos_k):
            key = tuple(int(x) for x in gv)
            nos_to_hr[gi] = g_rot_to_row2.get(key, -1)
        missing2 = int((nos_to_hr < 0).sum())
        umklapp_used = (kg0.tolist())
    else:
        missing2 = 0
        umklapp_used = [0, 0, 0]

    if (nos_to_hr < 0).any():
        # bail
        t1_results.append({
            'k_full': k_full, 'k_irr': k_irr, 'sym_idx': s_idx, 'k_nos': k_nos,
            'diff_lorrax_vs_hr': float(diff_ab), 'rel_lorrax_vs_hr': float(rel_ab),
            'umklapp': umklapp_used, 'missing_after_umklapp': missing2,
            'unitary_max_offdiag_nondegen': None,
            'note': f"could not map {missing2} nosym G's into hand-rolled basis",
        })
        print(f"[T1] kf={k_full}, sym={s_idx}: |Δ_lorrax_vs_hr|={diff_ab:.2e}, "
              f"BUT could not map {missing2} nosym G's to hand-rolled basis")
        continue

    # Now we can compare nosym to hand-rolled in the same G-axis ordering.
    # cnk_nos has shape (nb, ns, ngk_nos_k); cnk_hr_in_nos_order:
    cnk_hr_n = cnk_hr[:, :, nos_to_hr]  # (nb, ns, ngk_nos_k)

    # Build the gauge matrix U[m, n] = <ψ_nosym(m, k_full) | ψ_hr(n, k_full)>
    # over (spinor, G) inner product.  Restrict to ENERGY-DEGENERATE groups
    # using sym ψ_kbar's energies (which equal nosym ψ_k's energies up to ULP).
    nb = cnk_nos.shape[0]
    nb_test = min(nb, 80)  # cap to keep gauge matrix small
    # Reshape for inner product: (nb, ns*ngk)
    A = cnk_nos[:nb_test].reshape(nb_test, -1)
    B = cnk_hr_n[:nb_test].reshape(nb_test, -1)

    # Build U
    U = np.einsum('mi,ni->mn', np.conj(A), B)  # nosym^* @ hr  → (m, n)
    # Group by degenerate energy of sym IBZ k_irr.
    eners = np.asarray(w_sym.energies[0, k_irr, :nb_test])  # eV
    # group bands with |ΔE| < 1e-6 Ry-equivalent (1.36e-5 eV)
    tol_E = 1e-5
    groups = []
    used = np.zeros(nb_test, dtype=bool)
    for i in range(nb_test):
        if used[i]:
            continue
        g = [i]
        used[i] = True
        for j in range(i + 1, nb_test):
            if not used[j] and abs(eners[j] - eners[i]) < tol_E:
                g.append(j); used[j] = True
        groups.append(g)

    # For each non-trivial group, |U[grp,grp]| should be unitary; for cross-group
    # blocks, U[grp_i, grp_j] should be ~0 (different energies).
    worst_offdiag = 0.0
    worst_unit_dev = 0.0
    for g in groups:
        Ug = U[np.ix_(g, g)]
        # Unitary check: ||U U^† - I||
        Ut = Ug @ np.conj(Ug).T
        dev = np.max(np.abs(Ut - np.eye(len(g))))
        worst_unit_dev = max(worst_unit_dev, dev)
    # Cross-group off-diagonal
    for gi, g in enumerate(groups):
        for gj, gj_ in enumerate(groups):
            if gi >= gj:
                continue
            block = U[np.ix_(g, gj_)]
            worst_offdiag = max(worst_offdiag, float(np.max(np.abs(block))))

    t1_results.append({
        'k_full': k_full, 'k_irr': k_irr, 'sym_idx': s_idx, 'k_nos': k_nos,
        'diff_lorrax_vs_hr': float(diff_ab), 'rel_lorrax_vs_hr': float(rel_ab),
        'umklapp': umklapp_used, 'missing_after_umklapp': missing2,
        'n_groups': len(groups), 'nb_tested': nb_test,
        'worst_unitary_dev': float(worst_unit_dev),
        'worst_crossgroup_offdiag': float(worst_offdiag),
        'mat_S': np.asarray(sym.sym_mats_k[s_idx]).tolist(),
        'det_S': int(np.linalg.det(np.asarray(sym.sym_mats_k[s_idx]))),
    })
    print(f"[T1] kf={k_full:2d}, sym={s_idx} (det={int(np.linalg.det(np.asarray(sym.sym_mats_k[s_idx]))):+d}): "
          f"|Δ_lorrax_vs_hr|={diff_ab:.2e} (rel={rel_ab:.2e}); "
          f"worst-unitary-dev={worst_unit_dev:.2e}, worst-cross-block={worst_offdiag:.2e}")

with open(os.path.join(OUT, 'test1_psi_results.json'), 'w') as fh:
    json.dump(t1_results, fh, indent=2)


# ---------------------------------------------------------------------------
# Test 2 — ζ unfold algebraic check  (THE LOAD-BEARING ONE)
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("TEST 2 — ζ unfold algebraic check")
print("=" * 78)

f_sym_z = h5py.File(SYM_ZETA, 'r')
f_nos_z = h5py.File(NOSYM_ZETA, 'r')

# G-flat layout: zeta_q_G has shape (n_q, n_rmu, ngkmax); gvec_components is
# (n_q, 3, ngkmax); ngk is (n_q,).
assert f_sym_z['isdf_header/zeta_layout'][()] == b'G_flat'
assert f_nos_z['isdf_header/zeta_layout'][()] == b'G_flat'

z_sym = f_sym_z['zeta_q_G']                   # (8, 300, 13789)
gvc_sym = f_sym_z['isdf_header/gvec_components'][()]  # (8, 3, 13789)
ngk_sym_z = f_sym_z['isdf_header/ngk'][()]    # (8,)
z_nos = f_nos_z['zeta_q_G']                   # (36, 300, 13789)
gvc_nos = f_nos_z['isdf_header/gvec_components'][()]
ngk_nos_z = f_nos_z['isdf_header/ngk'][()]
r_mu_idx = f_sym_z['isdf_header/centroids/r_mu_fft_idx'][()]
fft_grid = w_sym.fft_grid

print(f"[T2] sym ζ: shape={z_sym.shape}, ngk={ngk_sym_z.tolist()}")
print(f"[T2] nosym ζ: shape={z_nos.shape}, ngk[:8]={ngk_nos_z[:8].tolist()}")
print(f"[T2] r_mu shape={r_mu_idx.shape}, fft_grid={tuple(fft_grid)}")

# Build centroid permutation π_s (length ntran rows; CrI3 has 0 TRS q's so
# we don't need extend_trs here).
from centroid.orbit_syms import compute_centroid_sym_perm
mu_perm = compute_centroid_sym_perm(
    r_mu_idx, sym.sym_matrices[:ntran], sym.translations[:ntran], fft_grid,
    extend_trs=False)
print(f"[T2] mu_perm shape={mu_perm.shape}, identity-on-row-0={np.array_equal(mu_perm[0], np.arange(mu_perm.shape[1]))}")
# Check at least one non-trivial row
print(f"[T2] mu_perm[3] (under -I) first 8 = {mu_perm[3, :8].tolist()}")

# Map sym q indices to nosym q indices using kvecs_asints lookup.
ki_sym_q = sym.kvecs_asints                  # (36, 3) int
ki_nos_q = sym_nosym.kvecs_asints            # (36, 3) int — for nosym
hash_sq = ki_sym_q[:, 0] * (kg[1] * kg[2]) + ki_sym_q[:, 1] * kg[2] + ki_sym_q[:, 2]
hash_nq = ki_nos_q[:, 0] * (kg[1] * kg[2]) + ki_nos_q[:, 1] * kg[2] + ki_nos_q[:, 2]
nos_pos_q = {int(h): i for i, h in enumerate(hash_nq.tolist())}
symq2nosq = np.array([nos_pos_q[int(h)] for h in hash_sq.tolist()], dtype=np.int64)
print(f"[T2] sym q-index → nosym q-index permutation built (first 8: {symq2nosq[:8].tolist()})")

# For each full-BZ q, build the algebraic predicted ζ_nosym from ζ_sym IBZ
# and compare to actual nosym ζ.
#
# Algebraic rule (symmorphic, τ=0):
#   For q_full at IBZ parent i = irr_idx_q[qf], sym row s = sym_idx_q[qf]:
#   For each G' in the nosym G-list at q_nos = symq2nosq[qf]:
#       Find G_ibz such that S G_ibz = G'  ⇒  G_ibz = S^{-1} G'
#       Then ζ_pred[π_s(μ), G'] = ζ_ibz[i, μ, G_ibz]
#   I.e., centroid axis permutes by π_s, G axis permutes by S^{-1}.

# Build per-IBZ-q G→row lookups (sym side).
def build_g_lookup(gvc_q, ngk_q):
    # gvc_q: (3, ngkmax) for one q
    gv = gvc_q[:, :ngk_q].T  # (ngk_q, 3)
    return {tuple(int(x) for x in row): i for i, row in enumerate(gv)}

g_lookup_sym = [build_g_lookup(gvc_sym[i], int(ngk_sym_z[i])) for i in range(8)]

# Read ALL sym IBZ ζ into memory once (8 × 300 × 13789 c128 ≈ 0.27 GB)
print("[T2] reading sym IBZ ζ into memory...")
z_sym_arr = z_sym[:]                          # (8, 300, 13789) c128
print(f"[T2]   done; {z_sym_arr.nbytes/1e9:.2f} GB")
print("[T2] reading nosym full-BZ ζ into memory (this is ~1.2 GB)...")
z_nos_arr = z_nos[:]                          # (36, 300, 13789) c128
print(f"[T2]   done; {z_nos_arr.nbytes/1e9:.2f} GB")

n_rmu = int(z_sym_arr.shape[1])

# Per-q tests
t2_results = []
n_q_full = sym.kvecs_asints.shape[0]
# Group q's by sym_idx to limit summary
q_by_sym = {}
for qf in range(n_q_full):
    s = int(sym.sym_idx_q[qf])
    q_by_sym.setdefault(s, []).append(qf)
test_qs = []
# Pick first occurrence of each sym index
for s in sorted(q_by_sym.keys()):
    test_qs.append(q_by_sym[s][0])
print(f"[T2] testing q reps (qf, sym_idx) = "
      f"{[(q, int(sym.sym_idx_q[q])) for q in test_qs]}")

# Also pick a few more q's for breadth
qf_extra = [qf for qf in range(n_q_full)
            if int(sym.sym_idx_q[qf]) != 0 and qf not in test_qs][:6]
test_qs.extend(qf_extra)

print(f"[T2] Testing {len(test_qs)} q's...")
for qf in test_qs:
    i_irr = int(sym.irr_idx_q[qf])
    s_idx = int(sym.sym_idx_q[qf])
    qn = int(symq2nosq[qf])
    ngk_n = int(ngk_nos_z[qn])
    gvc_n = gvc_nos[qn, :, :ngk_n].T          # (ngk_n, 3)
    ngk_i = int(ngk_sym_z[i_irr])

    if s_idx == 0 and i_irr == qf:
        # Trivial — verify bit-equality.
        delta = np.max(np.abs(z_sym_arr[i_irr, :, :ngk_i] - z_nos_arr[qn, :, :ngk_n]))
        t2_results.append({
            'qf': qf, 'i_irr': i_irr, 'sym_idx': s_idx, 'qn': qn,
            'kind': 'identity',
            'max_diff_pred_vs_nos': float(delta),
            'max_diff_lorrax_vs_pred': 0.0,
            'mat_S_kg': np.asarray(sym.sym_mats_k[s_idx]).tolist(),
            'det_S': int(np.linalg.det(np.asarray(sym.sym_mats_k[s_idx]))),
            'note': 'identity'})
        print(f"[T2] qf={qf:2d}, sym={s_idx} (identity): |Δ| = {delta:.3e}")
        continue

    # Hand-rolled prediction on nosym G-list at q_nos:
    S = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int64)
    Sinv = np.rint(np.linalg.inv(S)).astype(np.int64)
    pred = np.zeros((n_rmu, ngk_n), dtype=np.complex128)
    # For each G' in nosym list, find G_ibz = Sinv @ G' in sym IBZ list.
    g_pre = (Sinv @ gvc_n.T).T  # (ngk_n, 3) — the predicted G_ibz for each row in nosym
    lk = g_lookup_sym[i_irr]
    miss = 0
    for gp_idx, gibz in enumerate(g_pre):
        key = tuple(int(x) for x in gibz)
        gi = lk.get(key, -1)
        if gi < 0:
            miss += 1
            continue
        # Apply centroid permutation π_s: pred[π_s(μ), g_p] = ζ_ibz[i_irr, μ, gi]
        pred[mu_perm[s_idx], gp_idx] = z_sym_arr[i_irr, :, gi]
    # Compare prediction vs nosym ζ at qn (within the matched G slots)
    matched_mask = np.zeros(ngk_n, dtype=bool)
    for gp_idx, gibz in enumerate(g_pre):
        key = tuple(int(x) for x in gibz)
        if key in lk:
            matched_mask[gp_idx] = True
    if not matched_mask.any():
        print(f"[T2] qf={qf}, sym={s_idx}: ZERO G-vectors matched! Skipping.")
        continue
    delta = np.max(np.abs(pred[:, matched_mask] - z_nos_arr[qn, :, :ngk_n][:, matched_mask]))
    rel = float(delta / (np.max(np.abs(z_nos_arr[qn, :, :ngk_n][:, matched_mask])) + 1e-30))
    # Where is worst?
    worst_2d = np.unravel_index(
        np.argmax(np.abs(pred[:, matched_mask] - z_nos_arr[qn, :, :ngk_n][:, matched_mask])),
        (n_rmu, int(matched_mask.sum())))
    worst_mu = int(worst_2d[0])
    worst_gpos = int(np.where(matched_mask)[0][worst_2d[1]])
    t2_results.append({
        'qf': qf, 'i_irr': i_irr, 'sym_idx': s_idx, 'qn': qn,
        'kind': 'nontrivial',
        'max_diff_pred_vs_nos': float(delta),
        'rel_diff_pred_vs_nos': rel,
        'matched_G_count': int(matched_mask.sum()),
        'total_G_nos': ngk_n,
        'unmatched_G': int(miss),
        'worst_mu': worst_mu,
        'worst_gpos_in_nos': worst_gpos,
        'worst_G_vec': gvc_n[worst_gpos].tolist(),
        'mat_S_kg': np.asarray(sym.sym_mats_k[s_idx]).tolist(),
        'det_S': int(np.linalg.det(np.asarray(sym.sym_mats_k[s_idx]))),
    })
    print(f"[T2] qf={qf:2d}, sym={s_idx} (det={int(np.linalg.det(np.asarray(sym.sym_mats_k[s_idx]))):+d}): "
          f"max|Δ_pred_vs_nos|={delta:.3e} (rel={rel:.3e}); "
          f"matched G={int(matched_mask.sum())}/{ngk_n}, unmatched={miss}; "
          f"worst_mu={worst_mu}, worst_G_pos={worst_gpos}")

with open(os.path.join(OUT, 'test2_zeta_results.json'), 'w') as fh:
    json.dump(t2_results, fh, indent=2)

f_sym_z.close()
f_nos_z.close()

print("\n" + "=" * 78)
print("DONE.  See JSON files in", OUT)
print("=" * 78)
