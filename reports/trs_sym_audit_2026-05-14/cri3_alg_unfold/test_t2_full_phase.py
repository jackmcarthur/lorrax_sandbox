"""Test full ζ unfold rule including the (Sq+G)·τ phase and centroid perm conventions.

For symmorphic CrI3 (τ=0), τ-phase vanishes, but the "centroid permutation
direction" is still a question.  Try:

  (i) "forward π_s" = compute_centroid_sym_perm (which uses Rinv = inv(mtrx))
       — current LORRAX behavior
 (ii) "forward π'_s" using mtrx instead of mtrx_inv to define the real-space
       rotation of centroids

Also try the full disk-ζ unfold formula derived from first principles:

  ζ_disk[Sq, π_s(μ), G_full] = e^{-2πi (Sq+G_full)·τ}
                              × ζ_disk[q, μ, mtrx^{-T} · (Sq+G_full) - q]

For τ=0 and using the convention that Sq = mtrx·q  AND  q_irr = mtrx^{-T}·Sq:
  ζ_disk[Sq, π_s(μ), G_full] = ζ_disk[q, μ, mtrx^{-T} · G_full]

Compare with another candidate: ζ_disk[q, μ, mtrx^T · G_full]
"""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
import h5py
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_B/src')
from common.symmetry_maps import SymMaps
from file_io.wfn_loader import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm

SYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/WFN.h5'
NOSYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/WFN.h5'
SYM_ZETA = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/tmp/zeta_q.h5'
NOSYM_ZETA = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/tmp/zeta_q.h5'

w_sym = WfnLoader(SYM_WFN, backend='eager')
w_nos = WfnLoader(NOSYM_WFN, backend='eager')
sym = SymMaps(w_sym._sym_wfn_stub())
sym_n = SymMaps(w_nos._sym_wfn_stub())
ntran = int(w_sym.ntran)
kg = np.asarray(w_sym.kgrid)
ki_sym_q = sym.kvecs_asints
ki_nos_q = sym_n.kvecs_asints
hash_sq = ki_sym_q[:, 0]*kg[1]*kg[2] + ki_sym_q[:, 1]*kg[2] + ki_sym_q[:, 2]
hash_nq = ki_nos_q[:, 0]*kg[1]*kg[2] + ki_nos_q[:, 1]*kg[2] + ki_nos_q[:, 2]
nos_pos_q = {int(h): i for i, h in enumerate(hash_nq.tolist())}
symq2nosq = np.array([nos_pos_q[int(h)] for h in hash_sq.tolist()], dtype=np.int64)

f_sym = h5py.File(SYM_ZETA,'r')
f_nos = h5py.File(NOSYM_ZETA,'r')
z_sym = f_sym['zeta_q_G'][:]
z_nos = f_nos['zeta_q_G'][:]
gvc_sym = f_sym['isdf_header/gvec_components'][()]
ngk_sym = f_sym['isdf_header/ngk'][()]
gvc_nos = f_nos['isdf_header/gvec_components'][()]
ngk_nos = f_nos['isdf_header/ngk'][()]
r_mu_idx = f_sym['isdf_header/centroids/r_mu_fft_idx'][()]
fft_grid = np.asarray(w_sym.fft_grid, dtype=np.int32)

# Two flavors of mu_perm.
# Flavor A: LORRAX current — uses Rinv = inv(mtrx)
mu_perm_A = compute_centroid_sym_perm(
    r_mu_idx, sym.sym_matrices[:ntran], sym.translations[:ntran], fft_grid)
# Flavor B: with mtrx itself (no inverse)
mu_perm_B = compute_centroid_sym_perm(
    r_mu_idx,
    np.linalg.inv(sym.sym_matrices[:ntran]).round().astype(np.int32),
    sym.translations[:ntran], fft_grid)
print(f"mu_perm_A[1, :10] = {mu_perm_A[1, :10].tolist()}")
print(f"mu_perm_B[1, :10] = {mu_perm_B[1, :10].tolist()}")
print(f"mu_perm_A[3, :10] = {mu_perm_A[3, :10].tolist()}")
print(f"mu_perm_B[3, :10] = {mu_perm_B[3, :10].tolist()}")

# All G-rotation conventions to try
S_list = sym.sym_mats_k  # length 2*ntran; we only test [:ntran]
mtrx_list = sym.sym_matrices  # = wfn.sym_matrices[:ntran]
mtrx_inv_list = np.linalg.inv(mtrx_list).round().astype(np.int64)

# Per-q phase: BGW wrap for q_full and q_irr, in fractional units.
def bgw_wrap_frac(q_int, kg_arr):
    q = np.asarray(q_int, dtype=np.float64)
    return np.where(q > kg_arr/2, q - kg_arr, q) / kg_arr

q_full_frac = bgw_wrap_frac(sym.kvecs_asints, kg)              # (36, 3)
q_irr_frac  = bgw_wrap_frac(sym.q_irr_kgrid_int, kg)            # (8, 3)
r_mu_frac = r_mu_idx.astype(np.float64) / fft_grid[None, :]    # (n_rmu, 3) in [0,1)

# Now per-q test, with multiple G rotations and multiple mu-perm choices.
def build_lk(gvc_q, ngk_q):
    gv = gvc_q[:, :ngk_q].T
    return {tuple(int(x) for x in row): i for i, row in enumerate(gv)}
g_lk_sym = [build_lk(gvc_sym[i], int(ngk_sym[i])) for i in range(8)]

# G rotations to try; describe via name -> int matrix that maps G_full -> G_ibz
def make_g_conv(name, S_kg, mtrx, mtrx_inv):
    if name == 'mtrx_inv':       return mtrx_inv
    if name == 'mtrx':           return mtrx
    if name == 'mtrx_inv_T':     return mtrx_inv.T
    if name == 'mtrx_T':         return mtrx.T
    if name == 'sym_mats_k':     return S_kg
    if name == 'sym_mats_k_inv': return np.linalg.inv(S_kg).round().astype(np.int64)

n_rmu = 300
test_qs = [10, 11, 13, 4, 5, 6]   # mix of C3, S6, -I
for qf in test_qs:
    i_irr = int(sym.irr_idx_q[qf])
    s_idx = int(sym.sym_idx_q[qf])
    qn = int(symq2nosq[qf])
    ngk_n = int(ngk_nos[qn])
    gvc_n = gvc_nos[qn, :, :ngk_n].T
    S_kg = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int64)
    mtrx = np.asarray(mtrx_list[s_idx], dtype=np.int64)
    mtrx_inv = mtrx_inv_list[s_idx]
    print(f"\n=== qf={qf}, i_irr={i_irr}, sym={s_idx}, qn={qn}")
    print(f"    mtrx=\n{mtrx}")
    print(f"    mtrx_inv=\n{mtrx_inv}")
    print(f"    sym_mats_k=\n{S_kg}")
    print(f"    q_irr_frac={q_irr_frac[i_irr]}, q_full_frac={q_full_frac[qf]}")
    # Build prediction with multiple combinations
    for gname in ['mtrx_inv', 'mtrx', 'mtrx_inv_T', 'mtrx_T']:
        M = make_g_conv(gname, S_kg, mtrx, mtrx_inv)
        g_pre = (M @ gvc_n.T).T  # (ngk_n, 3) — predicted G_ibz for each row
        lk = g_lk_sym[i_irr]
        nos_to_ibz = np.array([lk.get(tuple(int(x) for x in g), -1) for g in g_pre], dtype=np.int64)
        matched = nos_to_ibz >= 0
        m_count = int(matched.sum())
        if m_count == 0:
            print(f"  G='{gname}': ZERO matched")
            continue
        for mu_name, mtbl in [('mu_perm_A_fwd', mu_perm_A), ('mu_perm_B_fwd', mu_perm_B)]:
            # Try with NO extra phase
            pred = np.zeros((n_rmu, ngk_n), dtype=np.complex128)
            for gi in range(ngk_n):
                j = nos_to_ibz[gi]
                if j < 0: continue
                pred[mtbl[s_idx], gi] = z_sym[i_irr, :, j]
            delta_noph = np.max(np.abs(pred[:, matched] - z_nos[qn, :, :ngk_n][:, matched]))
            # Try with phase exp(-2πi (Sq+G_full)·τ) — but τ=0 here so trivial
            # The bigger phase candidate is the "disk-phase difference":
            #   ζ_disk[Sq, π_s(μ), G] = e^{-2πi[(Sq)·R r_μ - q·r_μ]} ζ_disk[q, μ, ...]
            # For τ=0:
            #   delta_phase[μ, G] = exp(-2πi (Sq·r_{π_s(μ)} - q·r_μ))
            # Apply this phase to the pred. r_{π_s(μ)} = mu_perm-rotated r_μ.
            # Build r_perm:
            r_pi_s = r_mu_frac[mtbl[s_idx]]  # (n_rmu, 3)
            q_irr_v = q_irr_frac[i_irr]
            q_full_v = q_full_frac[qf]
            phase_mu = np.exp(-2j*np.pi*(np.einsum('mi,i->m', r_pi_s, q_full_v)
                                          - np.einsum('mi,i->m', r_mu_frac, q_irr_v)))
            pred_phase = pred * phase_mu[:, None]
            delta_ph = np.max(np.abs(pred_phase[:, matched] - z_nos[qn, :, :ngk_n][:, matched]))
            # Also: phase based on r_{π_s(μ)} and r_μ (= mu_perm[s, μ] selection)
            print(f"  G='{gname:11s}' μ='{mu_name:13s}': matched={m_count}/{ngk_n}, "
                  f"|Δ_no_phase|={delta_noph:.3e}, |Δ_phase|={delta_ph:.3e}")

f_sym.close(); f_nos.close()
