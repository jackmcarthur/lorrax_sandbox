"""Diagnostic: test multiple G-rotation conventions for ζ unfold.

For each (q_full, sym_idx) non-trivial, try four rules to map G_full → G_ibz:
  (A) G_ibz = mtrx_inv @ G_full          (column;  textbook BGW)
  (B) G_ibz = mtrx @ G_full              (just transpose-swap of A)
  (C) G_ibz = sym_mats_k @ G_full         (LORRAX's "k-action" matrix; = mtrx.T)
  (D) G_ibz = sym_mats_k^{-1} @ G_full    (inverse of C; = mtrx_inv.T)

Also try four conventions for μ permutation (forward/inverse of sym_perm).

Reports max |Δ_pred_vs_nos| over all matched G-slots, for each (A/B/C/D) × (forward/inverse perm) × q.
"""
from __future__ import annotations
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

# Maps for q
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
fft_grid = w_sym.fft_grid

mu_perm = compute_centroid_sym_perm(
    r_mu_idx, sym.sym_matrices[:ntran], sym.translations[:ntran], fft_grid)
inv_mu_perm = np.argsort(mu_perm, axis=-1)

# Pick a non-trivial q to test (avoid q_full = q_irr)
# qf=4 (sym=3, -I) maps q_irr=2 (000-1 0 wait let me re-pick)
# Use qf=10 sym=1, qf=4 sym=3 (-I), qf=13 sym=2, qf=11 sym=1
test_qs = []
for qf in range(36):
    if int(sym.sym_idx_q[qf]) > 0:
        test_qs.append(qf)
test_qs = test_qs[:6]
print(f"Testing q_full = {test_qs}")
print(f"Their (i_irr, sym_idx) = {[(int(sym.irr_idx_q[q]), int(sym.sym_idx_q[q])) for q in test_qs]}")

# Convention rules for the G-back-rotation
def conv_A_Sinv_mtrx(S_kg, gvc):
    """S^{-1} G' where S=mtrx.  S_kg here is sym_mats_k = mtrx.T, so mtrx = S_kg.T."""
    mtrx = S_kg.T  # = sym_matrices[s_idx]
    mtrx_inv = np.rint(np.linalg.inv(mtrx)).astype(np.int64)
    return (mtrx_inv @ gvc.T).T

def conv_B_mtrx(S_kg, gvc):
    """G' = mtrx G  (forward)"""
    mtrx = S_kg.T
    return (mtrx @ gvc.T).T

def conv_C_Skg(S_kg, gvc):
    """G' = sym_mats_k G  (= mtrx.T G)"""
    return (S_kg @ gvc.T).T

def conv_D_Skg_inv(S_kg, gvc):
    """G' = sym_mats_k^{-1} G"""
    S_inv = np.rint(np.linalg.inv(S_kg)).astype(np.int64)
    return (S_inv @ gvc.T).T

convs = {'A_mtrx_inv': conv_A_Sinv_mtrx,
         'B_mtrx': conv_B_mtrx,
         'C_sym_mats_k': conv_C_Skg,
         'D_sym_mats_k_inv': conv_D_Skg_inv}
mu_choice = {'fwd_perm[s,μ]': mu_perm, 'inv_perm[s,μ]': inv_mu_perm}

# Build per-IBZ G-list lookups
def build_lk(gvc_q, ngk_q):
    gv = gvc_q[:, :ngk_q].T
    return {tuple(int(x) for x in row): i for i, row in enumerate(gv)}

g_lk_sym = [build_lk(gvc_sym[i], int(ngk_sym[i])) for i in range(8)]

for qf in test_qs:
    i_irr = int(sym.irr_idx_q[qf])
    s_idx = int(sym.sym_idx_q[qf])
    qn = int(symq2nosq[qf])
    S_kg = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int64)
    ngk_n = int(ngk_nos[qn])
    gvc_n = gvc_nos[qn, :, :ngk_n].T
    print(f"\n=== qf={qf}, i_irr={i_irr}, sym={s_idx}  S_kg=\n{S_kg}")
    for cname, cfn in convs.items():
        g_pre = cfn(S_kg, gvc_n)
        lk = g_lk_sym[i_irr]
        nos_to_ibz = -np.ones(ngk_n, dtype=np.int64)
        for gi, gv in enumerate(g_pre):
            key = tuple(int(x) for x in gv)
            nos_to_ibz[gi] = lk.get(key, -1)
        matched = (nos_to_ibz >= 0)
        if not matched.any():
            print(f"  conv {cname:18s}: ZERO G matched")
            continue
        # For each mu convention, build prediction and compare
        for mname, mtbl in mu_choice.items():
            pred = np.zeros((mu_perm.shape[1], ngk_n), dtype=np.complex128)
            for gi in range(ngk_n):
                if nos_to_ibz[gi] < 0:
                    continue
                pred[mtbl[s_idx], gi] = z_sym[i_irr, :, nos_to_ibz[gi]]
            delta = np.max(np.abs(pred[:, matched] - z_nos[qn, :, :ngk_n][:, matched]))
            rel = float(delta / (np.max(np.abs(z_nos[qn, :, :ngk_n][:, matched])) + 1e-30))
            print(f"  conv {cname:18s} × μ {mname:14s}: matched={int(matched.sum()):5d}/{ngk_n}, "
                  f"max|Δ|={delta:.3e}  (rel={rel:.3e})")

f_sym.close(); f_nos.close()
