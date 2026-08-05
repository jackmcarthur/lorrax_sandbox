"""Test ψ unfold with alternate G-rotation convention.

LORRAX's wfn_loader does g_rot = sym_mats_k @ g_kbar = mtrx.T @ g_kbar.
Alternative: g_rot = mtrx @ g_kbar.

Check at sym_idx=1 (C3), 2 (C3^-1), 4 (S6), 5 (S6^-1) whether the alternative
makes the hand-roll ψ match nosym ψ up to a unitary on each degenerate group.
"""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('JAX_ENABLE_X64', '1')
import numpy as np
sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_B/src')
from common.symmetry_maps import SymMaps
from file_io.wfn_loader import WfnLoader

SYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_sym/WFN.h5'
NOSYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/07_M_6x6_30Ry_sym_vs_nosym_2026-05-14/run_nosym/WFN.h5'
w_sym = WfnLoader(SYM_WFN, backend='eager')
w_nos = WfnLoader(NOSYM_WFN, backend='eager')
sym = SymMaps(w_sym._sym_wfn_stub())
ntran = int(w_sym.ntran)

# Build sym↔nosym full-BZ k-map
kg = np.asarray(w_sym.kgrid)
unfolded = sym.unfolded_kpts
nosym_ks = np.asarray(w_nos.kpoints)
def as_int_kgrid(kfrac, kgrid):
    kw = np.mod(kfrac, 1.0)
    return np.mod(np.rint(kw * kgrid[None, :]).astype(np.int64), kgrid[None, :].astype(np.int64))
ki_sym = as_int_kgrid(unfolded, kg)
ki_nos = as_int_kgrid(nosym_ks, kg)
h_sym = ki_sym[:, 0]*kg[1]*kg[2] + ki_sym[:, 1]*kg[2] + ki_sym[:, 2]
h_nos = ki_nos[:, 0]*kg[1]*kg[2] + ki_nos[:, 1]*kg[2] + ki_nos[:, 2]
nos_pos = {int(h): i for i, h in enumerate(h_nos.tolist())}
sym2nos = np.array([nos_pos[int(h)] for h in h_sym.tolist()], dtype=np.int64)

# Quick utility
def raw_psi(wfn, k_idx):
    s = int(wfn._kpt_starts[k_idx])
    ngk = int(wfn.ngk[k_idx])
    raw = wfn._coeffs_raw[:, :, s:s+ngk, :]
    return raw[..., 0] + 1j*raw[..., 1]

def unfold_psi_hand(cnk, *, S_G, U):
    """Hand-roll: cnk has shape (nb, ns, ngk). Apply spinor U, G-rotation
    'S_G' which is the matrix that, when applied to g_ibz (column), gives
    the full-BZ G."""
    cnk_out = np.einsum('jk,nkl->njl', U, cnk)
    return cnk_out

# For each (k_full, sym_idx) with sym_idx in {1, 2, 3, 4, 5}, test both
# G-conventions: (LORRAX: g_rot = sym_mats_k @ g_ibz)
#               (alt: g_rot = mtrx @ g_ibz)
mtrx_list = sym.sym_matrices                    # mtrx[s]
mtrx_inv_list = np.linalg.inv(mtrx_list).round().astype(np.int64)

# Pick representatives
test_pairs = []
seen = set()
for kf in range(unfolded.shape[0]):
    s = int(sym.sym_idx_k[kf])
    if s not in seen:
        seen.add(s)
        test_pairs.append((kf, s))

n_bands_test = 80

for (kf, s_idx) in test_pairs:
    k_irr = int(sym.irr_idx_k[kf])
    k_nos = int(sym2nos[kf])
    ngk_irr = int(w_sym.ngk[k_irr])
    ngk_nos = int(w_nos.ngk[k_nos])
    g_ibz = w_sym._gvecs_raw[w_sym._kpt_starts[k_irr]:w_sym._kpt_starts[k_irr]+ngk_irr]
    g_nos = w_nos._gvecs_raw[w_nos._kpt_starts[k_nos]:w_nos._kpt_starts[k_nos]+ngk_nos]
    cnk_ibz = raw_psi(w_sym, k_irr)            # (nb, 2, ngk_irr)
    cnk_nos = raw_psi(w_nos, k_nos)            # (nb, 2, ngk_nos)
    nb = min(cnk_ibz.shape[0], cnk_nos.shape[0], n_bands_test)

    # LORRAX U_spinor at this sym
    U_lx = np.asarray(sym.U_spinor[s_idx]) if s_idx < ntran else None
    # Apply spinor rotation (LORRAX way)
    cnk_lx_rot = np.einsum('jk,nkl->njl', U_lx, cnk_ibz[:nb])

    # Compute energies (use sym IBZ energies)
    en_irr = np.asarray(w_sym.energies[0, k_irr, :nb])  # eV
    # Group by degenerate energy
    tol = 1e-5
    groups = []
    used = np.zeros(nb, dtype=bool)
    for i in range(nb):
        if used[i]: continue
        g = [i]; used[i] = True
        for j in range(i+1, nb):
            if not used[j] and abs(en_irr[j] - en_irr[i]) < tol:
                g.append(j); used[j] = True
        groups.append(g)

    print(f"\nkf={kf:2d}, k_irr={k_irr}, sym={s_idx}, det(mtrx)={int(np.linalg.det(mtrx_list[s_idx])):+d}: "
          f"nb={nb}, n_groups={len(groups)}")

    # Build two G-mappings: LORRAX (sym_mats_k @ g_ibz) and alt (mtrx @ g_ibz)
    for label, S_G in [('sym_mats_k', sym.sym_mats_k[s_idx]),
                        ('mtrx',       mtrx_list[s_idx])]:
        S_G = np.asarray(S_G, dtype=np.int64)
        g_rot = (S_G @ g_ibz.T).T  # (ngk_irr, 3)
        # Umklapp: kg0 makes the rotated k_irr land on k_full.
        # Use LORRAX's umklapp method for both — only the G-rotation we vary.
        sym_krep = np.asarray(sym.sym_mats_k[s_idx], dtype=np.int32)
        kg0 = sym._get_umklapp_vector(w_sym, kf, s_idx, k_irr, sym_krep)
        g_full_pred = g_rot - kg0[None, :]
        # Build lookup: g in nosym → row in g_full_pred
        lk = {tuple(int(x) for x in row): i for i, row in enumerate(g_full_pred)}
        nos_to_pred = np.array([lk.get(tuple(int(x) for x in g), -1) for g in g_nos], dtype=np.int64)
        matched = nos_to_pred >= 0
        n_match = int(matched.sum())
        if n_match < ngk_nos // 2:
            # alt try with -kg0 sign?
            g_full_pred2 = g_rot + kg0[None, :]
            lk2 = {tuple(int(x) for x in row): i for i, row in enumerate(g_full_pred2)}
            nos_to_pred2 = np.array([lk2.get(tuple(int(x) for x in g), -1) for g in g_nos], dtype=np.int64)
            n2 = int((nos_to_pred2 >= 0).sum())
            note = f"; with +kg0: {n2}"
        else:
            note = ""

        # Build cnk_hr in nosym G-order
        cnk_hr_n = np.zeros((nb, 2, ngk_nos), dtype=np.complex128)
        for gi in range(ngk_nos):
            j = nos_to_pred[gi]
            if j < 0: continue
            cnk_hr_n[:, :, gi] = cnk_lx_rot[:, :, j]
        # Build U[m, n] = <ψ_nos(m) | ψ_hr(n)>
        A = cnk_nos[:nb].reshape(nb, -1)
        B = cnk_hr_n.reshape(nb, -1)
        U = np.einsum('mi,ni->mn', np.conj(A), B)
        worst_unit = 0.0
        worst_off = 0.0
        for g in groups:
            Ug = U[np.ix_(g, g)]
            dev = np.max(np.abs(Ug @ np.conj(Ug).T - np.eye(len(g))))
            worst_unit = max(worst_unit, dev)
        for gi, g in enumerate(groups):
            for gj in range(gi+1, len(groups)):
                worst_off = max(worst_off, float(np.max(np.abs(U[np.ix_(g, groups[gj])]))))
        print(f"  G='{label:11s}': matched={n_match}/{ngk_nos}{note}, "
              f"worst-unitary={worst_unit:.3e}, worst-cross={worst_off:.3e}")
