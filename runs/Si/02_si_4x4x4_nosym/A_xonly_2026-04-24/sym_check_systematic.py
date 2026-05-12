"""Per-sym-op test: is U or U† correct? Compare R_cart input options.
Also test if using R_sym_matrices (real-space) instead of sym_mats_k gives
the correct U."""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")

import numpy as np
from file_io import WFNReader
from common import symmetry_maps

SYM = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5"
NOS = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5"
NBAND = 8

wfn_s = WFNReader(SYM); wfn_n = WFNReader(NOS)
sym = symmetry_maps.SymMaps(wfn_s)
ntran = wfn_s.ntran

# Also build alternate U_spinor using real-space sym_matrices (not sym_mats_k)
# to test if the orientation matters.
B_T = np.asarray(wfn_s.bvec)
B_T_inv = np.linalg.inv(B_T)
# sym_matrices = real-space; sym_mats_k = transpose
R_real_cart = np.einsum('ij,njk,kl->nil', B_T_inv, sym.sym_matrices, B_T)
U_spinor_alt = sym.get_spinor_rotations(wfn_s, np.around(R_real_cart, 10))


def kidx(kfrac, kps, tol=2e-3):
    for i, k in enumerate(kps):
        d = kfrac - k; d = d - np.round(d)
        if np.max(np.abs(d)) < tol: return i
    return None


def overlap(cA, gA, cB, gB, fft):
    nx, ny, nz = [int(x) for x in fft]
    def to(cc, gg):
        n = cc.shape[0]
        G = np.zeros((n, 2, nx, ny, nz), dtype=np.complex128)
        gx = np.mod(gg[:, 0], nx); gy = np.mod(gg[:, 1], ny); gz = np.mod(gg[:, 2], nz)
        for i in range(n):
            G[i, 0, gx, gy, gz] = cc[i, 0]
            G[i, 1, gx, gy, gz] = cc[i, 1]
        return G.reshape(n, -1)
    return to(cA, gA).conj() @ to(cB, gB).T


def rotate(nk_full, U_choice, use_alt=False):
    sym_idx, kbar_idx, sym_krep = sym._get_symmetry_context(nk_full)
    cnk = wfn_s.get_cnk_batch(kbar_idx, np.arange(NBAND))
    if sym_idx >= ntran:
        cnk = np.conj(cnk)
    else:
        phase = sym._get_fractional_translation_phase(wfn_s, nk_full, sym_idx, kbar_idx, sym_krep)
        if phase is not None:
            cnk = cnk * phase[None, None, :]
    U_src = U_spinor_alt if use_alt else sym.U_spinor
    U = U_src[sym_idx]
    if U_choice == 'U†': U = U.conj().T
    elif U_choice == 'U*': U = U.conj()
    elif U_choice == 'U.T': U = U.T
    cnk = np.einsum('jk,nkl->njl', U, cnk)
    return cnk, sym.get_gvecs_kfull(wfn_s, nk_full)


# Test 56 non-trivial k-points with various U options
results = []
for nk_full in range(64):
    sym_idx, kbar_idx, _ = sym._get_symmetry_context(nk_full)
    if sym_idx == 0:
        continue
    k_full = np.asarray(sym.unfolded_kpts[nk_full])
    nk_nos = kidx(k_full, wfn_n.kpoints)
    if nk_nos is None: continue
    cnos = wfn_n.get_cnk_batch(nk_nos, np.arange(NBAND))
    gnos = wfn_n.get_gvec_nk(nk_nos)
    energies = wfn_n.energies[0, nk_nos, :NBAND] * 13.6057
    blocks = []
    i = 0
    while i < NBAND:
        j = i + 1
        while j < NBAND and abs(energies[j] - energies[i]) < 5e-3:
            j += 1
        blocks.append((i, j))
        i = j

    def off_block_norm(S):
        ob = 0.0
        for (i0,i1) in blocks:
            for (j0,j1) in blocks:
                if i0 != j0: ob += float(np.sum(np.abs(S[i0:i1,j0:j1])**2))
        return ob

    scores = {}
    for src_label, use_alt in [('orig (sym_mats_k)', False), ('alt (sym_matrices)', True)]:
        for U_choice in ['U', 'U†', 'U*', 'U.T']:
            crot, grot = rotate(nk_full, U_choice, use_alt=use_alt)
            S = overlap(crot, grot, cnos, gnos, wfn_s.fft_grid)
            scores[(src_label, U_choice)] = off_block_norm(S)
    results.append((nk_full, sym_idx, scores))

# For each sym_idx, see what consistently wins
from collections import defaultdict
sym_best = defaultdict(list)
for nk, si, scores in results:
    best = min(scores.items(), key=lambda kv: kv[1])
    sym_best[si].append((nk, best[0], best[1]))
print(f"Per-sym-op best variant (minimum off-block²):")
print(f"{'sym':>4} {'n_ks':>5}  best variants (off²)")
for si in sorted(sym_best.keys()):
    combos = defaultdict(int)
    for nk, bc, ob in sym_best[si]:
        combos[bc] += 1
    top = sorted(combos.items(), key=lambda x: -x[1])[0][0]
    worst_off = max(ob for _, _, ob in sym_best[si])
    print(f"{si:>4} {len(sym_best[si]):>5}  {str(top):30s}  worst off²={worst_off:.4f}")

# Variant-averaged scores
print("\nGlobal per-variant mean off-block², over all 56 non-trivial k:")
sums = defaultdict(list)
for nk, si, scores in results:
    for k, v in scores.items(): sums[k].append(v)
for k, v in sorted(sums.items(), key=lambda kv: np.mean(kv[1])):
    print(f"  {str(k):45s}  mean={np.mean(v):.4f}  max={np.max(v):.4f}  #bad(>1e-2)={int(np.sum(np.array(v)>1e-2))}")
