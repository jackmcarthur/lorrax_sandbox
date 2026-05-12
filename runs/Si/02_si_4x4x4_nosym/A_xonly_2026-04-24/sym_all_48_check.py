"""Exhaustive test: for every spatial sym matrix in Si (48 ops + 48 TR),
find an IBZ k that can be mapped via that op to some full-BZ k, apply LORRAX's
rotation, and overlap against the nosym WFN at the target k.  Should be
block-unitary for every sym_idx after the fix.
"""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")

import numpy as np
from file_io import WFNReader
from common import symmetry_maps

wfn_s = WFNReader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5")
wfn_n = WFNReader("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5")
sym = symmetry_maps.SymMaps(wfn_s)
ntran = wfn_s.ntran
NBAND = 16

def kidx(kf, kps, tol=2e-3):
    for i, k in enumerate(kps):
        d = kf - k; d = d - np.round(d)
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


def custom_rotate(sym_idx, kbar_idx, target_kfrac):
    """Replicate LORRAX's get_cnk_fullzone_batch but letting us choose sym_idx.

    Matches the iσ_y·K convention for TR now live in symmetry_maps.py.
    """
    cnk = wfn_s.get_cnk_batch(kbar_idx, np.arange(NBAND))
    sym_krep = np.asarray(sym.sym_mats_k[sym_idx], dtype=np.int32)
    k_full = np.asarray(target_kfrac, dtype=np.float64)
    skbar = np.asarray(sym_krep @ wfn_s.kpoints[kbar_idx], dtype=np.float64)
    kg0 = np.rint(k_full - skbar).astype(np.int32)
    if not np.allclose(skbar + kg0, k_full, atol=1e-6):
        return None, None
    if sym_idx >= ntran:
        spatial_idx = sym_idx - ntran
        spatial_krep = np.asarray(sym.sym_mats_k[spatial_idx], dtype=np.int32)
        tau = np.asarray(wfn_s.translations[spatial_idx], dtype=np.float64)
        if np.any(np.abs(tau) > 1e-12):
            k_gvecs = wfn_s.get_gvec_nk(kbar_idx)
            rotated = np.einsum('ij,gj->gi', spatial_krep, np.asarray(k_gvecs, dtype=np.int32))
            phase = np.exp(-1j * rotated.astype(np.float64) @ tau)
            cnk = cnk * phase[None, None, :]
        # K then iσ_y: new[:,0]=conj(old[:,1]), new[:,1]=-conj(old[:,0])
        cnk = np.stack([np.conj(cnk[:, 1, :]), -np.conj(cnk[:, 0, :])], axis=1)
        cnk = np.einsum('jk,nkl->njl', sym.U_spinor[spatial_idx], cnk)
    else:
        tau = np.asarray(wfn_s.translations[sym_idx], dtype=np.float64)
        if np.any(np.abs(tau) > 1e-12):
            k_gvecs = wfn_s.get_gvec_nk(kbar_idx)
            rotated = np.einsum('ij,gj->gi', sym_krep, np.asarray(k_gvecs, dtype=np.int32))
            phase = np.exp(-1j * rotated.astype(np.float64) @ tau)
            cnk = cnk * phase[None, None, :]
        cnk = np.einsum('jk,nkl->njl', sym.U_spinor[sym_idx], cnk)
    # rotated G-list: same as before (sym_krep accounts for TR sign already)
    k_gvecs = wfn_s.get_gvec_nk(kbar_idx)
    gvecs_rot = np.einsum('ij,gj->gi', sym_krep, np.asarray(k_gvecs, dtype=np.int32))
    gvecs_rot = gvecs_rot - kg0[None, :]
    return cnk, gvecs_rot


# For each sym_idx, find any (kbar, k_full) pair and test
results = []  # (sym_idx, kbar, nk_full, off_block², unitarity_err)
nsym_total = len(sym.sym_mats_k)  # 2*ntran
tested = set()
for sym_idx in range(nsym_total):
    sym_krep = np.asarray(sym.sym_mats_k[sym_idx], dtype=np.int32)
    found = False
    for kbar_idx in range(wfn_s.nkpts):
        skbar = np.asarray(sym_krep @ wfn_s.kpoints[kbar_idx], dtype=np.float64)
        # Wrap into first BZ
        target = np.mod(skbar, 1.0)
        # Find in nosym WFN
        nn = kidx(target, wfn_n.kpoints)
        if nn is None:
            continue
        cnk_rot, gvecs_rot = custom_rotate(sym_idx, kbar_idx, target)
        if cnk_rot is None: continue
        cnos = wfn_n.get_cnk_batch(nn, np.arange(NBAND))
        gnos = wfn_n.get_gvec_nk(nn)
        S = overlap(cnk_rot, gvecs_rot, cnos, gnos, wfn_s.fft_grid)
        # block-diagonal test
        energies = wfn_n.energies[0, nn, :NBAND] * 13.6057
        blocks = []
        i = 0
        while i < NBAND:
            j = i + 1
            while j < NBAND and abs(energies[j] - energies[i]) < 5e-3:
                j += 1
            blocks.append((i, j)); i = j
        off = 0.0
        for i0, i1 in blocks:
            for j0, j1 in blocks:
                if i0 != j0: off += float(np.sum(np.abs(S[i0:i1, j0:j1]) ** 2))
        SS = S @ S.conj().T
        uerr = float(np.linalg.norm(SS - np.eye(NBAND)))
        results.append((sym_idx, kbar_idx, nn, off, uerr))
        found = True
        break
    if not found:
        results.append((sym_idx, -1, -1, float("nan"), float("nan")))

print(f"Tested {len([r for r in results if not np.isnan(r[3])])} / {nsym_total} sym ops "
      f"(ntran={ntran}: 0..{ntran-1} spatial, {ntran}..{nsym_total-1} time-reversed)")
bad = [r for r in results if not np.isnan(r[3]) and (r[3] > 1e-3 or r[4] > 1e-3)]
print(f"sym ops with bad rotation: {len(bad)} / {len([r for r in results if not np.isnan(r[3])])}")
if bad:
    print("\n  sym_idx  kbar  nk_full  off²       ‖SS†-I‖")
    for r in bad:
        print(f"  {r[0]:>7}  {r[1]:>4}  {r[2]:>7}  {r[3]:>9.4f}  {r[4]:>9.4f}")
# Also summarize by max off across all
offs = [r[3] for r in results if not np.isnan(r[3])]
print(f"\nmax off² across tested sym ops: {max(offs):.4e}")
print(f"mean off² across tested sym ops: {sum(offs)/len(offs):.4e}")

# Which sym ops were NEVER picked by irk_sym_map (would be untested in prod)?
picked = set(int(s) for s in sym.irk_sym_map)
all_idx = set(range(nsym_total))
unused = sorted(all_idx - picked)
print(f"\nsym ops picked by irk_sym_map: {sorted(picked)}")
print(f"sym ops NEVER picked (unused in prod): {unused}")
print(f"  (count: {len(unused)} / {nsym_total})")
