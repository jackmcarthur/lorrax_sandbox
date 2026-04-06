"""Test LORRAX's actual SymMaps rotation against nosym WFN.
Uses the real SymMaps class and WFNReader, not a reimplementation."""
import numpy as np, h5py, sys
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps

SYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic/qe/nscf/WFN.h5'
NOSYM_WFN = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5'

# Load sym WFN via LORRAX's WFNReader + SymMaps
wfn = WFNReader(SYM_WFN)
sym = SymMaps(wfn)

# Load nosym WFN (simple h5py reader for the reference)
with h5py.File(NOSYM_WFN, 'r') as f:
    nosym_kpts = f['mf_header/kpoints/rk'][:]
    nosym_ngk = f['mf_header/kpoints/ngk'][:]
    nosym_el = f['mf_header/kpoints/el'][0]  # (nk, nband)
    nosym_gvecs = f['wfns/gvecs'][:]
    nosym_coeffs = f['wfns/coeffs'][:]

nosym_starts = np.zeros(len(nosym_kpts), dtype=np.int64)
nosym_starts[1:] = np.cumsum(nosym_ngk[:-1])

def nosym_get_gvec(ik):
    s = nosym_starts[ik]; return nosym_gvecs[s:s+nosym_ngk[ik]]

def nosym_get_cnk(ik, ib):
    s = nosym_starts[ik]; e = s + nosym_ngk[ik]
    raw = nosym_coeffs[ib, :, s:e, :]
    return raw[:,:,0] + 1j * raw[:,:,1]

nosym_kmap = {}
for ink, k in enumerate(nosym_kpts):
    km = k % 1.0; km[km > 0.9999] = 0.0
    nosym_kmap[tuple(np.round(km, 6))] = ink

def overlap(ca, ga, cb, gb):
    bd = {tuple(g): j for j, g in enumerate(gb)}
    ov = 0.0j
    for i, g in enumerate(ga):
        j = bd.get(tuple(g))
        if j is not None:
            ov += np.conj(ca[0,i])*cb[0,j] + np.conj(ca[1,i])*cb[1,j]
    return ov

print(f"SymMaps: {sym.nk_tot} full k-pts, {sym.nk_red} IBZ, ntran={wfn.ntran}")
print(f"sym_mats_k shape: {sym.sym_mats_k.shape}")
print(f"U_spinor shape: {sym.U_spinor.shape}\n")

DEGEN_TOL = 0.001
print(f"{'ikf':>3s}  {'k':>20s}  {'irk':>3s}  {'sym':>3s}  {'G-match':>8s}  {'max_err':>10s}")

n_good = 0; n_bad = 0
for ikf in range(sym.nk_tot):
    irk = sym.irk_to_k_map[ikf]
    isym = sym.irk_sym_map[ikf]
    kf = sym.unfolded_kpts[ikf]
    kfm = kf % 1.0; kfm[kfm > 0.9999] = 0.0
    ink = nosym_kmap.get(tuple(np.round(kfm, 6)))
    if ink is None: continue

    # LORRAX's actual G-vector rotation
    gvecs_rot = sym.get_gvecs_kfull(wfn, ikf)
    gvecs_nosym = nosym_get_gvec(ink)
    g_match = len(set(map(tuple, gvecs_rot)) & set(map(tuple, gvecs_nosym)))

    E = nosym_el[ink]
    bands = list(range(min(16, len(E))))
    groups = []; ib = 0
    while ib < len(bands):
        nb = bands[ib]; grp = [nb]
        while ib+1 < len(bands):
            if abs(E[bands[ib+1]] - E[nb]) < DEGEN_TOL:
                grp.append(bands[ib+1]); ib += 1
            else: break
        groups.append(grp); ib += 1

    max_err = 0.0
    for grp in groups:
        nd = len(grp)
        O = np.zeros((nd, nd), dtype=complex)
        for ii, ni in enumerate(grp):
            cn = nosym_get_cnk(ink, ni)
            for jj, nj in enumerate(grp):
                # LORRAX's actual rotation
                cr = sym.get_cnk_fullzone(wfn, nj, ikf)
                O[ii,jj] = overlap(cn, gvecs_nosym, cr, gvecs_rot)
        fro2 = np.sum(np.abs(O)**2)
        max_err = max(max_err, abs(float(nd) - fro2))

    flag = " ***" if max_err > 0.05 else ""
    if max_err > 0.05: n_bad += 1
    else: n_good += 1
    print(f"{ikf:3d}  ({kfm[0]:5.2f},{kfm[1]:5.2f},{kfm[2]:5.2f})  {irk:3d}  {isym:3d}  {g_match:4d}/{len(gvecs_nosym):<4d}  {max_err:10.6f}{flag}")

print(f"\nGOOD: {n_good}, BAD: {n_bad}")
