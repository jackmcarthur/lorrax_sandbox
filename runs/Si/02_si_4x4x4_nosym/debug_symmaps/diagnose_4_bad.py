"""Deep dive into the 4 failing k-points.
Compare G-vectors, phases, spinor rotation, and overlaps in detail."""
import numpy as np, h5py
from common.wfnreader import WFNReader
from common.symmetry_maps import SymMaps

SYM_WFN = '../../01_si_4x4x4_nosymmorphic/qe/nscf/WFN.h5'
NOSYM_WFN = '../qe/nscf/WFN.h5'

wfn = WFNReader(SYM_WFN)
sym = SymMaps(wfn)

with h5py.File(NOSYM_WFN, 'r') as f:
    nosym_kpts = f['mf_header/kpoints/rk'][:]
    nosym_ngk = f['mf_header/kpoints/ngk'][:]
    nosym_gvecs = f['wfns/gvecs'][:]
    nosym_coeffs = f['wfns/coeffs'][:]
    nosym_el = f['mf_header/kpoints/el'][0]
nosym_starts = np.zeros(len(nosym_kpts), dtype=np.int64)
nosym_starts[1:] = np.cumsum(nosym_ngk[:-1])
def nosym_get_gvec(ik):
    s = nosym_starts[ik]; return nosym_gvecs[s:s+nosym_ngk[ik]]
def nosym_get_cnk(ik, ib):
    s = nosym_starts[ik]; e = s + nosym_ngk[ik]
    raw = nosym_coeffs[ib, :, s:e, :]; return raw[:,:,0] + 1j*raw[:,:,1]

nosym_kmap = {}
for ink, k in enumerate(nosym_kpts):
    km = k % 1.0; km[km > 0.9999] = 0.0
    nosym_kmap[tuple(np.round(km, 6))] = ink

# For each failing case, dump detailed info
bad_cases = [(4, 1, 5), (16, 1, 4), (30, 7, 7), (39, 7, 8)]

for ikf, irk, isym in bad_cases:
    kf = sym.unfolded_kpts[ikf]
    kfm = kf % 1.0; kfm[kfm > 0.9999] = 0.0
    ink = nosym_kmap[tuple(np.round(kfm, 6))]
    k_irk = wfn.kpoints[irk]
    
    print(f"\n{'='*70}")
    print(f"ikf={ikf}: k_full=({kfm[0]:.3f},{kfm[1]:.3f},{kfm[2]:.3f}), irk={irk}, sym={isym}")
    print(f"  k_irk = ({k_irk[0]:.6f}, {k_irk[1]:.6f}, {k_irk[2]:.6f})")
    print(f"  sym_mats_k[{isym}] =\n    {sym.sym_mats_k[isym].tolist()}")
    print(f"  U_spinor[{isym}] =\n    {sym.U_spinor[isym]}")
    
    # Check: S_k @ k_irk should give k_full (mod G)
    sk = sym.sym_mats_k[isym] @ k_irk
    print(f"  S_k @ k_irk = ({sk[0]:.6f}, {sk[1]:.6f}, {sk[2]:.6f})")
    print(f"  → mod 1 = ({sk[0]%1:.6f}, {sk[1]%1:.6f}, {sk[2]%1:.6f})")
    
    # G-vecs
    gvecs_rot = sym.get_gvecs_kfull(wfn, ikf)
    gvecs_nosym = nosym_get_gvec(ink)
    
    # Detailed 2x2 overlap for bands 0,1 (Kramers pair)
    O = np.zeros((2, 2), dtype=complex)
    for ii in range(2):
        cn = nosym_get_cnk(ink, ii)
        for jj in range(2):
            cr = sym.get_cnk_fullzone(wfn, jj, ikf)
            bd = {tuple(g): j for j, g in enumerate(gvecs_rot)}
            ov = 0.0j
            for i, g in enumerate(gvecs_nosym):
                j = bd.get(tuple(g))
                if j is not None:
                    ov += np.conj(cn[0,i])*cr[0,j] + np.conj(cn[1,i])*cr[1,j]
            O[ii,jj] = ov
    
    svs = np.linalg.svd(O, compute_uv=False)
    print(f"\n  2x2 overlap (bands 0,1):")
    print(f"    O = [[{O[0,0]:.4f}, {O[0,1]:.4f}],")
    print(f"         [{O[1,0]:.4f}, {O[1,1]:.4f}]]")
    print(f"    ||O||² = {np.sum(np.abs(O)**2):.4f}  (expected 2.0)")
    print(f"    svs = [{svs[0]:.6f}, {svs[1]:.6f}]")

    # Now check: what if we also try the PASSING k-point that uses the same sym?
    # For sym=5 (C3): also used at ikf=20 (irk=3, PASSES) and ikf=24 (irk=4, PASSES)
    # For sym=4 (C3): also used at ikf=17 (irk=3, PASSES) and ikf=33 (irk=4, PASSES)
    pass_ikf = None
    for ikf2 in range(64):
        if sym.irk_sym_map[ikf2] == isym and sym.irk_to_k_map[ikf2] != irk:
            # Check if it passes
            kf2 = sym.unfolded_kpts[ikf2]; kfm2 = kf2 % 1.0; kfm2[kfm2 > 0.9999] = 0.0
            ink2 = nosym_kmap.get(tuple(np.round(kfm2, 6)))
            if ink2 is None: continue
            irk2 = sym.irk_to_k_map[ikf2]
            gvecs_rot2 = sym.get_gvecs_kfull(wfn, ikf2)
            gvecs_nosym2 = nosym_get_gvec(ink2)
            O2 = np.zeros((2,2), dtype=complex)
            for ii in range(2):
                cn = nosym_get_cnk(ink2, ii)
                for jj in range(2):
                    cr = sym.get_cnk_fullzone(wfn, jj, ikf2)
                    bd = {tuple(g): j for j, g in enumerate(gvecs_rot2)}
                    ov = 0.0j
                    for i, g in enumerate(gvecs_nosym2):
                        j = bd.get(tuple(g))
                        if j is not None:
                            ov += np.conj(cn[0,i])*cr[0,j] + np.conj(cn[1,i])*cr[1,j]
                    O2[ii,jj] = ov
            fro2 = np.sum(np.abs(O2)**2)
            if fro2 > 1.9:
                print(f"\n  Same sym={isym} PASSES at ikf={ikf2} (irk={irk2}): ||O||²={fro2:.4f}")
                break
