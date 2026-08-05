"""Absolute vs relative in-plane Lorentz eigenvalue spread across transverse
centroid counts. Companion to _tt_covar_control.py: that one reports the
relative spread (max-min)/|mean| per orbit; this one also reports the ABSOLUTE
spread (max-min) and the mean eigenvalue magnitude, so a converging-residual
(absolute shrinks) is distinguishable from a relative-metric artifact (mean
shrinking faster than the spread).
"""
import sys, numpy as np, h5py
from types import SimpleNamespace
sys.path.insert(0, 'sources/lorrax_C/src')
WFN = 'runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
with h5py.File(WFN, 'r') as f:
    c, k, s = f['/mf_header/crystal'], f['/mf_header/kpoints'], f['/mf_header/symmetry']
    wfn = SimpleNamespace(avec=c['avec'][()], atom_crys=c['apos'][()], atom_types=c['atyp'][()],
        sym_matrices=s['mtrx'][()], translations=s['tnp'][()], ntran=int(s['ntran'][()]),
        kgrid=k['kgrid'][()], kpoints=k['rk'][()], nkpts=int(k['nrk'][()]), shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym = SymMaps(wfn)
iidx = np.asarray(sym.irr_idx_q); kvec = np.asarray(sym.kvecs_asints)
nq = len(iidx); kg = np.asarray(wfn.kgrid)
UNIQUE = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]

def load(DIR):
    with h5py.File(DIR, 'r') as f:
        M = np.zeros((nq, 3, 3), complex)
        for (i, j) in UNIQUE:
            tr = np.einsum('qmm->q', f[f'V_qmunu_TT_{i}{j}'][()]); M[:, i-1, j-1] = tr
            if i != j:
                M[:, j-1, i-1] = np.conj(tr)
    nkx, nky, nkz = [int(x) for x in kg]
    nested = [(qx, qy, qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)]
    lut = {(int(v[0]) % nkx, int(v[1]) % nky, int(v[2]) % nkz): r for r, v in enumerate(nested)}
    drow = np.array([lut[(int(kvec[q][0]) % nkx, int(kvec[q][1]) % nky, int(kvec[q][2]) % nkz)] for q in range(nq)])
    return M[drow]

from collections import defaultdict
orbits = defaultdict(list)
for q in range(nq):
    orbits[int(iidx[q])].append(q)
runs = {'102': 'runs/CrI3/C_cri3_full_bz_ref_2026-06-16/tmp/v_q_bispinor.h5',
        '206': 'runs/CrI3/C_cri3_covar_t204_2026-06-16/tmp/v_q_bispinor.h5',
        '308': 'runs/CrI3/C_cri3_covar_t306_2026-06-16/tmp/v_q_bispinor.h5'}
Ms = {kk: load(vv) for kk, vv in runs.items()}
print(f"{'orbit':>5} {'n':>2} | " + " | ".join(f"{kk+' cent':>28}" for kk in runs))
print(f"{'':>5} {'':>2} | " + " | ".join(f"{'absΔeig    relΔeig    meanEig':>28}" for kk in runs))
maxabs = {kk: 0.0 for kk in runs}; maxrel = {kk: 0.0 for kk in runs}
for ir, qs in sorted(orbits.items()):
    if len(qs) < 2:
        continue
    row = f"{ir:5d} {len(qs):2d} | "
    for kk in runs:
        M = Ms[kk]
        ev = np.array([np.sort(np.linalg.eigvalsh(M[q][:2, :2])) for q in qs]).real
        absd = max(ev[:, 0].max() - ev[:, 0].min(), ev[:, 1].max() - ev[:, 1].min())
        meane = np.mean(np.abs(ev))
        reld = absd / max(meane, 1e-12)
        maxabs[kk] = max(maxabs[kk], absd); maxrel[kk] = max(maxrel[kk], reld)
        row += f"{absd:.2e}  {reld:.2e}  {meane:.2e} | "
    print(row)
print("\nMAX over C3 orbits (in-plane Lorentz eigenvalue spread):")
for kk in runs:
    print(f"  {kk:>4} cent:  ABS spread = {maxabs[kk]:.3e}   REL spread = {maxrel[kk]:.3e}")
