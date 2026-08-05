"""Definitively pin the TT Lorentz-unfold convention via the gauge-clean
centroid-trace, aligning the two v_q files by exact integer q-vectors.

M[q]_{ij} = sum_mu V^{ij}(q)[mu,mu] is phase-free (L-phases cancel on the
diagonal; alpha is a bijection), so M_scalar-unfold[q] = M_parent exactly,
and M_full[q] = (R-convention)(M_parent). The DIRECT file is ground truth;
whichever of {R M R^T (derivation A5), R^T M R (live code)} reproduces it is
the correct convention. The live code applies R^T M R (einsum 'qai,qbj').
"""
import sys, numpy as np, h5py
from types import SimpleNamespace
sys.path.insert(0, 'sources/lorrax_C/src')

WFN = 'runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
IBZ = 'runs/CrI3/C_cri3_ibz_active_2026-06-16/tmp/v_q_bispinor.h5'
DIR = 'runs/CrI3/C_cri3_full_bz_ref_2026-06-16/tmp/v_q_bispinor.h5'

with h5py.File(WFN, 'r') as f:
    c, k, s = f['/mf_header/crystal'], f['/mf_header/kpoints'], f['/mf_header/symmetry']
    wfn = SimpleNamespace(
        avec=c['avec'][()], atom_crys=c['apos'][()], atom_types=c['atyp'][()],
        sym_matrices=s['mtrx'][()], translations=s['tnp'][()], ntran=int(s['ntran'][()]),
        kgrid=k['kgrid'][()], kpoints=k['rk'][()], nkpts=int(k['nrk'][()]),
        shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym = SymMaps(wfn)
Rp   = np.asarray(sym.R_proper)
sidx = np.asarray(sym.sym_idx_q)
iidx = np.asarray(sym.irr_idx_q)
qfull= np.asarray(sym.q_irr_full_idx)
kvec = np.asarray(sym.kvecs_asints)              # (nq,3) sym's full-BZ q order
nq   = sidx.shape[0]
kg   = np.asarray(wfn.kgrid)
print(f"nq={nq} n_irr={qfull.shape[0]} ntran={wfn.ntran} kgrid={kg.tolist()}")

UNIQUE=[(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]
def load_M(path):
    with h5py.File(path,'r') as f:
        M=np.zeros((nq,3,3),complex)
        for (i,j) in UNIQUE:
            tr=np.einsum('qmm->q', f[f'V_qmunu_TT_{i}{j}'][()])
            M[:,i-1,j-1]=tr
            if i!=j: M[:,j-1,i-1]=np.conj(tr)
    return M
M_ibz=load_M(IBZ)                                # sym order
M_dir_raw=load_M(DIR)                            # nested-loop (qx,qy,qz) order

# direct file q-order = nested loops qx in [0,nkx), qy, qz (v_q_g_flat.py:207)
nkx,nky,nkz=int(kg[0]),int(kg[1]),int(kg[2])
nested=np.array([(qx,qy,qz) for qx in range(nkx) for qy in range(nky)
                 for qz in range(nkz)],dtype=int)
# map each sym-order q to its row in the nested (direct) ordering
key=lambda v:(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz)
nested_lut={key(nested[r]):r for r in range(nq)}
dir_row=np.array([nested_lut[key(kvec[q])] for q in range(nq)])
assert len(set(dir_row.tolist()))==nq, "q-vector alignment not bijective"
M_dir=M_dir_raw[dir_row]                          # now in sym order
print(f"aligned direct->sym by integer q-vec (bijection OK)")

print("\nq  op det parent | relresid R M R^T(deriv) | R^T M R(code) | |M_ibz-M_dir|rel")
fail_d=fail_c=ntest=0
for q in range(nq):
    op=int(sidx[q]); p=int(qfull[iidx[q]])
    if op%wfn.ntran==0 and op<wfn.ntran:  # identity
        continue
    R=Rp[op]; Mp=M_dir[p]; nf=max(np.linalg.norm(M_dir[q]),1e-9)
    e_d=np.linalg.norm(R@Mp@R.T - M_dir[q])/nf
    e_c=np.linalg.norm(R.T@Mp@R - M_dir[q])/nf
    e_bug=np.linalg.norm(M_ibz[q]-M_dir[q])/nf
    ntest+=1; fail_d+=e_d>1e-3; fail_c+=e_c>1e-3
    if ntest<=12:
        print(f"{q:2d} {op:2d} {np.linalg.det(R):+.0f} {p:3d} |  {e_d:.2e}  |  {e_c:.2e}  | {e_bug:.2e}")
print(f"\nover {ntest} non-trivial q:  R M R^T (derivation) fails {fail_d}   "
      f"R^T M R (code) fails {fail_c}")
print("VERDICT: convention with ~0 failures is correct. Code uses R^T M R; "
      "if that's the failing one, flip einsum 'qai,qbj'->'qia,qjb' (R M R^T).")
