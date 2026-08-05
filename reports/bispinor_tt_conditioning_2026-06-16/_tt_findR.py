"""For a few C3-mapped q, find which op/convention actually relates M_dir[q]
to M_dir[parent], and print the 3x3 matrices to see the structure."""
import sys, numpy as np, h5py
from types import SimpleNamespace
sys.path.insert(0, 'sources/lorrax_C/src')
np.set_printoptions(precision=3, suppress=True, linewidth=140)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
DIR='runs/CrI3/C_cri3_full_bz_ref_2026-06-16/tmp/v_q_bispinor.h5'
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
        sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
        kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn)
Rp=np.asarray(sym.R_proper); Rc=np.asarray(sym.R_cart)
sidx=np.asarray(sym.sym_idx_q); iidx=np.asarray(sym.irr_idx_q); qfull=np.asarray(sym.q_irr_full_idx)
kvec=np.asarray(sym.kvecs_asints); nq=sidx.shape[0]; kg=np.asarray(wfn.kgrid)
UNIQUE=[(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]
with h5py.File(DIR,'r') as f:
    M=np.zeros((nq,3,3),complex)
    for (i,j) in UNIQUE:
        tr=np.einsum('qmm->q',f[f'V_qmunu_TT_{i}{j}'][()]); M[:,i-1,j-1]=tr
        if i!=j: M[:,j-1,i-1]=np.conj(tr)
nkx,nky,nkz=[int(x) for x in kg]
nested=np.array([(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)])
lut={(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz):r for r,v in enumerate(nested)}
drow=np.array([lut[(int(kvec[q][0])%nkx,int(kvec[q][1])%nky,int(kvec[q][2])%nkz)] for q in range(nq)])
M=M[drow]                                        # sym order
ntran=wfn.ntran

for q in [11,16]:
    op=int(sidx[q]); p=int(qfull[iidx[q]])
    print(f"\n===== q={q} op={op} parent={p} (kvec q={kvec[q].tolist()} parent={kvec[p].tolist()}) =====")
    print("M_dir[parent].real=\n",M[p].real)
    print("M_dir[q].real=\n",M[q].real)
    print(f"R_cart[op].real=\n{Rc[op].real}\n det={np.linalg.det(Rc[op]):.2f}")
    # search every op + both conventions for best reproduction of M[q] from M[p]
    best=[]
    for o in range(2*ntran):
        for conv,lab in [('a','R M R^T'),('b','R^T M R')]:
            R=Rp[o]
            pred=R@M[p]@R.T if conv=='a' else R.T@M[p]@R
            e=np.linalg.norm(pred-M[q])/max(np.linalg.norm(M[q]),1e-9)
            best.append((e,o,lab))
    best.sort()
    print("best transforms M[parent]->M[q]:")
    for e,o,lab in best[:4]:
        print(f"   resid {e:.2e}  op={o} ({lab})  det(Rp[op])={np.linalg.det(Rp[o]):+.0f}")
    # also: is M[q] reproduced from SOME other parent q' under op=sidx[q]?
    R=Rp[op]; bq=[]
    for qp in range(nq):
        e=np.linalg.norm(R@M[qp]@R.T - M[q])/max(np.linalg.norm(M[q]),1e-9)
        bq.append((e,qp))
    bq.sort()
    print(f"  using op={op} R M R^T, best source q': {[(round(e,3),qp) for e,qp in bq[:3]]}  (parent={p})")
