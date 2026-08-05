"""Independent verification of the discriminator: in-plane covariance + magnitude
vs transverse centroid count. Reports per-run: max in-plane eigenvalue spread over
C3 orbits (covariance) AND max |in-plane centroid-trace| magnitude (conditioning).
"""
import sys, numpy as np, h5py
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
RUNS=[('102','runs/CrI3/C_cri3_full_bz_ref_2026-06-16'),
      ('206','runs/CrI3/C_cri3_covar_t204_2026-06-16'),
      ('308','runs/CrI3/C_cri3_covar_t306_2026-06-16'),
      ('410','runs/CrI3/C_cri3_covar_t408_2026-06-16')]
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
        sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
        kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn); iidx=np.asarray(sym.irr_idx_q); kvec=np.asarray(sym.kvecs_asints)
nq=len(iidx); kg=np.asarray(wfn.kgrid)
nkx,nky,nkz=[int(x) for x in kg]
nested=[(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)]
lut={(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz):r for r,v in enumerate(nested)}
drow=np.array([lut[(int(kvec[q][0])%nkx,int(kvec[q][1])%nky,int(kvec[q][2])%nkz)] for q in range(nq)])
from collections import defaultdict
orbits=defaultdict(list)
for q in range(nq): orbits[int(iidx[q])].append(q)
UNIQUE=[(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]
def spread(a): a=np.asarray(a).real; return (a.max()-a.min())/max(abs(a.mean()),1e-30)
print(f"{'cent':>5} | {'n_T':>4} | {'max charge-tr':>13} | {'max z-z spr':>11} | {'max inplane-eig spr':>19} | {'max|inplane-tr|':>15} | {'max|V^11 entry|':>15}")
for name,rd in RUNS:
    p=f'{rd}/tmp/v_q_bispinor.h5'
    with h5py.File(p,'r') as f:
        nT=f['V_qmunu_TT_11'].shape[1]
        M=np.zeros((nq,3,3),complex); CC=np.einsum('qmm->q',f['V_qmunu_CC'][()])
        maxabs11=float(np.abs(f['V_qmunu_TT_11'][()]).max())
        for (i,j) in UNIQUE:
            tr=np.einsum('qmm->q',f[f'V_qmunu_TT_{i}{j}'][()]); M[:,i-1,j-1]=tr
            if i!=j: M[:,j-1,i-1]=np.conj(tr)
    M=M[drow]; CC=CC[drow]
    cc_s=[];zz_s=[];eg_s=[];ipmag=[]
    for ir,qs in orbits.items():
        if len(qs)<2: continue
        cc_s.append(spread([CC[q].real for q in qs]))
        zz_s.append(spread([M[q][2,2].real for q in qs]))
        ev=np.array([np.sort(np.linalg.eigvalsh(M[q][:2,:2])) for q in qs]).real
        eg_s.append(max(spread(ev[:,0]),spread(ev[:,1])))
        ipmag.append(max(abs(M[q][0,0]+M[q][1,1]) for q in qs))
    print(f"{name:>5} | {nT:>4} | {max(cc_s):>13.2e} | {max(zz_s):>11.2e} | {max(eg_s):>19.3e} | {max(ipmag):>15.3e} | {maxabs11:>15.3e}")
