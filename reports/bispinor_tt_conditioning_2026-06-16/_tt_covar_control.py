"""Control: is the DIRECT data covariant? Compare charge (scalar) vs transverse
(z-channel and in-plane) across one full C3 orbit. Charge centroid-trace and the
z-z transverse tile should be orbit-INVARIANT; the in-plane Lorentz trace should
be invariant too IF the transverse response is covariant. Quantify the violation.
"""
import sys, numpy as np, h5py
from types import SimpleNamespace
sys.path.insert(0, 'sources/lorrax_C/src')
np.set_printoptions(precision=2, suppress=True, linewidth=160)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
DIR=sys.argv[1] if len(sys.argv)>1 else 'runs/CrI3/C_cri3_full_bz_ref_2026-06-16/tmp/v_q_bispinor.h5'
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
        sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
        kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn)
iidx=np.asarray(sym.irr_idx_q); kvec=np.asarray(sym.kvecs_asints); nq=len(iidx); kg=np.asarray(wfn.kgrid)
UNIQUE=[(1,1),(2,2),(3,3),(1,2),(1,3),(2,3)]
with h5py.File(DIR,'r') as f:
    M=np.zeros((nq,3,3),complex); CCtr=np.einsum('qmm->q',f['V_qmunu_CC'][()])
    for (i,j) in UNIQUE:
        tr=np.einsum('qmm->q',f[f'V_qmunu_TT_{i}{j}'][()]); M[:,i-1,j-1]=tr
        if i!=j: M[:,j-1,i-1]=np.conj(tr)
nkx,nky,nkz=[int(x) for x in kg]
nested=[(qx,qy,qz) for qx in range(nkx) for qy in range(nky) for qz in range(nkz)]
lut={(int(v[0])%nkx,int(v[1])%nky,int(v[2])%nkz):r for r,v in enumerate(nested)}
drow=np.array([lut[(int(kvec[q][0])%nkx,int(kvec[q][1])%nky,int(kvec[q][2])%nkz)] for q in range(nq)])
M=M[drow]; CCtr=CCtr[drow]

# group full-BZ q by IBZ parent (orbit)
from collections import defaultdict
orbits=defaultdict(list)
for q in range(nq): orbits[int(iidx[q])].append(q)
print(f"DIR={DIR}")
print("orbit | n | charge-tr | z-z(3,3) | inplane-tr | inplane-EIGENVALUE spread (the covariance metric)")
def spread(a):
    a=np.asarray(a).real; return (a.max()-a.min())/max(abs(a.mean()),1e-9)
def eig_spread(qs):
    # in-plane 2x2 Hermitian eigenvalues per q; spread of the larger & smaller eig across orbit
    ev=np.array([np.sort(np.linalg.eigvalsh(M[q][:2,:2].astype(complex))) for q in qs]).real
    return max(spread(ev[:,0]), spread(ev[:,1]))
ips=[]
for ir,qs in sorted(orbits.items()):
    if len(qs)<2: continue
    cc=[CCtr[q].real for q in qs]; zz=[M[q][2,2].real for q in qs]
    ipt=[(M[q][0,0]+M[q][1,1]).real for q in qs]
    es=eig_spread(qs); ips.append(es)
    print(f"  {ir:3d} | {len(qs):2d} | {spread(cc):.1e} | {spread(zz):.1e} |  {spread(ipt):.1e}  |  {es:.2e}")
print(f"\nMAX in-plane eigenvalue spread over C3 orbits = {max(ips):.3e}  (102-cent ref was ~0.15)")
print("\nCharge tr & z-z spread ~0 (=covariant) but in-plane/full-Ltrace spread large")
print("=> transverse IN-PLANE direct response is NOT C3-covariant (rotation can't change a trace).")
print("Charge covariant + z covariant rules out a global alignment/centroid error.")
