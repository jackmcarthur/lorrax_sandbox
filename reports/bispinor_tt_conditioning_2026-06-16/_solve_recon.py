"""Confirm the SOLVE is the covariance-breaking site.

The ISDF fit:  zeta^i_q = (C^i_q + ridge)^{-1} Z^i_q,  then physical metric
reconstructed as  V^{ij}_recon = zeta^i^dag (Coulomb) zeta^j.  Here we drop the
Coulomb weight (gauge-inv S already channel/orbit covariant test) and reconstruct
the centroid metric M^{ij}_recon[mu,nu] = sum_r zeta^i(mu,r) conj(zeta^j(nu,r)).
The GAUGE-INVARIANT sum_ij ||M^{ij}||_F^2 must be orbit-constant.

We compare the reconstruction under:
  (A) the production ridge-LU solve (ridge=1e-12 rel), float64
  (B) an exact SVD pseudoinverse with GENEROUS rcond (well-conditioned modes only)
to show the production solve injects a q-dependent error (cond-driven), while an
honest pinv keeps S(q) orbit-constant -> the SOLVE is the site, not gamma/U/load.

Z^i_q here (q=0) is the centroid-cross zeta target: for the ISDF identity at q=0,
Z^i[mu,r] = sum_{mn} M^i_{mn}(mu) conj(M^i_{mn}(r)) = C^i (square, n_rmu x n_rmu).
So zeta = (C+ridge)^{-1} C ~ I (well-cond modes) and the reconstruction
M_recon = zeta C zeta^dag is what loses covariance when cond is high & q-varying.
"""
import os; os.environ['JAX_ENABLE_X64']='1'
import numpy as np, h5py, sys
from types import SimpleNamespace
sys.path.insert(0,'sources/lorrax_C/src')
np.set_printoptions(precision=4,suppress=True,linewidth=160)
WFN='runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5'
CENTS='runs/CrI3/C_cri3_ibz_active_2026-06-16/centroids_frac_102_current.txt'
with h5py.File(WFN,'r') as f:
    c,k,s=f['/mf_header/crystal'],f['/mf_header/kpoints'],f['/mf_header/symmetry']
    wfn=SimpleNamespace(avec=c['avec'][()],atom_crys=c['apos'][()],atom_types=c['atyp'][()],
      sym_matrices=s['mtrx'][()],translations=s['tnp'][()],ntran=int(s['ntran'][()]),
      kgrid=k['kgrid'][()],kpoints=k['rk'][()],nkpts=int(k['nrk'][()]),shift=k['shift'][()])
    fft_grid=f['/mf_header/gspace/FFTgrid'][()]
from common.symmetry_maps import SymMaps
sym=SymMaps(wfn); Rp=np.asarray(sym.R_proper)
sym_idx_k=np.asarray(sym.sym_idx_k); irr_idx_k=np.asarray(sym.irr_idx_k)
nx,ny,nz=[int(v) for v in fft_grid]
from common.gamma_matrices import gamma1,gamma2,gamma3
gam=[None,np.asarray(gamma1),np.asarray(gamma2),np.asarray(gamma3)]
from file_io.wfn_loader import WfnLoader
from file_io.centroids import load_centroids
from centroid.orbit_syms import compute_centroid_sym_perm
_,cents_idx,n_rmu=load_centroids(CENTS,tuple(int(v) for v in fft_grid)); cents_idx=np.asarray(cents_idx,np.int64)
sym_perm,_=compute_centroid_sym_perm(cents_idx,wfn.sym_matrices,wfn.translations,fft_grid,validate=False,extend_trs=True)
NB=24
loader=WfnLoader(WFN,backend='eager'); nb=min(NB,int(loader.nbands))
psi4=np.asarray(loader.load(bands=(0,nb),k='full_bz',sharding=None,bispinor=True))
gvecs_full=np.asarray(loader.gvecs(k='full_bz')); ngk_valid=np.asarray(loader.ngk_valid(k='full_bz'))
def sample4(ik):
    gv=gvecs_full[ik][:int(ngk_valid[ik])]; gi=(gv%np.array([nx,ny,nz]))
    out=np.zeros((nb,4,n_rmu),complex); box=np.zeros((nx,ny,nz),complex)
    for b in range(nb):
        for sp in range(4):
            box[:]=0.0; box[gi[:,0],gi[:,1],gi[:,2]]=psi4[ik,b,sp,:int(ngk_valid[ik])]
            r=np.fft.ifftn(box,norm='ortho'); out[b,sp,:]=r[cents_idx[:,0],cents_idx[:,1],cents_idx[:,2]]
    return out
def CCTi(psi_rmu,i):
    M=np.einsum('amu,ab,bnu->mnu',np.conj(psi_rmu).transpose(1,0,2),gam[i],psi_rmu.transpose(1,0,2))
    Mf=M.reshape(nb*nb,n_rmu); return Mf.conj().T@Mf
def solve_prod(C):           # production ridge-LU: zeta = (C+ridge)^{-1} C
    ridge=1e-12*np.abs(np.trace(C))/n_rmu
    return np.linalg.solve(C+ridge*np.eye(n_rmu),C)
def solve_pinv(C,rcond):     # honest SVD pinv with generous rcond
    return np.linalg.pinv(C,rcond=rcond)@C
def recon_S(psc,psp,solver,label):
    R=Rp[op]; alpha=sym_perm[op]
    # reconstructed metric per channel: M_recon^{ii} = zeta C zeta^dag (centroid-metric)
    def Mrec(ps,i):
        C=CCTi(ps,i); Z=C  # q=0 target = C
        zeta=solver(C); return zeta@C@zeta.conj().T
    Sc=sum(np.linalg.norm(Mrec(psc,i))**2 for i in (1,2))
    Sp=sum(np.linalg.norm(Mrec(psp,i))**2 for i in (1,2))
    print(f"    [{label}] in-plane gauge-inv S: child={Sc:.5e} parent={Sp:.5e} rel-diff={abs(Sc-Sp)/Sp:.2e}")
for op in (1,2):
    cand=np.where(sym_idx_k==op)[0]; child=None
    for cc in cand:
        if np.linalg.norm(wfn.kpoints[irr_idx_k[cc]])>1e-6: child=int(cc);break
    if child is None: child=int(cand[0])
    kbar=irr_idx_k[child]; pc=np.where((irr_idx_k==kbar)&(sym_idx_k==0))[0]
    parent=int(pc[0]) if len(pc) else int(np.where(irr_idx_k==kbar)[0][0])
    print("\n"+"="*90); print(f"OP {op}: child={child} parent={parent}")
    psc=sample4(child); psp=sample4(parent)
    print("  raw CCT (no solve) in-plane S rel-diff (covariant baseline):")
    Sc=sum(np.linalg.norm(CCTi(psc,i))**2 for i in (1,2)); Sp=sum(np.linalg.norm(CCTi(psp,i))**2 for i in (1,2))
    print(f"    child={Sc:.5e} parent={Sp:.5e} rel-diff={abs(Sc-Sp)/Sp:.2e}")
    recon_S(psc,psp,solve_prod,"PROD ridge-LU 1e-12")
    recon_S(psc,psp,lambda C:solve_pinv(C,1e-6),"pinv rcond=1e-6")
    recon_S(psc,psp,lambda C:solve_pinv(C,1e-3),"pinv rcond=1e-3")
