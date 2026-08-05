"""FINAL: is the in-plane non-covariance a sigma_y SIGN bug, or a RANK-DEFICIENT
indefinite-CCT reconstruction error from the fixed-ridge LU solve?

Build the transverse CCT  C^i_q[mu,nu] = sum_k M^i_k(mu) conj(M^i_{k-q}(nu))-style
metric honestly from loaded psi (q=0 special case: C^i[mu,nu]=sum_n
(psi^dag gamma^i psi)_n(mu) conj(...)(nu) summed over band pairs) and inspect:
  1. Is C^i_q covariant gauge-invariantly across an orbit? (||C^i||_F per channel)
  2. The singular spectrum of the in-plane vs z CCT: rank deficiency + conditioning.

If C^i is covariant (||C^i||_F orbit-constant) but the STORED tile S(q) is not,
the bug is the SOLVE (rank-deficient indefinite reconstruction), not the gamma/U.
If ||C^i||_F is itself orbit-non-constant, the bug is upstream in the CCT build.
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
def Mi(psi_rmu,i):
    # current matrix element per band pair at each centroid:
    # M^i_{mn}(mu)=sum_ab psi*_{m,a}(mu) gamma^i_{ab} psi_{n,b}(mu)
    return np.einsum('amu,ab,bnu->mnu',np.conj(psi_rmu).transpose(1,0,2),gam[i],psi_rmu.transpose(1,0,2))
def CCTi(psi_rmu,i):
    # q=0 transverse CCT: C^i[mu,nu]=sum_{mn} M^i_{mn}(mu) conj(M^i_{mn}(nu))
    M=Mi(psi_rmu,i)                                  # (nb,nb,n_rmu)
    Mf=M.reshape(nb*nb,n_rmu)
    return Mf.conj().T@Mf                            # (n_rmu,n_rmu) Hermitian PSD-ish
for op in (1,2):
    cand=np.where(sym_idx_k==op)[0]; child=None
    for cc in cand:
        if np.linalg.norm(wfn.kpoints[irr_idx_k[cc]])>1e-6: child=int(cc);break
    if child is None: child=int(cand[0])
    kbar=irr_idx_k[child]; pc=np.where((irr_idx_k==kbar)&(sym_idx_k==0))[0]
    parent=int(pc[0]) if len(pc) else int(np.where(irr_idx_k==kbar)[0][0])
    print("\n"+"="*90); print(f"OP {op}: child={child} parent={parent} IBZ={kbar}")
    psc=sample4(child); psp=sample4(parent); R=Rp[op]; alpha=sym_perm[op]
    # ||C^i||_F per channel, child vs parent (gauge-INVARIANT: Frobenius norm of CCT)
    for i in (1,2,3):
        Cc=CCTi(psc,i); Cp=CCTi(psp,i)
        fc=np.linalg.norm(Cc); fp=np.linalg.norm(Cp)
        # rank / conditioning of the CCT (singular values)
        svc=np.linalg.svd(Cc,compute_uv=False); svp=np.linalg.svd(Cp,compute_uv=False)
        rc=int((svc>1e-10*svc[0]).sum()); rp=int((svp>1e-10*svp[0]).sum())
        print(f"  i={i}: ||C^i||_F child={fc:.4e} parent={fp:.4e} rel-diff={abs(fc-fp)/fp:.2e} | "
              f"rank(>1e-10) child={rc} parent={rp} of {n_rmu} | cond child={svc[0]/svc[max(rc-1,0)]:.1e}")
    # The KEY covariance: sum_i ||C^i||_F^2 is channel-rotation-invariant (R orthogonal)
    # AND gauge-invariant -> must be orbit-constant for a correct CCT.
    Sc=sum(np.linalg.norm(CCTi(psc,i))**2 for i in (1,2,3))
    Sp=sum(np.linalg.norm(CCTi(psp,i))**2 for i in (1,2,3))
    Sin_c=sum(np.linalg.norm(CCTi(psc,i))**2 for i in (1,2))
    Sin_p=sum(np.linalg.norm(CCTi(psp,i))**2 for i in (1,2))
    print(f"  GAUGE-INV CCT scalar sum_i||C^i||_F^2: child={Sc:.5e} parent={Sp:.5e} rel-diff={abs(Sc-Sp)/Sp:.2e}")
    print(f"  in-plane only (i=1,2):                 child={Sin_c:.5e} parent={Sin_p:.5e} rel-diff={abs(Sin_c-Sin_p)/Sin_p:.2e}")
    print(f"  -> CCT covariant if rel-diff ~ 0 (machine). If so, the STORED-tile non-covariance is in the SOLVE.")
