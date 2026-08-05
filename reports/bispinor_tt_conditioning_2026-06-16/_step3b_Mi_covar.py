"""STEP 3b: The band-PAIRWISE transverse current matrix element is the object
the CCT squares.  Test it directly.

The CCT (transverse) at one centroid mu is built from, per band pair (m,n):
    M^i_{mn}(mu) = sum_{a,b} psi*_{m,a}(mu) gamma^i_{ab} psi_{n,b}(mu)   (4-spinor)
and C^i(mu,col) ~ sum_{mn} M^i_{mn}(mu) conj(M^i_{mn}(col)).

The CORRECT covariance is: for child k = S.parent, with psi(child) = Lam psi(parent)
(up to per-band U(1) phase from the unfold), and using the EXACT gamma^i the code uses,
    M^i_{mn}(child, mu) = e^{i(theta_m - theta_n)} * sum_j R[j,i] M^j_{mn}(parent, alpha(mu)).
The per-band phases theta cancel in the CCT bilinear |M|^2 sum.  So the gauge-invariant
covariant test is on the Gram of M over band pairs:
    g^{ij}(mu) = sum_{mn} M^i_{mn}(mu) conj(M^j_{mn}(mu))    (3x3 per centroid)
must satisfy  g_child(mu) = R^T g_parent(alpha(mu)) R.

This is EXACTLY the centroid-traced Gram the conditioning report measured as
lab-frame.  We compute it from the loaded psi + the code's gamma^i and decide.
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
sym=SymMaps(wfn); U=np.asarray(sym.U_spinor); Rp=np.asarray(sym.R_proper)
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
    # M^i_{mn}(mu) = sum_ab psi*_{m,a}(mu) gamma^i_{ab} psi_{n,b}(mu)
    # psi_rmu: (nb,4,n_rmu)
    g=gam[i]
    return np.einsum('amu,ab,bnu->mnu',np.conj(psi_rmu).transpose(1,0,2),g,psi_rmu.transpose(1,0,2))
def gram(psi_rmu):
    # g^{ij}(mu) = sum_{mn} M^i conj(M^j)  -> (3,3,n_rmu)
    M=np.stack([Mi(psi_rmu,i) for i in (1,2,3)],0)   # (3,nb,nb,n_rmu)
    return np.einsum('imnu,jmnu->iju',M,np.conj(M))   # (3,3,n_rmu)
for op in (1,2):
    cand=np.where(sym_idx_k==op)[0]; child=None
    for cc in cand:
        if np.linalg.norm(wfn.kpoints[irr_idx_k[cc]])>1e-6: child=int(cc);break
    if child is None: child=int(cand[0])
    kbar=irr_idx_k[child]
    pc=np.where((irr_idx_k==kbar)&(sym_idx_k==0))[0]
    parent=int(pc[0]) if len(pc) else int(np.where(irr_idx_k==kbar)[0][0])
    print("\n"+"="*100); print(f"OP {op}: child={child} parent={parent} IBZ={kbar}  R_proper=\n{Rp[op]}")
    gc=gram(sample4(child)); gp=gram(sample4(parent)); alpha=sym_perm[op]; R=Rp[op]
    # per-centroid covariant law: g_child(mu) = R^T g_parent(alpha) R
    dens=np.abs(gp[2,2]).real; pick=np.argsort(-dens)[:5]
    for mu in pick[:3]:
        a=int(alpha[mu]); gcm=gc[:,:,mu]; gpm=gp[:,:,a]
        pred=R.T@gpm@R
        print(f"\n  centroid mu={mu} alpha={a}:")
        print(f"    g_child(mu) real=\n{gcm.real}")
        print(f"    R^T g_parent(alpha) R real=\n{pred.real}  (covariant prediction)")
        print(f"    g_parent(alpha) real=\n{gpm.real}  (lab-frame)")
        print(f"    |g_child - R^T g_par R|={np.abs(gcm-pred).max():.3e}   |g_child - g_par|={np.abs(gcm-gpm).max():.3e}")
    # aggregate over all centroids, per matrix entry
    gpa=gp[:,:,alpha]                                  # (3,3,n_rmu) gather parent at alpha
    pred_all=np.einsum('ai,abu,bj->iju',R,gpa,R)       # R^T g R  (sum_ab R[a,i] g_ab R[b,j])
    num_cov=np.linalg.norm((gc-pred_all).real)
    num_lab=np.linalg.norm((gc-gpa).real)
    den=np.linalg.norm(gc.real)+1e-30
    print(f"\n  AGGREGATE: rel|g_child - R^T g_par R| = {num_cov/den:.3e}   rel|g_child - g_par(perm)| = {num_lab/den:.3e}")
    print(f"             -> {'COVARIANT (gamma+current fine)' if num_cov/den<1e-4 else ('LAB-FRAME (sigma pinned to lab)' if num_lab/den<1e-4 else 'NEITHER')}")
    # also report the off-diagonal in-plane g_xy specifically (the broken one in the report)
    print(f"    g_xy summed over centroids: child={gc[0,1].sum().real:+.3f}  R^TgR pred={pred_all[0,1].sum().real:+.3f}  parent={gpa[0,1].sum().real:+.3f}")
