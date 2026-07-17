import sys, os, json
import numpy as np, h5py
LROOT="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT,"src"))
import jax; jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from centroid.orbit_syms import compute_centroid_sym_perm
RY=13.6056980659; WFN=os.path.join(LROOT,"tests/regression/si_cohsex_debug/WFN.h5"); FFT=np.array([24,24,24])
wfn=WfnLoader(WFN,backend="eager"); sym=symmetry_maps.SymMaps(wfn); ntran=int(wfn.ntran)
tnp=np.asarray(wfn.translations[:ntran])/(2*np.pi); Uspin=np.asarray(sym.U_spinor[:ntran])
cfrac=np.loadtxt(f"{RUN}/work_sym/centroids_frac_792.txt"); ridx=np.rint(cfrac*FFT[None]).astype(np.int64)%FFT[None]
alpha,_=compute_centroid_sym_perm(ridx,wfn.sym_matrices[:ntran],wfn.translations[:ntran],FFT,validate=True)
with h5py.File(f"{RUN}/work_sym/tmp/isdf_tensors_792.h5","r") as f:
    psi=np.asarray(f["psi_full_y"][:]); enk=np.asarray(f["enk_full"][:])
    V0=np.asarray(f["V_qmunu"][0]); W0=np.asarray(f["W0_qmunu"][0])
symm_ops=[s for s in range(ntran) if np.linalg.norm(tnp[s])<1e-6]
def PsiG(bands): p=psi[0][bands]; return np.transpose(p,(1,2,0)).reshape(-1,len(bands))
def rot(bands,s):
    p=psi[0][bands]; pp=p[:,:,alpha[s]]; pr=np.einsum("ab,nbm->nam",Uspin[s],pp,optimize=True)
    return np.transpose(pr,(1,2,0)).reshape(-1,len(bands))
def Drep(bands,s):
    Psi=PsiG(bands); PsiR=rot(bands,s); G=Psi.conj().T@Psi
    D=np.linalg.solve(G,Psi.conj().T@PsiR); return D
def gamma_block(vb,cb):
    pv=psi[0][vb]; pc=psi[0][cb]; ev=enk[0][vb]; ec=enk[0][cb]; nv=len(vb); nc=len(cb)
    M=np.einsum("csm,vsm->cvm",np.conj(pc),pv,optimize=True)
    Kx=np.einsum("cvm,mn,CVn->cvCV",np.conj(M),V0,M,optimize=True)
    Pc=np.einsum("csm,Csm->cCm",np.conj(pc),pc,optimize=True); Pv=np.einsum("vsn,Vsn->vVn",pv,np.conj(pv),optimize=True)
    Kd=np.einsum("cCn,vVn->cvCV",np.einsum("cCm,mn->cCn",Pc,W0,optimize=True),Pv,optimize=True)
    D=np.zeros((nc,nv,nc,nv),complex)
    for c in range(nc):
        for v in range(nv): D[c,v,c,v]=ec[c]-ev[v]
    H=(D+Kx-Kd).reshape(nc*nv,nc*nv)
    return 0.5*(H+H.conj().T),D.reshape(nc*nv,nc*nv),Kx.reshape(nc*nv,nc*nv),Kd.reshape(nc*nv,nc*nv),nv,nc
def clusters(ev,gap=1e-3):
    g=[[0]]
    for i in range(1,len(ev)):
        (g.append([i]) if ev[i]-ev[i-1]>gap else g[-1].append(i))
    return g
for tag,(vb,cb) in {"4x4_CUT[4,8)x[8,12)":([4,5,6,7],[8,9,10,11]),
                    "6x6_CLOSED[2,8)x[8,14)":([2,3,4,5,6,7],[8,9,10,11,12,13])}.items():
    H,D,Kx,Kd,nv,nc=gamma_block(vb,cb)
    ev=np.sort(np.linalg.eigvalsh(H))*RY
    cl=clusters(ev)
    splits=[ (ev[c[-1]]-ev[c[0]])*1e6 for c in cl ]
    print(f"\n=== Γ on-site exciton block {tag} (dim {nv*nc}) ===")
    print(f"  eigenvalue multiplet sizes: {[len(c) for c in cl]}")
    print(f"  intra-multiplet splits (μeV): {[round(s,2) for s in splits]}")
    # commutator with U(R) built from closed subspaces
    if 'CLOSED' in tag or '4x4' in tag:
        cmx={"D":[], "Kx":[], "Kd":[], "H":[]}
        for s in symm_ops:
            Dc=Drep(cb,s); Dv=Drep(vb,s)
            U=np.einsum("Cc,Vv->CVcv",Dc,np.conj(Dv)).reshape(nc*nv,nc*nv)
            for nm,T in (("D",D),("Kx",Kx),("Kd",Kd),("H",H)):
                cmx[nm].append(np.linalg.norm(U@T-T@U)/(np.linalg.norm(T)+1e-30))
        for nm in ("D","Kx","Kd","H"):
            print(f"  ‖[U(R),{nm:>2}]‖/‖{nm}‖ max={np.max(cmx[nm]):.3e}")
