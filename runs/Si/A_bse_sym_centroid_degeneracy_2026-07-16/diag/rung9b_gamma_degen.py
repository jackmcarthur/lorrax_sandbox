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
print("Γ energies (eV) bands 0..15:")
eg=enk[0]*RY
for b in range(16): print(f"  band {b:2d}: {eg[b]:.5f}")
# degeneracy groups at Γ (1 meV)
groups=[]; cur=[0]
for b in range(1,20):
    if eg[b]-eg[b-1]>1e-3: groups.append(cur); cur=[b]
    else: cur.append(b)
groups.append(cur)
print("Γ degeneracy groups (bands):", groups[:8])
symm_ops=[s for s in range(ntran) if np.linalg.norm(tnp[s])<1e-6]
def PsiG(bands):
    p=psi[0][bands]; return np.transpose(p,(1,2,0)).reshape(-1,len(bands))
def rot(bands,s):
    p=psi[0][bands]; pp=p[:,:,alpha[s]]; pr=np.einsum("ab,nbm->nam",Uspin[s],pp,optimize=True)
    return np.transpose(pr,(1,2,0)).reshape(-1,len(bands))
def resid(bands):
    w=0; ws=None
    for s in symm_ops:
        Psi=PsiG(bands); PsiR=rot(bands,s)
        G=Psi.conj().T@Psi; D=np.linalg.solve(G,Psi.conj().T@PsiR)
        r=np.linalg.norm(PsiR-Psi@D)/np.linalg.norm(PsiR)
        if r>w: w=r; ws=s
    return w,ws
for bs in ([0,1],[2,3,4,5,6,7],[4,5,6,7],[0,1,2,3,4,5,6,7],[8,9,10,11,12,13],[8,9,10,11]):
    w,ws=resid(bs); print(f"  Γ-rot residual bands {bs}: worst={w:.3e} (op {ws})")
