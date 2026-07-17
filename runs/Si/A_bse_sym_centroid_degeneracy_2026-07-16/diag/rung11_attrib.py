import sys, os, json
import numpy as np, h5py
LROOT="/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
RUN="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"
sys.path.insert(0, os.path.join(LROOT,"src")); sys.path.insert(0, os.path.join(LROOT,"tests"))
import jax; jax.config.update("jax_enable_x64", True)
from file_io.wfn_loader import WfnLoader
from common import symmetry_maps
from bse import bse_io
RY=13.6056980659; WFN=os.path.join(LROOT,"tests/regression/si_cohsex_debug/WFN.h5"); GRID=(4,4,4); NK=64
wfn=WfnLoader(WFN,backend="eager"); sym=symmetry_maps.SymMaps(wfn)
irr=np.asarray(sym.irr_idx_k)
stars=[sorted(np.where(irr==u)[0].tolist()) for u in sorted(set(irr.tolist()))]
star_of=np.zeros(NK,int)
for si,st in enumerate(stars):
    for k in st: star_of[k]=si
inp=f"{RUN}/work_sym/cohsex_si_test.in"; restart=bse_io._find_restart_file(inp)
data=bse_io._load_ring_subset(restart,n_val=4,n_cond=4,px=1,py=1,input_file=inp)
psi_c=np.asarray(data["psi_c"]); psi_v=np.asarray(data["psi_v"])
eps_c=np.asarray(data["eps_c"]); eps_v=np.asarray(data["eps_v"])
V0=np.asarray(data["V_q0"]); Wf=np.asarray(data["W_q"]).reshape(792,792,NK)
nc=psi_c.shape[1]; nv=psi_v.shape[1]
M=np.einsum("kcsm,kvsm->kcvm",np.conj(psi_c),psi_v,optimize=True)
Dcvk=np.transpose(eps_c[:,:,None]-eps_v[:,None,:],(1,2,0))
lhs=np.einsum("kcvM,MN->kcvN",np.conj(M),V0,optimize=True)
Kx=np.einsum("kcvN,KCVN->cvkCVK",lhs,M,optimize=True)/NK
ck=np.array(np.unravel_index(np.arange(NK),GRID)).T
qidx=np.empty((NK,NK),int)
for k in range(NK): qidx[k]=np.ravel_multi_index(((ck[k][None]-ck)%np.array(GRID)).T,GRID)
Kd=np.zeros((nc,nv,NK,nc,nv,NK),complex)
for k in range(NK):
    Wq_k=np.transpose(Wf[:,:,qidx[k]],(2,0,1))
    Pc=np.einsum("csm,KCsm->KcCm",np.conj(psi_c[k]),psi_c,optimize=True)
    Pv=np.einsum("vsn,KVsn->KvVn",psi_v[k],np.conj(psi_v),optimize=True)
    tmp=np.einsum("KcCm,Kmn->KcCn",Pc,Wq_k,optimize=True)
    Kd[:,:,k,:,:,:]=np.einsum("KcCn,KvVn->cvCVK",tmp,Pv,optimize=True)/NK
N=nc*nv*NK
H6=(np.zeros((nc,nv,NK,nc,nv,NK),complex))
# H_{cvk,CVK} = D δ + Kx - Kd
Hfull=np.diag(Dcvk.reshape(-1).astype(complex))+Kx.reshape(N,N)-Kd.reshape(N,N)
Hfull=0.5*(Hfull+Hfull.conj().T)
ev,evec=np.linalg.eigh(Hfull); ev_eV=ev*RY
# lowest doublet = states 0,1 (they are the 518ueV manifold)
print(f"lowest 4 eV: {[round(x,6) for x in ev_eV[:4]]}  split01={((ev_eV[1]-ev_eV[0])*1e6):.2f} ueV")
A0=evec[:,0].reshape(nc,nv,NK); A1=evec[:,1].reshape(nc,nv,NK)
# H as (cv,k, cv,k)
Hk=Hfull.reshape(nc*nv,NK,nc*nv,NK)
a0=evec[:,0].reshape(nc*nv,NK); a1=evec[:,1].reshape(nc*nv,NK)
# per (k,k') contribution to <i|H|i>: c_i[k,k']=a_i[:,k]^H Hk[:,k,:,k'] a_i[:,k']
def contrib(a):
    return np.einsum("ak,akbl,bl->kl", np.conj(a), Hk, a, optimize=True)  # (k,k') real part meaningful after sum
c0=contrib(a0); c1=contrib(a1)
dsplit=(c1-c0).real*RY*1e6  # (k,k') contribution to (lam1-lam0) in ueV
tot=dsplit.sum()
print(f"reconstructed split from (k,k') sum: {tot:.2f} ueV (should match {((ev_eV[1]-ev_eV[0])*1e6):.2f})")
# attribute by star pair
nS=len(stars); starmat=np.zeros((nS,nS))
for k in range(NK):
    for l in range(NK):
        starmat[star_of[k],star_of[l]]+=dsplit[k,l]
# on-site (k=k') by star
onsite=np.zeros(nS)
for k in range(NK): onsite[star_of[k]]+=dsplit[k,k]
print("star sizes:",[len(s) for s in stars])
print("split contribution by star (on-site k=k', ueV):",[round(x,1) for x in onsite])
print(f"  Γ (star0, the CUT 6-fold multiplet) on-site contribution: {onsite[0]:.1f} ueV")
print(f"  total on-site: {onsite.sum():.1f} ueV ; total inter-k: {(tot-onsite.sum()):.1f} ueV")
# star-pair top contributors
flat=[(starmat[i,j],i,j) for i in range(nS) for j in range(nS)]
flat.sort(key=lambda z:-abs(z[0]))
print("top star-pair (Si,Sj) contributions to split (ueV):")
for val,i,j in flat[:6]:
    print(f"   star{i}(sz{len(stars[i])}) x star{j}(sz{len(stars[j])}): {val:.1f}")
