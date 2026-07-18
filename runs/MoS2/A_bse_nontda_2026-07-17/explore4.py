"""BSEPACK real transform: H_shao = [[A,B],[-B*,-A*]] -> real Hamiltonian
Hhat = [[Ah,Bh],[-Bh,-Ah]] with Ah,Bh REAL symmetric.  Then omega^2 = eig((Ah-Bh)(Ah+Bh))
with (Ah +- Bh) real symmetric PD -> clean matrix-free (A+B)-metric Lanczos.

Also derive Ah, Bh in terms of Re/Im of A,B so the matvec can produce (Ah +- Bh) U."""
import sys
import numpy as np
np.set_printoptions(precision=6, suppress=False, linewidth=150)
import jax; jax.config.update("jax_enable_x64", True)
RESTART, INPUT = sys.argv[1], sys.argv[2]
from bse import bse_io
data = bse_io._load_ring_subset(RESTART, n_val=2, n_cond=2, px=1, py=1, input_file=INPUT)
psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
W_q = np.asarray(data["W_q"]); V_q0 = np.asarray(data["V_q0"])
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"]); grid=(nkx,nky,nkz)
nk = nkx*nky*nkz; nc=psi_c.shape[1]; nv=psi_v.shape[1]; nmu=psi_c.shape[3]; N=nc*nv*nk
def qflat(k,kp):
    ck=np.array(np.unravel_index(k,grid)); ckp=np.array(np.unravel_index(kp,grid))
    return int(np.ravel_multi_index(tuple((ck-ckp)%np.array(grid)),grid))
M=np.einsum("kcsm,kvsm->kcvm",np.conj(psi_c),psi_v)
D=np.transpose(eps_c[:,:,None]-eps_v[:,None,:],(1,2,0))
Wf=W_q.reshape(nmu,nmu,nk); lhs=np.einsum("kcvM,MN->kcvN",np.conj(M),V_q0)
KxA=np.einsum("kcvN,KCVN->cvkCVK",lhs,M)/nk; KdA=np.zeros((nc,nv,nk,nc,nv,nk),complex); KdB=np.zeros_like(KdA)
for k in range(nk):
    for kp in range(nk):
        Wq=Wf[:,:,qflat(k,kp)]
        KdA[:,:,k,:,:,kp]=np.einsum("cCm,mn,vVn->cvCV",np.einsum("ctm,Ctm->cCm",np.conj(psi_c[k]),psi_c[kp]),Wq,np.einsum("vsn,Vsn->vVn",psi_v[k],np.conj(psi_v[kp])))/nk
        KdB[:,:,k,:,:,kp]=np.einsum("cVm,mn,vCn->cvCV",np.einsum("ctm,Vtm->cVm",np.conj(psi_c[k]),psi_v[kp]),Wq,np.einsum("vsn,Csn->vCn",psi_v[k],np.conj(psi_c[kp])))/nk
A=np.diag(D.reshape(-1).astype(complex))+KxA.reshape(N,N)-KdA.reshape(N,N)
B=np.einsum("kcvN,KCVN->cvkCVK",lhs,np.conj(M)).reshape(N,N)/nk - KdB.reshape(N,N)
H=np.block([[A,B],[-B.conj(),-A.conj()]])
pos=np.sort(np.linalg.eigvals(H).real[np.linalg.eigvals(H).real>1e-9])[:6]
def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))

I=np.eye(N)
Ws={
 "1/rt2[[I,iI],[I,-iI]]": (1/np.sqrt(2))*np.block([[I,1j*I],[I,-1j*I]]),
 "1/rt2[[I,I],[iI,-iI]]": (1/np.sqrt(2))*np.block([[I,I],[1j*I,-1j*I]]),
 "1/rt2[[iI,I],[-iI,I]]": (1/np.sqrt(2))*np.block([[1j*I,I],[-1j*I,I]]),
}
for nm,W in Ws.items():
    for tag,Hh in [("W^H H W", W.conj().T@H@W), ("W H W^H", W@H@W.conj().T)]:
        if np.max(np.abs(Hh.imag))<1e-8:
            Ah=Hh[:N,:N].real; Bh=Hh[:N,N:].real; s21=Hh[N:,:N].real; s22=Hh[N:,N:].real
            struct = (rel(s21,-Bh)<1e-8 and rel(s22,-Ah)<1e-8)
            AmB=Ah-Bh; ApB=Ah+Bh
            om=np.sort(np.abs(np.sqrt(np.linalg.eigvals(AmB@ApB).astype(complex)).real))[:6]
            print(f"{nm} {tag}: REAL, [[Ah,Bh],[-Bh,-Ah]]? {struct}  "
                  f"Ah sym {rel(Ah,Ah.T):.1e} Bh sym {rel(Bh,Bh.T):.1e}")
            print(f"   (Ah-Bh)PD {np.linalg.eigvalsh(AmB).min():.2e}  (Ah+Bh)PD {np.linalg.eigvalsh(ApB).min():.2e}")
            print(f"   sqrt(eig((Ah-Bh)(Ah+Bh)))6 = {om}")
            print(f"   target pos6                = {pos}   MATCH={np.allclose(om,pos,atol=1e-6)}")
            # relation to Re/Im parts
            AR,AI,BR,BI=A.real,A.imag,B.real,B.imag
            print(f"   Ah == AR+BR? {rel(Ah,AR+BR):.1e}  Ah==AR-BR? {rel(Ah,AR-BR):.1e} "
                  f" Ah==AR? {rel(Ah,AR):.1e}")
            print(f"   Bh == BR-AI? ... check combos:")
            for cn,cm in [("AR",AR),("AR+BR",AR+BR),("AR-BR",AR-BR),("BR",BR),("BR+AI",BR+AI),
                          ("BR-AI",BR-AI),("-BI",-BI),("BI",BI),("AR+BI",AR+BI),("AR-BI",AR-BI)]:
                if rel(Ah,cm)<1e-8: print(f"       Ah = {cn}")
                if rel(Bh,cm)<1e-8: print(f"       Bh = {cn}")
