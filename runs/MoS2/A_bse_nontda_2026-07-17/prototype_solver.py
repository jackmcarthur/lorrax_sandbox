"""Prototype + validate the two structure-preserving eigensolver algorithms
before the JAX implementation:

  (1) COMPLEX physical BSE (A Herm, B complex-sym): the Hermitian-PD definite
      pencil  K z = omega Sigma z,  K=[[A,B],[B*,A*]],  Sigma=diag(I,-I).
      -> real omega, eigenvectors z=[X;Y] with X^H X - Y^H Y = +1.
  (2) REAL BSE (A,B real symmetric, A+-B PD): the clean product reduction
      M=(A-B)(A+B), (A+B)-metric Lanczos, omega=sqrt(mu), recover X,Y, X^2-Y^2=1.

Validates against dense eig of [[A,B],[-B*,-A*]] (physical) and [[A,B],[-B,-A]] (real).
"""
import sys
import numpy as np
np.set_printoptions(precision=6, suppress=False, linewidth=150)
import jax; jax.config.update("jax_enable_x64", True)
RESTART, INPUT = sys.argv[1], sys.argv[2]
from bse import bse_io
data = bse_io._load_ring_subset(RESTART, n_val=2, n_cond=2, px=1, py=1, input_file=INPUT)
psi_c=np.asarray(data["psi_c"]); psi_v=np.asarray(data["psi_v"])
eps_c=np.asarray(data["eps_c"]); eps_v=np.asarray(data["eps_v"])
W_q=np.asarray(data["W_q"]); V_q0=np.asarray(data["V_q0"])
nkx,nky,nkz=int(data["nkx"]),int(data["nky"]),int(data["nkz"]); grid=(nkx,nky,nkz)
nk=nkx*nky*nkz; nc=psi_c.shape[1]; nv=psi_v.shape[1]; nmu=psi_c.shape[3]; N=nc*nv*nk
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

# ---------- (1) complex definite-pencil solver ----------
print("=== (1) COMPLEX physical BSE: definite pencil K z = w Sigma z ===")
K=np.block([[A,B],[B.conj(),A.conj()]]); Sig=np.diag(np.concatenate([np.ones(N),-np.ones(N)]))
wK=np.linalg.eigvalsh(K); print(f"K Herm err {np.linalg.norm(K-K.conj().T)/np.linalg.norm(K):.1e}  K PD min eig {wK.min():.3e}  (A-B) not needed")
# K^{-1/2} Sigma K^{-1/2} Hermitian -> eigenvalues 1/omega (+/- pairs)
w,U=np.linalg.eigh(K); Kmh=U@np.diag(1/np.sqrt(w))@U.conj().T
Shat=Kmh@Sig@Kmh; Shat=0.5*(Shat+Shat.conj().T)
mu,Vv=np.linalg.eigh(Shat)          # mu = 1/omega (real, +/- pairs)
posmask=mu>1e-12
inv_om=mu[posmask]; om=np.sort(1.0/inv_om)   # ascending omega
order=np.argsort(1.0/inv_om)
# recover z for the smallest few omega (largest 1/omega): y=Shat eigvec -> z=K^{-1/2} y
idx_pos=np.where(posmask)[0]
# smallest omega <-> largest mu(=1/omega)
sel=idx_pos[np.argsort(-mu[idx_pos])][:6]
Href=np.block([[A,B],[-B.conj(),-A.conj()]]); ev_ref=np.sort(np.linalg.eigvals(Href).real[np.linalg.eigvals(Href).real>1e-9])[:6]
print("omega6 (pencil):", np.sort(1.0/mu[sel])[:6])
print("omega6 (dense) :", ev_ref)
print("MATCH:", np.allclose(np.sort(1.0/mu[sel])[:6], ev_ref, atol=1e-6))
# eigenvector recovery + normalization X^H X - Y^H Y
for j in sel[:3]:
    y=Vv[:,j]; z=Kmh@y            # K^{-1/2} y solves Shat y = mu y  => K^{-1} Sigma z = mu z
    om_j=1.0/mu[j]
    X=z[:N]; Y=z[N:]
    snorm=(X.conj()@X - Y.conj()@Y).real
    z=z/np.sqrt(abs(snorm)); X,Y=z[:N],z[N:]
    snorm2=(X.conj()@X - Y.conj()@Y).real
    res=np.linalg.norm(Href@z - om_j*z)/np.linalg.norm(z)
    print(f"  omega={om_j:.6f}  X^HX-Y^HY={snorm2:+.4f}  Href residual={res:.2e}")

# ---------- (2) real BSE product-Lanczos ----------
print("\n=== (2) synthetic REAL BSE: M=(A-B)(A+B) (A+B)-metric Lanczos ===")
rng=np.random.default_rng(3); n=40
G=rng.standard_normal((n,n)); Ar=G@G.T/ n + 3*np.eye(n)      # SPD sym
Gb=rng.standard_normal((n,n))*0.15; Br=0.5*(Gb+Gb.T)          # sym, small
ApB=Ar+Br; AmB=Ar-Br
assert np.linalg.eigvalsh(ApB).min()>0 and np.linalg.eigvalsh(AmB).min()>0
Hr=np.block([[Ar,Br],[-Br,-Ar]]); ev_r=np.sort(np.linalg.eigvals(Hr).real[np.linalg.eigvals(Hr).real>1e-9])[:6]
# (A+B)-metric block Lanczos on M=(A-B)(A+B); inner <x,y>_+ = x^T(A+B)y (real)
def metric_lanczos(applyM, applyMetric, n, n_eig, bs=4, m=12, seed=0):
    rng=np.random.default_rng(seed)
    Q0=rng.standard_normal((n,bs))
    # metric-orthonormalize: G=Q^T P Q, P=metric Q ; R via eig
    def morth(W):
        P=applyMetric(W); Gm=W.T@P; Gm=0.5*(Gm+Gm.T)
        lam,Z=np.linalg.eigh(Gm); keep=lam>1e-14*lam.max()
        inv=np.where(keep,1/np.sqrt(np.where(keep,lam,1)),0.0)
        Tr=Z@np.diag(inv); return W@Tr, np.diag(np.where(keep,np.sqrt(np.where(keep,lam,0)),0))@Z.T
    Q,R0=morth(Q0); Qs=[Q]; alphas=[]; betas=[]; Pprev=None; betaprev=np.zeros((bs,bs)); Qprev=np.zeros_like(Q)
    for j in range(m):
        Wc=applyM(Qs[j]); P=applyMetric(Qs[j])
        alpha=Qs[j].T@applyMetric(Wc); alpha=0.5*(alpha+alpha.T)  # <Q,MQ>_+ ... use metric
        # actually <Q_j, W>_+ = Q_j^T (A+B) W
        alpha=Qs[j].T@applyMetric(Wc)
        Wc=Wc-Qs[j]@alpha-Qprev@betaprev.T
        for _ in range(2):
            for Qi in Qs:
                Wc=Wc-Qi@(Qi.T@applyMetric(Wc))
        Qn,beta=morth(Wc); alphas.append(alpha); betas.append(beta); Qprev=Qs[j]; betaprev=beta; Qs.append(Qn)
    mm=m; T=np.zeros((mm*bs,mm*bs))
    for j in range(mm):
        T[j*bs:(j+1)*bs,j*bs:(j+1)*bs]=alphas[j]
        if j+1<mm:
            T[(j+1)*bs:(j+2)*bs,j*bs:(j+1)*bs]=betas[j]
            T[j*bs:(j+1)*bs,(j+1)*bs:(j+2)*bs]=betas[j].T
    T=0.5*(T+T.T)
    muT,cT=np.linalg.eigh(T); idx=np.argsort(muT)[:n_eig]
    Qfull=np.concatenate(Qs[:mm],axis=1)
    U=Qfull@cT[:,idx]   # eigvecs of M (u=X+Y)
    return muT[idx], U
muT,Uu=metric_lanczos(lambda W:AmB@(ApB@W), lambda W:ApB@W, n, 6, bs=4, m=12)
om_l=np.sqrt(np.clip(muT,0,None))
print("omega6 (Lanczos):", np.sort(om_l)[:6])
print("omega6 (dense)  :", ev_r)
print("MATCH:", np.allclose(np.sort(om_l)[:6], ev_r, atol=1e-6))
# recover X,Y from u=X+Y: w=(1/omega)(A+B)u ; X=(u+w)/2, Y=(u-w)/2 ; normalize X^2-Y^2=1
for i in np.argsort(om_l)[:3]:
    u=Uu[:,i]; wv=(ApB@u)/om_l[i]
    # normalize so <u,u>_+ = omega  => X^2-Y^2 = 1
    scale=np.sqrt(om_l[i]/(u@(ApB@u))); u=u*scale; wv=wv*scale
    X=(u+wv)/2; Y=(u-wv)/2
    nrm=(X@X - Y@Y)
    res=np.linalg.norm(Hr@np.concatenate([X,Y]) - om_l[i]*np.concatenate([X,Y]))/np.linalg.norm(np.concatenate([X,Y]))
    print(f"  omega={om_l[i]:.6f}  X^2-Y^2={nrm:+.4f}  Hr residual={res:.2e}")
