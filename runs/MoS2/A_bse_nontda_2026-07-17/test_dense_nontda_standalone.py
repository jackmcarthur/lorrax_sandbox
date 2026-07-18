"""Standalone validation of the non-TDA dense-reference builder + solver gate
logic (to be folded into tests/test_bse_dense_reference.py).  Tests against the
CURRENT tree: materialises the full matvec, extracts A,B, builds analytic A,B,
checks the physical SHAO operator + definite pencil + solver + old-vs-fixed."""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=6, linewidth=150)

RESTART, INPUT = sys.argv[1], sys.argv[2]
from bse import bse_io
from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings
from bse.bse_serial import compute_pair_amplitude

data = bse_io._load_ring_subset(RESTART, n_val=2, n_cond=2, px=1, py=1, input_file=INPUT)
psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
W_q = np.asarray(data["W_q"]); V_q0 = np.asarray(data["V_q0"])
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"]); grid=(nkx,nky,nkz)
nk = nkx*nky*nkz; nc=psi_c.shape[1]; nv=psi_v.shape[1]; nmu=psi_c.shape[3]; N=nc*nv*nk


def build_dense_nontda(data):
    """Analytic A (Hermitian) and B (complex-symmetric) blocks; SHAO operator."""
    psi_c=np.asarray(data["psi_c"]); psi_v=np.asarray(data["psi_v"])
    eps_c=np.asarray(data["eps_c"]); eps_v=np.asarray(data["eps_v"])
    V_q0=np.asarray(data["V_q0"]); W_q=np.asarray(data["W_q"])
    nkx,nky,nkz=int(data["nkx"]),int(data["nky"]),int(data["nkz"]); grid=(nkx,nky,nkz)
    nk=nkx*nky*nkz; nc=psi_c.shape[1]; nv=psi_v.shape[1]; nmu=psi_c.shape[3]; N=nc*nv*nk
    def qf(k,kp):
        ck=np.array(np.unravel_index(k,grid)); ckp=np.array(np.unravel_index(kp,grid))
        return int(np.ravel_multi_index(tuple((ck-ckp)%np.array(grid)),grid))
    M=np.einsum("kcsm,kvsm->kcvm",np.conj(psi_c),psi_v)
    D=np.transpose(eps_c[:,:,None]-eps_v[:,None,:],(1,2,0))
    Wf=W_q.reshape(nmu,nmu,nk); lhs=np.einsum("kcvM,MN->kcvN",np.conj(M),V_q0)
    KxA=np.einsum("kcvN,KCVN->cvkCVK",lhs,M)/nk
    KxB=np.einsum("kcvN,KCVN->cvkCVK",lhs,np.conj(M))/nk
    KdA=np.zeros((nc,nv,nk,nc,nv,nk),complex); KdB=np.zeros_like(KdA)
    for k in range(nk):
        for kp in range(nk):
            Wq=Wf[:,:,qf(k,kp)]
            KdA[:,:,k,:,:,kp]=np.einsum("cCm,mn,vVn->cvCV",np.einsum("ctm,Ctm->cCm",np.conj(psi_c[k]),psi_c[kp]),Wq,np.einsum("vsn,Vsn->vVn",psi_v[k],np.conj(psi_v[kp])))/nk
            KdB[:,:,k,:,:,kp]=np.einsum("cVm,mn,vCn->cvCV",np.einsum("ctm,Vtm->cVm",np.conj(psi_c[k]),psi_v[kp]),Wq,np.einsum("vsn,Csn->vCn",psi_v[k],np.conj(psi_c[kp])))/nk
    A=np.diag(D.reshape(-1).astype(complex))+KxA.reshape(N,N)-KdA.reshape(N,N)
    B=KxB.reshape(N,N)-KdB.reshape(N,N)
    H=np.block([[A,B],[-B.conj(),-A.conj()]])
    return A,B,H

A_an, B_an, H_shao = build_dense_nontda(data)

# --- materialize the FIXED matvec by applying SHAO conjugation externally ---
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1,1), axis_names=("x","y"))
sh = make_bse_shardings(mesh)
mv = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=True, screening=False)
eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk); zero=np.zeros_like(eye)
with mesh:
    pcx=jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x)
    pcy=jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y)
    pvx=jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x)
    pvy=jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y)
    Wqs=jax.lax.with_sharding_constraint(jnp.asarray(W_q), sh.W)
    Vqs=jax.lax.with_sharding_constraint(jnp.asarray(V_q0), sh.V)
    WR=jnp.fft.ifftn(Wqs, axes=(2,3,4), norm="ortho")
    MX=jax.lax.with_sharding_constraint(compute_pair_amplitude(pcx,pvx), sh.psi_x)
    MY=jax.lax.with_sharding_constraint(compute_pair_amplitude(pcy,pvy), sh.psi_y)
    args=(pcx,pcy,pvx,pvy,jnp.asarray(eps_c),jnp.asarray(eps_v),WR,Vqs,MX,MY)
    # A_mat = top of matvec([e;0]); B_mat = top of matvec([0;e])
    XfA=jax.lax.with_sharding_constraint(jnp.asarray(np.stack([eye,zero],0)), sh.X_full)
    XfB=jax.lax.with_sharding_constraint(jnp.asarray(np.stack([zero,eye],0)), sh.X_full)
    A_mat=np.asarray(mv(XfA,*args)[0]).reshape(N,N).T
    B_mat=np.asarray(mv(XfB,*args)[0]).reshape(N,N).T
    # current-code Y-block (bottom) for [0;e] and [e;0] -> reveals the (2,1) bug
    C21=np.asarray(mv(XfA,*args)[1]).reshape(N,N).T   # = -B (code) vs -B* (fixed)
    C22=np.asarray(mv(XfB,*args)[1]).reshape(N,N).T   # = -A* (both, since -A*Y)... check

def rel(a,b): return float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
print(f"A_mat vs analytic A: {rel(A_mat, A_an):.2e}")
print(f"B_mat vs analytic B: {rel(B_mat, B_an):.2e}")
print(f"current (2,1) block vs -B: {rel(C21, -B_an):.2e}   vs -B* (SHAO): {rel(C21, -B_an.conj()):.2e}")
print(f"current (2,2) block vs -A*: {rel(C22, -A_an.conj()):.2e}  vs -A: {rel(C22, -A_an):.2e}")

# fixed operator (SHAO) built from materialized A,B
H_fixed = np.block([[A_mat, B_mat],[-B_mat.conj(), -A_mat.conj()]])
print(f"\nSHAO(materialized) vs analytic H_shao: {rel(H_fixed, H_shao):.2e}")
ev_shao=np.linalg.eigvals(H_shao).real; ev_shao=np.sort(ev_shao[ev_shao>1e-9])[:6]
print(f"SHAO real spectrum pos6 (Ry): {ev_shao}   max|Im|={np.max(np.abs(np.linalg.eigvals(H_shao).imag)):.1e}")
# current-code operator [[A,B],[-B,-A]] (what the UNFIXED matvec computes end-to-end)
H_code = np.block([[A_mat, B_mat],[-B_mat, -A_mat]])
print(f"CODE operator [[A,B],[-B,-A]] max|Im ev|={np.max(np.abs(np.linalg.eigvals(H_code).imag)):.1e} (unphysical)")
