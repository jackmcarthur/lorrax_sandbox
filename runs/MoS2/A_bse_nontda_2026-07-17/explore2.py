"""Nail the correct non-TDA operator + structure-preserving reduction for the
COMPLEX BSE (A Hermitian, B complex-symmetric), on the gnppm 2v2c fixture.

Loads A, B blocks (materialized) from a small numpy rebuild to avoid the 50 GB
matvec materialization; instead build A, B analytically (they are tiny N=36).
"""
import sys
import numpy as np
np.set_printoptions(precision=6, suppress=False, linewidth=150)
import jax
jax.config.update("jax_enable_x64", True)

RESTART, INPUT = sys.argv[1], sys.argv[2]
from bse import bse_io
data = bse_io._load_ring_subset(RESTART, n_val=2, n_cond=2, px=1, py=1, input_file=INPUT)
psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
W_q = np.asarray(data["W_q"]); V_q0 = np.asarray(data["V_q0"])
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
grid = (nkx, nky, nkz); nk = nkx * nky * nkz
nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]
N = nc * nv * nk

def qflat(k, kp):
    ck = np.array(np.unravel_index(k, grid)); ckp = np.array(np.unravel_index(kp, grid))
    return int(np.ravel_multi_index(tuple((ck - ckp) % np.array(grid)), grid))

M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)   # (k,c,v,mu)
D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
Wflat = W_q.reshape(nmu, nmu, nk)

# ---- A block: diag(D) + Kx_A - Kd_A  (standard pairing) ----
lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V_q0)
Kx_A = np.einsum("kcvN,KCVN->cvkCVK", lhs, M) / nk
Kd_A = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
for k in range(nk):
    for kp in range(nk):
        Wq = Wflat[:, :, qflat(k, kp)]
        Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
        Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
        Kd_A[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / nk
A = np.diag(D.reshape(-1).astype(np.complex128)) + Kx_A.reshape(N, N) - Kd_A.reshape(N, N)

# ---- B block: Kx_B - Kd_B  (coupling / conjugated ket pairing) ----
# Bx = (1/Nk) conj(M_I) V conj(M_J)  ;  ket leg conjugated (vs Kx_A's M_J)
Kx_B = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)) / nk
# Bd: [conj(psi_c(k)) psi_v'(k')]_mu W [psi_v(k) conj(psi_c'(k'))]_nu  (c'<->v' swap)
Kd_B = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
for k in range(nk):
    for kp in range(nk):
        Wq = Wflat[:, :, qflat(k, kp)]
        Pc_B = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])   # (c, v', mu)
        Pv_B = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))   # (v, c', nu)
        Kd_B[:, :, k, :, :, kp] = np.einsum("cVm,mn,vCn->cvCV", Pc_B, Wq, Pv_B) / nk
B = Kx_B.reshape(N, N) - Kd_B.reshape(N, N)

def rel(a, b): return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
print(f"analytic: N={N}")
print(f"A Herm err {rel(A, A.conj().T):.2e}  A sym err {rel(A, A.T):.2e}")
print(f"B Herm err {rel(B, B.conj().T):.2e}  B sym err {rel(B, B.T):.2e}")

def spec(O, tag):
    ev = np.linalg.eigvals(O); ev = ev[np.argsort(ev.real)]
    pos = np.sort(ev.real[ev.real > 1e-9])[:6]
    print(f"{tag:32s} max|Im|={np.max(np.abs(ev.imag)):.2e}  pos6={pos}")
    return pos

print("\n== candidate operators ==")
p_code = spec(np.block([[A, B], [-B, -A]]), "code [[A,B],[-B,-A]]")
p_task = spec(np.block([[A, B], [-B.conj().T, -A.conj().T]]), "task [[A,B],[-Bdag,-Adag]]")
p_shao = spec(np.block([[A, B], [-B.conj(), -A.conj()]]), "shao [[A,B],[-conjB,-conjA]]")

print("\n== structure-preserving reductions (which M gives sqrt->real spectrum?) ==")
def try_M(Mmat, tag):
    ev = np.linalg.eigvals(Mmat)
    om = np.sort(np.sqrt(ev.astype(complex)).real)
    om = np.sort(np.abs(om))[:6]
    print(f"{tag:40s} M max|Im(eig)|={np.max(np.abs(ev.imag)):.2e}  sqrt6={om}")
    return om
try_M((A - B) @ (A + B), "(A-B)(A+B)")
try_M((A - B.conj().T) @ (A + B), "(A-Bdag)(A+B)")
try_M((A - B) @ (A + B.conj().T), "(A-B)(A+Bdag)")

# ---- Shao real transformation test: Phi^H H_task Phi -> real [[Ah,Bh],[-Bh,-Ah]] ----
I = np.eye(N)
# candidate unitary (BSEPACK): Phi = 1/sqrt2 [[I, I],[i I, -i I]] etc. try a few
for name, Phi in [
    ("1/rt2[[I,I],[iI,-iI]]", (1/np.sqrt(2))*np.block([[I, I],[1j*I, -1j*I]])),
    ("1/rt2[[I,iI],[I,-iI]]", (1/np.sqrt(2))*np.block([[I, 1j*I],[I, -1j*I]])),
    ("1/rt2[[I,I],[-iI,iI]]", (1/np.sqrt(2))*np.block([[I, I],[-1j*I, 1j*I]])),
]:
    Ht = np.block([[A, B], [-B.conj().T, -A.conj().T]])   # task operator (real spectrum)
    Hr = Phi.conj().T @ Ht @ Phi
    imag = np.max(np.abs(Hr.imag))
    print(f"\nPhi={name}: max|Im(Phi^H H Phi)|={imag:.2e}")
    if imag < 1e-8:
        Ah = Hr[:N, :N].real; Bh = Hr[:N, N:].real
        s21 = Hr[N:, :N].real; s22 = Hr[N:, N:].real
        print(f"  Ah sym {rel(Ah, Ah.T):.1e}  Bh sym {rel(Bh, Bh.T):.1e}  "
              f"(2,1)vs-Bh {rel(s21, -Bh):.1e}  (2,2)vs-Ah {rel(s22, -Ah):.1e}")
        AmB = Ah - Bh; ApB = Ah + Bh
        wm = np.linalg.eigvalsh(AmB); wp = np.linalg.eigvalsh(ApB)
        print(f"  (Ah-Bh) eig>0? {wm.min()>0} [{wm.min():.2e},{wm.max():.2e}]  "
              f"(Ah+Bh) eig>0? {wp.min()>0} [{wp.min():.2e},{wp.max():.2e}]")
        Mr = AmB @ ApB
        om = np.sort(np.sqrt(np.linalg.eigvalsh(
            np.linalg.cholesky(AmB) is not None and Mr or Mr).astype(complex) if False else np.linalg.eigvals(Mr).astype(complex)).real)
        om = np.sort(np.abs(om))[:6]
        print(f"  sqrt(eig((Ah-Bh)(Ah+Bh)))6 = {om}   vs task pos6 = {p_task}")
