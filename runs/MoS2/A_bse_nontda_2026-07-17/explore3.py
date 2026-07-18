"""Find the structure-preserving reduction for the physical SHAO operator
H = [[A,B],[-conj B, -conj A]] (A Hermitian, B complex-symmetric)."""
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
nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]; N = nc * nv * nk

def qflat(k, kp):
    ck = np.array(np.unravel_index(k, grid)); ckp = np.array(np.unravel_index(kp, grid))
    return int(np.ravel_multi_index(tuple((ck - ckp) % np.array(grid)), grid))
M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)
D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
Wflat = W_q.reshape(nmu, nmu, nk)
lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V_q0)
Kx_A = np.einsum("kcvN,KCVN->cvkCVK", lhs, M) / nk
Kd_A = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
Kd_B = np.zeros_like(Kd_A)
for k in range(nk):
    for kp in range(nk):
        Wq = Wflat[:, :, qflat(k, kp)]
        Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
        Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
        Kd_A[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / nk
        Pc_B = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])
        Pv_B = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))
        Kd_B[:, :, k, :, :, kp] = np.einsum("cVm,mn,vCn->cvCV", Pc_B, Wq, Pv_B) / nk
A = np.diag(D.reshape(-1).astype(np.complex128)) + Kx_A.reshape(N, N) - Kd_A.reshape(N, N)
Kx_B = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)) / nk
B = Kx_B.reshape(N, N) - Kd_B.reshape(N, N)

H = np.block([[A, B], [-B.conj(), -A.conj()]])  # SHAO
ev = np.linalg.eigvals(H); evs = np.sort(ev.real)
pos = np.sort(ev.real[ev.real > 1e-9])[:8]
neg = np.sort(ev.real[ev.real < -1e-9])[::-1][:8]
print(f"SHAO max|Im ev|={np.max(np.abs(ev.imag)):.2e}")
print(f"pos8 {pos}")
print(f"-neg8 {-neg}  (== pos? {np.allclose(pos, -neg, atol=1e-6)})   <- +/- pairing")

# real transform T = 1/sqrt2 [[I,I],[I,-I]] then phase diag(I, iI) or (iI, I)
I = np.eye(N)
T = (1/np.sqrt(2)) * np.block([[I, I], [I, -I]])
G = T @ H @ T   # [[iP, S1],[S2, iQ]]
print(f"\nT H T real? max|Im|={np.max(np.abs(G.imag)):.2e} (expect imag diag blocks)")
# try to real-ify with phase on the imaginary diagonal blocks
for nm, Ph in [("diag(I,iI)", np.block([[I, 0*I],[0*I, 1j*I]])),
               ("diag(iI,I)", np.block([[1j*I, 0*I],[0*I, I]]))]:
    G2 = np.linalg.inv(Ph) @ G @ Ph
    print(f"  Ph={nm}: max|Im(Ph^-1 (THT) Ph)|={np.max(np.abs(G2.imag)):.2e}")

# The known real 2N problem: excitation energies omega^2 are eigenvalues of
#   the real symmetric-definite product.  Use S1=A_R-B_R, S2=A_R+B_R (symmetric),
#   with the imaginary parts folded in.  Build the guaranteed-correct real
#   Hamiltonian directly from the 2Nx2N complex->real via the standard mapping and
#   read omega from eig.  Here just test the CLEAN candidate that reuses A+-B:
AR, AI = A.real, A.imag
BR, BI = B.real, B.imag
S_plus = AR + BR      # symmetric
S_minus = AR - BR     # symmetric
# Bai-Li / Shao real form:  omega^2 = eig( (S_minus + i(AI - BI)) ... ) -- test the
# full real 2N Hamiltonian Hr = [[ (AI+BI), (AR-BR) ], [ -(AR+BR), (AI-BI) ]]*? scan:
def sqrt_pos6(Mm):
    e = np.linalg.eigvals(Mm)
    return np.sort(np.abs(np.sqrt(e.astype(complex)).real))[:6], float(np.max(np.abs(e.imag)))

print("\n== reduction candidates (want sqrt6 == pos8[:6]) ==")
print("target pos6:", pos[:6])
cands = {
    "(A-B)(A+B)": (A-B)@(A+B),
    "(A-B*)(A+B*) [B*=conjB]": (A-B.conj())@(A+B.conj()),
    "S_minus @ S_plus (real parts)": S_minus @ S_plus,
    "(A-B)(A+B) conj-sym": None,
}
for nm, Mm in cands.items():
    if Mm is None: continue
    s6, im = sqrt_pos6(Mm)
    print(f"  {nm:32s} sqrt6={s6}  M|Im|={im:.1e}")

# Definite pencil: H = Sigma K, K=[[A,B],[conjB,conjA]] Hermitian.  omega = eig(Sigma K).
Sig = np.diag(np.concatenate([np.ones(N), -np.ones(N)]))
K = np.block([[A, B], [B.conj(), A.conj()]])
print(f"\nK Hermitian? {np.linalg.norm(K-K.conj().T)/np.linalg.norm(K):.2e}  "
      f"K PD? min eig={np.linalg.eigvalsh(K).min():.3e}")
# omega = eig(Sigma K); K^{1/2} route: 1/omega = eig( K^{-1/2} Sigma K^{-1/2} )
w, U = np.linalg.eigh(K)
Kmh = U @ np.diag(1/np.sqrt(w)) @ U.conj().T   # K^{-1/2}
Shat = Kmh @ Sig @ Kmh
es = np.linalg.eigvalsh(Shat)   # 1/omega, real (Hermitian!)
inv_om = np.sort(es[es > 1e-12])
om_from_pencil = np.sort(1.0/inv_om[::-1])[:6]   # largest 1/omega -> smallest omega? invert
om_all = np.sort(np.abs(1.0/es[np.abs(es) > 1e-12]))[:6]
print(f"K^-1/2 Sigma K^-1/2 Hermitian eig (=1/omega); omega6 from |1/eig| = {om_all}")
print(f"target pos6 = {pos[:6]}")
