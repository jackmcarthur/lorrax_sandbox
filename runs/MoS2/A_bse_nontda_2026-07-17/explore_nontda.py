"""Exploration: materialize the full non-TDA optical BSE operator on the gnppm
2v2c fixture and determine the correct structure-preserving reduction.

Answers:
  (1) Are A and B Hermitian / complex-symmetric?
  (2) Does the code's matvec implement [[A,B],[-B,-A]] or [[A,B],[-B^H,-A^H]]?
  (3) Positive eigenvalues of the dense operator (the FIRST checked non-TDA-with-W
      numbers) vs Omega = sqrt(eig(M)), M=(A-B)(A+B).
  (4) Is M self-adjoint in the (A+B) inner product (so metric Lanczos applies)?
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

jax.config.update("jax_enable_x64", True)
np.set_printoptions(precision=6, suppress=False, linewidth=140)

RESTART = sys.argv[1]
INPUT = sys.argv[2]

from bse import bse_io
from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings
from bse.bse_serial import compute_pair_amplitude

data = bse_io._load_ring_subset(RESTART, n_val=2, n_cond=2, px=1, py=1, input_file=INPUT)
psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
eps_c = np.asarray(data["eps_c"]); eps_v = np.asarray(data["eps_v"])
W_q = np.asarray(data["W_q"]); V_q0 = np.asarray(data["V_q0"])
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
nk = nkx * nky * nkz
nc = psi_c.shape[1]; nv = psi_v.shape[1]
N = nc * nv * nk
print(f"fixture: nc={nc} nv={nv} nk={nk} N={N}  nspinor={psi_c.shape[2]} nmu={psi_c.shape[3]}")

mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
sh = make_bse_shardings(mesh)

def materialize_full(include_W):
    mv = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=include_W, screening=False)
    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(jnp.asarray(psi_c), sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(jnp.asarray(psi_v), sh.psi_y)
        W_qs = jax.lax.with_sharding_constraint(jnp.asarray(W_q), sh.W)
        V_q0s = jax.lax.with_sharding_constraint(jnp.asarray(V_q0), sh.V)
        W_R = jnp.fft.ifftn(W_qs, axes=(2, 3, 4), norm="ortho") if include_W else W_qs
        M_X = jax.lax.with_sharding_constraint(compute_pair_amplitude(psi_c_X, psi_v_X), sh.psi_x)
        M_Y = jax.lax.with_sharding_constraint(compute_pair_amplitude(psi_c_Y, psi_v_Y), sh.psi_y)
        # basis: 2N columns, part in {0,1}
        eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk)
        top = np.concatenate([eye, np.zeros_like(eye)], axis=0)  # (2N, nc,nv,nk)
        bot = np.concatenate([np.zeros_like(eye), eye], axis=0)
        Xf = jnp.asarray(np.stack([top, bot], axis=0))  # (2, 2N, nc, nv, nk)
        Xf = jax.lax.with_sharding_constraint(Xf, sh.X_full)
        out = mv(Xf, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                 jnp.asarray(eps_c), jnp.asarray(eps_v), W_R, V_q0s, M_X, M_Y)
        out.block_until_ready()
    out = np.asarray(out)  # (2, 2N, nc, nv, nk)
    # out[po, col] = block-po output for basis col.  columns 0..N-1 = resonant e_I,
    # N..2N-1 = antiresonant e_I.
    outf = out.reshape(2, 2 * N, N)  # (po, col, flat)
    # O[ (po,flat) , col ] = outf[po, col, flat] -> O row index (po,flat), col index col
    O = np.transpose(outf, (0, 2, 1)).reshape(2 * N, 2 * N)  # rows=(po,flat), cols=col
    return O

O = materialize_full(True)
A = O[:N, :N]
B = O[:N, N:]
C21 = O[N:, :N]  # should be -B (code) ; -B^H (physical)
C22 = O[N:, N:]  # should be -A (code) ; -A^H (physical)

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

print("\n== block structure ==")
print(f"||A - A^H||/||A|| = {rel(A, A.conj().T):.3e}   (A Hermitian?)")
print(f"||A - A^T||/||A|| = {rel(A, A.T):.3e}   (A symmetric?)")
print(f"||B - B^H||/||B|| = {rel(B, B.conj().T):.3e}   (B Hermitian?)")
print(f"||B - B^T||/||B|| = {rel(B, B.T):.3e}   (B symmetric?)")
print(f"(2,1) block vs -B      : {rel(C21, -B):.3e}   vs -B^H: {rel(C21, -B.conj().T):.3e}")
print(f"(2,2) block vs -A      : {rel(C22, -A):.3e}   vs -A^H: {rel(C22, -A.conj().T):.3e}")

print("\n== dense eigenvalues (Ry) ==")
ev_code = np.linalg.eigvals(O)
ev_code = ev_code[np.argsort(ev_code.real)]
print(f"code operator [[A,B],[-B,-A]] : max|Im(ev)| = {np.max(np.abs(ev_code.imag)):.3e}")
O_phys = np.block([[A, B], [-B.conj().T, -A.conj().T]])
ev_phys = np.linalg.eigvals(O_phys)
ev_phys = ev_phys[np.argsort(ev_phys.real)]
print(f"phys operator [[A,B],[-B^H,-A^H]]: max|Im(ev)| = {np.max(np.abs(ev_phys.imag)):.3e}")
pos_code = np.sort(ev_code.real[ev_code.real > 1e-9])[:8]
pos_phys = np.sort(ev_phys.real[ev_phys.real > 1e-9])[:8]
print("lowest-8 positive (code): ", pos_code)
print("lowest-8 positive (phys): ", pos_phys)

print("\n== product reduction M=(A-B)(A+B), Omega=sqrt(eig M) ==")
ApB = A + B
AmB = A - B
# PD checks
w_ApB = np.linalg.eigvalsh(0.5 * (ApB + ApB.conj().T))
w_AmB = np.linalg.eigvalsh(0.5 * (AmB + AmB.conj().T))
print(f"eig(Herm(A+B)) range: [{w_ApB.min():.4e}, {w_ApB.max():.4e}]  min>0? {w_ApB.min()>0}")
print(f"eig(Herm(A-B)) range: [{w_AmB.min():.4e}, {w_AmB.max():.4e}]  min>0? {w_AmB.min()>0}")
M = AmB @ ApB
ev_M = np.linalg.eigvals(M)
print(f"M eig: max|Im|={np.max(np.abs(ev_M.imag)):.3e}  min Re={ev_M.real.min():.4e}")
omega = np.sqrt(ev_M.astype(complex))
omega = omega[np.argsort(omega.real)]
om_pos = np.sort(np.abs(omega.real))[:8]
print("Omega=sqrt(eig M) lowest-8 |Re|: ", om_pos)

# metric self-adjointness of M in (A+B) inner product: <Mx,y>_+ = <x,My>_+
rng = np.random.default_rng(0)
x = rng.standard_normal(N) + 1j*rng.standard_normal(N)
y = rng.standard_normal(N) + 1j*rng.standard_normal(N)
lhs = (M @ x).conj() @ (ApB @ y)
rhs = x.conj() @ (ApB @ (M @ y))
print(f"\nM self-adjoint in (A+B) metric: |<Mx,y>+ - <x,My>+|/|.| = {abs(lhs-rhs)/max(abs(rhs),1e-30):.3e}")

# Compare the three "positive eigenvalue" lists
print("\n== RECONCILIATION ==")
print("dense-code pos   :", pos_code[:6])
print("dense-phys pos   :", pos_phys[:6])
print("sqrt(eig M) pos  :", om_pos[:6])
print("TDA A eigvalsh   :", np.sort(np.linalg.eigvalsh(0.5*(A+A.conj().T)))[:6])
