"""diag_refit5 — umklapp-pair hypothesis test.

The lab-frame pair (true Bloch functions, u at wrapped labels) carries a
per-k umklapp phase in its periodic part whenever k−q wraps:
    ρ̃_true(r) = e^{+2πi G_k·r} conj(ũ_{m,k'}) ũ_{n,k},
    G_k = (rk[k] − rk[q]) − rk[k']   (exact integer)
The torus pair (reference/build_cq, my refit) omits it.  Build the
umklapp-decorated C_u, Z_u at q1 with the STORED m-leg and residual-test
the stored ζ against them.  If r_stored(C_u,Z_u) ≈ 0 → the producer's
fit is umklapp/lab-convention and the refit must adopt the phases.
"""
import sys

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import Mesh

from bse import vq_interp as vqi

RD = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_exciton_bands_2026-07-17"
FX = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
      "05_lorrax_cohsex_native/tmp")
INP = f"{RD}/exciton_smoke.in"

mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
q0 = 1
qw = zx["qfr"][q0]
nk, nb, ns, n_mu = zx["nk"], zx["nb"], zx["ns"], zx["n_mu"]

with mesh:
    rst = vqi.refit_prepare(INP, mesh, zx)
k_frac = zx["k_int"].astype(np.float64) / zx["kgrid"][None, :]
kqs = np.array([vqi.kq_index_of_frac(zx, kf - qw) for kf in k_frac])
# exact integer umklapp: G_k = (rk[k] − rk[q0]) − rk[k']
G_k = (zx["qfr_raw"][:nk] - zx["qfr_raw"][q0][None, :]
       - zx["qfr_raw"][kqs])
assert np.max(np.abs(G_k - np.rint(G_k))) < 1e-6
G_k = np.rint(G_k).astype(int)
print("umklapp G_k per k:", G_k.tolist())

psi = zx["psi"]                                    # (nk, nb, ns, n_mu)
psi_r = np.asarray(jax.device_get(rst["psi_r"])).reshape(
    nk, nb, ns, rst["n_rp"])[:, :, :, : zx["n_rtot"]]

# per-k umklapp phases at centroids and on the r grid
ph_mu = np.exp(2j * np.pi * (zx["rmu_frac"] @ G_k.T)).T    # (nk, n_mu)
ph_r = np.exp(2j * np.pi * (zx["rfrac"] @ G_k.T)).T        # (nk, n_rtot)

# pairs: rho_u[k,m,n,x] = ph[k,x] * sum_s conj(psi[k',m,s,x]) psi[k,n,s,x]
X_u = np.einsum("kmsu,knsu,ku->kmnu", np.conj(psi[kqs]), psi, ph_mu,
                optimize=True).reshape(-1, n_mu)
C_u = np.conj(X_u).T @ X_u
C_t = vqi.build_cq(zx)[q0]
print(f"relF(C_umklapp, C_torus) = {vqi.relF(C_u, C_t):.3e}")

Z_u = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
rc = 4096
for r0 in range(0, zx["n_rtot"], rc):
    r1 = min(r0 + rc, zx["n_rtot"])
    rho = np.einsum("kmsr,knsr,kr->kmnr", np.conj(psi_r[kqs][:, :, :, r0:r1]),
                    psi_r[:, :, :, r0:r1], ph_r[:, r0:r1],
                    optimize=True).reshape(-1, r1 - r0)
    Z_u[:, r0:r1] = np.conj(X_u).T @ rho

# stored zeta in the torus frame with my phase convention
box = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
fi = vqi.flat_idx(zx, zx["gvec"][q0])
ng = int(zx["ngk"][q0])
box[:, fi[:ng]] = zx["ZG"][q0][:, :ng]
if_st = np.fft.ifftn(box.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                     axes=(1, 2, 3), norm="backward").reshape(n_mu, -1)
nZ = np.linalg.norm(Z_u)
for lbl, ph in (("frame+", +1.0), ("frame-", -1.0)):
    zst = np.exp(ph * 2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * if_st
    r = np.linalg.norm(C_u @ zst - Z_u) / nZ
    print(f"  umklapp eqs, stored zeta {lbl}: r = {r:.3e}")
# and my umklapp-refit solution's own residual + B vs stored
lam_tr = np.abs(np.trace(C_u))
L = np.linalg.cholesky(C_u + 1e-14 * lam_tr * np.eye(n_mu))
y = np.linalg.solve(L, Z_u)
zeta_u = np.linalg.solve(L.conj().T, y)
r_mine = np.linalg.norm(C_u @ zeta_u - Z_u) / nZ
print(f"  umklapp eqs, my solve: r = {r_mine:.3e}")
ztG = np.fft.fftn(zeta_u.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                  axes=(1, 2, 3), norm="backward").reshape(n_mu, -1)[:, fi[:ng]]
zt = np.exp(-2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * ztG
V_u = vqi.make_vq(zx, np.pad(zt, ((0, 0), (0, zx["ngkmax"] - ng))), q0)
x = vqi.gap_window_pairs(zx, q0)
print(f"  umklapp refit: tile relF vs stored = "
      f"{vqi.relF(V_u, zx['Vqmunu'][q0]):.3e}   B relF = "
      f"{vqi.relF(vqi.b_block(x, V_u), vqi.b_block(x, zx['Vqmunu'][q0])):.3e}")
print("DONE")
