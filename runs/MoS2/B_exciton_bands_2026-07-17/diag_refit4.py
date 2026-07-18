"""diag_refit4 — is the refit-vs-stored gap a regularization realization
of the ill-conditioned fit?

At q1, stored m-leg: solve MY normal equations with an SVD pseudoinverse
at a sweep of rcond cuts (and the ridge default), and score each ζ
against the stored fit at the B level.  Also: ζ Frobenius norms, C
spectrum tail, and the B-metric of my ridge solution vs each cut (self-
sensitivity — how much B the junk sector carries in the refit itself).
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
    kern = vqi._refit_kernels(nk, nb, ns, n_mu, rst["rank"], rst["r_chunk"])
    cq_and_x, _pm, z_chunk, solve_zeta = kern
    k_frac = zx["k_int"].astype(np.float64) / zx["kgrid"][None, :]
    kqs = np.array([vqi.kq_index_of_frac(zx, kf - qw) for kf in k_frac])
    psi_m_mu = jnp.asarray(zx["psi"][kqs])
    psi_r = rst["psi_r"].reshape(nk, nb, ns, rst["n_rp"])
    psi_m_r = psi_r[jnp.asarray(kqs)]
    psi_r_flat = rst["psi_r"].reshape(nk * nb, ns, rst["n_rp"])
    rc = rst["r_chunk"]
    Xf, C = cq_and_x(psi_m_mu, jnp.asarray(zx["psi"]))
    Z_parts = []
    for r0 in range(0, rst["n_rp"], rc):
        Z_parts.append(z_chunk(
            psi_m_r[:, :, :, r0:r0 + rc],
            psi_r_flat[:, :, r0:r0 + rc].reshape(nk * nb, ns * rc), Xf))
    Zp = jnp.concatenate(Z_parts, axis=1)
    C_h = np.asarray(jax.device_get(C))
    Z_h = np.asarray(jax.device_get(Zp))[:, : zx["n_rtot"]]

lam, R = np.linalg.eigh(0.5 * (C_h + C_h.conj().T))
print(f"C spectrum: max {lam.max():.3e}  min {lam.min():.3e}  "
      f"cond {lam.max()/max(lam.min(),1e-300):.1e}")
frac = [np.sum(lam > c * lam.max()) for c in (1e-4, 1e-6, 1e-8, 1e-10)]
print(f"modes above 1e-4/1e-6/1e-8/1e-10 of max: {frac}")

x = vqi.gap_window_pairs(zx, q0)
V_st = zx["Vqmunu"][q0]
B_st = vqi.b_block(x, V_st)
ph_mu = np.exp(-2j * np.pi * (zx["rmu_frac"] @ qw))[:, None]
fi = vqi.flat_idx(zx, zx["gvec"][q0])
ng = int(zx["ngk"][q0])


def V_from_zeta_r(zr):
    box = np.fft.fftn(zr.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                      axes=(1, 2, 3), norm="backward"
                      ).reshape(n_mu, zx["n_rtot"])
    zt = ph_mu * box[:, fi[:ng]]
    v = vqi.v_slab_on_set(zx, qw, zx["gvec"][q0][:, :ng])
    A = zt * np.sqrt(v)[None, :]
    return np.conj(A) @ A.T


# ridge solution (the current refit)
zeta_ridge = np.asarray(jax.device_get(
    kern[3](jnp.asarray(C_h), Zp)))[:, : zx["n_rtot"]]
print(f"\n||zeta_ridge||_F = {np.linalg.norm(zeta_ridge):.3e}")
V_ridge = V_from_zeta_r(zeta_ridge)
print(f"ridge:      B(refit vs stored) = "
      f"{vqi.relF(vqi.b_block(x, V_ridge), B_st):.3e}")

# stored zeta norm on the r grid (torus frame)
box = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
box[:, fi[:ng]] = zx["ZG"][q0][:, :ng]
if_st = np.fft.ifftn(box.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                     axes=(1, 2, 3), norm="backward").reshape(n_mu, -1)
zeta_st = np.conj(ph_mu) * if_st
print(f"||zeta_stored||_F (band-limited) = {np.linalg.norm(zeta_st):.3e}")

# rcond sweep (SVD-cut solutions of MY equations)
Uh = R.conj().T @ Z_h
for rcond in (1e-6, 1e-8, 1e-10, 1e-12):
    inv = np.where(lam > rcond * lam.max(), 1.0 / np.maximum(lam, 1e-300), 0.0)
    zr = R @ (inv[:, None] * Uh)
    Vr = V_from_zeta_r(zr)
    print(f"rcond {rcond:.0e}: ||zeta|| {np.linalg.norm(zr):.3e}  "
          f"B(vs stored) = {vqi.relF(vqi.b_block(x, Vr), B_st):.3e}  "
          f"B(vs ridge) = {vqi.relF(vqi.b_block(x, Vr), vqi.b_block(x, V_ridge)):.3e}")
print("DONE")
