"""diag_refit7 — flat-bottom confirmation (torus family, GPU).

E(ζ) = ‖ρ(r) − Σ_μ ζ̃_μ(r) ρ(μ)‖²/‖ρ‖² over the q1 torus pair family
(stored m-leg).  E_min from my exact-minimizer refit; E_stored from the
(band-limited) stored ζ.  E_stored − E_min ≈ 0 ⇒ both are the ISDF fit
within its meaningful resolution; the ζ/B differences live in the flat
bottom of the objective (rank-640 expansion floor ~11%).
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
    Z_parts, rho2 = [], 0.0
    for r0 in range(0, rst["n_rp"], rc):
        pm = psi_m_r[:, :, :, r0:r0 + rc]
        pchunk = psi_r_flat[:, :, r0:r0 + rc].reshape(nk * nb, ns * rc)
        Z_parts.append(z_chunk(pm, pchunk, Xf))
        rho = jnp.einsum("kmsr,knsr->kmnr", jnp.conj(pm),
                         pchunk.reshape(nk, nb, ns, rc))
        rho2 += float(jnp.sum(jnp.abs(rho) ** 2))
    Zp = jnp.concatenate(Z_parts, axis=1)
    zeta_mine = solve_zeta(C, Zp)[:, : zx["n_rtot"]]
    Z = np.asarray(jax.device_get(Zp))[:, : zx["n_rtot"]]
    C_h = np.asarray(jax.device_get(C))

box = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
fi = vqi.flat_idx(zx, zx["gvec"][q0])
ng = int(zx["ngk"][q0])
box[:, fi[:ng]] = zx["ZG"][q0][:, :ng]
if_st = np.fft.ifftn(box.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                     axes=(1, 2, 3), norm="backward").reshape(n_mu, -1)
z_st = np.exp(2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * if_st
zm = np.asarray(jax.device_get(zeta_mine))


def E(z):
    quad = float(np.real(np.trace(np.conj(z).T @ (C_h @ z))))
    lin = float(np.real(np.trace(np.conj(Z).T @ z)))
    return (rho2 - 2 * lin + quad) / rho2


Em, Es = E(zm), E(z_st)
print(f"E_min (my refit)        = {Em:.6e}")
print(f"E_stored (band-limited) = {Es:.6e}")
print(f"E_stored - E_min        = {Es-Em:.3e}  "
      f"({(Es-Em)/Em*100:.2f}% above the floor)")
print("DONE")
