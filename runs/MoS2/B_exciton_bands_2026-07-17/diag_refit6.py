"""diag_refit6 — which pair family does the stored ζ actually FIT?

For pair family T ∈ {torus, umklapp} build (C_T, Z_T, ‖ρ_T‖²) at q1
(stored m-leg) and evaluate the total ISDF expansion error
    E_T(ζ) = ‖ρ_T(r) − Σ_μ ζ̃_μ(r) ρ_T(μ)‖² / ‖ρ_T‖²
           = 1 + [Tr(ζ† C_T ζ) − 2Re Tr(ζ† Z_T)] / ‖ρ_T‖²
for ζ ∈ {stored (band-limited), my torus fit, my umklapp fit} — all in
the same torus frame (stored mapped with the e^{+iq·s_μ} phase).  The
stored ζ's low-E family is the producer's convention.
"""
import sys

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
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
G_k = np.rint(zx["qfr_raw"][:nk] - zx["qfr_raw"][q0][None, :]
              - zx["qfr_raw"][kqs]).astype(int)
psi = zx["psi"]
psi_r = np.asarray(jax.device_get(rst["psi_r"])).reshape(
    nk, nb, ns, rst["n_rp"])[:, :, :, : zx["n_rtot"]]

fam = {}
for name, ph_mu_k, ph_r_k in (
        ("torus", np.ones((nk, n_mu)), np.ones((nk, zx["n_rtot"]))),
        ("umklapp", np.exp(2j * np.pi * (zx["rmu_frac"] @ G_k.T)).T,
         np.exp(2j * np.pi * (zx["rfrac"] @ G_k.T)).T)):
    import jax.numpy as jnp
    Xj = jnp.einsum("kmsu,knsu,ku->kmnu", jnp.conj(jnp.asarray(psi[kqs])),
                    jnp.asarray(psi), jnp.asarray(ph_mu_k)).reshape(-1, n_mu)
    C = np.asarray(jnp.conj(Xj).T @ Xj)
    X = np.asarray(Xj)
    Z = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
    rho2 = 0.0
    rc = 4096
    pmr = jnp.asarray(psi_r[kqs]); pr = jnp.asarray(psi_r)
    phr = jnp.asarray(ph_r_k)
    for r0 in range(0, zx["n_rtot"], rc):
        r1 = min(r0 + rc, zx["n_rtot"])
        rho = jnp.einsum("kmsr,knsr,kr->kmnr", jnp.conj(pmr[:, :, :, r0:r1]),
                         pr[:, :, :, r0:r1], phr[:, r0:r1]).reshape(-1, r1 - r0)
        Z[:, r0:r1] = np.asarray(jnp.conj(Xj).T @ rho)
        rho2 += float(jnp.sum(jnp.abs(rho) ** 2))
    fam[name] = (C, Z, rho2)

# candidate zetas (torus frame)
box = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
fi = vqi.flat_idx(zx, zx["gvec"][q0])
ng = int(zx["ngk"][q0])
box[:, fi[:ng]] = zx["ZG"][q0][:, :ng]
if_st = np.fft.ifftn(box.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                     axes=(1, 2, 3), norm="backward").reshape(n_mu, -1)
z_stored = np.exp(2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * if_st

cands = {"stored": z_stored}
for name in ("torus", "umklapp"):
    C, Z, _ = fam[name]
    L = np.linalg.cholesky(C + 1e-14 * np.abs(np.trace(C)) * np.eye(n_mu))
    cands["fit_" + name] = np.linalg.solve(
        L.conj().T, np.linalg.solve(L, Z))

print(f"{'zeta':<14s} {'E_torus':>12s} {'E_umklapp':>12s}")
for zn, z in cands.items():
    row = []
    for name in ("torus", "umklapp"):
        C, Z, rho2 = fam[name]
        quad = np.real(np.trace(np.conj(z).T @ (C @ z)))
        lin = np.real(np.trace(np.conj(Z).T @ z))
        E = (rho2 - 2 * lin + quad) / rho2
        row.append(E)
    print(f"{zn:<14s} {row[0]:12.4e} {row[1]:12.4e}")
print("DONE")
