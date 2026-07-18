"""diag_refit2 — is the stored ζ a minimizer of MY normal equations?

At q1 with the STORED m-leg (formulation-null configuration):
  r_mine   = ‖C ζ_mine − Z‖_F / ‖Z‖_F      (my ridge-regularized solution)
  r_stored = ‖C ζ_st − Z‖_F / ‖Z‖_F        (stored fit mapped to my frame)
If r_stored ≈ r_mine: same objective, difference lives in near-null(C).
If r_stored ≫ r_mine: the producer optimizes a DIFFERENT objective.

Also: sphere-level ζ distance raw and Tikhonov-cleaned, and the B-metric
of the cleaned-tile swap (junk-sector localization).
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
C_all = vqi.build_cq(zx)
q0 = 1
qw = zx["qfr"][q0]
nk, nb, ns, n_mu = zx["nk"], zx["nb"], zx["ns"], zx["n_mu"]

with mesh:
    rst = vqi.refit_prepare(INP, mesh, zx)
    # --- rebuild C, Z with the stored m-leg (formulation-null config) ---
    kern = vqi._refit_kernels(nk, nb, ns, n_mu, rst["rank"], rst["r_chunk"])
    cq_and_x, psi_m_chunk, z_chunk, solve_zeta = kern
    k_frac = zx["k_int"].astype(np.float64) / zx["kgrid"][None, :]
    kqs = np.array([vqi.kq_index_of_frac(zx, kf - qw) for kf in k_frac])
    psi_m_mu = jnp.asarray(zx["psi"][kqs])
    psi_r = rst["psi_r"].reshape(nk, nb, ns, rst["n_rp"])
    psi_m_r = psi_r[jnp.asarray(kqs)]
    Xf, C = cq_and_x(psi_m_mu, jnp.asarray(zx["psi"]))
    Z_parts = []
    psi_r_flat = rst["psi_r"].reshape(nk * nb, ns, rst["n_rp"])
    rc = rst["r_chunk"]
    for r0 in range(0, rst["n_rp"], rc):
        Z_parts.append(z_chunk(
            psi_m_r[:, :, :, r0:r0 + rc],
            psi_r_flat[:, :, r0:r0 + rc].reshape(nk * nb, ns * rc), Xf))
    Z = jnp.concatenate(Z_parts, axis=1)[:, : zx["n_rtot"]]
    zeta_mine = solve_zeta(C, jnp.concatenate(Z_parts, axis=1))[:, : zx["n_rtot"]]

    # --- stored ζ mapped to my (torus) frame on the r grid ---
    # recon gives LAB ζ = e^{+iq·r}·IFFT(ZG); my frame: ζ̃ = e^{+iq·r_μ}·IFFT(ZG)
    box = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
    fi = vqi.flat_idx(zx, zx["gvec"][q0])
    ng = int(zx["ngk"][q0])
    box[:, fi[:ng]] = zx["ZG"][q0][:, :ng]
    if_st = np.fft.ifftn(box.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                         axes=(1, 2, 3), norm="backward"
                         ).reshape(n_mu, zx["n_rtot"])
    zeta_st = np.exp(2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * if_st

    C_h = np.asarray(jax.device_get(C))
    Z_h = np.asarray(jax.device_get(Z))
    zm = np.asarray(jax.device_get(zeta_mine))
    nZ = np.linalg.norm(Z_h)
    r_mine = np.linalg.norm(C_h @ zm - Z_h) / nZ
    r_st = np.linalg.norm(C_h @ zeta_st - Z_h) / nZ
    print(f"residuals: r_mine = {r_mine:.3e}   r_stored = {r_st:.3e}")

    # --- sphere-level ζ distance, raw + cleaned ---
    ztm = np.asarray(jax.device_get(
        jnp.fft.fftn(jnp.asarray(zm).reshape(n_mu, zx["nx"], zx["ny"],
                                             zx["nz"]),
                     axes=(1, 2, 3), norm="backward")
        .reshape(n_mu, zx["n_rtot"])))[:, fi[:ng]]
    ztm = np.exp(-2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * ztm
    zst = zx["ZG"][q0][:, :ng]
    print(f"zeta sphere raw relF(mine, stored)      = {vqi.relF(ztm, zst):.3e}")
    lam, R = np.linalg.eigh(0.5 * (C_all[q0] + C_all[q0].conj().T))
    g = lam ** 2 / (lam ** 2 + (1e-4 * lam.max()) ** 2)
    S = (R * g[None, :]) @ R.conj().T
    print(f"zeta sphere CLEANED relF(S·mine, S·stored) = "
          f"{vqi.relF(S @ ztm, S @ zst):.3e}")
    # --- B with cleaned tiles both sides ---
    x = vqi.gap_window_pairs(zx, q0)
    Vm = vqi.make_vq(zx, ztm, q0)
    Vs = zx["Vqmunu"][q0]
    Sc = np.conj(S)
    print(f"B relF raw tiles     = "
          f"{vqi.relF(vqi.b_block(x, Vm), vqi.b_block(x, Vs)):.3e}")
    print(f"B relF cleaned tiles = "
          f"{vqi.relF(vqi.b_block(x, Sc @ Vm @ Sc), vqi.b_block(x, Sc @ Vs @ Sc)):.3e}")
print("DONE")
