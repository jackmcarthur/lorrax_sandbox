"""diag_refit3 — which normal equations does the stored ζ satisfy?

Variants at q1 (stored m-leg everywhere):
  V0: m at k−q  (mine)          r = ‖C ζ_st − Z‖/‖Z‖
  Va: conj fit                  r = ‖C conj(ζ_st) − Z‖/‖Z‖
  Vd: m at k+q  (q-sign flip)   r = ‖C⁺ ζ_st − Z⁺‖/‖Z⁺‖
  Vda: sign flip + conj         r = ‖C⁺ conj(ζ_st) − Z⁺‖/‖Z⁺‖
Frame mapping for ζ_st fixed at qw(q1) in all variants; also tries the
opposite frame phase (e^{−iq·s_μ}) for each — 8 residuals total.
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
    cq_and_x, _pm, z_chunk, _sz = kern
    k_frac = zx["k_int"].astype(np.float64) / zx["kgrid"][None, :]
    psi_r = rst["psi_r"].reshape(nk, nb, ns, rst["n_rp"])
    psi_r_flat = rst["psi_r"].reshape(nk * nb, ns, rst["n_rp"])
    rc = rst["r_chunk"]

    def build_CZ(sign):
        kqs = np.array([vqi.kq_index_of_frac(zx, kf + sign * qw)
                        for kf in k_frac])
        psi_m_mu = jnp.asarray(zx["psi"][kqs])
        psi_m_r = psi_r[jnp.asarray(kqs)]
        Xf, C = cq_and_x(psi_m_mu, jnp.asarray(zx["psi"]))
        Z_parts = []
        for r0 in range(0, rst["n_rp"], rc):
            Z_parts.append(z_chunk(
                psi_m_r[:, :, :, r0:r0 + rc],
                psi_r_flat[:, :, r0:r0 + rc].reshape(nk * nb, ns * rc), Xf))
        Z = np.asarray(jax.device_get(
            jnp.concatenate(Z_parts, axis=1)))[:, : zx["n_rtot"]]
        return np.asarray(jax.device_get(C)), Z

    C_m, Z_m = build_CZ(-1.0)      # m at k−q  (mine)
    C_p, Z_p = build_CZ(+1.0)      # m at k+q

    box = np.zeros((n_mu, zx["n_rtot"]), dtype=np.complex128)
    fi = vqi.flat_idx(zx, zx["gvec"][q0])
    ng = int(zx["ngk"][q0])
    box[:, fi[:ng]] = zx["ZG"][q0][:, :ng]
    if_st = np.fft.ifftn(box.reshape(n_mu, zx["nx"], zx["ny"], zx["nz"]),
                         axes=(1, 2, 3), norm="backward"
                         ).reshape(n_mu, zx["n_rtot"])
    for ph_lbl, ph in (("+", +1.0), ("-", -1.0)):
        zst = np.exp(ph * 2j * np.pi * (zx["rmu_frac"] @ qw))[:, None] * if_st
        for lbl, Cv, Zv, z in (
                (f"m@k-q  frame{ph_lbl}      ", C_m, Z_m, zst),
                (f"m@k-q  frame{ph_lbl} conj ", C_m, Z_m, np.conj(zst)),
                (f"m@k+q  frame{ph_lbl}      ", C_p, Z_p, zst),
                (f"m@k+q  frame{ph_lbl} conj ", C_p, Z_p, np.conj(zst))):
            r = np.linalg.norm(Cv @ z - Zv) / np.linalg.norm(Zv)
            print(f"  {lbl}: r = {r:.3e}")
print("DONE")
