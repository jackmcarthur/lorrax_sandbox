"""diag_refit — locate the refit on-grid-null failure.

1. GAUGE CONSISTENCY: streamed WFN ψ at the centroid points vs the
   restart's psi_full_y (the refit mixes both sources — X-leg from the
   restart, ρ-leg from the stream; any gauge drift breaks the LSQ).
2. C DIRECTION: refit C_Q at on-grid q1 vs the reference-route C_q[1]
   and its transpose (q-sign convention check).
3. TILE VARIANTS: refit V vs stored V_qmunu[q1], its transpose, conj.
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
from bandstructure.bse_setup import compute_wfns_fi

RD = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_exciton_bands_2026-07-17"
FX = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
      "05_lorrax_cohsex_native/tmp")
INP = f"{RD}/exciton_smoke.in"

mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
C_ref = vqi.build_cq(zx)

with mesh:
    rst = vqi.refit_prepare(INP, mesh, zx)

    # ---- 1. streamed-psi @ centroids vs restart psi_full_y ----
    nk, nb, ns = zx["nk"], zx["nb"], zx["ns"]
    psi_r = np.asarray(jax.device_get(rst["psi_r"])).reshape(
        nk, nb, ns, rst["n_rp"])[:, :, :, : rst["n_rtot"]]
    at_mu = psi_r[:, :, :, zx["rmu_flat"]]              # (nk, nb, ns, n_mu)
    dv = vqi.relF(at_mu, zx["psi"])
    print(f"[1] streamed psi@centroids vs restart psi_full_y: relF = {dv:.3e}")
    per_k = [vqi.relF(at_mu[k], zx["psi"][k]) for k in range(nk)]
    print("    per-k: " + " ".join(f"{v:.1e}" for v in per_k))

    # ---- 2. C direction at on-grid q1 ----
    q0 = 1
    qw = zx["qfr"][q0]
    k_frac = zx["k_int"].astype(np.float64) / zx["kgrid"][None, :]
    qm = k_frac - qw[None, :]
    bundle = compute_wfns_fi(
        ctilde=rst["ctilde"], B_at_mu=rst["B_at_mu"],
        enk_sigma=rst["enk_sigma"], kgrid_co=rst["kgrid_co"],
        band_window_fi=(0, nb), mesh_xy=mesh, q_list=qm, return_coeffs=True)
    psi_m_mu = jnp.asarray(bundle.psi_rmu_Y)
    kern = vqi._refit_kernels(nk, nb, ns, zx["n_mu"], rst["rank"],
                              rst["r_chunk"])
    Xf, C = kern[0](psi_m_mu, jnp.asarray(zx["psi"]))
    C = np.asarray(jax.device_get(C))
    print(f"[2] refit C(q1) vs build_cq C[1]   : relF = {vqi.relF(C, C_ref[1]):.3e}")
    print(f"    refit C(q1) vs build_cq C[1].T : relF = {vqi.relF(C, C_ref[1].T):.3e}")
    print(f"    refit C(q1) vs conj(C[1])      : relF = {vqi.relF(C, np.conj(C_ref[1])):.3e}")

    # ---- 3. tile variants ----
    V_r = vqi.refit_vq(zx, rst, qw, mesh)
    V_st = zx["Vqmunu"][q0]
    for lbl, T in (("V", V_st), ("V.T", V_st.T), ("conj(V)", np.conj(V_st))):
        print(f"[3] refit V vs stored {lbl:8s}: relF = {vqi.relF(V_r, T):.3e}")
    # scale probe: best least-squares scalar between refit and stored
    s = np.vdot(V_st, V_r) / np.vdot(V_st, V_st)
    print(f"    best scalar V_r ≈ s·V_st: s = {s:.6f}, "
          f"resid after scale = {vqi.relF(V_r, s * V_st):.3e}")
print("DONE")
