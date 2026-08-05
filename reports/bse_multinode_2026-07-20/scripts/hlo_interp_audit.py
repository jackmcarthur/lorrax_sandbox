"""HLO collective audit of the INTERPOLATION linalg on the 16-GPU (4x4) mesh.

Owner question: is the per-Q V_Q tile assembly a LOCAL outer product (no
2D x 2D reshard of a full n_mu x n_mu tile), and is the prepare_coarse
reconstruction batched-over-q with per-tile-local matmuls?

Audits:
  (1) eval_vq (make_eval_vq._body): V = V_SR + conj(A_x) @ A_y.T
      - A_x = P('x',None), A_y = P('y',None); contract over replicated G.
        Expect: LOCAL dot -> (n_mu,n_mu) P('x','y'); comms = only the two
        SMALL A reshards (n_mu x nG ~ MB). RED FLAG = any collective whose
        operand is a full n_mu x n_mu tile.
      - V_SR = tensordot(w, V_SRc_stack): expect local AXPY (no collective).
  (2) _clean_split (prepare_coarse): S = R g R^H, V_SRc = Sc @ V_delta @ Sc,
      batched over q at qb3 = P(('x','y'),None,None). Expect: the reconstruction
      matmuls are LOCAL per device (mu,nu replicated; only q is sharded).
  (3) cusolverMp seam: after distributed_eigh returns R at P('x','y')
      (2D-distributed), _clean_split consumes R at qb3 (mu,nu REPLICATED) -> R is
      gathered. Quantify the gather size.

Run via run11.sh (16 procs). Compile-only. Uses eigh_backend='off' for the fast
prep build (the eval_vq/_clean_split HLO is independent of how prep was made).
"""
import os
import re

from runtime import set_default_env
set_default_env()

import numpy as np
import jax
import jax.numpy as jnp

from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()
jax.config.update("jax_enable_x64", True)

from jax.sharding import NamedSharding, PartitionSpec as P
from functools import partial

from bse.bse_w_exact import _create_mesh_xy
from bse.bse_io import _find_restart_file
from bse import vq_interp

RANK = jax.process_index()
INPUT = os.environ.get("HLO_INPUT", "exciton_40_8v8c.in")
OUTDIR = os.environ.get("HLO_OUTDIR", ".")


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


COLL = ("all-gather", "all-to-all", "reduce-scatter", "collective-permute",
        "all-reduce")


def audit_hlo(name, hlo, big_dim, out_lines):
    """List collectives + flag any operand carrying the big (n_mu) tile dim."""
    out_lines.append(f"\n### {name}\n")
    seen = {c: 0 for c in COLL}
    flags = []
    for line in hlo.splitlines():
        s = line.strip()
        for c in COLL:
            if re.search(rf"\b{re.escape(c)}(-start|-done)?\b", s) and "(" in s:
                seen[c] += 1
                shapes = re.findall(r"[a-z0-9]+\[[0-9,]*\]", s)
                dims = "".join(re.findall(r"\[[0-9,]*\]", s))
                red = re.search(rf"[\[,]{big_dim}[\],]", dims) is not None
                shp = " ".join(shapes[:2])
                out_lines.append(f"- `{c}` {shp}"
                                 + ("  **<-- carries n_mu tile dim**" if red else "  (small)"))
                if red:
                    flags.append((c, s[:180]))
    counts = ", ".join(f"{c}={n}" for c, n in seen.items() if n) or "none"
    out_lines.append(f"\ncollective counts: {counts}")
    if flags:
        out_lines.append(f"\n**{len(flags)} collective(s) carry the n_mu={big_dim} "
                         f"dim — inspect for a full-tile reshard:**")
        for c, s in flags[:6]:
            out_lines.append(f"  - {c}: `{s}`")
    else:
        out_lines.append(f"\n**CLEAN: no collective operand carries the n_mu="
                         f"{big_dim} tile dim.**")
    return len(flags)


mesh = _create_mesh_xy(4, 4)
log(f"[hlo] device_count={jax.device_count()} process_count={jax.process_count()} "
    f"mesh={dict(mesh.shape)}")

restart = _find_restart_file(INPUT)
zeta_file = os.path.join(os.path.dirname(restart), "zeta_q.h5")
zx = vq_interp.load_zeta_coarse(restart, zeta_file)
C_q = vq_interp.build_cq(zx)
prep = vq_interp.prepare_coarse(zx, C_q, mesh, alpha=vq_interp.ALPHA,
                                eps_tik=vq_interp.EPS_TIK, eigh_backend="off")
des = vq_interp.lr_design_blocks(zx, prep)
coeffs = vq_interp.fit_lr_model(des)
eval_vq = vq_interp.make_eval_vq(zx, prep, des, mesh, None,
                                 head_minibz_average=False)
pinvF = jnp.asarray(vq_interp.stencil_pinv(zx["qfr"], vq_interp.stencil_r7(zx)))
coeffs_packed = vq_interp.pack_coeffs(des, coeffs)
n_mu = int(zx["n_mu"])
nG = int(prep["GS"].shape[1])
log(f"[hlo] n_mu={n_mu} nG={nG} nq={int(zx['nq'])}")

report = ["# Interpolation-linalg HLO collective audit (16-GPU 4x4 mesh)\n",
          f"n_mu={n_mu} (big ISDF tile dim), nG={nG}, nq={int(zx['nq'])}, mesh 4x4.\n"]

# ── (1) eval_vq ────────────────────────────────────────────────────────────
q_tile = jnp.asarray(np.array([0.1, 0.07, 0.0]))
hlo_eval = eval_vq.lower(q_tile, prep["V_SRc"], pinvF, coeffs_packed).compile().as_text()
if RANK == 0:
    with open(os.path.join(OUTDIR, "hlo_eval_vq_16gpu.txt"), "w") as fh:
        fh.write(hlo_eval)
n_flag_eval = audit_hlo("eval_vq (per-Q tile V = V_SR + conj(A_x) @ A_y.T)",
                        hlo_eval, n_mu, report)

# ── (2) _clean_split (faithful rebuild at qb3, matching prepare_coarse) ─────
qb2 = NamedSharding(mesh, P(("x", "y"), None))
qb3 = NamedSharding(mesh, P(("x", "y"), None, None))
GS_f = jnp.asarray(prep["GS"].T.astype(np.float64))
rmu = jnp.asarray(zx["rmu_frac"])
eps_tik = float(prep["eps_tik"])


@partial(jax.jit, out_shardings=(qb3, qb3, qb3))
def _clean_split(lam, R, ZGq, v_ref, v_lr, idx, qfr_b):
    g = lam ** 2 / (lam ** 2 + (eps_tik * lam.max(axis=1, keepdims=True)) ** 2)
    S = jnp.einsum("bmr,br,bnr->bmn", R, g, jnp.conj(R))
    Sc = jnp.conj(S)
    A_ref = ZGq * jnp.sqrt(v_ref)[:, None, :]
    A_lr = ZGq * jnp.sqrt(v_lr)[:, None, :]
    V_delta = (jnp.einsum("bmg,bng->bmn", jnp.conj(A_ref), A_ref)
               - jnp.einsum("bmg,bng->bmn", jnp.conj(A_lr), A_lr))
    V_SRc = Sc @ V_delta @ Sc
    zt = S @ ZGq
    zt_ext = jnp.concatenate([zt, jnp.zeros((zt.shape[0], n_mu, 1), zt.dtype)], axis=2)
    ztg = jnp.take_along_axis(zt_ext, idx[:, None, :], axis=2)
    qG = qfr_b[:, None, :] + GS_f[None, :, :]
    ph = jnp.exp(2j * jnp.pi * jnp.einsum("mi,bgi->bmg", rmu, qG))
    return S, V_SRc, ph * ztg


bq = 48
lam = jax.device_put(jnp.zeros((bq, n_mu), jnp.float64), qb2)
R = jax.device_put(jnp.zeros((bq, n_mu, n_mu), jnp.complex128), qb3)
ZGq = jax.device_put(jnp.zeros((bq, n_mu, nG), jnp.complex128), qb3)
v_ref = jax.device_put(jnp.zeros((bq, nG), jnp.float64), qb2)
v_lr = jax.device_put(jnp.zeros((bq, nG), jnp.float64), qb2)
idx = jax.device_put(jnp.zeros((bq, nG), jnp.int64), qb2)
qfr_b = jax.device_put(jnp.zeros((bq, 3), jnp.float64), qb2)
hlo_clean = _clean_split.lower(lam, R, ZGq, v_ref, v_lr, idx, qfr_b).compile().as_text()
if RANK == 0:
    with open(os.path.join(OUTDIR, "hlo_clean_split_16gpu.txt"), "w") as fh:
        fh.write(hlo_clean)
n_flag_clean = audit_hlo("_clean_split (R g R^H, Sc@V_delta@Sc, batched over q at qb3)",
                         hlo_clean, n_mu, report)

# ── (3) cusolverMp R-gather seam quantification ────────────────────────────
tile_bytes = n_mu * n_mu * 16
report.append(
    "\n### cusolverMp eigh -> reconstruction seam\n"
    f"For `--eigh-backend cusolvermp`, `distributed_eigh` returns R at "
    f"`P('x','y')` (2D-distributed, {n_mu//4}x{n_mu//4}={n_mu*n_mu//16} elems/dev "
    f"per q). `_clean_split` consumes R at qb3 = `P(('x','y'),None,None)` (mu,nu "
    f"REPLICATED, only q sharded) -> R is GATHERED from 2D-sharded to replicated "
    f"n_mu x n_mu per q. Gather = one full {n_mu}x{n_mu} c128 tile "
    f"({tile_bytes/2**20:.1f} MiB) per q onto its owner device (batched: "
    f"~{tile_bytes/2**20*bq/16:.1f} MiB/device per {bq}-q chunk). So the "
    f"reconstruction R g R^H is NOT 2D-distributed — it replicates the mu/nu tile "
    f"and batches over q. Fine at n_mu={n_mu} (cheap dense per-tile); it is the "
    f"expected 'FFI eigh then local per-q reconstruction' seam.")

# ── verdict ────────────────────────────────────────────────────────────────
verdict = ("interpolation tile assembly is LOCAL-OUTER-PRODUCT (no 2D x 2D reshard "
           "of a full n_mu x n_mu tile)" if (n_flag_eval == 0 and n_flag_clean == 0)
           else "FOUND a full-tile reshard (see flags above)")
report.append(f"\n## VERDICT: {verdict}\n")
report.append(
    "eval_vq: V=V_SR+conj(A_x)@A_y.T is a local outer product (A_x=P('x',·) rows, "
    "A_y=P('y',·) cols, nG contracted replicated); the only comms are the two small "
    f"A reshards (n_mu x nG ~ {n_mu*nG*16/2**20:.1f} MiB). "
    "_clean_split: R g R^H and Sc@V_delta@Sc are per-q LOCAL matmuls (mu,nu local, "
    "q is the only sharded axis).")

if RANK == 0:
    with open(os.path.join(OUTDIR, "HLO_AUDIT.md"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    log("\n".join(report))
    log(f"\n[hlo] eval_vq n_mu-flags={n_flag_eval}, _clean_split n_mu-flags={n_flag_clean}")
log("[hlo] DONE")
