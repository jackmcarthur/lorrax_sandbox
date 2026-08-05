"""Dump + audit the compiled HLO of the production BSE stack matvec on the
16-GPU (4x4) mesh — prove the ISDF μ/ν dims stay sharded (no full-matrix
all-gather of V_q0 / W_R), i.e. the matvec is genuinely distributed.

Builds ``build_bse_stack_matvec(kernel='bse')`` (= D + V - W) with the REAL
loader V_q0 / W_R / psi_v (correct shapes + shardings) and dummy (zero)
conduction caches, lowers+compiles it for the 4x4 mesh, dumps the HLO, and
lists every collective with its operand shapes.  HLO structure is
value-independent, so the dummy conduction caches are faithful.

Run via run11.sh (16 procs).  Compile-only (no heavy execution).
"""
import os
import re
import sys

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

from bse.bse_w_exact import _create_mesh_xy
from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_ring_comm import make_bse_shardings
from bse.bse_serial import compute_pair_amplitude
from bse.bse_stack_matvec import build_bse_stack_matvec
from common.fft_helpers import make_sharded_ifftn_3d

RANK = jax.process_index()
OUT = os.environ.get("HLO_OUT", "hlo_matvec_16gpu.txt")
INPUT = os.environ.get("HLO_INPUT", "exciton_40_8v8c.in")
BLOCK = int(os.environ.get("HLO_BLOCK", "8"))


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


mesh = _create_mesh_xy(4, 4)
sh = make_bse_shardings(mesh)
log(f"[hlo] device_count={jax.device_count()} process_count={jax.process_count()} "
    f"mesh={dict(mesh.shape)}")

restart = _find_restart_file(INPUT)
data = load_bse_data_from_restart_sharded(
    restart, n_val=8, n_cond=8, mesh_xy=mesh, input_file=INPUT, inject_head=True)
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
nk = nkx * nky * nkz
nc_pad, nv_pad = int(data["n_cond_pad"]), int(data["n_val_pad"])
n_rmu_pad = int(data["n_rmu_pad"])
ns = int(data["psi_v_Y"].shape[2])
log(f"[hlo] nk={nk} ({nkx}x{nky}x{nkz}) nc_pad={nc_pad} nv_pad={nv_pad} "
    f"n_rmu_pad={n_rmu_pad} ns={ns} block={BLOCK}")

# Real W_R / V_q0 (loader), dummy (zero) conduction caches — HLO is
# value-independent, only shapes + shardings matter.
_ifftn = make_sharded_ifftn_3d(mesh, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm="ortho")
W_R = _ifftn(data["W_q"])
V_q0 = jax.device_put(data["V_q0"], sh.V)
psi_v_X, psi_v_Y = data["psi_v_X"], data["psi_v_Y"]
eps_v = data["eps_v"]

psi_c_X = jax.device_put(jnp.zeros((nk, nc_pad, ns, n_rmu_pad), jnp.complex128), sh.psi_x)
psi_c_Y = jax.device_put(jnp.zeros((nk, nc_pad, ns, n_rmu_pad), jnp.complex128), sh.psi_y)
eps_c = jax.device_put(jnp.zeros((nk, nc_pad), jnp.float64), sh.eps)
X = jax.device_put(jnp.zeros((BLOCK, nc_pad, nv_pad, nk), jnp.complex128), sh.X)
M_X = compute_pair_amplitude(psi_c_X, psi_v_X)
M_Y = compute_pair_amplitude(psi_c_Y, psi_v_Y)

matvec = build_bse_stack_matvec(mesh, nkx, nky, nkz, kernel="bse")

compiled = jax.jit(matvec).lower(
    X, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y, eps_c, eps_v, W_R, V_q0, M_X, M_Y
).compile()
hlo = compiled.as_text()

if RANK == 0:
    with open(OUT, "w") as fh:
        fh.write(hlo)
    log(f"[hlo] wrote {OUT} ({len(hlo)} chars)")

    # --- collective audit -------------------------------------------------
    COLL = ("all-gather", "all-to-all", "reduce-scatter", "collective-permute",
            "all-reduce")
    RMU = str(n_rmu_pad)                    # the big ISDF dim (red flag if gathered)
    log(f"\n[hlo] === collective audit (big ISDF dim = {RMU}) ===")
    seen = {c: 0 for c in COLL}
    flags = []
    for line in hlo.splitlines():
        s = line.strip()
        for c in COLL:
            # match the op name at an '= ...c(' or ' c-start(' boundary
            if re.search(rf"\b{re.escape(c)}(-start|-done)?\b", s) and "(" in s:
                seen[c] += 1
                # extract the result shape token (e.g. c128[8,144,1,160])
                shp = re.findall(r"[a-z0-9]+\[[0-9,]*\]", s)
                shp_s = " ".join(shp[:3])
                red = RMU in "".join(re.findall(r"\[[0-9,]*\]", s))
                tag = "  <-- RED FLAG: big ISDF dim" if red else ""
                log(f"  [{c}] {shp_s}{tag}")
                if red:
                    flags.append((c, s[:200]))
    log(f"\n[hlo] collective counts: "
        + ", ".join(f"{c}={n}" for c, n in seen.items() if n))
    if flags:
        log(f"[hlo] *** {len(flags)} collective(s) touch the big ISDF dim "
            f"{RMU} — POSSIBLE full-matrix gather; inspect: ***")
        for c, s in flags[:8]:
            log(f"    {c}: {s}")
    else:
        log(f"[hlo] CLEAN: no collective operand carries the big ISDF dim "
            f"{RMU}; only small band-dim gathers + μ/ν psum-scatters.")
log("[hlo] DONE")
