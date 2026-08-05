"""n_mu=640 correctness: the 2-D-distributed cuBLASMp reconstruction outputs
(S, V_SRc, Fch) vs the replicated-batched _clean_split on the MoS2 fixture.

Both paths use the SAME cusolverMp eigh (the only difference is the
reconstruction backend: replicated einsum vs cuBLASMp on 2-D-sharded tiles),
so they must agree to ~1e-9 (the cuBLASMp GEMMs reassociate at the ULP level;
this is the 640-case bit-match that gates the full exciton-band run).

Launch (16 proc / 4x4) via run11.sh in this dir.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from runtime import set_default_env  # noqa: E402
set_default_env()
from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend  # noqa: E402
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

from jax.sharding import Mesh, PartitionSpec as P  # noqa: E402
from bse import vq_interp as vqi  # noqa: E402

# MoS2 12x12 fixture (nq=144, divisible by the 16-device mesh — the
# replicated-batched baseline shards q over the flattened mesh so needs
# nq % 16 == 0; the 3x3 nq=9 fixture cannot run that baseline on 4x4).
FX = ("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/04_mos2_12x12_bands_"
      "2026-07-18/11_lorrax_exciton_bands_minibz_16gpu/tmp")


def log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def _relF(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def main():
    px = py = int(round(jax.process_count() ** 0.5))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), axis_names=("x", "y"))
    log(f"[dist] devices={jax.device_count()} procs={jax.process_count()} "
        f"mesh={px}x{py}")
    zx = vqi.load_zeta_coarse(f"{FX}/isdf_tensors_640.h5", f"{FX}/zeta_q.h5")
    log(f"  n_mu={zx['n_mu']} nq={zx['nq']} ngkmax={zx['ngkmax']}")
    C_q = vqi.build_cq(zx)

    with mesh:
        prep_r = vqi.prepare_coarse(zx, C_q, mesh, eigh_backend="cusolvermp",
                                    distributed_recon=False)
        prep_d = vqi.prepare_coarse(zx, C_q, mesh, eigh_backend="cusolvermp",
                                    distributed_recon=True)

    dS = _relF(prep_d["S"], prep_r["S"])
    dV = _relF(prep_d["V_SRc_np"], prep_r["V_SRc_np"])
    dF = _relF(prep_d["Fch"], prep_r["Fch"])
    # also the device V_SRc stack (what eval_vq consumes)
    dVdev = _relF(np.asarray(vqi._to_host(prep_d["V_SRc"])),
                  np.asarray(vqi._to_host(prep_r["V_SRc"])))
    log("\n=== distributed-recon vs replicated-recon (n_mu=640) ===")
    log(f"  relF(S)          = {dS:.3e}")
    log(f"  relF(V_SRc_np)   = {dV:.3e}")
    log(f"  relF(Fch)        = {dF:.3e}")
    log(f"  relF(V_SRc dev)  = {dVdev:.3e}")
    ok = max(dS, dV, dF, dVdev) <= 1e-9
    log(f"  {'PASS' if ok else 'FAIL'} at 1e-9")

    if jax.process_index() == 0:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "logs", "recon_bitmatch_640.npz")
        np.savez(out, dS=dS, dV=dV, dF=dF, dVdev=dVdev,
                 n_mu=zx["n_mu"], nq=zx["nq"])
        log(f"saved {os.path.abspath(out)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
