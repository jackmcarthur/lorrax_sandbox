"""Standalone potrf+potrs correctness check on an arbitrary mesh.

Keeps host copies of A and B before the (donating) FFI calls, so unlike
common.cusolvermp_batched_test it works with donate_argnums.

Usage: run from sources/lorrax_D/src with 16 ranks:
  lxrun python3 -u /path/potrs_4x4_check.py --nbatch 8 -n 1216 --mrhs 608 --mesh 4x4
"""
import argparse, os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
if not os.environ.get(_DIST):
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        kw = {"local_device_ids": [0]} if (cvd and "," not in cvd) else {}
        try:
            jax.distributed.initialize(**kw)
        except Exception:
            pass
    os.environ[_DIST] = "1"

import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils
from ffi.cusolvermp import batched_distributed_cholesky, batched_distributed_potrs

ap = argparse.ArgumentParser()
ap.add_argument("--nbatch", type=int, default=8)
ap.add_argument("-n", type=int, default=1216)
ap.add_argument("--mrhs", type=int, default=608)
ap.add_argument("--mesh", type=str, default="4x4")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--illcond", action="store_true",
                help="make A ill-conditioned (cond ~ 1e10)")
args = ap.parse_args()

px, py = (int(v) for v in args.mesh.split("x"))
devs = np.array(jax.devices()[: px * py]).reshape(px, py)
mesh = Mesh(devs, ("x", "y"))

rng = np.random.default_rng(args.seed)
nb, n, m = args.nbatch, args.n, args.mrhs
G = rng.standard_normal((nb, n, n)) + 1j * rng.standard_normal((nb, n, n))
if args.illcond:
    # Hermitian PD with spectrum spanning 1e-10..1
    w = np.logspace(-10, 0, n)
    A_host = np.zeros((nb, n, n), dtype=np.complex128)
    for q in range(nb):
        Q, _ = np.linalg.qr(G[q])
        A_host[q] = (Q * w) @ Q.conj().T
        A_host[q] = 0.5 * (A_host[q] + A_host[q].conj().T)
else:
    A_host = G @ np.conj(np.swapaxes(G, -1, -2)) / n + 2.0 * np.eye(n)[None]
B_host = (rng.standard_normal((nb, n, m))
          + 1j * rng.standard_normal((nb, n, m)))

sh = NamedSharding(mesh, P(None, "x", "y"))
A = jax.device_put(jnp.asarray(A_host), sh)
B = jax.device_put(jnp.asarray(B_host), sh)

L = batched_distributed_cholesky(A, mesh=mesh)  # donates A
X = batched_distributed_potrs(L, B, mesh=mesh)  # donates B
X_full = np.asarray(multihost_utils.process_allgather(X, tiled=True))

res = np.linalg.norm(A_host @ X_full - B_host) / np.linalg.norm(B_host)
# reference conditioning-aware tolerance
if jax.process_index() == 0:
    print(f"mesh={args.mesh} nbatch={nb} n={n} mrhs={m} illcond={args.illcond}")
    print(f"  rel residual |A X - B|/|B| = {res:.3e}")
    ok = res < 1e-8 if not args.illcond else res < 1e-4
    print("  PASS" if ok else "  FAIL")
