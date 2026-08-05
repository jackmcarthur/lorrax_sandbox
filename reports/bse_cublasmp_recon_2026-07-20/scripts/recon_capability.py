"""Capability + timing proof for the 2-D-distributed cuBLASMp V_Q
reconstruction (bse.vq_interp._distributed_prims / _recon_body).

THE POINT: a single n_mu x n_mu tile need NEVER fit on one proc.  We run the
reconstruction GEMMs (S = R g R^H, C^2, S@C^2) on RANDOM Hermitian C_q at
n_mu large enough that a replicated per-proc tile is a large fraction of a
40 GB A100 — 2-D-sharded via cuBLASMp it is n_mu/Px x n_mu/Py per proc.

  * DISTRIBUTED consistency (no gather, purely sharded reductions):
      C_reb = R diag(lam) R^H  ->  ||C_reb - C|| / ||C||          (eigh+GEMM)
      S     = R diag(g)  R^H,   g = lam^2/(lam^2 + c^2), c = eps*lam_max
      identity  S (C^2 + c^2 I) = C^2  (exact):
          ||S@(C@C) + c^2 S - C@C|| / ||C@C||                     (all cuBLASMp)
  * MEMORY: per-proc shard bytes replicated (n_mu^2*16) vs distributed
    (n_mu^2/(Px*Py)*16); the size where replicated OOMs.
  * TIMING: distributed recon wall vs replicated recon wall -> crossover n_mu.

Launch (16 proc / 4x4, one GPU each) via run11.sh:
    JID=<jid> ./run11.sh <dir> python3 -u -m recon_capability [sizes...]
(the report scripts dir is on PYTHONPATH through run11.sh's -m by cwd; we add
 it explicitly below so `-m` is not required).
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
if not os.environ.get(_DIST) and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    kw = {"local_device_ids": [0]} if cvd and "," not in cvd else {}
    try:
        jax.distributed.initialize(**kw)
    except Exception:
        pass
    os.environ[_DIST] = "1"

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from ffi.cublasmp import batched_distributed_gemm

EPS_TIK = 1e-4


def log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def _mesh(px, py):
    return Mesh(np.asarray(jax.devices()).reshape(px, py), axis_names=("x", "y"))


def _sharded_random_hermitian(key, n, sh2):
    """Random Hermitian C (n,n), generated 2-D-sharded (out_shardings=sh2,
    no replicated intermediate) then symmetrised via a transpose reshard."""
    @jax.jit
    def _gen(k):
        k1, k2 = jax.random.split(k)
        A = (jax.random.normal(k1, (n, n), dtype=jnp.float64)
             + 1j * jax.random.normal(k2, (n, n), dtype=jnp.float64))
        A = jax.lax.with_sharding_constraint(A, sh2)
        C = 0.5 * (A + jnp.conj(A).T) + n * jnp.eye(n, dtype=jnp.complex128)
        return jax.lax.with_sharding_constraint(C, sh2)
    return _gen(key)


def _shard_bytes(x):
    sh = x.addressable_shards[0].data
    return tuple(sh.shape), int(sh.nbytes)


def distributed_recon_check(n, mesh, sh2, sh3, key):
    """Run the distributed reconstruction on a random Hermitian C(n,n) and
    return (consistency dict, timing dict, per-proc shard info)."""
    Px = int(mesh.shape["x"])
    Py = int(mesh.shape["y"])
    C = _sharded_random_hermitian(key, n, sh2)
    C = jax.block_until_ready(C)
    Cb = C[None]                                        # (1, n, n) P(None,x,y)
    Cb = jax.lax.with_sharding_constraint(Cb, sh3)

    from ffi.cusolvermp.eigh import distributed_eigh

    def _z(shape):
        return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                       out_shardings=sh3)()

    def gemm(A, B, transa="N", transb="N"):
        m = A.shape[1] if transa == "N" else A.shape[2]
        nn = B.shape[2] if transb == "N" else B.shape[1]
        return batched_distributed_gemm(A, B, _z((A.shape[0], int(m), int(nn))),
                                        mesh=mesh, transa=transa, transb=transb)

    def gram(evec, gvec):                               # R diag(g) R^H (raw buf)
        Bs = jax.lax.with_sharding_constraint(gvec[:, :, None] * evec, sh3)
        return gemm(evec, Bs, transa="C", transb="N")

    t = {}
    t0 = time.perf_counter()
    w, Qraw = distributed_eigh(C, mesh=mesh)            # 2-D sharded eigh
    lam = w[None]                                       # (1, n)
    Qb = jax.lax.with_sharding_constraint(Qraw[None], sh3)
    jax.block_until_ready((lam, Qb))
    t["eigh"] = time.perf_counter() - t0

    lam_max = lam.max(axis=1, keepdims=True)
    c2 = (EPS_TIK * lam_max) ** 2                       # (1,1)
    g = lam ** 2 / (lam ** 2 + c2)

    t0 = time.perf_counter()
    C_reb = gram(Qb, lam)                               # R Lam R^H  (== C)
    S = gram(Qb, g)                                     # R g   R^H
    C2 = gemm(Cb, Cb)                                   # C @ C
    SC2 = gemm(S, C2)                                   # S @ C^2
    resid = SC2 + c2[:, :, None] * S - C2               # S(C^2+c^2I) - C^2
    jax.block_until_ready((C_reb, S, C2, resid))
    t["recon_gemms"] = time.perf_counter() - t0

    # sharded Frobenius norms (scalar out; NEVER gather a full tile)
    def _fro(x):
        return float(jnp.sqrt(jnp.sum(jnp.abs(x) ** 2)))
    nrm_C = _fro(Cb)
    nrm_C2 = _fro(C2)
    cons = {
        "eigh_reb_rel": _fro(C_reb - Cb) / nrm_C,
        "filter_identity_rel": _fro(resid) / nrm_C2,
    }
    ashape, abytes = _shard_bytes(S)
    info = {
        "per_proc_shard_shape": ashape,
        "per_proc_bytes": abytes,
        "per_proc_mib": abytes / 2 ** 20,
        "replicated_tile_bytes": n * n * 16,
        "replicated_tile_gib": n * n * 16 / 2 ** 30,
        "Px": Px, "Py": Py,
    }
    del C, Cb, C_reb, S, C2, SC2, resid
    return cons, t, info


def replicated_recon_attempt(n, mesh):
    """Attempt the REPLICATED-batched recon on one q at n_mu = n: a full
    n x n tile per device + eigh workspace + the R g R^H reconstruction.
    Returns ('ok', wall) or ('OOM', msg)."""
    repl = NamedSharding(mesh, P())                     # fully replicated
    try:
        @jax.jit
        def _run(key):
            k1, k2 = jax.random.split(key)
            A = (jax.random.normal(k1, (n, n), dtype=jnp.float64)
                 + 1j * jax.random.normal(k2, (n, n), dtype=jnp.float64))
            C = 0.5 * (A + jnp.conj(A).T) + n * jnp.eye(n, dtype=jnp.complex128)
            lam, R = jnp.linalg.eigh(C)                 # replicated eigh
            g = lam ** 2 / (lam ** 2 + (EPS_TIK * lam.max()) ** 2)
            S = jnp.einsum("mr,r,nr->mn", R, g, jnp.conj(R))
            return jnp.linalg.norm(S)
        t0 = time.perf_counter()
        out = jax.jit(_run, out_shardings=repl)(jax.random.key(1))
        out = jax.block_until_ready(out)
        return "ok", time.perf_counter() - t0
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
        return "OOM", msg


def main():
    sizes = [int(x) for x in sys.argv[1:]] or [640, 4096, 16384]
    px = py = int(round(jax.process_count() ** 0.5))
    if px * py != jax.process_count():
        log(f"need a square process count; got {jax.process_count()}")
        return 2
    mesh = _mesh(px, py)
    sh2 = NamedSharding(mesh, P("x", "y"))
    sh3 = NamedSharding(mesh, P(None, "x", "y"))
    log(f"=== cuBLASMp distributed-recon capability proof ===")
    log(f"devices={jax.device_count()} procs={jax.process_count()} "
        f"mesh={px}x{py}")

    results = {}
    for n in sizes:
        if n % (px * py) != 0 or n % np.lcm(px, py) != 0:
            log(f"[skip] n={n} not divisible by mesh")
            continue
        cons, t, info = distributed_recon_check(
            n, mesh, sh2, sh3, jax.random.key(n))
        log(f"\n--- n_mu = {n} (mesh {px}x{py}) ---")
        log(f"  DISTRIBUTED per-proc shard: {info['per_proc_shard_shape']} "
            f"= {info['per_proc_mib']:.1f} MiB  (replicated tile would be "
            f"{info['replicated_tile_gib']:.2f} GiB/proc)")
        log(f"  eigh reconstruct  ||RLamR^H - C||/||C||   = "
            f"{cons['eigh_reb_rel']:.3e}")
        log(f"  filter identity   ||S(C^2+c^2I)-C^2||/||C^2|| = "
            f"{cons['filter_identity_rel']:.3e}")
        log(f"  time: eigh={t['eigh']:.2f}s recon_gemms={t['recon_gemms']:.3f}s")
        results[f"dist_{n}"] = dict(cons=cons, t=t, info=info)

    # replicated OOM demonstration at the large sizes
    for n in sorted(set(sizes)):
        if n < 16384:
            continue
        status, detail = replicated_recon_attempt(n, mesh)
        gib = n * n * 16 / 2 ** 30
        log(f"\n  REPLICATED recon attempt n_mu={n} "
            f"(one {gib:.2f} GiB tile/proc + eigh workspace + R g R^H): "
            f"{status}")
        if status == "OOM":
            log(f"    -> {detail}")
        results[f"repl_{n}"] = dict(status=status, detail=detail,
                                    tile_gib=gib)

    if jax.process_index() == 0:
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "logs", "recon_capability.npz")
        np.savez(out, results=np.array([results], dtype=object),
                 sizes=np.array(sizes), mesh=np.array([px, py]))
        log(f"\nsaved {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
