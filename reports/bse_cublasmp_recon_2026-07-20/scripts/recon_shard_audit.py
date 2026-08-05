"""N_mu^2-DISTRIBUTION AUDIT (owner directive): prove that EVERY intermediate
of size proportional to n_mu^2 in the distributed-recon path is sharded across
ALL P procs (per-proc shard = n_mu/Px x n_mu/Py), and NONE is replicated,
half-distributed (one axis), or on a sub-mesh.

For each intermediate we print the logical shape, the PartitionSpec, the ACTUAL
per-proc addressable-shard shape (the empirical proof, not an assertion), and a
verdict.  The only things allowed to be replicated are O(n_mu) or smaller
(eigenvalues lam, filter weights g).

Launch (16 proc / 4x4) via run11.sh.
"""
from __future__ import annotations

import os
import sys

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
from bse import vq_interp as vqi


def log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def _report(name, x, n_mu, Px, Py):
    """Print logical shape / spec / per-proc shard shape + a verdict."""
    spec = x.sharding.spec
    shard = x.addressable_shards[0].data.shape
    mib = x.addressable_shards[0].data.nbytes / 2 ** 20
    # n_mu^2-carrying axes are the last two; expected full-mesh shard divides
    # BOTH by Px and Py.
    r, c = shard[-2], shard[-1]
    lr, lc = x.shape[-2], x.shape[-1]
    full = (r == lr // Px and c == lc // Py)
    repl = (r == lr and c == lc)
    half = (not full and not repl and (r == lr or c == lc))
    verdict = ("FULL-MESH 2-D (ok)" if full else
               "REPLICATED (DEFEAT)" if repl else
               "HALF-DISTRIBUTED (DEFEAT)" if half else
               "OTHER (check)")
    log(f"  {name:<12s} shape={str(tuple(x.shape)):<18s} spec={str(spec):<22s}"
        f" shard/proc={str(tuple(shard)):<16s} {mib:8.2f} MiB  {verdict}")
    return full


def main():
    px = py = int(round(jax.process_count() ** 0.5))
    mesh = Mesh(np.asarray(jax.devices()).reshape(px, py), axis_names=("x", "y"))
    Px, Py = px, py
    n_mu, ngk, nq = 640, 340, 2
    sh2 = NamedSharding(mesh, P("x", "y"))
    sh3 = NamedSharding(mesh, P(None, "x", "y"))
    log(f"=== N_mu^2-distribution audit: n_mu={n_mu} mesh {Px}x{Py} "
        f"(one tile replicated = {n_mu*n_mu*16/2**20:.0f} MiB; "
        f"full-mesh shard = {n_mu*n_mu*16/(Px*Py)/2**20:.0f} MiB/proc) ===")

    # synthetic sharded Hermitian C + random ZG/v (per q), on device only.
    rng = np.random.default_rng(0)
    from ffi.cusolvermp.eigh import distributed_eigh
    prims = vqi._distributed_prims(mesh)
    lams, Qs = [], []
    C_list = []
    for b in range(nq):
        A = rng.standard_normal((n_mu, n_mu)) + 1j * rng.standard_normal((n_mu, n_mu))
        C = 0.5 * (A + A.conj().T) + n_mu * np.eye(n_mu)
        Cd = jax.device_put(jnp.asarray(C), sh2)
        C_list.append(Cd)
        w, Qraw = distributed_eigh(Cd, mesh=mesh)
        lams.append(w)
        Qs.append(Qraw)
    lam = jnp.stack(lams)
    evec = jax.lax.with_sharding_constraint(jnp.stack(Qs), sh3)
    ZG_h = (rng.standard_normal((nq, n_mu, ngk))
            + 1j * rng.standard_normal((nq, n_mu, ngk)))
    ZG = jax.device_put(jnp.asarray(ZG_h), sh3)
    vref = jnp.asarray(np.abs(rng.standard_normal((nq, ngk))))
    vlr = 0.3 * vref

    # re-run the _recon_body sequence with the distributed prims, capturing
    # each intermediate (mirrors bse.vq_interp._recon_body exactly).
    gram = prims["gram"]; gram_outer = prims["gram_outer"]
    gemm = prims["gemm"]; conj = prims["conj"]; constr = prims["constr"]
    g = lam ** 2 / (lam ** 2 + (vqi.EPS_TIK * lam.max(axis=1, keepdims=True)) ** 2)
    S = gram(evec, g)
    Sc = conj(S)
    A_ref = constr(ZG * jnp.sqrt(vref)[:, None, :])
    A_lr = constr(ZG * jnp.sqrt(vlr)[:, None, :])
    Vd_ref = gram_outer(A_ref)
    V_delta = Vd_ref - gram_outer(A_lr)
    T1 = gemm(Sc, V_delta)
    V_SRc = gemm(T1, Sc)
    zt = gemm(S, ZG)
    jax.block_until_ready((evec, S, Sc, A_ref, A_lr, V_delta, T1, V_SRc, zt))

    log(f"\n  {'tensor':<12s} {'logical shape':<18s} {'PartitionSpec':<22s}"
        f" {'per-proc shard':<16s}")
    oks = []
    oks.append(_report("Qraw(evec)", evec, n_mu, Px, Py))
    oks.append(_report("S", S, n_mu, Px, Py))
    oks.append(_report("Sc", Sc, n_mu, Px, Py))
    oks.append(_report("A_ref", A_ref, n_mu, Px, Py))
    oks.append(_report("A_lr", A_lr, n_mu, Px, Py))
    oks.append(_report("V_delta", V_delta, n_mu, Px, Py))
    oks.append(_report("T1=Sc@Vd", T1, n_mu, Px, Py))
    oks.append(_report("V_SRc", V_SRc, n_mu, Px, Py))
    oks.append(_report("zt", zt, n_mu, Px, Py))
    # O(n_mu) replicated operands (ALLOWED)
    log(f"\n  (allowed replicated, O(n_mu)):")
    log(f"  {'lam':<12s} shape={str(tuple(lam.shape)):<18s} "
        f"spec={str(lam.sharding.spec):<22s} shard/proc="
        f"{str(tuple(lam.addressable_shards[0].data.shape))}")
    log(f"  {'g':<12s} shape={str(tuple(g.shape)):<18s} "
        f"spec={str(g.sharding.spec):<22s} shard/proc="
        f"{str(tuple(g.addressable_shards[0].data.shape))}")

    log(f"\n  VERDICT: {'ALL n_mu^2 tensors FULL-MESH 2-D sharded' if all(oks) else 'DEFEAT — some tile not full-mesh'}"
        f" ({sum(oks)}/{len(oks)})")

    # also dump the compiled-HLO sharding of one cuBLASMp gemm (S = evec^H g evec)
    # to show the annotation, not just the runtime shard.
    if jax.process_index() == 0:
        try:
            from ffi.cublasmp import batched_distributed_gemm
            def _z(shape):
                return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                               out_shardings=sh3)()
            Bs = jax.lax.with_sharding_constraint(g[:, :, None] * evec, sh3)
            low = jax.jit(lambda a, b, c: batched_distributed_gemm(
                a, b, c, mesh=mesh, transa="C", transb="N")).lower(
                    evec, Bs, _z((nq, n_mu, n_mu)))
            txt = low.compile().as_text()
            keep = [ln for ln in txt.splitlines()
                    if "sharding" in ln.lower() or "custom-call" in ln.lower()
                    or "all-to-all" in ln.lower() or "collective" in ln.lower()]
            out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "logs", "recon_gram_hlo.txt")
            with open(out, "w") as fh:
                fh.write("\n".join(keep[:120]))
            log(f"\n  wrote gram-GEMM HLO sharding lines -> {os.path.abspath(out)}")
        except Exception as exc:
            log(f"  (HLO dump skipped: {type(exc).__name__}: {exc})")
    return 0 if all(oks) else 1


if __name__ == "__main__":
    sys.exit(main())
