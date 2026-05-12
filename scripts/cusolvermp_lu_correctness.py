"""Standalone correctness micro-bench for batched_distributed_solve_lu.

Goal: demonstrate that cuSolverMp 0.7.2 produces correct answers on a
2×2 mesh for the indefinite Hermitian regime we care about for the
ζ-fit transverse channels.  The original FFI binding docstring warns
that 0.6.0 produced garbage on Px>1 AND Py>1; this script either
confirms 0.7.2 fixes it (path to wiring solver_kind='cusolvermp_lu'
into solve_zeta), or reveals it doesn't.

Run via:
    lxrun python -u scripts/cusolvermp_lu_correctness.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sources', 'lorrax_A', 'src'))

from runtime import set_default_env, init_jax_distributed
set_default_env()
import jax
init_jax_distributed()

import numpy as np
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ffi.cusolvermp import batched_distributed_solve_lu


def _mesh(px: int, py: int) -> Mesh:
    devs = np.array(jax.devices()).reshape(px, py)
    return Mesh(devs, axis_names=('x', 'y'))


def _build_indefinite_hermitian(nq: int, n: int, seed: int) -> np.ndarray:
    """Indefinite Hermitian (Nq, N, N) C128 — both signs of eigenvalue.

    Generates A = (M + M^H)/2 + α I with α small relative to spectral
    norm, then shifts by −trace/N to center the spectrum so we get
    both positive and negative eigenvalues.  This mirrors the actual
    transverse-channel CCT structure where γ̃^i has eigenvalues ±1.
    """
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((nq, n, n)) + 1j * rng.standard_normal((nq, n, n))
    A = 0.5 * (M + M.conj().transpose(0, 2, 1))
    shifts = np.einsum('qii->q', A) / n
    A = A - shifts[:, None, None] * np.eye(n)[None]
    return A.astype(np.complex128)


def _build_rhs(nq: int, n: int, nrhs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1000)
    B = rng.standard_normal((nq, n, nrhs)) + 1j * rng.standard_normal((nq, n, nrhs))
    return B.astype(np.complex128)


def _shard(A: jax.Array, mesh: Mesh, spec: P) -> jax.Array:
    return jax.device_put(A, NamedSharding(mesh, spec))


def run_case(mesh: Mesh, nq: int, n: int, nrhs: int, seed: int) -> dict:
    A_np = _build_indefinite_hermitian(nq, n, seed)
    B_np = _build_rhs(nq, n, nrhs, seed)

    # Reference: per-q LU on a fully-replicated copy.  Distribute then
    # allgather so the values are reproducible across ranks (each rank
    # owns the same A/B for the reference solve).
    A_rep = _shard(jnp.asarray(A_np), mesh, P(None, None, None))
    B_rep = _shard(jnp.asarray(B_np), mesh, P(None, None, None))
    X_ref = jnp.linalg.solve(A_rep, B_rep)

    # cuSolverMp distributed LU on sharded inputs.
    A_d = _shard(jnp.asarray(A_np), mesh, P(None, 'x', 'y'))
    B_d = _shard(jnp.asarray(B_np), mesh, P(None, 'x', 'y'))
    X_cmp = batched_distributed_solve_lu(A_d, B_d, mesh=mesh)

    # Compute reduction-norm quantities on-device (sharded), then
    # allgather the small (nq,) results.  Avoids fetching the full X.
    A_for_resid = _shard(jnp.asarray(A_np), mesh, P(None, None, None))
    B_for_resid = _shard(jnp.asarray(B_np), mesh, P(None, None, None))
    # Bring X_cmp to fully-replicated so we can dot with full A.
    target_rep = NamedSharding(mesh, P(None, None, None))
    X_cmp_rep = jax.lax.with_sharding_constraint(
        jax.jit(lambda x: x, out_shardings=target_rep)(X_cmp), target_rep)
    X_ref_rep = jax.lax.with_sharding_constraint(X_ref, target_rep)

    AX_minus_B = jnp.einsum('qij,qjk->qik', A_for_resid, X_cmp_rep) - B_for_resid
    resid_per_q = (jnp.linalg.norm(AX_minus_B, axis=(1, 2))
                   / jnp.linalg.norm(B_for_resid, axis=(1, 2)))
    diff_per_q = (jnp.linalg.norm(X_cmp_rep - X_ref_rep, axis=(1, 2))
                  / jnp.linalg.norm(X_ref_rep, axis=(1, 2)))

    # process_allgather to gather these tiny per-q vectors to every rank
    from jax.experimental.multihost_utils import process_allgather
    resid_g = np.asarray(process_allgather(resid_per_q, tiled=False))
    diff_g = np.asarray(process_allgather(diff_per_q, tiled=False))
    # process_allgather returns shape (procs, nq); all procs agree so take row 0.
    resid = resid_g[0] if resid_g.ndim == 2 else resid_g
    diff = diff_g[0] if diff_g.ndim == 2 else diff_g

    # Conditioning + indefiniteness check (numpy-only, on rank 0).
    eigs_min = np.array([np.linalg.eigvalsh(A_np[q]).min() for q in range(nq)])
    eigs_max = np.array([np.linalg.eigvalsh(A_np[q]).max() for q in range(nq)])

    return dict(
        n=n, nrhs=nrhs, nq=nq, seed=seed,
        max_resid=float(resid.max()),
        max_diff_vs_ref=float(diff.max()),
        eigs_min=eigs_min.tolist(),
        eigs_max=eigs_max.tolist(),
        is_indefinite=bool((eigs_min < 0).any() and (eigs_max > 0).any()),
    )


_PRINT = jax.process_index() == 0
def _p(*args, **kw):
    if _PRINT:
        print(*args, **kw)


def main():
    nd = jax.device_count()
    if nd != 4:
        _p(f"FAIL: expected 4 GPUs, got {nd}")
        return 1
    # Sweep all 4-GPU mesh shapes, with progressively-larger N on 2×2.
    # N must be divisible by both Px and Py; NRHS by Py.
    sweeps = [
        (2, 2, [
            (2,  16,   8,  0),
            (4,  32,  16,  2),
            (2,  64,  32,  4),
            (1, 128,  64,  5),
            (1, 512, 256,  6),
        ]),
        (1, 4, [
            (2,  16,   8, 10),
            (4,  64,  16, 11),
            (1, 256,  64, 12),
        ]),
        (4, 1, [
            (2,  16,  16, 20),
            (4,  64,  64, 21),
            (1, 256,  64, 22),
        ]),
    ]
    _p(f"procs={jax.process_count()}, devices={nd}, dtype=C128")
    _p()
    _p(f"{'mesh':>5} {'case':>22} {'max_resid':>14} {'max_diff':>14}  indef  λmin..λmax")
    all_pass = True
    for px, py, cases in sweeps:
        mesh = _mesh(px, py)
        for nq, n, nrhs, seed in cases:
            r = run_case(mesh, nq, n, nrhs, seed)
            passed = r['max_resid'] < 1e-9 and r['max_diff_vs_ref'] < 1e-7
            all_pass = all_pass and passed
            tag = f"nq={nq:>2} N={n:>4} NRHS={nrhs:>3} s={seed}"
            eig_summary = f"{min(r['eigs_min']):+.2e} .. {max(r['eigs_max']):+.2e}"
            _p(f"{px}x{py:<3} {tag:>22} {r['max_resid']:14.3e} {r['max_diff_vs_ref']:14.3e}  "
               f"{'Y' if r['is_indefinite'] else 'n'}     {eig_summary}  "
               f"{'PASS' if passed else 'FAIL'}")

    _p()
    _p('ALL PASS' if all_pass else 'SOME FAILED')
    return 0 if all_pass else 1


if __name__ == '__main__':
    raise SystemExit(main())
