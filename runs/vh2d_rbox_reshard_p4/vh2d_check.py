"""Certify the 2-D band-sharded <mk|V_H|nk> against the local k-parallel plan.

Same k, same V(r), same psi -- the ONLY difference is the work split, so any
disagreement is the sharding.  Owner tolerance is 1e-12 relative; the
contraction is deliberately reassociated by the sharding, so bit-identity is
neither expected nor demanded (see the numerical-tolerance ruling).

Also reports the per-rank peak RSS of each plan, because the headline claim of
the distributed plan is not speed but that no rank ever materialises the
full-band FFT box (wall W1) or the (nb,nb) tile (wall W2).
"""
import os
import sys
import time

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,   # noqa: E402
                                resolve_mesh)
from psp.get_DFT_mtxels import compute_local_V_k, compute_local_V_k_2d  # noqa: E402

NB = int(os.environ.get("VH2D_NB", "32"))
NS = int(os.environ.get("VH2D_NS", "2"))
GRID = tuple(int(v) for v in os.environ.get("VH2D_GRID", "12,12,20").split(","))
NGK = int(os.environ.get("VH2D_NGK", "97"))
NPAD = int(os.environ.get("VH2D_NPAD", "3"))       # D10 fixed-shape pad rows


def vmhwm_gib():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / (1024.0 * 1024.0)
    return float("nan")


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    nx, ny, nz = GRID
    ngkmax = NGK + NPAD

    p0(f"[vh2d] world={world} mesh=({px},{py}) nb={NB} ns={NS} grid={GRID} "
       f"ngk={NGK} pad={NPAD} (nb%px={NB % px}, nb%py={NB % py})")

    # Deterministic inputs, identical on every rank.
    rng = np.random.default_rng(12345)
    gv = np.zeros((ngkmax, 3), dtype=np.int32)
    cells = rng.choice(nx * ny * nz, size=NGK, replace=False)
    for i, c in enumerate(cells):
        gv[i] = [c // (ny * nz), (c // nz) % ny, c % nz]
    g_mask = np.zeros(ngkmax, dtype=np.float64)
    g_mask[:NGK] = 1.0
    g_index = np.full((1, nx, ny, nz), ngkmax, dtype=np.int32)
    for i in range(NGK):
        g_index[0, gv[i, 0], gv[i, 1], gv[i, 2]] = i

    psi = (rng.standard_normal((1, NB, NS, ngkmax))
           + 1j * rng.standard_normal((1, NB, NS, ngkmax))).astype(np.complex128)
    psi[..., NGK:] = 0.0
    V_r = rng.standard_normal(GRID)
    volume = 137.035

    # ---- local plan: needs the FULL FFT box (this IS wall W1) -------------
    box = np.zeros((NB, NS, nx, ny, nz), dtype=np.complex128)
    for i in range(NGK):
        box[:, :, gv[i, 0], gv[i, 1], gv[i, 2]] = psi[0, :, :, i]
    t0 = time.time()
    V_local = np.asarray(compute_local_V_k(
        jnp.asarray(box), jnp.asarray(gv), jnp.asarray(V_r), volume,
        g_mask=jnp.asarray(g_mask)))
    t_local = time.time() - t0
    hwm_local = vmhwm_gib()

    # ---- distributed plan: G-sphere in, sharded (nb,nb) out --------------
    psi_j = jax.device_put(
        jnp.asarray(psi), NamedSharding(mesh, P(None, ('x', 'y'), None, None)))
    # warm-up (compile) before timing
    compute_local_V_k_2d(psi_j, psi_j, g_index, gv, GRID, jnp.asarray(V_r),
                         volume, mesh=mesh, g_mask=g_mask).block_until_ready()
    t0 = time.time()
    V_dist_sharded = compute_local_V_k_2d(
        psi_j, psi_j, g_index, gv, GRID, jnp.asarray(V_r), volume,
        mesh=mesh, g_mask=g_mask)
    V_dist_sharded.block_until_ready()
    t_dist = time.time() - t0
    hwm_dist = vmhwm_gib()

    # ---- OWNER'S SCHEME: corner sentinel, NO mask ------------------------
    # Pad the G table with a Miller index at the box corner (nx//2, ny//2,
    # nz//2) instead of (0,0,0).  It can never intersect the G-sphere, so the
    # question is whether the mask becomes unnecessary -- and whether the
    # result is BIT-IDENTICAL to the masked run, not merely close.
    #
    # Note what does and does not make it inert.  phi_G at the sentinel is NOT
    # zero: multiplying by V(r) in real space spreads support over the whole
    # box.  What kills the pad contribution is the M SIDE: psi_m is the stored
    # sphere, whose pad coefficients are exact zeros, so 0 * anything = 0.
    gv_corner = gv.copy()
    gv_corner[NGK:] = [nx // 2, ny // 2, nz // 2]
    g_index_corner = np.full((1, nx, ny, nz), ngkmax, dtype=np.int32)
    for i in range(NGK):
        g_index_corner[0, gv[i, 0], gv[i, 1], gv[i, 2]] = i

    V_corner = compute_local_V_k_2d(
        psi_j, psi_j, g_index_corner, gv_corner, GRID, jnp.asarray(V_r),
        volume, mesh=mesh, g_mask=None)
    V_corner.block_until_ready()

    # Negative control: SAME thing but pad rows left at (0,0,0) = Gamma.  If
    # this also matched, the test would prove nothing about the sentinel.
    try:
        V_gamma = compute_local_V_k_2d(
            psi_j, psi_j, g_index, gv, GRID, jnp.asarray(V_r), volume,
            mesh=mesh, g_mask=None)
        V_gamma.block_until_ready()
        gamma_refused = False
    except ValueError as exc:
        gamma_refused = True
        p0(f"[vh2d] (0,0,0)-pad + no mask REFUSED as designed: "
           f"{str(exc).splitlines()[0][:90]}")

    def shard_max_delta(a, b):
        d = 0.0
        exact = True
        for sa, sb in zip(a.addressable_shards, b.addressable_shards):
            xa, xb = np.asarray(sa.data), np.asarray(sb.data)
            d = max(d, float(np.abs(xa - xb).max()))
            exact = exact and bool(np.array_equal(xa, xb))
        return d, exact

    d_c, exact_c = shard_max_delta(V_dist_sharded, V_corner)
    p0(f"[vh2d] corner-sentinel, NO mask   vs masked: max|delta| = {d_c:.3e}"
       f"   BIT-IDENTICAL = {exact_c}")
    if not gamma_refused:
        d_g, exact_g = shard_max_delta(V_dist_sharded, V_gamma)
        p0(f"[vh2d] (0,0,0)-pad,      NO mask vs masked: max|delta| = "
           f"{d_g:.3e}   BIT-IDENTICAL = {exact_g}   <- control, expect WRONG")

    # Compare each rank's OWN shard against the matching slice of the local
    # plan.  A global device_get is not merely inconvenient in a
    # multi-process run -- gathering the very tile the design exists to avoid
    # would defeat the point, and every rank checking its own block is the
    # stronger test anyway (a plan that put the right answer on the wrong
    # rank would pass a rank-0-only check).
    spec = V_dist_sharded.sharding.spec
    shards = [tuple(s.data.shape) for s in V_dist_sharded.addressable_shards]
    d = 0.0
    for sh in V_dist_sharded.addressable_shards:
        blk = np.asarray(sh.data)[0]                 # (nb_x, nb_y)
        idx = sh.index                                # (slice_k, slice_m, slice_n)
        ref = V_local[idx[1], idx[2]]
        if blk.shape != ref.shape:
            p0(f"[vh2d] SHARD SHAPE MISMATCH {blk.shape} vs {ref.shape}")
            d = float("inf")
            break
        d = max(d, float(np.abs(blk - ref).max()))
    s = float(np.abs(V_local).max()) or 1.0

    # Every rank must agree; a rank-0-only verdict would miss a single bad block.
    from common.collectives import psum_replicate
    bad = float(np.asarray(psum_replicate(
        np.asarray([0.0 if (d / s) <= 1.0e-12 else 1.0]), mesh))[0]
    ) if world > 1 else (0.0 if (d / s) <= 1.0e-12 else 1.0)
    ok = (bad == 0)

    p0(f"[vh2d] out spec={spec} local shard shapes={shards[:2]} "
       f"(full tile would be {(NB, NB)})")
    p0(f"[vh2d] ranks disagreeing = {int(bad)} of {world}")
    p0(f"[vh2d] max|delta| = {d:.3e}   relative = {d / s:.3e}   "
       f"scale = {s:.3e}")
    p0(f"[vh2d] wall  local {t_local:.3f} s | distributed {t_dist:.3f} s")
    print(f"[vh2d rank={rank:03d}] VmHWM after local {hwm_local:.3f} GiB, "
          f"after distributed {hwm_dist:.3f} GiB", flush=True)
    p0(f"[vh2d] VERDICT: {'PASS' if ok else 'FAIL'} "
       f"(owner tolerance 1e-12 relative)")
    return 0 if ok else 1


if __name__ == "__main__":
    finalize_process(main())
