"""Bisect the fit_one_rchunk compile: jit each constituent piece in
isolation at the 4GB-chooser shapes and time each compile.

Lowest-level pieces first.  Whichever one hangs past its wall-clock
budget (the launcher kills the process at 60s) is the culprit.

Shapes taken from the 4GB-chooser output for MoS2 3×3 nosym:
    kgrid=(3,3,1) → n_k = 9
    fft_grid = (24, 24, 80) → n_r = 46080
    n_rmu = 640
    n_s = 2
    n_b = 80, band_chunk = 40 → nbc = 2
    chunk_r = 23040 (= n_r/2)
    q_chunk_size = 8 (triggers the python-unrolled q-loop path)
"""
from __future__ import annotations

import os, sys, time

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

# Reuse gw_jax's distributed-init routine so the 4 ranks see 4 devices.
from gw.gw_jax import _maybe_init_jax_distributed  # noqa: E402
_maybe_init_jax_distributed()

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update("jax_enable_x64", True)

from common.load_wfns import Meta
from common import isdf_fitting
from common.load_wfns import get_sharded_wfns_rchunk_slice

# ---------------------------------------------------------------------------
# Setup — match the 4GB chooser's picks.
# ---------------------------------------------------------------------------
devs = jax.devices()
assert len(devs) >= 4, f"need 4 devices, got {len(devs)}"
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))

nk = 9
fft_grid = (24, 24, 80)
nx, ny, nz = fft_grid
n_r = nx * ny * nz       # 46080
n_rmu = 640
n_s = 2
n_b = 80
bc = 40                  # band_chunk
chunk_r = 23040          # half n_r
nq = nk                  # Γ-centered → nq = n_k

meta = Meta(
    rank=0, n_proc=mesh.size,
    b_id_0=0, b_id_1=0, b_id_2=0, b_id_3=n_b, b_id_4=n_b,
    fft_grid=fft_grid, cell_volume=1.0,
    n_rtot=n_r, n_rmu=n_rmu,
    npol=1, nfreq=1,
    nspin=n_s, nspinor=n_s, nspinor_wfnfile=n_s,
    nkx=3, nky=3, nkz=1, nk_tot=nk,
    nbnd_jax=n_b, n_rtot_jax=n_r, n_rmu_jax=n_rmu,
)
kvecs_frac = np.array(np.meshgrid(
    np.arange(3)/3, np.arange(3)/3, np.arange(1)/1, indexing="ij",
)).reshape(3, -1).T.astype(np.float64)
kvecs_frac = np.ascontiguousarray(kvecs_frac)


def _sh(spec):
    return NamedSharding(mesh, spec)


def time_compile(label: str, fn, args, in_shardings=None, out_shardings=None,
                 budget_s: float = 55.0):
    """Lower+compile `fn(*args)` under an explicit partial-jit wrapper;
    print t_lower / t_compile.  ``args`` are ShapeDtypeStruct specs."""
    kwargs = {}
    if in_shardings is not None:
        kwargs['in_shardings'] = in_shardings
    if out_shardings is not None:
        kwargs['out_shardings'] = out_shardings
    jitted = jax.jit(fn, **kwargs)
    t0 = time.perf_counter()
    lowered = jitted.lower(*args)
    t1 = time.perf_counter()
    compiled = lowered.compile()
    t2 = time.perf_counter()
    print(f"[{label:40s}] lower {t1-t0:6.2f}s  compile {t2-t1:6.2f}s",
          flush=True)
    return compiled


# ---------------------------------------------------------------------------
# Piece 1 — einsum accumulate (low level)
# ---------------------------------------------------------------------------
def piece_accumulate(P_acc, psi_L, psi_R):
    P_sh = _sh(P(None, 'x', 'y'))
    L_sh = _sh(P(None, 'x', None, None))
    R_sh = _sh(P(None, None, None, 'y'))
    P_acc = jax.lax.with_sharding_constraint(P_acc, P_sh)
    psi_L = jax.lax.with_sharding_constraint(psi_L, L_sh)
    psi_R = jax.lax.with_sharding_constraint(psi_R, R_sh)
    res = P_acc + jnp.einsum('kmns,knsv->kmv', psi_L, psi_R, optimize=True)
    return jax.lax.with_sharding_constraint(res, P_sh)

# ---------------------------------------------------------------------------
# Piece 2 — ZCT (FFT + conj + mul + FFT)
# ---------------------------------------------------------------------------
def piece_zct(P_l, P_r):
    return isdf_fitting.compute_ZCT_from_left_right_zchunk(
        P_l, P_r, (3, 3, 1), mesh)

# ---------------------------------------------------------------------------
# Piece 3 — solve_zeta fast path (q_chunk_size >= nq)
# ---------------------------------------------------------------------------
def piece_solve_fast(L_q, Z_col):
    return isdf_fitting.solve_zeta_from_L_q(L_q, Z_col, mesh, nq)

# ---------------------------------------------------------------------------
# Piece 4 — solve_zeta slow path (q_chunk_size < nq; python-unrolled loop)
# ---------------------------------------------------------------------------
def piece_solve_slow(L_q, Z_col):
    return isdf_fitting.solve_zeta_from_L_q(L_q, Z_col, mesh, 8)

# ---------------------------------------------------------------------------
# Piece 4b — same slow path but with lax.scan instead of python-for
# (user request: compile-time comparison with the py-loop version).
# Inlined here so the production function isn't touched yet.
# ---------------------------------------------------------------------------
from functools import partial
from jax.experimental.shard_map import shard_map
def piece_solve_slow_scan(L_q, Z_col):
    q_batch = 8
    nq_local = L_q.shape[0]
    nq_padded = ((nq_local + q_batch - 1) // q_batch) * q_batch  # 16 when nq=9
    pad_q = nq_padded - nq_local
    if pad_q:
        L_q = jnp.pad(L_q, ((0, pad_q), (0, 0), (0, 0)))
        Z_col = jnp.pad(Z_col, ((0, pad_q), (0, 0), (0, 0)))
    zeta = jnp.zeros_like(Z_col)
    q_starts = jnp.arange(0, nq_padded, q_batch, dtype=jnp.int32)

    L_rep_sh = _sh(P(None, None, None))
    Z_sh     = _sh(P(None, None, ('x', 'y')))

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, None, None), P(None, None, ('x', 'y'))),
             out_specs=P(None, None, ('x', 'y')))
    def _chol_batch(L_batch, Z_batch):
        def solve_single(L, Z):
            y = jax.scipy.linalg.solve_triangular(L, Z, lower=True)
            return jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
        return jax.vmap(solve_single)(L_batch, Z_batch)

    def _body(zeta_carry, q0):
        L_slice = jax.lax.dynamic_slice_in_dim(L_q, q0, q_batch, axis=0)
        Z_slice = jax.lax.dynamic_slice_in_dim(Z_col, q0, q_batch, axis=0)
        L_rep = jax.lax.with_sharding_constraint(L_slice, L_rep_sh)
        batch_result = _chol_batch(L_rep, Z_slice)
        zero_i32 = jnp.int32(0)
        zeta_new = jax.lax.dynamic_update_slice(
            zeta_carry, batch_result, (q0, zero_i32, zero_i32))
        return zeta_new, None

    zeta, _ = jax.lax.scan(_body, zeta, q_starts)
    if pad_q:
        zeta = zeta[:nq_local, :, :]
    return zeta

# ---------------------------------------------------------------------------
# Piece 5 — get_sharded_wfns_rchunk_slice (FFT + slice + 2-stage reshard)
# ---------------------------------------------------------------------------
def piece_rchunk(psi_G, r_start):
    return get_sharded_wfns_rchunk_slice(
        psi_G, meta, r_start, chunk_r, kvecs_frac, mesh, (0, bc))


# ---------------------------------------------------------------------------
# Driver — run pieces sequentially.
# ---------------------------------------------------------------------------
pair_shape = (nk, n_rmu, chunk_r)
pair_sh    = _sh(P(None, 'x', 'y'))

centroid_shape = (nk, n_rmu, bc, n_s)
centroid_sh    = _sh(P(None, 'x', None, None))

rchunk_Y_shape = (nk, bc, n_s, chunk_r)
rchunk_Y_sh    = _sh(P(None, None, None, 'y'))

Lq_shape = (nq, n_rmu, n_rmu)
Lq_sh    = _sh(P(None, 'x', 'y'))

Zcol_shape = (nq, n_rmu, chunk_r)
Zcol_sh    = _sh(P(None, None, ('x', 'y')))

psiG_shape = (nk, bc, n_s, nx, ny, nz)
psiG_sh    = _sh(P(None, ('x', 'y'), None, None, None, None))

rep_sh = _sh(P())


pieces = [
    ("accumulate (einsum)",
     piece_accumulate,
     (jax.ShapeDtypeStruct(pair_shape, jnp.complex128, sharding=pair_sh),
      jax.ShapeDtypeStruct(centroid_shape, jnp.complex128, sharding=centroid_sh),
      jax.ShapeDtypeStruct(rchunk_Y_shape, jnp.complex128, sharding=rchunk_Y_sh))),

    ("rchunk_slice (FFT+slice+reshard)",
     piece_rchunk,
     (jax.ShapeDtypeStruct(psiG_shape, jnp.complex128, sharding=psiG_sh),
      jax.ShapeDtypeStruct((), jnp.int32))),

    ("ZCT (FFT+conj+mul+FFT)",
     piece_zct,
     (jax.ShapeDtypeStruct(pair_shape, jnp.complex128, sharding=pair_sh),
      jax.ShapeDtypeStruct(pair_shape, jnp.complex128, sharding=pair_sh))),

    ("solve fast (q_chunk>=nq)",
     piece_solve_fast,
     (jax.ShapeDtypeStruct(Lq_shape, jnp.complex128, sharding=Lq_sh),
      jax.ShapeDtypeStruct(Zcol_shape, jnp.complex128, sharding=Zcol_sh))),

    ("solve slow (q_chunk<nq, pyloop)",
     piece_solve_slow,
     (jax.ShapeDtypeStruct(Lq_shape, jnp.complex128, sharding=Lq_sh),
      jax.ShapeDtypeStruct(Zcol_shape, jnp.complex128, sharding=Zcol_sh))),

    ("solve slow (q_chunk<nq, lax.scan)",
     piece_solve_slow_scan,
     (jax.ShapeDtypeStruct(Lq_shape, jnp.complex128, sharding=Lq_sh),
      jax.ShapeDtypeStruct(Zcol_shape, jnp.complex128, sharding=Zcol_sh))),
]


# ---------------------------------------------------------------------------
# REPRO — compile the FULL _make_fit_one_rchunk_kernel at the exact 4GB
# config (nbc=2, chunk_r=23040, q_chunk_size=8) that hangs in production.
# If this is fast, the hang must come from something OUTSIDE the jit body
# (runtime env, io_callback, driver loop).  If this hangs, we have a
# self-contained reproducer to iterate on.
# ---------------------------------------------------------------------------
from common.isdf_fitting import _make_fit_one_rchunk_kernel

band_chunk_ranges = ((0, bc), (bc, 2 * bc))  # nbc=2, uniform width 40
band_range_left = band_range_right = band_range_full = (0, n_b)
q_chunk_size_repro = 8  # matches the 4GB chooser pick

full_kernel = _make_fit_one_rchunk_kernel(
    mesh, meta, band_chunk_ranges,
    band_range_left, band_range_right, band_range_full,
    chunk_r, q_chunk_size_repro, kvecs_frac,
)

# psi_G_tuple: one ShapeDtypeStruct per band chunk
psiG_bc_spec = jax.ShapeDtypeStruct(
    (nk, bc, n_s, nx, ny, nz), jnp.complex128, sharding=psiG_sh)

full_specs = (
    (psiG_bc_spec, psiG_bc_spec),  # psi_bc_G_tuple
    jax.ShapeDtypeStruct((nk, n_rmu, n_b, n_s), jnp.complex128, sharding=_sh(P(None, 'x', None, None))),
    jax.ShapeDtypeStruct((nk, n_rmu, n_b, n_s), jnp.complex128, sharding=_sh(P(None, 'x', None, None))),
    jax.ShapeDtypeStruct(Lq_shape, jnp.complex128, sharding=Lq_sh),
    jax.ShapeDtypeStruct((n_b,), jnp.float64, sharding=rep_sh),
    jax.ShapeDtypeStruct((n_b,), jnp.float64, sharding=rep_sh),
    jax.ShapeDtypeStruct((), jnp.int32),
)

print("=" * 80, flush=True)
print("REPRO: full _make_fit_one_rchunk_kernel compile (4GB config)", flush=True)
print("=" * 80, flush=True)
t0 = time.perf_counter()
lowered = full_kernel.lower(*full_specs)
t1 = time.perf_counter()
print(f"  lowered in {t1-t0:.2f}s", flush=True)
compiled = lowered.compile()
t2 = time.perf_counter()
print(f"  compiled in {t2-t1:.2f}s (TOTAL {t2-t0:.2f}s)", flush=True)


print(f"mesh={mesh}, devices={len(devs)}", flush=True)
print(f"shapes: nk={nk} n_rmu={n_rmu} n_s={n_s} n_b={n_b} bc={bc} "
      f"chunk_r={chunk_r} n_r={n_r}", flush=True)
print("-" * 80, flush=True)

for label, fn, specs in pieces:
    t0 = time.perf_counter()
    try:
        time_compile(label, fn, specs)
    except Exception as e:
        print(f"[{label:40s}] FAILED: {e!r}", flush=True)
    print(f"    total wall = {time.perf_counter()-t0:.2f}s", flush=True)
