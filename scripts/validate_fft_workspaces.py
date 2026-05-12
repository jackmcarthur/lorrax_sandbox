"""T1 + T2: ground-truth workspace measurements for wfn FFT and ZCT.

Compares the memory model's fudges to AOT-measured workspace of the
isolated kernels:
  T1 — wfn FFT: how much does XLA allocate for `get_sharded_wfns_
       rchunk_slice`'s local_ifftn over (nk, bc, ns, 24³), bc-sharded?
  T2 — ZCT:     how much does `compute_ZCT_from_left_right_zchunk`
       allocate on top of the 2×P_l/P_r args and 1×Z_q output?

T1 measurement gets compared to query_fft_peak_bytes to see if the
helper is accurate for this shape class.
T2 measurement gets compared to ZCT_FFT_WS=3 · α_pair · cr to see if
the fudge is close.  If not, replace with the measured coefficient.
"""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.gw_jax import _maybe_init_jax_distributed
_maybe_init_jax_distributed()

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))


def aot_temp(jit_fn, *specs):
    """AOT-compile + return memory_analysis as a dict in GB."""
    compiled = jit_fn.lower(*specs).compile(
        compiler_options={"xla_gpu_memory_limit_slop_factor": 10000})
    m = compiled.memory_analysis()
    return dict(
        arg=m.argument_size_in_bytes / 1e9,
        temp=m.temp_size_in_bytes / 1e9,
        out=m.output_size_in_bytes / 1e9,
        alias=m.alias_size_in_bytes / 1e9,
    )


# --- T1: wfn FFT over (nk, bc, ns, 24, 24, 24) sharded on bands ---
from common.fft_helpers import (
    make_jittable_local_fftn_3d,
    query_fft_peak_bytes,
)

def t1_wfn_fft(nk, bc, ns, fft_grid):
    """Measure what a STANDALONE wfn-shape FFT costs, compare with helper."""
    nx, ny, nz = fft_grid
    spec = P(None, ('x', 'y'), None, None, None, None)
    sh = NamedSharding(mesh, spec)
    sds = jax.ShapeDtypeStruct(
        (nk, bc, ns, nx, ny, nz), jnp.complex128, sharding=sh)

    fftn = make_jittable_local_fftn_3d(
        mesh, spec, spec, axes=(-3, -2, -1), norm=None)
    jit_fft = jax.jit(fftn, out_shardings=sh)
    m = aot_temp(jit_fft, sds)
    helper_peak_gb = query_fft_peak_bytes(
        input_shape=(nk, bc, ns, nx, ny, nz),
        fft_axes=(-3, -2, -1), sharding=sh) / 1e9
    return m, helper_peak_gb


# --- T2: ZCT over (nk, μ, cr) sharded P(None, 'x', 'y') ---
# Simulate compute_ZCT_from_left_right_zchunk as a standalone jit —
# takes P_l, P_r; returns Z_q of same shape.  What's the temp cost?
from common.isdf_fitting import compute_ZCT_from_left_right_zchunk

def t2_zct(nk, mu, cr, kgrid):
    """Measure standalone ZCT kernel's temp vs the α_pair · cr fudge."""
    spec = P(None, 'x', 'y')
    sh = NamedSharding(mesh, spec)
    P_l_sds = jax.ShapeDtypeStruct((nk, mu, cr), jnp.complex128, sharding=sh)
    P_r_sds = jax.ShapeDtypeStruct((nk, mu, cr), jnp.complex128, sharding=sh)

    @jax.jit
    def zct_standalone(P_l, P_r):
        return compute_ZCT_from_left_right_zchunk(P_l, P_r, kgrid, mesh)

    m = aot_temp(zct_standalone, P_l_sds, P_r_sds)
    # α_pair per device (bytes):
    alpha_pair_bytes = 16 * nk * mu / 4  # /P = /4 for our 2x2 mesh
    # Fudge prediction: (4 + ZCT_FFT_WS=3) · α_pair · cr, but
    # arg = 2 · α_pair · cr (P_l + P_r) and out = α_pair · cr (Z_q),
    # so "temp on top" fudge = ZCT_FFT_WS · α_pair · cr + (4 - 2 - 1) · α_pair · cr
    #                        = (ZCT_FFT_WS + 1) · α_pair · cr
    #                        = 4 · α_pair · cr
    fudge_temp_gb = 4 * alpha_pair_bytes * cr / 1e9
    return m, fudge_temp_gb, alpha_pair_bytes * cr / 1e9


if jax.process_index() == 0:
    print(f"\n=== T1 — wfn FFT workspace ===")
    print(f"{'nk':>5}{'bc':>4}{'ns':>3}{'fft_grid':>12}  "
          f"{'arg':>6}{'temp':>6}{'out':>6}  {'helper':>7}  {'total':>7}  {'hlp/tot':>7}")
    for (nk, bc, ns, fft_grid) in [
        (9, 80, 2, (24, 24, 80)),   # MoS2-scale
        (1000, 4, 1, (24, 24, 24)), # Si 10³ tiny bc
        (1000, 8, 1, (24, 24, 24)), # Si 10³ nominal
        (1000, 20, 1, (24, 24, 24)),
        (1000, 60, 1, (24, 24, 24)),# Si 10³ full nb
    ]:
        m, helper = t1_wfn_fft(nk, bc, ns, fft_grid)
        total = m['arg'] + m['temp'] + m['out'] - m['alias']
        print(f"{nk:>5}{bc:>4}{ns:>3}{str(fft_grid):>12}  "
              f"{m['arg']:>6.2f}{m['temp']:>6.2f}{m['out']:>6.2f}  "
              f"{helper:>7.2f}  {total:>7.2f}  {helper/max(total,1e-9):>7.2f}")

    print(f"\n=== T2 — ZCT standalone kernel workspace ===")
    print(f"{'nk':>5}{'mu':>5}{'cr':>7}{'kgrid':>12}  "
          f"{'arg':>6}{'temp':>6}{'out':>6}  {'fudge':>7}  "
          f"{'ratio':>7}  {'pair_gb':>8}")
    for (nk, mu, cr, kgrid) in [
        (9, 640, 23040, (3, 3, 1)),      # MoS2 mid-cr
        (9, 640, 46080, (3, 3, 1)),      # MoS2 max-cr
        (1000, 480, 192, (10, 10, 10)),  # Si 10³ small cr (like our earlier HLO dump)
        (1000, 480, 440, (10, 10, 10)),  # Si 10³ 8GB chooser pick
        (1000, 480, 1880, (10, 10, 10)), # Si 10³ 28GB chooser pick
    ]:
        m, fudge, alpha_cr = t2_zct(nk, mu, cr, kgrid)
        ratio = m['temp'] / max(fudge, 1e-9)
        print(f"{nk:>5}{mu:>5}{cr:>7}{str(kgrid):>12}  "
              f"{m['arg']:>6.2f}{m['temp']:>6.2f}{m['out']:>6.2f}  "
              f"{fudge:>7.2f}  {ratio:>7.2f}  {alpha_cr:>8.3f}")
    print("\n  (ratio = temp / 4·α_pair·cr — shows how much of the model's "
          "ZCT-workspace fudge is real temp.  1.0 ≈ perfect; <1 ≈ over-fudging.)")
