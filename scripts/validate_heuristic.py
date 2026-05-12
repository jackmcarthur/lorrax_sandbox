"""Validate the 20/80 heuristic chooser against the real
memory_analysis() peaks captured in ``fit_one_rchunk__ortho__samples.json``,
plus compare to MoS2 3×3 and Si 4×4×4 production configs where I
have runtime GPU peak measurements from earlier smoke tests.
"""
import sys
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs, choose_chunks_heuristic, describe_chunks,
)

# Config list: each has (name, sys, mesh, runtime_peak_GB | None)
configs = [
    # MoS2 3x3 nosym production — measured 3.06 GB runtime (though that's
    # a post-jit nvidia-smi — see prior measurement-bug note)
    ("MoS2 3x3 nosym 2x2", SysDims(
        kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
        n_rmu=640, n_s=2, n_b=80, n_b_sum=160, n_r=80 * 72 * 8,
    ), MeshSpec(2, 2), 28.0, 3.06),

    # Si 4x4x4 nosym production — measured 4.04 GB runtime
    ("Si 4x4x4 nosym 2x2", SysDims(
        kgrid=(4, 4, 4), fft_grid=(24, 24, 24),
        n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24 ** 3,
    ), MeshSpec(2, 2), 28.0, 4.04),

    # Si 4x4x4 on 1x4 mesh — hypothetical, no runtime measurement
    ("Si 4x4x4 nosym 1x4", SysDims(
        kgrid=(4, 4, 4), fft_grid=(24, 24, 24),
        n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24 ** 3,
    ), MeshSpec(1, 4), 28.0, None),

    # Si 4x4x4 on 4x1 mesh
    ("Si 4x4x4 nosym 4x1", SysDims(
        kgrid=(4, 4, 4), fft_grid=(24, 24, 24),
        n_rmu=480, n_s=1, n_b=60, n_b_sum=120, n_r=24 ** 3,
    ), MeshSpec(4, 1), 28.0, None),

    # Tight-budget MoS2 (4 GB instead of 28 GB) — exercises the
    # rchunk_budget-bound branch of the heuristic
    ("MoS2 3x3 nosym 2x2 @ 4GB", SysDims(
        kgrid=(3, 3, 1), fft_grid=(80, 72, 8),
        n_rmu=640, n_s=2, n_b=80, n_b_sum=160, n_r=80 * 72 * 8,
    ), MeshSpec(2, 2), 4.0, None),

    # Hypothetical 10x10x10 Si with larger μ — extrapolation test
    ("Si 10x10x10 nosym 4x4", SysDims(
        kgrid=(10, 10, 10), fft_grid=(36, 36, 36),
        n_rmu=2400, n_s=1, n_b=260, n_b_sum=520, n_r=36 ** 3,
    ), MeshSpec(4, 4), 40.0, None),

    # Same Si 10x10x10 on a smaller 2x2 mesh — much tighter
    ("Si 10x10x10 nosym 2x2", SysDims(
        kgrid=(10, 10, 10), fft_grid=(36, 36, 36),
        n_rmu=2400, n_s=1, n_b=260, n_b_sum=520, n_r=36 ** 3,
    ), MeshSpec(2, 2), 40.0, None),
]

print(f"{'config':<30s} {'budget':>7s} {'cr':>6s} {'bc':>4s} {'kc':>4s} "
      f"{'pred_GB':>8s} {'runtime_GB':>10s} {'util':>5s}")
print("-" * 90)
for name, sys_, mesh, budget_gb, runtime_gb in configs:
    try:
        choice = choose_chunks_heuristic(
            sys_, mesh, budget_bytes=budget_gb * 1e9 * 0.97,
        )
        rt = f"{runtime_gb:.2f}" if runtime_gb is not None else "-"
        util = 100 * choice.peak_bytes / (budget_gb * 1e9 * 0.97)
        print(f"{name:<30s} {budget_gb:>5.0f}GB  {choice.chunk_r:>5} "
              f"{choice.band_chunk:>4} {choice.k_chunk:>4} "
              f"{choice.peak_bytes/1e9:>7.2f} {rt:>10s} {util:>4.0f}%")
        print(f"  note: {choice.note}")
    except Exception as e:
        print(f"{name:<30s} FAILED: {e}")
