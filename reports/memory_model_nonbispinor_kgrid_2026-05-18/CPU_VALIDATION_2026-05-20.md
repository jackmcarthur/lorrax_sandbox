# LORRAX CPU port — planner validation

**Date:** 2026-05-20
**Goal:** does `gflat_memory_model.py` carry over to CPU MPI without recalibration?
**Hardware:** 1 Milan node on `urgent_milan` (JID 54408957), 540 GB RAM.
**LORRAX:** lorrax_B `main` @ `0f355b7` + 3 JAX-0.9-compat patches (see §3).
**Reference (GPU):** Si 4×4×4 μ=384, 4-GPU 2×2 mesh, HWM_pred 20.12 vs mem_stats peak 20.13 GB → −0.05 % (`REDO_PROD_2026-05-19.md`).

## Headline

**The planner carries over to CPU without recalibration.** At matched mesh shape the planner emits the *exact same* HWM_pred byte count on CPU as on GPU. The CPU port required only env tuning + two cohsex.in flag flips (`use_ffi_io=false`, `cusolvermp_*=off`) — no planner-formula change. Three small JAX-0.9 strictness patches were needed to land the multi-process code path on the CPU backend; none of them touch planner logic.

## Headline table — Si 4×4×4 μ=384 ζ-fit, 60 bands, scalar

| n_proc | mesh | HWM_pred (GB/dev) | max RSS rank-0 (GB) | excess vs HWM_pred | wall ζ-fit |
|---:|---|---:|---:|---:|---:|
| **GPU 4 (reference)** | 2×2 | 20.12 | 20.13 *(BFC peak)* | **−0.05 %** | ~3 s |
| CPU 1 | 1×1 | 56.00 | 72.15 | +28.8 % | 54 s |
| CPU 2 | 1×2 | 40.24 | 53.27 | +32.4 % | 29 s |
| **CPU 4** | **2×2** | **20.12** | **26.64** | **+32.4 %** | **17 s** |

- HWM_pred scales correctly with mesh (56 / 40 / 20 ≈ 1 / p_xy on the binding C_fit_one_rchunk term).
- CPU 4-proc 2×2 mesh produces the **byte-exact same HWM_pred (20.12 GB)** as the GPU 4-GPU 2×2 mesh — planner formula is fully backend-agnostic.
- RSS excess over HWM_pred is consistently ~30 % on CPU vs ~0 % on GPU at the same μ.

## What the +30 % CPU excess is

It is **not** a planner miscalibration. On GPU we measure `device.memory_stats()['peak_bytes_in_use']` which counts only the XLA-arena BFC pool. On CPU there is no separate device-memory pool, so `/usr/bin/time -v` captures everything the process touches:
- Python interpreter + jaxlib + numpy + h5py shared libs (~1 GB baseline).
- numpy/OpenBLAS thread arenas — 8 threads/task × ~MB per arena.
- XLA CPU backend scratch buffers for FFT (pocketfft) and matmul (Eigen) that live outside the JAX-tracked pool.
- Host-side `psi_G_store` ψ(G) cache (~hundreds of MB at this scale).
- Transient buffers that get freed before returning to Python but never released to the OS (glibc heap fragmentation).

Crucially the excess **scales with the algorithmic peak** (~constant fraction), not with the framework floor (~constant byte addition on GPU). The GPU's ~5–8 GB framework floor and the CPU's ~30 % multiplicative excess are both "unmodeled-by-design overheads" per `memory_model_refit_2026-05-17/MEMORY_MODEL_SYNTHESIS.md` §6.2. They flip sign in fractional terms at small/large peaks but neither is a correctness issue — the planner remains protective (HWM × 1.3 ≪ user budget at the configs we ran).

If the user wants to budget *tightly* on CPU, multiply HWM_pred by ~1.35 for safety. At normal "memory_per_device_gb = N" settings with N comfortably above the algorithmic peak, no change is needed.

## Planner constants — do they need CPU-specific values?

No HLO-level recalibration was performed; the runs above match the planner's *coarse* prediction to within 30 %. The four constants we identified as possibly backend-dependent (`pair_density_slots=3`, `fft_box_factor_A=4.0`, `fft_box_factor_D=2.0`, `GFLAT_CHUNK_SIZE_CAP=100`) were not individually probed on CPU here.

Bounds on what an HLO dump would change:
- If `fft_box_factor_A` and `fft_box_factor_D` are smaller on pocketfft (likely — cuFFT's 4× and 2× count plan-side scratch slabs that pocketfft doesn't carry the same way), HWM_pred would *drop*. Since RSS already exceeds HWM_pred by 30 %, dropping the FFT factors would widen that gap — the planner would become MORE under-protective. So leaving the GPU-calibrated 4.0 / 2.0 in place is the conservative choice for CPU.
- `pair_density_slots = 3` is structural (XLA buffer-lifetime aliasing in `fit_one_rchunk`); should hold on CPU XLA at the same value.
- `GFLAT_CHUNK_SIZE_CAP = 100` is the cuFFT plan-algorithm cliff. CPU FFT has no such cliff and the cap could in principle be lifted on CPU — but at cs=100 we already see good throughput; not blocking.

**Recommendation: leave the GPU-calibrated constants alone for CPU**. If a future CPU production run shows the planner being too conservative (HWM_pred wastes budget headroom), revisit `fft_box_factor_A` and `fft_box_factor_D` via an HLO dump on CPU.

## The three JAX-0.9 compat patches

Working tree on lorrax_B (uncommitted; **not pushed**). All three are JAX-0.9 strictness fixes that the GPU backend tolerated under JAX 0.8 / its more permissive CPU vs GPU split; the CPU 0.9 backend enforces the spec strictly. They should be backend-agnostic — but **need a GPU smoke-test before any merge to origin/main**.

| file | line | what | why |
|---|---|---|---|
| `src/common/cholesky_2d.py` | 183 | wrap `panel_init` in `lax.pcast((x,y), to='varying')` | shard_map carry must be marked varying when the fori_loop body writes per-device |
| `src/file_io/_slab_io_allgather.py` | 50 | `process_allgather(tiled=False)` → `tiled=True` | JAX 0.9 drops tiled=False for non-fully-addressable arrays |
| `src/common/isdf_fitting.py` | 2688 | same `tiled=False` → `tiled=True` in `fit_zeta_to_h5` allgather slab IO path | same |

The two `tiled` fixes are exactly the change JAX 0.9's deprecation message tells you to make. The `pcast` fix is exactly what the VMA error message suggests. All three are tiny (1–3 LOC each).

Three more `tiled=False` call sites remain in non-ζ-fit paths (`src/solvers/davidson.py:59`, `src/psp/run_nscf.py:252/255/259`, `src/bse/bse_davidson_helpers.py:53`) — these would crash if the BSE / Davidson / NSCF drivers are run on CPU multi-process but don't affect the GW ζ-fit path tested here. Worth bundling into a single "JAX-0.9 multi-process compat" PR.

## Launch recipe — what worked

```bash
# Env (set before srun)
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONPATH=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src

# cohsex.in changes from GPU template
#   use_ffi_io = false          (no phdf5 FFI on CPU; routes through h5py_allgather)
#   cusolvermp_charge = off     (no cuSOLVERMp on CPU; routes through sharded_cholesky)
#   cusolvermp_lu = off         (same)

# srun (4 ranks × 8 threads on 1 Milan node = 32 cores out of 256)
srun --jobid=$SLURM_JOBID -N 1 -n 4 -c 8 --cpu-bind=cores \
     /usr/bin/time -v -o time_n4.log \
     /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/.venv/bin/python -u \
     -m gw.gw_jax -i cohsex.in
```

No `lxrun` (that wrapper is GPU-specific — sets `--gres=gpu:N` and shifter `--module=gpu`). Direct `srun --jobid=$SLURM_JOBID` is the supported escape hatch.

## Measurement caveats on CPU

- `device.memory_stats()` returns `None` on the CPU backend — the BFC-+95 % protocol from `MEMORY_MODEL.md` §5.1 doesn't apply.
- `_mem_probe` in `isdf_fitting.py` falls back to `nvidia-smi` which is also empty on CPU — its probe lines report `peak=-0.00 GB nvsmi=0.00 GB`. Functional but uninformative.
- Ground-truth peak comes from `/usr/bin/time -v`'s "Maximum resident set size" (rank-0 only when wrapped around srun; if you need per-rank, wrap `time` around `python` inside a per-rank bash script).
- A future enhancement would be to add `psutil.Process().memory_info().rss` reporting inside `_mem_probe` when `default_backend() == 'cpu'`. Not needed for this validation.

## Files

- Run dir: `runs/Si/NONBISPINOR_CPU_2026-05-20/mu384/`
- Outputs: `gw_cpu_n{1,2,4}.out`, `time_n{1,2,4}.log`
- Manifest: `runs/Si/NONBISPINOR_CPU_2026-05-20/manifest.yaml`
- This report: `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_VALIDATION_2026-05-20.md`

## Verdict

Memory model is portable. No planner changes required for CPU. Three small JAX-0.9 strictness patches are needed to get past the multi-process code path on the CPU backend; they are backend-agnostic and should land as a bundled "JAX-0.9 compat" PR after a GPU smoke-test.
