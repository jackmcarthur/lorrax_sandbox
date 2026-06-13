# Profiling stack — CPU addendum

The main `SKILL.md` was written for the GPU path (`lxrun` + Shifter + nvidia-smi
+ `device.memory_stats()`). On the CPU backend most of the stack works but a few
specifics differ. This addendum tells you what changes and what to look at
instead.

## Launch recipe — CPU

`lxrun` is GPU-specific (sets `--gres=gpu:N` and Shifter `--module=gpu`). On a
CPU interactive node use raw `srun` after `salloc`:

```bash
# Get the alloc (no lxalloc on cpu)
salloc --nodes=1 --qos=interactive --constraint=cpu --time=02:00:00 \
       --account=m2651 -J "lx-alloc-$USER" bash -c "sleep 100000" &
# Wait for RUNNING in squeue, then:
export SLURM_JOBID=<jid>

# Env
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONPATH=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src
export PY=/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/.venv/bin/python

# Single-rank profiled run
srun --jobid=$SLURM_JOBID -N 1 -n 1 -c 8 --cpu-bind=cores \
     --export=ALL,PY=$PY \
     $PY -u /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
        --out profile --no-trace \
        -m gw.gw_jax -i cohsex.in

# Multi-rank
srun --jobid=$SLURM_JOBID -N 1 -n 4 -c 8 --cpu-bind=cores \
     --export=ALL,PY=$PY \
     $PY -u /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
        --out profile --no-trace \
        -m gw.gw_jax -i cohsex.in
```

`--no-trace` is recommended on CPU: the jax.profiler trace produces an
`xplane.pb` that is structurally fine, but the cpu_after_optimizations HLO
dumps already cover the per-jit memory information you usually want, and the
trace adds overhead with little payoff on CPU.

If running ζ-fit only (the typical planner-validation case), set
`LORRAX_MAX_RCHUNKS=1 LORRAX_EXIT_AFTER_ZETA=1` before srun.

## What's empty on CPU and what to look at instead

The CPU backend exposes a different and smaller set of runtime memory
introspection APIs. Empty fields are not failures — they're just unavailable.

| field | what you see on CPU | what to use instead |
|---|---|---|
| `jax.devices()[0].memory_stats()` | returns `None` | process RSS via `psutil` / `/proc/self/status` |
| `nvidia-smi` (e.g. in `_mem_probe`) | returns 0 (no GPUs) | per-rank `/usr/bin/time -v` Maximum resident set size |
| `XLA_PYTHON_CLIENT_ALLOCATOR=platform / default` | irrelevant — CPU XLA uses libc malloc directly | the only allocator-relevant env is `MALLOC_ARENA_MAX` (glibc) |
| `device.peak_bytes_in_use` | 0 | `peak_rss_bytes` field in `memory_timeline.txt` (sampler-tracked) |

The patched `pf.py:_LiveArraySampler`:
- detects backend on first sample;
- when backend is CPU, treats `max(bytes_in_use, rss_bytes)` as `bytes_in_use`
  so the peak-finder still works;
- always also records `rss_bytes` as a separate column in
  `memory_timeline.txt` so you can see both signals on both backends.

The output `memory_timeline.txt` on CPU includes a `backend=cpu` note in the
peak block and an explanatory tail.

## HLO dump — works on CPU

CPU XLA produces:
- `xla_dump/module_NNNN.<jit_name>.cpu_after_optimizations.txt`
- `xla_dump/module_NNNN.<jit_name>.cpu_after_optimizations-memory-usage-report.txt`
- `xla_dump/module_NNNN.<jit_name>.cpu_after_optimizations-buffer-assignment.txt`
- `xla_dump/jit_<name>_NNN/{module.mlir,compile_options.pb,topology.pb}` (per-jit MLIR; pre-XLA)

The naming is `cpu_after_optimizations` (vs GPU's `gpu_after_optimizations`),
flat in the `xla_dump/` root (vs GPU's per-jit subdirs).
`scripts/profiling/analyze_hlo_dump.py` matches both via a suffix check and
works unmodified on CPU dumps:

```bash
$PY -u scripts/profiling/analyze_hlo_dump.py <my_run_dir>/profile
```

Produces the same `hlo_summary.md`, `memory_details.txt`, etc., with peaks and
allocations in CPU XLA's accounting.

## What CPU XLA accounts differently from GPU XLA

Heads up on numbers when porting GPU-calibrated planner constants to CPU:

1. **`pair_density_slots`** — CPU XLA places **4 concurrent slots** in
   `fit_one_rchunk`'s `preallocated-temp` where GPU XLA places **3**. Per-slot
   bytes match the planner formula exactly. Confirmed across n=1, n=2, n=4
   mesh shapes on Si μ=384 (see
   `reports/memory_model_nonbispinor_kgrid_2026-05-18/CPU_OVERHEAD_DECOMP_2026-05-20.md`).
   If you're auditing per-jit memory on CPU expect total `preallocated-temp`
   to exceed the planner's predicted slot count × per-slot bytes by exactly
   one slot.

2. **`fft_box_factor_{A, D}`** — CPU XLA's FFT (Eigen pocketfft) does not have
   cuFFT's plan-side scratch slabs. The GPU-calibrated 4.0 / 2.0 factors are
   conservative (over-estimates) on CPU. Leave them; tightening them makes the
   planner less protective.

3. **`GFLAT_CHUNK_SIZE_CAP = 100`** — empirically chosen to dodge a cuFFT plan
   algorithm cliff on GPU. CPU FFT has no such cliff; the cap is purely
   conservative on CPU.

4. **No `device.memory_stats()` peak** — there's no XLA-arena equivalent to
   read peak from. The OOM-relevant metric is per-rank max RSS from
   `/usr/bin/time -v`. The 4 ranks all share the same node memory; budget
   accordingly.

## Per-rank max RSS — the canonical CPU OOM metric

Wrap each rank's python call with `/usr/bin/time -v` so the kernel records
`Maximum resident set size` independently per process:

```bash
# rank_wrap.sh
#!/bin/bash
set -e
RANK=${SLURM_PROCID:-0}
TAG=${RSS_TAG:-default}
exec /usr/bin/time -v -o "time_rank${RANK}_${TAG}.log" "$PY" -u "$@"

# launch
srun --jobid=$SLURM_JOBID -N 1 -n 4 -c 8 --cpu-bind=cores \
     --export=ALL,PY=$PY,RSS_TAG=tag \
     ./rank_wrap.sh -m gw.gw_jax -i cohsex.in
```

Then `grep 'Maximum resident' time_rank*_tag.log` gives per-rank RSS in KiB.
This is the only metric that captures the full process footprint (Python +
XLA + glibc fragmentation + thread stacks).

## Don't

- Don't use `lxrun` on CPU nodes — it sets GPU-specific flags and will fail
  or run in surprising configurations.
- Don't rely on `device.memory_stats()` on CPU.
- Don't expect HLO `.txt` files inside per-jit subdirs — they're flat at the
  top of `xla_dump/`.
- Don't read the `bytes_in_use` column in `memory_timeline.txt` as JAX-arena
  bytes when the backend is CPU; on CPU it's whichever of `stats.bytes_in_use`
  and `rss_bytes` is larger (the patched sampler's behaviour for peak-finding).
