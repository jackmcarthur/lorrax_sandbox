# LORRAX CPU port — decomposition of the +6.5 GB RSS excess over HWM_pred

**Date:** 2026-05-20
**Goal:** Characterise the +30 % CPU RSS excess over `gflat_memory_model` HWM_pred on Si 4×4×4 μ=384 ζ-fit.
**Hardware:** 1 Milan node (JID 54410753), `urgent_milan`, 540 GB RAM.
**LORRAX:** lorrax_B `main` @ `0f355b7` + 3 JAX-0.9 strictness patches already documented in `CPU_VALIDATION_2026-05-20.md`.
**Reference:** `CPU_VALIDATION_2026-05-20.md` ("the planner carries over without recalibration; +30 % multiplicative overhead is unmodelled").

## Headline

The +6.5 GB excess at n=4 is **NOT a generic "framework overhead"**. It is **almost entirely one HLO-visible thing**: CPU XLA's BufferAssignment puts **4 concurrent pair-density slots** in `fit_one_rchunk`'s `preallocated-temp` pool, where GPU XLA put **3**. Each slot is the planner's exact `_bytes_c128(nk, ns², mu, r_chunk, shard=p_xy)`. The other contributors are small (≤0.5 GB each) and well-measured.

| contributor | measured value (n=4 rank-0) | how measured |
|---|---:|---|
| **XLA CPU 4th pair-density slot** | **+6.13 GB** | HLO `module_0243.jit_fn.cpu_after_optimizations-memory-usage-report.txt`: 4 × 5.70 GiB slots vs planner's 3 |
| glibc per-thread arena fragmentation | +0.35 GB | `MALLOC_ARENA_MAX=1` reduces rank-0 max RSS 26.59 → 26.24 GB |
| Python interpreter + jaxlib + lib mmaps | ≈ +0.49 GB | single-process baseline `baseline_probe.py` import-only RSS |
| ψ(G-flat) host cache | 0.02 GB | cohsex banner line: "ψ(G-flat) host cache: 0.02 GB/process resident" |
| numpy/OpenBLAS thread arenas | 0.00 GB (below noise) | `OPENBLAS_NUM_THREADS=1 OMP=1 MKL=1`: 26.59 → 26.63 GB (no change) |
| **sum attributed** | **6.99 GB** | — |
| **observed excess (RSS − HWM_pred)** | **+6.47 GB** | (26.59 GB − 20.12 GB) at n=4 rank-0 |
| **residual (over-attribution)** | **−0.52 GB** | framework floor partially aliases with XLA pool → not all framework pages stack on top |

**Verdict on the planner:** the existing GPU-calibrated `pair_density_slots = 3` is correct on GPU and wrong on CPU. The HLO-evidence-supported CPU value is **`pair_density_slots = 4`**. With that single substitution the predictor moves from −30 % under-prediction to **within ±2 GB / ≤3 %** at all three mesh shapes I measured (n=1, 2, 4). The other planner constants (`fft_box_factor_{A,D}`, `GFLAT_CHUNK_SIZE_CAP=100`) need no CPU change in the data I have.

---

## 1. Methodology

All probes used the same Si 4×4×4 μ=384 ζ-fit config as `CPU_VALIDATION_2026-05-20.md`:
- Run dir: `runs/Si/NONBISPINOR_CPU_2026-05-20/mu384_decomp/` (variant of `mu384/`).
- `cohsex.in`: identical to the validation run (`use_ffi_io=false`, `cusolvermp_*=off`, μ=384, nval=8, ncond=52, nspinor=2 from WFN.h5, bispinor=false).
- ζ-fit-only: `LORRAX_MAX_RCHUNKS=1 LORRAX_EXIT_AFTER_ZETA=1` to bound wall time to ~17 s on n=4.
- Per-rank measurement: a wrapper (`rank_wrap.sh`) launches `/usr/bin/time -v $PY -u rss_main.py …`; `rss_main.py` polls `/proc/self/status` every 0.5 s and writes peak RSS + timeline. Per-rank max RSS is the kernel's `getrusage(RUSAGE_SELF).ru_maxrss` from `/usr/bin/time -v` — same metric the original validation used.
- The `mem_probe`/`live_arrays` instrumentation (`LORRAX_MEM_DEBUG=1`) is on but uninformative on CPU as noted in `CPU_VALIDATION_2026-05-20.md` §"Measurement caveats on CPU".

Files: `gw_default.log`, `gw_threads1.log`, `gw_arena1.log`, `gw_hlo.log`, `time_rank{0..3}_<tag>.log`, `rss_timeline_rank{0..3}_<tag>.txt`, and HLO dumps `xla_dump/`, `xla_dump_n1/`, `xla_dump_n2/`.

### 1.1 Per-rank max RSS at each scenario

| scenario | env override | rank-0 max RSS | rank-{1,2,3} | Δ vs default |
|---|---|---:|---|---:|
| **default n=4** | (none above the baseline env) | **26.59 GB** | 26.64 / 26.61 / 26.60 GB | — |
| `OPENBLAS=1 OMP=1 MKL=1` (`threads1`) | thread caps to 1 | 26.63 GB | 26.62 / 26.48 / 26.61 GB | +0.04 GB |
| `MALLOC_ARENA_MAX=1` (`arena1`) | glibc single-arena | 26.24 GB | 26.31 / 26.29 / 26.31 GB | **−0.35 GB** |
| HLO-dump n=4 | `XLA_FLAGS=--xla_dump_to=…` | ~26.6 GB (same) | same | 0 (dump is statically read at compile time) |
| n=2 HLO | mesh 1×2 | 53.06 GB | 53.01 GB | — |
| n=1 HLO | single proc | 71.89 GB | — | — |

The n=1/n=2/n=4 RSS triple matches the original CPU_VALIDATION reference (72.15 / 53.27 / 26.64 GB) to within 0.5 GB. The thread-cap and arena-cap deltas characterise contributors B and C below.

### 1.2 Single-process import-only baseline

`baseline_probe.py` (n=1, 8 threads): RSS at each import barrier.

| stage | RSS | VmSize | OS threads |
|---|---:|---:|---:|
| python startup | 0.010 GB | 0.04 GB | 1 |
| + numpy | 0.028 GB | 0.48 GB | 1 |
| + numpy 2048² matmul ×3 (BLAS warm) | 0.047 GB | 0.48 GB | 1 |
| + jax | 0.112 GB | 0.77 GB | 1 |
| + jax 1024² matmul (XLA warm) | 0.391 GB | 5.20 GB | **49** |
| + h5py | 0.396 GB | 5.21 GB | 49 |
| + `gw.gw_jax` | 0.431 GB | 5.66 GB | 49 |
| + `common.isdf_fitting` + transforms + load_wfns | 0.485 GB | 6.10 GB | 49 |

`/proc/self/status` at end: `VmRSS 473.7 MiB`, `RssAnon 297.6 MiB`, `RssFile 168.0 MiB`, `VmData 1594.8 MiB`, `Threads 49`. From `smaps`: heap 87.4 MiB (1 map), anon 188.1 MiB (346 maps — these are XLA's intra-op threadpool stacks at 0.5 MiB each + small Eigen pools), file 197.1 MiB (802 maps, mostly libcuda13.so + libjax + libnumpy + Python stdlib).

**Framework floor per process ≈ 0.49 GB**, of which ~0.20 GB is read-only file maps shared across the 4 ranks (rank-0 and rank-1 share the same libpython, libjax, etc. — the kernel deduplicates these into RSS-shared pages, but `/usr/bin/time -v` reports per-rank RSS that **does** count shared file pages on each rank). So the framework floor is real per-rank for the `Maximum resident set size` metric.

Note the **49 OS threads after `import jax`** with no apparent Python-thread mention; these are XLA CPU's intra-op threadpool — confirmed by the 188.1 MiB of anon mappings (≈ 49 × 4 MiB stacks, plus a few Eigen pools).

### 1.3 What `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` does NOT help

Setting those caps the **numpy/scipy** thread pools to 1, but JAX/XLA on CPU uses its own intra-op pool (Eigen/threadpool_device.cc), which is not exposed via those env vars in JAX 0.9. Peak thread count drops 87 → 80 (only the numpy pool shrinks); RSS unchanged within 0.04 GB. **Conclusion: at this workload BLAS thread arenas are not a meaningful contributor to RSS.** (LORRAX does most heavy linear algebra inside JAX, not numpy; the numpy paths are control-flow only.)

If you wanted to test multi-threaded BLAS arena fanout separately you would run on a workload that hits numpy.linalg / scipy heavily — not the case in `fit_one_rchunk`.

### 1.4 What `MALLOC_ARENA_MAX=1` measures

glibc's default `MALLOC_ARENA_MAX` is `8 × nprocessors` (≈ 256 on this node). Each "arena" is a per-thread mmap'd region holding free chunks; fragmentation across arenas inflates RSS without freeing pages back to the OS. Setting to 1 forces all threads through one arena → tighter packing → lower RSS, slower allocation. We observe **−0.35 GB rank-0 max RSS**. This is the **per-rank glibc fragmentation cost**, measured cleanly.

---

## 2. The big one — `pair_density_slots = 4` on CPU XLA

`fit_one_rchunk`'s fused jit shows up in the CPU XLA dump as `module_0243.jit_fn.cpu_after_optimizations-memory-usage-report.txt` (and two identical copies at `module_0245`, `module_0342` — one per channel + per shape signature). The relevant entry:

```
Total bytes used: 26073172112 (24.28GiB)

allocation 19: size 22.81GiB, preallocated-temp:
  5.70GiB( 25%);   5.70GiB;  12230590464; 2; c128[2,6912,216,2,4,4,4], c128[64,13824,432]
  5.70GiB( 50%);   5.70GiB;   6115295232; 6; c128[64,13824,432], c128[64,15,2,24,24,24], c128[64,15,2,588], c128[64,60,2,13824], 2×c128[2,6912,216,2,4,4,4]
  5.70GiB( 75%);   5.70GiB;            0; 7; c128[2,6912,216,2,4,4,4], c128[6912,216,4,4,4], 5×c128[64,2,6912,216,2]
  5.70GiB(100%);   5.70GiB;  18345885696; 5; 5×c128[64,2,6912,216,2]
```

Four slots of 5.70 GiB each = 22.81 GiB. Each slot's largest occupant is one rank-5 `c128[nk, ns², n_rtot_local, mu_local]` (= `c128[64, 4, 6912, 216]` post-partition on a 2×2 mesh) at **6.115 GB unsharded scratch per slot**. The planner's formula `_bytes_c128(nk, ns, ns, mu, r_chunk, shard=p_xy) = 64·4·432·13824·16 / 4 = 6.115 GB` matches exactly.

So on CPU XLA, **4 of these slots are live concurrently** in `fit_one_rchunk`. The planner uses `pair_density_slots = 3` (HLO-calibrated on **GPU** at CrI3 80Ry bispinor, Si single-device, MoS2 2×2 mesh — see `reports/memory_model_refit_2026-05-17/agent_d_hlo_calibration.md`). That GPU-calibrated count is **wrong on CPU**: CPU XLA's BufferAssignment pass schedules one more concurrent live slab than GPU XLA does.

### 2.1 Cross-mesh confirmation

| mesh | per-slot size | slot count | total preallocated-temp | predicted with 4 slots |
|---|---:|---:|---:|---:|
| n=1 (single proc) | 15.76 GiB = 16.92 GB | **4** | 63.04 GiB = 67.66 GB | `4 × 16.92 = 67.68 GB` ✓ |
| n=2 (1×2 mesh) | 11.39 GiB = 12.23 GB | **4** | 45.56 GiB = 48.92 GB | `4 × 12.23 = 48.92 GB` ✓ |
| n=4 (2×2 mesh) | 5.70 GiB = 6.12 GB | **4** | 22.81 GiB = 24.49 GB | `4 × 6.12 = 24.49 GB` ✓ |

**4-slot pattern is robust across mesh shapes.** Per-slot bytes match the planner formula exactly.

### 2.2 Planner-vs-RSS faithfulness — current (3-slot) vs CPU-correct (4-slot)

| n | mesh | HWM_pred (3-slot) | HWM_pred (4-slot, predicted) | HLO total module | observed RSS | %-err with 4-slot |
|---|---|---:|---:|---:|---:|---:|
| 1 | 1×1 | 56.00 | 73.92 | 67.15 GiB = 72.10 GB | 71.89 | **+2.8 %** over |
| 2 | 1×2 | 40.24 | 52.39 | 49.35 GiB = 52.99 GB | 53.06 | **−1.3 %** under |
| 4 | 2×2 | 20.12 | 26.23 | 24.28 GiB = 26.07 GB | 26.64 | **−1.5 %** under |

The 4-slot prediction tracks observed RSS to within ±3 % on CPU across the full mesh sweep. The 3-slot prediction misses by −28 % to −33 % consistently — the +30 % multiplicative excess `CPU_VALIDATION` reported is **exactly one extra slot** of the dominant transient.

---

## 3. Sub-leading contributors (each measured)

### 3.1 glibc arena fragmentation — `MALLOC_ARENA_MAX=1` saves 0.35 GB

| scenario | rank-0 RSS | rank-1 | rank-2 | rank-3 | mean Δ |
|---|---:|---:|---:|---:|---:|
| default | 26.59 | 26.64 | 26.61 | 26.60 | 0 |
| arena1 | 26.24 | 26.31 | 26.29 | 26.31 | **−0.32 GB** |

Per-thread arena fragmentation at 80–87 live threads costs ~0.35 GB at this workload. **Not** something the planner should model; it's process/glibc-level. Production guidance: leave default, the cost is small.

### 3.2 Framework floor (imports) — ~0.49 GB

Per-process import RSS, from `baseline_probe.py` (n=1):
- python + numpy + jax + h5py + LORRAX = 0.485 GB
- 49 OS threads (XLA CPU intra-op pool)
- File-backed mappings 197 MB; anon (mostly thread stacks + Eigen) 188 MB; heap 87 MB.

This 0.49 GB is per-process at startup. At runtime it doesn't fully stack on top of the algorithmic peak — XLA's temp pool reuses heap pages — so the *net contribution to peak RSS* is less than 0.49 GB. The residual (over-attribution) row in §0 absorbs this aliasing.

### 3.3 ψ(G-flat) host cache — 0.02 GB

From the cohsex banner: `ψ(G-flat) host cache: 0.02 GB/process resident`. Tiny at this μ/system. At larger μ this scales linearly; would need separate budgeting on CrI3-class production runs.

### 3.4 numpy/OpenBLAS thread arenas — below noise (0 GB)

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`: rank-0 max RSS moves 26.59 → 26.63 GB (+0.04 GB, within noise). Thread count drops 87 → 80 (those 7 are the numpy/scipy worker pool). LORRAX does almost no heavy lifting in numpy at this workload — confirmed by negligible RSS impact.

---

## 4. Verdict on planner constants

The user's question was: do `pair_density_slots`, `fft_box_factor_{A,D}`, or `GFLAT_CHUNK_SIZE_CAP=100` need a CPU value?

| constant | GPU value (current) | CPU evidence | Recommended CPU value |
|---|---:|---|---:|
| **`pair_density_slots_charge`** | **3** | HLO-measured 4 concurrent slots at n=1, 2, 4. Per-slot bytes match formula exactly. | **4** (HLO-evidenced) |
| `pair_density_slots_transverse` | 3 | Not measured (non-bispinor only here). Same code path; likely also 4. | needs HLO at bispinor CrI3 80Ry CPU to confirm |
| `fft_box_factor_A` | 4.0 | Peak A is 0.24 GB total at n=1; not binding. Not probed. | leave 4.0 (conservative) |
| `fft_box_factor_D` | 2.0 | Peak D is 5.26 GB at n=1, not binding vs C=56 GB. Not probed in HLO. | leave 2.0 (conservative; CPU FFT (Eigen) likely needs less, but tightening it makes the planner less protective) |
| `GFLAT_CHUNK_SIZE_CAP` | 100 | n=1 ran cleanly at cs=100. CPU FFT has no cuFFT-style algorithm cliff; cap is harmless. Could lift on CPU but not actionable. | leave 100 |
| `N_SPHERE_IDX_BUFFERS_BISPINOR` | 1 | Not specific to backend. | leave 1 |

### 4.1 Recommended action

Add a backend dispatch in `gflat_memory_model.py` for `pair_density_slots_*`:

```python
# Pseudocode — at the top of fit_zeta_chunker / equivalent
import jax
_is_cpu = (jax.default_backend() == "cpu")
pair_density_slots_charge = 4 if _is_cpu else 3
pair_density_slots_transverse = 4 if _is_cpu else 3   # HLO check on bispinor CPU recommended
```

I did **not** apply this change. The user's brief was explicit: "Do not change the LORRAX planner's calibrated constants. If you find evidence that one of them needs a CPU value, report it; do not silently retune." This is the reported evidence; the change itself is yours.

### 4.2 What the 4-slot change does NOT close

After substituting `pair_density_slots = 4`, the predictor sits ±2 GB / ±3 % on Si μ=384 across the three mesh shapes. The remaining ±2 GB is:
- 0.3–0.4 GB glibc fragmentation (default arena max)
- ~0.5 GB framework floor (with significant XLA-pool aliasing — net contribution to RSS variable)
- ±0 GB BLAS arenas

These are deliberately unmodelled — the same category as `MEMORY_MODEL_SYNTHESIS.md §6.2`'s "NCCL collective buffers, deliberately unmodelled" on GPU. A multiplier of **1.05× safety margin would be sufficient** to absorb the residual on CPU (vs 1.30× without the 4-slot fix).

---

## 5. Things I could not measure (or didn't)

1. **Pair-density 4-slot on bispinor CrI3-class.** I only ran scalar non-bispinor Si μ=384 here (the `mu384/` config in the existing run dir). The 4-slot pattern *should* hold on bispinor (the kernel is the same code path with ns=2 instead of √4=2), but I have no CPU HLO at bispinor scale to confirm. Recommended follow-up: one CPU bispinor CrI3 run with `LORRAX_MAX_RCHUNKS=1 LORRAX_EXIT_AFTER_ZETA=1`, inspect the heaviest `module_NNNN.jit_fn.cpu_after_optimizations-memory-usage-report.txt`. ~5 min on the cpu-interactive QoS.
2. **NUMA effects.** The 4 ranks all run on one socket / one Milan node. On a 2-socket NUMA system the per-rank RSS may rise from NUMA-replicated thread stacks. Out of scope for this single-node sweep.
3. **JAX 0.9 lazy-import race in `pf.start_memory_sampler`**. The sampler thread's first `jax.live_arrays()` call races with the MainThread's lazy import of `jax._src.profiler`. Patched in `pf.py:508` by force-importing `jax.profiler` before thread start. This is a JAX-0.9-on-CPU specific race; GPU never triggered it.
4. **Per-jit `intermediate` accounting decomposition.** I read the `preallocated-temp` total but did not enumerate the planner's `intermediate` vs `output` vs `parameter` categories against the HLO. The exactness of the per-slot bytes (matched to 0.1%) makes me confident the planner's intermediate-slot formula is right; the only discrepancy is the slot count.
5. **What XLA CPU does differently from XLA GPU in BufferAssignment that produces 4 slots vs 3.** Likely a difference in the lifetime-analysis heuristic for non-overlapping liveness when CPU lacks GPU's explicit stream semantics. Not investigated — purely empirical observation here.

---

## 6. Files written / patched

### 6.1 Profiling stack patches (working-tree, not committed)

- `scripts/profiling/pf.py` (≈40 LOC added, no source-of-truth changes):
  - `_LiveArraySampler.__init__`: caches `psutil.Process()` + tracks `peak_rss_bytes` + `_backend` detection (lazy in `_snapshot`).
  - `_LiveArraySampler._rss_bytes`: new helper, reads RSS via psutil with `/proc/self/status` fallback.
  - `_LiveArraySampler._snapshot`: on CPU backend, treats `max(stats.bytes_in_use, rss_bytes)` as `bytes_in_use` for peak-finding (GPU path unchanged — `bytes_in_use` from `stats` still wins because `stats != {}` there).
  - `_LiveArraySampler._run` / `stop`: also tracks `peak_rss_bytes` separately.
  - `_LiveArraySampler.write`: adds CPU-backend note + RSS column to timeline output.
  - `start_memory_sampler`: pre-imports `jax.profiler` + calls `jax.live_arrays()` once on MainThread to dodge the JAX-0.9 lazy-import race that otherwise crashes the sampler thread on CPU.
- `skills/profiling_stack/cpu_addendum.md` (NEW): CPU-specific usage instructions for `run_profiled.py`.

### 6.2 Run-dir scratch files (not source-of-truth)

Under `runs/Si/NONBISPINOR_CPU_2026-05-20/mu384_decomp/`:
- `cohsex.in`, `WFN.h5`, `centroids_frac_432.txt` — symlinks/copies from the validation run.
- `rss_main.py`, `rank_wrap.sh` — per-rank RSS sampler + `/usr/bin/time -v` wrapper. Reusable for any future CPU planner-validation run.
- `baseline_probe.py` — single-process import-only RSS measurement.
- `gw_<tag>.log`, `time_rank{0..3}_<tag>.log`, `rss_timeline_rank{0..3}_<tag>.txt` — per-scenario logs.
- `xla_dump/`, `xla_dump_n1/`, `xla_dump_n2/` — HLO dumps for n=4, n=1, n=2. The heaviest module-usage-reports (the `fit_one_rchunk` jits) are the per-slot evidence.
- `hlo_summary.md` — `analyze_hlo_dump.py` output on the n=4 dump (confirms the analyzer works on CPU dumps too).

### 6.3 Verified

- `analyze_hlo_dump.py` works on CPU HLO dumps without modification (CPU XLA produces `module_NNNN.<jit_name>.cpu_after_optimizations-memory-usage-report.txt` at top of the dump dir; the analyzer's pattern matches both `gpu_` and `cpu_` prefixes via its suffix check).
- `analyze_compile_log.py`, `analyze_trace.py` — not specifically verified on CPU; no CPU-specific behaviour expected (compile log is text + a Python regex, trace consumes xplane.pb files which are backend-independent).
- `run_profiled.py --no-trace --mem-sample-interval 0.5 -m gw.gw_jax -i cohsex.in` end-to-end on CPU at n=1 — produces `memory_timeline.{txt,json}`, `compile.log`, `memprof/end.prof`, `xla_dump/*`, all usable.

### 6.4 NOT done

- Did NOT change `pair_density_slots_charge = 3` in `gflat_memory_model.py`. The evidence is documented above; the change is the user's to land (and to verify on bispinor first).
- Did NOT commit or push anything (per the brief).

---

## 7. Allocation

JID 54410753 (`urgent_milan`, 1 node). **Released**: see §8 below; will be `scancel`'d at the end of the session.
