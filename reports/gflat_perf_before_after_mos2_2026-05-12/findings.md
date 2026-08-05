# Profiling findings — MoS2 3×3 bispinor, 8 r-chunks (2026-05-12)

**Setup:** `runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12/`, 1×4 GPUs, 2×2 mesh,
locked `r_chunk_size = 5760` (8 chunks per channel).
**Profile artifacts:** `profile/hlo_summary.md`, `profile/compile_summary.md`,
`profile/trace_summary.md`, `profile/retrace_details.txt`,
`profile/collectives_details.txt`.

This is a quick scan for next-round optimization targets. The opts I shipped
this session (#1 phase-on-slice, #3 pre-read IBZ ζ̃, #4 G-chunk scan) saved
~3% on the accumulator total at 8-chunk; the items below are an order of
magnitude bigger.

## Top runtime sinks (per `trace_summary.md`)

The xprof trace (156.9 s GPU duration) ranks GPU work — high-impact items
*not* obvious from `timing.section` because they fire inside fused kernels:

| Rank | Op | Total GPU ms | Calls | Where |
|---:|---|---:|---:|---|
| 1 | `all-reduce-start` in `jit__batched_chol` | **26 428** | 306 | `cholesky_2d.cholesky_2d_batched` panel loop |
| 2 | `all-gather-start` in `jit__kernel` (`_solve_all_at_once`) | 3 651 | 53 | LU back-solve in fit_one_rchunk |
| 3 | `all-gather-start.5` in `jit__kernel` (transpose) | 1 995 | 32 | `wfn_transforms.py:109` post-`_box_kernel` |
| 4 | `all-to-all.5` in `jit__kernel` (`_reshard_z`) | 1 497 | 33 | ζ-solve reshard |
| 5 | `custom-call.135.0` (cusolver LU) | 1 097 | 48 | per-q LU inside `_solve_all_at_once` |
| 6 | `cutlass …z884gemm…` | 915 | 332 | GEMM (real compute) |
| 7 | `loop_transpose_fusion_5` | 661 | 64 | various |
| 8 | `loop_reduce_fusion` (einsum/dot_general) | 318 | 78 | various |
| 9 | various `fft.NN.0` | ~260 each × 6 ops | 96–128 ea | cuFFT calls inside `_kernel` |

**Observations:**

* **#1 dwarfs everything.** Cholesky panel all-reduces total ~26.4 s of
  GPU time (>10× the next item). The wall-clock impact is bounded by
  the longest dependency chain (cholesky alone reports ~1.28 s in the
  section timing for charge channel), but 26 s of GPU stream time means
  the device is saturated with NCCL traffic across the cholesky run.
  Max single call: 4.45 s (NCCL warm-up on first call; typical is
  ~72 ms across 305 other calls). Source:
  [`cholesky_2d.py:cholesky_2d_batched`](../../sources/lorrax_D/src/common/cholesky_2d.py).
* **GEMM is only 0.9 s.** The arithmetic intensity per-call is high,
  but total compute time is dwarfed by communication. Classic
  comm-bound regime.
* **All-gather at `wfn_transforms.py:109`** (the transpose after the
  `_box_kernel` `jnp.take`) lands a 506 MiB per-call collective and
  fires ~32× across the run. The transpose `(nb, ns, n_k, …) → (n_k, nb, ns, …)`
  changes which axis is sharded, forcing the all-gather. Pre-existing
  issue, not introduced by my opts.

## Host↔device overlap — **completely unoverlapped**

| Direction | Count | Total time | Exposed | Overlap frac |
|---|---:|---:|---:|---:|
| H2D | 2535 | 520 ms | 520 ms | **0.000** |
| D2H | 1052 | 182 ms | 182 ms | **0.000** |

**Every H2D and D2H is on the critical path.** No async overlap is
happening — copies aren't dispatched ahead of compute. Total exposed
copy: ~700 ms. Small in absolute terms but indicative of a sync wfn
loader pattern. With CrI3-class systems doing 100s of GB of copies
(ψ(G) host caches → device), this would dominate.

## Compile-time issues (per `compile_summary.md` + `retrace_details.txt`)

Total XLA compile: **20 s** (12% of 168 s wall). 668 cache misses.
Top miss locations:

| Source | #misses | Pattern |
|---|---:|---|
| `wfn_loader.py:587` | 30 | `read_kchunk_union_sharded.<locals>._per_rank` — fresh closure per call |
| `fft_helpers.py:325` & `:343` | 22 + 20 | `for fft … tracing context doesn't match` — different jit-context cache misses |
| `gamma_matrices.py:122`, `:99`, `:46` | 22 + 17 + 16 | `_where` / `dynamic_slice` / `cumsum` traced fresh from `gamma_apply` / `gamma_perm_phase` |
| `_slab_io_ffi.py:697`, `:585` | 20 + 17 | `_FfiBackend.{read,write}_slab.<locals>._per_rank` — closure trap |
| `wfn_loader.py:883`, `:956`, `:598` | 15 + 15 + 10 | `dynamic_slice` / `_where` inside wfn_loader |

**Closure-defined-inside-function trap** is the dominant pattern.
Example — `_FfiBackend.read_slab` defines a fresh `_per_rank` closure
on every call:

```python
# _slab_io_ffi.py:670-686
sm = self._sm_cache.get(cache_key)
if sm is None:
    def _per_rank(offset_local, valid_shape_local, _ds_id=..., ...):
        return ffi_read_call(...)
    sm_bare = shard_map(_per_rank, mesh=mesh, ...)
    sm = jax.jit(sm_bare)
    self._sm_cache[cache_key] = sm
```

The `self._sm_cache` *should* protect against re-trace, but the cache
key includes `partition_spec` (which is sometimes a freshly-constructed
`PartitionSpec` with a non-stable hash). Result: many cache misses on
keys that are structurally identical, leading to recompile.

Looking at `retrace_details.txt`, **12+ separate `_per_rank` modules
are compiled with literally identical input signatures**:

```
module_0684.jit__per_rank  peak=129.73 MiB
  sig: Arg_0.1: c128[9,5760,656], Arg_1.2: s64[3], Arg_2.3: s64[3]
module_0765.jit__per_rank  peak=129.73 MiB
  sig: Arg_0.1: c128[9,5760,656], Arg_1.2: s64[3], Arg_2.3: s64[3]
... (and 10 more identical sigs)
```

Each compile is ~0.025–0.1 s; 12+ × ~75 ms = ~1 s wasted on this
exact pattern. Across all closure-trap sites: ~5–10 s waste total.

## fit_one_rchunk kernel retracing — same-signature recompile

24 separate `jit__kernel` modules under fit_one_rchunk. Grouped by
input signature:

| #modules | Signature | Where |
|---:|---|---|
| 3 | `c128[9,640,80,4], c128[9,640,80,4], c128[9,640,640]` | charge (μ_L=0, n_rmu=640) |
| 9 | `c128[9,656,80,4], c128[9,656,80,4], c128[9,656,656], s32[4], c128[4]` | transverse (μ_L=1,2,3, n_rmu=656; +γ̃ perm/phase) |
| 4 | `c128[9,656,656], c128[9,656], c128[9,46080,656], c128[9,46080,656], …` | V_q kernel, transverse off-diag (`same_zeta=False`) |
| 4 | `c128[9,656,656], c128[9,656], c128[9,46080,656], …` | V_q kernel, transverse diag (`same_zeta=True`) |
| 4 | `c128[9,640,640], c128[9,640], c128[9,46080,640], …` | V_q kernel, charge (CC tile) |

**These should compile 5 times total (one per shape group), not 24.**
For example the 9 transverse fit_one_rchunk compiles should be ONE
compile shared across all 3 channels × 8 r-chunks since γ̃ is a
runtime arg. The 4 V_q recompiles per tile shape (diag, off-diag, CC)
should be 1 each — all 3 tiles of the same shape group hit one cache
entry.

Wasted: ~19 redundant compiles × ~0.6 s avg = ~11 s of compile time.
**This is the biggest concrete compile-time win.**

## Memory model — top jit modules by peak HBM

| Module | Peak HBM | Top alloc |
|---|---:|---|
| `jit__kernel` (many) | 10.40 GiB | preallocated-temp |
| `jit__kernel` (charge) | 10.11 GiB | preallocated-temp |
| `jit__kernel` (V_q off-diag) | 5.14 GiB | preallocated-temp |
| `jit__kernel` (V_q diag/charge) | 3.06 GiB | preallocated-temp |

The 10.4 GiB peak is well below the 40 GB A100 limit but a substantial
fraction of `memory_per_device_gb = 28` budget. On a per-process basis,
this leaves only ~17 GB headroom for other live tensors. The
`preallocated-temp` allocation type means XLA is reserving a single
big buffer for use across the kernel — a known pattern for fused
kernels with multiple intermediate FFT/contract steps.

## Ranked recommendations for next round

| Priority | Where | Cost | Estimated win |
|---|---|---|---|
| **1** | Cholesky panel all-reduces (26 s GPU time) | medium | Investigate sharding/panel size in `cholesky_2d_batched`. Possibly switch to cuSolverMp Cholesky for `μ_L=0` channel where it's currently sharded (cuSolverMp is one big call, replacing many small NCCL ones). |
| **2** | Fix closure-defined-inside-function retraces | small (~1 hr/site × 3 sites) | ~5–10 s compile time saved per run |
| **3** | Fix fit_one_rchunk and V_q kernel cache-key bug (24× recompile) | small | ~11 s compile time per run |
| **4** | H2D/D2H async overlap is 0.0 | medium | Investigate whether wfn loader can pre-fetch the next r-chunk's ψ(G) while compute runs. Bigger win at CrI3 scale. |
| **5** | All-gather at `wfn_transforms.py:109` (transpose after `_box_kernel`) | small | Try `with_sharding_constraint` on the gather output to keep the band-axis sharded post-transpose, or rewrite the gather + transpose as one op. |

## What this exercise validated

* The profiling pipeline (`run_profiled.py` → `analyze_*.py` → `*_summary.md`)
  produced ranked, source-located findings in <1 minute per run. The
  setup is correct and reliable. Both `hlo_summary.md` and
  `trace_summary.md` agree on the top sharding hotspots.
* My opts #1 and #4 are correctness-preserving (bit-identical eqp0) and
  show measurable wins on the kernels they target, even at MoS2 scale.
  They just don't move the needle because MoS2 work is dominated by
  the items above (Cholesky comm, compile waste).

## What I'd do next

If I were continuing this session: pick item **#3 (cache-key bug for
fit_one_rchunk)** first — it's small, localised, and predicts ~11 s
wall savings per run on MoS2 (~7% wall). Then item **#1 (Cholesky)**,
which is the biggest absolute target but needs more thought (changing
to cuSolverMp may have memory tradeoffs).
