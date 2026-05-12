# V_q tile consolidation + memory model + flat-q V_qmunu — 2026-05-08

End-of-session report.  All work is on `lorrax_A`, branch `agent/phdf5_padded` (built off `agent/v_q_perf`), 17 commits ahead of main.

---

## 1. What landed

### Consolidation (the headline change)

Replaced two divergent V_q tile implementations on lorrax_A and lorrax_B with a single source of truth on lorrax_A's `gw/v_q_tile.py`:

| Before | After |
|---|---|
| lorrax_A: `_make_V_q_caseA_kernel` + `_make_V_q_caseB_kernel`, plus `_compute_all_V_q_sharded` + `_compute_all_V_q_replicated` (~920 LOC) | `_make_V_q_tile_kernel` (one inner kernel) + `compute_V_q_tile` (one outer driver, single nested loop covering both Case A and Case B) (~700 LOC) |
| lorrax_B: separate `gw/v_q_tile.py` for the bispinor wrapper | thin shim, will rebase to import from this single source when bispinor lands on A |

Net: −1536 / +990 lines on `agent/phdf5_padded`. One inner kernel handles same-ζ (V^{0,0} self-contraction) and distinct-ζ (bispinor V^{μ_L,ν_L}) via a Python-static `same_zeta` flag; one nested driver loop covers `(q_chunk, μ_block, ν_block)` for either Case A or Case B as parameterizations of the same loop body.  Externalised `v_per_G_fn` and `phase_fn` callables let any Coulomb dimensionality (3D/2D/0D) or signed bispinor transverse-projector weight plug in identically.

### Allgather (h5py) backend honors `partition_spec`

`_AllgatherBackend.read_slab` now accepts a `partition_spec`.  PHDF5 reads sharded directly; allgather rank-0 reads then `device_put`s with the same sharding.  The unified driver issues the same `read_slab(..., partition_spec=…)` regardless of backend.

### Padded SlabIO ported from lorrax_B

A subagent cherry-picked B's four padded-SlabIO commits onto agent/phdf5_padded (`fcbe0ea` write-offset fix, `41d4a94` SlabIO contract, `f41c737` padded extents, `19322f6` documentation).  The C++ FFI .so was rebuilt via `src/ffi/common/cpp/run_shifter.sh bash src/ffi/common/cpp/build.sh` on the login node (~30 s) — required because the new handlers take a `valid_shape` Buffer arg (3-arg ABI vs old 2-arg).  Future rebuilds when bispinor's μ counts don't divide the mesh evenly.

### Flat-q V_qmunu

V_qmunu shape changed from legacy 8-D `(1, npol, npol, nkx, nky, nkz, μ, μ)` to 3-D `(nq, μ, μ)` throughout the gw/cohsex pipeline.  The `(1, npol, npol)` axes were always vestigial in scalar mode; bispinor will use a structured `V_q_bispinor` NamedTuple (CC + CT(3) + TT(3,3)) since μ_C and μ_T differ across polarisation tiles.

- `compute_V_q` returns `(nq, μ, μ)` directly.
- `flatten_V_qmunu` is a back-compat shim (handles 8-D / 6-D / 3-D inputs).
- `tagged_arrays.write_restart_state_to_h5` writes the new 3-D form and a top-level `kgrid` dataset so BSE downstream can recover (nkx, nky, nkz) without re-reading the WFN.
- `bse/bse_io.py` updated to consume both legacy 8-D and new flat-q layouts (replaces the previous TODO with a real fix).

### Memory model — fully system-parameterised

After today's work, every term in `_choose_v_q_chunks` is either a physical constant, system geometry, or **AOT-measured for the actual shape**.  No empirical magic constants remain in the chooser.

Three accuracy levels, each replacing the previous:

| Era | FFT-stage model | Gather-stage model |
|---|---|---|
| Pre-consolidation | `_Q_COMPUTE_COEF_FFT × N_zeta × μ × n_rtot / p_prod`, hand-tuned 5×; required env-var override | `_Q_COMPUTE_COEF_GATHER = 4.4 × N_zeta × μ × n_G / p_min` |
| Mid-session (slope+intercept) | AOT-measure `make_sharded_fftn_3d` standalone primitive at Q=1 and Q=2; linear regression | analytical `(1/p_x + 1/p_y + 1/(p_x · p_y)) · μ · n_G · 16` |
| **End of session (full-kernel AOT)** | **AOT-compile the *whole* `_make_V_q_tile_kernel` at the candidate q_chunk, read `compiled.memory_analysis()`** | analytical (kept as fallback, plus `max(gather, fft)` is now subsumed by the full-kernel pass when AOT succeeds) |

Empirical AOT accuracy on MoS2 3×3 cohsex (mesh 2×2, μ=640, n_G=2419):

| Model | Predicted peak/rank | Actual XLA `memory_analysis` | Error |
|---|---|---|---|
| Manual `LORRAX_V_Q_FFT_COEF=2.0` | (config-dependent) | 2.99 GiB | — |
| AOT slope+intercept | **2.14 GB** | 2.99 GiB | **−28 %** |
| AOT full-kernel (this session) | **3.21 GB** | 2.99 GiB | **+7 %** |

The AOT-of-full-kernel path captures cuFFT scratch (XLA tracks plan workspace through `temp_size_in_bytes`), gather temps, contract intermediates, DUS overlap, and any aliasing XLA negotiates between the kernel's stages — none of which the standalone-FFT measurement saw.

Validated CrI3 q=18 OOM stress test still works as expected (the prior slope+intercept predicted 60.9 GB at q=18 vs actual XLA-requested 56.7 GB; full-kernel AOT will be re-validated on next CrI3 run).

### Diagnostic env knobs (all default-off except where noted)

- `LORRAX_V_Q_FFT_COEF=<float>` — override the slope-fallback FFT-stage coefficient.
- `LORRAX_V_Q_TIME_STAGES=1` — host-blocked per-stage timing in the V_q tile loop.
- `LORRAX_V_Q_Q_CHUNK=<int>` — force a specific Case-A q_chunk.  Used to validate the AOT model via OOM stress test.
- `LORRAX_V_Q_AOT_VERBOSE=1` — print AOT slope/intercept and full-kernel peak predictions.
- `LORRAX_V_Q_AOT_FULL_KERNEL=0` — opt out of full-kernel AOT (fall back to slope+intercept). **Default on.**

---

## 2. Bench results

CrI3 6×6×1 80 Ry, 16-GPU 4×4 mesh (alloc 52664413, hbm80g):

| Configuration | q_chunk | V_q tile wall |
|---|---|---|
| Bispinor wrapper (lorrax_B, pre-consolidation) | 12 | **892 s** |
| Manual `LORRAX_V_Q_FFT_COEF=2.0` | 12 | 218 s |
| Default `LORRAX_V_Q_FFT_COEF=5.0` | 5 | 274 s |
| AOT v3 (slope+intercept, production primitive) | 12 | 152 s |
| Full-kernel AOT (default budget 45 GB, alloc 52707589) | 8 | **1640 s** (Lustre-slow; kernel 12.6 s, read 2127 s host-blocked) |
| Full-kernel AOT (raised budget 70 GB, alloc 52707589) | 13 | **FAIL: cuFFT plan scratch alloc** (predicted 66.32 GB; cuFFT scratch pushed past 80 GB ceiling on first invocation) |
| Forced q_chunk=18 (override) | 18 | OOM at 56.7 GB (predicted 60.9 GB) |

The 4× speedup over the bispinor wrapper at the same q_chunk is the consolidation effect, not chooser changes.  Stage breakdown on the 218 s baseline (host-blocked, isolated):

```
read   = 346.8 s
kernel =  14.1 s
v+phase=   1.3 s
total  = 362.3 s
actual wall = 226 s   (→ JAX async dispatch hides 137 s of read time)
```

MoS2 3×3 80 bands cohsex (alloc 52690853, hbm80g, 1 node, mesh 2×2):

| Stage | wall |
|---|---|
| ζ-fit | ~245 s |
| V_q tile | 0–4 s (one batch covers all 9 q's; small system) |
| σ_X | <2 s |

V_q=0 trace **842442655.9280** matches across all variants (correctness preserved).

---

## 3. Outstanding & open

### CrI3 V_q 10× slowdown (2134 s outlier)

Subagent forensics report at `reports/v_q_slowdown_forensics_2026-05-08/report.md`.  Conclusion: **almost certainly a transient Lustre/OST contention event during the 19:42–20:18 window**, not a code regression.  Evidence:

- Same job (52664413), same nodes, same chooser pick (q_chunk=12 with predicted peak 40.70 GB), same kernel module.
- ζ-fit in the same run completed in 221 s (faster than several reference runs at 251 s) — same SlabIO/PHDF5/FFI write path was healthy in the same window.
- All three V_q batches uniformly slow (711 s/batch vs 51 s/batch fast) — rules out compile/JIT one-off cost.
- Padded-SlabIO read code path adds ≤ tens of µs of host-side overhead per call; cannot account for 660 s/batch.

Recommended next steps (from the forensics report, ranked):
1. Reproduce with `LORRAX_V_Q_TIME_STAGES=1` to see if `read=` is the inflated term.
2. Back-to-back A/B in one allocation; if run-to-run wall varies by 10× across the same job, it's system fluctuation.
3. Bisect: `git checkout` to last pre-padded SlabIO and re-run; if still slow when system loaded, regression is conclusively excluded.

### Ring/ppermute V_q algorithm

Subagent design report at `reports/v_q_ring_algorithm_2026-05-08/report.md`.  Conclusion: **viable but not worth it now**.  Memory savings on the gather stage are 2.5×–8.5× depending on mesh, but the FFT stage dominates `max(gather, fft)` for all current workloads, so ring would save memory in a stage that isn't binding.  ~80 LOC prototype if you want it.  Triggers that would justify it: μ ≥ 4000, mesh ≥ 4×8, or after the planned NUFFT/flat-k chi0 refactor (which deflates the FFT stage and exposes gather).

### BSE consumer

The flat-q V_qmunu shape change rippled into BSE's `bse_io.py`; the consumer now handles 8-D (legacy) / 6-D (transitional) / 3-D (new flat-q) layouts via a small dispatch shim that recovers (nkx, nky, nkz) from either the dataset shape (legacy) or a top-level `kgrid` dataset on the restart file (new) or the WFN (fallback).  Not exercised by the cohsex regression test; verify on an actual BSE run.

### Bispinor

The B-side `v_q_lorentz.py` outer Lorentz orchestrator and bispinor σ_X^B / σ_H^B paths are intentionally **not** ported.  When the bispinor branch on lorrax_B is rebased onto a future main that includes `agent/phdf5_padded`, the import path becomes a one-line change (`from .v_q_tile import compute_V_q_tile`), since A and B now share the same primitive.

---

## 4. Branch and commit map

`agent/phdf5_padded` (this branch, 17 commits past main):

```
723e1df gw/v_q_tile: full-production-kernel AOT chooser via memory_analysis()
9acac0d gw+bse+file_io: V_qmunu becomes 3-D flat-q; analytical gather; BSE compat
a052a1c gw+file_io: V_qmunu becomes flat-q throughout the gw/cohsex pipeline
11c58b9 gw/v_q_tile: env-gated diagnostics — Q_CHUNK override + AOT_VERBOSE
6989a19 Document SlabIO padded extent contract              ← from B
a16f8eb Support padded PHDF5 slab extents                   ← from B
6cdf077 Harden PHDF5 SlabIO contract                        ← from B
fe1eefd Fix PHDF5 write offset sharding                     ← from B
99e10c5 gw/v_q_tile: AOT FFT model uses production primitive (make_sharded_fftn_3d)
29d9381 gw/v_q_tile: AOT FFT model = (slope · Q + intercept), not flat coefficient
03663a8 gw/v_q_tile: AOT-exact FFT workspace via query_fft_peak_bytes in chooser
686c188 gw/v_q_tile: env-gated per-stage timing (LORRAX_V_Q_TIME_STAGES)
6761e45 gw/v_q_tile: chooser cleanups — 5× FFT workspace, same-vs-distinct, dropped Case A/B tag
387bbd9 gw/v_q_tile: merge Case A and Case B into one nested loop
aac6315 gw: consolidate V_q tile to one unified driver + kernel
```

Pre-merge cleanup suggestion: squash the three `03663a8`/`29d9381`/`99e10c5` AOT-FFT iterations into one (the v3 result is the only behavior-active state); `723e1df` (full-kernel AOT) sits on top of v3 and supersedes the slope+intercept path for the active code path, but keeps it as fallback.

---

## 5. Recommended next steps

In priority order, things you might want to act on:

1. **(DONE: commit bbb2925 on `agent/aot-cufft-workspace`)** Principled cuFFT scratch via `cufftGetSize` on jaxlib's loaded `libcufft.so`.  See `reports/v_q_memory_model_open_2026-05-08/report.md` §Closure.  Model predicts every observed CrI3 outcome (Q=8 ran, Q=13 cuFFT bust, Q=18 OOM) and matches MoS2 observed XLA peak within 0.07 GB.
2. **Don't raise `memory_per_device_gb` in CrI3 templates.** With the new accurate model, Q=12 lands right at the 80 GB ceiling and Q=13 is over it.  At the existing 45 GB budget the chooser will correctly pick a smaller Q.
3. **Re-run CrI3 V_q on a clean Lustre window** to get a real wall measurement.  Today's runs (Q=8 at 1640 s and yesterday's 2134 s outlier) both hit Lustre-side read slowdowns; need a baseline uncontaminated by I/O contention.
4. **Reuse follow-up:** call `aot_kernel_peak_bytes` from the chi0/W chooser and the σ chunker — same primitive, different kernels.  Single source of truth for FFT-aware AOT memory.
5. **Squash the AOT-FFT iterations** into one commit before pushing, per the cleanup suggestion above.
6. **CHANGELOG.md** updated under the 2026-05-08 heading; ready for review.
7. Subagent reports (forensics + ring + memory model open + bispinor plan) live alongside this one.

---

## 6. Files and reports

- `/pscratch/sd/j/jackm/lorrax_sandbox/reports/v_q_consolidation_2026-05-08/report.md`  ← this file
- `/pscratch/sd/j/jackm/lorrax_sandbox/reports/v_q_slowdown_forensics_2026-05-08/report.md` ← CrI3 outlier
- `/pscratch/sd/j/jackm/lorrax_sandbox/reports/v_q_ring_algorithm_2026-05-08/report.md` ← ring design
- `/pscratch/sd/j/jackm/lorrax_sandbox/reports/v_q_memory_model_open_2026-05-08/report.md` ← memory model open + 70 GB distinguishing test verdict (case 3: AOT under-counts cuFFT scratch)
- `/pscratch/sd/j/jackm/lorrax_sandbox/reports/v_q_bispinor_plan_2026-05-08/report.md` ← bispinor V_qmunu implementation plan
- `/pscratch/sd/j/jackm/lorrax_sandbox/CHANGELOG.md` ← top entry under 2026-05-08
- `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_perf_2026-05-07/run_logs/cri3_full_aot_20260508_094522.log` ← Q=8 successful run (Lustre-slow)
- `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_perf_2026-05-07/run_logs/cri3_70gb_20260508_102736.log` ← Q=13 cuFFT failure
