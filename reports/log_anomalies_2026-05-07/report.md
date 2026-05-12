# Log anomalies report — 2026-05-07

Survey of every non-progress / non-banner line emitted by recent CrI3 + MoS2 lorrax runs (110 logs, last 24 h) against lorrax_B and lorrax_D source. Sorted by severity.

## TL;DR

| # | Class | Severity | Source | Action |
|---|-------|----------|--------|--------|
| 1 | `cuFFT batched plan failed` | **fatal** | XLA fft_thunk | Real OOM on long-running alloc; restart needed |
| 2 | `GPU_0_bfc ran out of memory` | **fatal** | XLA bfc_allocator | Old D 6×6 80Ry nband=150 OOM (from prior session, fixed by chooser patch) |
| 3 | `WFN symmetry data incomplete: N k-points… identity fallback` | **silent correctness risk** | lorrax common/symmetry_maps.py:296 | The original `00_cri3_4x4_6x6_test` WFN: 27/36 k-points fall back to identity. **This is what caused the original σ_X "regression"** |
| 4 | `Some donated buffers were not usable` | medium | jax mlir.py:1178 | Many shapes seen across all runs. Real perf cost; not correctness |
| 5 | `Constant folding > Ns` slow alarms | medium | XLA slow_operation_alarm | AOT compile path; first-run cost only |
| 6 | `γ (runtime / AOT-pred) = 0.3–0.5` | medium | lorrax internal report | AOT memory model **over-predicts peak by 2–3×** |
| 7 | `Reducing band_chunk to 16` | low | lorrax chunker | Auto-downsize, expected on tight budgets |
| 8 | `SymMaps: time-reversal symmetry` | low | lorrax symmetry_maps.py:324 | Informational; consequences unclear without `noinv` |
| 9 | `FFI backend: chunks/attrs … no-op` | benign | lorrax _slab_io_ffi.py | Future feature, not used in current code path |
| 10 | `pw2bgw: resetting diag_nmax to … 78` | benign | QE pw2bgw | When NSCF nbnd < requested vxc range |
| 11 | `Zap's log subsystem` | benign | Cray MPICH | Slingshot init noise, no effect |
| 12 | `bash: cannot set terminal process group` | benign | bash -i without TTY | Harmless interactive shell warning |

---

## 1. cuFFT batched plan failure (fatal)

```
INTERNAL: RET_CHECK failure (.../fft_thunk.cc:176) fft_plan != nullptr
Failed to create cuFFT batched plan with scratch allocator
```

Seen at 12:20 (L 3×3 80Ry run) and 12:21 (some D run).

**Reading the source**: `fft_thunk.cc:176` is where XLA asks cuFFT to allocate a batched FFT plan. If GPU memory is too fragmented or insufficient, cuFFT's plan allocation returns null and XLA aborts.

**Trigger here**: not a real OOM — the GPU still has memory listed as free, but cuFFT needs a contiguous workspace (~hundreds of MB to a few GB) and the BFC allocator's pool is fragmented from prior runs in the same allocation. The state survives across `srun` steps because nothing flushes the GPU between them.

**Fix**: just-restarted alloc fixes it. Long-term, set `XLA_PYTHON_CLIENT_PREALLOCATE=true` to force a single big arena (already done in lorrax env), or call `jax.clear_caches()` between steps.

## 2. BFC allocator OOM (fatal, but for one specific old run)

```
W external/.../bfc_allocator.cc:501] Allocator (GPU_0_bfc) ran out of memory
trying to allocate 32.19GiB (rounded to 34560000000) requested by op
```

Seen 4× in the 2026-05-06 lorrax_D CrI3 6×6 80Ry runs. **Different from cuFFT failure**: BFC is XLA's general-purpose allocator. 32 GB allocation on a 40 GB GPU = real OOM.

This is what motivated my V_q chooser FFT-stage fix earlier this session. After the patch, lorrax_D no longer asks for a 32 GB allocation in `_compute_all_V_q_sharded` — peak is now ~12 GB.

## 3. WFN symmetry data incomplete (silent correctness risk)

```
UserWarning: WFN symmetry data incomplete: 27 k-points could not be
mapped via stored symmetries (ntran=12). Using identity fallback.
First unmatched: [0. 0.16666667 0.]
```

**Reading [common/symmetry_maps.py:285-298](sources/lorrax_B/src/common/symmetry_maps.py#L285-L298)**:
The full BZ unfolding tries every stored sym op to map each full-BZ k onto an IBZ rep. If no op works, it falls back to the **nearest IBZ k-point with the identity sym op** — i.e. it pretends the unmatched full-BZ point IS its nearest IBZ neighbor.

**This is the source of the original σ_X "regression".** The `00_cri3_4x4_6x6_test/00_lorrax_kmeans/WFN.h5` WFN had `ntran=12` (with inversion) but the stored ops couldn't unfold to all 36 BZ points. 27 of 36 points silently used identity-fallback wave functions belonging to the wrong k-point. Pair-density and V_q thus computed wrong overlaps → −298 eV instead of −18.5 eV at HOMO.

The fresh M build (no_t_rev + noinv → ntran=6, IBZ 8) doesn't trigger this warning because the orbits actually close correctly under the smaller stored op set.

**Severity**: **HIGH but bounded** — only triggers on certain WFN.h5 files. If the warning fires, the run's σ values are *not* trustworthy. Any agent should treat this warning as a hard stop, not a soft note. Worth promoting to a `raise` or at minimum a banner.

## 4. "Some donated buffers were not usable" (perf, recurrent)

```
/opt/jax/jax/_src/interpreters/mlir.py:1178: UserWarning: Some donated buffers
were not usable: ShapedArray(complex128[36,160,160])
```

JAX emits this when a `donate_argnums=` argument was passed but XLA can't actually reuse the buffer in-place (input layout/shape != output layout/shape, or both donor and result are needed simultaneously by the schedule).

**Shapes I observed**:
| shape | likely site | what it costs |
|---|---|---|
| `(36, 160, 160)` | Σ_X k×n_band×n_band accumulator | ~14 MB extra alloc per batch |
| `(36, 375, 17500)` | μ-on-x ζ-fit intermediate | **2 GB** |
| `(36, 375, 8192)` | reshard buffer | 1 GB |
| `(36, 900, 900)` | V_q tile (μ_chunk × μ_chunk) | 230 MB |
| `(36, 16, 17500)` | tile contraction | 80 MB |
| `(4, 375, 17500)` | per-q ζ slab | 200 MB |

Pattern: it's almost always the donor argument of `fit_one_rchunk`-style or V_q kernel jits. Likely cause: `with_sharding_constraint` injects layout boundaries that prevent in-place reuse. Fixing requires aligning input/output layouts via explicit `in_shardings=out_shardings` or refactoring donate calls.

**Severity**: medium perf, not correctness. The 2 GB extra alloc per batch is real wasted memory.

## 5. XLA slow_operation_alarm (medium, first-run only)

```
E external/.../slow_operation_alarm.cc:73] Constant folding an instruction
is taking > 1s:  ... took 4.502115191s
```

Seen 12+ times across recent runs, always during the first call of a freshly-jitted kernel.

**What it means**: XLA's compile pipeline is folding constants (precomputed integer indices, FFT phase tables, sphere indices) into the HLO. Some of these are large arrays (sphere indices for n_G = 62k, FFT frequency tables for 75×75×200) and constant-folding takes seconds.

**Severity**: medium. First run pays this; subsequent runs hit the JAX compilation cache at `$JAX_COMPILATION_CACHE_DIR=/pscratch/sd/j/jackm/.jax_cache`. Could be reduced by passing these as runtime args instead of jit closure constants in the FFT helper / V_q kernel.

## 6. γ (runtime / AOT-pred) = 0.3–0.5 (memory model over-predicts)

```
GPU high-water mark: 7.96 GB / 45.00 GB budget (18%)
Memory estimate: peak 33.34 GB (budget 45.00 GB), bottleneck=fft
γ (runtime / AOT-pred) = 0.486 (AOT predicted 16.37 GB)
```

The AOT memory model in `compute_optimal_chunks` predicts peak 16–25 GB; runtime hits 7–8 GB. So **γ ≈ 0.4** systematically.

**This is the OPPOSITE of the V_q chooser bug I fixed earlier**: that one *under*-predicted, this one *over*-predicts. Mismatched calibration of two different memory models for two different stages.

**Consequence**: chunker downsizes work too aggressively → more chunks than necessary → unnecessary kernel-launch overhead. Should re-fit the model coefficients against measured peaks (see `aot_memory_model/artifacts/*.json` for measurement infrastructure that already exists).

## 7. "Reducing band_chunk to 16" (informational)

```
Reducing band_chunk to 16 (bands/device=1)
```

Auto-chunker downsizing band_chunk to fit memory budget. Expected behavior; only worth flagging if it goes excessively small.

## 8. SymMaps time-reversal warning (informational)

```
SymMaps: 4/9 full-BZ k-points require time-reversal symmetry for unfolding.
Non-symmorphic phases are NOT applied for these k-points. Use noinv=.true.
in QE to avoid this.
```

[symmetry_maps.py:321-329](sources/lorrax_B/src/common/symmetry_maps.py#L321-L329) emits this when `irk_sym_map` indices ≥ ntran (i.e. the matched op was a TR-augmented one, not stored in WFN).

**For MoS2 (3×3=9 BZ pts)**: 4/9 use TR. Result still matches BGW to 1 meV — apparently the missing non-symmorphic phase doesn't matter for MoS2 (no fractional translations).

**For CrI3 (6×6=36 BZ pts)**: would matter more, since CrI3 has non-symmorphic ops (nontrivial tnp). The fresh M WFN has noinv=true and avoids this warning entirely.

**Severity**: low for symmorphic systems (Si, MoS2), potentially medium for non-symmorphic. A correctness audit on CrI3 specifically requires confirming the unfolded ψ phases.

## 9. FFI backend chunks/attrs no-op (benign)

```
UserWarning: FFI backend: chunks/attrs on create_dataset currently no-op;
pre-create with h5py if you need explicit chunking or attrs.
```

[_slab_io_ffi.py:229-241](sources/lorrax_D/src/file_io/_slab_io_ffi.py#L229-L241): the FFI-backed PHDF5 SlabIO accepts `chunks=` and `attrs=` for API compatibility but doesn't actually pass them through to the underlying HDF5 API. Datasets are created un-chunked. Read perf is fine because the FFI uses MPI-IO not HDF5's chunked-direct path.

**Severity**: benign. Future feature. Not affecting correctness.

## 10. pw2bgw: resetting diag_nmax (benign)

```
WARNING: resetting diag_nmax to max number of bands 78
```

When `vxc_diag_nmax` in pw2bgw.in exceeds the actual nbnd in the QE save, pw2bgw clamps it. This is what bit me on M's NSCF when the save-dir was stale (NSCF nbnd=160 in input but save dir from SCF had nbnd=78).

**Severity**: in normal use, benign. In this session it was a useful diagnostic that the save dir was stale.

## 11. Cray MPICH "Zap's log subsystem" (benign)

```
WARNING: Failed to create Zap's log subsystem.
WARNING: Failed to create zap_sock's log subsystem. Error 17.
```

Cray Slingshot/MPICH startup noise on Perlmutter. "Error 17" = EEXIST on socket bind (multiple ranks racing on the log socket). Doesn't affect runtime. Same warning prints on every QE launch in the sandbox.

## 12. bash terminal process group (benign)

```
bash: cannot set terminal process group (-1): Inappropriate ioctl for device
bash: no job control in this shell
```

`bash -i -c` running without an attached TTY (which is how the agent invokes lxrun under `module use` / `module load`). Bash can't set up job-control TTY but everything else works fine.

---

## Recommendations

1. **Promote `WFN symmetry data incomplete` to a hard error** unless explicitly bypassed. It's the silent culprit that took us hours to isolate this session, and the identity-fallback path produces *wrong-but-running* results.

2. **Audit donate_argnums on the big-shape kernels** (`(36, 375, 17500)` and similar). 2 GB extra alloc per fit_one_rchunk batch is the largest single perf cost in the donor-failure list.

3. **Re-calibrate the AOT memory model.** γ ≈ 0.4 across all recent CrI3 runs means the chunker is consistently leaving 2–3× headroom. Either retune the `aot_memory_model/artifacts/*.json` coefficients or add a runtime feedback loop that adjusts γ.

4. **Add `noinv=.true.` to the build_inputs SKILL** as the default for all 2D and 3D WFN generation. Today it's only mentioned as a fix-after-warning in symmetry_maps.py; making it the default in scf.in / nscf.in templates avoids the silent fallback path entirely.

5. **The slow_operation_alarm constant-folding warnings** could be silenced for known-large constants (sphere indices, FFT phase tables) by emitting them as runtime args rather than jit closure values. Not high-priority.
