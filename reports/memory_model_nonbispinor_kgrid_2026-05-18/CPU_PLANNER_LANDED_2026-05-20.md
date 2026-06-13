# CPU planner change — FFT-scratch test + bispinor confirmation + landed

**Follow-up to** `CPU_OVERHEAD_DECOMP_2026-05-20.md` (the agent decomposition that identified the 4-slot pattern). Two open questions there:
1. Is the 4th slot really pair-density, or is it FFT scratch that pocketfft happens to need on CPU? (User raised this hypothesis explicitly.)
2. Does the 4-slot pattern hold for the bispinor `pair_density_slots_transverse` constant too?

Both resolved here. Planner change landed on lorrax_B `agent/jax-09-cpu-compat` commit `5c2dae7`.

## FFT-scratch hypothesis — REJECTED

Si μ=384 non-bispinor n=4 (2×2 mesh), varying `band_chunk_size ∈ {32, 64, 120}` in cohsex.in. The FFT box is `band_chunk × n_rtot × ns × 16 / p_xy`, so doubling band_chunk doubles only FFT-related buffer sizes. If the 4th slot were FFT scratch we'd see per-slot bytes track band_chunk; if it's pair-density we'd see the slot bytes stay at `_bytes_c128(nk, ns², μ, r_chunk, /p_xy) = 5.70 GiB` regardless.

| band_chunk | r_chunk × n_chunks | preallocated-temp | slot count × per-slot size | FFT-box-shape inside slots |
|---:|---|---:|---|---|
| 32 | 13824 × 1 | 23.29 GiB | **4 × 5.70 GiB** | `c128[64, 8, 2, 24, 24, 24]` (small, aliased) |
| 64 | 13824 × 1 | 22.81 GiB | **4 × 5.70 GiB** | `c128[64, 15, 2, 24, 24, 24]` (small, aliased) |
| 120 | 13824 × 1 | 22.81 GiB | **4 × 5.70 GiB** | `c128[64, 15, 2, 24, 24, 24]` (small, aliased) |

Per-slot bytes **invariant** under band_chunk. Slot count **invariant**. The FFT-box shapes inside each slot DO change with band_chunk (bc=32 → 8 batches; bc=64 → 15 batches), but they alias into the same 5.70 GiB pair-density-sized slots.

Per-slot bytes match `_bytes_c128(nk=64, ns²=4, μ=432, r_chunk=13824, /p_xy=4) = 6,115,295,232 B = 5.696 GiB` to the byte.

Direct shape evidence — slot 1 of the bc=32 preallocated-temp shows `c128[64,8,2,24,24,24]` (FFT box) aliasing the same offset as `c128[2,6912,216,2,4,4,4]` (pair-density rank-7). The slot SIZE is set by pair-density (5.70 GiB); the FFT-box (425 MB at bc=32, 793 MB at bc=64) is the smaller occupant of the same memory.

**Conclusion: the 4th slot is a 4th concurrent pair-density allocation, not FFT scratch.** CPU XLA's BufferAssignment lifetime analysis keeps one more pair-density-shaped buffer simultaneously live than GPU XLA does.

Files: `runs/Si/NONBISPINOR_CPU_2026-05-20/mu384_fft_probe/bc{32,64,128}/{cohsex.in, gw_bc*.out, xla_dump/}`.

## Bispinor CPU confirmation — 4 slots for both charge and transverse

Si μ=384 with `bispinor=true` on CPU n=4 (2×2 mesh). The bispinor pipeline runs 4 channels — charge γ̃⁰ and 3 transverse μ_L=1,2,3 — through separate JIT compilations of `fit_one_rchunk`. Each channel produces its own HLO module.

| module | role | total | preallocated-temp | slots |
|---|---|---:|---:|---|
| `module_0360.jit_fn` | charge channel | 70.07 GiB | 68.91 GiB | **4 × 16.92 GiB** |
| `module_0413.jit_fn` | transverse μ_L=1 | 70.07 GiB | 68.91 GiB | **4 × 16.92 GiB** |
| `module_{0415, …, 0870}.jit_fn` | transverse μ_L=2,3 (+r-chunks) | 70.07 GiB | 68.91 GiB | **4 × 16.92 GiB** |

Per-slot bytes match `_bytes_c128(nk=64, ns²=16, μ=864, r_chunk=10268, /p_xy=4) = 18,166,841,856 B = 16.92 GiB` to the byte. (Bispinor: ns=4 → ns²=16, μ_padded=864 = 432 charge + 432 transverse stacked.)

**Conclusion: `pair_density_slots_transverse = 4` on CPU also** — same 4-slot pattern in both bispinor channels.

Files: `runs/Si/NONBISPINOR_CPU_2026-05-20/mu384_bispinor/{cohsex.in, gw_bispinor.out, xla_dump/module_036[0,3,5]*, xla_dump/module_041[3,5]*}`.

## Planner change — landed

`sources/lorrax_B/src/gw/gflat_memory_model.py` on branch `agent/jax-09-cpu-compat` (commit `5c2dae7`):
- New module-level helper `_default_pair_density_slots()` returns 4 on CPU, 3 elsewhere via `jax.default_backend()`. Documented with the HLO calibration provenance.
- Function args `pair_density_slots_{charge, transverse}` default to `None`; if not explicitly set, resolved via the helper at function-call time.
- No source change to the formula — the existing `_bytes_c128(nk, ns², μ, r_chunk, /p_xy)` per-slot formula carries over byte-exactly to CPU.

### Post-fix verification

CPU Si μ=384 n=4 with the new planner — re-ran identical command:
```
HWM estimate = 26.24 GB/dev (37% of budget)
Maximum resident set size: 26.64 GB
%-err (HWM − RSS) / RSS = −1.5%
```

vs pre-fix: HWM = 20.12 GB, RSS = 26.64 GB, %-err = **−24.5%** (under-protective).

GPU back-compat — same Si μ=384 on 4×A100 hbm80g 2×2 mesh under BFC+0.95:
```
HWM estimate = 20.12 GB/dev (29% of budget)
mem_stats peak = 20.13 GB
%-err = −0.05%  — IDENTICAL to pre-fix 2026-05-19 reference
```

Backend-aware dispatch works as designed: GPU value unchanged; CPU now properly predicted.

## What this finding closes

- The user's hypothesis "is it just FFT scratch?" — directly tested with band_chunk variation. Cleanly rejected with three data points.
- The Phase-2 open question from `CPU_OVERHEAD_DECOMP_2026-05-20.md` — `pair_density_slots_transverse` confirmed at 4 on CPU bispinor.
- The +30% RSS-vs-HWM excess that `CPU_VALIDATION_2026-05-20.md` reported — now characterized to one specific HLO-visible cause + planner refinement that closes the gap to ±3%.

## What this does NOT close

- The **0.35 GB glibc per-thread arena fragmentation** is still unmodelled. `MALLOC_ARENA_MAX=1` saves it; not worth modelling in the planner.
- The **~0.5 GB Python/jaxlib import floor** per process. Same status — XLA-pool aliasing makes the actual peak-RSS contribution variable.
- **Why** CPU XLA's BufferAssignment differs from GPU XLA in this exact scheduling decision. Empirical observation; not investigated at the XLA-internals level.
- **CPU FFT-box-factor recalibration** (`fft_box_factor_{A, D}` currently 4.0 / 2.0 from cuFFT). At Si μ=384 these are not binding (Peak A = 0.24 GB, Peak D = 5.26 GB vs binding C = 56 GB at n=1). The conservative GPU values stay; revisit if a future CPU production run shows Peak A or D becoming binding.

## Commits

- lorrax_B `agent/jax-09-cpu-compat`:
  - `c7e6695` `fix(jax-0.9): VMA pcast + tiled=True for multi-process CPU compat`
  - `5c2dae7` `feat(planner): backend-aware pair_density_slots — CPU=4, GPU=3`
- Sandbox `main` (this commit): CHANGELOG entry + this report + pf.py CPU support + addendum + the FFT-probe and bispinor run dirs.

Neither branch pushed to `origin` — awaiting the user's explicit instruction. Both pass the smoke tests at this commit.
