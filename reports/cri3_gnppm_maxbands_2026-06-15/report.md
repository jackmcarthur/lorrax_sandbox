# CrI3 6×6 80 Ry GN-PPM — max bands before OOM on 16× A100-80GB

**Date:** 2026-06-15 · **Branch:** `lorrax_A` `agent/cri3-ppm-maxbands` (rebased to `e85be60`)
**Run dir:** `runs/CrI3/6x6_80Ry_gnppm_maxbands_2026-06-15/` · **Alloc:** JID 54541850 (4 nodes / 16× A100-80GB, 4 h)

## Summary

Goal: how many bands a CrI3 6×6 GN-PPM GW completes before OOM on 16× A100-80GB,
with `n_centroids (μ) = 10·nband`, for non-bispinor **and** bispinor; preliminary goal
to identify the binding memory peak.

**Three headline results:**

1. **Binding peak = `C_fit_one_rchunk`** (the ISDF pair-density fit transient
   `3·16·nk·ns²·μ·r_chunk / p_xy`), *not* the V_q buffer (Peak E). Confirmed: the
   production driver's own planner printed `HWM = 56.00 GB/dev, bottleneck = C_fit_one_rchunk`
   at nb150 — byte-matching a standalone `plan_gflat_chunks` sweep.

2. **Non-bispinor ceiling: empirically 700–798 bands** (16× A100-80GB, BFC pool 76 GB).
   **nb700 (μ≈6986) FITS** — passed ζ-fit + V_q (n_q_ibz=8) in ~24 min, stopped only at the
   head; **nb798 (μ≈7980) OOMs** at the ζ-fit (a single 60.5 GB allocation fails, ~3 min).
   The planner's ~640 prediction is **conservative by ~60–150 bands**: it assumes full-BZ
   V_q (36 q) but the runtime cascade uses IBZ (8 q), and it over-charges the Peak C
   transient (nb700 fit at planner HWM=79.2 GB = 113% of budget; nb798 OOM'd at 91.0 GB =
   130%). The binding peak is still `C_fit_one_rchunk` — `r_chunk` floors at `max_chunks=64`
   (≈17.5k) and HWM climbs with μ past nb600.

3. **Bispinor OOMs ~3× SOONER than non-bispinor — empirical ceiling 200–300 bands.**
   nb200 (μ2000) clears the binding charge ζ-fit; nb300 (μ2992) OOMs at it (`fit_zeta`,
   90.8 GB alloc). The production driver picks r_chunk=19536 (58 chunks) for bispinor vs
   73824 (16) for non-bispinor at nb150 — a ~3.8× steeper Peak C slope, because the 4 ζ
   channels (charge + 3 transverse) inflate the pair-density transient. CrI3 being fully
   relativistic (`nspinor=2` in both modes) means the *non-bispinor* ceiling is already as
   high as it gets; the transverse channels are what pull the bispinor ceiling down.

   > **Note / caveat:** A standalone `plan_gflat_chunks(is_bispinor=True, ...)` call
   > (per the test harness) does **not** reproduce this — it returns the *same* r_chunk as
   > non-bispinor (the picker's `α_C` uses `pair_density_slots_transverse`, which also
   > resolves to 3 on GPU). The ~3.8× only appears in the in-driver (`gw_init.py:606`) call.
   > The bispinor ceiling above is derived from the **production** r_chunk=19536 data point
   > + the chunk-floor climb model, not from the standalone sweep. The exact value needs a
   > live bispinor sweep (blocked by the centroid-prune OOM). This standalone-vs-production
   > bispinor discrepancy is itself worth a planner follow-up.

## Setup

| | |
|---|---|
| System | CrI3 monolayer, 2D, P-3 (6 ops), nval=70, 8 atoms |
| Cutoff / grid | ecutwfc=80 Ry, 6×6×1 k (8 IBZ / 36 full BZ), ngkmax=59990, FFT (75,75,200), n_rtot=1.125M |
| NSCF | **800 bands** (multiple of 16), npool=8 (one GPU per IBZ k-pool), JOB DONE in 28 s; reused 80 Ry SCF charge density |
| WFN.h5 | 12.3 GB, verified mnband=800, nrk=8, nspinor=2, ngkmax=59990 |
| GW mesh | 4×4 device mesh (p_xy=16), `compute_mode=gn_ppm`, `memory_per_device_gb=70`, `bare_coulomb_cutoff=80` |

Pipeline: `qe+gwjax` (no BGW). **This is the first CrI3 GN-PPM run in the sandbox** —
all prior CrI3 runs were x-only / static COHSEX.

## Planner prediction (driving real `gw.gflat_memory_model.plan_gflat_chunks`)

Budget 70 GB/dev, GPU `pair_density_slots=3`, μ=10·nband. This is the **non-bispinor**
curve (validated below: nb150 production r_chunk=73824 matches). The standalone sweep
returns the same for bispinor, but production charges bispinor ~3.8× more (see Result 3) —
so the table below is the **non-bispinor ceiling (~600–640)**; bispinor lands ~3× lower:

| nband | μ | r_chunk | n_chunks | HWM GB/dev | fit (≤70 → ≤80 phys) |
|---|---|---|---|---|---|
| 150 | 1504 | 74496 | 16 | 56.0 | ✅ |
| 300 | 3008 | 34624 | 33 | 56.0 | ✅ |
| 500 | 5008 | 18512 | 61 | 56.0 | ✅ |
| 600 | 6000 | 17568 | 65* | 64.8 | ✅ |
| 700 | 7008 | 17568 | 65* | **76.1** | ❌ OOM |
| 798 | 7984 | 17568 | 65* | 87.2 | ❌ OOM |

\*`r_chunk` floored by `max_chunks=64`. Per-peak at nb500 (μ=5008): Peak C 56.0 (P_pair
40.1 + gflat_acc 10.8), Peak D 19.5, Peak A 19.4 (fft_box 18.4), Peak E 14.7, Peak B 9.9.
Raising `max_chunks` would push the ceiling higher at runtime cost — the ceiling is a
chunk-count floor, not a hard μ² wall.

## Empirical validation (16× A100-80GB, BFC + MEM_FRACTION=0.95 → 76 GB pool)

Ran the GW driver across the band sweep (μ ≈ 10·nband; centroids generated with
`--oversample 1.0` — see Blockers). OOM is detected **fast** (~3 min — the ζ-fit's first
big pair-density buffer fails immediately); fitting runs take ~24 min (non-bisp) / ~90 min
(bispinor, 4 ζ channels) to clear ζ-fit + V_q. A run that fits stops at the q=0 Coulomb head
(`Failed to resolve q=0 Coulomb head`) — **not** OOM — for lack of dipole.h5/overrides; the
head is downstream of the binding ζ-fit/V_q, so "stopped at head" = "did **not** OOM".

| mode | nband | μ | planner HWM (% of 70 budget) | result |
|---|---|---|---|---|
| non-bisp | 150 | 1500 | 56.0 (80%) | FIT (ζ+V_q, stop at head) |
| non-bisp | **700** | 6986 | 79.2 (113%) | **FIT** — ζ-fit + V_q (IBZ, n_q=8) in ~24 min |
| non-bisp | **798** | 7980 | 91.0 (130%) | **OOM** at ζ-fit (60.5 GB single alloc, ~3 min) |
| bispinor | 150 | 1504 | 56.0 (80%) | FIT (charge ζ passed) |
| bispinor | **200** | 2000 | 67.6 (97%) | **FIT** — passed binding charge ζ-fit (transverse+V_q ran) |
| bispinor | **300** | 2992 | 101.5 (145%) | **OOM** at ζ-fit `fit_zeta` (90.8 GB alloc, ~3 min) |

**Empirical ceilings (the answer):**
- **Non-bispinor: 700–798 bands** — 700 fits, 798 OOMs.
- **Bispinor: 200–300 bands** — 200 clears the binding ζ-fit, 300 OOMs at it. **~3× lower**
  than non-bispinor, confirming the steeper bispinor Peak C slope (4 ζ channels).

Both ceilings are set by **`C_fit_one_rchunk`** (the ISDF pair-density transient) — the OOM
tracebacks are in `fit_zeta` / `RESOURCE_EXHAUSTED` during the ζ-fit, exactly the predicted
binding peak.

**The planner is conservative** for non-bispinor (~640 predicted vs 700–798 measured): nb700
fits at planner HWM = 113% of budget; nb798, the first OOM, is at 130%. Two reasons — the
runtime V_q cascade is IBZ-reduced (n_q_ibz=8) while the planner assumes full-BZ (36 q), and
it over-charges the Peak C transient. For **bispinor the in-driver planner tracks well**
(nb300 @145% OOMs, nb200 @97% fits) — it *does* capture the bispinor slope; only the
*standalone* `plan_gflat_chunks` call doesn't (Result 3 note).

## Timing breakdown (16× A100-80GB unless noted)

**Pipeline setup (one-time):**

| step | config | wall |
|---|---|---|
| QE NSCF | 800 bands, 8 IBZ k, npool=8 (8 GPUs) | **28 s** |
| wfn2hdf (WFN binary → WFN.h5, 12.3 GB) | 1 GPU | **45 s** |
| centroid gen (`--oversample 1.0`, Lloyd) | 1 GPU | 46 s (μ2000) · 100 s (μ6000) · 130 s (μ7980) |
| planner prediction (standalone) | 1 GPU | ~1–2 min |

**GW runs — the binding cost is the ζ-fit, which scales ~linearly with μ:**

| run | result | total wall | breakdown |
|---|---|---|---|
| non-bisp nb700 (μ6986) | FIT | **~27 min** | compile+load+C_q+Cholesky ~1 min · **ζ-fit 1245 s (≈21 min)** · V_q + Σ + head ~5 min |
| non-bisp nb798 (μ7980) | OOM | **~3 min** | OOM at the ζ-fit's first pair-density alloc (60.5 GB) — fails at compile/first-exec |
| bispinor nb300 (μ2992) | OOM | **~3 min** | OOM at charge `fit_zeta` (90.8 GB) |
| bispinor nb200 (μ2000) | FIT (full) | **~36 min** | **4 ζ-fits: 448 + 419 + 415 + 416 s = ≈28 min** (charge + γ̃¹⁻³) · 7-tile V_q + Σ^B + head ~6 min |

**Takeaways:**
- **OOM is detected in ~3 min** — the ζ-fit's first big pair-density buffer fails at
  compile/first-execution, so probing the boundary is cheap (fitting runs are the slow ones).
- A *fitting* run is **dominated by the ζ-fit**, which scales ~linearly with μ: 1245 s at
  μ6986 vs 448 s at μ2000 (per channel, both 65 r-chunks at the `max_chunks=64` floor).
- **Bispinor costs ~4× the ζ-fit wall** of non-bispinor at the same nband (4 *sequential* ζ
  channels, ~7 min each) plus the 7-tile V_q — the "~4× runtime, lower ceiling" picture.
- V_q + sigma + head are a small tail (~5–6 min) in both modes.

## Centroid generation for large μ (how the boundary was unblocked)

The documented `kmeans_cli <N> --seed 42` OOMs at the pivoted-Cholesky prune in **two**
places (both logged to `KNOWN_SANDBOX_ERRORS.md`):
1. **cuFFT scratch OOM** (16.5 GB plan) under the platform allocator. **Fixed** by a new
   `--prune-mem-gb` flag (threaded `memory_per_device_gb` through
   `kmeans_cli → prune_candidates_by_pivoted_cholesky → build_gram_q0_via_loadwfns`) — a low
   budget shrinks the FFT chunk `cs ∝ budget` (the OOM's `cs=984` came from the 71 GB
   auto-budget). Commit on `agent/cri3-ppm-maxbands`.
2. **Gram-build OOM** (173 GB for ~9000 candidates) — exposed once (1) is fixed; the prune's
   candidate-axis Gram is unchunked. Not yet fixed.

**Workaround used for the sweep:** `--oversample 1.0` skips the prune entirely (kmeans runs
N_c reps directly). Valid here because an OOM/memory test depends only on the centroid
**count**, not the prune-improved values. Generated μ ∈ {2000, 2992, 5984, 6986, 7980}
(+ current-density) this way, ~1–2 min each.

## Remaining blocker — full QP completion only (NOT the boundary)

**dipole.h5 generation OOM** — GN-PPM needs a q=0 Coulomb head from dipole.h5 / eps0mat.h5 /
explicit overrides. `psp.get_dipole_mtxels` (the only runnable entry — `get_dipole_mtxels_chunked.py`
does **not exist**, contra the skill) materializes the full 181 GiB ψ-box and doesn't shard
→ OOM on 1 and 16 GPUs. So every run stops at the head. **This is downstream of the binding
ζ-fit/V_q, so it does not affect the OOM-ceiling result** — it only blocks producing eqp0.dat.
Fix: k-chunk the `read_Gvecs_to_devices` call in `get_dipole_mtxels.main()` (the function
already takes a `k_range`; the per-k loop at line 582 already processes one k at a time).

Also logged: JAX version triple-mismatch, stale QE module names, wrong `*_chunked`
preprocessing module names.

## Status

- [x] Rebase old-main checkout → current main `e85be60`
- [x] QE NSCF 800 bands → WFN.h5
- [x] Planner OOM-vs-nband (both modes) — bottleneck = `C_fit_one_rchunk`
- [x] Fixed centroid-prune cuFFT OOM (`--prune-mem-gb`); large-μ centroids via `--oversample 1.0`
- [x] **Empirical OOM ceilings: non-bisp 700–798, bispinor 200–300** (both set at the ζ-fit)
- [ ] Full QP completion (eqp0.dat) — blocked by dipole/head (downstream, non-memory)

## Next steps

- k-chunk `psp.get_dipole_mtxels` (use its `k_range`) → head resolves → runs complete to eqp0.dat.
- Narrow the ceilings further if wanted: non-bisp nb750; bispinor nb250.
- Chunk the prune's candidate-axis Gram build so `kmeans_cli` works at large μ without `--oversample 1.0`.
- Optionally raise `max_chunks` (>64) to trade runtime for a higher ceiling.
