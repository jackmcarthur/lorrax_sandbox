# Round 4 — Agent 4: Prioritized next-tasks queue + profiling plan

Synthesis of Path D landing day (2026-05-13).  Path D shipped on
`lorrax_B` `agent/zeta-bc-scan-shardmap` @ `5cadd4b`: 200.35 GiB → 48.63
GiB peak preallocated-temp at CrI3 6×6 80 Ry, OOM → progressing
through 16 r-chunks at ~1 min each.  The structural fix worked on its
own terms (Agent 1 `round4_improvements.md` §2: 58 → 1 FFT-box slots,
~152 GiB eliminated).  What's left is the **next-layer principle
violations** that the bc-loop collapse exposed at coarser
granularity, plus the morning Defects 4–5 that were always going to
land next, plus the profiling baseline that should anchor any
further structural work.

References (all in `reports/zeta_rchunk_memory_model_2026-05-13/`):
- Agent 1 `round4_improvements.md` — HLO before/after, where the 48 GiB lives.
- Agent 2 `round4_code_state.md` — lorrax_A↔B reconciliation, cohsex.in surface diff.
- Agent 3 `round4_memory_model_state.md` (in progress at write-time;
  full headline already in `round4_discussion.md`).  Three planner
  bookkeeping bugs identified, plus confirmation that the 12.07 GiB
  unsharded FFT-box slot is the cost-side of the remat warning and
  is *XLA-SPMD-emergent* — the planner can't model it directly.
- Agent 4 `defect_catalog.md` (morning) — Defects 4 / 5 / 6.

## 1. Top of the queue (land before the next structural change)

### T1.  Remat boundary fix at `gflat_to_rchunk` → `z_q_from_psi_sm`

**Where**: `src/common/wfn_transforms.py:765` (helper's `out_spec`),
`src/common/isdf_fitting.py:1292` (consumer's slice into L/R).
**Cost**: ~30 GiB of the AFTER pool (Agent 1 §3a) is two copies of
`psi_Y_full` `c128[36, 160, 2, 73648]` (14.85 GiB each) live
simultaneously.  Source: helper emits `P(None, ('x','y'), None,
None)` (band-flat-XY); consumer wants `P(None, 'x', None, None)`
(L-side) or `P(None, None, None, 'y')` (R-side).  XLA can't plan the
band-XY → r-Y axis-swap and remats the full tensor instead.  32
`Involuntary full rematerialization` warnings in `gw.out`.
**Payoff**: ~15 GiB measured (eliminating the second materialization);
~5 GiB more if the helper can emit directly in the consumer's
sharding so no reshard is needed.  Brings the pool from 44.56 GiB to
~25 GiB.
**Rationale**: this is the §0 principle violation introduced *by*
Path D.  It's the immediate next defect, not a coupon to spend later.
**Effort**: ~half-day.  Two approaches sketched in
`round4_improvements.md` §5 — preferred is teaching `gflat_to_rchunk`
to take a target-output `out_spec` (e.g. `P(None, None, None,
('x','y'))`) and emit two-step reshards inside the shard_map that
XLA can plan; the slice + downstream shard_map then sees the right
layout for free.  Fallback: emit a *second* helper variant
(`gflat_to_rchunk_r_sharded`) with the r-axis-sharded output for the
consumer; keep band-XY for any other caller.

### T2.  Engage `chunk_size` on `gflat_to_rchunk` / `gflat_to_rmu`

**Where**: `src/gw/gw_init.py:635-646` — the auto-picker for
`gflat_to_rchunk_chunk_size`.  Agent 3 confirms the picker is
principled (per-row FFT-box bytes vs 50% budget) but at CrI3 6×6 80
Ry on a 4×4 mesh it picks cs_auto=854 vs N_rows=360 → effective
**one-shot**, materializing the 12.07 GiB `c128[360, 2, 75, 75, 200]`
slot Agent 1 §4 observed.
**Cost** today: the auto-picker over-budgets — it sized against the
single-helper 50% budget without accounting for the +2× remat
multiplier T1 still has to fight.  Once T1 lands and the 30 GiB
remat goes away, the auto-picker's headroom is genuine; before that
it underestimates because `cs=N` lets two `psi_Y_full` slots live
concurrently.
**Payoff**: ~10 GiB further (Agent 1 §4).  Combined with T1 brings
pool to ~12-15 GiB on a 4×4 mesh, comfortably inside HBM40.
**Approaches** (pick one after T1):
- (a) Lower the auto-picker's budget fraction so it returns cs < N
  at CrI3 scale.  Single-constant change in `gw_init.py:635-646`.
- (b) Leave the picker, but expose `gflat_to_rchunk_chunk_size` in
  cohsex.in (already named — Agent 2 §3 confirms) and document a
  recommended setting (e.g. cs ≈ nk · ⌈nb_local / 4⌉ on this mesh).
- (c) Refactor toward N3 (io_chunk / fft_chunk split) and pick
  fft_chunk from cuFFT throughput profiling (P1).  Strategically
  preferred but blocks on P1.
**Effort**: ~2 hours for (a)/(b), ~half-day for (c) gated by P1.

### T3.  Planner bookkeeping fixes (one commit, three sites)

**Where** (per Agent 3 in `round4_discussion.md`):
- `gflat_memory_model.py:L148` `_peak_B_cct_chol["centroids_persistent"]`
  — uses `(nk, ns, μ, ns)` should be `(nk, ns, μ, nb_total)`.
- `gflat_memory_model.py:L184` `_peak_C_fit_one_rchunk["centroids_persist"]`
  — same typo (uses nk as band dim).  Under-counts centroids by
  `nb_total/nk` ≈ 10× at CrI3 6×6 80 Ry.
- `gflat_memory_model.py:_peak_D_accumulate` — remove `fft_box_factor=4`
  multiplier on the accumulate-FFT term.  XLA-fused FFT, no 4×
  scratch (Agent 3).
- Also: cherry-pick `_bytes_centroids_LR` from `lorrax_A` `ff5873c`
  (Agent 2 §4) — the formula `nk·nb·ns·μ/p_y + nk·μ·nb·ns/p_x` for
  the two persistent centroid copies on disjoint mesh axes.  Lands
  in the same commit; together they fix the doubly-broken centroids
  accounting Agent 3 flagged.

**Cost** today: planner's HWM prediction is 51.93 GB vs measured
48.63 GiB (within 7%, Agent 1 §4) but the agreement is *coincidental*
— two compensating bugs (centroids under-counted by ~10×; Peak D's
`fft_box_factor=4` over-counts accumulate scratch).  After T1+T2 the
real HWM drops to ~15 GiB and the planner needs to track that
honestly or it'll keep picking conservative chunks.

**Payoff**: ~5 GiB of *predicted* HWM recovered (Agent 2 §4); planner
picks slightly smaller r_chunk after the Peak B/C centroids fixes,
slightly larger gflat_chunk_size after the Peak D `fft_box_factor`
removal.  Net: planner accuracy.

**Rationale**: low-risk, one-file, mechanical (~10 lines per Agent 3).
The remat boundary @ wfn_transforms.py:765 is the only line item the
planner CAN'T model and must be fixed structurally (T1); these
bookkeeping changes make the planner honest about everything *else*.

**Effort**: ~1 hour.

## 2.  Next-priority block (after the top of queue lands)

### N1.  Defect 4 — `solve_zeta` q-batch Python loop

`isdf_fitting.py:1118-1141`.  **Defect 4 in `defect_catalog.md`**.
Recorded 3-way trap (scan-unroll-8 → 2×preallocated-temp alive;
unrolled scan → SPMD replication ~88 GB; fori_loop → WhileOp
replication).  All three pre-date Path D and likely tried the
*outer-level* scan, not scan-inside-shard_map.

**Payoff**: unbounded at scale.  At CrI3 6×6 80 Ry the q-batch
materializes `c128[q_batch, μ_padded, n_rchunk] = c128[1, 1504, 73648]`
≈ 1.7 GiB per iter × N_qchunks live concurrent slots.  Not the
current OOM driver (the bc-loop is what blew up); becomes the limiter
at 8×8 k-grids where μ ~3000 and r_chunk ~150k.

**Effort**: medium-high (half- to full-day).  Open question whether
the Path D pattern (scan-inside-shard_map) transfers cleanly — the
q-axis is the leading axis and the inner cho_solve is already
shard_map-decorated, so re-spec'ing the sharding inside the scan
body is the natural attempt.  Alternative: drop the loop entirely
and use the existing `solve_all_at_once` path with a stricter
feasibility gate when the cuSolverMp batched LU/Chol is available.

### N2.  Defect 5 — `_v_q_per_q_g_chunked_jit` G-loop

`gw/compute_vcoul.py:589-625`.  **Defect 5 in `defect_catalog.md`**.
Cheap `lax.fori_loop` swap, ~300 MB on full mesh.  Single-output
carry, no donation hazard.

**Payoff**: small (~300 MB CrI3-scale) but **fast cleanup**.  Good
"confidence-builder" follow-up after Path D's fori_loop / scan
ergonomics lessons land.

**Effort**: ~1 hour.  Replace `for i in range(n_chunks)` with
`V = lax.fori_loop(0, n_chunks, body_with_idx, V_acc)`.  Add a unit
test mirroring the n_chunks divisor / non-divisor / pad cases from
`test_gflat_to_rchunk_chunked_matches_oneshot`.

### N3.  io_chunk / fft_chunk split (user's idea, this morning's orchestrator convo)

**Where**: `src/common/wfn_transforms.py` `gflat_to_rchunk` and
`gflat_to_rmu` (both currently use a single `chunk_size` for both
the io_callback batch and the FFT batch).

**Motivation**: in the `gw.out` planner output for CrI3 6×6 80 Ry,
`G-flat ζ sphere: ngkmax=59990, min ngk=59826, max ngk=59990 (5.332%
of n_rtot)`.  ψ(G_sph) is ~5–10% of the FFT-box payload — so the IO
batch can be **~10–20× larger** than the FFT batch without raising
the per-iter HBM footprint.  Today a single `chunk_size` couples
them.

**Sketch**: nested scan inside the shard_map body —
```text
shard_map:
  pull psi_G slice via io_callback (large io_chunk rows)
  lax.scan over io_chunk in fft_chunk-sized inner blocks:
    body:
      _box_kernel + ifftn + slice/sample + Bloch phase
      dynamic_update_slice into out_flat
```
Outer scan = I/O batching (cheap; one ψ(G_sph) carry).  Inner scan =
FFT batching (expensive; per-iter FFT box at fft_chunk-sized extent).

**Open question — needs measurement first** (see P1 below):
- If cuFFT throughput at `fft_chunk=1` is acceptable, the inner scan
  is the path; chunk size becomes a tuning knob, default ~1 minimizes
  HBM.
- If cuFFT throughput collapses at `fft_chunk=1`, find a knee
  experimentally and bake that as the inner-scan default.

**Payoff**: another factor of ~10× on the FFT-box slot — but only
matters at the scales where Path D's current `chunk_size` is still
the limiter (e.g. 8×8 k-grids).  At CrI3 6×6, T1+T2 are enough; this
is the structural-cleanup follow-up.

**Effort**: medium.  ~half-day for the implementation if cuFFT
profiling (P1) says inner scan at fft_chunk≪io_chunk pays.  Could be
deferred behind T1 / T2 / N1 if profiling shows the simple
chunk_size knob is sufficient.

### N4.  Stopgap-term comment cleanup + COHSEX_INPUT.md doc

**Where**: `src/gw/gflat_memory_model.py` `_peak_C_fit_one_rchunk` —
the `band_fft_pool` / `band_fft_unsharded` term is already absent on
`lorrax_B` per Agent 2 §3, but the planner doc / in-file comments
still describe the term as if it were live.  Strip the stale
commentary.  Also `docs/docs_gwjax/COHSEX_INPUT.md` needs
`gflat_to_rchunk_chunk_size` documented and `psig_k_chunk_size`
marked deprecated (Agent 2 §3 — proposes a deprecation warning in
`gw_config.py` for stale cohsex.in fields).

**Payoff**: zero memory; reduces user-visible confusion and removes a
dead branch from the planner.  Pair with the T2 default-picker —
they touch the same file.

**Effort**: ~1 hour.

### N5.  Defect 6 — Davidson `_ortho_expand` CGS2 unroll

`solvers/davidson.py:151-178`.  **Defect 6 in `defect_catalog.md`** —
2-iter unroll inside `@jax.jit`.  Confirmed leave-alone: BSE-side,
intermediates are small `(m, n)` rank-2 buffers, XLA aliases the 2
iters trivially.  Not an OOM driver at any scale.

**Action**: confirm with a one-line comment in the source code
noting "intentional 2-iter unroll, see `defect_catalog.md` §Defect
6" so future audits don't re-flag.  Or leave it as the catalog entry
is authoritative.

### N6.  Pair-density carry — `c128[36, 16, 2, 59990]` ×8 slots

Agent 1 §3c: 8 distinct preallocated-temp slots of 1.03 GiB each
(8.2 GiB total) inherited from `z_q_from_psi_sm`'s post-pair pipeline
(rank-7 IFFT/FFT chain).  Not in Path D scope but now visible.  Has
the same shape signature as `pair_density_slots`-class but with
`nk=36` (un-k-chunked) instead of `nk=6` (k-chunked).

**Effort**: medium-high; structural inside `z_q_from_psi_sm`.  Defer
behind everything above unless Agent 3's planner audit flags it as a
limiter at 8×8 k-grids.

## 3.  Profiling block (measurements to ground the next structural change)

### P1.  cuFFT throughput vs `chunk_size` (user's explicit ask)

**Question**: can we run `lax.scan(fft_chunk=1)` cheaply, or does
cuFFT throughput collapse below some knee?  Drives the
**fft_chunk=1 vs nested-scan decision** for N3.

**Measurement**: write a minimal jax program that times
`jnp.fft.ifftn` on `c128[N_rows, ns, nx, ny, nz]` for
- shape: `ns=4`, `nx=ny=96`, `nz=120` (CrI3 6×6 80 Ry FFT grid).
- `N_rows ∈ {1, 2, 4, 8, 16, 32, 64, 128, 360}`.
- mesh: 4×4 (16 ranks), one A100/rank.
- norm: 'ortho'.

Report tFLOPS achieved + wall time per IFFT × N_rows; look for the
knee.  Run command:
```bash
lxrun --jid=$JID python tests/bench_cufft_chunk_size.py \
  --fft-grid 96 96 120 --ns 4 --rows 1,2,4,8,16,32,64,128,360
```
(test script doesn't exist yet; would write under `tests/` or
`runs/profiling/`).

**What we'd learn**: whether N3 (nested scan) is justified vs T2
(single chunk_size knob) at this physics scale.

### P2.  End-to-end wall time per r-chunk (decompose the 1-min cost)

**Question**: where are the seconds going inside `fit_one_rchunk`?
Path D made the kernel fit; making it fast is the next question.

**Measurement**: enable existing `timing.section` blocks in the
`fit_one_rchunk` kernel — the wrappers are present in `load_wfns.py`
(`load_centroids.loader_load`, `.gflat_to_rmu`, `.reshard`) and need
to be added at the per-stage boundary in
`fit_one_rchunk._kernel`.  Stages:
- (a) `gflat_to_rchunk` (bc-load FFT + Bloch phase)
- (b) `z_q_from_psi_sm` pair density + IFFT/FFT chain
- (c) `solve_zeta` Cholesky / LU
- (d) `accumulate_rchunk_to_gflat` (downstream of the kernel; the
  outer driver section)

Run the same CrI3 6×6 80 Ry config; print per-r-chunk breakdown.

**What we'd learn**: which of (a-d) dominates the 1 min/r-chunk
cost; ranks the structural-next work alongside the §1/2 priorities.
If (a) dominates → T2/N3 directly help wall time; if (b) dominates →
N6 / Agent 3's planner refinements take priority; if (c) → Defect 4
becomes higher than its memory ranking suggests.

### P3.  HBM HWM via `nvidia-smi` — measured vs planner

**Question**: is the 51.93 GB planner prediction conservative or
optimistic against the *actual* HBM peak (including XLA's reserved
scratch + cuFFT plan caches not in the memory-usage-report)?

**Measurement**: during the CrI3 6×6 80 Ry run, sample every 1s:
```bash
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader \
  --loop-ms=1000 > hbm_trace.csv &
```
Take the per-GPU peak across the run.  Compare against the
memory-usage-report's 48.63 GiB and the planner's 51.93 GB.

**What we'd learn**: the gap (if any) between XLA's accounting and
the real OS-visible HBM peak.  If `nvidia-smi` peak >> 48.63 GiB,
there's hidden scratch we haven't modeled.  If it matches or is
lower (e.g. XLA over-reserves), the planner can be tightened.

### P4.  NCCL collectives volume — count + size in new HLO vs baseline

**Question**: did the bc-loop → scan-inside-shard_map fix add new
all-gathers / all-reduces at the boundary?  Agent 1 §3b implies the
remat is one-sided (no collective, just remat-replicate); we should
verify.

**Measurement**: dump HLO with `XLA_FLAGS="--xla_dump_to=...
--xla_dump_hlo_pass_re=spmd"` already enabled.  Grep
`module_0394.jit__kernel.*-after-optimizations.txt` for `all-gather`,
`all-reduce`, `all-to-all`.  Sum the per-call byte volumes.  Compare
to the same grep on `module_0408.jit__kernel.*` (baseline).

**What we'd learn**: whether Path D's net network volume is up or
down.  Memory dropped 4× but if collective volume went up 2×, wall
time may not scale.

### P5.  Compile time at scale

**Question**: how long is the first-call HLO compile in the new
pipeline vs baseline?  Path D introduces scan-inside-shard_map which
JAX historically compiles slower than the unrolled equivalent.

**Measurement**: log wall time of the first `_kernel` jit-compile in
the CrI3 6×6 80 Ry run (already in `gw.out` if `timing.section`
captures the first call; otherwise add a `jax.block_until_ready` +
timer around the first `fn(...)` invocation).  Compare to
`lorrax_A`'s baseline first-compile time.

**What we'd learn**: whether scan-inside-shard_map has acceptable
compile overhead at this scale.  Path D §1 ("compile times can grow
non-linearly") was a concern flagged in `PATH_D_PICKUP.md`.

## 4.  Out of scope for now (deferred deliberately)

- **Davidson `_ortho_expand` 2-iter unroll** (Defect 6).  See N5;
  no measurable cost.
- **The 11 "not-violations" in `defect_catalog.md` §4**.  V_q tile
  q/μ/ν driver loops, sigma τ loop, kpath loops, kmeans axis-name
  loops, etc.  None have new evidence to re-flag.
- **CCT-side rewrite of `c_q_from_psi_sm`**.  Out-of-scope per
  `round3_integration.md` §"Out of scope".  Wait for T1 + N3 to
  settle the helper sharding-spec story; the CCT side will inherit
  whatever pattern proves out there.
- **`io_callback` async pipeline** in `load_centroids_band_chunked`.
  Tried at MoS2 3×3 scale (`load_wfns.py:747-754` comment), 0.000
  overlap observed.  Revisit only if P2 shows IO time dominates.
- **3D bulk MC-averaged head correction** (`compute_vcoul.py:324-326`
  triple loop, pure numpy host).  Setup-time only, no JAX trace
  pollution.
- **Per-q V_q tile q/μ/ν driver loops**.  Donation chain pattern
  already prevents pile-up; not Path D's mechanism.

## 5.  The CrI3 6×6 80 Ry validation gate ("Path D genuinely done")

A single pass/fail box:

1. **End-to-end completes cleanly**.  No OOM; no SPMD warnings
   (T1 removes the remat warnings); writes `eqp0.dat` + `WFN_qp.h5`
   + `zeta.h5` + `V_q.h5` to disk.
2. **HBM HWM < 28 GB / rank** (the HBM40-class budget, with
   `target_utilization=0.80`).  Measured by P3.
3. **Σ matches lorrax_A baseline** to existing tolerance — i.e.
   eqp0.dat values agree within COHSEX tolerance with a known-good
   baseline (the lorrax_A `agent/zeta-r-chunk-fixes-2026-05-13`
   run, or a pre-Path-D commit before the planner stopgap landed).
4. **No remat warnings in `gw.out`** — the 32 lines that appear
   today must be gone.
5. **Compile time < 15 min** — soft gate, indicative.  Hard fail at
   60 min.

Reaching all five flips the "Path D genuinely done" bit.  T1 + T2
alone are likely enough to flip 1-4; T3 + Agent 3's planner work
tightens the model for downstream chunk-size picks at larger
scales.

## 6.  Priority recap (single column)

1. **T1** Remat boundary fix `gflat_to_rchunk` → consumer (~15 GiB, half-day)
2. **T2** Engage `chunk_size` on new helpers (~10 GiB, 2 hours)
3. **T3** Cherry-pick `_bytes_centroids_LR` (~5 GiB predicted, 1 hour)
4. **P1** cuFFT throughput sweep (decides N3 design)
5. **P2** Per-r-chunk wall-time decomposition (decides N1 vs N6 priority)
6. **N1** Defect 4 — solve_zeta q-batch
7. **N3** io_chunk / fft_chunk nested scan (gated by P1)
8. **N2** Defect 5 — `_v_q_per_q_g_chunked_jit` (cheap cleanup)
9. **N4** Planner / docs cleanup + cohsex.in deprecation
10. **N6** Pair-density carry rewrite (gated by P2 if it dominates wall time)
11. **P3 / P4 / P5** profiling baselines that should run alongside any structural change above

After 1-3 + 4 + 5 land, take the **CrI3 6×6 80 Ry validation gate** (§5)
and call Path D done.  N1 / N3 / N6 are the *next* structural epoch.

Agent 4 round 4 done
