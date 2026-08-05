# Agent L — Round 5 live verification of Round-4 commits

**Branch:** `agent/bispinor-ibz` (lorrax_B HEAD: `81817e2`)
**System:** CrI3 6×6×1 80 Ry SOC bispinor, 16 GPUs (4×4 mesh, hbm80g)
**Run dir:** `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**JID:** 53082914 (consumed ~1:30 / 2:00 wall)

## Verdict

**ROUND 6 NEEDED: planner sphere-idx constant `N_SPHERE_IDX_BUFFERS_*=1` undercounts the
post-fix steady-state — actual is 3 buffers, not 1.** All four configs ran
cleanly with no OOM and good live-vs-predicted agreement, but the model's
sphere-idx persistent term is short by `2 × 0.162 GB/dev = 0.324 GB/dev`
(`5.18 GB global`). The Round-4 prediction that "agent_k V1 picks r_chunk=78272"
is **also wrong** — the natural picker selects **r=20688**, not 78272. Cache fix
itself works (sphere count is bounded at 3, no growth across channels).

## 1. Per-config table

`live_total` = `Σ jax.live_arrays()` worst across all probes/channels (global GB).
`peak_bytes_in_use` returns `-0.00 GB` on this JAX/CUDA stack (already noted by
agent_j) — cross-check uses live-arrays composition + Round-3 baselines.
HWM-pred-vs-runtime cross-check via live composition: live_total = pre_rchunk_loop
baseline + Peak C zeta_chunk transient. The `Δ_live` column compares against
Round-3 (Agent J) results at the matching chunk plan.

| config | r_chunk | b_chunk | cs (plan/run) | HWM_pred (GB/dev) | bottleneck | live_total worst (GB global) | Δ_live vs R3 | sphere-buffer count | verdict |
|---|---|---|---|---|---|---|---|---|---|
| **W4** safety net | 20256 | 64 | 100 / 100 | **54.83** | C | **76.79** | -0.82 (R3 V4: 77.61) | **3** | PASS (R3-reproduce) |
| **W2** sweet spot | 24576 | 32 | 100 / 100 | **66.41** | C | **80.53** | -0.83 (R3 V3: 81.36) | **3** | PASS |
| **W3** cs=200 override | 24576 | 32 | 200 / 200 | **66.41** | C | **80.53** | n/a (R3 V5: r=20256) | **3** | PASS + cap-warning fired |
| **W1** planner-natural | **20688** (not 78272) | 64 | 100 / 100 | **55.99** | C | **77.16** | +0.70 (R3 V1: 76.46) | **3** | PASS (but agent_k claim wrong) |

**HWM-pred deltas R3 → R5 are uniformly ~-1.13 GB/dev** (exactly 7 sphere
buffers × 0.162 GB/dev). The planner removed sphere buffers in its formula
correctly. Live-arrays peaks drop by ~0.82 GB global = 5 buffers worth, NOT
7 buffers worth — observed sphere count is 3, not 1.

W3 cap-warning capture (also reproduced verbatim in §3):
```
[plan_gflat_chunks] WARNING: gflat_chunk_size overridden to 200 (cap was 100);
past the cuFFT plan-algorithm crossover at cs ~ 1000 cuFFT scratch grows
non-linearly (agent_f cs=1414 OOM verified).  Peak D at overridden cs ≈
12.75 GB/dev (budget 70.00 GB/dev).
```

W2 unexpectedly also fired an `r_chunk` override warning — the planner correctly
recognises that `r_chunk=24576 > 20688 = headroom-cap` and recomputes Peak C
at the override. Reasonable behavior.

## 2. Sphere-idx count audit (end-of-run, all 4 channels)

| stage / channel | sphere-buffer count | observed bytes (global) | predicted (planner) |
|---|---|---|---|
| zeta_fit_start ch0 (charge) | 2 | 0.32 GB | 1 buffer / 0.162 GB |
| pre_rchunk_loop ch0 | 3 | 0.49 GB | 1 buffer / 0.162 GB |
| after_fit ch0..ch3 (all r-chunks) | **3** (stable) | 0.49 GB | 1 buffer / 0.162 GB |
| zeta_fit_end ch3 (last) | **3** | 0.49 GB | 1 buffer / 0.162 GB |

**Cache fix verified working** — count is BOUNDED at 3 (no growth from ch0 to
ch3). Pre-fix agent_h §3 measured 2 → 3 → 6 → 7 → 8 monotonic growth.
**But the post-fix asymptote is 3, not 1** as claimed by agent_k's report.

Root cause inspection of commits `d1fcd20`/`94542c2`:
- `wfn_loader.box_index_dev` correctly produces 1 shared device buffer per
  `(k_set, mesh)`. Confirmed.
- `wfn_transforms._cached_gindex_dev` content-hashes the numpy `g_arr`. Three
  separate numpy arrays of distinct content evidently reach it: (i) the
  WfnLoader g_index buffer captured by closure (via `box_index_dev` path,
  shared with psi_G_store), (ii) the sphere index passed to `gflat_to_rmu`
  closures (different bytes from g_index), (iii) at least one more variant
  observed at zeta_fit_start (count=2 before pre_rchunk_loop adds the third).
  These three buffers all match the same `(nq, nx, ny, nz) i32` shape but
  differ in **bytes**, so the content-hash dedup keeps them separate.

The cache fix is doing its job — it bounds the count at 3 (eliminating the
8× growth), but the FLOOR is 3 not 1. The Round-4 commit message claim "1-2
buffers" was optimistic.

## 3. W3 cap warning — verbatim capture

```
[plan_gflat_chunks] WARNING: r_chunk overridden to 24576 (cap was 20688);
Peak C at overridden r ≈ 66.41 GB/dev (budget 70.00 GB/dev).
[plan_gflat_chunks] WARNING: gflat_chunk_size overridden to 200 (cap was 100);
past the cuFFT plan-algorithm crossover at cs ~ 1000 cuFFT scratch grows
non-linearly (agent_f cs=1414 OOM verified).  Peak D at overridden cs ≈
12.75 GB/dev (budget 70.00 GB/dev).
```

Both warnings fire on every call (planner is invoked once per channel = 4×
per run). Peak D recomputed correctly: at cs=200, accumulate_fft_box ≈ 7.2 GB/dev
(2× the cs=100 value of 3.6 GB/dev), keeping D ≪ C so HWM = C remains.

## 4. Acceptance criteria check

- **W4** live_total within ±5% of R3 Agent J V1: target 76.46 GB; observed 76.79.
  Δ = +0.43%. **PASS** (the +0.43% is sphere going 8→3 in transverse channels;
  R3 V4 was also transverse, with sphere=8 → 77.61. R5 W4 has sphere=3 →
  77.61 - 0.81 = 76.80. Matches predicted-from-Round-3-modulo-sphere-fix.)
- **W2** live_total within ±10% of HWM_pred × 16: target 1062 GB global. Observed
  80.53 GB global. **N/A** — HWM_pred is the in-jit-transient peak which XLA
  frees before fit_one_rchunk returns (agent_j footnote); the live_arrays probe
  can't see it. live_total tracks the persistent + zeta_chunk peak, not the
  pair-density slot peak. So this check as specified is inapplicable; the
  meaningful check is "does fit_one_rchunk complete without OOM at HWM_pred",
  which it does (Δ vs 70 GB/dev budget = 95% utilization, fit ran fine).
- **W3** cap warning prints: **PASS**. Peak D recomputed at cs=200 (12.75 GB/dev)
  correctly.
- **W1** planner-natural runs through 3 r-chunks without OOM: **PASS**. The
  *agent_k claim* that the picker would select r=78272 is FALSE — picker
  selects r=20688 (n_rtot/55), close to the pre-Round-4 r=20256.
- **Sphere-idx count = 1** at end-of-run: **FAIL** — observed 3. (Round 6 fix
  candidate, see §5).

## 5. Diagnosis of the two prediction errors in agent_k's report

### 5a. r_chunk=78272 claim (FALSE; observed 20688)

`headroom_C` formula (`gflat_memory_model.py:713`):
```
α_C = 3 * c128(nk, ns, ns, mu, shard=p_xy)  +  c128(nq, mu, shard=p)
c_C_const = 4 * c128(nk, ns, mu, nb_total, shard=p_xy) + c128(nq, mu, mu, shard=p_xy)
            + sphere_idx_replicated_bytes(...)
headroom_C = target - c_C_const = 0.94 * 70e9 - c_C_const
r_chunk = headroom_C / α_C
```

Pre-Round-4 with `n_sphere_buffers=8`, `c_C_const` was higher by 7 × 0.162 GB.
Post-Round-4 with `n_sphere_buffers=1`, `c_C_const` drops by `7 × 1.62e8 = 1.13 GB`.
The increment in `r_chunk` is `1.13 GB / α_C`. With `α_C ≈ 3 ×
36×2²×1520×16/16 (sharded) = 2.74e7 bytes/r-unit`, the r_chunk increment is
about `1.13e9 / 2.74e7 ≈ 41` units. So r_chunk goes from ~20256 → ~20297, not
20256 → 78272. Agent_k's report is off by ~3 orders of magnitude on this claim
— almost certainly a transcription error or mismatched test inputs.

### 5b. HWM=17.79 GB/dev for V3 claim (FALSE; observed 66.41 GB/dev)

Agent_k's report row says "V3 ... new HWM_pred 17.79 GB/dev". The actual
predicted value at r=24576/b=32/cs=100 is **66.41 GB/dev**. The 17.79 figure
doesn't correspond to any peak component I can identify; possibly the
zeta_chunk transient alone (`c128(36, 1504, 24576, shard=p_xy) ≈ 21.3 GB/dev
unsharded → 1.33 GB/dev at p_xy=16`, off by ~13×) but not a match. Likely
transcription error in agent_k.

### 5c. Sphere-buffer count = 1 claim (FALSE; observed 3)

`_cached_gindex_dev` keys on the content-hash of the numpy g_arr. Three
*distinct* numpy buffers reach this cache during steady-state (likely:
full-BZ g_index, charge-channel sphere idx variant, transverse sphere idx
variant), so content-hash dedup correctly preserves them. The Round-4 expectation
of "1 buffer" was over-optimistic — the loader-side `box_index_dev` collapses
ONE source of leak (psi_G_store), the content-hash cache collapses growth to a
small constant (3) but cannot collapse three *content-distinct* numpy
sources into one. Fix would be to share a single canonical sphere idx across
all call sites (e.g., a `WfnLoader.sphere_idx_dev` mirror of `box_index_dev`),
not to fix the content-hash cache.

## 6. Final verdict

**ROUND 6 NEEDED:** the cache fix lands and works correctly (3 buffers bounded,
no monotonic growth), but the planner's `N_SPHERE_IDX_BUFFERS_BISPINOR=1`
is over-aggressive — the post-fix asymptote is **3 buffers**, not 1.

Recommended Round 6 work, in priority order:
1. **Set `N_SPHERE_IDX_BUFFERS_BISPINOR = 3` and `N_SPHERE_IDX_BUFFERS_CHARGE = 3`**
   (the safe undercount-by-zero choice, since the fix bounds the count at 3 and
   no growth was observed). Re-run W1/W2/W4 sanity check; expected HWM/dev rises
   by 0.324, no chunk-plan change since C ≫ sphere.
2. **Correct the agent_k report**: r_chunk=78272 / HWM=17.79 / sphere=1 are all
   incorrect. The branch is otherwise sound — the Round-4 code commits do what
   they claim, just less of it than claimed.
3. **(Optional)** add a `WfnLoader.sphere_idx_dev` accessor symmetric to
   `box_index_dev`, route all wfn_transforms call sites through it, and reduce
   sphere count to truly 1. Saves 0.324 GB/dev. Not blocking.

**No rollback needed.** The Round 4 commits (`d1fcd20`, `94542c2`, `48ee189`,
`81817e2`) are all safe, useful, and produce real improvements. Only the
planner constant `N_SPHERE_IDX_BUFFERS_*` overshoots reality.

## 7. Artifacts

| file | purpose |
|---|---|
| `agent_l_w1.out`, `agent_l_w2.out`, `agent_l_w3.out`, `agent_l_w4.out` | per-config gw.out logs |
| `w1_sb/`, `w2_sb/`, `w3_sb/`, `w4_sb/cohsex.in` | per-config sandbox dirs |
| `_launch_agent_l.sh` | Round-5 launcher (module init → lxrun) |
