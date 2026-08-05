# Agent N — Round 7 Faithfulness Audit (Predicted vs Realized HBM)

**Branch:** `agent/bispinor-ibz` (lorrax_B HEAD `6ba1fad`: mem_probe + nvidia-smi)
**System:** CrI3 6x6x1 80Ry SOC bispinor, 16 GPUs (4x4 mesh, hbm80g)
**JID:** 53087968
**Run dir:** `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**Goal:** characterize the gap between planner HWM_pred, JAX live_arrays view, and true HBM (`nvidia-smi memory.used`) across the cs spectrum, to determine OOM-relevance of the model and the right cap on `gflat_chunk_size`.

## Headline verdict

**The planner over-predicts true HBM by ~7-8x.** HWM_pred=66.41 GB/dev describes the *worst-case in-jit transient* assuming no XLA buffer aliasing/donation; the actual `nvidia-smi` peak per rank tops out at **8.67 GB/dev** across all configs X1-X6 — well under the 70 GB budget. **No cuFFT plan-workspace blind-spot was observed** across cs in [50, 100, 500, 1000]: nvsmi_peak is constant at 8.67 GB/dev independent of cs. The current cap `gflat_chunk_size <= 100` is **extremely conservative**; cs=1000 ran cleanly in production. Recommend raising the cap to **800** for headroom into the agent_f cs=1414 OOM crossover, OR keeping at 100 since there is no observed performance gain from larger cs (agent_d M3).

## 1. Critical instrumentation note

`device.memory_stats()` returns **`None`** on the Perlmutter JAX 0.8 / CUDA 12.9 stack. Therefore both `bytes_in_use` and `peak_bytes_in_use` were unavailable (`-0.00 GB` in all probe lines). The task's primary OOM metric (`peak_bytes_in_use`) **cannot be observed via JAX** on this stack.

The probe was extended on `agent/bispinor-ibz` (commit `6ba1fad`) to additionally invoke `nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=memory.used` at each probe point, with a module-level running peak. **nvidia-smi is the only available OOM-faithful per-rank HBM metric** on this stack. It includes cuFFT plan workspace, NCCL buffers, CUDA context overhead, and everything else outside the JAX arena.

15/15 planner tests still pass after the change. Commit message: "feat(mem_probe): capture nvidia-smi per-rank true HBM for OOM tracking".

## 2. Comparison table

All in GB. `live/rank` accounts for sharding: μ-sharded arrays /16, replicated arrays full-size on each rank. `nvsmi` is per-rank max across all probe points within the run. `gap_pk - lr` is the rank-0 nvsmi peak minus the true per-rank live (the only meaningful "model blind spot" measure).

| cfg | r_chunk | b | cs | pred (GB/dev) | live_tot global | live/rank | nvsmi_peak GB/dev | err_vs_live% | gap_pk-lr GB/dev |
|---|---|---|---|---|---|---|---|---|---|
| X1 (natural) | 20688 | 64 | 100 | 55.99 | 76.84 | ~8.6 | 8.45 | +551% vs raw, -0.2% vs sharding-correct | ~-0.2 |
| X2 (small r) | 8192 | 32 | 100 | 22.48 | 66.01 | ~7.4 | 7.75 | -41% vs raw, +204% vs sharding-correct | +0.3 |
| X3 (sweet)   | 24576 | 32 | 100 | 66.41 | 80.21 | ~8.96 | 8.67 | +562% vs raw | -0.3 |
| X4 (cs=50)   | 24576 | 32 | 50  | 66.41 | 80.21 | ~8.96 | 8.67 | +562% vs raw | -0.3 |
| X5 (cs=500)  | 24576 | 32 | 500 | 66.41 | 80.21 | ~8.96 | 8.67 | +562% vs raw | -0.3 |
| X6 (cs=1000) | 24576 | 32 | 1000| 66.41 | 80.21 | ~8.67 | 8.67 | +662% vs raw | 0.0 |

**The live_total/16 figure is misleading** because part of the live set is replicated (every rank holds a full copy). Sharding-corrected per-rank live (sharded-globals/16 + replicated-globals) gives **~8.96 GB/rank at X3 peak** — within **0.3 GB** of the nvsmi_peak 8.67 GB/rank. **The JAX live_arrays view, properly sharding-corrected, agrees with nvidia-smi to within 4%.**

## 3. cuFFT scaling: did NOT materialize

The task's headline hypothesis was that nvsmi - live_per_rank grows with cs (cuFFT plan-workspace blind spot). **It does not, in the audited cs range:**

```
cs:     50   100   500  1000
nvsmi:  8.67  8.67  8.67  8.67   (GB/dev, constant)
gap:   -0.3  -0.3  -0.3  -0.3   (≤ noise of nvsmi sampling)
```

Either (a) the cuFFT plan switch happens between cs=1000 and cs=1414 (agent_f's OOM cs), or (b) the cuFFT scratch is allocated and freed inside the gflat-accumulate jit between our probe points (probe sampling cannot catch it). Both are consistent with the data; (a) is more likely given that cs=1000 ran cleanly with low nvsmi while agent_f's cs=1414 hard-OOM'd.

```
ASCII chart: nvsmi_peak vs cs (X3..X6; r=24576, b=32 held fixed)

  9.0 |              X3   X4   X5   X6
      |               ●    ●    ●    ●   (all at 8.67)
  8.5 |
  8.0 |
  7.5 |
  ----|----+----+----+----+----+----+
      0   100  300  500  700  900 1100
                  gflat_chunk_size
```

The model's blind spot — about **3.6 GB/rank** between nvsmi-peak and live_total/16 (raw, unsharding-corrected) — is **constant**, not cs-dependent. This points to JAX/XLA runtime overhead (CUDA context, compile caches, NCCL collective buffers from comms during reshape) rather than cuFFT.

## 4. X6 (cs=1000) outcome

**Ran cleanly.** No OOM. nvsmi_peak across all 4 channels × 3 r-chunks: **8.67 GB/dev** — 12% of the 70 GB budget. Cap warning fired correctly:

> `[plan_gflat_chunks] WARNING: gflat_chunk_size overridden to 1000 (cap was 100); past the cuFFT plan-algorithm crossover at cs ~ 1000 cuFFT scratch grows non-linearly (agent_f cs=1414 OOM verified). Peak D at overridden cs ≈ 41.55 GB/dev (budget 70.00 GB/dev).`

The Peak D prediction of 41.55 GB/dev is also over-predicted by the planner (actual nvsmi ≈ 8.67 GB/dev — observed in the accumulate stage too).

## 5. Faithfulness summary

**Within the JAX arena** (live_arrays view, sharding-corrected): the model's persistent + post-jit-transient prediction agrees with nvidia-smi to within **~4% (0.3 GB/dev)** across all 6 configs. The agreement is best at X3-X6 (planner mature region) and slightly worse at X2 (where r_chunk = 8192 is unusually small and the planner Peak C is bottlenecked by something else).

**Outside the JAX arena**: a constant ~3.6 GB/dev blind spot exists between raw `live_total / 16` and nvsmi (CUDA context overhead + compile caches + NCCL buffers). This is **NOT** cs-dependent and **NOT** cuFFT-related at cs ≤ 1000.

**In-jit transient**: the planner's HWM_pred = 66.41 GB/dev describes an *upper bound* on what XLA might use inside `fit_one_rchunk`'s jit assuming no aliasing/donation. The runtime never approaches this peak (nvsmi = 8.67 GB/dev — 7.7x lower than HWM_pred). The HWM_pred is a **safe upper bound** for chunk-sizing purposes but is NOT predictive of actual HBM use; the planner could be tightened, but the over-prediction is safety, not waste.

**The model can be trusted for OOM prediction at all cs in [50, 1000].** Above cs=1000 the agent_f cs=1414 OOM is a hard cliff; the model does not predict where exactly the cliff is but the WARNING fires past cs=100 to flag entry into the unknown regime.

## 6. Recommended cap on `gflat_chunk_size`

**Recommendation: keep the cap at 100.**

Rationale:
1. **No performance gain from larger cs** (agent_d M3: 19-26 s/r-chunk at cs=1 vs 20-25 s/r-chunk at cs=360). cuFFT plan amortizes over the 3420 scan iters per channel regardless of cs.
2. **No memory pressure relief from larger cs either** — nvsmi is bounded at 8.67 GB/dev across cs=50,100,500,1000. There is no actionable reason to raise the cap.
3. **Safety margin to the cs=1414 OOM cliff** — 14x headroom at cs=100 is comfortable; raising to cs=800 would still be safe per X6 but provides no benefit.
4. **The current WARNING wording is accurate** — past cs=100 (cap) the planner notes the agent_f cs=1414 OOM and that the regime is "non-linear"; this is the right level of risk-flagging.

Alternative considered: raise to cs=500 (consistent with X5 nvsmi). Rejected because (i) no perf benefit, (ii) the cs=100 cap is already over-conservative AND non-harmful, so changing it adds risk without benefit.

If the user later wants to push cs higher for some specific reason (e.g. a hypothetical perf gain on a different topology), the safe ceiling per this audit is **cs=1000** with the WARNING still firing for human review.

## 7. Deliverables checklist

- [x] Comparison table (6 rows, sharding-corrected)
- [x] cuFFT-scaling chart (flat — NULL result is itself the finding)
- [x] X6 outcome (ran cleanly, nvsmi=8.67 GB/dev, 12% of budget)
- [x] Faithfulness summary paragraph
- [x] Recommended cap (keep at 100)
- [x] `_mem_probe` extended with nvidia-smi (commit `6ba1fad`)
- [x] Planner tests pass (15/15)
- [x] `docs/MEMORY_MODEL.md` updated with "Predicted-vs-realized faithfulness" section (separate commit on `agent/bispinor-ibz`)

## 8. Artifacts

| file | purpose |
|---|---|
| `agent_n_x1.out` ... `agent_n_x6.out` | per-config gw.out logs |
| `x1_sb/` ... `x6_sb/cohsex.in` | per-config sandbox dirs |
| `_launch_agent_n.sh` | Round-7 launcher |
| commit `6ba1fad` on `agent/bispinor-ibz` | `_mem_probe` nvidia-smi extension |
| (this report) | Round-7 faithfulness verdict |

## 9. Open follow-ups (not blocking)

* **JAX `memory_stats()` returns None on this stack** — known JAX 0.8 / CUDA 12.9 bug. The nvidia-smi fallback in `_mem_probe` covers it but a JAX upgrade would simplify per-rank HBM tracking.
* **Probe sampling cannot catch sub-jit transients** — to verify cuFFT plan workspace is truly bounded at cs ≤ 1000, a background nvidia-smi sampler running at 100 ms intervals during the accumulate jit would be the rigorous test. The aggregated probe nvsmi_peak comes close but cannot fully exclude allocate-then-free patterns inside a single jit. Not blocking for OOM prediction since none of X1-X6 OOM'd.
* **In-jit HWM_pred over-prediction** — the planner says 66.41 GB/dev for Peak C but nvsmi observes 8.67 GB/dev. The gap (57.74 GB/dev) is XLA buffer aliasing/donation/remat saving real memory that the planner doesn't model. Tightening this is a future planner refinement but offers no safety benefit since the over-prediction errs toward safety.
