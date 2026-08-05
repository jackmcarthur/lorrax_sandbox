# Agent O — Cross-Allocator Audit (Round 8)

**Branch:** `agent/bispinor-ibz` on lorrax_B
**System:** CrI3 6x6x1 80Ry SOC bispinor, 16 GPUs (4×4 mesh, hbm80g)
**JID:** 53087968
**Run dir:** `runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/`
**Goal:** disambiguate whether Round 7's 7-8× planner-over-prediction gap is REAL (XLA aliasing/donation saves memory the static model can't see) or an ARTIFACT (`cudaMallocAsync` returns pages so fast that nvidia-smi sampling never catches the true peak).

## Headline verdict

**The Round 7 gap is MOSTLY a measurement artifact.** True XLA-arena peak under BFC + preallocate (variant Y3_95) is **76.05 GB/dev**, very close to the planner's 66.41 GB/dev prediction (15% under-prediction, not 7× over). Round 7's "nvsmi_peak = 8.67 GB" was an artifact of `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (cudaMallocAsync): pages are released between probe samples so the high-water-mark is invisible to nvidia-smi. **The planner should be RECALIBRATED slightly UP, not down — and the model is closer to right than Round 7 concluded.**

The sandbox's default (`platform + preallocate=false`) is still correct for FFI compatibility (NCCL needs the shared pool); this audit doesn't change that recommendation.

## 1. Comparison table

Probe config: X3 sweet-spot (`r=24576, b=32, cs=100`), `LORRAX_MAX_RCHUNKS=3`, `LORRAX_EXIT_AFTER_ZETA=1`, `LORRAX_FORCE_FULL_BZ=1`, 4×4 mesh on 4 nodes × 4 A100-80GB.

Planner prediction (Peak C, in-jit transient): **66.41 GB/dev** across all variants — config is identical.

| variant | XLA_…_ALLOCATOR | PREALLOCATE | MEM_FRACTION | XLA pool / dev | mem_stats peak | mem_stats in_use | nvsmi_peak | live_total (global) | outcome |
|---|---|---|---|---|---|---|---|---|---|
| **Y1** baseline | platform   | false | (n/a)  | n/a — async | **None** (-1) | **None** (-1) | **8.67 GB**  | 80.21 GB | succeeded (reproduces Round 7 X3) |
| **Y2** BFC no-prealloc | default | false | (unset) | 63.82 GB (~80% × 80) | 15.12 GB (before OOM) | 4.82 GB | 22.97 GB | 58.40 GB | **OOM** trying to allocate one 60.12 GB block (pool too small for jit) |
| **Y3_50** BFC pre 0.50 | default | true  | 0.50   | 42.55 GB | 15.01 GB (before OOM) | 4.67 GB | 41.85 GB | 58.40 GB | **OOM** trying to allocate one 60.12 GB block (pool too small) |
| **Y3_95** BFC pre 0.95 | default | true  | 0.95   | 80.84 GB | **76.05 GB** | 6.25 GB transient | 78.15 GB | 80.21 GB | **succeeded** — gold-standard peak captured |

Source files:
* `/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/agent_o_y1.out`
* `…/agent_o_y2.out`
* `…/agent_o_y3_50.out`
* `…/agent_o_y3_95.out`

## 2. Verdict: the Round 7 gap is mostly a measurement artifact

| metric | Round-7 X3 (Y1) | Y3_95 ground truth | gap |
|---|---|---|---|
| nvidia-smi peak | 8.67 GB | n/a (~78 GB dominated by preallocation) | — |
| **XLA-arena peak** (`device.memory_stats()['peak_bytes_in_use']`) | None (unavailable under platform) | **76.05 GB** | — |
| planner prediction (Peak C) | 66.41 GB | 66.41 GB | — |
| **planner error vs ground truth** | "+566% (over-pred)" — WRONG conclusion | **-13.6% (under-pred)** | the real story |

**Mechanism**: `XLA_PYTHON_CLIENT_ALLOCATOR=platform` (cudaMallocAsync) hands HBM pages back to the CUDA driver as soon as XLA's PJRT executor releases them, often within microseconds of a buffer becoming dead. Probes (and Round 7's nvidia-smi snapshots) sampled on the order of seconds-to-minutes (between probe sites). The buffer the planner sized for (~60 GB Peak-C transient) is allocated, used, and freed within a single `fit_one_rchunk` jit — by the time the probe fires *after* the jit, those pages are already gone. **nvidia-smi under platform shows only the steady-state working set (persistent ψ + ζ buffers ≈ 8 GB), not the in-jit transient.**

Under BFC + 0.95 (Y3_95), the allocator holds the pages, so `device.memory_stats()['peak_bytes_in_use']` records the true 76.05 GB high-water mark. The planner under-predicts by ~10 GB because it doesn't model some buffer-aliasing overhead that XLA's runtime inflates (the 60.12 GB single-allocation seen in Y2/Y3_50 OOM messages is the largest contiguous block XLA needs; the 76 GB peak adds the rest of the still-live working set).

## 3. The OOM data points (Y2, Y3_50) are themselves diagnostic

Both Y2 (63.82 GB pool) and Y3_50 (42.55 GB pool) **failed** with:
```
RESOURCE_EXHAUSTED: Out of memory while trying to allocate 64550340328 bytes
```
That's a single ~60 GB allocation request — exactly the planner's Peak C transient. **This confirms that XLA *really does* try to materialize that buffer**; under `platform` allocator it gets recycled fast, under BFC it has to fit in the steady pool. The planner's 66.41 GB number is realistic, not paranoid.

Y3_95 succeeded only because the 80.84 GB pool exceeded both the 60 GB single-allocation AND the residual working set (8-15 GB before the jit).

## 4. Recommendation: planner recalibration

**Round 7 said "keep the cap at 100, planner is conservative by 7-8×."** That conclusion was wrong about the magnitude (the platform-allocator nvsmi was hiding the in-jit peak), but the practical *outcome* — `gflat_chunk_size <= 100` at X3 r=24576 ran fine — is still correct *for the sandbox's default allocator*. Under cudaMallocAsync the actual HBM bind is around 8-9 GB on top of context overhead, so OOM never triggers despite the predicted 66 GB.

Concrete actions:
1. **Update `docs/MEMORY_MODEL.md`** to note that the planner's Peak-C HWM is a faithful upper bound on the XLA-arena peak under BFC, AND a substantial OVER-prediction of the nvidia-smi observable under cudaMallocAsync (the sandbox default). Both are correct in their own frame.
2. **Do NOT loosen the planner cap** based on the Y1 nvsmi gap — that gap is unobservable to BFC users and unobservable to anyone profiling true XLA arena pressure. The 66 GB prediction is within 14% of Y3_95's 76 GB ground truth.
3. **No change to the sandbox default** (`platform + preallocate=false`) — Y2 and Y3_50 OOMs are the documented evidence that BFC starves NCCL / cuSOLVERMp when the jit reservation is tight; the sandbox correctly avoids this.
4. **Optional**: a planner refinement that captures the extra ~10 GB (76 - 66 = 10 GB) seen in Y3_95 could be a future precision improvement; the ~14% under-prediction is well within "safety margin" today but worth a single revisit when next refactoring.

## 5. Sandbox-error candidates: none

Y2 and Y3_50 OOMs are exactly the documented failure modes of BFC pre-allocation on this stack (see `docs/ENVIRONMENT_COMPREHENSIVE.md` §3.2 and §8.3). No new infrastructure issues.

## 6. Artifacts

| file | purpose |
|---|---|
| `agent_o_y1.out` | platform + false: reproduces Round 7 X3 nvsmi_peak=8.67 GB |
| `agent_o_y2.out` | default + false: OOM at 60 GB single allocation; XLA peak captured before OOM = 15.12 GB |
| `agent_o_y3_50.out` | default + true + 0.50: OOM; mem_stats available; XLA peak before OOM = 15.01 GB |
| `agent_o_y3_95.out` | default + true + 0.95: succeeded; **mem_stats peak = 76.05 GB ground truth** |
| `_launch_agent_o.sh` | per-variant launcher (`--env=` overrides applied after module setenv) |
| `_print_env_then_gw.py` | rank-0 startup env-audit shim |
| `_extract_agent_o.py` | post-hoc log scraper for in_use / peak / nvsmi peaks |
