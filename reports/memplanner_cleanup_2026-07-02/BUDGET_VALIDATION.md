# Memory-planner budget-sweep validation — `plan_gflat_chunks`

**Branch:** `agent/memplanner-cleanup` (`sources/lorrax_D`, `src/gw/gflat_memory_model.py`)
**Date:** 2026-07-02
**System:** MoS2 monolayer 3×3, nspinor=2, 82-band WFN, 642 centroids (real WFN, planner actually chunks)
**Mode:** static COHSEX (`do_screened=true`, no PPM), 1 GPU, 1×1 mesh (`LORRAX_NGPU=1` → planner sees `budget_gb = per_device_gb`)
**Allocation:** SLURM `55407468` (m2651, 4×A100-40GB, interactive)
**Run dir:** `reports/memplanner_cleanup_2026-07-02/budget_sweep/`

---

## TL;DR verdict

**`plan_gflat_chunks` PASSES — the cleanup is safe to push.** The ζ-fit planner it governs is
faithful: at the production budget (28 GB) it predicts a 22.40 GB high-water mark and the real
device peak is **22.26 GB (0.6 % under-prediction)**, hitting exactly the `C_fit_one_rchunk`
bottleneck it named. As the budget drops it shrinks `r_chunk` monotonically (3→5→8→13 chunks),
predicted HWM tracks **80 % of budget at every point** (never >100 %, never a wasteful 10 %),
and **every run completed EXIT 0 with a valid 730-line eqp0.dat — no OOM anywhere.**

**One honest caveat (not a planner defect, does not block the push):** `plan_gflat_chunks` models
only the ζ-fit + V_q peaks (its "5 HBM peaks", by design). There is a separate **~18.08 GB
upstream (pre-ζ-fit) pipeline transient** — set before `Computing C_q`, during WFN load / ψ→r
transforms — that this budget knob does **not** chunk. So the *whole-run* peak floors at ~18 GB:
lowering the budget below ~18 GB shrinks the ζ-fit stage exactly as modeled but does **not** make
the whole run fit that budget. On the production 40 GB A100 at the default 28 GB budget this is a
non-issue (22.26 < 28 < 40); it only bites at artificially low budgets you would never set on that
card. No OOM occurred because the physical card (40 GB) exceeds every budget tested.

---

## Results table

`peak_in_use` = JAX `memory_stats()['peak_bytes_in_use']` (the true XLA *logical* concurrent-live
peak — the same quantity the planner models), captured under the **BFC allocator** with
`LORRAX_MEM_DEBUG=1` so the retained arena makes `memory_stats()` return real numbers.

| Budget (GB) | band_chunk | n_r_chunks | gflat_chunk | bottleneck | Predicted HWM (GB) | Actual peak_in_use (GB) | Pred util | EXIT |
|-------------|-----------|-----------|-------------|-----------|--------------------|--------------------------|-----------|------|
| 28 | 16 | 3  | 100 | C_fit_one_rchunk | 22.40 | **22.26** | 80 % | OK (730-line eqp) |
| 16 | 16 | 5  | 100 | C_fit_one_rchunk | 12.80 | 18.08 † | 80 % | OK (730-line eqp) |
| 10 | 16 | 8  | 100 | C_fit_one_rchunk | 8.00  | 18.08 † | 80 % | OK (730-line eqp) |
| 6  | 16 | 13 | 100 | C_fit_one_rchunk | 4.80  | 18.08 † | 80 % | OK (730-line eqp) |

† The ζ-fit stage itself shrinks as modeled (at budget 6 its live `in_use` during
`fit_one_rchunk` is only 0.26–0.44 GB). The 18.08 GB figure is the **all-time** `peak_bytes_in_use`,
which is pinned by the upstream pre-ζ-fit transient (already 18.08 GB at `zeta_fit_start`, before
the r-chunk loop) — not the ζ-fit stage. Only at budget 28 is the ζ-fit stage the binding whole-run
peak, and there predicted (22.40) ≈ actual (22.26).

### Against the four acceptance criteria

- **(a) predicted ≤ budget always** — YES. 22.40≤28, 12.80≤16, 8.00≤10, 4.80≤6. Always exactly 80 %.
- **(b) actual ≤ budget / no OOM** — YES for the stage the planner governs and for the only
  production-relevant case (budget 28: 22.26 ≤ 28). No run OOMed; all EXIT 0, valid eqp. Whole-run
  peak *exceeds* budget at 16/10/6 (18.08 GB floor), but from the unchunked upstream stage, not the
  ζ-planner, and never as an actual OOM (40 GB card).
- **(c) monotonic shrink** — YES. Chunks: n_r_chunks 3→5→8→13; predicted HWM 22.40→12.80→8.00→4.80.
  (Whole-run actual floors at 18 GB due to the upstream stage.)
- **(d) utilization decent** — YES. 80 % of budget predicted at every point (not 10 %, not >100 %).
  Faithfulness at the binding budget: actual/predicted = 22.26/22.40 = **99.4 %**.

`band_chunk` stays 16 and `gflat_chunk_size` stays 100 (its cap) across the sweep — expected: for
this 80-band system the band-FFT-box (Peak A) and the cuFFT-plan cap (Peak D) are not the binding
axis; `r_chunk` (Peak C) is, and that is the axis that moves.

---

## Evidence (quoted from `run.log`)

### Budget 28 — planner block + faithful device peak
```
    Memory estimate: peak 27.16 GB (budget 28.00 GB), bottleneck=zct
  G-flat memory model — chunk plan + HWM estimate
    band_chunk         = 16
    r_chunk            = 18276  (3 chunks)
    gflat_chunk_size   = 100
    budget             = 28.00 GB/dev
    HWM estimate       = 22.40 GB/dev (80% of budget) [bottleneck: C_fit_one_rchunk]
...
[mem_probe after_fit_one_rchunk chunk=0] in_use=1.20 GB  peak=22.26 GB  live_count=60 ...
[mem_probe post_v_q]                     in_use=0.09 GB  peak=22.26 GB  live_count=84 ...
```
→ predicted 22.40 GB, actual peak_bytes_in_use 22.26 GB, peak reached at the named
`C_fit_one_rchunk` bottleneck (chunk 0 of the fit loop). **99.4 % faithful.**

### Budget 16 / 10 / 6 — chunks shrink, predicted tracks 80 %
```
[16]  r_chunk = 10288 (5 chunks)   HWM estimate = 12.80 GB/dev (80% of budget)
[10]  r_chunk =  6294 (8 chunks)   HWM estimate =  8.00 GB/dev (80% of budget)
[ 6]  r_chunk =  3632 (13 chunks)  HWM estimate =  4.80 GB/dev (80% of budget)
```

### Budget 6 — ζ-fit stage shrank; whole-run peak pinned upstream
```
[mem_probe zeta_fit_start]              in_use=0.03 GB  peak=18.08 GB   (BEFORE the r-chunk loop)
  Computing C_q via shard_map pipeline (open-spin, charge γ̃^0=I)
[mem_probe after_fit_one_rchunk chunk=0] in_use=0.44 GB  peak=18.08 GB  (ζ-fit transient tiny)
```
→ the ζ-fit stage is cheap at budget 6 exactly as planned; the 18.08 GB all-time peak predates it.

---

## Two measurement traps found (documented so nobody misreads the runtime line)

1. **The in-code `GPU high-water mark: X GB / budget (NN%)` line is unreliable on this stack.**
   - Under the **production/default allocator** (`XLA_PYTHON_CLIENT_ALLOCATOR=platform`,
     `TF_GPU_ALLOCATOR=cuda_malloc_async`, set by the `lorrax_D` modulefile) it is a **single
     `nvidia-smi` sample taken *after* the chunk loop** (`isdf_fitting.py` `_track_peak`, ~L2472).
     The async allocator has already freed the transients, so it reads **0.75 GB / 3 %** at budget
     28 — a 30× *under*-report of the real 22.26 GB peak. This is the cudaMallocAsync artifact noted
     in prior memory; do **not** read it as "planner is over-conservative."
   - Worse, `_track_peak` queries `nvidia-smi --id=0` (hardcoded GPU 0), which under BFC read an
     **identical 32.31 GB for budgets 28, 16 AND 10** — a constant that cannot be this run's peak.
     On a shared node GPU 0 may not even be this rank's device (assigned via
     `CUDA_VISIBLE_DEVICES`/`select_gpu.sh`), so that line can report a foreign process's memory.
   - **The trustworthy number is `peak_bytes_in_use`** from `mem_probe`/`_mem_report`
     (`LORRAX_MEM_DEBUG=1` / `LORRAX_MEM_PROFILE=1`) run under the **BFC allocator**
     (`XLA_PYTHON_CLIENT_ALLOCATOR=default`), which is what this report uses. On 1 GPU there is no
     NCCL, so the BFC-vs-async concern in the modulefile comment does not apply.

2. **BFC total process HBM (`nvsmi_peak`) was a flat 30.09 GB across all budgets** — that is BFC
   arena + CUDA context + cuFFT plan cache, allocator overhead independent of the ζ-fit chunk sizes,
   not a planner metric. It is why the raw `nvidia-smi` reading is not comparable to the planner's
   logical model.

---

## Reproduction

```
salloc --nodes=1 --qos=interactive --time=01:00:00 --constraint=gpu --gpus=4 \
       --account=m2651 -J lx-alloc-jackm bash -c "sleep 3600" &
export SLURM_JOBID=<jid>
# per budget:  reports/memplanner_cleanup_2026-07-02/budget_sweep/run_one.sh <BUDGET> bfc
# (run_one.sh loads lorrax_D+lorrax_agent, swaps allocator platform→default for a faithful
#  peak_bytes_in_use, sets LORRAX_MEM_DEBUG=1, runs `LORRAX_NGPU=1 lxrun python3 -m gw.gw_jax`)
```
Artifacts: `budget_sweep/budget_{28,16,10,6}_bfc/` (BFC + debug, faithful peaks) and
`budget_sweep/budget_28/` (production-default allocator, showing the 0.75 GB artifact).
