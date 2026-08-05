# Round 6 — Agent 3: HLO validator

Read `round6_discussion.md` first. Read `round5_unified_plan.md` §4 (your contribution from Round 5).

## Your role

You don't write the kernel code. You **validate the HLO of Agent 2's new kernel against the predictions in `round5_unified_plan.md` §4** — gate G2 from the discussion file's validation gates.

## Workflow

### Phase 1: Prep (while Agent 2 implements)

Don't sit idle. Re-read your Round 4 `round4_improvements.md` so you have the baseline HLO numbers fresh:
- Current `lorrax_B` `5cadd4b` HLO at CrI3 6×6 80 Ry: 48.63 GiB total, 30 GiB from `psi_Y_full` materialized twice, 12 GiB unsharded FFT box (remat cost), ~20 "Involuntary full rematerialization" warnings.
- Predicted for Round 6: ~10–15 GiB total, 2 × 3.71 GiB carry slots, 1 FFT box aliased inside scan, 0 remat warnings.

Write your **HLO comparison checklist** in `round6_discussion.md` under "Agent 3 → others" so when the new HLO lands you can fill it in rapidly.

### Phase 2: Set up the HLO dump (when Agent 2 commits)

When Agent 2 posts "Agent 2 round 6 done":

1. Build a new run-variant dir: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_round6_hlo_2026-05-13/`. Symlink the standard WFN.h5/kin_ion.h5/centroids/qp_wfn_rotations from the parent `lorrax_A_perf_2026-05-07/`. cohsex.in: same recipe as the morning's lorrax_A baseline + the lorrax_B_path_d run (memory_per_device_gb=60, band_chunk=16, r_chunk=0, gflat_chunk_size=64). NO psig_k_chunk_size (retired). NO gflat_to_rchunk_chunk_size knob this round (the helper is gone).
2. Launch via the existing pattern (`module load lorrax_B` + `lxattach` + `lxrun` with `XLA_FLAGS=--xla_dump_to=$PWD/xla_dump`). Reference: prior `run_path_d_hlo.sh` script in the lorrax_B_path_d_hlo dir.
3. Wait for the jit__kernel module memory-usage-report to appear.

### Phase 3: G2 validation

For the largest `jit__kernel` memory-usage-report:
- **Total bytes** — predicted ≤ 15 GiB. Pass if so.
- **FFT-box-class slot count** — predicted ≤ 2 (one aliased inside scan + maybe one fusion-adjacent). Pass if ≤ 3.
- **Pair-density slot count** — predicted 2 × 3.71 GiB carry slots. Confirm.
- **`psi_Y_full` materialization** — predicted ABSENT (no concat). Confirm there's no `c128[nk, nb_total, ns, ...]` slot.
- **Remat warnings in stderr** — `grep "Involuntary full rematerialization" gw.out`. Predicted count: **0**. PASS REQUIRES ZERO.

Write the comparison table to `round6_discussion.md` "Agent 3 → others" + a copy to `reports/zeta_rchunk_memory_model_2026-05-13/round6_hlo_validation.md`. If predictions match, post "Agent 3 G2 passed". If a slot count is wrong, write a `[CONCERN]` note for Agent 2 to investigate.

## Output

Print "Agent 3 round 6 done" when the HLO validation table is complete and posted. ~30 min active time once Agent 2 commits.

Constraints: read-only on `sources/`. The HLO dump itself requires the SLURM allocation, but the validation reading is CPU.
