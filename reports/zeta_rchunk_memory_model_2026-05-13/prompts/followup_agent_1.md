# Agent 1 — independent HLO interpretation verification

Context you already have: CONTEXT.md, your own agent_1.md (v1) and agent_1_v2.md (v2), the consensus.md, and the orchestrator's general-case-framing broadcast.

**New artifact to read:**
- `reports/zeta_rchunk_memory_model_2026-05-13/hlo_findings.md` — my (orchestrator's) interpretation of the CrI3 HLO dump.
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/xla_dump/module_0408.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt` — the actual HLO report. Three identical 200.35 GiB modules at indices 0297, 0299, 0408 — read one, they're the same kernel.
- `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/gw.out` — the run log including the planner's `gflat_plan.format()` output and the OOM trace.

**Your task: independent verification of three claims I asserted in hlo_findings.md.**

The fix commit (`agent/zeta-r-chunk-fixes-2026-05-13`, ff5873c on lorrax_A) leaned on these claims. If any are wrong, the fix is wrong. Be skeptical.

## Claim 1: `pair_density_slots = 3` is exactly right

I said the dump shows three 14.79-GiB slots that hold rank-5 `c128[ns, ns, μ·r_local, nk]`-shape values, matching the existing default. Verify by:
- Counting in the HLO report exactly how many distinct slots (rows in the "Allocations sorted by size with their values" section) host a pair-density-shaped value. The shapes to look for are `c128[nk, ns², n_rmu_local, r_chunk_local]` and its reshape variants, including the `c128[2, 6892832, 2, 36]` flat form.
- If you find 2, 3, 4, or 5 — say so explicitly with line citations.

## Claim 2: `band_fft_slots ≈ 3` (the S_fft constant in the formula)

I asserted the band-FFT pool has 58 slots = `N_BC · S_fft` with `N_BC=20` (from `nb_total=310 / band_chunk=16`) and `S_fft=3` (three shape variants per bc-iter: `c128[6,16,2,75,75,200]` band-FFT box, `c128[6,16,2,59990]` G-sphere variant, `c128[6,32,1125000]` n_rtot-flat variant).

Verify by:
- Counting the 3.22-GiB slots in the HLO report. Is it 58, 60, or something else?
- Mapping shape variants to slots: do you see exactly 3 distinct shapes recur across the slots, or more? (My interpretation of "3 shape variants per bc-iter" might be wrong — there could be 4 or 5 with one being rare.)
- Computing `slot_count / N_BC`. If it's 3.0 ± 0.1, my claim holds. If it's significantly different (e.g., 2.9 = 58/20, 4.0, 5.0), update the constant.

## Claim 3: `psi_Y_full` aliases cleanly with the band-FFT pool

I claimed `psi_Y_full` (the concatenation across bc-iterations, shape `(nk, nb_total, ns, r_chunk_local)` post-concat, sharded on `p_y`) does NOT have its own dedicated slot — it shares the band-FFT slot lifetimes. Verify by:
- Searching the HLO for any rank-4 buffer with `nb_total ≈ 310` (or the padded equivalent) in one of its dims AND a r_chunk-related dim like 73328 or 18332. Is there a dedicated slot?
- If yes, what's its size? Is it co-located with the band-FFT slots or in its own slot?

## Output

Write your findings to `reports/zeta_rchunk_memory_model_2026-05-13/agent_1_hlo_verify.md`. Structure:

1. **Verification of Claim 1** — final answer with citations.
2. **Verification of Claim 2** — final S_fft value, with method of derivation.
3. **Verification of Claim 3** — psi_Y_full slot count + bytes.
4. **Issues found in hlo_findings.md or consensus.md** — anywhere my interpretation was sloppy or wrong.
5. **Confidence assessment** — given what you found, do you support the commit's formula `nb_total · S_fft · psig_k_chunk_eff · ns · n_rtot · 16` for the band-FFT pool? Any caveats?

**Constraints:**
- Read-only on `sources/lorrax_A/`.
- No compute (no srun, lxrun, etc.). HLO files are static — read with cat / grep / less.
- Do NOT read agent_2_*.md (they're working a different follow-up task) until you've finished your own.
- Stay in this tmux pane.
- Print "Agent 1 HLO verification done" when finished.

Be terse, cite line numbers in the HLO report, and be willing to say "the orchestrator is wrong" if that's what you find. The fix on `agent/zeta-r-chunk-fixes-2026-05-13` (commit ff5873c) is reversible — better to catch an error now.
