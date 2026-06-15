# Memory model robustness — non-bispinor + k-grid scaling

Sister initiative to `reports/memory_model_refit_2026-05-17/` (which calibrated
the planner on bispinor CrI3 + Si). The bispinor work landed `pair_density_slots=3`,
`fft_box_factor_D=2.0`, and the persistent-array refit, with predicted-vs-realized
%-err in the −0.5 % to −10.8 % range on Si μ-sweep (see
`reports/memory_model_refit_2026-05-17/MEMORY_MODEL_SYNTHESIS.md`).

## Goal of this initiative

1. **Non-bispinor parity.** Verify the planner is equally accurate on
   non-bispinor Si (scalar `nspinor=1`) as on bispinor. The bispinor work
   only spot-checked ns=2 on CrI3 (agent_d M4); we want a clean Si
   `noncolin=false` μ-sweep against mem_stats peak.
2. **k-grid scaling.** Hold μ density roughly fixed and vary the k-grid
   2×2×2 → 3×3×3 → 4×4×4 → 6×6×6 with symmetries enabled. Check that each
   persistent buffer (centroids, L_q, gflat_acc, sphere_idx, V_acc) and each
   peak component (Peak A–E breakdowns) scales how the planner says it
   should. Any superlinear or unexpected scaling = potential leak or
   missing planner term.
3. **Band-count sensitivity.** Re-run select configs at nbnd=100 and
   nbnd=200 to confirm the linear-in-nb pieces (centroids, pair-density on
   `nb_l + nb_r`) scale correctly.

## Agent assignments

| Agent | LORRAX clone | Branch | Run dir prefix | Report file |
|---|---|---|---|---|
| **A** — non-bispinor μ-sweep + HLO | `sources/lorrax_A` | `agent/si-nonbispinor-mu-sweep` | `runs/Si/MU_nonbispinor_2026-05-18/` | `agent_a_si_nonbispinor_mu_sweep.md` |
| **B** — k-grid scaling | `sources/lorrax_B` | `agent/si-kgrid-scaling` | `runs/Si/KGRID_nonbispinor_2026-05-18/` | `agent_b_si_kgrid_scaling.md` |
| **C** — band-count sensitivity | `sources/lorrax_C` | `agent/si-band-sensitivity` | `runs/Si/BANDS_nonbispinor_2026-05-18/` | `agent_c_si_band_sensitivity.md` |

`sources/lorrax_D` is reserved for the env-var cleanup branch — do not touch.

## Mandatory protocol (all agents)

1. **Frequent rebase.** Before every commit, fetch `origin/main` and rebase
   your feature branch onto it. The main work landed yesterday (commits
   `0f355b7`, `e2de150`, `685f11b`, `9afa11e`, `da7b41f`, `81817e2`,
   `94542c2`, `48ee189`, `d1fcd20`, `409be4f`, `652b004`, `5c884ac`,
   `21f2ed6`, `381e010` — see `git log origin/main`); main may still be
   moving.
2. **Pool-aware allocations.** `module load lorrax_X lorrax_agent` then
   `lxattach` (if a tagged allocation exists) or `lxalloc N HH:MM:SS` in
   the background. Use `lxrun` (NOT bare `srun --jobid=`) to launch jobs.
   Never grab a foreign JID directly — share via the pool.
3. **Measurement protocol** (from
   `reports/memory_model_refit_2026-05-17/MEMORY_MODEL_SYNTHESIS.md` §5):
   - Standard sandbox env (platform allocator) is correct for the run
     itself.
   - For OOM-relevant HWM, override `LORRAX_SHIFTER` with
     `XLA_PYTHON_CLIENT_ALLOCATOR=default`,
     `XLA_PYTHON_CLIENT_PREALLOCATE=true`,
     `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` so
     `device.memory_stats()['peak_bytes_in_use']` is exposed.
   - `LORRAX_MEM_DEBUG=1 LORRAX_RCHUNK_DEBUG=1 LORRAX_MAX_RCHUNKS=3
     LORRAX_EXIT_AFTER_ZETA=1` to bound runtime to ~60–90 s on Si.
   - For HLO dumps, use the profiling skill's launcher (`scripts/profiling/run_profiled.py`).
4. **Trust production log over mock.** Grep `gw.out` for
   `"G-flat memory model — chunk plan + HWM estimate"` to recover the
   planner's per-component breakdown. Do NOT hand-roll a planner replay.
5. **Reports.** Each agent writes its report in this dir (file name in
   table above). Mirror the structure of
   `reports/memory_model_refit_2026-05-17/agent_t_si_bispinor_sweep.md`
   (Headline verdict → table → per-channel timings → planner breakdown →
   interpretation).
6. **Checkpoint discipline.** Every ~5 commits or major milestone, follow
   `skills/checkpoint/SKILL.md`: pytest, commit, report update, CHANGELOG.

## File-naming conventions for run dirs

```
runs/Si/<TAG>_nonbispinor_2026-05-18/
    manifest.yaml
    qe/   (or _NxNxN/qe/ if k-grid sweep)
        scf/
        nscf/
    mu<MU>/
        cohsex.in
        centroids_frac_<N>.txt
        gw_platform_false.out      # production allocator
        gw_bfc_pre95.out           # mem_stats peak
        xla_dump/                  # if HLO needed
```

## Cross-agent state to avoid colliding on

- WFN.h5 inputs: each agent's run dirs are isolated. Don't symlink into
  another agent's qe/ subdir.
- Centroids: per-MU files. Regenerate locally.
- Probe env vars: set them per-launcher, not in shell rc.
- Allocations: use the lorrax_agent pool — `lxstatus` to inspect.

## Concrete questions this initiative should answer

1. Does `pair_density_slots = 3` hold at `ns=1` (true scalar Si)? Or do
   ns² hidden factors break the count?
2. Does `fft_box_factor_D = 2.0` hold across Si k-grids and at ns=1?
3. How does Peak C's `P_pair_concurrent_slots` (the binding peak in every
   measured config so far) scale with kgrid at fixed μ density: linear in
   nk, super-linear, sub-linear?
4. Does `sphere_idx_replicated` (the only fully replicated array in the
   persistent set) stay at 1 buffer post-canonical-accessor across all
   k-grids? Or does the leak come back at small or large kgrid?
5. Does `gflat_acc` (which is `c128(nq_disk, mu, ngkmax)`) scale correctly
   with nq_disk as kgrid grows?
6. Is the planner's %-err vs mem_stats systematically biased
   under-prediction (as observed on bispinor) across all kgrid + band-count
   variations?
7. Are there band-count leaks (cross-bc residue) that grow with nb?

## Out of scope

- Compute V_q + Σ_X correctness vs BGW — this is a memory-model audit, not
  a physics audit. Agents should run through ζ-fit + first V_q tile only
  (LORRAX_MAX_RCHUNKS=3 LORRAX_EXIT_AFTER_ZETA=1) unless the planner
  prediction for Peak E specifically needs validation.
- Bispinor — separate effort, already covered in `memory_model_refit_2026-05-17`.
- The `gflat_chunk_size = 100` cap — that's an empirical hard cap, not
  scaling-dependent.
