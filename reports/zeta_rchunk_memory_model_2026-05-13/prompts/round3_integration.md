# Round 3 — Agent 2: Integrate `gflat_to_rchunk` into `_make_fit_one_rchunk_kernel`

You just shipped `gflat_to_rchunk` (commit `d7eaf1c` on `sources/lorrax_B` branch `agent/zeta-bc-scan-shardmap`). The orchestrator added a follow-on commit `3d0636c` (qvec content-hash fix in `accumulate_rchunk_to_gflat`). All 27 tests pass.

**Now**: replace `_make_fit_one_rchunk_kernel._kernel`'s Python-unrolled bc-loop with a single `gflat_to_rchunk` call. This is the structural fix landing — the principle violation goes from "stopgap budgeted by the planner" to "doesn't exist."

## Comm protocol (file-polling, no orchestrator routing)

- **Shared discussion**: `reports/zeta_rchunk_memory_model_2026-05-13/round3_discussion.md`. Read it at the start of each work cycle (Read tool, top to bottom). Respond to anything new in the "Agent 4 → Agent 2" section before continuing your own work.
- **Your work log**: `reports/zeta_rchunk_memory_model_2026-05-13/integration_log.md` (create it). Append as you go.
- **Blockers needing the human**: append a line starting with `BLOCKER:` to the "Either → Orchestrator" section of `round3_discussion.md`, then stop.

## Scope

Per `parallel_helpers_design.md §4` and `parallel_helpers_impl.md §4`. Three sub-tasks:

1. **`PsiGStore.psi_G_device_full`** — new lazy property on the store. Returns the `(nk, nb_total, ns, ngkmax)` device tensor with band-axis sharding `P(None, ('x','y'), None, None)`. Pulls all band-chunks via the existing io_callback pattern once per `begin_rchunk` cycle; invalidated by `end_rchunk` or by `_clear_tiles`. Add a small `tests/test_psi_g_store.py` test that round-trips.

2. **Replace the bc-loop in `_make_fit_one_rchunk_kernel._kernel`** (`isdf_fitting.py:~1273-1286`). Delete the `for bc_range in band_chunk_ranges: psi_Y_parts.append(...)` + `jnp.concatenate(...)` block. Call `gflat_to_rchunk` with the closure-captured store's `psi_G_device_full` + `g_index` + a `chunk_size` chosen from `cfg.memory.memory_per_device_gb`. Per-side band slices (`psi_l_Y_sm`, `psi_r_Y_sm`) come from slicing the full output.

3. **Remove the `band_fft_pool` accommodation** from `gflat_memory_model.py` (and its `psig_k_chunk` plumbing). The structural fix means this term is now zero in practice; carrying it codifies a defect that no longer exists. **Bundle this with the `_kernel` rewrite in one commit** so the "before/after" is atomic.

### Out of scope this round (flag if you see them)

- The `c_q_from_psi_sm` callers at `isdf_fitting.py:1661, 1667` use a different inner shape (centroid indices, not flat-r slab) and need a separate `gflat_to_rmu` helper. Don't fix here — log it.
- Cherry-picking the `band_fft_pool` removal back to `lorrax_A`'s `agent/zeta-r-chunk-fixes-2026-05-13` is a separate followup; do not touch lorrax_A this round.

## Chunk-size policy

`gflat_to_rchunk(..., chunk_size=cs)`. Defaults:
- `cs = None` ⇒ one-shot (whole flat axis, single scan iter). Use when memory allows.
- For CrI3 80 Ry on 16-GPU 4×4 mesh: `N = nk · nb_local ≈ 36 · (310/16) ≈ 698 rows/rank`, per-rank FFT box at one-shot ≈ `698 · ns=2 · n_rtot=1.125M · 16 ≈ 25 GB`. **One-shot still doesn't fit on 28 GB budget.** Pick `cs` so per-iter FFT box ≤ ~50% of `memory_per_device_gb`.
- Heuristic (orchestrator suggests): `cs_target = floor(0.5 * budget_bytes / (ns * n_rtot * 16))`. At CrI3 80Ry / 28 GB budget: `cs ≈ 350` ⇒ 2 scan iters at ~12.5 GB FFT box each.
- Site-pick this in `_kernel` from `cfg.memory.memory_per_device_gb`, not in the helper itself.

## Validation gates (in order; do NOT proceed past a failed gate)

1. **CPU pytest** for `test_wfn_transforms.py`, `test_rchunk_gflat_pair.py`, `test_psi_g_store.py`, `test_aot_memory.py`. All must pass.
2. **MoS2 3×3 ζ-fit bit-identity** on lorrax_B branch vs the baseline ζ produced by lorrax_A `agent/zeta-r-chunk-fixes-2026-05-13` head. Use whatever existing test scaffold exists — if none, run gw.gw_jax end-to-end at MoS2 3×3 and diff the `zeta_q.h5` outputs to `atol=1e-12` per element. **This is the load-bearing test** — bit-identity proves the structural change doesn't move numerics.
3. **HLO slot-count test** at synth scale predicting ≤ 3 FFT-box slots in the new `_kernel`. **DO NOT do the CrI3 HLO test yet** — that's a separate orchestrator-scheduled run.

## Commit & log

- Single commit on `agent/zeta-bc-scan-shardmap` covering all three sub-tasks. Commit message must enumerate the three changes and quote the bit-identity test result.
- Append final status to `integration_log.md`: what's committed, what's deferred, any gotchas.
- Print "Agent 2 integration done" when you've committed AND the integration_log.md is published.

## Constraints

- **Work on `sources/lorrax_B` only**. Do not modify lorrax_A this round.
- Branch stays at `agent/zeta-bc-scan-shardmap`. Don't create a new branch.
- Re-read `round3_discussion.md` before each substantive edit cycle. Respond to Agent 4 messages.
- The principle: zero replicated intermediates. After your commit, the `_kernel`'s HLO should show ≤ 3 FFT-box-class slots, not the current 58.

Start by reading `round3_discussion.md` and updating it with your "Agent 2 starting on integration" announcement so Agent 4 sees you're live.
