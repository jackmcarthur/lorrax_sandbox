# Shared context — zeta-fit r-chunk memory model (4-agent study)

You are one of four agents working independently on the same problem. The user-orchestrator has set this up so each of you produces an honest, from-scratch derivation, then we compare drafts and surface what's actually unknown vs. what's just under-documented.

This file is your shared briefing. Read it once, then read what it points at.

---

## 1. The problem in one paragraph

LORRAX's GW driver (`gw.gw_jax`) fits an ISDF representation `ζ(r)` of pair densities by looping over chunks of the real-space grid (the `r_chunk` loop in `fit_zeta_to_h5`). The chunk size sets the binding peak HBM per device. Every "extra" r-chunk pays a fixed FFT tax on both the wavefunction-fetch and accumulator sides, so we want the **largest** `r_chunk` that fits under the memory budget — never smaller. The same fit must handle the bispinor case (μ_L ∈ {0,1,2,3}) where the charge and transverse channels have different `n_rmu` and different factorization (Cholesky vs LU). The existing planner is a mix of a 6-stage `compute_optimal_chunks` in `gw_init.py`, a `gflat_memory_model.py` module, and a separate `aot_memory_model/` tree. The user characterizes it as "outdated / a mess" and wants a clean memory model derived from the *current* code.

Your deliverable: a complete, defensible memory model that produces `(r_chunk, band_chunk, gflat_chunk_size, psig_k_chunk_size)` given `(B, problem geometry, mesh)`, with every tensor accounted for and every magic constant either justified from the HLO or flagged as an open question.

---

## 2. Read order

1. **`/pscratch/sd/j/jackm/lorrax_sandbox/AGENTS.md`** — project conventions (read-only on sources, no edits outside your report, etc.).
2. **`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/AGENTS.md`** — module map for the LORRAX source.
3. **`/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_v_q_g_flat_reference_2026-05-12/report.md`** — the *current* living reference. §5 is the existing prose memory model. **This is the artifact you are critiquing and rebuilding.** Treat it as a primary source but assume it has bugs and gaps.
4. **`/global/homes/j/jackm/scratchperl/lorrax_sandbox/reports/v_q_bispinor_plan_2026-05-08/report.md`** — bispinor physics + V_q plan. Use §1, §4.b, §5 for the bispinor pieces.
5. Code (see §3 below).

---

## 3. Code map (in `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/`)

### r-chunk loop and zeta fit

- `src/common/isdf_fitting.py:1458–2260` — `fit_zeta_to_h5`. Outer r-chunk loop. The thing you are modeling.
- `src/common/isdf_fitting.py:1335–1410` — `fit_one_rchunk`. The fused per-chunk kernel (pair density + factor + solve).
- `src/common/isdf_fitting.py:254–375` — `c_q_from_psi_sm`. Monolithic shard_map for CCT (charge or transverse via γ̃ perm/phase).
- `src/common/isdf_fitting.py:376–475` — `z_q_from_psi_sm`. Same structure, produces ZCT.
- `src/common/isdf_fitting.py:80–115` — `pair_density`. Standalone rank-5 helper (centroid Gram path only).
- `src/gw/gw_init.py:532–700` — `fit_zeta`. Driver that calls the chunker, then `fit_zeta_to_h5`.

### Current memory model — multiple competing pieces

- `src/gw/gw_init.py:154–361` — `compute_optimal_chunks`. The "6-stage" model: per-stage cost `base + αᵢ·cr + cᵢ`, picks `cr` by `min_i (headroom − cᵢ)/αᵢ`. **Look at the α coefficients and per-stage HBM accounting carefully — this is one of the two competing models.**
- `src/gw/gw_init.py:291–361` — `_find_r_chunk` (inner). Returns `{chunk_r, peak, ...}`.
- `src/gw/gflat_memory_model.py` (416 lines total) — `GFlatChunkPlan` + `estimate_gflat_plan`. The "Peak A/B/C/D" per-rank model. **The other competing model.**
- `src/gw/aot_memory_model/core.py` — `AotMemoryModel` class.
- `src/gw/aot_memory_model/chooser.py:109–200` — `choose_chunks_aot` (grid search / NNLS).
- `src/gw/aot_memory_model/kernels/` — per-kernel cost models: `fit_one_rchunk.py`, `pair_density.py`, `load_psi_rchunk.py`, `cct_lr.py`, `zct_lr.py`, `solve_q.py`, `slab_write.py`, `vq_mu_chunk.py`. Each declares its primitives + scalings.

### Bispinor

- `src/common/gamma_matrices.py` — γ̃ perm/phase encoding, `gamma_double_contract`.
- `src/common/bispinor_init.py:12–42` — small-spinor lift `(α/2)(σ·(k+G))ψ_L`.
- `src/gw/v_q_bispinor.py:244–650` — V_q bispinor orchestrator + reader. Not in the zeta-fit memory pool but useful for the per-channel n_rmu plumbing.
- `src/gw/sigma_x_bispinor.py` — Σ_x bispinor consumer.

### Cohsex input surface

- `templates/cohsex.in` and `docs/docs_gwjax/COHSEX_INPUT.md` — input flags: `memory_per_device_gb`, `band_chunk_size`, `r_chunk_size`, `psig_k_chunk_size`, `gflat_chunk_size`, `vq_g_chunk_size`. All default `0` → planner picks. Non-zero overrides planner.

---

## 4. CrI3 80 Ry size point (validation target)

Typical sizes for the CrI3 6×6 80 Ry reference run on 80 GB A100s, 4×4 mesh (`p_x · p_y = 16`):

| Quantity                | CrI3 6×6 80 Ry            | Notes                                                          |
|-------------------------|---------------------------|----------------------------------------------------------------|
| `n_rmu_C` (charge)      | ≈ 1800                    | ~8·n_band                                                      |
| `n_rmu_T` (transverse)  | ≈ 1200                    | smaller than charge                                            |
| FFT grid                | 75 × 75 × 200             | `n_rtot ≈ 1.125 M`                                             |
| `n_G_sph` (per q)       | several thousand          | `ngkmax`                                                       |
| `n_k = n_q`             | 36                        | reduced from up to 400 by symmetry                             |
| `n_band`                | ~400+ (not all loaded)    |                                                                |
| `nspinor` (ns)          | 2                         | bispinor                                                       |
| Mesh                    | 4 × 4 = 16                |                                                                |
| `memory_per_device_gb`  | 60 GB (of 80 GB HBM)      | headroom for cuFFT scratch + XLA aliasing slack                |

Your model should land on chunk sizes that fit under 60 GB/rank at this size. Per `report.md` §7, the empirical working config is `band_chunk=16, r_chunk=auto(~12500), gflat_chunk_size=64, psig_k_chunk_size=6`. Treat this as a sanity check, not a target — your job is to *derive* sizes, not to back-fit.

For comparison, MoS2 3×3 (the easy point) has `n_rtot ≈ 46k`, `n_rmu ≈ 328`, `n_q = 9`, ns=1, on 4× A100-40GB — fits in one r-chunk with defaults.

---

## 5. What's known to be wrong/fragile in the current model

These are flagged in `report.md` §5.8 and elsewhere. Independent verification of each is useful:

- **Unsharded FFT box at CrI3 scale.** `psi_G_store.fetch_psi_rchunk` materializes a band-chunked FFT box unsharded on every rank; W_wfn jumps by a factor of P. Current planner doesn't model the unsharded case — `psig_k_chunk_size=6` is the manual cap.
- **`pair_density_slots = 3`.** Magic constant from one HLO dump. Was 5 in the legacy decomposed chain. Any kernel-structure edit can shift it. Currently hard-coded.
- **`fft_factor = 4.0`.** Single scalar covering cuFFT scratch across multiple call sites with different fusion neighborhoods. Empirical within ~10%.
- **cuSolverMp internal scratch** (~n_rmu² class). Not modeled.
- **No W_vq term in the same planner pass** — V_q chunker runs separately with its own `_pick_g_chunk(ngkmax)` capped at 4096.
- **50/50 split between W_wfn and W_zeta** in `plan_gflat_chunks` — the §5.5 aliasing analysis says `W_pool / pair_density_slots` is the principled ceiling. Two different models give different answers.

Don't take these on faith. Verify in the code. Add your own.

---

## 6. Constraints

- **Read-only on `sources/lorrax_A/`.** No edits to any file under `sources/`. If you discover a code bug, note it in your report — don't fix it.
- **No compute.** Desk research only. No `srun`, no `lxrun`, no Python execution that runs JAX. Reading code, reading docs, reading HLO dumps if any exist on disk — fine.
- **Stay in your own report file.** Write to `reports/zeta_rchunk_memory_model_2026-05-13/agent_<N>.md` only. **Do NOT read `agent_<M>.md` for M ≠ your N.** You'll see the other drafts in the discussion phase after everyone finishes.
- **Stay in your own pane.** No tmux escapes, no talking to the other agents directly.

---

## 7. Output structure (suggested, not required)

Your `agent_<N>.md` should cover these. Header structure is yours.

1. **Tensor catalog.** Every persistent and transient workspace tensor in the zeta-fit. For each: shape (with axis names), sharding, lifetime (which step/jit it's alive in), per-rank c128 bytes formula. Distinguish charge vs transverse where they differ.
2. **Aliasing analysis.** Which transients overlap in lifetime (sum) vs alias (max). Be specific about *why* — point to the XLA module / shard_map structure or the lifetime offsets.
3. **Budget equations.** `B_persist`, `W_pool`, and `W_wfn(band_chunk, …)`, `W_zeta(r_chunk)`, `W_accum(gflat_chunk_size)`, etc. Plus any new terms you find that the current model misses.
4. **r_chunk picker procedure.** Given `(B, mesh, geometry)`, the explicit algorithm. Including how to handle the bispinor charge/transverse difference. Including how to handle the unsharded-FFT-box pathology at CrI3 scale (or explicitly flag it as unmodeled).
5. **Validation at CrI3 80 Ry.** Plug §4 numbers into your formulas. Does your `r_chunk` come out near 12500? If not, why?
6. **Diff against the current code.** Which of (`compute_optimal_chunks`, `gflat_memory_model.py`, `aot_memory_model/`) is closest to right? What needs to go? What needs to be added?
7. **Open questions.** **This is the most important section.** Things you genuinely could not resolve by reading code + the two reports. Magic constants you had to assume. Sharding behavior you'd need to confirm from HLO. Bispinor cases not yet exercised. Be specific. Honest "I don't know" beats confident-but-wrong.

---

## 8. Tone

You're writing for an expert reader who knows JAX sharding, ISDF, GW. Don't pad with background. Be terse where you can, exhaustive where you must (the tensor catalog should be exhaustive — no "etc.").

Now go read the reports, read the code, and write your draft.
