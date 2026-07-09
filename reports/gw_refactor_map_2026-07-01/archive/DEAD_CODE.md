# LORRAX GW — dead code, suspected bugs, refactor targets

From 1049 adversarial verdicts (`_raw_verdicts.json`), each a reader claim that a high-effort
verifier tried to disprove (enumerate all call mechanisms before "dead"; explicit index/sign
math before "bug"). Verdict tally:

| verdict | n | meaning |
|---|---|---|
| intentional-convention | 295 | leave alone (BGW convention, documented) |
| confirmed-dead | 212 | delete candidates |
| benign-cruft | 208 | harmless; opportunistic cleanup |
| refactor-target | 185 | consolidation targets (see MAP.md §4) |
| suspected-bug | 74 | **needs a physics/owner look** |
| test-only | 48 | keep, isolate from prod path |
| alive | 27 | reader was wrong; not dead |

**Read order: §1 suspected bugs first** (these can be live wrong-answer paths — a refactor that
"preserves behavior" preserves the bug). Then §2 confirmed-dead (the delete pass). §3 refactor targets.

---

> **Updated 2026-07-02** — reconciled against `agent/memplanner-cleanup`
> (`git log e7b6c7d..HEAD`, verified against diffs + current code, not DONE_SUMMARY).
> The big delete passes have LANDED: the whole `aot_memory_model/` package (554bcbe), the entire
> r-space V_q "tile" subsystem (`v_q_tile.py` + FFT-era trio + tile factories, 8369ecc/d4bb3ba/6809e64),
> 5 verified-dead modules (dd03ce6), `gw/vcoul.py` strip (8a079c7), cohsex.in parser single-source
> (4ea6cb6), and the config knobs `use_aot_chunk_chooser`/`chunk_chooser_mode`/`async_prefetch`.
> Deleted items are marked ✅**DELETED** with the commit. Two §1 suspected bugs went **MOOT** with the
> tile deletion. **Everything still-open is surfaced in the box below** — that is the live backlog.
>
> **Updated 2026-07-03** — reconciled against `agent/memplanner-cleanup` HEAD (`bb04399`),
> verified against the diffs `61ae4b8`, `d55c4cb`, `c9fb0e2`, `62ce45e`, `dfb6b90`, `94cc354`,
> `a697abc`, `842c3c6`, `04ff0a9`, `ab5d11f`, `bb04399` + current code. New landings:
> **`wfn_writer` factor-of-2 occupation FIXED** (61ae4b8; `nelec//2` for nspinor=1, confirmed at
> `wfn_writer.py:102,141`) — the run_nscf nk>1 half stays open. **ISDF split**: `isdf_fitting` moved
> `common/`→`gw/` (c9fb0e2) and its neutral core carved into a NEW `src/isdf/` package (dfb6b90) —
> all `isdf_fitting.py` path/line refs below now resolve to `gw/isdf_fitting.py` OR `isdf/core.py`.
> **Reader cleanup**: `common/load_wfns.py` facade DELETED (bb04399); the wfn_loader inline σ·p lift
> and ψ-unfold copies single-sourced away (842c3c6 → `bispinor_init.lift_to_4spinor`; a697abc →
> `symmetry_maps.trs_augment_U`/`tau_phase_row`); wave-1 boilerplate copies collapsed (94cc354).
> `common/bispinor_init.py` is **no longer a delete candidate** — S1 promoted it to the production
> single-source. **Zeta-class merge stays DEFERRED** (genuine divergence; needs padded-μ fixture).

### ⏳ STILL-OPEN — the actionable backlog (nothing below here is done)

Attack order is §1 suspected bugs (live wrong-answer risk) before §3 consolidation. The highest-value
open targets:

1. **`head_correction.py:320-332` (`fit_head_ppm`) negative-Ω² branch flips the head sign** — live
   wrong-answer on any q where the head plasmon fit goes imaginary; `head_correction.py` untouched by
   the branch (still present, verify line nos). The new GN-PPM regression gate (e7646e1) will not catch
   this unless a gate system exercises an imaginary head.
2. **`ppm_pipeline.py:365 / head_correction.compute_ppm_head_sigma_kij` — streamed Σc silently skips the
   analytic q→0 head.** `_inject_analytic_head` (ppm_pipeline.py:109) returns early in streamed
   `_StreamedH5Accumulator` mode ⇒ streamed GN-PPM writes Σc with no head, uncompensated. Both symbols
   still live.
3. ~~**`wfn_writer.py` factor-of-2 occupation on nspinor=1**~~ ✅**FIXED** (61ae4b8) — `n_occ`/`ifmax`
   now `nelec//2` for nspinor=1 (`wfn_writer.py:102,141`). The coupled **`run_nscf.py` nk>1 offset
   index math STILL OPEN** — re-verify lines (file was edited; the per-rank reduce loop is now around
   `run_nscf.py:219-267`), corrupts the no-QE / pseudobands WFN path when nk>1.

Then work down §2 "dead functions within live files" (most whole-file deletes are DONE) and the §3
consolidation list (QP/eqp math triplicated; `LorraxConfig`/`PPMSigmaRuntimeOptions`/`LORRAX_*` shadow).

---

## 1. Suspected bugs (74) — triage before moving code

Grouped by likely severity. `⚠` = plausibly changes a physical answer today; `○` = latent
(unreached at current call sites) or contract/comment mismatch. Full evidence: `_raw_verdicts.json`
(filter `verdict=="suspected-bug"`), and per-file notes in `files/*.md`.

### ⚠ Live wrong-answer candidates

- **`ppm_pipeline.py:365` streamed Σc silently skips analytic q→0 head.** `compute_ppm_head_sigma_kij`
  (in `head_correction.py`) has one caller (`_inject_analytic_head`, `ppm_pipeline.py:109`) which returns
  `(None, None)` when `sigma_c_omega is None` (streamed `_StreamedH5Accumulator` mode). Streamed GN-PPM
  runs write Σc with **no head**. Not compensated elsewhere. **STILL OPEN** — both symbols verified present.
  (Line moved 126→365 since original scan.)
- **`head_correction.py:320-332` (`fit_head_ppm`) negative-Ω² branch flips head sign.** With
  Ω_h²<0 the code coerces `ω_h=|Ω_h²|^½` and `R_h=B_h/(2|Ω_h²|^½)`; explicit algebra shows the
  static-limit head sign inverts vs the physical `W^c(ω)=B_h/(ω²−Ω_h²)` pole. Fires on any q where
  the head plasmon fit goes imaginary. **STILL OPEN.**
- ~~**`compute_vcoul.py:231` `_v_munu_kernel_cache` key omits `vcoul_cutoff_ry`, ...**~~ ✅**MOOT** —
  `make_v_munu_chunked_kernel` + `_v_munu_kernel_cache` were the r-space tile kernel, **DELETED** in
  8369ecc (r-space V_q tile subsystem). No cache, no stale-key bug.
- ~~**`wfn_writer.py:99-100,135-137` factor-of-2 occupation on nspinor=1.**~~ ✅**FIXED** (61ae4b8):
  `n_occ = nelec if nspinor==2 else nelec//2` at `wfn_writer.py:102,141`. The separate
  `run_nscf` nk>1 offset-index-math half remains **STILL OPEN** (re-verify lines — file edited;
  per-rank reduce now ~`run_nscf.py:219-267`). Affects the no-QE / pseudobands WFN path.
- **`coulomb/box_0d.py:29,51` 0D q0-average divides by `cell_volume` while Bulk3D/Slab2D use the
  volume-free `8π/|G|²` convention** — a real convention split with a live wrong-answer path for 0D box runs.
  **STILL OPEN.**
- **`get_DFT_mtxels.py:533,556` hardcode `truncation_2d=True`** (V_H, V_loc) while `:717` threads
  `ctx.truncation_2d` and `:738` hardcodes False — inconsistent truncation in kin_ion.h5 generation.
  Note: `get_DFT_mtxels.py` was edited this branch (97-line diff, get_dipole_mtxels sign work) —
  **re-verify line numbers** before touching.
- **`charge_density.py:242` NLCC gate `str(cc)!='UpfLogical.T'`** misfires: the enum has TRUE/T/FALSE/F;
  a `.true.`-spelled UPF core-correction flag is not `'UpfLogical.T'` ⇒ NLCC silently skipped for those pseudos.
  **STILL OPEN.**
- **`charge_density.py:159-171` density star-average drops the non-symmorphic `e^{-2πiG·t}` phase**
  though `SymMaps` carries the translations — wrong symmetrized density on non-symmorphic groups. **STILL OPEN.**
- **`build_projectors_qe.py:743` defaults `j=l+½` when the `jjj` annotation is absent** (cascades from
  `load_upf.py:34-95` silently dropping `PP_RELBETA` on any parse hiccup) ⇒ wrong SOC splitting for the
  down-j projector. **STILL OPEN.**
- **`ffi/slate/eigh.py`** — heev eigenvectors wrong by an unpinned layout transform; **known,
  documented, unresolved**. Any SLATE-eigh path (QSGW Hermitise) is suspect.
- **`sternheimer_solve.py:372-377` JVP has a flipped `b_dot` sign** (double-negation error) — wrong
  linear-response derivative; affects Sternheimer-based screening/head (future path). **STILL OPEN.**

### ○ Latent / contract-mismatch (unreached now, or comment-vs-code)

- **`gw_jax.py:237` `quad,e_ref` bind only under `if config.do_screened:` but consumed at L421 (dynamic)
  and by SC** — latent `NameError` on the `do_screened=false` + dynamic/SC combination. (gw_jax.py had a
  42-line diff this branch — re-verify lines.)
- **`qsgw_utils.py:128-136` fixed-point convergence metric** mixes `mix·|ΔE|` (in-loop) vs `|ΔE|`
  (post-loop) — inconsistent tolerance; unreachable at current call sites.
- **`zeta_reader.py` G-flat branch validates caller sphere by size only** — a same-size but
  different-ordering sphere reads wrong ζ data (couples to the A2→A3 layout contract, MAP.md §3).
  **STILL OPEN** — re-verify line (file untouched by wave-1/2 except the `bind_mf_attrs`/`bind_isdf_attrs`
  boilerplate adoption). The `zeta_loader`/`zeta_reader` **merge is DEFERRED** (genuine divergence:
  backend semantics + valid_mu zero-fill; needs a padded-μ fixture — READER_CLEANUP_PLAN.md step 4).
- ~~**`v_q_tile.py:458` Case-B fallback `mu_chunk=max(snap, mu_chunk_max-mod)`**~~ ✅**MOOT** —
  `v_q_tile.py` **DELETED** (8369ecc). Whole file gone.
- **`symmetry_maps.py:1121-1127` unmapped k silently assigned NEAREST irreducible point** (warn only)
  — the exact failure class as the historical TRS-blind bug; should hard-fail. **STILL OPEN.**
  (Lines shifted from 1050-1063 — `symmetry_maps.py` gained `trs_augment_U`/`tau_phase_row`/public
  `get_umklapp_vector` in the S2 reader cleanup, a697abc.)
- **`gpu_utils.py:97 vs 149`** budget policy inconsistency (`0.9·bytes_limit` vs `0.9·bytes_available`)
  with a live per-rank-divergence failure path.
- **`async_io.py:89-95` `close()` drains before poison-pill** ⇒ worker can survive on error path.
- **`_slab_io_ffi.py:525-538` read-after-write hazard** when `ds_id` cached (skips `_drain_pending`).
- Comment-vs-code (harmless but misleading trails, fix while touching): the "SVD pseudoinverse"
  trail moved with the ISDF split — now `gw/isdf_fitting.py:309,419` + `isdf/core.py:992` (the
  `solve_zeta` dispatch does have an SVD-pinv branch, so re-check whether this is still a real
  mismatch before touching), `w_isdf.py:456`
  host-sync vs "no host gathers" docstring, `fft_helpers.py:85-100` "logged so caller notices" (no log),
  `dos.py:262` no-op `−E_cross+E_cross`, `lanczos.py:281` final-column clamp overwrite,
  `head_wing_schur.py:126` "gw_jax passes 2.0" (false).

### Stale flags / self-admitted convention debt

- **`kmeans_cli.py` flag is `--orbit`/`--no-orbit`, but `v_q_bispinor.py` + `docs/theory/symmetry.md:505`
  tell users `--orbit-aware`** (doesn't exist) — stale user-facing instruction. Note: `v_q_bispinor.py` was
  heavily cut this branch (351-line diff — `compute_V_q_bispinor_to_h5` + 2 factories removed);
  **re-grep for the stale `--orbit-aware` string** before fixing.
- **`qe_save_reader.py:24-26`** self-admits 24/48 non-symmorphic translations have a sign mismatch vs pw2bgw.
- **`qe_save_reader.py:137`** `alat=|a1|` assumption inconsistent with XML-read b-vectors.
- **`LORRAX_FORCE_FULL_BZ`** — confirmed this branch to be a **DEV/gate seam only** (user decision, NOT a
  user-facing flag). It now honors the tile-free G-flat path (0d7ba06) and backs the new IBZ-vs-full-BZ
  equivalence gate (1479162). Do not document it as user-facing; do not delete it.

---

## 2. Confirmed-dead (212) — the delete pass

Zero live callers after the verifier checked imports, string-dispatch, CLI (`python -m`,
console_scripts), tests/, tools/, scripts/, and the sandbox `skills/`+`runs/*.sh`. **The whole-file
tier has largely LANDED on `agent/memplanner-cleanup`** — see marks below.

### Whole files (Tier F — delete outright)
| File | Evidence | Status |
|---|---|---|
| `psp/archive/projector_pipeline.py` | unimportable (broken relative imports), zero callers | ✅**DELETED** dd03ce6 |
| `psp/archive/charge_density.py` | unimportable (stale import path) | ⏳ **STILL PRESENT** — not in the dd03ce6 batch; safe to delete next pass |
| `psp/archive/build_projectors.py` | zero callers | ✅**DELETED** dd03ce6 |
| `solvers/cg_posdef.py` | zero importers; algorithm reimplemented inline in `sternheimer_solve` | ✅**DELETED** dd03ce6 |
| `common/bispinor_init.py` | ~~legacy lift; test oracle → move into test~~ | ✅**KEPT / PROMOTED** (842c3c6): S1 made it the production single-source (`lift_to_4spinor` + `HALFALPHA`, `bispinor_init.py:16,51`); wfn_loader's inline copy DELETED. **No longer a delete candidate.** |
| `centroid/centroid_io.py` | orphan duplicate of `file_io/centroids.py` | ✅**DELETED** dd03ce6 |
| `common/chi_sos.py` | fully unwired SOS χ₀ head/wing backend; 5 public fns, 0 callers | ✅**DELETED** dd03ce6 |
| `gw/experimental/head_wing_schur.py` | production-quality but **0 production callers** (test-only); decide promote-or-delete | ⏳ **STILL PRESENT** — promote-or-delete decision open |
| `gw/v_q_tile.py` (1662 L) | entire r-space V_q "tile" driver | ✅**DELETED** 8369ecc (survivor `_unfold_g0_ibz_to_full` relocated to `v_q_g_flat.py:556`) |
| `aot_memory_model/` (whole package, ~8.7k L incl artifacts) | superseded planner; `chooser.py` was **dead by call-order clobber** | ✅**DELETED** 554bcbe — `gflat_memory_model.py` is now the SOLE planner; `runtime/aot_memory.py` kept (live cuFFT query); planner self-check fixed (ee6c447) |

### r-space V_q subsystem (~3k+ L) — ✅**DELETED** (8369ecc / d4bb3ba / 6809e64)
The entire FFT-era / tile V_q path is gone; live V_q is **G-flat only**:
- `v_q_tile.py` (whole file) ✅
- `v_q_bispinor.compute_V_q_bispinor_to_h5` + its 2 factories ✅ (survivor: `compute_V_q_bispinor_g_flat_to_h5`)
- `compute_vcoul.make_v_munu_chunked_kernel` + the FFT-era trio
  (`fft_integer_axes` / `_v_q_per_q_g_chunked_jit` / `compute_v_q_per_q_g_chunked`) ✅ (grep: 0 hits)
- `compute_all_V_q` r-space tail → now raises `NotImplementedError` for non-G_flat layout
  (`compute_vcoul.py:126/182`); dispatches to `v_q_g_flat.compute_all_V_q_g_flat` (charge) /
  `v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` (bispinor) ✅
- `coulomb_sphere.compute_bare_coulomb_sphere_idx` ✅ (grep: 0 hits)
- `gw_init` legacy else-branch ✅ (297-line cut, gw_init.py)

### Config knobs — ✅**DELETED**
- `use_aot_chunk_chooser`, `chunk_chooser_mode` — ✅ gone (grep: 0 hits)
- `async_prefetch` (compat no-op) — ✅ removed (grep: 0 hits)
- `compute_all_V_q` dead params `n_rmu` / `n_rtot` / `budget_bytes` / `use_g_flat_zeta` — ✅ stripped

### Reader cleanup — duplicate copies ✅**DELETED / single-sourced** (94cc354 / a697abc / 842c3c6 / 04ff0a9→bb04399)
The reader boilerplate and ψ-loading duplication is collapsed. Net −59 LOC (wave 1) + the whole
`load_wfns.py` facade gone:
- `common/load_wfns.py` (facade, ~522 L) — ✅**DELETED** (bb04399). Its 5 live ψ-loading helpers
  (`get_enk_bandrange`, `load_kpoint_fftbox`, `read_Gvecs_to_devices`, `iter_psi_rchunk_bandwise`,
  `load_centroids_band_chunked`) relocated byte-for-byte to `common/wfn_transforms.py` (04ff0a9),
  22 consumer sites repointed (ab5d11f); `common.get_enk_bandrange` stays public via `__init__`.
- wfn_loader inline **σ·p small-component lift** copy + its `_HALFALPHA` — ✅ gone (842c3c6);
  single-sourced to `bispinor_init.lift_to_4spinor` / `HALFALPHA`.
- wfn_loader inline **ψ-unfold rule** (spinor + τ-phase) copy — ✅ gone (a697abc); single-sourced to
  `symmetry_maps.trs_augment_U` / `tau_phase_row`. The eager slurp was also made lazy (OOM fix) and
  `_get_umklapp_vector` promoted to public `get_umklapp_vector`.
- wave-1 boilerplate copies — ✅ single-sourced (94cc354): 35-field mf_header mirror →
  `mf_header.bind_mf_attrs`; isdf_header mirror → `isdf_header.bind_isdf_attrs`; `_rank0`/`_barrier`
  → `_slab_io_ffi`; `kpt_starts` prefix-sum → `mf_header.kpt_starts`.
- **Zeta-class merge (`zeta_loader`/`zeta_reader`)** — ⏳ **DEFERRED** (genuine divergence; needs a
  padded-μ fixture; READER_CLEANUP_PLAN.md step 4).

### Dead functions within live files (top offenders; full list in `_raw_verdicts.json`)
Still to sweep alongside each file's next edit (whole-file tier above already accounts for several).
**Path note**: `isdf_fitting.py` split into `gw/isdf_fitting.py` (orchestrator) + `isdf/core.py`
(primitives) — its dead-fn cluster now spans both; the neutral primitives live in `isdf/core.py`.
`ppm_sigma.py` (6), `head_correction.py` (4), `isdf_fitting.py`→`gw/`+`isdf/core.py` (4),
`pseudobands_v2.py` (4), `w_isdf.py` (3), `gw_init.py` (3), `compute_vcoul_0d.py` (3),
`cholesky_2d.py` (3), `psi_G_store.py` (3), `ffi/phdf5/read.py` (3), `ffi/common/ffi_loader.py` (3),
`get_dipole_mtxels.py` (3), `build_projectors_qe.py` (5), `load_upf.py` (3), `radial_jax.py` (3),
`run_sternheimer.py` (3).
(`compute_vcoul.py` dead-fn cluster and `vcoul.py` cluster below are now resolved.)

- `gw/vcoul.py` — ✅**RESOLVED** (8a079c7): stripped 234→68 L, kept the 2 live helpers
  (`wrap_points_to_voronoi`, `compute_q0_averages`); the commented-out legacy builder + 0-caller fns gone.
- `gw/compute_vcoul.py` — ✅ mostly resolved by the r-space deletion (967-line cut); the tile dead-fn
  cluster is gone. Remaining is the dispatcher + G-flat entry only.
- `cohsex.in` parser — ✅**SINGLE-SOURCED** (4ea6cb6): was x2, now x1.

## 3. Test-only (48) — keep, isolate

Referenced only by tests/benches. Keep them but ensure they're not on any prod import path:
`solvers/minres.py` (superseded by inline CG), `solvers/projectors.py`, `common/*_test.py` benches,
`mixing/benchmark_synthetic.py`. (~~`common/bispinor_init.py`~~ — no longer test-only; it is now the
production single-source for the σ·p lift, S1/842c3c6.) The ~20 `common/*_{test,bench}.py`
FFI validation CLIs (cusolvermp/slate/cublasmp) are diagnostics (Tier E), not pytest — fine to keep,
consider moving under `tools/` or `tests/` so `common/` stops hosting a diagnostics category.

**New gates added this branch** (keep, do not treat as dead): GN-PPM regression gate
(`test_gw_jax_regression.py`, parametrized cohsex + gnppm, e7646e1 + gate-0 crash fix 6dbb3b4 +
re-frozen ref 143dd99) and the IBZ-vs-full-BZ equivalence gate (`test_ibz_full_bz_equivalence`,
1-GPU MoS2, 1479162). Deleted stale tests: `tests/archive/test_chunked_wfn_loading.py`,
`test_compute_all_V_q_g_flat.py`, `test_per_q_sphere.py`, `test_v_q_transverse_unfold.py`.
**Added since (2026-07-03):** the first BGW-anchored **sys_dim=3** e2e gate — Si 4×4×4 no-SOC COHSEX
(`tests/regression/si_cohsex_debug/`, `si_cohsex_3d` case, native Σ body + BGW head scalars, anchored
~few-meV, atol 1e-3, d55c4cb). Suite discipline (25067f4, "red means red"): 4 container-JAX env
failures now **conditionally skip-marked** (not silent), 3 golden gates wired into
`skills/checkpoint/SKILL.md`; baseline **242 passed / 24 skipped / 0 failed**.

## 4. Refactor targets (185)

The consolidation list — see **MAP.md §4** for the ranked, mechanism-annotated version. Highest-count
files after this branch's cuts: `sigma_x_bispinor.py` (7), `gw_jax.py` (9 across two spellings),
`gw_driver_helpers.py` (4), `sc_iteration.py` (4), `qsgw_utils.py` (4), `cohsex_sigma.py` (4).
(`v_q_tile.py` (4) — ✅ moot, file deleted.) Recurring theme:
QP/eqp math duplicated across `eqp_bgw`/`gw_output`/`gw_jax`; the cohsex.in parser is now single-source
(was 3 copies — one collapse landed 4ea6cb6, verify no third copy remains);
`LorraxConfig` mirrored by `PPMSigmaRuntimeOptions` + a shadow `LORRAX_*` env surface (still open).
Also landed: 4 `_rotate_psi` variants → 2 (`_rotate_psi_bandlast` / `_rotate_psi_bandfirst`,
byte-identical einsums, 466040a).
