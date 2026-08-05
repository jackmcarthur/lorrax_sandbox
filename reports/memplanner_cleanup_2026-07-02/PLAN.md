# Memory-planner cleanup — PLAN

_2026-07-02 · repo `sources/lorrax_D` · read/grep-only survey_

Three distinct "memory model" bodies live in the tree. Naming them precisely
matters because the task title collapses two of them:

| Name | Path | LOC | Kind | Production status |
|------|------|-----|------|-------------------|
| **gflat** | `src/gw/gflat_memory_model.py` | 1007 | closed-form 5-peak analytic model | **LIVE** — sole driver of `band_chunk`/`chunk_r`/`gflat_chunk_size` |
| **aot_memory_model** | `src/gw/aot_memory_model/` (24 files) | 3488 | DOE + NNLS-fit + preset + chooser framework | **DEAD-by-clobber** (see §1) |
| **runtime/aot_memory** | `src/runtime/aot_memory.py` | 519 | cuFFT-scratch AOT query via `cufftGetSize` | **LIVE** — used by V_q tile chooser |

Total under review ≈ 5000 LOC + a 1010-line doc.

---

## 1. Verdict: gflat vs aot_memory_model — REDUNDANT, delete the package

**They are redundant, and `aot_memory_model/` should be deleted (absorb nothing).**

### The clobber is confirmed

In `gw_init.fit_zeta` the two run back-to-back:

- `gw_init.py:574` `_apply_aot_chunk_model(...)` — if `use_aot_chunk_chooser`
  (default **False**), it sets `chunks['chunk_r']` and `chunks['band_chunk']`
  from `choose_chunks_{heuristic,analytic}`.
- `gw_init.py:606` `plan_gflat_chunks(...)` — runs unconditionally, then at
  `gw_init.py:635-636` **unconditionally overwrites** `chunks['band_chunk']`
  and `chunks['chunk_r']` with the gflat plan.

So even with `use_aot_chunk_chooser=True`, the AOT chooser's picks are
immediately clobbered. The flag defaults False; no run dir sets it. The
chooser has **zero** effect on production sizing.

The only surviving effect of the whole `aot_memory_model` import is the
**predict-only γ-calibration print**: `_apply_aot_chunk_model` returns
`predict_kernel_peak(...)`/1e9, consumed at `gw_init.py:739-742` to print
`γ = peak_gb / aot_peak_gb`. That is one diagnostic log line backed by 3488
LOC + 13 JSON artifact pairs.

### What each uniquely provides that is actually USED

- **gflat**: everything. It is the production planner. Keep.
- **aot_memory_model**: nothing that reaches a production decision. Its
  `predict_kernel_peak` feeds one γ print; `choose_chunks_*` are clobbered;
  `doe`/`sweep`/`presets`/`predict_cli`/`cost`/all `kernels/*` are offline
  calibration scaffolding with no in-tree caller outside the package + its own
  CLI. `choose_chunks_aot` / `build_doe_axes` / `aot_measure` are exported but
  have **no importer anywhere** (grep-clean outside the package).
- **runtime/aot_memory** (separate, keep): genuinely live — `v_q_tile.py:220`
  calls `aot_kernel_peak_bytes(compiled)` inside `_v_q_full_kernel_aot` to add
  real cuFFT plan scratch to `memory_analysis()` for the V_q chooser. This is
  the ONE AOT idea worth keeping and is already isolated in `runtime/`.

**Decision:** delete `src/gw/aot_memory_model/` wholesale (~3488 LOC + artifacts).
Replace the γ-calibration line with the gflat HWM the planner already computes
(or drop it). Keep `runtime/aot_memory.py`. Keep gflat as the single planner.

---

## 2. De-overengineering delete / inline list (ranked, ponytail)

Ranked by LOC removed and safety. "Caller impact" = what breaks.

| # | Target | Move | Caller impact | LOC |
|---|--------|------|---------------|-----|
| 1 | `src/gw/aot_memory_model/` entire package (core, cost, chooser, doe, sweep, presets, predict_cli, kernels/*, artifacts/*) | **delete** | Only `gw_init._apply_aot_chunk_model` + `gw_config` flags import it; both removed in #2/#3 | **~3488** |
| 2 | `gw_init._apply_aot_chunk_model` (gw_init.py:413-501) + its call site (574) + γ block (739-742) | **delete**; if γ print wanted, log `gflat_plan.hwm_bytes` instead | Self-contained; `fit_zeta` keeps only the gflat call | ~110 |
| 3 | `gw_config` chooser knobs: `use_aot_chunk_chooser`, `chunk_chooser_mode` (defaults 228-235, dataclass fields 612-613, parse 969-970, doc 222-233) | **delete** flag + field + parse + comment | cohsex.in stops accepting two dead knobs; MemoryConfig shrinks | ~25 |
| 4 | gflat `use_query_fft_peak_bytes` path (`_peak_D_accumulate` 483-499 + param plumbing 433, 658, 697-698, 931) | **delete** — never set True anywhere; gflat_chunk_size capped at 100 makes it "mostly cosmetic" per its own docstring | Peak D uses the constant `factor_D=2.0` formula it already falls back to | ~30 |
| 5 | gflat legacy alias `fft_box_factor` (638,647,705-706) + `band_chunk_override` dead-branch tail (887-888 `if...: pass`, 890-898 re-check) | **inline**: keep one `fft_box_factor_A`; drop no-op `pass` and the duplicate override warning | Caller (`gw_init.py:618`) passes `fft_box_factor=4.0`; rename to `fft_box_factor_A` | ~20 |
| 6 | gflat two sphere-buffer constants both `= 1` (`N_SPHERE_IDX_BUFFERS_BISPINOR`/`_CHARGE`, 207-208) + `is_bispinor` branch at 726 | **inline** to a single `1` (post-Round-6 both collapsed to 1; the branch is a no-op) | none — value identical | ~10 (+~40 comment) |
| 7 | gflat `GFlatChunkPlan.format()` per-peak component dump (244-258) | **simplify** — keep peak totals, drop the grouped per-term pretty-printer unless a run actually reads it | log gets shorter | ~15 |
| 8 | gflat override-warning recompute blocks (r_chunk 794-800, cs 845-859) | **simplify** to one-line warnings; the inline peak recompute duplicates `_peak_*` formulas | warning text less precise | ~25 |
| 9 | `runtime/aot_memory.py` R2C/C2R/`f32`/`f64` cuFFT branches (`_cufft_type_for`, RFFT/IRFFT dist logic 418-424) | **delete** — LORRAX FFTs are all c128 C2C (Z2Z); real-FFT support is speculative | HLO parser + query only ever see FFT/IFFT c128 | ~30 |
| 10 | Three parallel FFT-scratch estimators: gflat Peak-D factor, `v_q_tile._aot_fft_model` (slope/intercept fit), `runtime.aot_memory` (cufftGetSize) | **consolidate** onto `runtime.aot_memory` / `fft_helpers.query_fft_peak_bytes` (real query); retire the slope/intercept two-point fit in `_aot_fft_model` if the direct query covers V_q | V_q chooser switches to the direct-query peak it already computes at 221 | ~60 |

**Estimated removable: ~3800 LOC** (dominated by #1), plus ~100 LOC of comment
bloat trimmed from gflat's docstrings. gflat itself shrinks from 1007 → ~800.

---

## 3. Consolidation target: one planner home

**Home:** keep `src/gw/gflat_memory_model.py` as the single analytic planner,
renamed conceptually to "the memory planner" (module docstring already owns the
A–E peak taxonomy). It is closed-form, microsecond, unit-tested, and is the
default per the docs. Everything else folds toward it or toward
`runtime/aot_memory.py` for the one real-measurement need.

What moves / what each embedded site needs as a seam:

- **`gw_init.py`** (`fit_zeta`, `_apply_aot_chunk_model`): remove the AOT
  detour entirely; `fit_zeta` calls `plan_gflat_chunks` once and reads
  `band_chunk`/`chunk_r`/`gflat_chunk_size`/`hwm` off the returned plan. Seam:
  the existing `GFlatChunkPlan` dataclass — already the clean boundary.
- **`v_q_tile.py`** (`_choose_v_q_chunks`, `_v_q_full_kernel_aot`,
  `_aot_fft_model`): Peak E (V_q) already lives in gflat
  (`_peak_E_v_q_per_tile_transient`). The V_q chooser and gflat's Peak E are
  two models of the same kernel. Consolidate: let gflat's Peak E be the sizing
  source and keep `runtime.aot_memory` only for the final live cuFFT-scratch
  correction. Seam: a small `v_q_peak_bytes(...)` entry in gflat that the
  chooser calls, plus one `aot_kernel_peak_bytes` add-on for cuFFT scratch.
- **`isdf_fitting.py`**: no model logic to move — it is the runtime being
  modeled. Seam is unchanged: it receives `chunk_r`, `band_chunk_size`,
  `q_chunk_size` scalars.
- **`fft_helpers.query_fft_peak_bytes`**: this is the real-measurement FFT
  primitive. Keep it as the ONE FFT-scratch query; have gflat Peak D and the
  V_q chooser both call it (or `runtime.aot_memory`) instead of maintaining
  three separate factor/slope/query estimators (see delete-list #10).

Net: two homes with a clear split — **gflat** = closed-form sizing (all peaks
A–E), **runtime/aot_memory + fft_helpers** = live cuFFT-scratch correction when
a real number is needed. No third framework.

---

## 4. Docs revision outline: `memory-model.md` (1010 lines) → lean

Keep the physics/derivation; cut the dead-framework prose and fix contradictions.

**Keep (core, ~500 lines):**
- Intro + Stage Summary + per-process footprint (1-116)
- Band/R/Q/μ chunk derivations (117-312) — the actual formulas
- IBZ cascade memory (313-361), ψ(G) host store (362-382)
- **G-Flat Memory Model** section (383-538) — this is the live planner; make
  it the centerpiece, promote to top
- Automatic Sizing Algorithm + Recipe + worked example (567-650)
- live_arrays appendix (940-999) — verification ground truth, keep

**Cut (~350 lines):**
- Entire **AOT Memory Model** section (701-886): architecture, covered
  kernels, NNLS-in-practice, "When to trust which chooser", γ calibration,
  Status(2026-05-15). Replace with a 3-line note: "cuFFT plan scratch is
  measured live via `runtime/aot_memory.py`; see V_q chooser."
- XProf Workflow (651-669) and Model Corrections (670-700) — merge the one
  still-true correction into the G-Flat section, drop the rest.

**Merge:** the "Predicted-vs-realized faithfulness" audit (887-939) → a short
subsection under G-Flat (the 14%-conservative HWM finding is worth one para).

**Code-vs-doc contradictions to fix:**
- Docs (795-864) present four live choosers ("G-flat / legacy heuristic / AOT
  20/80 / AOT analytic") and advise "when to trust which." In code only gflat
  drives sizing; the AOT choosers are clobbered and the legacy heuristic's
  `chunk_r`/`band_chunk` are also overwritten by gflat. **Doc oversells three
  dead paths.**
- Docs "Status" (865-886) call the AOT model "scaffolded and calibrated";
  reality is it is unreachable in production. Delete.
- Docs still reference `gflat_to_rchunk_chunk_size` knob semantics that
  `gw_init.py:645` says Round-6 deleted — scrub stale knob references.

Target: ~1010 → ~550 lines, single planner narrative.

---

## 5. Attack order

**Phase 0 — establish the green gate (no edits).**
Run and record baselines: COHSEX regression gate, GN-PPM regression gate, and
planner unit tests `test_band_chunk_size_floor.py`,
`test_planner_refit_2026-05-17.py`, `test_aot_memory.py`. These gate every
subsequent phase. `test_rchunk_gflat_pair.py` guards the accumulate kernel.

**Phase 1 — safe deletes (gated by the above).** Order by isolation:
1. Delete `src/gw/aot_memory_model/` package + artifacts (delete-list #1).
2. Delete `gw_init._apply_aot_chunk_model` + call + γ block (#2); swap γ print
   for `gflat_plan.hwm` or drop.
3. Delete `gw_config` chooser knobs (#3).
4. Delete gflat `use_query_fft_peak_bytes` path (#4) and
   `runtime/aot_memory` real-FFT branches (#9).
   → Re-run all Phase-0 gates. `test_aot_memory` must stay green after #9
   (only real-FFT parse cases change — verify none are asserted).

**Phase 2 — gflat simplification (gated).**
5. Collapse sphere-buffer constants + `is_bispinor` branch (#6), legacy alias +
   dead override branches (#5, #8), format() trim (#7).
   → `test_planner_refit_2026-05-17` + `test_band_chunk_size_floor` must stay
   green (they assert x4 centroids, cs cap=100, mesh floor, sphere-idx in every
   peak — all preserved).

**Phase 3 — consolidation (gated + re-verify a real run).**
6. Fold V_q sizing onto gflat Peak E + `runtime.aot_memory` cuFFT correction;
   retire `_aot_fft_model` (#10). This touches live V_q sizing → re-run a small
   CrI3/Si V_q to confirm HWM within ~10% and no OOM before committing.

**Phase 4 — docs.** Rewrite `memory-model.md` per §4 once code is settled so the
doc matches the single-planner reality; fix the three contradictions.

Checkpoint (pytest + commit + report + CHANGELOG) after Phase 1 and Phase 3.
