# Round 1 — Agent 2: Compartmentalize: split ppm_sigma.py along the 8 stage boundaries

**Lens**: module decomposition. Where do the file boundaries go so that each concern of the
Σ_PPM engine can be read, tested, and modified without paging through the other four?

**Code state audited**: `sources/lorrax_D`, branch `agent/memplanner-cleanup` @ `3cad3dd`.
NOTE for the round: the map (`SIGMA_PPM_MAP.md`) is slightly stale against this tree —
2A dead-delete has landed (`ppm_sigma.py` is **1632 L**, not 1702; `fit_gn_ppm`,
`_ReduceScatterGpuAccumulator`, `mask_B_mode="explicit"` are already gone), `ppm_invalid_mode`
is wired for `zero`/`2ry` (`ppm_sigma.py:1465-1483`, `_prepare_sigma_state` at :306-358), and
**Bug A is fixed** (`head_correction.py:322-327`, commit `3cad3dd`). **Bug B is still pending**
(`ppm_pipeline.py:126-127` early-returns `None, None` in stream mode). All line refs below are
against this tree.

**Verdict on the lead's framing**: agree — the 8-stage teleology is good bones, and the split
should NOT be one-file-per-stage. Stages S0–S1 are ~125 L combined and S2 is ~50 L; eight files
would scatter the branch→window→τ sign-convention chain that has to be read as one narrative.
The right cut is **four files along the three seams the code already documents about itself**:
host-window-build vs device-τ-kernel (the file's own comment at `ppm_sigma.py:1082-1087`),
device-kernel vs storage (the accumulator protocol comment at :889-895), and driver vs all of
the above. 1632 L → ~360 + ~260 + ~330 + ~600.

---

## 1. The problem — concern inventory, with evidence of the cognitive cost

The file's own docstring already confesses the five concerns (`ppm_sigma.py:36-48`, "File
layout"). Measured against the current tree:

| # | Concern | Stage(s) | Lines (current) | ~L |
|---|---------|----------|-----------------|----|
| 1 | Pole birth + physics state | S0, S1 | `fit_ppm` :679-736, `_SigmaPhysicsState`/`_prepare_sigma_state` :293-358, `PPMBuildResult` :131-140 | 125 |
| 2 | Branch/window construction (host) | S2, S3 | `_SigmaBranch`/`_iter_branches` :186-234, `_SigmaWindow` :151-177, `_materialize_window_mask_B` :265-280, `_build_single_sigma_window` :743-786, `_build_three_sigma_windows` :789-886, `_build_windows_for_branch` :1089-1166 | 360 |
| 3 | Device τ-kernel | S4 | `_make_project_ri_reduce_scatter` :427-512, `_get_sigma_kij_kernel` :515-574, `_get_sigma_tau_kernel` (+`_build_W_t_q`) :577-620, `precompile_sigma` :623-672, kernel caches :283-284 | 260 |
| 4 | ω-projection + accumulators | S5 | `_AccumMode`/`_select_accum_mode` :78-128, `_combine_coeff_with_sigma_tau` :361-393, `_project_tau_onto_omega` :396-424, `_project_tau_onto_omega_np` :923-945, `_SigmaAccumulator` :897-920, `_HostOmegaAccumulator` :948-1026, `_StreamedH5Accumulator` :1029-1079 | 330 |
| 5 | τ-loop + top-level driver | S4/S5 glue, assemble | `minimax_tau_integrate_sigma` :1169-1224, `_integrate_tau_windows_for_branch` :1227-1289, `_run_sigma_branch` :1292-1390, `compute_sigma_c_ppm_omega_grid` :1397-1631, `SigmaOmegaResult` :143-148, host utils :237-262 | 560 |

Concrete costs of the current interleaving (not abstract "big file bad"):

- **Definition/use distance.** `_SigmaWindow` (:151) is consumed 590 lines later by
  `_build_windows_for_branch` (:1089) and 835 lines later by the accumulator that reads its
  `omega_sign/prefactor/project_code` fields (`_HostOmegaAccumulator.begin_window` :986-991).
  Reading any consumer means scrolling past two unrelated concerns.
- **The sign/mask story is told three times in one file**: module docstring :23-34 (branch
  table), `_iter_branches` docstring :204-214 ("Why the flipped signs?"), and
  `_combine_coeff_with_sigma_tau` :368-384 (project-code semantics). Dense bookkeeping that
  isn't first-pass readable is partly a *placement* problem: there is no single place where the
  Σc(−ω)=−Σc(ω)* decomposition is derived once and then merely *referenced*.
- **The jax/numpy mirror pair** `_project_tau_onto_omega` (:396) and
  `_project_tau_onto_omega_np` (:923) must stay in lock-step (the np docstring says "Matches
  the jax version exactly") but sit 520 lines apart with the entire PPM fit and window builder
  between them.
- **The perf-critical SPMD knowledge** (reduce-scatter layout doc :427-476, the
  scan-regression note :463-470 and :1180-1183) is embedded mid-file; it's the material an
  HLO/perf session needs and nothing else does.

`head_correction.py` (815 L) and `ppm_pipeline.py` (409 L) are addressed in §3 — the short
version is they are already *at* module boundaries and mostly need to be left alone.

## 2. The proposal — four modules, three seams

### 2.1 Target layout

Naming follows the existing `gw/` convention (`ppm_sigma.py`, `ppm_pipeline.py`,
`cohsex_sigma.py` — flat `ppm_*` family, no subpackage):

```
gw/ppm_windows.py       (~360 L)  S2+S3   host-side branch + window construction
gw/ppm_tau_kernel.py    (~260 L)  S4      device kernels, kernel caches, AOT precompile
gw/ppm_accumulators.py  (~330 L)  S5      ω-projection + the two accumulators + stream-h5 setup
gw/ppm_sigma.py         (~600 L)  S0,S1,  fit_ppm, _prepare_sigma_state, τ-loop,
                                  driver  _run_sigma_branch, compute_sigma_c_ppm_omega_grid
gw/ppm_pipeline.py      (unchanged role)  S6+S7 sequencer (head injection, at-DFT eval, h5 write)
gw/head_correction.py   (unchanged, see §3)
```

`ppm_sigma.py` keeps its name and its two public exports (`compute_sigma_c_ppm_omega_grid`,
`fit_ppm`, `precompile_sigma` — the exact set `ppm_pipeline.py:36-40` imports), so the only
import-site churn in the whole repo is `precompile_sigma` moving to `ppm_tau_kernel`
(footprint check: no test or script imports `gw.ppm_sigma` directly; the golden gates drive it
through `gw_jax.main`).

### 2.2 What moves where (exact manifest)

**`gw/ppm_windows.py`** — everything that turns (E_A, Ω_q stats, ω-grid, quadrature params)
into host-side `_SigmaBranch` / `_SigmaWindow` lists. Moves:

- `_SigmaWindow` (:151-177) — the module's central vocabulary type; defined here because this
  file is its *producer* and the semantics of every field (`mask_B_mode`, `project`,
  `omega_sign`, `prefactor`) are decided here.
- `_SigmaBranch` + `_iter_branches` (:186-234).
- `_materialize_window_mask_B` (:265-280) — it *interprets* `_SigmaWindow.mask_B_mode`; keeping
  interpretation next to definition is the single-source-of-truth move, even though its caller
  is the driver's τ loop. It's 16 L of jnp.
- `_build_single_sigma_window` (:743-786), `_build_three_sigma_windows` (:789-886),
  `_build_windows_for_branch` (:1089-1166), `_masked_stats_device` (:254-262).
- **New, consolidation not code**: one module-level docstring that is *the* derivation of the
  4-branch Σc(−ω)=−Σc(ω)* decomposition + the three-window (core/a_stripe/b_slab) picture +
  the project full/imag rule — merged from `ppm_sigma.py:23-34`, :204-214, :368-384, leaving
  one-line pointers at the two consumer sites. This is the cheap 80% of the "τ-kernel
  bookkeeping not first-pass readable" complaint: the convention becomes one narrative in the
  file where every sign is *chosen*.

Imports: `minimax_screening` (`MinimaxNodes`, `solve_laplace_minimax_interval`,
`solve_phase_minimax_bandwidth`), numpy, jnp (mask/stat helpers only). No imports from the
other new modules — this is the leaf.

**`gw/ppm_tau_kernel.py`** — the device stage. Moves:

- `_sigma_tau_kernel_cache` / `_sigma_kij_kernel_cache` (:283-284),
  `_make_project_ri_reduce_scatter` (:427-512), `_get_sigma_kij_kernel` (:515-574),
  `_get_sigma_tau_kernel` + `_build_W_t_q` (:577-620), `precompile_sigma` (:623-672).
- These five move **as a unit**: `_get_sigma_tau_kernel` calls `_get_sigma_kij_kernel` (:591),
  and `precompile_sigma` must hit the *same* cache dicts as the runtime path or the AOT
  compile at `ppm_pipeline.py:350-351` silently stops pre-warming (perf bug, not wrong answer).

Imports: `w_isdf` (compilation cache), `greens_function_kernel.build_G_tau`,
`wavefunction_bundle` FFT specs, `common.fft_helpers`. This becomes the *only* Σ_PPM file
where SPMD/sharding/HLO expertise is required — the reduce-scatter layout doc, the deferred
scan/collective-flush notes (:463-475) all live here.

**`gw/ppm_accumulators.py`** — "what happens to σ^τ after the kernel returns". Moves:

- `_AccumMode` + `_select_accum_mode` (:78-128).
- The ω-projection pair, **now adjacent**: `_combine_coeff_with_sigma_tau` (:361-393),
  `_project_tau_onto_omega` (:396-424), `_project_tau_onto_omega_np` (:923-945). The
  must-stay-in-sync jax/np mirror becomes 60 contiguous lines.
- `_SigmaAccumulator` protocol (:897-920), `_HostOmegaAccumulator` (:948-1026),
  `_StreamedH5Accumulator` (:1029-1079).
- **Cohesion move (step 4, not a pure move)**: the stream-h5 dataset setup currently inlined
  in the driver (`compute_sigma_c_ppm_omega_grid` :1543-1560) plus the
  `_accumulate_kij_stream` writer closure (:1563-1569) become
  `open_sigma_kij_stream(path, omega_req, nk, nb, omega_batch) -> (writer, close)` here. After
  this, *every* byte of streamed-Σ storage logic is in one file — which matters because **Bug B
  lives at this seam** (see §5 interaction).

Imports: `ppm_windows` (for the `_SigmaWindow` type hint read in `begin_window` — no cycle,
windows is a leaf), h5py, numpy, jnp.

**`gw/ppm_sigma.py` (slimmed driver, ~600 L)** — keeps stages S0/S1 as its prologue plus the
loop nest. Stays:

- `PPMBuildResult` / `SigmaOmegaResult` (:131-148), `fit_ppm` (:679-736),
  `_SigmaPhysicsState` + `_prepare_sigma_state` (:293-358) — S0/S1 are the driver's own
  setup, 125 L, not worth a file; and the invalid-pole gate (:1465-1483 validation +
  `keep_invalid` wiring :1483-1493) reads as one story only if the string-validation and the
  jit that consumes it are in the same file.
- `minimax_tau_integrate_sigma` (:1169-1224) and `_integrate_tau_windows_for_branch`
  (:1227-1289) — the τ loop is *orchestration* (it binds window × kernel × accumulator and
  owns the Python-loop-not-scan decision :1180-1183), not a kernel; it stays with the driver.
- `_run_sigma_branch` (:1292-1390), `compute_sigma_c_ppm_omega_grid` (:1397-1631),
  `_to_host_np` / `_to_host_scalar` (:237-251).

Post-split the driver reads as the 8-stage teleology verbatim: `fit_ppm` → `_prepare_sigma_state`
→ `_iter_branches` (imported) → per branch `_build_windows_for_branch` (imported) → τ loop with
`_get_sigma_tau_kernel` (imported) feeding an accumulator (imported) → assemble
`SigmaOmegaResult`. That is the "main() as physics outline" standard.

### 2.3 Import graph (acyclic, one direction: driver → stages → engine)

```
gw_jax ──► ppm_pipeline ──► ppm_sigma (driver) ──► ppm_windows ──► minimax_screening
                │                   ├────────────► ppm_tau_kernel ──► w_isdf, greens_function_kernel
                │                   └────────────► ppm_accumulators ──► ppm_windows (type only)
                └──► head_correction        (unchanged)
w_isdf ──► minimax_screening                (unchanged — the shared minimax ENGINE stays put)
```

What stays shared and untouched: `minimax_screening.py` remains the engine (nodes, tables,
`fit_gn_ppm_from_wc_pair`, `solve_*`); `MinimaxNodes` remains its type; `minimax_config.py`
remains the two config dataclasses; `w_isdf.build_*_quadrature` untouched. **Do not** merge
`fit_ppm` into `minimax_screening` — `fit_ppm` is sharding+logging glue over
`fit_gn_ppm_from_wc_pair` (:706-712), i.e. driver-side, and the engine must stay
mesh-agnostic.

One dedup flagged, not bundled: `_to_host_np` is duplicated at `minimax_screening.py:32-38`
(no `tiled` arg) vs `ppm_sigma.py:237-245`. Candidate for a `common/` host-gather helper —
separate commit, or skip this round (it changes the engine file, out of scope).

### 2.4 Judge: does this reduce load or scatter it?

Reduces it, by three measurable criteria:

1. **Expertise locality.** Post-split, the skill set per file is uniform: `ppm_windows` =
   host numpy + minimax interval algebra (unit-testable without a GPU — window edges,
   `T = ω_max + edge·ξ` thresholds :1124-1130, the crossing ξ-scaling :852-853);
   `ppm_tau_kernel` = SPMD/shard_map/FFT; `ppm_accumulators` = D2H pipelining + h5;
   driver = physics sequencing. Today all four skill sets are needed to *navigate* the one file.
2. **Change locality against the known roadmap.** Every pending work item touches exactly one
   file: invalid-mode `static_limit` (Wc0 retention) → driver + `PPMBuildResult`; Bug B → 
   `ppm_accumulators` (+8 L in `ppm_pipeline`); the deferred collective-flush SlabIO
   accumulator (:1035-1044) → `ppm_accumulators`; m-chunking/τ-scan experiments (:463-470) →
   `ppm_tau_kernel`; 2B config collapse → driver signature only (§5).
3. **No new abstraction is introduced** — zero new classes, zero wrappers, the existing
   `_SigmaAccumulator` protocol and `_SigmaWindow`/`_SigmaBranch` carriers are simply
   relocated. This is a pure compartmentalization, compliant with the no-new-API-layers rule.

Rejected granularities:

- **8 files (one per stage)**: S0 (58 L), S1 (66 L), S2 (50 L) files would be stubs; the
  sign-convention chain S2→S3→S5-projection would spread over 3 files *plus* glue; churn with
  no locality gain. Over-split.
- **2 files (host/device)**: leaves a 900-L "host" file where windows, accumulators, stream-h5
  and driver still interleave — the accumulator seam is where both the pending bug and the two
  deferred perf projects live; it earns its own file.
- **Subpackage `gw/ppm/`**: fights the flat-module convention of `gw/` for zero benefit at 4
  files.

## 3. head_correction.py (815 L, 3 head channels): mostly leave it alone

Verdict: `head_correction.py` is **already a correct module boundary** — it is exactly the
"parallel scalar track" of the map, one topic (q→0 head) at one altitude, with clearly-labeled
internal sections (resolution :87-273, PPM fits :280-477, static COHSEX :484-608, dynamic head
Σ :611-711, rank-1 injection :714-815). Its 4 consumers import *disjoint* symbol sets
(`gw_jax.py:53-58` resolver+static; `ppm_pipeline.py:68-73,128` fits+dynamic;
`cohsex_sigma.py:26` static-to-kij; `bse_io.py:502,816` rank-1) — that's evidence the internal
sections are real seams, but none of the consumers suffers from the others' presence.

The one split that would carry its weight, offered as **optional / low priority**:

- `gw/head_sources.py` (~250 L): `HeadSample`, `resolve_head_override`, `resolve_head_sample`,
  `HeadResolver`, `format_head_sample_diagnostics` (:33-41, :87-273). This is the only part
  of the file doing **file IO and config resolution** (imports `EPSReader`,
  `chi_from_dipole`, `vcoul.compute_q0_averages` :130-176), with try/except fallback-chain
  control flow alien to the rest of the file (pure closed-form math on 4 floats). The math
  remainder (~560 L) keeps the `head_correction.py` name and all fit/injection functions.

Do it only if a later round touches the resolver anyway (e.g. bispinor screened-W adding head
channels per the roadmap). Do **not** split the three math channels (fits / static / rank-1)
apart: they share the `HeadGNParams`/`w1 = wcoul0 − vc0` vocabulary and are each <150 L.

`ppm_pipeline.py` (409 L): correct size and role (the S6/S7 sequencer). One hygiene item:
`sc_iteration.py:664` imports the *private* `_write_sigma_omega_h5` (and stubs its argument
with a `SimpleNamespace` :673) — rename to public `write_sigma_omega_outputs` and give
`sigma_kij_h5_path` a `None` default so the stub dies. 5-line change, fold into step 4.

## 4. Migration path + keeping the gates green

Ground rule: **pure moves first** — steps 1–3 are cut/paste + import edits, zero logic diffs,
one commit each, each independently gate-checked. Branch: `agent/sigma-ppm-split` (do not mix
into `agent/memplanner-cleanup`).

| Step | Commit | Content | Verification |
|------|--------|---------|--------------|
| 0 | — | Baseline capture: run the gnppm golden gate on the pre-split tree; stash `eqp*.dat` / `sigma_c_kij` for bit-compare | 3 golden gates green |
| 1 | pure move | Create `ppm_accumulators.py` (§2.2 manifest, *without* the stream-h5 setup — that's step 4); `ppm_sigma` imports from it | `pytest -q` + 3 golden gates; eqp bit-identical to step-0 baseline |
| 2 | pure move | Create `ppm_windows.py`; move S2+S3 + `_materialize_window_mask_B` | same |
| 3 | pure move | Create `ppm_tau_kernel.py`; move the 5-symbol kernel unit; update `ppm_pipeline.py:36-40` to import `precompile_sigma` from it **in the same commit** (no re-export shim left behind) | same, **plus** check the timing report still shows nonzero `sigma.compile` (AOT cache identity, §6 risk 2) |
| 4 | cohesion | (a) `open_sigma_kij_stream` extracted from driver :1543-1569 into `ppm_accumulators`; (b) sign-convention doc consolidation into `ppm_windows` header; (c) `_write_sigma_omega_h5` → public rename + `sc_iteration.py:664-675` cleanup | gates + one **stream-mode** run (`omega_accumulation kij_stream` on the MoS2 gate fixture) since (a) touches the streamed path the gates don't exercise |
| 5 | checkpoint | Report + CHANGELOG per `skills/checkpoint/SKILL.md` | — |

Why bit-identical is achievable and worth asserting for steps 1–3: the driver deliberately
preserves reduction/traversal ordering (`per_half` comment :1599-1601), and a pure move cannot
reorder branch traversal, τ order, or accumulate order. Any eqp diff at steps 1–3 means the
move wasn't pure — revert, don't rationalize. (Gates are 1-GPU by policy — no 16-GPU gating —
so before merging the branch, run one multi-GPU MoS2 smoke to cover the reduce-scatter path
`ppm_tau_kernel` now owns; pure moves can't change its math, but the import-identity risks in
§6 are multi-process-visible.)

Total: 5 commits, each self-contained, ~1 session including gate wall-time.

## 5. Interaction with the other three lenses

- **Config-seam lens (2B, `PPMSigmaRuntimeOptions` collapse).** Strong synergy, one ordering
  constraint. The getattr grab-bag sits in the driver (`ppm_sigma.py:1435-1441`) and stays in
  the driver file under my split — so if the split lands first, the 2B collapse touches
  exactly one Σ-side file (driver signature: `ppm_options` → `config.ppm` + `config.debug` +
  explicit `omega_grid_ry`) plus `gw_driver_helpers.py` (delete :16-34, :230-269) plus the
  three `ppm_options` consumers in `ppm_pipeline` (:141, :201, :263-276) and
  `sc_iteration.py:668`. The ω-grid derivation (`gw_driver_helpers.py:244-250`) needs a new
  home — I propose a plain function in `ppm_pipeline` (the sequencer is its only consumer).
  **Serialize, don't parallelize**: either order works, but concurrent edits collide on the
  driver tail. My preference: split first (pure moves rebase trivially under them, not over).
- **τ-kernel sign/mask readability lens.** My step-4(b) consolidation gives that lens a single
  target file (`ppm_windows.py`) for the branch/sign narrative; the residual 2-file spread is
  `_combine_coeff_with_sigma_tau` landing in `ppm_accumulators` (it's the projector's
  consumer-side half). If that lens wants to rename `_SigmaBranch` fields or restructure
  `kernel_sign/scale` into something more self-evident, that's a value-level change — it should
  land **after** the pure moves so the diff is legible, and it conflicts with nothing else here.
- **Bugs/physics lens (Bug B; `static_limit` mode).** Bug B's fix location *changes* under my
  split, deliberately: after step 4(a) the stream dataset handle and writer live in
  `ppm_accumulators.open_sigma_kij_stream`, so "inject the analytic head into the stream" is a
  natural `add_dense(head_kij)` capability on that seam, called from
  `ppm_pipeline._inject_analytic_head` (:109-127) instead of its current early-return. If the
  bug lens wants to fix B *before* the split lands, fixing it in-place at
  `ppm_pipeline.py:126-127` (rank-0 h5 read-modify-write of the head) is fine — the move then
  carries the fix. `static_limit` (NotImplementedError :1471-1476) needs `Wc0` retained through
  `fit_ppm` → `PPMBuildResult` (:131-140) — pure driver-file work, no conflict.
- **Shared caution for all**: the map's line numbers are pre-2A; ground any patch against
  `3cad3dd`, not the map (§ code-state note at top).

## 6. Risks

1. **Two module instances of the kernel caches.** The caches are module-level dicts keyed on
   `id(mesh_xy)` (:283-284, :529, :586). If any call site imports the new module by a different
   path (`gw.ppm_tau_kernel` vs relative `.ppm_tau_kernel` resolving through a stale
   `sys.path` entry), Python creates two module objects → two caches → double compile and,
   worse, `precompile_sigma` warming the wrong one. Mitigation: relative imports only (repo
   convention), and step 3's `sigma.compile`-timing check.
2. **AOT prewarm silently decoupled.** Same mechanism, observable consequence: gates stay
   green but the first τ dispatch pays full compile inside `sigma.exec`. The step-3
   verification line exists for exactly this; it's the only pure-move failure mode that isn't
   caught by bit-compare.
3. **"Pure move" scope creep.** The tempting micro-fixes en route (the `_to_host_np` dedup
   §2.3; `getattr` defaults; docstring rewrites) each turn a bit-identical diff into a
   judgment diff. Discipline: steps 1–3 diffs must show `ppm_sigma.py` shrinking by exactly
   the moved line count; everything else is step 4+ or another round.
4. **Streamed path under-gated.** Steps 1–3 don't touch it and are safe; step 4(a) does, and
   `KIJ_STREAM` is exercised by no golden gate (`_select_accum_mode` :95-128 falls back to
   HOST multi-process, and the gates' grids are small → `auto` picks HOST). Hence the explicit
   stream-mode run in step 4's verification — and this same fixture is what the Bug B fix
   needs anyway, so build it once, in coordination with the bugs lens.
5. **Import cycle via type hints.** `ppm_accumulators` → `ppm_windows` for `_SigmaWindow` is
   safe today (windows is a leaf). Keep it that way: if a future window builder ever wants an
   accumulator-side constant, pass it as an argument, don't import backward.

## 7. Summary of recommended actions (this round)

1. Adopt the 4-file split (§2.1-2.3): `ppm_windows` / `ppm_tau_kernel` / `ppm_accumulators` /
   slim `ppm_sigma` driver. 5 commits, pure moves first, gates + bit-compare per step (§4).
2. Consolidate the sign-convention narrative into `ppm_windows`'s header (step 4b) — the cheap
   majority of the readability complaint, zero behavior change.
3. Leave `head_correction.py` intact this round; optional `head_sources.py` IO split only if a
   later round touches the resolver (§3).
4. Sequence: this split → 2B config collapse (driver-file-only after the split) → Bug B fix at
   the now-consolidated accumulator seam.
