# GW refactor — STATUS ledger (updated 2026-07-09)

Single place to see where the refactor stands. Overlays progress onto MAP.md's
attack order (§6) and target list (§4). Read `HANDOFF.md` first, then this, then
the sub-reports.

## 2026-07-08 overnight (most recent — see MORNING_SUMMARY.md)

Four programs landed after this ledger's 2026-07-06 body was written: the
**device-invariance bug fix** (padded-μ root cause, NOT accumulation order) +
**padding consolidation** (−553 L), the **qp_solver toggle** (one_shot_dft
default), and **ppm_invalid_mode complete with static_limit as the new default**
(BGW three-way parity PASS). Suite: 257 passed / 9 skipped / 0 failed. Details:
`MORNING_SUMMARY.md` (this dir); evidence:
`reports/device_invariance_2026-07-08/`,
`reports/bgw_invalid_mode_refs_2026-07-08/`, `G0W0_SC_TOGGLE_DESIGN.md`. The
sections below predate this and are accurate through 2026-07-06.

## Branches / what's on origin

| Branch | State | Contents |
|---|---|---|
| **`agent/memplanner-cleanup`** | HEAD `bb95bc3` (local; earlier line `dd03ce6` pushed to origin) | gate-0 (crash fix + Eo + re-freeze) → GN-PPM 1-GPU fix + gate → parser single-source → vcoul strip → **aot_memory_model delete (~8.7k L)** → memory-model.md docs → gflat docstring → gflat Phase 2 → **planner self-check fix** → **CONVENTIONS** → **delete-pass (5 modules)** → isdf move + `isdf/` core split → load_wfns delete + reader de-physics → **memory-model REDESIGN (one ISDF planner, `4c833e4`)** → **bispinor e2e gate (`3fc93b4`)** → **Σ_PPM tightening program WS0–WS4 (`fddf8c0`…`bb95bc3`)** |

**Landed 2026-07-06 (three programs, see dedicated sections below):** (1) the
memory-model **redesign** collapsed the planner to ONE ISDF model and deleted the
legacy band/r `compute_optimal_chunks` (closes NEXT_TARGETS #5); (2) the **first
bispinor e2e regression gate** (closes NEXT_TARGETS #3); (3) the **Σ_PPM tightening
program** (SIGMA_PPM_MAP → 4-agent consensus → WS0–WS4 executed) — 3 physics bugs
fixed, the `PPMSigmaRuntimeOptions` config mirror deleted, all named ±1 sign-vars
removed, the 1631-L `ppm_sigma.py` monolith split into 4 single-concern files, 12
gates green. This begins attack-order **step 5 (cross-cutting)**: the streamed-Σc
head seam and the config-mirror seam are now closed.

**Branches reconciled 2026-07-02:** `gw-delete-pass` (delete-pass + CONVENTIONS)
was re-done fresh onto the main stack — the original commit had accidentally
committed 13 binary profile-trace junk files, so it was NOT cherry-picked; the
deletes (grep-verified zero real importers) + CONVENTIONS were re-applied clean.
Subsumed branches `gate-0-qpwfn`, `gnppm-1gpu-mask-fix`, `gw-delete-pass` deleted.
Everything is now on the one pushed branch. Nothing is on `main`.

## Attack order status (MAP.md §6)

| Step | Status | Notes |
|---|---|---|
| **0. Gate-first** | ✅ DONE + gate #4 added | e2e gate was RED (write_qp_wfn_h5 crash + stale W-drift ref); fixed crash, re-froze reference, added an `Eo` column. **COHSEX + GN-PPM regression gates green**, plus **gate #4 `test_ibz_full_bz_equivalence`** (`1479162`) — IBZ vs `FORCE_FULL_BZ` on MoS2, exact to ~1e-9 (COHSEX); the only gate that catches a wrong symmetry unfold. Surfaced + fixed a `FORCE_FULL_BZ` drift bug on the tile V_q path (`0d7ba06`). Gate #2 (3D COHSEX) still missing. |
| **1. Delete pass** | ✅ DONE + r-space V_q subsystem CONSOLIDATED | 5 verified-dead modules; **+ deleted the ~3k-line r-space V_q "tile" subsystem** (`8369ecc`+`d4bb3ba`, MAP §4 #3): v_q_tile.py, legacy compute_V_q_bispinor_to_h5, make_v_munu_chunked_kernel, compute_bare_coulomb_sphere_idx, the r-space tail of compute_all_V_q. Live survivor `_unfold_g0_ibz_to_full`→v_q_g_flat. **Then a birds-eye audit (workflow) + finish-cleanup (`6809e64`): removed a dead FFT-era kernel trio (~135 L) the first pass missed, stripped dead params, swept stale docs.** **BISPINOR VERIFIED e2e** (real MoS2 bispinor run, 4 ζ / 7 tiles, bit-identical to pre-deletion ref). V_q(G) 2×|G| cutoff feature verified structurally (guarded, unit-tested). Single G-flat V_q path now. 3 gates + 241 unit green. |
| **2. Single-source duplicates** | 🟡 load_wfns DONE, zeta DEFERRED | ✅ cohsex.in parser ×2→1 (reconciled sys_dim 3→2, ecutrho→WFN ecutwfc). ✅ gw/vcoul.py shim stripped 234→72. ✅ eqp0/eqp1/Z math — found ALREADY single-sourced (stale map claim). ✅ **`load_wfns` migration DONE (bb04399)** — `common/load_wfns.py` facade DELETED, 5 helpers → `common/wfn_transforms`; B3 σ·p lift single-sourced to `bispinor_init.lift_to_4spinor`, B4 ψ-unfold single-sourced to `symmetry_maps.trs_augment_U`/`tau_phase_row`; eager `coeffs[:]` slurp now lazy; `get_umklapp_vector` public. Wave-1 reader boilerplate single-sourced (94cc354; `bind_mf_attrs`/`kpt_starts` → `file_io/mf_header`). 🟡 **`zeta_loader`/`zeta_reader` merge DEFERRED** — genuine divergence (loader adds `auto/eager/phdf5` backend selector; reader owns `valid_mu` trailing-μ zero-fill pad), loader already delegates into `zeta_reader._do_disk_to_G`; safe merge needs a **padded-μ fixture** (→ bispinor value gate). |
| **3. Extract smeared concerns** | ✅ memory planner DONE (redesign landed); IBZ symmetry mostly a non-target | **Memory planner (C6): FULLY single-homed** — the earlier pass deleted the aot_memory_model model; the **2026-07-06 redesign (`4c833e4`) collapsed to ONE ISDF planner** = persistent floor + max-over-stage-transients + rank floor, **deleted the legacy `compute_optimal_chunks` band/r model** (closes NEXT_TARGETS #5), fixed the **centroid ÷√P sharding bug**, made the util **ns²-aware** (bispinor). Validated no perf/accuracy regression on 4- and 16-GPU (bit-identical where chunk count matches). Docs: `SHARDING_RULES.md`, `MEMORY_MODEL_DESIGN.md` (this dir), `MEMORY_MODEL_VALIDATION.md` (`reports/memory_model_refit_2026-07-03/`). Phase 3 (V_q consolidation) **CANCELLED** — two different live kernels, not redundant. **IBZ symmetry (B4): the truly-redundant part done, the rest is NOT a consolidation target** (`466040a`) — 4 byte-identical `_rotate_psi` variants collapsed to 2; but the "≥6 parallel unfold helpers" (MAP §4) are conceptually-parallel, operationally-distinct: scalar `unfold_v_q` (device shard_map+all_to_all, μ/ν perm), transverse `_unfold_v_q_ij_ibz_to_full` (rank-2 R_cart tensor), zeta `_unfold_q_full_bz` (jit+out_shardings, r/μ perm), `unfold_psi` (host numpy spinor+phase+umklapp+TRS). A unified sym-action over these = over-abstraction hiding real differences (sharding, tensor rank, host-vs-device), NOT recommended. See finding below. |
| **4. Stage-by-stage move** | 🟡 isdf DONE + SPLIT | ✅ `isdf_fitting.py` (A2 ζ-fit stage) moved common/→gw/ (c9fb0e2 move + 62ce45e cleanup), no circular import; **then SPLIT (dfb6b90): core primitives extracted to a new standalone `src/isdf/core.py` (1733 L, 7 public primitives + 6 jit caches), leaving `gw/isdf_fitting.py` (1030 L) as the thin GW orchestrator (`fit_zeta_to_h5` + stage-coupled mem probes).** Byte-identical move confirmed by body-diff (15 fns, 0 differences); core imports only stdlib/jax/np + `common.*` → no isdf→gw cycle. First BGW-anchored 3D gate added (d55c4cb). ✅ load_wfns A1 ingest DONE (see step 2); 🟡 wfn_loader mostly de-physics'd. Remaining: zeta reader merge (deferred, needs padded-μ fixture). |
| **5. Cross-cutting last** | 🟡 STARTED via the Σ_PPM program | **Config seam DONE** (`f607f6c`): the `PPMSigmaRuntimeOptions` mirror is deleted, `compute_sigma_c_ppm_omega_grid` takes one signature (`config.ppm` + derived ω-grid, no getattr defaults). **Streamed-Σc head seam DONE** (`099bcd5`, Bug B): the q→0 analytic head is now injected into the streamed Σc h5. Still open: `sigma_at_dft_energies` authoritative-QP wiring (gw_jax:649) and the `LORRAX_SC_*` shadow env-flag surface → `LorraxConfig`. |

## Deferred / cancelled (with reasons)

- **Phase 3 V_q consolidation — CANCELLED.** gflat Peak E models the G-flat path;
  `_choose_v_q_chunks` sizes the r-space *tile* kernel — different live kernels,
  not redundant. Merging would break tile/bispinor V_q sizing. (`memplanner_cleanup_2026-07-02/MAINTAINABILITY.md`.)
- **`load_wfns` migration — ✅ DONE (bb04399).** Facade deleted, 5 helpers rehomed
  to `wfn_transforms`, physics (B3 lift, B4 ψ-unfold) single-sourced out of the
  reader. See step 2.
- **`zeta_loader`/`zeta_reader` merge — DEFERRED (genuine interdependency, not
  papered over).** The loader is a superset wrapper (backend selector) that already
  delegates into `zeta_reader._do_disk_to_G`; the reader owns the `valid_mu`
  trailing-μ zero-fill pad. A safe merge needs a **padded-μ fixture** (genuine
  backend + valid_μ divergence) — the same fixture the bispinor value gate wants.
  Confirmed a real divergence by the correctness audit, not a hidden breakage.
- **MoS2 2D sub-meV BGW agreement — parked.** Native COHSEX tops out ~62 meV;
  it's a 2D-head/static-CH-partition research problem, not a config fix. Lead:
  read `whead` directly from BGW `DEBUG HEAD TERMS`. (`archive/GATE_AUDIT.md §7-11`.)

## Open findings / gaps (not blocking, worth a future pass)

- ~~**Device-invariance robustness**~~ ✅ **FIXED 2026-07-08** — and the root
  cause written here at filing time (chunk-order-sensitive Σc round-off) was
  **wrong**: it was the PADDED μ extent being P-dependent. See the 2026-07-08
  overnight section above, NEXT_TARGETS TIER 0★ B, and
  `reports/device_invariance_2026-07-08/`. Residual: the on-pole census
  robustness item (needs a user physics decision).

- ~~**Broken planner self-check**~~ ✅ FIXED (`ee6c447`): now reads JAX
  `peak_bytes_in_use` (faithful `22.26/28 GB, 79%` under BFC, matches mem_probe),
  never `--id=0`; under cuda_async (which hides the peak) the line self-flags
  `[cuda_async under-reports — rerun with XLA_PYTHON_CLIENT_ALLOCATOR=default]`
  instead of printing a misleading 3%.
- **Planner budget covers only ζ-fit/V_q**, not a ~18 GB upstream transient →
  whole-run peak floors ~18 GB regardless of budget. Narrower than the name.
- **Underived planner coefficients** (`factor_D=2.0`, `pair_density_slots 3/4`):
  empirical, no maintainable derivation; the "derive + HLO-pin each coefficient"
  pass is the deeper memory-model maintainability gap. (`MAINTAINABILITY.md`.)
- **Two intermingled V_q drivers** (`v_q_g_flat` vs `v_q_tile`) — pick one before
  any real V_q-sizing consolidation.
- **A/B/C modulefiles** still carry the purged-`$SCRATCH` FFI defaults (lorrax_D
  fixed). ~~Config parser inline-`#`~~ ✅ FIXED (61ae4b8).
- 212 confirmed-dead + 74 suspected-bug items from the audit remain (`archive/DEAD_CODE.md`)
  — only 5 modules + the aot package deleted so far.

## New `isdf/` mini-library (2026-07-03, dfb6b90)

The A2 ζ-fit core is now a **standalone reusable library at `src/isdf/core.py`**
(1733 L, 7 public primitives — `c_q_from_psi_sm`, `factor_c_q`, `fit_one_rchunk`,
… — + 6 module jit caches), imported by the thin `gw/isdf_fitting.py` orchestrator.
`isdf/core.py` depends only on stdlib/jax/np + `common.*` (no `gw`/`file_io`/
`centroid`), so **no isdf→gw import cycle** and the library is consumable by a
second client — direct-BSE is the intended second consumer. The move was verified
byte-identical (15 fns, 0 body differences vs pre-split rev 62ce45e). `common/
bispinor_init.py` is reclassified **NO-LONGER-DEAD** — it is now the production
single-source home for the B3 σ·p lift (`lift_to_4spinor` + `HALFALPHA`).

## Gate scaffolding + small fixes (2026-07-03)

"Red means red" now holds (commit 25067f4): the 4 container-JAX env failures
(reshard jit-form, aot_memory libcufft probe) are CONDITIONALLY skip-marked (auto-run
when the env is fixed), the 3 golden gates are wired into skills/checkpoint/SKILL.md.
**New `si_cohsex_3d` gate (d55c4cb)** — first BGW-anchored + first `sys_dim=3` e2e
gate (Si 4×4×4, atol 1e-3 eV, native Coulomb body + BGW head scalars); flips the
entire 3D column of the coverage matrix from synthetic to E2E. Full suite now =
**242 passed / 24 skipped / 0 failed**. Two verified small fixes landed (61ae4b8):
parser inline-# stripping + wfn_writer nspinor=1 factor-of-2 occupation.

## Correctness audit (2026-07-03) — branch COHERENT

Fresh-context audit of the isdf split + load_wfns deletion: **all imports resolve,
no cycle, byte-identical core move confirmed by body-diff, deleted facade has no
dangling importer, single-sourced helpers each have exactly one def with importing
adopters, and the deferred zeta merge is a real interdependency — not a hidden
breakage.** Only cleanup item is a batch of **LOW-severity stale source-attribution
comments/docstrings** (no code impact): heaviest in `gw/gflat_memory_model.py`
(`:265,296` cite deleted `load_wfns.py`; `:404,657` cite out-of-range
`isdf_fitting.py` line numbers now in `isdf/core.py`), plus `common/psi_G_store.py:20-21`,
`common/symmetry_maps.py:124`, `common/wfn_transforms.py:922` (refs
`_make_pair_pipeline_sm`, likely pre-existing staleness), and stale
`SOURCES.txt`/`PKG-INFO` generated artifacts. A comment-refresh pass would tidy
these; none are load-bearing.

## Memory-model redesign (2026-07-06, `4c833e4`)

The C6 planner was collapsed to **ONE ISDF planner**: `persistent floor +
max-over-stage-transients + rank floor`. The legacy `compute_optimal_chunks`
band/r model (a second, parallel planner) was **deleted** — this is exactly
NEXT_TARGETS #5, now done. Three substantive fixes rode along: the **centroid
÷√P** sharding bug (per-device centroid count was under-divided), an **ns²-aware**
utilization estimate (bispinor's 4-channel ζ cost), and the unified floor/transient
accounting. **Validated end-to-end on 4- and 16-GPU** — no perf or accuracy
regression, bit-identical where the chunk count matches. Design/validation docs:
`SHARDING_RULES.md` + `MEMORY_MODEL_DESIGN.md` (this dir);
`MEMORY_MODEL_VALIDATION.md` under `reports/memory_model_refit_2026-07-03/`.

## Σ_PPM tightening program (2026-07-06, `fddf8c0`…`bb95bc3`)

The session's largest program: SIGMA_PPM_MAP.md (8-stage flow + cleanup ledger) →
a 4-agent consensus (`reports/sigma_ppm_tighten_2026-07-04/consensus_draft.md`) →
executed as workstreams WS0–WS4 on this branch. Net: **3 physics bugs fixed, the
config mirror deleted, all named ±1 sign-vars removed, the 1631-L `ppm_sigma.py`
monolith split into 4 single-concern files, 12 gates green.**

| Commit | WS | What |
|---|---|---|
| `fddf8c0` | 2A | dead-code delete (SIGMA_PPM_MAP §2A) — delete-only hygiene |
| `3cad3dd` | 2C/2D | wired `ppm_invalid_mode` (`zero`/`2ry`); **fixed Bug A** (head sign-flip on Ω²<0) |
| `08cee1a` | WS0 | keystone gates G1 (kij↔stream parity) / G2 (per-branch/window tiles) / G3 (head negative-branch) + h5py fillvalue fix |
| `abfcc5f` | WS0⟂ | engine delete-pass on `minimax_screening` (**−231 L**, no behavior change) |
| `099bcd5` | WS1 | **fixed Bug B** — inject the analytic q→0 head into the streamed Σc h5 (4.13 eV head-drop) |
| `f607f6c` | WS2 | config-seam collapse: killed the `PPMSigmaRuntimeOptions` mirror, one signature |
| `92fead6` | WS3 | split the 1631-L monolith → `ppm_sigma`/`ppm_windows`/`ppm_tau_kernel`/`ppm_accumulators` (pure moves) |
| `c3c721c` | WS5 (lead directive) | eliminated `kernel_sign`/`scale` ±1 fields; signs inlined at the physics; `S=E_A+Ω` convention |
| `bb95bc3` | WS4 | unified accumulators to one `_TauAccumulator` + one ω-projector (deleted the jax/numpy mirror); **fixed Bug C** (shard-0-only assumption, now correct on arbitrary device counts); G1 tightened `1e-8 → 5e-12`; **~38% streamed `sigma.exec` speedup** (12.8s → 8.0s) |

**Bugs fixed:** A (head sign-flip on negative-Ω² fit), B (streamed head-drop), C
(shard-0-only accumulator). **Structural:** config mirror gone (one signature, no
getattr defaults), no ±1 sign-vars (signs inline, mapped to `docs/docs_gwjax`),
one accumulator + one projector. **Remaining Σ_PPM work → WS6 physics tail** (see
NEXT_TARGETS): `static_limit` invalid-pole default (+ analytic −½·Wc0 term),
`sigma_at_dft_energies` authoritative-QP wiring.

## Meta-lesson (recurring — heed before the next "consolidate N helpers" task)

Three times now the MAP's "N parallel X = redundant, merge them" premise was
**wrong on inspection**: (1) the two "V_q models" (Phase 3) were different live
kernels; (2) the "≥6 parallel unfold helpers" are conceptually-parallel but
operationally-distinct; only (3) the aot_memory_model package and the 4
`_rotate_psi` variants were *actually* redundant and safe to collapse. The map
catalogues by **surface shape** (same verb: "unfold", "model", "rotate"); real
redundancy needs **operational identity** (same object, same math, same
device/sharding). Rule going forward: before any consolidation, diff the cores —
byte-identical / value-identical → merge; conceptually-similar-but-different →
leave it, a forced unification is over-abstraction. The safe wins were the
byte-identical ones; every "these look parallel" case that turned out distinct
would have broken physics if merged.

## Sub-report index

- `MAP.md` — dataflow spine, concern×stage matrix, ranked refactor targets (§4), attack order (§6).
- `SIGMA_PPM_MAP.md` — the Σ_PPM 8-stage flow + cleanup ledger (Part 2); driver of the WS0–WS6 program. Consensus/execution plan: `reports/sigma_ppm_tighten_2026-07-04/consensus_draft.md`.
- `MEMORY_MODEL_DESIGN.md` + `SHARDING_RULES.md` — the one-ISDF-planner redesign; validation at `reports/memory_model_refit_2026-07-03/MEMORY_MODEL_VALIDATION.md`.
- `archive/FEATURES.md` — exhaustive feature catalog. `archive/FLAGS.md` — 128 input flags.
- `archive/DEAD_CODE.md` — 212 confirmed-dead / 185 refactor-target / 74 suspected-bug verdicts.
- `archive/GATE_AUDIT.md` — gate coverage + the gate-0 / MoS2-exact-agreement saga (§1-11).
- `archive/BUGS_FOUND.md` — this-session bug ledger (gate-0, cusolvermp, GN-PPM 1-GPU, planner self-check…).
- `memplanner_cleanup_2026-07-02/` — `PLAN.md` (attack plan) + `MAINTAINABILITY.md` (physics + honest gaps + Phase-3 correction) + `BUDGET_VALIDATION.md` (the end-to-end proof).
- `archive/READER_CLEANUP_PLAN.md` — the A1/B4 reader un-smear plan; step 2 (load_wfns delete + helper rehome) DONE, step 4 (zeta_loader/zeta_reader merge) now UNBLOCKED (padded-μ gate landed 2026-07-08: `tests/test_mu_pad_invariance.py`). New standalone ζ-fit library lives at `src/isdf/core.py`.
