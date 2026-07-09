# GW refactor — RANKED remaining targets (synthesized 2026-07-02, updated 2026-07-09)

> **2026-07-09 update.** The 2026-07-08 overnight program landed (device-invariance
> fix, padding consolidation, qp_solver toggle, static_limit default) — see
> `HANDOFF.md` for the current entry point and `MORNING_SUMMARY.md` for the record.

Synthesis of MAP §4/§6, STATUS, B4_ATTACK, GATE_AUDIT, FLAGS, DEAD_CODE,
VQ_CONSOLIDATION_AUDIT (source audits now under `archive/`) after the
`agent/memplanner-cleanup` line landed. Read STATUS.md first for what is
already done. This doc is the *what's left*, ranked by
(value × safety/leverage), with an honest "not worth it" section at the end.

Legend for gate prereq: **Gated-now** = an existing green gate would catch a
wrong refactor; **Needs gate** = must add coverage first or move blind.

Attack-order note (from MAP §6): stage moves (steps 4–5) are only safe once each
touched stage has a gate a wrong-but-plausible refactor would fail. So the gate
items below (#1, #2) are deliberately ranked above the big stage moves even though
they are smaller — they are the linchpins.

---

## TIER 0★ — current top priorities (updated 2026-07-09)

### ★ Driver transparency revision (B → C) — NEW TOP ITEM (2026-07-09)
- **What:** make `gw_jax.main()` read as the twelve-line physics scaffold. Spec:
  **IDEAL_SCAFFOLD_VS_LORRAX.md §5**, recommendation "B then C, each gated".
  - **Phase B — pure moves, ~350 L out of main():** the 90-line IBZ slice/unfold
    block into a `solve_w` wrapper; restart flush into `persist_restart`; the
    ~100-line freq-debug table into `gw_output`; degeneracy-averaging into a
    helper. Zero seam changes; existing e2e gates (incl. IBZ-equivalence) pin
    bit-identity.
  - **Phase C — the real unification:** one-shot main() consumes the SC path's
    `screening_requests_for` + `compute_sigma_xc` dispatch; `fit_ppm` lifts to
    top level; the three QP branches collapse into `solve_qp(...)`. This is
    *deletion of a duplicate pipeline* (the no-redundancy rule), not new
    architecture.
- **Gates:** Phase B = bit-identical under the existing e2e suite; Phase C =
  SC-iteration-1 ≡ one-shot on a fixture + a basis-rotation regression (the
  documented silent-breakage risk in the QP block).
- **Don't-list (physics integrity):** don't materialize G(t); don't promote the
  head to a stage; don't put W on an ω-grid.
- **Size:** Medium-large (two gated phases). **Gate prereq:** Gated-now for B;
  C needs the SC≡one-shot fixture gate first.

### A. ✅ DONE (2026-07-08, `fdf89c2` impl + `9925d43` default flip) — Σ_PPM WS6 physics tail
_`static_limit` invalid-pole mode LANDED and **is now the DEFAULT** (= BGW mode 3
= BGW's default). Implementation keeps **BOTH** the static SEX and CH terms
(Σ_static = sigma_sx(G_occ, Wc0·inv_mask) + sigma_coh(Wc0·inv_mask), reusing the
two cohsex_sigma kernels verbatim; `Wc0` retained on `PPMBuildResult`). **Three-way
BGW parity PASS** on Si (zero/2ry/static_limit; refs + mode-diff tables in
`reports/bgw_invalid_mode_refs_2026-07-08/`; the earlier research note's "SX pole
→ 0" claim was WRONG — see SIGMA_PPM_MAP §2C). gnppm golden re-frozen (mean 40 meV
/ max 0.322 eV, in-family with the validated 2ry−zero delta); statics, Si-3D,
bispinor, IBZ gates unmoved. `sigma_at_dft_energies` wiring was separately DONE
2026-07-08 as the `qp_solver` toggle (G0W0_SC_TOGGLE_DESIGN.md,
`archive/g0w0_sc_toggle_impl/`)._
- **Only remaining sub-item:** **`project_code` → node-derived cleanup** (small) —
  derive the project code from the node kind instead of the free-string path
  (SIGMA_PPM_MAP WS6 leftover).

### B. ✅ FIXED (2026-07-08, `083d209`..`7801d46` + consolidation `6c850bd`..`620b501`) — Device-invariance bug
_FOUND and FIXED. The **correct root cause** was **NOT** the accumulation-order
hypothesis this ticket was filed with (measured chunk-order effects ≤6e-5 eV vs
the 3.8 eV signature) — it was the **PADDED μ extent being P-dependent**: pad
rows appear only at some device counts (1204→1216 at P=16; transverse 668→672),
and (1) the transverse ζ LU solved at padded extent, (2) the PPM mode
census/window stats ran over padded μ². Fix: **solves + census on the LOGICAL
extent**. Bispinor 4g↔16g eqp: **2.535 eV → 1.45e-7 eV**. Padding then
consolidated (−553 L: logical-extent restarts → P-portable, census masking at
fit birth, one `round_up` / one `solve_at_logical`, dead PadAxis API deleted).
New gates: pad-flip bit-identity (in suite, 1 GPU:
`tests/test_mu_pad_invariance.py`), cross-P script (P=1 vs P=4), restart
pad-roundtrip. Full story: `reports/device_invariance_2026-07-08/`._
- **Remaining residual (OPEN — needs a physics decision from the user):** Fix-3
  on-pole census robustness — on-pole charge bands remain P-sensitive via
  near-threshold PPM mode flips amplified by GN-PPM ill-posedness. Options:
  census hysteresis / census at a fixed reference / document as ill-posed.
  See `reports/device_invariance_2026-07-08/ROOT_CAUSE.md` AS-FIXED §.

---

## TIER 0 — gate/safety scaffolding (do first; cheap, unblocks everything)

### 1. ✅ DONE (2026-07-03, commit 25067f4) — Wire the 3 green gates into the checkpoint skill + skip-mark the container-env red tests
- **What:** The three green 1-GPU gates (`test_gw_jax_regression[cohsex]`,
  `[gnppm]`, `test_ibz_full_bz_equivalence`) are not referenced by
  `skills/checkpoint/SKILL.md`, so "run pytest" still goes red on two
  container-JAX env failures (`tests/test_reshard_all_to_all.py` — `jax.jit`
  kwargs form; `tests/test_aot_memory.py` — libcufft probe) that are *not* real
  regressions. "Red means red" is currently false.
- **Files:** `skills/checkpoint/SKILL.md`, `tests/test_reshard_all_to_all.py`,
  `tests/test_aot_memory.py` (add `pytest.mark.skipif` on JAX version / missing
  libcufft), `pyproject.toml` (jax pin vs container 0.5.3 mismatch).
- **Why:** Without this, every downstream move lands against a red suite and the
  gate discipline the whole refactor depends on is unenforceable.
- **Size:** Small (~1–2 h). **Gate prereq:** none — this *is* the gate wiring.

### 2. ✅ DONE (2026-07-03, commit d55c4cb) — 3D BGW-anchored COHSEX gate (Si 4×4×4)
- **What:** Every value gate today runs on a *single 2D MoS₂ IBZ/charge fixture*.
  The 3D branch of `compute_vcoul` and the `sys_dim=3` head/wing treatment have
  **zero e2e coverage**. Freeze a downsized Si (e.g. 4×4×4, few bands) COHSEX eqp
  reference — BGW-anchorable at ~0.12 meV (unlike 2D's ~62 meV ceiling), so it is
  a real physics gate, not a self-consistency freeze.
- **Files:** new `tests/regression/` fixture + parametrization in
  `tests/test_gw_jax_regression.py`; seed run under `runs/Si/`.
- **Why:** Unblocks all 3D / truncation-off stage work (Coulomb, head/wing,
  `w_isdf`); highest remaining gate value. Must stay ≤1 GPU (never 16).
- **Size:** Medium (compute + freeze). **Gate prereq:** N/A (it is the gate).

### 3. ✅ DONE (2026-07-06, commit `3fc93b4`) — First bispinor Σ_X (+Σ^B) e2e regression gate
_Closes the "bispinor was verified-once-by-hand but never gated" gap. Adds a 1-GPU
(`LORRAX_NGPU=1`, `memory_per_device_gb=30`) MoS2 3×3 nspinor=2 gate: screened-charge
COHSEX (Σ_SX+Σ_COH on W⁰⁰) + bare Breit Σ^B, exercising the 4 ζ-channel / 7 V_q-tile
/ transverse-γ̃ machinery the scalar gates never touch (Σ^B folds into the sigSX
column). Fixture `tests/regression/bispinor_debug/` = full WFN truncated 82→34 bands,
**verified bit-identical** to the full-WFN run. **atol=1e-6 pure freeze** (deterministic
run) — note this is a self-consistency freeze, not a BGW/covariance parity pin, so the
`bispinor_tt_noncovariance` full-BZ-direct-transverse concern is NOT yet gated. Suite:
243 passed / 24 skipped / 0 failed (regression subset now 5 gates)._

---

## TIER 1 — high-value structural moves (the real refactor)

### 4. ✅ DONE + SPLIT (2026-07-03; c9fb0e2 move + 62ce45e cleanup + dfb6b90 split) — Move `common/isdf_fitting.py` → `gw/`; then extract core to a standalone `isdf/` library
_STEP 1 pure move (10 rel-imports + 6 importers, no cycle) + STEP 2 (deleted dead _mem_report, docstringed 5 caches). **STEP 3 (dfb6b90): the ζ-fit core primitives were extracted to a NEW standalone `src/isdf/core.py` (1733 L, 7 public primitives + 6 jit caches), leaving `gw/isdf_fitting.py` (1030 L) as the thin GW orchestrator** (`fit_zeta_to_h5` + stage-coupled mem probes stayed). Byte-identical move confirmed by body-diff (15 fns, 0 differences); `isdf/core.py` imports only stdlib/jax/np + `common.*` → no isdf→gw cycle, so the library is reusable (direct-BSE the intended 2nd consumer). DECLINED the mem_probe extraction — it shares _nvsmi with the stage-coupled _track_peak, so lifting it would split a cohesive cluster. 4 gates green (incl. `isdf.core` import test)._
- **What:** An entire pipeline stage (ζ-fit) lives in `common/` with embedded
  memory probes, 6 hand-keyed jit caches, and C3 FFI solves. MAP §4 #2 — the
  single biggest un-started stage move. Move file to `gw/`, lift the memory
  probes into the C6 planner, name the jit caches.
- **Files:** `common/isdf_fitting.py` → `gw/isdf_fitting.py`; update importers
  (`gw/gw_init.py`, `gw/v_q_g_flat.py`, others via grep).
- **Why:** Largest smeared file; A2 is otherwise homeless in `common/`. Clarifies
  the stage boundary the whole MAP is organized around.
- **Size:** Large. **Gate prereq:** Gated-now for charge/IBZ/GN-PPM (gates #1
  suite). ζ-fit feeds every gate, so a value-breaking move fails today. Safe to
  start; keep it a pure move (no logic change) first, extract caches second.

### 5. ✅ DONE (2026-07-06, commit `4c833e4`) — Memory-model redesign: one ISDF planner
_The C6 planner is now fully single-homed. Collapsed to **ONE ISDF planner** =
persistent floor + max-over-stage-transients + rank floor; **deleted the legacy
`compute_optimal_chunks` band/r model** (the second parallel planner this item
targeted). Also fixed the **centroid ÷√P** sharding bug and made the util
**ns²-aware** (bispinor 4-channel ζ). Validated no perf/accuracy regression on 4-
and 16-GPU, bit-identical where the chunk count matches. Docs: `SHARDING_RULES.md`
+ `MEMORY_MODEL_DESIGN.md` (this dir), `MEMORY_MODEL_VALIDATION.md`
(`reports/memory_model_refit_2026-07-03/`). The 4- vs 16-GPU eqp divergence this
validation surfaced is now tracked as **TIER 0★ item B** above._

### 6. ✅ DONE (2026-07-03, bb04399 + 94cc354) — A1 ingest cleanup: delete `common/load_wfns.py`, de-physics `file_io/wfn_loader.py`
_MAP §4 #10 + #4. `common/load_wfns.py` facade DELETED — 5 helpers rehomed to
`common/wfn_transforms`; **B3 σ·p lift single-sourced to `bispinor_init.lift_to_4spinor`**
(+`HALFALPHA`), **B4 ψ-unfold single-sourced to `symmetry_maps.trs_augment_U`/
`tau_phase_row`**, the eager `coeffs[:]` host slurp made **lazy** (latent OOM
closed), `get_umklapp_vector` made public. Wave-1 reader boilerplate single-sourced
(94cc354; `bind_mf_attrs`/`kpt_starts` → `file_io/mf_header`). Correctness audit:
no dangling importer, each helper has exactly one def with importing adopters.
`wfn_loader.py` now mostly de-physics'd (🟡 residual cleanup only). No bispinor-lift
gate (#3) was needed — the lift is a pure symbol move to `bispinor_init`, unit-safe._

### 7. 🟢 UNBLOCKED (2026-07-08) — `zeta_loader.py` vs `zeta_reader.py` merge (the top remaining single-source target)
- **What:** MAP §4 #8. Both classes are live: `ZetaLoader` is a superset wrapper
  adding a `backend` selector (`auto`/`eager`/`phdf5`, phdf5 needs a Mesh) and
  **already delegates into `zeta_reader._do_disk_to_G`** for the G_flat path;
  `ZetaReader` owns the `valid_mu` trailing-μ zero-fill **pad** logic. The
  correctness audit confirmed this is a **genuine interdependency, not redundant
  duplication** — so a safe merge is NOT a mechanical swap.
- **Blocker LIFTED:** the padded-μ gate this merge was waiting for now EXISTS —
  **`tests/test_mu_pad_invariance.py`** (pad-flip bit-identity, 1 GPU), landed with
  the 2026-07-08 padding consolidation, plus the restart pad-roundtrip gate. The
  merge can proceed gated. ~−350 L.
- **Files:** `file_io/zeta_loader.py` (delete after merge), `file_io/zeta_reader.py`,
  `gw/v_q_g_flat.py`, `file_io/__init__.py`.
- **Why:** Last real single-source target; removes a duplicated reader + the ζ half
  of the B4 smear. (B4_ATTACK found the *worst* B4 drift, the `_resolve_ibz_q_list`
  copy, already died with the deleted r-space tile subsystem.)
- **Size:** Medium-large. **Gate prereq:** **Needs gate** (padded-μ fixture); the
  existing IBZ-equivalence gate #4 covers the *un-padded* charge ζ unfold only.

---

## TIER 2 — worthwhile, smaller / cross-cutting

### 8. ✅ MOSTLY DONE — Live-wrong-answer bug triage (NOT refactor — physics look first)
_The three highest-exposure bugs are now FIXED by the Σ_PPM program (Bugs A/B/C):_
- ✅ **Bug A — head sign-flip** (`head_correction.py`, negative-Ω² branch) — FIXED
  (`3cad3dd`; pinned green by gate G3, which had zero negative-branch coverage before).
- ✅ **Bug B — streamed-Σc no-head** (`ppm_pipeline.py`) — FIXED (`099bcd5`; the q→0
  analytic head is now injected into the streamed h5; pinned by gate G1, whose absence
  was the *proven* reason Bug B survived).
- ✅ **Bug C — shard-0-only accumulator** — FIXED (`bb95bc3`; now correct on arbitrary
  device counts).
- ✅ **Bug D — wfn_writer factor-of-2 occupation** on nspinor=1 — FIXED (`61ae4b8`),
  along with the config parser inline-`#` bug.
- **`gw/gw_jax.py:~237` latent `NameError`** — verified FALSE (clean call), not a bug.
_Nothing open here. Note the distinct 4- vs 16-GPU eqp divergence is TIER 0★ item B,
not part of this triage list._

### 9. ✅ SUPERSEDED by the Σ_PPM program (2A/2C) — Second dead-key delete pass
_The initial blanket delete (`5725aca`) was **REVERTED** (`76e9c2a`): two of the four
knobs are class-(c) **disconnected-but-wanted**, not dead. Correct resolution landed
via the Σ_PPM program: `ppm_sigma_scale`/`ppm_sigma_flip_neg` (truly dead) deleted in
the **2A** dead-pass (`fddf8c0`); **`ppm_invalid_mode` WIRED** (`3cad3dd`, zero/2ry
modes) and **`sigma_at_dft_energies` is now WS6** authoritative-QP wiring (TIER 0★ A) —
neither is dead. The `PPMSigmaRuntimeOptions` mirror is gone entirely (`f607f6c`).
Remaining truly-dead sweep below is the only leftover._
_Original notes (still partly valid for the non-PPM keys):_ **NOT dead (kept):**
`head_wing_schur.py` is staged-and-tested ('intended for promotion', has
`test_head_wing_schur.py`) — NOT abandoned; `chunk_size` live (`gw_jax:152`);
`bispinor_init.py` live (wfn_loader B3 lift, post-#6). **Still open:** `screening_method`
(left — `ScreeningConfig.method` documents a real axis); the ~8 `debug_*`/`w_copies`
DebugConfig keys (unverified — a separate pass; some may be live debug plumbing)._
- **What:** FLAGS #1 — 15 keys in `gw/gw_config.py` `_DEFAULTS` /
  `DebugConfig` / `PPMSigmaRuntimeOptions` are parsed-but-never-read (`chunk_size`,
  `screening_method`, `ppm_sigma_scale/flip_neg`, `ppm_invalid_mode`,
  `sigma_at_dft_energies`, 8 dead `debug_*`/`w_copies`). Plus DEAD_CODE
  leftovers: `common/bispinor_init.py` (test-oracle only) and
  `gw/experimental/head_wing_schur.py` (zero prod callers) still present.
- **Files:** `gw/gw_config.py`, `common/bispinor_init.py`,
  `gw/experimental/head_wing_schur.py`.
- **Why:** Pure hygiene, grep-verifiable zero real consumers; shrinks the flag
  surface the doc rewrite (#12) has to cover.
- **Size:** Small. **Gate prereq:** none (delete-only, run full suite after).

### 10. Move `w_isdf.build_*_quadrature` → the minimax engine (B1 out of A4)
- **What:** MAP §4 #9 — `build_static_quadrature` / `build_imag_quadrature` /
  `build_real_quadrature` are B1 frequency code hosted inside the A4 stage file.
- **Files:** `gw/w_isdf.py` → `gw/minimax.py` (or `minimax_screening.py`).
- **Why:** Clean B1/A4 separation; `w_isdf` and `ppm_sigma` both consume minimax
  — the builder belongs with the engine, not one consumer.
- **Size:** Small-medium. **Gate prereq:** GN-PPM gate exercises the quadrature
  consumers; a broken move fails there. **Honesty:** `w_isdf.solve_w` itself has
  no dedicated gate — keep this a pure symbol move, don't touch solve logic.

### 11. Shadow env-flag surface → `LorraxConfig` (`LORRAX_SC_*`) — **DONE 2026-07-08**
Promoted to the `SCConfig` group (`sc_max_iter` / `sc_tol_ev` / `sc_accelerator`
/ `sc_history_depth` / `sc_mixing` / `sc_dump_dir`) in the `qp_solver` toggle
commit series; the envs remain deprecated overrides (a note is printed when one
is active). SC behavior unchanged (RMS trajectories bit-identical, see
`archive/g0w0_sc_toggle_impl/`).
- **What (was):** MAP §6 step 5 / FLAGS #3 — the self-consistent loop's entire
  convergence surface (5 env-only knobs, `gw_jax.py:536-540`) lives outside
  `cohsex.in` / `LorraxConfig`. Promote to real input keys.
- **Files:** `gw/gw_jax.py`, `gw/gw_config.py`, `templates/cohsex.in`.
- **Why:** Config should be single-source; env flags bypass the frozen config and
  are invisible to reproducibility. (Note the V_Q_* env flags are already dead —
  their modules were deleted; SC_* is what remains.)
- **Size:** Small-medium. **Gate prereq:** SC path is ungated e2e — promote the
  *plumbing* (env→config) without changing SC behavior; a value change here is
  currently invisible, so keep it mechanical.

### 12. `gw_driver_helpers.py` fan-out + `get_DFT_mtxels.py` C10/driver split
- **What:** MAP §4 #6/#7. `gw/gw_driver_helpers.py` (285 L) is a grab-bag
  (config mirror `PPMSigmaRuntimeOptions` + C8 runtime init + A3 BGW-grid glue) —
  fan out to three homes. `psp/get_DFT_mtxels.py` (926 L): parser already
  single-sourced (done), but the C10 kernel library is still fused with a stale
  debug driver — split the driver out / delete it.
- **Size:** Medium. **Gate prereq:** low-risk; get_DFT_mtxels is A0 (upstream),
  exercised by fixture regen.

### 13. `docs/docs_gwjax/COHSEX_INPUT.md` rewrite (doc-only, lead with `compute_mode`)
- **What:** FLAGS #2 — doc is a full mode-axis behind: documents the legacy
  `do_screened`/`use_ppm_sigma` trio, not the canonical `compute_mode`
  (`x_only|cohsex|gn_ppm|hl_ppm`); ~36 live keys (incl. new `ecutrho`) undocumented.
- **Size:** Medium (doc). **Gate prereq:** none. Best done *after* #9 so it
  documents the pruned key set, not the dead one.

---

## NOT worth doing (explicitly de-scoped — honesty section)

- **Unify the B4 unfold KERNELS into one sym-action** (`unfold_v_q` /
  `_unfold_v_q_ij_ibz_to_full` / `_unfold_q_full_bz` / `unfold_psi`). B4_ATTACK
  Layer 3 + STATUS meta-lesson: conceptually parallel but operationally distinct
  (device shard_map vs host numpy, scalar vs rank-2 tensor, μ/ν vs r/μ perm,
  umklapp+TRS). A forced unification is a leaky/fat abstraction that would break
  physics. The map's "≥6 parallel helpers → merge" premise was wrong on
  inspection **three times**. Leave separate; only route them to Layer-1 tables
  (that's #6/#7, not a merge).
- **Phase-3 V_q "model" consolidation** — CANCELLED. The two V_q sizing models
  were different live kernels, and one (`v_q_tile`) is now deleted anyway.
- **eqp0/eqp1/Z math single-source (MAP §4 #5)** — MAP says OPEN, but STATUS
  found it **already single-sourced** (`gw_jax` imports `compute_eqp_diag` /
  `compute_z_factor_from_omega_grid` from `eqp_bgw`). Verify with a diff before
  spending time; likely a stale map claim, not real work.
- **`_resolve_ibz_q_list` single-source (B4 Layer 2)** — mostly MOOT: the drifted
  `compute_vcoul` copy lived in the now-deleted r-space tile subsystem. The live
  bispinor path already runs the gate-verified resolver. Nothing to consolidate.

---

## One-line prioritization

**Done since last revision (2026-07-08 overnight):** **TIER 0★ A** static_limit
default + BGW three-way parity (`fdf89c2`/`9925d43`), **TIER 0★ B**
device-invariance fix + padding consolidation (`083d209`..`620b501`), **#11**
LORRAX_SC_* → SCConfig (part of the `qp_solver` toggle). Earlier: **#5
memory-model redesign** (`4c833e4`), **#3 first bispinor e2e gate** (`3fc93b4`),
**the whole Σ_PPM program WS0–WS4**, #1 gate wiring, #2 3D-COHSEX gate, #4 isdf
move + `isdf/` core split, #6 load_wfns delete.

**Top-3 remaining, ranked:** (1) **★ Driver transparency revision B → C**
(IDEAL_SCAFFOLD_VS_LORRAX.md §5: Phase B ~350 L pure moves out of main(), Phase C
unify one-shot onto the SC dispatch; each phase gated). (2) **Fix-3 on-pole
census robustness** (TIER 0★ B residual — needs a user physics decision). (3)
**#7 zeta_loader/zeta_reader merge** — the last real single-source target, now
UNBLOCKED by `tests/test_mu_pad_invariance.py`. The rest (**A's `project_code`
leftover, #10, #12, #13**) are small, safe hygiene/cross-cutting cleanups. Skip
the four de-scoped items.

**Correctness note (2026-07-03 audit):** the isdf split + load_wfns delete are
COHERENT — no import cycle, byte-identical core move, no dangling importers. Only
residue is LOW-severity stale source-attribution comments/docstrings (heaviest in
`gw/gflat_memory_model.py`, plus `psi_G_store.py`, `symmetry_maps.py:124`,
`wfn_transforms.py:922` → dead `_make_pair_pipeline_sm` ref, and stale egg-info
artifacts). A comment-refresh pass is optional cleanup, not blocking.
