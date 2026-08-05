# Consensus — Σ_PPM tighten (quadrature API · module split · config seam · kernel legibility)

Synthesised from four Round-1 proposals and four Round-1 responses
(`round1_agent_{1..4}_*.md`, 2026-07-04/06). Ground truth for every line ref:
`sources/lorrax_D` @ `3cad3dd` (`ppm_sigma.py` **1631 L** — the map's 1702 is
pre-2A; Bug A is FIXED at `head_correction.py:320-338`; `ppm_invalid_mode` is
wired for `zero`/`2ry` at `ppm_sigma.py:1465-1483`; Bug B is still live at
`ppm_pipeline.py:126-127`, re-verified this session). This document is the
team's position for the lead to finalize.

## 1. TL;DR

The four lenses converge with unusually little residue. All four agree:

- **The 8-stage teleology is good bones; nobody proposes restructuring the
  physics.** The rot is (a) five concerns in one 1631-L file, (b) a drifted
  config mirror consumed via `getattr(..., default)` — the same silent-default
  layering that produced the invalid_mode bug (`ppm_sigma.py:1435-1441`, incl.
  a *fourth* copy of the regularization default as the magic literal
  `0.018374661087827496`), (c) a sign story told in 7 places with the ±ω prose
  **wrong-as-stated** (the `Σc(−ω)=−Σc(ω)*` claim at :29-30/:211-213 is a
  false *global* theorem; the true identity is per pole term — Agent 4's single
  best catch), and (d) ~70%-duplicated accumulators with a jax/numpy projector
  pair kept in sync by comment discipline alone (:361-424 vs :923-945).
- **The load-bearing consensus rule**: *no behavior-adjacent Σ_PPM change lands
  before G1 (kij vs kij_stream parity, expected RED today) and G2 (per-branch
  reference tiles) exist.* G1's absence is *why* Bug B survived — nothing ever
  compared the two accumulator paths. Grep-verified 0-caller engine deletes are
  exempt (pytest + golden gates suffice).
- **One signature commit** (Agent 1's R2, adopted by all): the
  `compute_sigma_c_ppm_omega_grid` contract changes exactly once, owned by
  Agent 3, carrying Agent 1's merged-`MinimaxConfig` required-`quad` and
  satisfying Agent 4's no-getattr demand. `debug_cfg` is OUT (settled 3-1,
  A4 conceded — zero `config.debug` reads survive below `gw_jax`).
- **Order**: gates → Bug B in-place → the signature/seam commit → the 4-file
  pure-move split → content changes (K2 accumulators, K0 ledger, quadrature
  unification) in the new small files → physics tail. Seam-before-split and
  split-before-K2 were both contested in Round 1 and both resolved by
  concession in the responses.
- **The don't-force-merge list is part of the consensus** (Agent 1 §5,
  unchallenged): the two τ-loop mechanics (chi0 `lax.scan`-in-jit vs Σ's
  Python loop — a monolithic scan regressed MoS2 3×3 ~80%,
  `ppm_sigma.py:1180-1183`), interval derivation per side, node kinds, the two
  accuracy-knob families (`screening.minimax_*` vs `ppm.sigma_*` — one *class*
  after the merge, two *instances*), and the three energy-reference *values*.

Net effect: ~−300 L of dead/duplicated code, one config path instead of four
default layers, a 1631-L file becoming four single-concern files, and the two
remaining live bugs (B streamed-head-drop, C shard-0 accumulator suspect)
closed under new gates.

## 2. The agreed end-state

### 2a. Unified quadrature API (Agent 1, amended by 2/3/4)

`minimax_screening.py` becomes **exactly the engine**: cached solvers
(`solve_laplace_minimax_interval:560`, `..._imag_interval:618`,
`solve_phase_minimax_bandwidth:662`), the two quadrature value objects, the
`MinimaxNodes` currency, and **one** two-point pole fit. Specifically:

- **Delete the dead vestigial layer** (~200 L, all grep-zero-caller):
  `MinimaxWindowPair` (:258-287), `EnergyWindow` (:243-254),
  `build_imag_freq_minimax_window_pair` (:775-835),
  `extract_gn_ppm_parameters(_from_Wc)` (:838-935 — includes a duplicate
  host-side W-solve). `build_static_minimax_window_pair`'s body moves into its
  only caller, `w_isdf.build_static_quadrature` (:446) — interval derivation
  lives with the physics on both sides, symmetric with Σ's
  `_build_*_sigma_windows`.
- **One config class, two instances**: `MinimaxConfig` and
  `SigmaQuadratureConfig` (`minimax_config.py:8-34`) merge into one frozen
  `MinimaxConfig`; both `gw_config` properties (:737, :748) keep their names
  and knob sources but return the one class. The Σ defaults exist at **one**
  site (today: three — dataclass, `gw_config.py:751-757`, and the
  `sigma_window_quad=None` fallback at `ppm_sigma.py:1424-1433`, which is
  deleted by making `quad` a required argument).
- **All `MinimaxNodes` born in one accessor**:
  `to_minimax_nodes(time_axis=..., alpha_scale=..., exp_fold=..., t_scale=...)`
  replaces the hand-rolled construction in `compute_chi0`
  (`w_isdf.py:597-605`) and the manual crossing ξ-rescale
  (`ppm_sigma.py:851-853`). Bit-identical by construction (same ops, same
  order); the MoS2 chi0 regression hash (`w_isdf.py:170-173`) is the gate.
  The per-window `α_eff = α·exp(−i·E_ref_sum·t)` fold at :1210-1212 stays
  where it is (per-window, post-axis-cast — moving it changes the locked path
  for zero gain).
- **One `two_point_pole_fit`**: extract the shared Ω²/B algebra;
  `fit_gn_ppm_from_wc_pair` (:408) = it + fallback policy + tally;
  `head_correction.fit_head_ppm` (:280) = the 1-element case + its **named**
  continuation policy + the `B_h = 2Ω·B_tensor` normalization stated in one
  comment. Bug-A-class defects become structurally impossible — one place
  decides the sign of B. Gated by the full sub-case grid
  ({GN imag probe, HL real probe} × {Ω²>0, Ω²<0, denom≈0}), per the
  audit-failure-modes discipline — no "matches at ULP" hand-waving.
- **Shipped-table honor holes closed**: `build_real_quadrature`
  (`w_isdf.py:521,532`) passes `use_shipped_tables` through (today
  `regenerate_minimax_tables=true` silently doesn't regenerate the HL ±branch
  quads); the imag solver gains the same keyword + a docstring note that no
  shipped family exists.

### 2b. Module layout (Agent 2, manifests amended per responses)

Four files, three seams the code already documents about itself; flat `ppm_*`
naming, no subpackage; **acyclic** import graph
(driver → stages → engine):

```
gw/ppm_windows.py       ~360 L  S2+S3  _SigmaWindow/_SigmaBranch/_iter_branches,
                                       _build_{single,three}_sigma_windows,
                                       _build_windows_for_branch, mask_B interp.
                                       Leaf; imports only minimax_screening.
                                       GPU-free unit-testable. Home of the
                                       branch-story sign ledger + _BRANCH_TABLE.
gw/ppm_tau_kernel.py    ~260 L  S4     the 5-symbol kernel unit (caches :283-284,
                                       _make_project_ri_reduce_scatter :427,
                                       _get_sigma_{kij,tau}_kernel, precompile_sigma)
                                       moved AS A UNIT (shared cache dicts — AOT
                                       prewarm at ppm_pipeline.py:350 must keep
                                       hitting the same dicts). The only file
                                       needing SPMD/HLO expertise.
gw/ppm_accumulators.py  ~330 L  S5     _AccumMode/_select_accum_mode, the
                        (→~250 post-K2) projector(s), the accumulator(s).
                                       K2's rewrite happens INSIDE this file.
gw/ppm_sigma.py         ~600 L  S0,S1, fit_ppm, _prepare_sigma_state (the
                        driver         invalid-pole gate), the τ loop
                                       (orchestration: binds window × kernel ×
                                       accumulator, owns Python-loop-not-scan),
                                       _run_sigma_branch, the driver. Reads as
                                       the 8-stage teleology verbatim.
gw/ppm_pipeline.py      unchanged role (S6+S7 sequencer)
gw/head_correction.py   intact — already a correct module boundary; optional
                        head_sources.py IO split DEFERRED until a round touches
                        the resolver.
```

Settled placement questions: the τ loop stays in the driver (A1's co-location
request withdrawn/rejected on import-graph grounds — merging it into
`ppm_windows` would make the leaf import kernel caches and the accumulator
protocol); only the driver may import `PPMConfig` — **`ppm_windows`,
`ppm_tau_kernel`, `ppm_accumulators` must not import from `gw_config`**
(explicit rule in the split PR). Rejected granularities stand: not 8 files
(stubs + scattered sign chain), not 2 (a 900-L rump), not a subpackage.

### 2c. Config contract (Agent 3 §3a, adopted verbatim by all)

> `config.ppm` is the only home for Σ_PPM scalar knobs. Values are validated
> once, at `PPMConfig.__post_init__` (normalization at the parse site).
> Derived values (ω-grid) are `LorraxConfig` properties with exactly one
> formula (the live builder's length-stable formula; Ry derived from eV —
> the dead, numerically-different arange properties at `gw_config.py:758-775`
> are replaced). Resolved values (paths) are resolved at the driver seam where
> `input_dir` lives and passed as explicit args. The kernel reads scalars by
> direct attribute access — **`getattr(x, 'field', default)` on config-like
> objects is banned in `gw/`**. No object may copy a `config` field it does
> not itself derive or resolve.

`PPMSigmaRuntimeOptions` + `build_ppm_sigma_runtime_options`
(`gw_driver_helpers.py:16-34, 230-269`) are deleted (census: 6/15 fields dead
mirrors, 7/15 verbatim relays, only 2 genuinely resolved). The one signature,
written once:

```python
compute_sigma_c_ppm_omega_grid(
    wfns, ppm, meta, mesh_xy, *,
    ppm_cfg: PPMConfig,              # validated frozen scalars — the ONLY config object
    quad: MinimaxConfig,             # REQUIRED (merged class; :1424-1433 fallback deleted)
    omega_grid_ry: np.ndarray,       # derived data (config.omega_grid_ry, evaluated once)
    sigma_kij_h5_path: str | None,   # resolved data; "" normalized to None at the seam
    print_fn=print,
) -> SigmaOmegaResult
```

No `debug_cfg` (zero reads below `gw_jax.py:817/:920`). Value-vs-capability
split for `invalid_mode`: values validate in `__post_init__`; capability
gating (`static_limit`/`infinity` → `NotImplementedError` until Wc0 retention
lands) stays in the kernel at :1470-1482. When `static_limit` is implemented,
`Wc0` goes on `PPMBuildResult` — a **data-seam** change, never a config field.
The `sc_iteration.py:664-675` `SimpleNamespace` mirror-stub dies in the same
commit (public `write_sigma_omega_outputs` rename rides along).

### 2d. Kernel legibility + the one accumulator (Agent 4, homes per A2's split)

- **K0 (zero math change)**: `_DenomKind` enum replaces the triple-overloaded
  `kernel_sign` int (ω-kernel sign at :780 / crossing-vs-single dispatch at
  :1124 / x_max extension at :763-766); `_BRANCH_TABLE` module constant;
  project codes become an IntEnum whose value **derives from the node kind**
  (`CrossingMinimaxQuadrature` → IMAG) instead of free strings at :854/:868;
  `inv_sqrt_nk` at :540 renamed to carry its hidden global minus sign; the
  E_ref factorization identity commented at the α_eff build (:1212).
- **The sign ledger is partitioned by who decides the sign** (A4's C3,
  accepted by A1 and A2): branch story (corrected per-term ±ω derivation,
  `_BRANCH_TABLE`, window prefactors, the `T = ω_max + edge·ξ` sentence) in the
  `ppm_windows.py` header; the three alpha-fold conventions + `time_axis`
  semantics in the `MinimaxNodes` docstring; the E_ref identity at the driver's
  fold site; the −1/√N_k row in `ppm_tau_kernel`. One-line cross-pointers, no
  restating — a single mega-doc would recreate the told-three-times problem.
- **K2 (behavior-changing, the riskiest single item)**: one `_TauAccumulator`
  (async-D2H deque verbatim from :993-1015) + two sinks (`_MemoryTileSink` ≡
  today's host tail; `_H5Sink` replacing `_StreamedH5Accumulator`). Deletes
  the jitted projector pair `_project_tau_onto_omega` /
  `_combine_coeff_with_sigma_tau` (:361-424; only runtime caller is
  `_StreamedH5Accumulator.add_tau:1068` — re-verified) — the numpy projector
  becomes the single source of truth. Streamed-path D2H drops n_ω-fold and h5
  RMW drops ~n_τ-fold (the perf asymmetry at :1563-1569 nobody had written
  down). Carries the **Bug C** fix (shard-0-only `addressable_data(0)` at
  :997-998 vs `make_array_from_process_local_data` at :1024-1026 — wrong or
  crashing on single-process multi-GPU; 20-minute 2-GPU repro decides which,
  fix or loud assert either way — this matters because the code ships to
  arbitrary device counts, per the no-16-GPU-gating rule). The τ loop and
  `_SigmaAccumulator` protocol seam (:897-920) are untouched.

## 3. Points still contested + recommended resolution

Almost everything was settled by concession in the response round. The record,
plus the one live item:

| # | Question | Positions | Resolution |
|---|----------|-----------|------------|
| 1 | Seam vs split order | A2 wanted split-first; A1/A3/A4 seam-first | **SETTLED: seam → split** (A2 conceded). The winning reason is A1's R2 one-signature-change rule — the split then cuts along the *final* contract — NOT A3's original stated reason (the split never moves the getattr block; A4's correction, accepted). |
| 2 | Split vs K2 order | A4 wanted K2-first ("moves code twice"); A2 split-first | **SETTLED: split → K2** (A4 conceded). Move-count is symmetric; the criterion is where the riskiest diff is reviewed — a ~330-L `ppm_accumulators.py`, not the monolith. Cost accepted: the 64-L jax projector is relocated then deleted; K2 lands the same session so the corpse doesn't linger. |
| 3 | `debug_cfg` in the signature | A4 wanted it; A1/A2/A3 no | **SETTLED: no** (A4 conceded; grep confirms zero `config.debug` reads in `ppm_sigma.py`). If a debug read ever appears below the driver, the parameter is added in that commit, next to its first reader. |
| 4 | A2 step 4(a) `open_sigma_kij_stream` extraction | A2 proposed; A3/A4 flagged as churn | **SETTLED: struck** — K2's `_H5Sink` subsumes it; extracting code the next scheduled PR deletes is two review passes over the same 60 lines. |
| 5 | Co-locate window-build + τ-integrate | A1 requested; A2 rejected | **SETTLED: rejected, A1 withdrew.** τ loop is orchestration and stays in the driver; the shared-convention worry is answered by the partitioned ledger. |
| 6 | **Bug B timing** (the one live disagreement) | A3+A4: immediately after G1, in-place at `ppm_pipeline.py:126-127`; A1+A2 tables place it after the seam commit | **RECOMMEND: immediately after G1** (Workstream 1 below). It is a live physics wrong-answer (streamed Σc silently head-less, with **three** mis-handling consumers: the h5 itself, the at-DFT eval at ppm_pipeline.py:209-220, the undocumented QP-solve skip at gw_jax.py:628); `ppm_pipeline.py` is untouched by every other stream until the seam commit, so merge risk is zero, and the seam commit simply rebases the fix's two new explicit args (`sigma_kij_h5_path`, `meta`). A2 already granted "the move carries the fix"; A1's dependency edge was preference, not requirement. Fix shape: rank-0 ω-batched h5 RMW add + `head_injected` idempotence attr; hard error if streaming with no h5 path. The "forbid streaming" alternative is **rejected** — large-ω-grid streamed runs are exactly where the ω-dependent head matters. |
| 7 | Head-vs-body invalid-policy skew (body drops Ω²<0 poles; head continues through with a different formula) | surfaced by A1 §2.4 | **Parked as an open physics question** — not part of any refactor commit. §3.3's unification makes the two policies visible as named arguments; the decision then goes to the BGW-parity track with the `static_limit`/Wc0 work. |
| 8 | `sigma_freq_debug_file` cwd-vs-input_dir | A3 raised | **Parked** in the physics-tail bucket; decide when someone uses the flag. |

## 4. THE DIVISION OF TASKS

Consensus ground rule: **no behavior-adjacent Σ_PPM change lands before
G1 + G2 exist** (grep-zero-caller engine deletes exempt). All gates run on the
MoS2 1-GPU fixture class (no-16-GPU-gating rule); before merging the split
branch, one multi-GPU MoS2 smoke covers the reduce-scatter path. Every phase
ends pytest-green + 3-golden-gates-green. WS0–WS3 are strictly serial on the
Σ-side files; ⟂ marks parallel-safe work.

| WS | Content | Owner | Kind | Files | Depends on | Gate |
|----|---------|-------|------|-------|-----------|------|
| **0** | **Gates first**: G1 kij↔kij_stream parity (expect RED at head), G2 per-branch/window reference `.npz` (all 4 branches × 3 windows non-empty, asserted), G3 head negative-branch regression (GREEN — pins the Bug-A fix; `tests/test_head_correction.py` has zero negative-branch coverage today) | A4 | tests only | tests/ | — | G3 green; G1 red-documented |
| **0⟂** | Engine delete-pass: the ~200-L `MinimaxWindowPair` dead layer + dead imports (`ppm_sigma.py:68,71,74`) + static-interval body into `w_isdf.build_static_quadrature` | A1 | pure delete | minimax_screening, w_isdf, 3 L ppm_sigma | — | pytest + golden gates + grep-zero |
| **1** | **Bug B fix in-place** (head → stream h5, idempotence attr, hard-error on pathless streaming; +1 sentence documenting the gw_jax:628 streamed no-QP-solve fallthrough) | A4 | behavior (physics fix) | ppm_pipeline only (~30 L) | 0 | **G1 RED→GREEN** |
| **2** | **The one signature commit series**: `PPMConfig.__post_init__` + parse-site normalization; single ω-grid property; A1's merged `MinimaxConfig` + required `quad`; getattr block :1435-1441 → direct reads; `""`→`None`; mirror + builder + `PPMOutputs.ppm_options` deleted; `sc_iteration` stub kill + writer rename; `sigma_dispatch.py:227,246-247` / `gw_jax.py:451-455` → `config.omega_grid_*` | A3 (A1 reviews the quad type; A3 owns both `__post_init__`s with a stated validation boundary) | pure plumbing | gw_config, minimax_config, gw_driver_helpers, ppm_pipeline, ppm_sigma driver tail, sigma_dispatch, gw_jax, sc_iteration | 0, 1 | `sigma_mnk.h5` **bit-identical**; G1 re-run; grep-zero on `PPMSigmaRuntimeOptions`; `hash(config.ppm)` doesn't raise |
| **3** | **The 4-file split, pure moves only** (A2 steps 1-3; 4(a) struck, 4(b)→WS5, 4(c) rode WS2). Discipline: each diff shows `ppm_sigma.py` shrinking by exactly the moved line count; any eqp diff = impure move, revert don't rationalize | A2 | pure move | new ppm_windows / ppm_tau_kernel / ppm_accumulators; slim ppm_sigma | 0⟂, 2 | G2 bit-identical per move; **AOT-prewarm timing check** (nonzero `sigma.compile`, none inside `sigma.exec`) |
| **4** | **K2 accumulator unification**: `_TauAccumulator` + 2 sinks; delete jax projector pair; Bug-C repro → fix or loud assert; per-ω-half injection killing the `per_half` tuple-key dict (:1599-1611) — micro-commit if it muddies the diff | A4 | **behavior** (streamed data movement redesign) | ppm_accumulators (+driver call sites :1359-1375) | 3 | G1 + G2 + golden; `sigma.exec` ±3%; 2-GPU single-process check |
| **5** | **K0 ledger**: `_BRANCH_TABLE`, `_DenomKind`, corrected ±ω derivation replacing :29-30/:211-213, window spec-table flatten of :815-869, IntEnum project derived from node kind, `neg_inv_sqrt_nk` rename, E_ref comment; ledger partitioned per §2d | A4 (+A2 doc merge) | renames + comments, zero math | ppm_windows, ppm_tau_kernel, driver comments | 4 (documents the post-K2 single projector) | G2 bit-identical; **no retrace** |
| **5⟂** | Quadrature content: node accessor (chi0 fold + crossing ξ-rescale, now targeting w_isdf + ppm_windows); `two_point_pole_fit` unification with the sub-case grid; shipped-table holes; (optional, drop-if-over-budget) reference-resolver share | A1 | behavior-neutral, numerically gated | minimax_config, minimax_screening, w_isdf, ppm_windows, head_correction | 2, 3; grid test + G3 before the pole-fit step | chi0 locked hash; sub-case grid; G3; eqp bit-diff |
| **6** | **Physics tail**, each its own gated commit, flags default off: `static_limit` (+`Wc0` on `PPMBuildResult`, analytic −½·Wc0 term) with a new invalid-pole fixture + BGW parity; `sigma_at_dft_energies` wiring at gw_jax:649 (same pattern as its sibling at :673); head-vs-body invalid-policy decision (BGW parity); `sigma_freq_debug_file` path decision | A3 + physics lens | **behavior** | driver, gw_jax, fit_ppm | 2 | new fixtures + BGW parity; golden gates unchanged with flags off |

WS4, WS5, WS5⟂ are parallelizable across branches (disjoint files after WS3);
K2 should land in the same session as WS3 so the relocated-then-doomed jax
projector doesn't linger. Branch discipline per repo rules: feature branches
(`agent/sigma-ppm-*`), never `main`; checkpoint every ~5 commits.

**Pure-move vs behavior-changing, at a glance**: WS0⟂/WS2/WS3 must be
bit-identical (h5/eqp diff = the gate); WS5/WS5⟂ are value-identical with
numeric gates (hash, sub-case grid); WS1/WS4/WS6 change behavior and each has
a dedicated red→green or parity gate.

## 5. The single highest-leverage first step

**Build G1 — the kij vs kij_stream accumulator-parity gate (with G2/G3 riding
the same commit), before touching any source.** One small MoS2 fixture run
twice (`omega_accumulation=kij` vs `kij_stream` + `sigma_kij_h5_file`), assert
`sigma_c_kij_ry` equal to ~1e-12 and eqp consistent. It is the keystone of the
whole program: (a) it does not exist today, and its absence is the *proven*
mechanism by which Bug B survived — `KIJ_STREAM` is exercised by no golden
gate (`_select_accum_mode:95-128` falls back to HOST on the gates' small
grids); (b) it is the acceptance test for WS1 (RED→GREEN is the Bug-B fix's
definition of done); (c) it gates WS2's `""`→`None` normalization, WS4's
entire streamed-path redesign, and every later stream-touching change; (d) it
is tests-only, needs no allocation beyond 1 GPU, and unblocks A1's 0⟂
delete-pass to proceed in parallel the same day. Everything else in §4 queues
behind it by the consensus rule — so it is also literally the critical path.

## 6. Pointers

- `round1_agent_1_quadrature_api.md` — engine inventory (what's already
  shared), the 5 duplication classes, the don't-force-merge list (§5).
- `round1_agent_2_module_split.md` — the 4-file manifests with exact line
  ranges, the acyclic import graph, rejected granularities, kernel-cache /
  AOT-prewarm risk (§6).
- `round1_agent_3_config_seam.md` — the 15-field mirror census, Option-1
  verdict + the 3a contract, the two-ω-grid-formulas finding (§1b),
  jit-friendliness requirements (§4).
- `round1_agent_4_kernel_legibility.md` — the 7-row sign ledger, the false
  ±ω-theorem catch (§2a), K2 design + the streamed-path perf asymmetry, Bug B
  blast radius (3 consumers) + fix, Bug C, the G1/G2/G3 gate definitions.
- `round1_agent_{1..4}_response.md` — the concession record backing every
  "SETTLED" row in §3.

---

## 7. LEAD FINALIZATION (2026-07-04)

Division **accepted** as written. Adjustments/notes:
- **WS0 (G1/G2/G3) is building now** on `agent/memplanner-cleanup` — the keystone,
  approved. G1 is expected RED (strict-xfail, documents Bug B); G2/G3 green.
- **WS0⟂ (engine delete-pass) follows WS0** (sequential, not parallel, to avoid two
  agents racing the same branch). Grep-zero re-verified before each delete (the
  parsed-but-unread lesson: verify machinery-absence, not just zero-callers).
- **One open priority call — static_limit timing.** Consensus parks it in WS6 (last);
  the lead flagged it may jump to right-after-WS2 (its only real dependency is the
  config seam) because it's the user's preferred *default*. Pending the user's call;
  does not change WS0/WS0⟂/WS1.
- Branch: staying on `agent/memplanner-cleanup` (continuity with the Σ_PPM cleanup
  already there) rather than a fresh `agent/sigma-ppm-*`, unless the program grows
  large enough to warrant its own branch.

## 8. LEAD DIRECTIVE — WS5 sign approach (supersedes §2d K0 / §4 WS5)

User directive (2026-07-04): **no named sign variables.** Eliminate
`_SigmaBranch.kernel_sign` and `.scale` entirely — do NOT replace them with a
`_DenomKind` enum + `_BRANCH_TABLE` (that just renames the abstraction). Instead the
four physical branches (cond/val × ±ω) are written as explicit cases with the
physically-correct sign baked directly into the denominator / prefactor where the
physics puts it. The only signs in the code are the ones that make it physically
correct, inline, mapped to the physics in `docs/docs_gwjax` — not carried as ±1
integers through a tuple.

- Rationale: sign-carrying variables read as leftover debug hooks (the class of
  `ppm_sigma_flip_neg`/`ppm_sigma_scale`, already deleted in 2A). Options/conventions
  must map to the physics docs, not to a folkloric ±1 ledger.
- Behavior-preserving: the current signs are physically CORRECT (gates match the
  BGW-anchored refs); they are just laundered through opaque `kernel_sign`/`scale`
  and mis-documented (the false global `Σc(−ω)=−Σc(ω)*` claim). WS5 makes them
  legible + the prose correct — bit-identical, G2 + golden gates as the pins.
- The word "ledger" is retired. WS5 = "inline the correct branch signs + fix the ±ω
  prose + doc-map"; nothing is catalogued because there are no toggles.
- Still-true precondition: lands after WS3 (the split), so the inlined signs go into
  the clean ppm_windows/ppm_tau_kernel, not the monolith.
