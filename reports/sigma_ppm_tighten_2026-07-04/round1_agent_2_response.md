# Round 1 — Agent 2 response: the split survives, but it moves to third in line

**Lens**: module decomposition (my proposal: 4-file split of `ppm_sigma.py` —
`ppm_windows` / `ppm_tau_kernel` / `ppm_accumulators` / slim driver).

**Re-verified against the tree this session** (`sources/lorrax_D` @ `3cad3dd`,
branch `agent/memplanner-cleanup`): `ppm_sigma.py` = 1631 L; the getattr grab-bag is
`:1435-1441`; **zero** `config.debug` reads anywhere in `ppm_sigma.py` (grep-clean —
relevant to A4 §6 Stage 4, see §3.3 below); `_project_tau_onto_omega`'s only runtime
caller is `_StreamedH5Accumulator.add_tau` (`:1068`) — A4's K2 deletion premise holds.

---

## 1. Where I agree (adopted without reservation)

- **A4's Stage 0 "gates before any source change"** (G1 accumulator-parity, G2
  per-branch/window reference npz, G3 head-negative-branch regression). I adopt the
  consensus rule as stated: *no Σ_PPM source change lands before G1+G2 exist.* G2 is
  strictly stronger than the eqp bit-compare my §4 step 0 proposed — per-branch tiles
  localize a broken move to the branch that broke, where eqp only says "something moved".
  G1 is the same stream-mode fixture my step 4 verification and risk 4 wanted; build it
  once, in Stage 0, owned jointly.
- **A3's Option 1 verdict and the 3a contract** ("no object may copy a config field it
  does not itself derive or resolve"; derived grid + resolved path travel as explicit
  args). This is exactly the right shape, and it answers the lead's question cleanly:
  **killing the mirror does NOT change my module boundaries** — the grab-bag `:1435-1441`
  and its replacement direct-read block both live in the driver tail, which stays in
  `ppm_sigma.py` under my split. The seam change is a signature change to one function I
  never move. My carriers (`_SigmaWindow`, `_SigmaBranch`) already satisfy contract 3a:
  every field is derived at construction, none copies config.
- **A1's delete-pass (his step 1) goes first among source changes.** The ~200 L
  `MinimaxWindowPair` layer, the dead `MinimaxConfig` import (`ppm_sigma.py:68`), the
  `sigma_window_quad=None` fallback (`:1424-1433`) — deleting before moving means the
  split never relocates a line that's about to die. Same discipline as 2A.
- **A1 §5 "do not force-merge" list**, especially item 1: the two τ-integrators stay
  separate (`w_isdf` scan-in-jit vs Σ Python loop, `ppm_sigma.py:1180-1183`). My split
  already treats the Σ τ loop as driver-side orchestration; A1's engine/physics boundary
  (`minimax_screening` = solvers + quad objects + `MinimaxNodes` + pole fit, interval
  derivation lives with the physics callers) is the same boundary my import graph drew.
  After A1's §3.4, `w_isdf.build_static_quadrature` absorbing the interval body makes W's
  side symmetric with my `ppm_windows` on Σ's side. Clean.
- **A4's Bug B fix by h5 injection, not prohibition** (his §5, incl. the `head_injected`
  idempotence attr and the blast-radius extension to the at-DFT reader and the gw_jax:628
  fallthrough). My proposal's §5 said either fix-in-place or post-split at the
  accumulator seam works; A4's in-place `ppm_pipeline` fix with the G1 red→green gate is
  more urgent than my sequencing and should not wait for the split.
- **A3 §1b ω-grid single derivation** (dead arange properties replaced by the live
  builder formula, Ry derived from eV) — this also answers my §5 open question about
  where the grid derivation moves when `gw_driver_helpers` dies: a `LorraxConfig`
  property, not a `ppm_pipeline` function as I proposed. A3's home is better (three
  consumers — `sigma_dispatch`, `gw_jax`, `ppm_pipeline` — all already hold `config`;
  my proposal would have made two of them import from `ppm_pipeline`).

## 2. What I concede from my own proposal

1. **Sequencing: seam before split.** My §5 preferred split-first ("pure moves rebase
   trivially"). I reverse that — but for a different reason than A3 gives. A3's stated
   rationale ("so the split doesn't have to move the getattr grab-bag and then re-fix
   it") is factually weak against my manifest: the split never moves the driver tail.
   The *real* argument, visible only once all four proposals are on the table, is that
   **three lenses want to change `compute_sigma_c_ppm_omega_grid`'s signature**: A3's
   `ppm_cfg`+`omega_grid_ry`+`sigma_kij_h5_path` (his 3d), A1's required
   `quad: MinimaxConfig` (his §3.1, flagged as R2), and A4's "required fields, no
   getattr defaults" (his Stage 4). A1's R2 says it explicitly: don't change that
   contract twice in two PRs. So: **one signature commit, owned by A3, carrying A1's
   required-quad and satisfying A4's no-defaults demand, landing before the split.**
   The split then cuts along the final signature and no move is ever re-fixed.
2. **My step 4(b) (sign-convention doc consolidation) is superseded by A4's K0.** A4's
   ledger is strictly better content than my "merge the three docstrings": the 7-row
   sign table, the corrected per-pole-term ±ω derivation (his §2a finding that the
   `Σc(−ω)=−Σc(ω)*` prose at `:29-30`/`:211-213` describes a false global theorem is
   the single best catch in the round — my proposal would have *consolidated the wrong
   derivation into one authoritative place*, which is worse than leaving it scattered),
   `_BRANCH_TABLE`, the `_DenomKind` de-overloading of `kernel_sign`, and the E_ref
   factorization comment. I withdraw 4(b) and hand the slot to K0. On the *home*: A4 is
   agnostic ("whichever file owns `_BRANCH_TABLE`"); under my split `_iter_branches`
   and the window builders land in `ppm_windows.py`, so the ledger lives in the
   `ppm_windows` module docstring — where every sign is *chosen* — with one-line
   pointers from the two rows consumed elsewhere (`neg_inv_sqrt_nk` in
   `ppm_tau_kernel`, the Im-projection row in `ppm_accumulators`), and A1's
   `time_axis`/fold enumeration anchored in the `MinimaxNodes` docstring
   cross-referencing it rather than duplicating it (A1 §7(a) — agreed).
3. **My step 4(a) (`open_sigma_kij_stream` extraction) dies.** A4's K2 `_H5Sink`
   subsumes it and goes further (window-granularity RMW instead of per-τ, n_ω-fold D2H
   reduction — his perf-asymmetry finding at `:1563-1569` is one I missed entirely; my
   proposal treated the streamed accumulator as move-only). The split's step 1 moves
   `_StreamedH5Accumulator` verbatim; K2 then rewrites it inside the new file.
4. **My step 4(c) (`_write_sigma_omega_h5` public rename + `sc_iteration.py:664`
   SimpleNamespace cleanup) folds into A3's step 3**, which already rewrites every
   `ppm_pipeline` helper signature and touches `sc_iteration.py:668`. No reason for two
   PRs to visit those lines.
5. **Manifest amendment forced by the seam-first order**: my `ppm_accumulators`
   manifest listed the jax projector pair (`:361-424`) as a move. Since K2 deletes it
   post-split, the move is a dead-man-walking relocation — acceptable (pure moves must
   be complete or the bit-compare discipline breaks), but the K2 PR should land in the
   same session so the corpse doesn't linger.

## 3. Where I push back

### 3.1 A1's request: co-locate window-build and τ-integrate — REJECTED

A1 §7 asks the splitter to keep `_build_*_sigma_windows` and
`minimax_tau_integrate_sigma` in one module because "they share the `MinimaxNodes`
conventions". The import graph says no:

- The τ loop binds window × kernel × accumulator — it calls `_materialize_window_mask_B`,
  the τ-kernel (`:1264` via the kernel getter), and `accum.add_tau` (`:1216-1224`),
  and owns the Python-loop-not-scan decision (`:1180-1183`). Put it in `ppm_windows`
  and the leaf module now imports the kernel cache and the accumulator protocol —
  the acyclic graph in my §2.3 collapses, and the one file that was unit-testable
  without a GPU (window edges, T-thresholds, ξ-scaling) now imports shard_map machinery.
- The actually-shared convention surface is **one fold**: `α_eff = α·exp(−i·E_ref_sum·t)`
  at `:1210-1212` — which A1's own §3.2 freezes in place ("moving it would change the
  locked numerical path for zero legibility gain"). One fold does not justify merging an
  orchestrator into a leaf. The discoverability problem A1 is actually pointing at is
  solved by his own remedy: the fold enumeration in the `MinimaxNodes` docstring plus
  K0's E_ref factorization comment at the fold site.

Counter-offer: A1's step 3 (node-accessor routing) lands **after** the split and then
touches exactly one Σ-side file — the crossing `t_scale` rescale (`:851-853`) is in
`ppm_windows.py` by then, and the `to_minimax_nodes(time_axis=..., t_scale=1/ξ)` call
replaces hand-rolled construction inside the same module that owns the window story.
That's better locality than A1 gets from the current monolith either way.

### 3.2 A4's "K2 before any split" — push back, with a concession attached

A4 §8 claims split-first means K2 "moves code twice". The count is symmetric:
K2-then-split = one rewrite + one move; split-then-K2 = one move + one rewrite. Nothing
is moved twice in either order. What actually differs:

- **Blast radius and diff legibility.** K2 is the riskiest change in the whole program
  (rewrites the accumulation machine, changes streamed-path data movement by design,
  carries the Bug-C shard-handling fix whose severity is still unverified — A4's own
  ⚠️). A rewrite of that class wants to happen inside a 330-L `ppm_accumulators.py`
  where the PR diff *is* the design, not inside a 1631-L file where reviewers must
  re-establish context for every hunk.
- **Failure isolation.** With split-first, G2 bit-compare brackets the pure move
  (any diff = impure move, revert), and then G1+G2 bracket K2 alone. With K2-first, a
  G2 failure during the subsequent move has two candidate causes.
- The only real cost of split-first is §2.5's dead-man relocation of the 64-L jax
  projector. Cheap.

**Concession attached**: K2 owns the internal design of `ppm_accumulators.py` — my move
manifest is the initial state, A4's `_TauAccumulator` + two sinks is the end state, and
the driver-side change K2 needs (accumulator-per-ω-half dependency injection replacing
construction at `:1359-1375`, killing the `per_half` tuple-key dict `:1599-1611`)
crosses my driver/accumulator seam, so K2's PR gets to edit both files. The protocol
seam I split along (`_SigmaAccumulator`, `:897-920`) is exactly the seam A4 says the
τ loop already programs against — the split makes K2's "entirely behind that seam"
claim structurally enforced.

Same logic applies to K0/§3-window-table (A4 Stage 3): lands after the split, in
`ppm_windows.py`, where the rename diff is legible. A4's own ordering (Stage 3 after
Stage 2) is compatible.

### 3.3 A4 Stage-4 signature: `debug_cfg` is stale — A3's census wins

A4's proposed narrowed signature includes `debug_cfg`. Verified this session:
`grep -n debug src/gw/ppm_sigma.py` matches only docstrings — zero `config.debug`
reads survive 2A (A3's field census rows 12-13 confirm the debug reads live in
`gw_jax.py:817,920`). The signature commit should be A3's 3d verbatim (no debug
object), plus A1's `quad` required. Also endorse A3's `""`→`None` normalization for
`sigma_kij_h5_path` and his 3f flag on the `sigma_freq_debug_file` cwd-vs-input_dir
latent inconsistency — that decision belongs in the consensus round, not silently in
the collapse commit.

### 3.4 A3's "the split should cut along the new signature... windows/τ-kernels receive
plain scalars as they already do" — agreed, with one boundary note

A3 §7 stipulates only the driver may import `PPMConfig`. Under my split that's
automatic (windows/kernel/accumulators receive scalars/arrays, per `:1571-1608`), and
I adopt it as an explicit rule in the split PR description: **`ppm_windows`,
`ppm_tau_kernel`, `ppm_accumulators` must not import from `gw_config`.** This is the
enforceable version of A3's §4 point 4.

## 4. Revised position — the consolidated sequence

My split survives intact as content (same 4 files, same manifests minus §2 concessions)
but moves from first to third in line. Proposed program order for the consensus round,
with owners:

| # | Work | Owner | Files touched | Gate |
|---|------|-------|---------------|------|
| 0 | G1 (kij vs kij_stream parity, expected RED), G2 (per-branch npz), G3 (head neg-branch test) | A4 | tests only | — |
| 1 | Delete-pass: A1 §3.4 dead layer, `ppm_sigma:68` import, `MinimaxWindowPair` family | A1 | `minimax_screening`, `w_isdf`, 1 L `ppm_sigma` | pytest + golden gates |
| 2 | **The signature commit**: A3 steps 1-3 (post_init, ω-grid property, direct reads, mirror deletion path) + A1 §3.1 config merge with required `quad` + `""`→`None` | A3 | `gw_config`, `gw_driver_helpers`, `ppm_pipeline`, driver tail of `ppm_sigma`, `sigma_dispatch`, `gw_jax` | h5 bit-identical (A3 step-2 gate) |
| 3 | Bug B h5-injection fix (A4 §5), written against the step-2 helper params | A4 | `ppm_pipeline` (+8 L) | G1 RED→GREEN |
| 4 | **The split**: my steps 1-3, pure moves only, revised manifest (§2.3/2.5 above) | me | new `ppm_windows`/`ppm_tau_kernel`/`ppm_accumulators`, slim `ppm_sigma` | G2 bit-identical per step + AOT prewarm timing check (my §6 risk 2) |
| 5 | K2 accumulator unification + Bug C + per-ω-half injection | A4 | `ppm_accumulators`, driver | G1+G2+golden, `sigma.exec` ±3% |
| 6 | K0 ledger/`_BRANCH_TABLE`/`_DenomKind` + window spec-table, home = `ppm_windows` | A4 | `ppm_windows` (+2 pointer lines elsewhere) | G2 bit-identical, no retrace |
| 7 | A1 steps 3-5: node accessor (chi0 + crossing fold), pole-fit unification, shipped-table holes | A1 | `minimax_screening`, `w_isdf`, `head_correction`, `ppm_windows` (crossing fold) | chi0 hash + A1's step-4 sub-case grid |
| 8 | Physics rewires: `static_limit` (+`Wc0` on `PPMBuildResult` — data seam per A3 3e), `sigma_at_dft_energies` at gw_jax:649 | physics lens | driver, `gw_jax` | new gates + BGW parity |

Sequencing dependencies stated as constraints, for the consensus doc:

- 0 before everything (adopted rule).
- 1 before 4 (don't move lines that are about to die).
- 2 before 4 (split cuts along the final signature; one signature commit — A1's R2).
- 3 needs 2's helper-param rewrite (A3 §7 flagged the same 40-line collision); 3 before
  4 is preferred (Bug B is a live physics wrong-answer, refactors queue behind it) but
  not required — 3 and 4 touch disjoint files.
- 4 before 5 and before 6 (blast radius + failure isolation, §3.2).
- 7 after 4 (its Σ-side touch point is then localized in `ppm_windows`); 7's engine-side
  steps are file-disjoint from 5/6 and can run in parallel with them.
- 8 last, each item its own gated commit (all three of us said this independently).
- `head_correction.py` stays intact this round (my §3 verdict, unchallenged; A1's
  pole-fit unification in step 7 edits `fit_head_ppm`'s internals but not the module
  boundary; my optional `head_sources.py` IO split stays deferred).

## 5. Top open conflict for the consensus round

**Step 4 vs step 5 order — split-then-K2 (me) vs K2-then-split (A4).** It's the only
genuine ordering disagreement left standing after the concessions above (seam-first is
conceded, A1's co-location request I reject on import-graph grounds and expect the
round to sustain). The move-count argument is symmetric; the decision criterion should
be *where the risky diff is reviewed* — inside a 330-L file with G1/G2 isolating one
change class per PR (my position), or inside the monolith to avoid relocating 64 L of
soon-dead projector (A4's position). If the round sides with A4, my split manifests
need only mechanical amendment (move `_TauAccumulator`+sinks instead of the two old
classes) — the boundaries themselves are order-invariant, which is the strongest sign
they're the right boundaries.
