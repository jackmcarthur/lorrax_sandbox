# Round 1 response — Agent 4 (kernel legibility lens) reacting to Agents 1/2/3

**Read**: all four Round-1 proposals + SIGMA_PPM_MAP.md, re-verified against
`sources/lorrax_D` @ `3cad3dd` (ppm_sigma.py 1631 L; getattr block :1435-1441; driver
`compute_sigma_c_ppm_omega_grid` :1397; accumulators :948/:1029; jax projector pair
:361-424 / numpy mirror :923-945; `_inject_analytic_head` ppm_pipeline.py:109).

**Headline**: I concede my biggest ordering demand (K2-before-split) to Agent 2, I take
Agent 3's side on the seam-vs-split ordering question (with a correction to their stated
reason), and I claim one merge: Agent 2's step 4(a) (`open_sigma_kij_stream` extraction)
is subsumed by my K2 and must not land as a separate commit. The top conflict in the
round is the three-way ownership of the accumulator/stream seam (Agent 2 step 4a, my K2,
Bug B) — resolved below by making K2 the single owner and Bug B an in-place pre-fix.

---

## 1. Where I AGREE (adopt as-is)

**A2 (module split): the 4-file cut is right, and it improves my lens.** The seams Agent
2 chose are exactly the seams my proposal programs against: `ppm_accumulators.py` is
bounded by the `_SigmaAccumulator` protocol (:897-920), which is the interface K2 hides
behind; `ppm_windows.py` is where `_BRANCH_TABLE` and the sign ledger want to live
(their step 4(b) and my K0 are the same deliverable — merge them, one commit, their
file). Their rejection of one-file-per-stage is correct for legibility reasons I care
about: the S2→S3→S5 sign chain must stay a readable narrative, and 4 files keep
window-birth (`_build_three_sigma_windows` :789) and window-consumption
(`begin_window` :986-991) each in a single-concern file. Their §1 "definition/use
distance" evidence matches my §2 "signs live in 7 places" evidence — same disease,
two measurements.

**A3 (config seam): Option 1 + both amendments, verbatim.** The getattr block
:1435-1441 is item 5 on my incidental-complexity list and Agent 3's census is
strictly better than my treatment: I flagged the magic literal at :1435
(`0.018374661087827496`), they found it is the *fourth* copy of the default and traced
all 15 mirror fields. Their value-vs-capability split for `invalid_mode` (§3e —
values validated in `PPMConfig.__post_init__`, capability `NotImplementedError` stays
in the kernel at :1470-1482) is the correct generalization of the "3 silent default
layers" defect class I described. Their §1b two-formula ω-grid finding is new to me and
I endorse the fix (adopt the live builder formula, Ry derived from eV). Also adopt
their `"" → None` normalization for `sigma_kij_h5_path` — my K2's `_H5Sink` wants the
`str | None` contract, not "falsy string handled at :118/:123 by accident".

**A1 (quadrature): §3.3 pole-fit unification is the structural closure of Bug A.** My
G3 regression test (negative-Ω² branch of `fit_head_ppm`, continuity of `R_h` across
Ω²→0) pins the *symptom*; their `two_point_pole_fit` extraction removes the *mechanism*
(two implementations of the same algebra). Both should land; G3 first (it then guards
their step 4). Their §4 step-4 unit-test grid (full {probe kind}×{Ω² sign}×{denom≈0}
enumeration, no "matches at ULP" hand-waving) is exactly the audit discipline the memory
notes demand — G3 becomes one row of that grid. I also adopt their §5 item 1 without
reservation: the τ-loop mechanics (chi0 scan vs Σ Python loop) must never merge — that's
my "essential complexity" row 4 (:1180-1183) stated from the other side.

**A1 §7(b) hands me a design point and I accept it**: `_SigmaWindow.project` /
`project_code` should *derive* from the node kind (`CrossingMinimaxQuadrature` → IMAG)
instead of being a free parallel field set by string at :854/:868. Post-K2 there is
exactly one projector (the numpy one), so the derivation has one producer
(`ppm_windows` row-builder, §3 of my proposal) and one consumer. My IntEnum item
(`_Project.FULL/IMAG`) survives as the *type* of the derived value. This also
discharges their §2.4 hand-off: the head-vs-body invalid-policy skew is a physics
question — I agree it's real, agree it needs a BGW parity check, and agree it is NOT
part of any refactor commit in this effort. Park it in the report's open-physics list.

**Gates**: nobody objected to G1/G2/G3 (my Stage 0), and two agents independently
documented why G1 is mandatory: Agent 2's risk 4 ("KIJ_STREAM is exercised by no golden
gate... `_select_accum_mode` :95-128 falls back to HOST on the gates' small grids") and
Agent 3's risk table (streamed smoke test required for the `or None` change). I restate
the consensus rule with one softening (see §3): **no behavior-adjacent Σ_PPM change
lands before G1 (kij vs kij_stream parity) + G2 (per-branch reference tiles) exist.**

## 2. Where I CONFLICT (and how I'd resolve each)

### C1 — top conflict: three owners of the accumulator/stream seam

Agent 2's step 4(a) extracts the driver's stream-h5 setup (:1543-1569) into
`ppm_accumulators.open_sigma_kij_stream`; my K2 *deletes* `_StreamedH5Accumulator`
(:1029-1079) and the jitted projector pair (:361-424) and replaces the whole
arrangement with `_TauAccumulator` + `_MemoryTileSink`/`_H5Sink` — where the h5
dataset setup is `_H5Sink.__init__`. If step 4(a) lands as written, K2 rewrites it a
commit later: extraction wasted, two review passes over the same 60 lines, and the
step-4(a) verification run (stream-mode MoS2) paid twice.

**Resolution — merge, don't sequence**: Agent 2's steps 1-3 (pure moves) land as
proposed, but step 4(a) is struck and replaced by "K2 lands inside
`ppm_accumulators.py`". K2's diff is then file-local (~330 L file, net ≈ −80 L), which
answers the legibility standard Agent 2 holds pure moves to: the one non-pure change at
this seam happens exactly once, in one file, gated by G1+G2. Step 4(b) (sign-doc
consolidation) and 4(c) (`_write_sigma_omega_h5` rename) survive unchanged.

### C2 — seam-first vs split-first (Agent 3 vs Agent 2): I side with Agent 3, but their stated reason is wrong

Agent 3's argument for seam-first is "so the split doesn't have to move the getattr
grab-bag and then re-fix it" (§7). Factually weak: Agent 2's manifest keeps
`compute_sigma_c_ppm_omega_grid` (:1397-1631) — including the getattr block :1435-1441
— in the slimmed `ppm_sigma.py` (§2.2, "stays"). The split never moves the grab-bag.

The *correct* arguments for seam-first, and why I still land on Agent 3's side:

1. **One-signature-change rule.** Agent 1's risk R2 and my Stage 4 both demand the
   `compute_sigma_c_ppm_omega_grid` contract change exactly once. That change bundles
   three agents' edits: Agent 3's `ppm_cfg`/`omega_grid_ry`/`sigma_kij_h5_path` args,
   Agent 1's "quad required, delete the :1430-1433 None-fallback" (they propose the
   identical edit — one owner: Agent 3's step 2), and my "no getattr defaults"
   requirement. Doing this on the pre-split tree means the split's pure moves are cut
   against the *final* signature, and no post-split commit re-touches the driver head.
2. **The seam is small and bit-identical-gated** (Agent 3's h5 bit-diff gate), so it
   costs the split nothing to rebase over; whereas landing the seam after the split
   forces the seam PR to touch 4 files (driver + pipeline + dispatch + gw_jax) *plus*
   re-verify the freshly-moved imports.
3. **2C/2D physics work is blocked on a trustworthy single-path `invalid_mode`**
   (Agent 3 §5 "before 2C/2D") — physics fixes shouldn't queue behind 5 move commits.

Concession to Agent 2 inside this resolution: their real concern was *concurrent* edits
colliding on the driver tail ("serialize, don't parallelize" §5) — granted absolutely.
The orders differ only in which serialized block goes first; nobody's content changes.

### C3 — home of the sign ledger (me vs Agent 1 vs Agent 2): partition, don't pick

Three candidate homes were proposed: my "wherever `_BRANCH_TABLE` lives", Agent 2's
`ppm_windows.py` header (step 4b), Agent 1's `MinimaxNodes` docstring as "the anchor"
for the sign story (§7). A single mega-doc in any one place re-creates the
told-three-times problem (module docstring :23-34, `_iter_branches` :204-214,
`_combine_coeff_with_sigma_tau` :368-384). Partition by *who decides the sign*:

- **`ppm_windows.py` header** (Agent 2's home wins for the branch story): the corrected
  per-term ±ω derivation (my §2a — the current "Σc(−ω) = −Σc(ω)*" prose at :29-30 and
  :211-213 states a false global theorem), `_BRANCH_TABLE`, window prefactors, the
  `T = ω_max + edge·ξ` sentence. Rows 2, 3, 4, 7 of my 7-row ledger.
- **`MinimaxNodes` docstring** (Agent 1's home wins for the fold story): the three
  alpha-fold conventions (their §2.3) + `time_axis` semantics. My §3.2-adjacent row 6
  cross-references it.
- **Driver, at the α_eff build (:1212)**: the E_ref factorization identity
  (my K0 last bullet), because its three factors live in three files (:1212, :564-567
  via `build_G_tau`, :600 in `_build_W_t_q`) and only the driver sees all three.
- **`ppm_tau_kernel.py`** keeps row 1 (the global −1/√N_k at :540, renamed).

Each site carries one-line pointers to the other three. No duplication, per the repo's
single-source rule; the "one page you can audit signs from" is the ppm_windows header,
which *links* rather than restates.

### C4 — Bug B timing (soft conflict with Agent 2): fix in-place NOW, the move carries it

Agent 2 offers two options (§5: fix at ppm_pipeline.py:126-127 pre-split, or as
`add_dense` on the post-split stream seam) and prefers the split-integrated one via
their sequencing ("split → 2B → Bug B", §7 item 4). I hold my position: Bug B is a
**live physics bug** (streamed Σc silently head-less, and my §5 showed the blast radius
is three consumers, including the at-DFT eval at ppm_pipeline.py:209-220 reading the
head-less h5). It should not wait behind ~10 refactor commits. The in-place fix is
~30 L of rank-0 h5 RMW + the `head_injected` idempotence attr, entirely in
`ppm_pipeline.py`, which **no other workstream touches until Agent 3's step 3** — zero
merge risk. Agent 2 already granted this is fine ("the move then carries the fix").
Locking it: Bug B = Phase 1, immediately after G1 exists (and G1's RED→GREEN flip is
the fix's acceptance test). I also re-flag my push-back for the record: any
"forbid streaming without a head path" resolution is rejected — large-ω-grid streamed
runs are precisely where the ω-dependent head matters.

### C5 — minor: Agent 1's node accessor (§3.2) crosses the split boundary

Their accessor rewires the crossing rescale at ppm_sigma.py:851-853, which Agent 2's
step 2 moves into `ppm_windows.py`. Trivial resolution: Agent 1's step 3 (node
accessor) lands **after** the split, editing `ppm_windows.py` + `w_isdf.py`. Their
steps 1-2 (dead-layer delete in `minimax_screening.py`, config-class merge in
`minimax_config.py`/`gw_config.py`) touch no file the split moves and can run in
parallel with anything. One caveat I add from my lens: their config-class merge (§3.1)
changes what `sigma_quadrature_config` returns — Agent 3's step 2 makes `quad` a
required kernel arg of that type. Those two edits must agree on the class name/shape
before either lands; propose Agent 1's merged `MinimaxConfig` is settled in Round 2 and
Agent 3's signature uses it from day one (avoids typing the signature twice).

## 3. What I CONCEDE from my own Round-1 proposal

1. **K2-before-split → split-before-K2.** My §8 argued "K2 first or the code moves
   twice". Agent 2's counter is better: pure moves are mechanical and bit-gated, and a
   post-split K2 is a *file-local* rewrite of `ppm_accumulators.py` — strictly more
   reviewable than a 1631-L-file rewrite followed by moving the fresh code. The
   moves-twice cost is one cheap mechanical commit; the review-legibility gain is real.
   (Conditional on C1: step 4(a) is struck so K2 is the only non-pure change at that
   seam.)
2. **Drop `debug_cfg` from my Stage-4 target signature.** Agent 3's field census (§1,
   rows 12-13) shows the debug fields are read in `gw_jax.py:817/:920`, never in the
   kernel — my proposed kwargs list was carrying a dead parameter. Adopt their §3d
   signature verbatim (`ppm_cfg`, `quad`, `omega_grid_ry`, `sigma_kij_h5_path`).
3. **Soften my consensus-rule proposal** ("no Σ_PPM source change before G1+G2") to
   *behavior-adjacent* changes: Agent 1's step 1 is 0-caller deletions in
   `minimax_screening.py` verified by grep + pytest — holding that hostage to a
   stream-parity gate protects nothing. The rule binds the seam change, the split, K2,
   K0, and all physics fixes.
4. **My per-half accumulator fold (Stage 2, killing the `per_half` tuple-key dict at
   :1599-1611) stays in K2 but demoted to "if it doesn't grow the diff"** — Agent 2's
   discipline (each commit's diff must be attributable to one intent) is right, and the
   per_half kill is driver-side, not accumulator-side; if it muddies K2's review, it
   becomes its own micro-commit after K2.

Not conceded: G1/G2 as hard prerequisites for everything behavior-adjacent (both other
implementation lenses independently justified it); Bug B before the refactor train
(C4); Bug C (single-process multi-GPU `addressable_data(0)` vs
`make_array_from_process_local_data`, :997-998/:1024-1026) staying inside K2 with the
20-minute 2-GPU repro first.

## 4. Revised position — consolidated sequencing for the round

Phases are serialized on the Σ-side files; ⟂ marks work that may run in parallel
because it touches disjoint files. Every phase ends pytest-green + golden-gates-green.

| Phase | Content | Owner (lens) | Files | Gate |
|---|---|---|---|---|
| 0 | G1 (kij vs kij_stream parity — expect RED at head), G2 (per-branch reference tiles, all 4 branches × 3 windows non-empty), G3 (Bug-A negative-branch regression) + baseline eqp/h5 stash | me | tests only | G3 green; G1 red-documented |
| 0⟂ | A1 step 1: dead-layer delete (`MinimaxWindowPair` etc., ~200 L) + dead imports | A1 | minimax_screening, w_isdf | pytest + grep-zero |
| 1 | **Bug B fix in-place** (head → stream h5, idempotence attr) | me | ppm_pipeline only | G1 flips GREEN |
| 2 | **Config seam** (A3 steps 1-4) incl. A1's "quad required" edit and the merged `MinimaxConfig` shape agreed in Round 2 | A3 (+A1 for quad type) | gw_config, driver tail, ppm_pipeline, gw_driver_helpers, gw_jax, sigma_dispatch | h5 bit-identical to stash |
| 3 | **Module split, pure moves** (A2 steps 1-3; step 4a struck per C1; 4b deferred to phase 5; 4c rides along) | A2 | new ppm_windows / ppm_tau_kernel / ppm_accumulators | bit-identical + AOT-prewarm timing check |
| 4 | **K2 accumulator unification** (delete jax projector pair, `_TauAccumulator` + 2 sinks, Bug C fix or loud assert) — file-local in ppm_accumulators | me | ppm_accumulators (+driver call sites) | G1 + G2 + golden gates; sigma.exec ±3% |
| 5 | **K0 sign ledger + `_BRANCH_TABLE` + window spec table** = A2 step 4(b) merged; ledger partitioned per C3 | me (+A2 doc merge) | ppm_windows, ppm_tau_kernel, driver comments | G2 bit-identical; no retrace |
| 5⟂ | A1 steps 2-4: config-class merge, node accessor (now targets ppm_windows), `two_point_pole_fit` unification | A1 | minimax_config, minimax_screening, w_isdf, ppm_windows, head_correction | chi0 hash; A1's §4 sub-case grid; G3 |
| 6 | Physics re-wires, each own commit + gate: `static_limit` (Wc0 on `PPMBuildResult` — data seam per A3 §3e), `sigma_at_dft_energies` at gw_jax:649, head-vs-body invalid-policy decision (BGW parity) | physics round | driver, gw_jax | new fixtures + BGW check |

Dependencies stated once: 0→1 (G1 is Bug B's acceptance test); 2→3 (split cuts against
final signature, C2); 3→4 (K2 is file-local only post-split, concession 1); 4→5 (the
ledger documents the *post*-K2 single projector, not the doomed pair); A1's accessor
after 3 (C5); phase 6 after 2 (needs the trustworthy invalid_mode path).

**My lens's deliverables in one line each**: G1/G2/G3 (phase 0), Bug B (phase 1), K2 +
Bug C (phase 4), K0 ledger + corrected ±ω derivation + window spec table + project-code
derivation from node kind (phase 5).

*Agent 4 response. All line refs re-verified against `sources/lorrax_D` @ `3cad3dd`,
2026-07-06.*
