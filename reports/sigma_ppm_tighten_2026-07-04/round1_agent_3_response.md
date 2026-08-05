# Round 1 — Agent 3 RESPONSE: config seam vs the other three lenses

**Lens**: config/options seam (`PPMSigmaRuntimeOptions` kill). Responding to Agent 1
(quadrature), Agent 2 (module split), Agent 4 (kernel legibility). All refs against
`sources/lorrax_D` @ `3cad3dd` (re-verified this session: getattr block is
`ppm_sigma.py:1435-1441`, `sigma_window_quad: ... | None = None` at `:1404` with the
fallback at `:1424-1433`, mirror at `gw_driver_helpers.py:17/:230`).

---

## 1. Where I AGREE (and fold in)

**A4's "gates before any source change" rule — adopt as the consensus ground rule.**
G1 (kij vs kij_stream parity) is *exactly* the streamed-mode smoke my §6 risk table
demanded for the `""`→`None` normalization of `sigma_kij_h5_path`; G2 (per-branch
reference `.npz`) subsumes my step-0 baseline stash. My step 0 is deleted; I build on
G1/G2/G3. One addition: G1 must be run once **against the post-seam tree too** (my
steps 2-3 touch `_select_accum_mode`'s inputs at `ppm_sigma.py:118-123`), which it
will be anyway if the sequencing in §4 holds.

**A4's Bug-B fix shape and its parameter needs.** His fix gives
`_inject_analytic_head` (`ppm_pipeline.py:109`) explicit `sigma_kij_h5_path` + `meta`
args — precisely the two values my step 3 was going to make explicit when the mirror
dies. Same edit, two motivations. He lands first (see §4); my step 3 rebases on his
signature rather than the current `ppm_options` one. His `""`-vs-None observation
("works by accident", his §8) is my 3d `or None` normalization — one owner (me), his
G1 gates it.

**A1's config merge + required-quad must ride the same commit as my signature swap.**
His R2 ("don't change `compute_sigma_c_ppm_omega_grid`'s contract twice in two PRs")
is the sharpest sequencing observation in the round and I accept it as binding: my
step 2 (kill getattr, explicit kwargs) and his step 2 (one `MinimaxConfig` class, quad
becomes required, delete the `:1424-1433` third-default-copy) are **one commit series
with one owner**. I volunteer to carry his §3.1 (it's ~30 L in `minimax_config.py` +
`gw_config.py:748-757`); he keeps everything below the signature (node accessor §3.2,
pole-fit unification §3.3, shipped-table holes §3.5) on his own branch — none of those
touch the seam.

**A1's "build quad_config once in `ppm_pipeline`, don't re-derive via property per
call".** Accepted, and consistent with my contract 3a: the property is the single
*formula* home; the pipeline calls it once and passes the instance. Same treatment for
`config.omega_grid_ry` (my 3c property): `ppm_pipeline` evaluates it once, passes the
array. Properties = derivation site, not per-consumer allocator.

**A2's split shape (4 files) and where the seam lives under it.** His statement that
the getattr grab-bag "sits in the driver and stays in the driver file under my split"
(his §5) matches my contract exactly: post-seam, *only* the driver function may hold
`PPMConfig`; `ppm_windows`/`ppm_tau_kernel`/`ppm_accumulators` receive plain scalars
(already true at `:1571-1588`). No content conflict at all — only ordering (§2).

**A2's `_write_sigma_omega_h5` public rename + `sc_iteration.py:664-675` cleanup.**
Strongly agree and want it *in my step 3*, not his step 4: the `SimpleNamespace` stub
at `sc_iteration.py:673` exists **only to fake a `PPMSigmaRuntimeOptions`** — it is a
second, hand-rolled instance of the mirror and dies naturally when the writer's
signature becomes `(config, ...)`. Deleting the mirror while leaving a stub that
imitates it would fail my own grep-zero gate (step 4).

**A4's Bug-C flag (shard-0 vs process-local in `_HostOmegaAccumulator`).** Out of my
lane, but I confirm my seam neither worsens nor masks it — the accumulator selection
inputs change spelling, not values. His 2-GPU single-process check should join the
shared gate set.

## 2. Where I CONFLICT

### C1 (top conflict of the round) — seam-vs-split ordering. A2 wants split→seam; I hold seam→split.

A2's argument: after the split, 2B touches exactly one Σ-side file. True but weak —
count my blast radius under both orders. Seam-first: `gw_config.py`,
`gw_driver_helpers.py` (delete), `minimax_config.py`, `ppm_pipeline.py`,
`ppm_sigma.py` driver tail (~40 L), `sigma_dispatch.py:227-247`, `gw_jax.py:451-455`,
`sc_iteration.py:664-675`. Split-first: the **identical set** — the split doesn't
relocate the driver tail, `ppm_pipeline`, or any of the out-of-file consumers. The
split buys my diff nothing.

What seam-first buys the *split*: (a) the pure-move manifest cuts along the **final**
signature — `precompile_sigma` and the driver move exactly once with their
final contracts, and A1's R2 (one signature owner) is satisfied by construction
rather than by rebase discipline; (b) A2's own bit-identity discipline ("steps 1-3
diffs must show ppm_sigma.py shrinking by exactly the moved line count", his risk 3)
is *easier* to audit when the moved code no longer contains a getattr block that the
next PR is contracted to delete; (c) pure moves rebase trivially over a plumbing
change — a plumbing change rebasing over relocated files means re-resolving every
hunk's new home by hand. A2 himself concedes "either order works"; given that, the
tiebreakers all point seam-first. **Ask: A2 accepts seam→split.**

### C2 — A4's Stage-4 signature includes `debug_cfg`. Push back: no.

My field census (proposal §1, rows 12-13) shows **zero** debug fields read inside
`compute_sigma_c_ppm_omega_grid` or below — `sigma_freq_debug_*` are consumed in
`gw_jax.py:817/:920` directly off `config.debug`. A `debug_cfg` parameter with no
reader is the mirror-rot mechanism in miniature: a slot that exists "for later" and
drifts. Contract 3a tie-breaker applies: a parameter may exist only when a read
exists. If K0's sign ledger later wants a config-gated debug print, `debug_cfg` gets
added *in that commit*, next to its first reader. Until then the signature is
`(wfns, ppm, meta, mesh_xy, *, ppm_cfg, quad, omega_grid_ry, sigma_kij_h5_path,
print_fn)` — my 3d with A1's merged `MinimaxConfig` as `quad`'s type.

### C3 — A4 schedules the config seam at his Stage 4 (after K2+K0). Too late.

K2 rewrites the accumulators and their selection path; `_select_accum_mode`
(`ppm_sigma.py:95-128`) consumes `omega_accumulation` + `sigma_kij_h5_path` — two of
the getattr-defaulted values (`:1438-1439`) with the `""`-falsy quirk (`:118,:123`).
Rebuilding that machinery **on top of** a silent-default seam means the new
accumulator inherits the old failure mode (typo → getattr default → wrong mode,
unobservable); rebuilding it on a fail-loud seam means a wiring mistake in K2 raises
at first call. The invalid_mode lesson says defaults-layering bugs are found years
late; don't construct new machinery over the layer we've agreed to demolish.
**Ask: seam lands before K2.** (K0 — renames/comments — is genuinely
order-independent of me; only the shared touch on `_prepare_sigma_state`'s docstring
needs a heads-up.)

### C4 — A2's step 4(a) (`open_sigma_kij_stream` extraction) vs A4's K2. Arbitration: drop 4(a).

Both claim the driver's stream-h5 setup (`:1543-1569`). A2 extracts it verbatim into
`ppm_accumulators`; A4's K2 then **deletes `_StreamedH5Accumulator` entirely** and
replaces the per-τ h5 RMW with an `_H5Sink` that writes per-window — the extracted
function's body doesn't survive K2 in recognizable form. Extracting code that the
next scheduled PR deletes is churn. A2's split should move the stream setup *as part
of the driver's pure move only if it stays in the driver*, or (cleaner) leave `:1543-
1569` untouched in step 1-3 and let K2 be the commit that consolidates streamed
storage into `ppm_accumulators`. A2's step 4(b) (sign-doc consolidation) and 4(c)
(writer rename — which I've claimed for my step 3, §1) survive; 4(a) is superseded.

### C5 — minor, A1: `MinimaxConfig` merged-class fields vs `__post_init__` placement.

A1 puts the merged class in `minimax_config.py` with defaults as the single site; my
3b puts Σ *value* validation in `PPMConfig.__post_init__` (`gw_config.py:541`). No
collision — quad accuracy knobs validate in the merged class, physics-mode strings
(`invalid_mode`, `fermi_reference`, `omega_accumulation`) validate on `PPMConfig` —
but the boundary must be stated in the commit so the two `__post_init__`s don't grow
overlapping checks. I own both since they ride my commit series (§1).

## 3. What I CONCEDE from my own proposal

1. **`quad`'s type**: `SigmaQuadratureConfig` → A1's merged `MinimaxConfig` (one
   class, two instances). My 3d text updates; the "required, no None-fallback"
   substance is unchanged and was independently proposed by both of us.
2. **My step-0 baseline** → superseded by A4's G1/G2/G3. Better gates than mine
   (mine compared only h5 bit-identity; his G1 adds cross-mode parity, which my
   `or None` change specifically needs).
3. **Bug-B precedence**: I had claimed my seam should land "before 2D's bug fixes".
   Wrong for Bug B specifically — it's a 30-L in-place fix wanted *now*, and my step 3
   carries/rebases it for free (A2's §5 made the same point). Bug B goes first.
4. **`sigma_freq_debug_file` cwd-vs-input_dir decision** (my 3f): I proposed settling
   it in this round; conceding it's a behavior question orthogonal to all four
   workstreams — file it as a one-line entry in the physics-tail bucket, decide when
   someone actually uses the flag.
5. **Timing softness**: I claimed seam-before-split as a requirement; downgraded to a
   strong preference with the three tiebreakers in C1 — if consensus goes split-first
   the seam still works, at the cost of R2 discipline living in rebase instead of
   structure.

## 4. Revised position — the merged sequence

Consensus rule (from A4, adopted): **no Σ_PPM source change lands before G1+G2 exist.**
Tie-breaker contract (mine, unchanged, now with A4/A2 compliance verified): *bundles
carry only what they derive or resolve; anything readable off `config` travels as
`config.ppm` or a scalar; `getattr(x, 'field', default)` on config-like objects is
banned in `gw/`.* A4's `_TauAccumulator`+sinks comply (constructed from scalars); A2's
relocated carriers comply (no new fields).

| # | Work | Owner | Depends on | Gate |
|---|------|-------|-----------|------|
| 0 | G1 (kij↔kij_stream parity, expected RED), G2 (branch/window `.npz`), G3 (Bug-A regression test) | A4 | — | gates exist + G3 green |
| 1 | Bug B fix in-place at `ppm_pipeline.py:126-127` (h5 head injection + idempotence attr) | A4 | 0 | G1 red→green |
| 2 | Delete-pass: A1 §3.4 dead layer (~200 L `minimax_screening`), dead imports (`ppm_sigma.py:68,71,74`) | A1 | 0 | pytest + golden gates; grep-zero |
| 3 | **The one signature commit series**: my steps 1-4 (post_init, ω-grid property, getattr→direct reads, mirror delete, `sc_iteration` stub kill) + A1 §3.1 (merged `MinimaxConfig`, required `quad`, `:1424-1433` fallback delete) | me (A1 reviews) | 1, 2 | pytest; `sigma_mnk.h5` bit-identical; G1 re-run; grep-zero on `PPMSigmaRuntimeOptions` |
| 4 | Module split, pure moves (A2 steps 1-3) + 4(b) doc consolidation; 4(a) dropped per C4 | A2 | 3 | bit-compare per move; AOT-prewarm timing check |
| 5 | K2 accumulator unification + Bug-C handling, inside `ppm_accumulators.py` | A4 | 3, 4 | G1, G2, golden gates; `sigma.exec` ±3% |
| 6 | K0 sign ledger + branch table + window spec-table, in `ppm_windows.py` | A4 | 4 | G2 bit-identical; no retrace |
| 7 | A1 §3.2 node accessor + §3.3 pole-fit unification (+§3.5 table holes) | A1 | 2 (not 3-6) | chi0 hash; sub-case-grid unit test |
| 8 | Physics tail: `sigma_at_dft_energies` wiring at `gw_jax.py:649` (mine), `static_limit` + `Wc0` on `PPMBuildResult` (data seam, not config) | me / physics | 3 | new flag gates; golden gates with flags off |

Rows 5-7 are parallelizable across branches (disjoint files after row 4). Rows 0-4
are strictly serial. Row 8 is the only behavior change and defaults off.

My proposal's §3 contract, §3b-3f edits, and §5 gate table otherwise stand as
written, with the two amendments above (quad type, Bug-B precedence).
