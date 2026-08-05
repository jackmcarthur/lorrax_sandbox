# Agent 3 round 2 — critique + revision

I've now read `agent_1.md`, `agent_2.md`, `agent_4.md`.  The
biggest lesson: my v1 §5 plug-in had a 10× arithmetic error in
`B_persist` (I forgot to divide by `p_y` / `p_x` on each centroid
copy).  All three other agents got that right.  This Round-2 doc
makes the corrections explicit and consolidates what's still open.

---

## 1. Cross-reading notes

### 1.1  agent_1.md

**Stronger than my v1.**
- §1c spells out the **rank-7 reshape as a bitcast alias** of the
  rank-5 P-pair buffer.  I had three concurrent rank-5 slots but
  did not note that `P_l_3d` shares storage with `P_l`.  This
  changes the count of *distinct* slots vs *concurrent live*
  buffers — the magic-3 is concurrent live, not allocated.
- §1f gives explicit numbers for the unsharded ψ_G box:
  `83 GB` at `bc=16, kc=nk` and `14 GB` at `bc=16, kc=6`.  My v1
  carried only the symbolic formulas and didn't realise
  `bc/P = 1` at CrI3 4×4 so the sharded path needs `bpd`
  semantics.
- §2 explicitly separates **sum vs alias** with concrete XLA
  BufferAssignment rules.  My v1 said "aliasable" qualitatively;
  agent_1 gives the matching-shape requirement and the
  donation-as-source-of-alias mechanism.
- §5b candidate enumeration ("six explanations") for why
  empirical is 12,500 — better-organised than my v1 §5.5.
- §5b candidate-5 catches that `plan_gflat_chunks(max_chunks=64)`
  forces `cr ≥ ⌈n_rtot / 64⌉ = 17,578` for CrI3 — so the
  empirical 12,500 **cannot** be the gflat planner output.  I
  missed this entirely.
- §6 cleanly tells which model to retire (`compute_optimal_chunks`),
  evolve (`gflat_memory_model`), and keep as sanity check
  (`aot_memory_model/chooser`).

**Weaker / wrong.**
- §3c definition of `α_zeta = (pair_density_slots + 1) · …`
  conflates the "3 rank-5 slots" with "1 more for Z_q".  Z_q is
  rank-3 (16·nq·μ·cr/P), not rank-5; the `+1` is wrong by a
  factor of `ns² · μ / (nq)` — at CrI3 the ratio is `4·1800/36 =
  200×`, so adding "+1 rank-5 slot" for Z_q is order-of-magnitude
  wrong.  See my §3 disagreement.
- §5a uses `nb_L = nb_R = nb_full ≈ 400`, treating `nb_full` as
  the sum.  Per agent_2 §6 and code (`gw_init.fit_zeta:588`),
  `nb_total_chunker = (b3-b0) + (b4-b1) ≈ 2·nb_F` — i.e. the
  centroid sum is double what `nb_full` reports.  Agent_1's
  `B_centroids_total ≈ 0.83 GB` is therefore ~2× under.

**New (not in my v1).**
- The `n_bc · α_psi_Y_bc · cr` "cumulative reshard slab across
  Python unroll" is treated as **schedule-dependent** and quotes
  measured 17 GB miss on CrI3 at `n_bc=5`.  My v1 §1.4 noted
  per-bc concatenation but didn't trace it back to the
  17-GB-miss in `_fft_moment` docstring.
- The 1-D mesh failure mode in §7.11 — I hadn't considered that
  cuSolverMp falls back when `p_x = 1` or `p_y = 1`.
- Maintenance-cost framing of the AOT DoE infrastructure as a
  "footprint maintenance burden" — useful evaluative criterion.

### 1.2  agent_2.md

**Stronger than my v1.**
- §6 catches a concrete bug in `gflat_memory_model._peak_C_fit_one_rchunk`:
  the `centroids_persist` formula uses `2 · _bytes_c128(nk, ns,
  mu, nk, shard=p_xy)` — `nk` appears twice (as the leading axis
  and again as a stand-in for `nb_total`), with `shard=p_xy`
  instead of separate `p_x` / `p_y`.  This is a **source-code
  bug**, not just a modelling gap.  My v1 §6 noted the planner
  used `meta.n_rmu` once and missed bispinor; I didn't read
  `_peak_C` carefully enough to catch the nk-twice typo.
- §5 candidate-enumeration includes "Possibility 5 = unsharded
  box NOT fully aliased into a freed pair-density slot" with
  numbers (`14 GB`) — concretely closes one of the gaps that my
  v1 §5 left open.
- §3a `psi_Y_full / rank-5` ratio computation (`nb_F /
  (3·ns·μ) = 0.037 at CrI3 charge`) — gives a clean reason to
  drop the term.  My v1 kept it in the budget; agent_2 shows it's
  3.7% of W_zeta.

**Weaker / wrong.**
- §5 plug-in computes B_persist = 4.91 GB but only counts L+R
  *once* (single `psi_rmu_Y` + single `psi_rmuT_X`).  Per
  agent_1 and agent_4 there are **four** copies (L on Y, L on X,
  R on Y, R on X) — agent_2 under-counts B_persist by a factor
  of 2 on the centroid terms.  Within "negligible at CrI3" so
  the conclusion is unchanged, but the formula is off.
- §3 W_C(cr, bc, kc) formula `+ max(0, W_Yfull(cr) − slot_overlap)`
  — `slot_overlap` is named but never defined.  Either drop the
  term (agent_2 §3a does drop it later) or define `slot_overlap`
  = `one_rank5_slot` size.  My v1 had the same problem (kept
  `psi_Y_full` as an additive term).
- §5 says "Possibility 7 — n_bc × 3 slots inside the shard_map"
  then correctly dismisses it.  Detour adds noise.

**New (not in my v1).**
- §3c notes cuSolverMp scratch is ~`O(mu²)` per panel and gives
  a numerical bound at CrI3: `16·1800²/16 ≈ 3.2 MB` — small.
  My v1 left scratch as a "k_factor ∈ [2,4]" unknown.  Agent_2
  is closer to right (it really is small at MoS2/CrI3 sizes; the
  worry is asymptotic).
- §7.10 "are `psi_rmu_Y` and `psi_rmuT_X` really two copies?" —
  same question as my Q-implicit but agent_2 gives a concrete
  resolution path (check JAX view aliasing of the load).
- §7.11 "`target_utilization` interpretation" — frames the
  0.80 / 0.95 / 1.0 choice as "lands at the budget edge without
  OOM in query_fft_peak_bytes calibration runs".  My v1 treated
  η as a hand-tuned constant; this is the right design.

### 1.3  agent_4.md

**Stronger than my v1.**
- §2.b sharply identifies the L3 ↔ L4 separation via
  `zeta_chunk.block_until_ready()` at `isdf_fitting.py:2152` —
  this is the single clearest argument for `peak = max(C, D)`
  rather than `sum(C, D)`.  My v1 §2.4 had the right intuition
  but missed the explicit sync.
- §3a defines `B_persist` with the correct **per-axis** shard
  divisors (`/ p_y` on `psi_rmu_Y`, `/ p_x` on `psi_rmuT_X`,
  `/ P` on `L_q` and `gflat_acc`).  My v1 §3.1 also had this,
  but agent_4's table form is cleaner and matches code line
  numbers.
- §5b explicit r_chunk_max ≈ 75,000 at CrI3 with the correct
  small `B_persist ≈ 0.93 GB`.  My v1 got 23,600 from a
  10×-too-large `B_persist`.  Agent_4 is right.
- §5d shows `cs = 64` is "wildly under" the budget at the
  empirical r_chunk = 12500 (W_accum = 5 GB ≪ W_pool = 55 GB),
  i.e. there's massive slack in the empirical config.  My v1
  attributed the cs=64 to F_fft >> 4; agent_4's reading is more
  parsimonious (the empirical setting is conservative, not
  binding).
- §5.e MoS2 sanity check (`r_chunk → cap at n_rtot, one chunk`)
  is a useful cross-validation point.  My v1 didn't include MoS2.
- §7.10 "Cohsex.in override interaction: who wins?" — concretely
  notes that the empirical config explicitly **overrides three
  of four knobs**, so the comparison "empirical 12,500 vs my
  prediction" is **not apples-to-apples**.  This is a critical
  point that recasts the entire "why doesn't my formula land at
  12,500" discussion.

**Weaker / wrong.**
- §5.a uses `nq_disk ≈ 7` for charge IBZ at CrI3 6×6×1.  For a
  6×6×1 monkhorst grid, |IBZ| depends on the spatial point
  group; for a hexagonal 6×6 grid the IBZ is typically 7 or 13
  (depends on the specific group), so 7 is plausible but not
  given.  CONTEXT says `n_k = n_q = 36`; the IBZ reduction
  shrinks this for `nq_disk` but the exact value isn't pinned.
  Same issue in my v1 (Q9).
- §3.b `W_solve = (1 or 2) · B·nq·μ·r_chunk/P` — the "(1 or 2)"
  branch on cuSolverMp vs shard_map fallback is conceptually
  right, but at CrI3 distributed scale the cuSolverMp path is
  default (per `report.md §2b.4`), so it's (1).  Branching
  hint is good but not actually deployed.
- §5.b possibility-5 says "`S_pd_effective ≈ 5–6` at CrI3 (cuFFT
  scratch on the rank-7 IFFT/FFT)".  Plausible but speculative;
  same conjecture as my v1's "F_fft ≈ 5" route, with the same
  lack of evidence.

**New (not in my v1).**
- §6.b spots the **50/50 ceiling vs `W_pool/S_pd` ceiling**
  debate explicitly — this is `gflat_memory_model.plan_gflat_chunks:297`
  hard-coding `0.5 * target` vs report §5.5 advocating
  `W_pool / pair_density_slots`.  My v1 §6.1 noted the split
  exists but didn't trace it to the line number.
- §7.10 the override-vs-auto interaction (cited above).
- §6 "Three half-rights → one whole right" sentence is the
  cleanest synthesis: keep gflat_memory_model's A/B/C/D shape,
  add aot_memory_model's per-channel μ, keep
  compute_optimal_chunks's `query_fft_peak_bytes` integration.

---

## 2. Revisions to my v1

**R1.  §5 arithmetic error in `B_persist`.**  My v1 §5.1 wrote
`P1+P2 = 16·36·1800·400·2 / 4 = 10.4 GB`.  Correct value is
`830 MB / 4 = 208 MB`.  I miscomputed `5.184e7 / 4`.  All four
centroid copies together come to ~0.83 GB (not 20.8 GB).
B_persist for CrI3 charge ≈ **1.2 GB**, not 25.4 GB.  Every
downstream conclusion ("cr ≈ 23,600", "F_fft ≈ 5 explains gap")
falls.  *Evidence:* agent_1 §5a, agent_2 §5, agent_4 §5a all land
near 1 GB.  *On reflection:* my arithmetic was sloppy and I did
not spot-check against the order-of-magnitude
`B/total_GPU ≈ 25/60 ≈ 0.4` which would have been a red flag —
empirical CrI3 peak is 28 GB, of which most is the chunk-loop
peak, not persistent.

**R2.  `cr` from corrected `B_persist` is in the 50–75k range,
not 12,500.**  With the right B_persist (~1.2 GB) and
W_pool ≈ 56 GB at η=1: `K1 = 3.73e5 bytes/cr` (my v1's value
is correct) → `cr_max ≈ 56e9 / 3.73e5 ≈ 150,000`.  Even with
the unsharded FFT box subtracted (W_wfn ≈ 13.8 GB) → `cr_max
≈ 113,000`.  My v1 conclusion "F_fft ≈ 5 closes the gap" is
wrong — the gap was an arithmetic error, not an F_fft issue.

**R3.  The empirical 12,500 may not be planner-derived at all.**
Per agent_4 §7.10, cohsex.in `band_chunk_size = 16,
gflat_chunk_size = 64, psig_k_chunk_size = 6` are **manual
overrides**; only `r_chunk_size = 0` lets the planner pick.  And
per agent_1 §5b candidate-5, the `gflat_memory_model.max_chunks=64`
floor forces `cr ≥ ⌈n_rtot/64⌉ = 17,578` — so 12,500 cannot come
from `plan_gflat_chunks`.  Either:
- (a) `compute_optimal_chunks` is still the active picker for cr
  in this code path, and its 5-stage moment inversion lands at
  ~12,500 because its `_zct_moment` counts 5 slots (2 + 3) and
  the `_fft_moment` adds the `n_bc · α_psi_Y_bc · cr` term that
  `gflat_memory_model` is missing; or
- (b) a manual override.  *Evidence:* agent_1 §7.1 makes the
  same point.  *On reflection:* my v1 §7 Q-on-12,500 was
  correctly flagged as an unknown, but I should have read
  `gw_init.fit_zeta` more carefully to see which planner's
  `chunks['chunk_r']` actually flows to `fit_zeta_to_h5`.
  Reading `gw_init.py:574–626` again: `_apply_aot_chunk_model`
  may override `chunks['chunk_r']` if `cfg.memory.use_aot_chunk_chooser=True`;
  then `plan_gflat_chunks` overrides it *unconditionally*
  (`gw_init.py:617: chunks['chunk_r'] = int(gflat_plan.r_chunk)`)
  — so the gflat planner always wins.  Therefore the empirical
  12,500 must reflect `plan_gflat_chunks`'s output, which means
  either the `max_chunks=64` floor doesn't bind (e.g.
  W_pool/α_zeta < 17,578) OR I'm mis-reading the floor.  This
  contradicts my "either (a) or (b)" framing — needs another
  pass to resolve.  Promoted to §4 open question.

**R4.  Concurrent rank-5 *distinct* vs *live*.**  My v1 §1.4
listed 4 distinct rank-5 buffers (Z1 P_l, Z3 P_l_R, Z4
P_l_R_conj, Z5 P_r→P_r_R) but said "3 concurrent slots at peak".
Agent_1 §1c clarifies that `P_l_3d` is a **bitcast alias** of
`P_l` — same physical buffer.  So the chain is:
`P_l → (bitcast) P_l_3d → (new) P_l_R → (new) P_l_R_conj` plus
the `del P_l, del P_l_3d, del P_l_R` calls free intermediates
deterministically.  At the γ̃-contract step, **only `P_l_R_conj`
and `P_r_R` plus one XLA scratch** are live, hence S=3.  Update
my v1 §1.4 catalog: Z2 is alias of Z1, Z3 is new, Z4 is new
*replacing* Z3, Z5 mirrors L-side.

**R5.  `nb_total` in centroid B_persist is `nb_L + nb_R`, not
`nb_full`.**  Per agent_2 §6 + code (`gw_init.fit_zeta:588`),
`nb_total_chunker = (b3 - b0) + (b4 - b1)`.  For symmetric GW
this is ≈ `2·nb_full` (overlapping ranges).  My v1 §1 catalog
used `nb_L` and `nb_R` separately for X-fit and Y-fit — correct.
But my §5 plug-in used `nb_full = 400` once, which under-counts
when `nb_L + nb_R = 2·nb_full`.  Corrected: centroid sum at
CrI3 ≈ `4 × 208 MB ≈ 0.83 GB` (still small).

**R6.  `B_persist` is dominated by gflat_acc, not centroids.**
My v1 §5 had the centroid term at 20.8 GB and gflat_acc at
4.5 GB.  Corrected: centroids ≈ 0.83 GB, gflat_acc dominates at
0.4–4 GB depending on `n_q_disk · ngkmax`.  At full-BZ bispinor
transverse channels with `n_q_disk = 36, ngkmax = 67500`,
gflat_acc ≈ 4.4 GB.  Charge IBZ with `n_q_disk = 7`,
gflat_acc ≈ 0.85 GB.

**R7.  cuSolverMp scratch is small at CrI3.**  My v1 §1.3 left
this as `[2, 4]` factor × `nq·μ²/P` (~unknown).  Agent_2 §3c
gives the numerical bound at CrI3: `16·1800²/16 ≈ 3.2 MB`
per rank for the dense factor, with workspace of the same order.
Negligible.  Keep as flag for asymptotic safety but stop
expecting it to close the cr-prediction gap.

**R8.  My §5 "F_fft ≈ 5" inference is unsupported.**  Was a
fudge to close a gap that didn't exist (it was an arithmetic
error).  Drop the inference; keep the open question of per-call-
site F_fft but without the fake numeric calibration.

**R9.  The `max_chunks=64` floor.**  My v1 §6.2 and §7
discussed planner shape but didn't note this floor.  Per
agent_1 §5b/§7.9 and `gflat_memory_model.py:334`
(`r_chunk = max(r_chunk, math.ceil(n_rtot / max_chunks))`), the
floor pegs `cr ≥ n_rtot / 64` ≈ 17,578 at CrI3.  This is opaque
and probably wrong — drop it or make it adaptive.

---

## 3. Where I still disagree

**D1.  agent_1 §3c: `α_zeta = (pair_density_slots + 1) · …`**

*Agent_1's claim (§3c):* the linear-in-cr coefficient is
`(S+1) · 16·nk·ns²·μ/P`, adding "+1" for the Z_q transient.

*My counter-claim:* Z_q has shape `(nq, μ, cr)` not `(nk, ns², μ,
cr)`.  Per-rank bytes: `16·nq·μ·cr/P`.  Compared to one rank-5
slot `16·nk·ns²·μ·cr/P` at `nq = nk`, the ratio is `1/ns² = 1/4`
for bispinor.  And Z_q's lifetime is *after* the γ̃-contract and
FFT, when the rank-5 slots are dead — so it aliases into a freed
slot rather than adding.  Correct coefficient: just
`S · 16·nk·ns²·μ/P` (no +1).

*Evidence to resolve:* HLO dump of `z_q_from_psi_sm` —
specifically, look at the BufferAssignment for the post-FFT
phase.  Does Z_q occupy its own slot or alias a freed rank-5?
For CrI3 4×4 mesh, dump from
`module_*.jit__kernel.sm_*.memory-usage-report.txt` and count
slots whose lifetime extends past the γ̃-contract.

**D2.  agent_4 §3a: "balanced mesh balances both copies to
`/√P`".**

*Agent_4's claim (§3a, sentence "For a balanced mesh p_x = p_y
= √P both copies cost B·nk·μ·nb·ns/√P each"):* sum of
X-sharded + Y-sharded copies has divisor `√P`.

*My counter-claim:* the X-sharded copy is `/p_x`, the Y-sharded
copy is `/p_y`.  Their **sum** is `B·nk·nb·ns·μ·(1/p_x + 1/p_y)`
= `B·nk·nb·ns·μ·(p_x + p_y)/(p_x · p_y)`.  At `p_x = p_y = 4`,
this is `2/4 = 1/2 = 1/√P` only if I read the formula loosely.
For a 1×16 mesh (`p_x = 1, p_y = 16`), sum is `(1 + 1/16) · B·…
≈ B·nk·nb·ns·μ` (no shard, because the X-sharded copy is
unsharded).  Agent_4's "balanced" qualifier saves it for the
4×4 case but the formula is misleading.  My v1 §3.1 had the
correct per-axis formula.

*Evidence to resolve:* trivial — just read the
`load_centroids_band_chunked` PartitionSpec output (it's literal
`P(None, None, None, 'y')` and `P(None, 'x', None, None)` —
no "balanced" assumption).

**D3.  Whether the `gflat_memory_model.max_chunks=64` floor
binds at CrI3.**

*Implicit claim in agent_1 §7.9:* the floor binds (`r_chunk ≥
17,578`), so empirical `12,500` cannot come from this planner.

*My counter-claim:* I haven't verified the floor is actually
binding.  Reading `gflat_memory_model.py:332–342`:

```python
r_from_budget = (int(headroom_C / α_C) if α_C > 0 else n_rtot)
r_chunk = max(r_lo, min(r_hi, r_from_budget))
r_chunk = max(r_chunk, math.ceil(n_rtot / max_chunks))
r_chunk = min(r_chunk, n_rtot)
```

The line `r_chunk = max(r_chunk, ceil(n_rtot / 64))` only
applies *after* `r_from_budget` is computed.  If the budget
allows `r_from_budget = 15,000`, the max with `17,578` upgrades
it.  But if `r_from_budget = 50,000`, the floor is silently
satisfied.  So at CrI3 we'd expect `r_chunk = max(50000, 17578)
= 50,000`, not 12,500.  The floor only binds at *very tight*
budgets where `r_from_budget` is itself small.

So the empirical 12,500 cannot come from this code path
(`r_from_budget` is much larger; the floor's max doesn't
*reduce* anything).  **Agreement with agent_1's conclusion**:
12,500 is not from `plan_gflat_chunks`.

*Evidence to resolve:* one-line dump from the actual CrI3 run
log.  Specifically the lines emitted by `gflat_plan.format()`
at `gw_init.fit_zeta:615` and the
`compute_optimal_chunks` print at `:557`.

---

## 4. Consolidated open questions

Cross-referencing all four drafts:

**Q-1.  Where does empirical r_chunk = 12,500 actually come
from?**  (my v1 §7-implicit; agent_1 §5b/§7.1; agent_2 §5
"Possibility list"; agent_4 §7.10).  Three competing hypotheses:
- (a) `plan_gflat_chunks` with hidden tight constraint that
  agents are missing.
- (b) `compute_optimal_chunks` legacy chunker (but `gw_init.py:617`
  unconditionally overrides — see my R3).
- (c) The empirical "12500" in CONTEXT §4 is a stale recipe
  number, not the auto-pick at the current code state.

This is **the single most load-bearing unknown**.  Without
resolving it, no model can be validated.

**Q-2.  `pair_density_slots` at CrI3 scale: 3, 5, 7, or
something else?**  (my v1 §7 Q10; agent_1 §7.2; agent_2 §7.1;
agent_4 §7.3).  Currently sourced from a MoS2 3×3 bispinor HLO
dump.  All four agents agree this is the model's biggest
fragility.  Either the constant is still 3 at CrI3 scale (in
which case there must be other terms I'm missing for the
prediction to land at 12,500), or it's larger at scale (cuFFT
scratch on a 75×75×200 rank-7 box).

**Q-3.  Unsharded FFT box — root cause and detection.**  (my
v1 §7 Q1; agent_1 §7.3 + §1f; agent_2 §7.5; agent_4 §7.2).
All four flag this.  The fix is at the buffer-creation site in
`to_rchunk` / `psi_G_store.fetch_psi_rchunk`.  Until fixed, the
planner must conservatively assume unsharded above some
`n_rtot · nk` threshold (agents differ on the threshold).

**Q-4.  cuSolverMp internal scratch.**  (my v1 §7 Q3; agent_1
§7.4; agent_2 §7.3; agent_4 §7.4).  Agreed across agents that
this is at most O(μ²/P) and unlikely to dominate at current
sizes.  Worth a one-shot HLO probe; not blocking.

**Q-5.  `fft_factor = 4.0` site-dependence.**  (my v1 §7 Q2 +
Q7; agent_1 §7.6; agent_2 §7.4; agent_4 §7.5).  Three call
sites (loader / fit jit / accumulate jit) have different cuFFT
neighborhoods.  Per-call-site F_fft is the §5.10 follow-up.

**Q-6.  ngkmax estimate before sphere construction.**  (agent_1
§7.7; agent_2 §7.7; agent_4 §7.9).  `gw_init.fit_zeta:596`
falls back to `0.06·n_rtot` because the real sphere is built
inside `fit_zeta_to_h5`.  Hoist the sphere construction or pass
ngkmax in.

**Q-7.  Bispinor per-channel re-picking.**  (my v1 §7 Q5;
agent_4 §4 "bispinor handling"; agent_2 §4 step 5).  Currently
unclear whether `plan_gflat_chunks` is called once with
`n_rmu = n_rmu_C` and reused, or per channel with
`n_rmu_chan`.  Code read needed:
`gw_init.fit_zeta` is called once per
`vertex_mu_L`?  If yes, `plan_gflat_chunks` runs per channel and
picks chunks from `meta.n_rmu` which is per-channel.  If no, it
runs once for charge and re-uses for transverse.  The bispinor
4-channel orchestrator (presumably in a higher-level driver)
controls this.

**Q-8.  `psi_Y_full` aliasing — does it sum or alias inside
the fused jit?**  (my v1 §7 Q6; agent_1 §2a; agent_2 §3a +
§7.2; agent_4 §2.c).  All four say "negligible at CrI3" but
flag it as unverified.  agent_2 quantifies the ratio at 3.7%.

**Q-9.  `max_chunks = 64` floor — what is it for?**  (agent_1
§7.9; my §3 disagreement D3).  No obvious physical motivation.
Likely an arbitrary cap to prevent runaway chunk counts at
small problems; at CrI3 it doesn't bind (per D3).  Drop or
make budget-derived.

**Q-10.  `target_utilization` = 0.80 vs 0.97 vs 1.0.**  (my v1
§5 implicit; agent_2 §7.11).  `gflat_memory_model` uses 0.80;
`compute_optimal_chunks` uses 0.97; the §5 reference algorithm
uses no slack.  Right answer is "whatever lands at the budget
edge without OOM in calibration".

**Q-11.  IBZ vs full-BZ `n_q_disk` for bispinor transverse.**
(agent_4 §7.7).  `write_ibz_only=False` for transverse μ_L>0
currently — so `gflat_acc` is 5× larger for those channels
than for charge.  Not in the reference doc explicitly; agent_4
catches it.  Adds ~1.5 GB extra to W_pool budget on each
transverse channel.

---

## 5. Recommended next steps

Each item: *what action, on what artifact, expecting what result.*

**N1.  Dump and read a CrI3 6×6 80 Ry HLO memory-usage report
for `fit_one_rchunk`.**
- *How:* set
  `XLA_FLAGS="--xla_dump_to=/tmp/hlo_cri3 --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.* --xla_gpu_dump_xspace_to=/tmp/hlo_cri3"`
  (or via `jax.config.update('jax_compilation_cache_dir', …)`
  + the dump flags), run one r-chunk's worth of `fit_zeta_to_h5`
  on the 16-GPU CrI3 reference, then `grep -c "c128\[" /tmp/hlo_cri3/module_*.jit__kernel.memory-usage-report.txt`
  for the rank-5 pair-density-shape lifetime slots.
- *Expected:* either confirm `pair_density_slots = 3` at this
  scale, or surface a larger number that explains the
  empirical 12,500.  Resolves Q-2 and informs Q-1.

**N2.  Read the CrI3 run log line emitted by
`gflat_plan.format()` at `gw_init.fit_zeta:615` for the
`A_ffi_boundary_cusolvermp_profile_2026-05-13` (or any recent
CrI3) run.**
- *How:* check `runs/MoS2/00_mos2_3x3_cohsex/A_*_cri3_*/lorrax.log`
  for "G-flat memory model — chunk plan + HWM estimate" and
  record `(r_chunk, n_r_chunks, bottleneck, HWM, peak
  breakdown)`.
- *Expected:* the planner's actual pick at CrI3.  If it's
  ≈ 12,500 then `plan_gflat_chunks` *is* the source; if it's
  larger then a downstream override is in play.  Resolves Q-1.

**N3.  Grep `to_rchunk` for the FFT box buffer allocation site
and check whether its sharding annotation propagates.**
- *How:* `grep -n "_box_kernel\|jnp.zeros\|jnp.fft.ifftn" sources/lorrax_A/src/common/wfn_transforms.py`,
  read lines 380–430, identify the buffer that materialises the
  `(k_chunk, bpd, ns, nx, ny, nz)` shape.  Then dump HLO for one
  `to_rchunk` call (use `psig_k_chunk_size=nk` for unmitigated
  case) at CrI3 sizes and check whether the buffer has a
  per-rank or replicated allocation.
- *Expected:* either confirm the unsharded materialization (and
  identify which op forces it — likely the `jnp.zeros` for the
  pad buffer or the `dynamic_update_slice` scatter) or rule it
  out.  Resolves Q-3.

**N4.  Fix `gflat_memory_model._peak_C_fit_one_rchunk.persistent.centroids_persist`
typo and re-run plan.**
- *How:* line 184 currently reads
  `2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy)`.  Should be
  `_bytes_c128(nk, ns, mu, nb_total, shard=p_y) +
   _bytes_c128(nk, mu, nb_total, ns, shard=p_x)` (per-axis
  shards, with nb_total = nb_L + nb_R).  Patch and rerun the
  CrI3 chunk-plan log line.
- *Expected:* B_persist for CrI3 drops from "however much the
  bug currently reports" to ~1 GB; `r_chunk_max` increases
  proportionally.  May explain why the planner currently picks
  small values.  Tests Q-1 from a different angle.

**N5.  Read `gw_init.fit_zeta` for per-channel re-picking.**
- *How:* `grep -n "plan_gflat_chunks\|compute_optimal_chunks\|vertex_mu_L" sources/lorrax_A/src/gw/gw_init.py`.
  Check whether `fit_zeta` is itself called inside a μ_L loop or
  whether it loops internally.
- *Expected:* confirm or deny per-channel re-picking.  If
  per-channel, the planner uses each channel's `meta.n_rmu`
  (assuming the centroid loader updated it); if once, it uses
  whatever was set at problem-setup time.  Resolves Q-7.

**N6.  Add a planner regression test.**
- *How:* in a new test file under `sources/lorrax_A/tests/`,
  build a stub `Meta` matching CrI3 6×6 80 Ry geometry and
  call `plan_gflat_chunks` with `budget_gb=60`.  Assert
  `12,000 ≤ r_chunk ≤ 100,000` (wide band initially, narrow
  after N1/N2 calibrate).  Also assert
  `B_persist ≤ 5 GB` and `gflat_plan.hwm_bytes ≤ 60 GB`.
- *Expected:* a fail-loud guardrail that catches future drift.
  Concrete instance of "Calibration test" from §5.10 of the
  reference doc.

---

Agent 3 round 2 done
