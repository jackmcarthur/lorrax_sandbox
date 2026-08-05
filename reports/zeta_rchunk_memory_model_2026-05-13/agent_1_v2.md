# Agent 1 — round 2 critique + revision

Cross-read of `agent_2.md`, `agent_3.md`, `agent_4.md` against my
`agent_1.md`. Where citations like "agent_3.md §5.3" appear, the §
numbering refers to that agent's own section structure.

---

## 1. Cross-reading notes

### Agent 2 (`agent_2.md`)

- **Stronger:** §1 tensor table is more line-number-referenced
  than my §1 (every entry pinned to `isdf_fitting.py:NNN`). §6
  catches a concrete bug in `gflat_memory_model._peak_C_fit_one_rchunk`
  that I missed: the centroid term uses `nk` in *both* the band-count
  and the load-axis position (`2·c128(nk, ns, mu, nk, …)`), an
  apparent typo for `nb_F` — confidence-shaking for the model's
  current output. §5 explicitly tabulates the 5 candidate
  explanations for the r_chunk=12500 mismatch in a way mine
  conflates.
- **Weaker/wrong:** §3a's claim that `psi_Y_full` is "3.7% of rank-5
  slots, negligible" mirrors my §5 conclusion but is derived
  without showing the size math — verifiable. §5 lands at
  r_chunk ≈ 70,700 vs my 53,800 — agent_2 used `mu = 1800` not
  padded to `1808` and `target_utilization = 1.0`; my number used
  utilization-corrected B. Both are wrong in the same direction by
  the same factor.
- **New:** the `psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)`
  observation (§2 Tier 2). I didn't catalogue this tensor at all.
  Agent 2 argues it's the cumulative sum of per-bc reshard slabs,
  alive concurrently because the concat needs them all — same
  underlying quantity as my "n_bc · α_psi_Y_bc · r_chunk" but at
  rank `[nk, nb_F, ns, cr]/p_y` rather than my per-bc accumulation
  formula. Agent_2's formulation is cleaner. Also flags the
  `nb_total_chunker = nb_L + nb_R` issue in `gw_init.fit_zeta:588`
  — the planner sees `~2·nb_F` not `nb_F`, which agrees with my
  §3a but agent_2 makes it explicit.

### Agent 3 (`agent_3.md`)

- **Stronger:** §5.3 numerical walk is the most disciplined of any
  agent — derives `K0`/`K1` separately, computes `α_zeta`,
  `α_psi_Y`, `α_zcol`, and back-solves for what `F_fft` would have
  to be to land at 12500. Concrete answer: `F_fft ≈ 5`, not 4.
  This is a *measurable* prediction. §1.4 explicitly distinguishes
  the four pair-density-shape buffers (Z1, Z3, Z4, Z5) with
  per-rank bytes and notes XLA's allocator must keep 3 of them live.
  §2.5 aliasing table is the right compact summary; I should have
  written one. §6 has a per-model rubric (right / wrong) that is
  more disciplined than mine.
- **Weaker/wrong:** §1.1 counts ψ centroids at `nb_L + nb_R`
  shard=`p_x` (or `p_y`) — agrees with mine after I also corrected
  for the dual-copy. But §5.1 plugs `P5 = 16·36·1800²/16 ≈ 0.12 GB`
  which uses `n_rmu = 1800` (logical) not `n_rmu_padded ≈ 1808` —
  a 1% rounding error, fine. The Q4 claim that the `r_chunk ≥
  n_rmu` floor is "redundant / wrong" is contestable; the floor
  comes from "don't pay per-iter overhead for less than `n_rmu`
  units of work", an *algorithmic* argument, not a memory one —
  see §3 below.
- **New:** Q1's alternate hypothesis — the "unsharded" framing may
  be a *misdiagnosis*: the FFT box may always be a per-rank
  dense `(k_chunk, bc/P, ns, nx, ny, nz)` box with `n_rtot`
  positions per rank (rank-local, not globally-replicated).
  If so, there is no "factor of P" — the box is just locally dense
  in the r-axis, as it must be for a local cuFFT. **This is a
  significant alternative reading of report.md §5.8** that nobody
  else proposed. Also: explicit per-channel re-pick verification
  (Q5) and HLO-dump pipeline question (Q10).

### Agent 4 (`agent_4.md`)

- **Stronger:** §4 picker is the cleanest pseudocode of any agent
  — actually compilable. The `sharded_fft_box_holds(meta, mesh,
  bc)` gate is a defensible API: makes the unsharded pathology a
  boolean, not a continuous fudge. §5.b is the most agnostic about
  the 12500 mismatch — lists 5 explanations and explicitly says
  "I'd want an HLO memory-usage-report to resolve this".
  §7.10 is the single biggest insight in any agent's report:
  **the empirical CrI3 config froze `band_chunk = 16,
  gflat_chunk_size = 64, psig_k_chunk_size = 6` via cohsex.in
  overrides**, leaving only `r_chunk_size = 0`. So r_chunk = 12500
  is the planner's pick *conditional on those constraints*. My
  v1 §5b/§7.1 missed that the comparison is constrained, not
  free.
- **Weaker/wrong:** §5.a is sloppy about ngkmax — agent_4 uses
  30,000 (a guess), getting `B_persist ≈ 0.93 GB`, way smaller
  than mine (2.17 GB) and agent_3's (25.4 GB at ngkmax=70k). The
  spread reflects ngkmax uncertainty — agent_4 underweights this.
  §5.b lands at `r_chunk ≈ 75,664` (≈ mine) but uses
  `target_utilization = 1.0` and `n_q_disk = 7`. §3.b's claim that
  `W_solve = (1 or 2) · B · nq · μ · r_chunk / P` collapses the
  cuSolverMp vs shard_map cost difference into a 2× factor, which
  is too coarse — the difference in donation patterns has been
  measured at 2× elsewhere (e.g. Si 4×4×4 31→16 GB per device),
  not just on the Z_q reshard.
- **New:** the cohsex.in-override observation in §7.10 (above).
  §7.6 confirms via line-number reference that `ζ_chunk` is alive
  during `_kernel` (`isdf_fitting.py:2181` for `del zeta_chunk`).
  §7.7 notes that bispinor μ_L > 0 currently uses `write_ibz_only
  = False`, so `nq_disk = nq_full = 36` for transverse channels —
  the persistent `gflat_acc` is ~5× larger for transverse than for
  charge at CrI3. I didn't make this asymmetry explicit.

---

## 2. Revisions to my v1

- **`agent_1.md §1a` — missed `psi_Y_full` term.** All other agents
  (agent_2 §1, agent_3 §1.4, agent_4 §1.c) catalogue this as a
  separately-tracked tensor of shape `(nk, nb_full, ns, cr) /p_y`
  built by `jnp.concatenate(psi_Y_parts, axis=1)` before
  `z_q_from_psi_sm` consumes it. Agent_2 §3a shows it's ~3.7% of
  rank-5 size at CrI3 charge — negligible *in this regime* but not
  zero. My v1 folded this into a hand-wave "n_bc · α_psi_Y_bc"
  unroll cost without making the post-concat buffer a distinct
  tensor. **New view:** add `psi_Y_full = 16·nk·nb_full·ns·cr/p_y`
  as a discrete entry, aliased into one pair-density slot only if
  XLA's BufferAssignment confirms — otherwise summed. Evidence:
  `isdf_fitting.py` r-chunk loop body, confirmed by all three
  agents.

- **`agent_1.md §5b` — apples-to-apples error.** My validation
  compared my picker's pick (`r_chunk ≈ 53,800`) to the empirical
  12500 without noting that the empirical config explicitly froze
  `band_chunk=16, psig_k_chunk_size=6, gflat_chunk_size=64` via
  cohsex.in. Agent_4 §7.10 makes this explicit: with three knobs
  frozen, the planner's pick of `r_chunk = 12500` is conditioned
  on those values, not the free optimum. **New view:** to validate
  the model honestly, re-run the picker constrained to
  `(bc=16, kc=6, cs=64)` and see what r_chunk falls out. The
  unsharded W_wfn at `(bc=16, kc=6)` is ~13.8 GB; that's the
  dominant constant in K0, not a slot-budget concern. With ngkmax
  near 30k (agent_4) vs 70k (agent_3) the prediction can shift by
  ~1.5 GB on `B_persist`.

- **`agent_1.md §1f` / §7.3 — "unsharded" framing may be wrong.**
  Agent_3 Q1 raises the possibility that the FFT box isn't
  "unsharded across ranks" but rather "dense in the r-axis on each
  rank by design" — every rank doing its own local cuFFT on its
  own per-rank `(k_chunk, bc/P, ns, nx, ny, nz)` box. If so, the
  per-rank bytes formula `16·kc·bpd·ns·n_rtot · fft_factor`
  *without* `/P` is correct, but the framing "factor of P from
  XLA refusing to shard" is misleading; rather it's "the r-axis
  cannot be sharded in a per-rank cuFFT, so the FFT-box r-axis is
  full n_rtot on every rank, by construction". **New view:** I
  should not conflate "unsharded → bug" with "r-axis dense per
  rank → unavoidable". The fix per report.md §5.8 (locate-and-
  shard at the creation site) only makes sense if the box's
  *band* or *k* axis is supposed to be sharded but isn't.
  Evidence: needs HLO dump; my v1 §4d step 1 ("always use the
  unsharded formula") is still the right defensive choice
  regardless.

- **`agent_1.md §6` — missed the `nk`-vs-`nb_F` typo in
  `_peak_C_fit_one_rchunk`.** Agent_2 §6 / agent_3 §6.1 both flag
  `2·c128(nk, ns, mu, nk, shard=p_xy)` at
  `gflat_memory_model.py:184`. If accurate that's a literal bug —
  uses `nk` where `nb_F` was intended. **New view:** confirmed by
  reading `gflat_memory_model.py:184`:

  ```python
  "centroids_persist":
      2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy),  # L+R approx
  ```

  Comment "L+R approx" hints this was a quick stand-in. At
  CrI3 nb_F ≈ 400, nk = 36, so this *under-counts* persistent
  centroids by ~11× and shards on `p_xy` (= P) rather than the
  per-axis `p_x` or `p_y` for each copy — *over-counts* the
  sharding savings by 4× on a balanced 4×4 mesh. Net: probably
  under-counts B_persist by ~3×. Should be filed as a code bug
  (per AGENTS.md, noted in this report, not fixed here).

- **`agent_1.md §7.1` (the 12500 mystery) — narrower hypothesis.**
  After reading agent_3's quantitative back-solve (F_fft = 5 →
  cr_max ≈ 14,200) and agent_4's cohsex-override observation,
  the most likely combination is:
  1. The empirical 12500 was picked by the planner with three
     knobs frozen by user override (cohsex.in),
  2. With those constraints, W_wfn_unsharded ≈ 13.8 GB eats most
     of W_pool,
  3. F_fft is closer to 5 than 4 at the CrI3 1.125 M-cell FFT
     box scale,
  4. cuSolverMp scratch + `psi_Y_full` add ~1–2 GB more,
  5. `max_chunks = 64` floor in `plan_gflat_chunks` may or may
     not bind — I previously claimed it should force
     r_chunk ≥ 17,578, but the floor only applies when the
     budget allows a larger value; under tight constraints
     the budget bound wins.
  Net prediction: my model with these corrections lands near
  agent_3's 14,200, ~13% above 12,500 — well within the
  uncertainty of F_fft. **The model is more right than my v1
  validation made it look.**

- **`agent_1.md §3a` — per-channel `mu` plumbing not asserted.**
  Agent_3 Q5 / agent_4 §4 note that `plan_gflat_chunks` is called
  with `meta.n_rmu` (single value) but the bispinor pipeline
  needs to re-pick per channel because `mu_T < mu_C`. My v1 §4c
  said "always size to charge — transverse fits with slack" which
  is correct but suboptimal: transverse can afford a *larger*
  r_chunk because `α_zeta ∝ mu`. **New view:** the right answer
  is per-channel re-pick (allow transverse to grow r_chunk by
  `mu_C/mu_T ≈ 1.5×`), not "use charge everywhere". Whether the
  *current* code does this needs verifying — agent_3 Q5 says it
  doesn't.

---

## 3. Where I still disagree

- **Agent 3 §Q4** claims the `r_chunk ≥ n_rmu` lower bound from
  `report.md §5.4` and `gflat_memory_model.py:326` is "redundant /
  wrong" because Σ_μν isn't the output of zeta-fit and the
  divisibility floor (`cr ≥ P`) is the only structural
  requirement. **I disagree.** The `r_chunk ≥ μ` floor is an
  *algorithmic* lower bound based on per-iteration overhead, not a
  divisibility one. The point is that each r-chunk pays a fixed
  FFT tax (ψ-fetch IFFT + accumulate FFT), and once
  `r_chunk < μ` the per-r-unit overhead grows above the per-r-unit
  ζ-fit work, so we're losing efficiency. Distinct from the `cr ≥
  P` divisibility requirement. Removing the floor doesn't crash
  the code but it makes the picker pick pathologically small
  r_chunks at the budget edge. **Resolution:** read
  `gflat_memory_model.py:325-326` ("Lower bound: r_chunk ≥ μ (per
  user note — Σ_μν output dominates any savings from finer
  chunking)") in context with `report.md §5.1`'s "the smallest
  possible chunk count". Both rationales support the floor; the
  Σ_μν phrasing is an analogy, not a literal claim about the fit's
  output shape. The floor stays.

- **Agent 4 §5.b explanation 5** ("`pair_density_slots` is actually
  higher under bispinor at CrI3 — cuFFT may force more scratch
  slots on a 75·75·200 rank-7 box") proposes `S_pd_eff = 5–8`,
  giving r_chunk ~ 25–35k. **I disagree this is the dominant
  explanation.** Agent_3's back-solve gets to ~14k by raising
  F_fft from 4 to 5, with S_pd unchanged at 3. Both candidates
  reach the same r_chunk only by coincidence; mechanically they
  pin different terms. **Counter-claim:** S_pd is about
  *concurrent rank-5 buffer lifetime overlap*, which is set by
  the static dataflow of the shard_map body (`P_l_R_conj`,
  `P_r_R`, and one cuFFT/contract scratch). cuFFT scratch *size*
  scales linearly with box size and can absolutely grow at CrI3
  rank-7 dimensions, but it shouldn't add a new *lifetime slot* —
  it changes the bytes of one of the three slots, not the count.
  So F_fft growing 4→5 is mechanistically more likely than S_pd
  growing 3→6. **Resolution:** dump
  `module_*.jit__kernel.memory-usage-report.txt` from a CrI3
  fit_one_rchunk compile, count slots holding pair-density-
  shaped buffers. If it's 3, F_fft is the missing factor; if it's
  5–6, agent_4 is right.

- **Agent 3 §Q1** hypothesises the "unsharded FFT box" framing is
  a misdiagnosis — the box may always be per-rank dense in r,
  not globally replicated. **I disagree this is the right
  reading.** `psig_k_chunk_size = 6` measurably reduces peak
  HBM by ~`(36-6)/36 ≈ 5.5×` on the W_wfn term (from ~83 GB
  unsharded at kc=36 down to ~14 GB at kc=6 — both my numbers and
  agent_3's match). If the box were "per-rank dense in r by
  construction", reducing kc would only linearly reduce the
  *k-axis*-multiplied bytes of one rank's box — but that's
  exactly the relationship we observe. So either reading is
  *consistent with the empirical mitigation*. The distinction is
  whether the *band* axis is sharded (per-rank `bc/P = 1` at
  CrI3) or replicated (per-rank `bc = 16`). If `bc/P` is sharded,
  the box is `16·6·1·2·n_rtot·4 = 0.86 GB` per rank — fits trivially.
  If `bc` is replicated (the "unsharded" pathology), it's
  `16·6·16·2·n_rtot·4 = 13.8 GB`. **The 13.8 GB number is the
  one that matches the manual mitigation,** suggesting the band
  axis IS being replicated. So agent_3's "no factor of P"
  reading is wrong in this specific dimension. **Resolution:**
  HLO grep for the FFT-thunk in `to_rchunk`; check whether its
  band-axis bytes are `bpd = bc/P` or `bc`.

---

## 4. Consolidated open questions

Promoted to **team-level open questions** because at least two of
us flagged each independently:

- **Q4.1 — `pair_density_slots` value at CrI3 scale.** Mine §7.2,
  agent_2 §7.1, agent_3 §1.4/Q1, agent_4 §7.3. All four cite a
  range of 3 (MoS2 HLO measurement, current) to 5-8 (hypothetical
  CrI3 scaling). **Single best test:** HLO dump from one CrI3 6×6
  80 Ry `fit_one_rchunk` compile; count rank-5/rank-7 buffer
  lifetime slots holding pair-density-shape values. Resolves Q1,
  Q3, and several flow-on uncertainties.

- **Q4.2 — Why does the planner pick r_chunk ≈ 12500?** Mine §7.1,
  agent_2 §5 "Possibility 1–7", agent_3 §5.3 explanations a–e,
  agent_4 §5.b and §7.10. We all predict 23k–75k; empirical is
  12500. Agent_4 §7.10 is the most likely partial answer: the
  comparison is constrained by frozen `band_chunk = 16,
  gflat_chunk_size = 64, psig_k_chunk_size = 6`. **Two tests:**
  (a) re-run the planner with cohsex.in `band_chunk_size = 0` and
  see if r_chunk grows; (b) attach a debug log printing each
  stage's binding peak and the chosen knob so the actual
  bottleneck is visible. The 12500 may not be reproducible if
  the planner is non-deterministic across cohsex configurations.

- **Q4.3 — Unsharded vs per-rank-dense FFT box.** Mine §7.3,
  agent_3 §Q1, agent_4 §7.2. Three distinct hypotheses (band
  axis unsharded; r axis necessarily local; both). **HLO grep
  for `to_rchunk`'s FFT thunk** at CrI3 scale resolves all three
  by reporting per-rank bytes by axis. Single test, three
  hypotheses falsifiable.

- **Q4.4 — `fft_factor = 4` is one constant for three call sites.**
  Mine §7.6, agent_2 §7.4, agent_3 §Q2, agent_4 §7.5. All four
  flag this. Agent_3 §5.3 back-solves `F_fft ≈ 5` for the
  fit_one_rchunk box; agent_4 §5.b speculates F_fft ≥ 6 for the
  accumulate FFT. **Single test:** `query_fft_peak_bytes` (already
  in `compute_optimal_chunks`!) called at each of the three call
  sites at planner time, replacing the global constant.

- **Q4.5 — cuSolverMp internal scratch.** Mine §7.4, agent_2 §7.3,
  agent_3 §Q3, agent_4 §7.4. Unmodeled by all three current
  planners. Agent_3 puts it at `k_factor ∈ [2,4]·n_rmu²/P` per
  rank. **Single test:** snapshot device memory before/after one
  `potrs` / `getrs` call at CrI3 scale; subtract baseline.

- **Q4.6 — Per-channel `(r_chunk, …)` re-pick for bispinor.** Mine
  §7.5, agent_2 §5 implicit, agent_3 §Q5, agent_4 §4 explicit.
  Open as a code-design question (does the chunker get re-called
  with the right `meta.n_rmu` for each `vertex_mu_L`?). **Test:**
  grep `gw_init.fit_zeta` and outer caller for the bispinor loop;
  verify each channel re-runs `plan_gflat_chunks`.

- **Q4.7 — `psi_Y_full` is a real concurrent-live tensor.** Mine
  was implicit, agent_2 §1/§2 explicit, agent_3 §1.4, agent_4
  §1.c. Need to confirm via HLO whether `jnp.concatenate(psi_Y_parts,
  axis=1)` materialises a new buffer or fuses into the next
  consumer. If new buffer, my Peak C is missing a
  `nk·nb_full·ns·cr/p_y` term (~280 MB/rank at CrI3 charge, small
  but real).

- **Q4.8 — `centroids_persist` bug in `gflat_memory_model.py:184`.**
  Agent_2 §6, agent_3 §6.1. Possible code-level error: uses `nk`
  where `nb_F` was intended. **Single read of source confirms** —
  trivially fixed; the planner has been silently miscounting B_persist
  by ~3× at CrI3 charge.

- **Q4.9 — IBZ vs full-BZ `n_q_disk` for bispinor.** Mine §7.7
  silent, agent_4 §7.7 explicit: transverse channels use
  `write_ibz_only = False`, so `n_q_disk = 36` (full BZ) not ~10
  (IBZ). Transverse `B_persist` is ~5× the charge channel on the
  `gflat_acc` term alone. Single source read confirms.

- **Q4.10 — `max_chunks = 64` floor rationale.** Mine §7.9 only.
  No other agent picked this up. Promoting because the floor is
  the only explanation I can find for why `gflat_memory_model`
  doesn't pick `r_chunk = n_rtot` at CrI3 — and even with the
  floor, it should pick ≥ 17,578, not 12,500. **Why does the
  floor exist, and is it actually binding at CrI3?**

---

## 5. Recommended next steps

These are concrete, each pinning *what to measure / read* and
*what to expect*. Listed in priority order.

1. **Dump HLO `memory-usage-report.txt` from a CrI3 6×6 80 Ry
   `fit_one_rchunk` compile.** Specifically:
   ```bash
   XLA_FLAGS="--xla_dump_to=/tmp/lorrax_hlo --xla_dump_hlo_pass_re=.*"
   lxrun python -m gw.gw_jax -i cohsex.in   # run 1 r-chunk and bail
   ls /tmp/lorrax_hlo/module_*.jit__kernel.*memory-usage-report.txt
   grep -A 2 "preallocated-temp" /tmp/lorrax_hlo/...memory-usage-report.txt
   ```
   Then count: (a) the number of distinct lifetime offsets holding
   pair-density-shaped buffers (`c128[36, 2, 2, 1808/p_x, cr_loc, …]`),
   (b) the total bytes of the FFT thunk for `psi_R_bc_box`,
   (c) the cuFFT scratch size. **Resolves Q4.1, Q4.3, Q4.4 in one
   shot.**

2. **Read `gflat_memory_model.py:182-189` line by line and confirm
   the centroid term bug** (Q4.8). Specifically check whether
   `2*_bytes_c128(nk, ns, mu, nk, shard=p_xy)` was a typo for
   `2*_bytes_c128(nk, ns, mu, nb_total, shard=p)` (or per-axis
   `p_x`/`p_y`). If a bug, file in `KNOWN_SANDBOX_ERRORS.md` per
   AGENTS.md rule #3. **Doesn't require any compute.**

3. **Re-run the planner with all four chunk knobs unfrozen** at
   CrI3 6×6 80 Ry to test Q4.2:
   ```ini
   r_chunk_size = 0
   band_chunk_size = 0
   gflat_chunk_size = 0
   psig_k_chunk_size = 0
   ```
   Capture the planner's pick at startup. Expect (per my §2
   revision): r_chunk ≈ 14,000 ± 30%, band_chunk ≈ 8 (not 16
   because the planner's W_wfn-budget will be tighter without
   the manual `psig_k_chunk_size=6`), gflat_chunk_size some
   single-digit-hundreds value. **If the planner produces
   r_chunk ≈ 12500 ± 10%, my model corroborates; if it
   produces 70k+, the planner has a guardrail I haven't
   modelled (probably the `max_chunks = 64` floor — Q4.10).**

4. **Add a planner unit test that asserts predicted r_chunk for
   MoS2 3×3 charge equals `n_rtot` (one chunk) and for CrI3 6×6
   80 Ry charge lands within ±25% of the production value.**
   Test inputs: `(B = 60e9 · 0.97, nk=36, ns=2, n_rmu=1800,
   n_rtot=1.125e6, nb_F=400, mesh=4×4)`. Expected:
   `r_chunk ∈ [10000, 16000]` if the model is honest. Failing
   such a test flags drift (either in source or model).

5. **Run `query_fft_peak_bytes` at each of the three FFT call
   sites at planner-time** to replace the global `fft_factor =
   4.0`. The function already exists at
   `common/fft_helpers.query_fft_peak_bytes`. Three calls per
   plan: (a) the band-load FFT box (Peak A), (b) the per-bc IFFT
   inside `fit_one_rchunk` (Peak C, already called once in
   `compute_optimal_chunks`), (c) the per-cs FFT inside
   `accumulate_rchunk_to_gflat` (Peak D, never called). **One
   AOT lowering per call site at startup; cheap. Closes Q4.4.**

6. **Read `gw_init.fit_zeta` and its bispinor caller to verify
   per-channel `plan_gflat_chunks` re-runs (Q4.6).** If the
   chunker is only called once with `meta.n_rmu = n_rmu_C`,
   transverse channels are over-budgeted by ~`(mu_C/mu_T)² ≈
   2.25×` on the W_zeta term. Trivial code edit (move the
   chunker call inside the per-channel loop) once verified.

Agent 1 round 2 done
