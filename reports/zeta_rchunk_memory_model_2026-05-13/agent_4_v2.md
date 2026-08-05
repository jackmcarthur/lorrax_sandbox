# Agent 4 — Round 2: critique + revision

I've read `agent_1.md`, `agent_2.md`, `agent_3.md`. Below: what each had that I didn't,
where I revise my v1, where I still disagree, the consolidated open-question set, and
concrete next steps.

---

## 1. Cross-reading notes

### Agent 1 (the most thorough — 941 lines)

- **Stronger than my v1**
  - `§1c` explicitly catalogues the `n_bc · α_psi_Y_bc · r_chunk` cumulative-live cost
    from the Python-unrolled bc-loop, citing `gw_init._fft_moment` docstring as evidence
    that XLA holds all `n_bc` reshard slabs live concurrently. My `agent_4.md §1.c` had
    a per-bc `psi_l_Y_bc` row but missed the *cumulative-across-unroll* part.
  - `§2b` cleanly distinguishes "W_wfn aliases into one freed pair-density slot" vs
    "W_wfn sums with the pair-density slots" — the optimistic/pessimistic interpretation
    of report.md §5.5. My v1 implicitly assumed the optimistic case (max not sum).
  - `§7.2` enumerates the *three* competing values of `pair_density_slots` in tree (3
    in gflat_memory_model, ~5 in `_zct_moment`, 4 in the aot heuristic). My §7.3 only
    flagged the magnitude as uncertain.
  - `§7.9` flags `plan_gflat_chunks`'s opaque `max_chunks=64` floor as a candidate
    explanation for the empirical r_chunk = 12500 (= n_rtot/64 = 17,578 lower bound).
    I had missed this knob entirely.
- **Weaker / wrong**
  - `§5b` computes `α_zeta = (S+1)·16·nk·ns²·mu/P = 1.04 MB/r-unit` but the `+1` is for
    Z_q `(nq, mu, r_chunk)/P = 16·nq·mu·r_chunk/P` which is *rank-3 with no ns² factor*.
    Z_q's per-r contribution is `nq·mu·16/P`, not `nk·ns²·mu·16/P`. So `α_zeta` is
    overstated by `(nk·ns²)/nq ≈ ns² = 4`. Real `α_zeta ≈ 0.78 MB/r-unit`.
- **New material**
  - Per-channel mu argument (§4c): charge always sets the binding budget because
    `mu_C > mu_T`; one planner pass with `mu = mu_C` suffices. I had explicitly *not*
    asserted this and recommended per-channel re-picks; Agent 1's optimisation is real.
  - `§5a`: explicit `ngkmax ≈ 0.06·n_rtot ≈ 67,500` heuristic from
    `gw_init.py:596`. I had guessed 30,000.
  - `§6` table format for the diff. Lifts both `compute_optimal_chunks` and
    `gflat_memory_model` strengths clearly; my §6 conflated them.

### Agent 2 (the most surgical — 480 lines)

- **Stronger than my v1**
  - `§3a / §5` distinguishes between **logical** `r_chunk` and per-rank-local
    `r_chunk/p_y` and walks the per-rank slot bytes explicitly. My v1 was sloppy about
    which `μ` and `r_chunk` get divided by which mesh axis.
  - `§6` catches a **bug in `gflat_memory_model._peak_C_fit_one_rchunk`**: uses
    `2·_bytes_c128(nk, ns, mu, nk, shard=p_xy)` — i.e. **`nk` appears in place of `nb`**.
    Per the function signature `_bytes_c128(*dims)` multiplies all dims, so the
    "centroids persistent" line accidentally scales by `nk·nk` instead of `nk·nb`. At
    CrI3 (nk=36, nb=400) this under-counts by `400/36 ≈ 11×`. Real bug in source. I
    missed it.
  - `§5` tries five concrete candidate explanations for the 5.6× discrepancy with
    the empirical 12500 and rules each out with arithmetic. Honest "I don't fully
    resolve this."
- **Weaker / wrong**
  - `§3 W_iter(cr, bc, kc)` formula adds `max(0, W_Yfull − slot_overlap)` — but
    `slot_overlap` is undefined (`one_rank5_slot` vs `psi_Y_full` have completely
    different shapes; XLA can't alias them just because their byte counts happen to
    match). The aliasing claim needs more than equal-bytes.
  - `§5` predicts `gflat_chunk_size = 760` and concludes "empirical 64 is 12× off" —
    likely cuFFT scratch factor is much bigger for the 1-D 1.1M FFT than `fft_factor = 4`.
    Same conclusion as agent 3.
- **New material**
  - `§3a` explicit aliasing of the `psi_R_bc_box` into one rank-5 slot, *conditional on
    shape-match*. My v1 hand-waved the aliasing without checking shapes.
  - `§7.11` flags `target_utilization = 0.80` as doing implicit work — it accounts for
    ~12 GB of unmodelled scratch at CrI3 scale. I missed this lever.
  - `§5` notes `nb_total_chunker = nb_L + nb_R = 2·nb_F` in
    `gw_init.fit_zeta:588` — so band counts in `B_persist` need to use `nb_L + nb_R`,
    not `nb_F`. I used `nb_full` ambiguously.

### Agent 3 (also thorough, 966 lines)

- **Stronger than my v1**
  - `§5.3` walks the linearisation of Peak C through to an explicit
    `K0 + K1 · cr` form with K1 broken into the three sub-coefficients
    (`α_psi_Y + S·α_pair + α_zcol`). This is the cleanest closed-form
    derivation in any agent's report.
  - `§5.3` arrives at `cr_max ≈ 23,600` and explicitly tests
    `F_fft = 5` → `cr ≈ 14,200`, finding the F_fft = 4 constant is the
    most plausible culprit for the 5× discrepancy. **More plausible diagnosis than my v1.**
  - `§2.3` lifetime sequence narrative through one fit_one_rchunk jit is clearer than
    my §2.c sketch.
  - `§5.5` explicitly says "η does the work an explicit scratch term should do" —
    the target_utilization = 0.80 cushion absorbs unmodelled cuFFT + NCCL scratch at
    CrI3 scale.
- **Weaker / wrong**
  - `§5.1 B_persist` arithmetic appears to be **off by ~50×** for the centroid
    terms: claims `16·36·1800·400·2 / 4 = 10.4 GB`, but `16·36·1800·400·2/4 =
    207,360,000 = 0.21 GB`. They land at `B_persist ≈ 25.4 GB`, which is two orders
    of magnitude too high. The "matches empirical 28 GiB peak" coincidence is therefore
    suspect.
  - `§Q1` proposes the unsharded FFT box may be a **misdiagnosis** — that the box is
    actually per-rank-local but full-n_rtot dense (not factor-of-P unsharded), so the
    "factor of P" framing in report.md §5.8 is wrong. **Plausible** but unverified. If
    true, my v1 §5 unsharded-formula assumption (factor of P) is over-conservative.
- **New material**
  - `§4.1.7` includes a budget-aware **V_q chunker** in the same algorithm. My v1
    omitted V_q entirely (deferred to a future planner).
  - `§Q4` challenges the `r_chunk ≥ n_rmu` floor — argues it's a misreading of report.md
    §5 ("Σ_μν output is n_rmu²·nq·16" — but Σ_μν isn't the output of zeta-fit). My v1
    inherited this floor uncritically.
  - `§Q9` flags the ambiguity in CONTEXT §4's "n_k=n_q=36 (reduced from up to 400 by
    symmetry)" — is "400" the unreduced count or just hand-waving? I had assumed 36 was
    the full BZ count without questioning.

---

## 2. Revisions to my v1

| my §  | v1 claim | revised position | evidence |
|---|---|---|---|
| §1.a | "psi_rmu_Y and psi_rmuT_X are views of the same data; persistent centroids ≈ 416 MB / rank" | They are **two distinct buffers** (same data, different shardings). At CrI3 charge the persistent ψ-centroid bytes are 2 × (16·nk·nb·ns·μ / p_axis) — both X-form and Y-form, each sharded on one mesh axis only. Total ~0.4 GB/rank, **not** ~50 MB I implied. | Agent 2 §1, Agent 3 §1.1. Caller `gw_init.fit_zeta` passes both as distinct args; the function header at `isdf_fitting.py:1466-1467` confirms two arrays. |
| §1.c | Implicit per-bc `psi_l_Y_bc`, no cumulative-across-unroll term | Add `α_psi_Y_cumulative = 16·nk·(nb_L+nb_R)·ns·r_chunk / p_y` as a separate term in W_iter. The Python-unrolled bc loop holds **all** `psi_l_Y_bc` slabs live concurrently after concat. | Agent 1 §1c, Agent 3 §1.4a; `gw_init.py:_fft_moment` docstring. |
| §3.b | `W_zeta = S_pd · α_zeta · r_chunk` with `α_zeta = nk·ns²·μ·16/P` | `α_zeta` should also include `α_psi_Y_cumulative + α_zcol ≈ 16·nq·μ/P`. Per Agent 3 §5.3, K1 ≈ 3.73e5 bytes/r-unit at CrI3 charge, with the psi_Y term dominating (230 KB/r-unit) over the rank-5 slots (78 KB/r-unit). **psi_Y is the dominant lever**, not the rank-5 slots. | Agent 3 §5.3, my own §5.b. |
| §4 picker step 3 | "Use `W_pool / S_pd` as the band_chunk ceiling per §5.5" | I shouldn't take this as principled until HLO confirms aliasing. The 50/50 split in current source is a defensible conservative fallback. Agent 1 §6b takes the same line. | Agent 1 §6b. |
| §5.b | "My picker says r_chunk ≈ 75,000" | Corrected: with the missing `psi_Y_cumulative` term (Agent 1, Agent 3), `K1 ≈ 3.7e5 B/r-unit`. r_chunk_max in the 23-55k range depending on `F_fft` and unsharded-FFT cost. Still 2-4× larger than empirical 12,500 — gap explained by `F_fft ≈ 5–6` at CrI3 grids (Agent 3 §5.3) plus the `target_utilization = 0.80` cushion (Agent 2 §7.11). | Agent 3 §5.3, Agent 2 §7.11. |
| §5.d | "gflat_chunk_size ≤ 794 fits, so empirical 64 is conservative" | Wrong direction: my formula **over-budgets** cs because the 1-D 1.1M FFT inside `accumulate_rchunk_to_gflat` needs cuFFT scratch *much* bigger than `F_fft = 4`. Per Agent 3 §5.4, `F_fft_accum ≥ 6` (likely 8); per Agent 2 §5, the scratch may be ~50× the box. Need a per-call-site F_fft. | Agent 2 §5, Agent 3 §5.4. |
| §6 / §7.7 | "`ngkmax = 30,000` is my guess for CrI3 80 Ry" | `gw_init.py:596` uses `0.06·n_rtot ≈ 67,500` as the default fallback when `meta.ngkmax` is missing. My estimate was 2× low. Doesn't change the binding peak but inflates `B_persist`. | Agent 1 §5a, Agent 3 §5.1. |
| §7.1 (open Q) | "Why r_chunk = 12500" listed as the #1 open question | Promote to **team-level** consensus: all four agents independently derive r_chunk in the 20-75k range and admit they can't reach 12,500 from desk math. The two most-plausible single explanations: (a) F_fft is per-site and the accumulate-FFT site is ≥ 6–8 (Agent 3); (b) `target_utilization = 0.80` is the cushion (Agent 2). Most likely both. | All four agents §5. |

---

## 3. Where I still disagree

### 3.a Agent 1 §4c — "fitting charge fits transverse" with one planner pass

**Agent 1's claim** (`§4c, §1e`): one planner pass with `mu = mu_C` (charge), reused for
all four channels. Charge is always the bigger channel → transverse fits with slack.

**My counter-claim**: this leaves performance on the table for transverse. With
`mu_T ≈ 1200` vs `mu_C ≈ 1800`, `α_zeta` is 33% smaller for transverse, so transverse
*could* tolerate a 50% larger r_chunk — meaning ~1.5× fewer r-chunks → ~1.5× less FFT
tax on the transverse fits. Three transverse fits × ~30% wall time savings each is real.
Per-channel re-picking (my v1 §4) is the right answer; the bispinor pipeline already
calls `fit_zeta_to_h5` sequentially per channel (`vertex_mu_L` arg at
`isdf_fitting.py:1477`), so picking per-channel costs nothing.

Agent 2 §4 step 5 agrees with me. Agent 3 §4.1.6 agrees with me.

**What resolves it**: trivial — just call the planner once per channel. The asymmetry is
genuine; Agent 1 underestimates the transverse-side gain.

### 3.b Agent 3 §Q1 — the "unsharded FFT box" may be a misdiagnosis

**Agent 3's claim**: the report.md §5.8 phrasing "factor of P unsharded" may be a
misdiagnosis — the box is plausibly per-rank-local-but-full-n_rtot-dense, not factor-of-P
unsharded.

**My counter-claim**: this contradicts the empirical observation that
`psig_k_chunk_size = 6` is required at CrI3 80 Ry. If the box were just
"per-rank-local but full n_rtot dense" with `bpd = bc/P = 1`, then per-rank cost is
`16·nk·1·ns·n_rtot·F_fft = 16·36·1·2·1.125M·4 = 5.2 GB`. That fits under 60 GB without
any `psig_k_chunk_size` mitigation. But empirically the system OOMs without the
mitigation, per report.md §7. So either:
- The box really is replicated across ranks (factor of P), giving 83 GB per rank → OOM
  → mitigation needed.
- Or there's a different unsharded-by-P intermediate elsewhere.

Agent 1 §1f computes the same 83 GB per rank with the factor-of-P assumption and
arrives at the same conclusion. Agent 2 §3b agrees.

**What resolves it**: HLO grep of the actual `to_rchunk` call at CrI3 sizes, looking
for the FFT-input buffer's allocation shape. If shape is `c128[36, 1, 2, 75, 75, 200]`
(per-rank dimensions, /16 sharded), Agent 3 is right. If it's
`c128[36, 16, 2, 75, 75, 200]` (full bc, replicated), the factor-of-P framing is right.

### 3.c Agent 2 §6 — `_peak_C_fit_one_rchunk` "centroids_persist" bug

**Agent 2's claim**: `gflat_memory_model._peak_C_fit_one_rchunk` line
`2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy)` uses `nk` in place of `nb`, under-counting.

I confirmed this from source — `gflat_memory_model.py:184` reads
`2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy)`. The second `nk` should be `nb_total`
(bands), since persistent centroids are shape `(nk, n_b, ns, mu)`. At CrI3 with
`nk=36, nb=400`, the bytes scale as `36·400 = 14,400` vs the buggy `36·36 = 1,296` —
the model under-counts the persistent centroid term by ~11×. **I have no disagreement
with Agent 2 here; this is a real bug.** Note also that the same line uses
`shard=p_xy = P` for *both* X-form and Y-form copies, but each copy is sharded on only
one mesh axis (`p_x` or `p_y` separately) — so even when nb is right, the divisor
should be `p_x` for one and `p_y` for the other. At a balanced 4×4 mesh that's a 4×
further under-count.

**Action**: flag for direct fix when the planner gets rewritten.

### 3.d Agent 1 §3c — FFT box "aliases into one freed pair-density slot"

**Agent 1's claim**: when applying the §5.5 ceiling, treat `W_wfn_box` as fitting
inside *one* freed pair-density slot, justified by lifetime-disjoint ordering.

**My counter-claim**: aliasing requires both lifetime disjoint *and* shape match. The
FFT box shape `c128[nk_slice, bc/P, ns, nx, ny, nz]` and the rank-5 pair-density slot
shape `c128[kx, ky, kz, ns, r_chunk_loc, μ_loc, ns]` have different per-axis layouts —
XLA's allocator is shape-aware and won't alias unequal-rank buffers in the same slot
unless the larger one's bytes accommodate the smaller (which works only one direction).
The "1/S ceiling" is more like "FFT box must fit in *the remaining W_pool after S slots
allocated*", not "fits in one slot".

**What resolves it**: HLO grep of one fit_one_rchunk compile, looking at slot
assignments — does any single preallocated-temp slot host both an FFT-box-shaped value
and a pair-density-shaped value across non-overlapping lifetimes?

---

## 4. Consolidated open questions

Cross-referenced. Bold = load-bearing for the planner.

| # | Question | Owners (my §, others' §) |
|---|---|---|
| **OQ1** | **Why does the empirical r_chunk land at ~12,500?** All four agents derive 20-75k from first principles. | my §7.1, agent_1 §5b §7.1, agent_2 §5, agent_3 §5.3/5.5 |
| **OQ2** | **`pair_density_slots`: 3, 4, or 5?** Three different values in tree, all hand-extracted from different HLO dumps. | my §7.3, agent_1 §7.2, agent_2 §7.1, agent_3 §2.1 |
| **OQ3** | **Per-call-site `fft_factor`** — one global F=4 is the wrong abstraction. CrI3 evidence: `F_fft_accum ≥ 6–8` for the 1-D 1.1M FFT inside `accumulate_rchunk_to_gflat`. | my §7.5, agent_1 §7.6, agent_2 §7.4, agent_3 §Q2/Q7 |
| **OQ4** | **Unsharded FFT-box pathology** — is the factor-of-P framing in report.md §5.8 right, or is Agent 3's "per-rank-but-dense" diagnosis right? Either way, what's the actual root-cause site? | my §7.2, agent_1 §1f/§7.3, agent_2 §3b/§7.5, agent_3 §1.4a/§Q1 |
| **OQ5** | **cuSolverMp internal scratch** for `potrs`/`getrf`/`getrs` distributed factor/solve — `O(mu²)`-class but exact size unknown; transverse LU has higher scratch than charge Cholesky. | my §7.4, agent_1 §7.4, agent_2 §7.3, agent_3 §Q3 |
| **OQ6** | **`psi_Y_full` aliasing & cumulative-bc-unroll cost**: do the `n_bc` per-bc Y-slabs sum or alias after `jnp.concatenate`? Affects Peak C by `(n_bc·bc·nk·ns·r_chunk/p_y)`-class. | (not in my v1), agent_1 §1c/§2a/§7.8, agent_2 §3a/§7.2, agent_3 §1.4a/§Q6 |
| OQ7 | **`gflat_memory_model._peak_C_fit_one_rchunk` source bug** at line 184 (`nk` typo for `nb`, wrong shard divisor for L+R centroids). | agent_2 §6 (confirmed via direct source read in §3.c above) |
| OQ8 | **`max_chunks = 64` floor** in `plan_gflat_chunks` — opaque, possibly the proximate cause of empirical r_chunk = 12,500? `n_rtot/64 = 17,578` is the lower-bound it imposes. | agent_1 §7.9 |
| OQ9 | **`target_utilization = 0.80`** is doing the work an explicit scratch term should do; cushion of ~12 GB at CrI3 absorbs uncatalogued overhead. | agent_2 §7.11, agent_3 §5.5 |
| OQ10 | **Per-channel mu in the bispinor pipeline** — is `meta.n_rmu` per-channel or globally fixed at problem-setup? Does `fit_zeta` get re-called per `vertex_mu_L`? | my §1.b ("per-channel"), agent_2 §5 step 5, agent_3 §Q5 |
| OQ11 | **`ngkmax` is estimated, not measured** at planner time — `gw_init.py:596` uses `0.06·n_rtot`. Real `ngkmax` from `zeta_cutoff_ry`-derived sphere is only known inside `fit_zeta_to_h5`. Affects `B_persist` (gflat_acc term) by up to 2×. | my §7.9, agent_1 §5a/§7.7, agent_3 §Q9 |
| OQ12 | **W_vq budget** is a separate problem; `_pick_g_chunk` caps at 4096 by fiat, no budget-aware pick. Should fall out of the same planner. | my §6 ("Add"), agent_1 §3f/§7 ("V_q"), agent_2 §7.9, agent_3 §4.1.7 |
| OQ13 | **`r_chunk ≥ n_rmu` floor** — Agent 3 §Q4 argues this comes from a misreading of report.md §5; the real floor is `r_chunk ≥ P` for sharding divisibility. Worth a test run with override. | agent_3 §Q4 |
| OQ14 | **Bispinor 4-channel orchestrator status** — is the per-channel ζ-fit + V_q assembly landed on `lorrax_A/main`, or only on agent-B? Affects whether the per-channel re-planner is forward-looking or shippable today. | agent_3 §Q8 |
| OQ15 | **`n_q_disk` semantics** — does CONTEXT §4's "n_k=n_q=36 (reduced from up to 400 by symmetry)" mean full-BZ 400, IBZ 36, or something else? | agent_3 §Q9 |

---

## 5. Recommended next steps

Six concrete actions, ordered by what would close the most open questions per unit
effort. Each says *what* to do, *on what*, and *what result decides*.

1. **Run one CrI3 6×6 80 Ry `fit_zeta_to_h5` invocation with HLO dump enabled** to
   answer OQ1, OQ2, OQ4, OQ6 simultaneously. Concretely:
   - `XLA_FLAGS="--xla_dump_to=/tmp/hlo_dump --xla_dump_hlo_pass_re=.*"` for one
     auto-planner CrI3 run (let the planner pick r_chunk).
   - Open the resulting `module_NNNN.jit__make_fit_one_rchunk_kernel.memory-usage-report.txt`.
   - **Decision criteria:**
     - Count distinct preallocated-temp slots holding a `c128[nk, ns, ns, mu_loc,
       r_chunk_loc]`-shape value → that is `pair_density_slots` for CrI3 (OQ2).
     - Grep for the FFT-box buffer: does it carry shape `[nk, bc/P, ns, nx, ny, nz]`
       (sharded) or `[nk, bc, ns, nx, ny, nz]` (replicated)? → resolves OQ4.
     - Grep for `psi_Y_full` slots: are they aliased to a pair-density slot, or summed?
       → resolves OQ6.
     - Compare planner's logged `(r_chunk, band_chunk, gflat_chunk_size,
       psig_k_chunk_size)` with the HLO peak slot bytes — quantify the gap → OQ1.

2. **Read `gw_init.fit_zeta` (lines 532–700) and confirm which planner is invoked** in
   the current build and with what arguments. Specifically: is `compute_optimal_chunks`
   still in the path, or has it been replaced by `plan_gflat_chunks`? Where does
   `r_chunk = 12,500` actually come from? Look for any
   `cohsex.r_chunk_size = 12500` default in `cohsex.in` templates that might explain
   the empirical value without invoking either planner. **Decision criteria:** locating
   the source of the 12,500 value pin-points OQ1 to one of (compute_optimal_chunks
   pick / plan_gflat_chunks pick / cohsex default override / `max_chunks=64` floor).

3. **Fix `gflat_memory_model._peak_C_fit_one_rchunk:184` typo** — replace
   `_bytes_c128(nk, ns, mu, nk, shard=p_xy)` with two separate terms
   `_bytes_c128(nk, ns, mu, nb_total, shard=p_x) + _bytes_c128(nk, nb_total, ns, mu,
   shard=p_y)`. Closes OQ7. Add a regression unit test that this term scales as `nb`
   not `nk`. Cheap. (Read-only constraint in this exercise — flag for follow-up.)

4. **Calibrate per-call-site `fft_factor` via `query_fft_peak_bytes`** at the three call
   sites (Peak A loader FFT, Peak C bc-FFT inside fit_one_rchunk, Peak D accumulate
   FFT). The infrastructure already exists in `common.fft_helpers.query_fft_peak_bytes`
   and is used by `compute_optimal_chunks`. Closes OQ3.
   - For each site, query at the target geometry and compare the returned bytes to
     the box's own size (`nk·bc·ns·n_rtot·16/P` etc.). The ratio *is* `F_fft_site`.
   - **Decision criteria:** for CrI3 80 Ry expect `F_fft_loader ≈ 4`,
     `F_fft_kshmap ≈ 4–5`, `F_fft_accum ≥ 6`.

5. **Add a planner regression test with two recorded golden values**, MoS2 3×3 and
   CrI3 6×6 80 Ry:
   ```python
   def test_chunker_at_known_sizes():
       p = plan_zeta_fit_chunks(meta_mos2_3x3, mesh_2x2, B=60e9, …)
       assert p.r_chunk == n_rtot_mos2_3x3        # one chunk
       p = plan_zeta_fit_chunks(meta_cri3_6x6_80, mesh_4x4, B=60e9, …)
       assert 10000 <= p.r_chunk <= 15000         # ±20% of empirical
   ```
   Run this before/after every planner edit and after every fit_one_rchunk
   kernel-structure change. Catches accidental shifts in `pair_density_slots` or
   `fft_factor`. Closes OQ2 + OQ4 over time as drift gets caught.

6. **Resolve OQ13 (the `r_chunk ≥ n_rmu` floor) with one one-shot CrI3 test**: set
   `r_chunk_size = P = 16` in `cohsex.in` and run. If the system completes, the
   `≥ n_rmu = 1800` floor is folklore and can be dropped from both planners. If it
   crashes or wastes time, the floor has a real reason worth documenting. Quick smoke
   test; cheap.

---

Agent 4 round 2 done.

