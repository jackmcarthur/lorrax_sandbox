# Agent 2 — round 2: critique + revision

I read `agent_1.md`, `agent_3.md`, `agent_4.md`. All four of us land on:
"my from-scratch derivation predicts r_chunk in the 23k–75k range,
but the empirical value is 12,500." That's the central unresolved
question.

## 1. Cross-reading notes

### agent_1.md

**Stronger than my v1.**
- §1c, §2a, §2b: explicit `n_bc · α_psi_Y_bc · r_chunk` *Python-unroll
  cumulative-live* term for the reshard slab. I had `psi_Y_full` as
  a Tier-2 transient and explicitly dismissed it as ~4% of rank-5
  (`agent_2.md §3a`). Agent 1 cites `gw_init._fft_moment`'s
  docstring quoting a 17 GB underestimate on CrI3 16-GPU at
  `chunk_r=112016, band_chunk=16, n_bc=5`. **The per-bc reshard
  slab from XLA's straight-line unroll is a real, measured cost
  the gflat planner misses entirely.** This is a new failure mode
  for my model.
- §5b enumerates five candidate explanations for the 12500
  discrepancy with quantitative arithmetic on each — much more
  systematic than my §5 narrative.
- §6 comparison table is sharper than my §6: explicit "keep / drop
  / add" buckets.
- §7.9: notices `plan_gflat_chunks` has a `max_chunks=64` floor
  that *itself* lower-bounds r_chunk at `ceil(n_rtot/64) = 17,578`.
  **That means the empirical 12,500 cannot have come from the
  gflat planner under any setting.** Either from
  `compute_optimal_chunks`, AOT chooser, or a user override I
  didn't trace. I missed this entirely.

**Weaker / wrong.**
- §3d / §5b — Agent 1 uses `α_zeta = (pair_density_slots + 1) · …`
  by adding a "Z_q transient" rank-3 term as an additional slot.
  Z_q is `(nq, mu, r_chunk)/P`; the rank-5 slots are
  `(nk, ns², mu·r_chunk)/P` — Z_q is ~1/ns² = 1/4 the size of a
  single rank-5 slot at bispinor. Adding +1 to a slot count is a
  ~25% over-estimate, not a 1-for-1 swap.
- §5a — uses `n_q_disk = 10` for charge CrI3 IBZ. The mesh is
  6×6 = 36 q-points without further symmetry above translation;
  IBZ shrinkage at this size is more like ~7 (P4mm/m point group)
  to ~12 (lower symmetry) depending on system. The exact number
  isn't critical but 10 is a guess. (My v1 used 36 full-BZ; Agent
  4 used 7. Spread.)

**New material I didn't have.**
- The §6 verdict that `compute_optimal_chunks._zct_moment` uses
  **5 slots** (2 persistent P_l/P_r + 3 transient) while the
  gflat planner uses 3 and the AOT heuristic uses 4 — three
  competing constants live in tree. I named the conflict but
  hadn't enumerated 5 vs 3 vs 4 explicitly.
- The `query_fft_peak_bytes` mechanism in `compute_optimal_chunks`
  that *queries XLA's emitted FFT thunk* for the per-rank peak
  including cuFFT scratch. This is the right way to handle
  `fft_factor` per-call-site; I had it as an open question only.
- §7.11: shard_map rank-7 buffer's *actual* per-axis sharding
  (`/p_x` on one axis, `/p_y` on another). I had `/P` as a
  combined divisor, which is right by product but doesn't surface
  the per-axis structure.

### agent_3.md

**Stronger than my v1.**
- §3.1 spells `B_persist` out with **both X-fit AND Y-fit
  centroids** (`P1+P2 + P3+P4`) as separate divisors, then notes
  the reference §5 under-counts by missing one of the two copies.
  My v1 had both copies but didn't call out the under-count in
  prose.
- §1.3 introduces an explicit `k_factor ∈ [2, 4]` for cuSolverMp
  scratch with a placeholder formula. My v1 only flagged it.
- §4.1 step 3 introduces a **`sharded_fft_box_holds: bool` gate**
  the planner must set conservatively. Cleaner than my "heuristic
  flag based on nr ≥ 5e5".
- §7.Q1 makes a falsifiable claim: the FFT-box code path inside
  `to_rchunk` may not be "unsharded across ranks" at all — it may
  be **locally dense per rank** (i.e. the per-rank bytes ARE
  `nk·bpd·ns·n_rtot · fft_factor` because each rank holds its own
  full FFT box, sharded only on the band axis). If true, "the
  factor of P" framing of §5.8 is a misdiagnosis.

**Weaker / wrong.**
- §5.1 — **B_persist ≈ 25 GB at CrI3 charge.** Arithmetic check:
  `16·36·1800·400·2 / 4 = 16·36·1800·400·2 / 4`. Let me compute:
  `16·36 = 576; ·1800 = 1.04e6; ·400 = 4.15e8; ·2 = 8.29e8; /4 =
  2.07e8 ≈ 0.21 GB`. Agent 3 wrote "10.4 GB" for this term. **That's
  a ~50× arithmetic error.** The actual per-buffer is 0.21 GB, not
  10.4 GB, and B_persist is on the order of 1–5 GB, not 25 GB.
  Everything downstream of §5.1 (W_pool, K1, r_chunk_max, the F_fft
  back-fit at 5) is built on this error. The 22.6 GB → 23,600 r_chunk
  conclusion in §5.3 is therefore unreliable. (Agents 1, 4, and my
  own v1 all land near 0.2 GB per buffer.)
- §6.1 table claims `compute_optimal_chunks` "only counts 2" pair
  slots in `_fft_moment` (`+ 2·α_pair·cr`) — but the moment
  counts pair-density slots as alive PLUS the FFT box, so the "2"
  isn't the slot count; it's the pair-density factor distinct from
  the FFT factor. Mis-reading of the moment formula.

**New material.**
- §1.4-c — the Z_col "shard_map fallback" branch adds **two extra
  rank-3 buffers** during reshard scratch and per-q L_q replicated
  slice (`16·n_rmu²` per active q). I had this as "solve scratch
  (cuSolverMp internal)" without splitting cuSolverMp from
  shard_map-fallback. Worth pulling apart in the model.
- §4.3 — the *threshold* for when to switch from sharded to
  unsharded W_wfn is left undefined ("heuristic") — Agent 3 is
  honest about this, where I had a specific numeric (`nr ≥ 5e5`)
  with no justification.
- §6 — explicit five-bullet "what to add" list including
  `cusolvermp_k_factor` plumbed into the planner API. Useful for
  the rewrite.
- §7.Q9 — flags `n_q_disk` vs `nq` ambiguity in CONTEXT §4 ("up
  to 400"). I assumed `n_q_disk = 36`; Agent 1 assumed 10; Agent 4
  assumed 7. **Real spread of ~5× in the gflat_acc term**, which
  matters because gflat_acc dominates `B_persist`.

### agent_4.md

**Stronger than my v1.**
- §3.b distinguishes **cuSolverMp branch (Z_q donated, 1×) vs
  shard_map fallback (input+output co-live, 2×)** with the
  Si 4×4×4 31 → 16 GB measurement from `report.md §6` as evidence.
  My v1 lumped these together.
- §4 has a **clean per-channel pseudocode** with the actual
  fallback chain (`while W_wfn > ceiling: bc //= 2; … while still
  > ceiling: k_chunk //= 2`). My §4 was prose with steps; Agent 4
  wrote runnable algorithm.
- §5.b explanation #5: **`pair_density_slots` may be 6–8 at CrI3
  scale** because cuFFT scratch on a 75·75·200 rank-7 box is
  larger than on a small box. Specific mechanism: the constant 3
  was extracted on MoS2 3×3, and the rank-7 IFFT/FFT inside the
  shard_map may force more concurrent scratch slots at large nr.
  I had this as "pair_density_slots is bigger than 3" without
  pinpointing cuFFT-on-large-grid as the candidate.
- §7.10 — flags that the empirical CrI3 config has **`band_chunk
  = 16, gflat_chunk_size = 64, psig_k_chunk_size = 6` all
  user-frozen**, so only `r_chunk = 12,500` is planner-derived in
  apples-to-apples comparison. My §5 validation didn't account
  for this — I compared all four of my picks against numbers that
  only one of which was planner-derived.

**Weaker / wrong.**
- §5.a — uses `n_q_disk ≈ 7` (deep IBZ) which gives gflat_acc =
  0.38 GB, much smaller than Agent 1's 1.22 GB at n_q_disk=10 or
  my 4.37 GB at full-BZ 36. The right number depends on whether
  `write_ibz_only` is True (charge) or False (bispinor) — Agent 4
  notes both but uses 7 anyway. For the bispinor case (transverse
  channels) `write_ibz_only=False` so n_q_disk = 36 = nq.
- §5.b — has the same r_chunk ≈ 75k prediction as my §5 (so
  technically not "weaker" but doesn't probe the gap any further
  than I did).
- §6.a — "α inflation across stages that should alias" for
  `compute_optimal_chunks` — I think Agent 1 has it more precisely
  (the moments are mathematically clean and the issue is they
  match a stage decomposition that no longer exists, not that they
  inflate α's).

**New material.**
- §1.b note 1 mentions `psi_rmu_Y` reused by
  `gw.wavefunction_bundle.build_wavefunctions` after the fit —
  hence the L0 lifetime (cross-channel). I called them "C"
  persistent but didn't track that they survive *across* channels,
  which matters for the bispinor sequential 4-pass loop.
- §1.d — the `phx, phy, phz` per-axis phase tables are closure
  constants computed once in `_RCHUNK_TO_GFLAT_CACHE`. I had them
  in the catalog but didn't note the closure-once aspect.
- §7.1 enumeration of explanations is the cleanest in any of the
  four drafts; uses my candidate-2 (unsharded box) but adds
  candidate-5 (cuFFT-on-large-box S_pd inflation) as the most
  probable.
- §5.e MoS2 sanity check showing r_chunk_max ≈ 1.4M → caps at
  46k = n_rtot → "one r-chunk with defaults" matches recipe. I
  didn't sanity-check MoS2.

## 2. Revisions to my v1

For each: my v1 section + new view + evidence.

**R1. `n_bc · α_psi_Y_bc · r_chunk` cumulative-live term.**
- v1: §3a — dismissed `psi_Y_full` as ~4% of rank-5, said
  negligible.
- new view: this *is* the missing budget term that gflat planner
  lacks. The Python-unrolled bc-loop straight-line trace keeps
  all `n_bc` per-bc reshard slabs (after `to_rchunk` finishes and
  before the post-bc-loop pair-density einsum starts) alive
  concurrently. At CrI3 (`nb_F ≈ 400`, `bc = 16`, `n_bc = 25`,
  `cr = 12500`): `n_bc · 16 · nk · ns / p_y · cr = 25 · 16 · 36 ·
  2 / 4 · 12500 ≈ 90 MB`. So small at CrI3, NOT the 4× gap
  explanation. **But** Agent 1's quoted "17 GB underestimate at
  chunk_r = 112016" puts this at the binding term when r_chunk is
  large. My formula needs the term; my arithmetic at CrI3 just
  doesn't trigger it.
- evidence: `gw_init._fft_moment` docstring (per agent 1 §1c).

**R2. `max_chunks = 64` floor in `plan_gflat_chunks`.**
- v1: never mentioned.
- new view: at CrI3 `n_rtot = 1.125e6`, `r_chunk ≥ 1.125e6 / 64 ≈
  17,578`. The gflat planner *cannot* output 12,500. Either
  `compute_optimal_chunks` produced it (via the 6-stage moment
  inversion), or it's a user override, or it's the AOT chooser.
- evidence: `gflat_memory_model.py:334` `r_chunk = max(r_chunk,
  math.ceil(n_rtot / max_chunks))` (agent_1.md §7.9).

**R3. cuSolverMp branch vs shard_map fallback differ in working set.**
- v1: §3c lumped cuSolverMp scratch into a single open-question
  bucket.
- new view: the **shard_map fallback** explicitly doubles Z_q's
  per-rank cost (input + output co-live) unless donate is wired —
  the Si 4×4×4 60 Ry 31→16 GB measurement in `report.md §6` is
  exactly the donation effect. My budget needs a branch:
  `W_solve = α_solve · cr · {1 if cusolvermp else 2}` plus the
  cusolvermp_k_factor scratch.
- evidence: `report.md §2b.4, §6` and agent_4.md §3.b.

**R4. `pair_density_slots` at CrI3 scale.**
- v1: §7 question 1 — flagged that slots=3 might be wrong, did not
  propose a mechanism.
- new view: agent_4.md §5.b candidate-5 is the most plausible
  mechanism — cuFFT scratch on the rank-7 IFFT/FFT at
  `(kx, ky, kz, …)` shape on a 75×75×200 grid is **larger** than
  on a small grid because cuFFT can choose multi-radix/Bluestein
  plans with bigger workspaces. So `S_pd_effective` at CrI3 may be
  5–8 even if it's 3 on MoS2. **The constant is not
  problem-size-invariant.**
- evidence: cuFFT documentation behaviour (general knowledge),
  agent_4.md §5.b.

**R5. The empirical 12500 isn't an apples-to-apples comparison.**
- v1: §5 narrative compared my predicted 70k vs empirical 12500
  as if they were both planner outputs at the same constraints.
- new view: empirical config from `report.md §7` has `band_chunk
  = 16, gflat_chunk_size = 64, psig_k_chunk_size = 6` all
  user-overridden. Only `r_chunk = 0 → auto` is planner-derived.
  So the planner picked 12500 *given* a user-frozen bc=16 and
  kc=6, with `compute_optimal_chunks`'s 5-slot ZCT moment — not
  the gflat planner's 3-slot model.
- evidence: agent_4.md §7.10 + `report.md §7` cohsex.in recipe.

**R6. The X-fit / Y-fit centroid duplication.**
- v1: §1 catalog had both `psi_rmu_Y` (Y-shard) and `psi_rmuT_X`
  (X-shard) but didn't make explicit that this is **two
  full-size copies** of the same ψ data, divided by `p_y` and
  `p_x` respectively (not by `P`).
- new view: at balanced mesh `p_x = p_y = √P`, both copies cost
  `16·nk·μ·nb·ns/√P` each; sum is `2·16·nk·μ·nb·ns/√P`. Reference
  doc §5.2 collapses to "centroids at centroids" as one term —
  which would only hold if XLA aliased the two copies (it
  cannot — they have different sharding axes).
- evidence: agent_3.md §3.1 and §6.1, agent_4.md §3.a.

**R7. `n_q_disk` choice for the bispinor case.**
- v1: §5 used n_q_disk = 36 (full BZ).
- new view: for charge (write_ibz_only=True) the actual is ~7–10.
  For transverse channels (write_ibz_only=False, per `report.md
  §4`) it's 36. So `B_persist` differs by ~5× across channels —
  and at the transverse end, my full-BZ figure was right. At
  charge it was 5× over.
- evidence: `report.md §4` and `gw_init.fit_zeta:644`.

## 3. Where I still disagree

**D1. `pair_density_slots = 3` as the kernel-structural count.**
- Other claim: Agent 1 §7.2 reports three different repo values
  (3 in gflat, 5 in compute_optimal_chunks's ZCT moment, 4 in
  aot_memory_model heuristic), framing as "one of these is right."
- My counter-claim: Agent 4 §5.b candidate-5 is the right reading
  — `S_pd_effective(geometry)` is **not** a single integer. The
  HLO-extracted "3" is right *as a static slot count* on MoS2 3×3;
  the **dynamic** memory cost includes cuFFT scratch whose
  per-call size grows with FFT box volume. The 5 / 4 / 3 spread
  isn't model-version disagreement; it's three call sites
  measuring slightly different things at different grid sizes.
- Resolution: a **single CrI3 HLO `memory-usage-report.txt`** —
  count pair-density-shaped lifetime slots AND sum cuFFT
  preallocated scratch in those slots' shape class. The
  size-class total ≥ slot count; report both.

**D2. Whether `psi_Y_full` (post-concat) aliases with the first
P-pair slot.**
- Other claim: Agent 3 §2.3 says they are "disjoint at shard_map
  entry" hence aliasable; Agent 1 §2a says "live during the
  pair-density einsum" hence summed.
- My counter-claim: Agent 1 is right *by lifetime*. The shard_map
  ingests `psi_l_Y_sm` as an in_spec arg; that arg is *alive when
  the first inner allocation fires*. XLA can free the input only
  after the einsum reads it. The first rank-5 P_l buffer is
  allocated **inside** the einsum's output — they co-exist for
  the duration of the GEMM. So this is "sum, not max".
- Resolution: inspect HLO at the `c_q_from_psi_sm._local` entry
  boundary; specifically does the input buffer get freed before
  or after the GEMM completes? A `module_*.jit__kernel`
  memory-usage-report at any size will show this.

**D3. The `mu_C` vs `mu_T` channel branching in the planner.**
- Other claim: Agent 1 §4c says "run planner once with `mu =
  n_rmu_C`, re-use for all four channels — charge is the larger
  budget so transverse fits trivially." Agents 3 and 4 (and my v1
  §4 step 5) say "run planner once per channel, pick a
  per-channel triple."
- My counter-claim: Agent 1's argument that "fitting charge fits
  transverse" loses ~30% performance on the transverse passes —
  `α_zeta` is **linear in μ**, so transverse can use a `cr ≈ 1.5×`
  larger (n_rmu_C / n_rmu_T) than charge, paying fewer FFT taxes
  on the wfn-fetch and accumulate sides. Three of the four
  passes are transverse, so this is ~75% of the loop wall time.
  Per-channel picking is worth the extra arithmetic in the
  planner.
- Resolution: time `fit_zeta_to_h5` on a CrI3 transverse pass
  with `r_chunk = r_charge` vs `r_chunk = ~1.5·r_charge`. If the
  larger value runs faster (smaller `n_rchunks`), per-channel
  picking wins. One-GPU smoke not feasible at CrI3 80 Ry; the test
  belongs on the 4×4 mesh.

**D4. Should the unsharded FFT-box pathology be treated as a
mode bit or a continuous risk?**
- Other claim: Agent 3 §4.1 step 3 + §7.Q1 — "set
  `sharded_fft_box_holds = False` conservatively at large
  geometry; HLO-grep is the only way to know for sure." Plus
  Agent 3's specific hypothesis that the box may be **locally
  dense per rank, not actually unsharded** (i.e. the per-rank
  bytes are right but the framing in §5.8 is misleading).
- My counter-claim: my v1 §4 had a heuristic threshold
  `nr ≥ 5e5 AND bc·nr·ns·16 ≥ 0.5·B`. That's the right
  *operational* choice — engage the conservative bound at large
  geometry — but Agent 3's "the box might just be locally dense"
  hypothesis is testable and worth knowing the answer to: if the
  box is *always* locally dense (i.e. one full-r_n_rtot box per
  rank, sharded only on band), then my formula
  `16·kc·bc·ns·nr·fft_factor` (no `/P`) is the correct one
  *always*, and the "unsharded pathology" of §5.8 isn't a
  pathology at all — just a discovery that the band-axis
  sharding does what it says and the r-axis is per-rank-local.
- Resolution: HLO grep of the `to_rchunk` shard_map at any
  problem size, looking at the per-rank shape of the
  `_box_kernel` allocation. If it's `(k_chunk, bc/P, ns, nx, ny,
  nz)` — sharded version is the truth and CrI3 is a regression.
  If it's `(k_chunk, bc, ns, nx, ny, nz)` per rank (no `/P` on bc
  either) — then the "/P" on the band axis was never real, and
  my formula needs revisiting *everywhere*. Either way, the HLO
  dump answers it definitively.

## 4. Consolidated open questions

The team-level open question set. Items where multiple agents
flagged the same thing are merged.

**Q1. What actually picks r_chunk = 12,500 at CrI3 80 Ry?**
(My §7.1, Agent 1 §7.1, Agent 4 §7.1.) Three of us derive
20k–75k. Agent 1's `max_chunks=64` floor observation shows the
gflat planner *cannot* output 12,500. So either:
- compute_optimal_chunks's 6-stage moment model;
- AOT chooser with stale fit coefficients;
- A cohsex.in override I didn't trace;
- A combination of user-overridden `band_chunk=16, kc=6` plus
  one of the planners' residual r-chunk picker.
Need: a CrI3 run log + the planner's actual log line.

**Q2. `pair_density_slots` — geometry-dependent?**
(My §7.1 (folded into Q1), Agent 1 §7.2, Agent 3 §2.1 and §7.Q10,
Agent 4 §7.3.) Repo has three values (3 / 4 / 5) live in different
files. Open question is whether the "right" value is geometry-
invariant or grows with FFT box volume due to cuFFT scratch slot
selection. Need: HLO `memory-usage-report.txt` from CrI3 6×6 80 Ry,
specifically count pair-density-shape lifetime slots + cuFFT scratch
in the same shape class.

**Q3. Does XLA shard the band-FFT box, or is it locally dense per
rank?** (My §7.5, Agent 1 §7.3, Agent 3 §7.Q1, Agent 4 §7.2.) The
"/P" divisor in W_wfn is correct only if XLA emits a sharded FFT
box; the §5.8 mitigation `psig_k_chunk_size=6` suggests it doesn't.
Agent 3's contrarian read: the box may be locally dense
(per-rank-full-n_rtot, band-axis-sharded) by design and §5.8's
"unsharded pathology" framing is wrong. The right answer changes
the planner's W_wfn formula entirely. Need: HLO grep of
`to_rchunk` allocation at any size.

**Q4. cuSolverMp internal scratch — bound how?**
(My §7.3, Agent 1 §7.4, Agent 3 §1.3 and §7.Q3, Agent 4 §7.4.)
Bound is `O(n_rmu²/P)` per panel × some panel count `k_factor ∈
[1,4]`. Plus LU has pivoting workspace ≥ Cholesky's. Today: not
modelled. Need: a single nvprof / runtime memory probe on a
sized-up run, then a calibration constant in the planner.

**Q5. Per-channel chunking (charge vs transverse) — is the
planner re-run per-channel?**
(My §7.6, Agent 3 §7.Q5, Agent 4 §4 pseudocode.) Charge and
transverse have different `n_rmu`. `gw_init.fit_zeta:597-625`
calls `plan_gflat_chunks(meta=meta, …)` exactly once — passing
the global `meta.n_rmu`. If `meta.n_rmu` is set to
`max(n_rmu_C, n_rmu_T) = n_rmu_C` at problem-setup, transverse
channels get charge-budgeted chunks (correct but suboptimal). If
it's set per-channel, the planner needs to be invoked in the
bispinor loop. **Need: read the bispinor outer loop in
`gw_jax.py` to see whether `meta.n_rmu` flips between channels.**
Stronger fix: thread `n_rmu_chan` into the planner explicitly.

**Q6. `n_q_disk` ambiguity in CONTEXT §4.**
(Agent 3 §7.Q9, picked up by all of us in §5 with different
guesses 7 / 10 / 36.) "n_k = n_q = 36 (reduced from up to 400 by
symmetry)" — is 400 the full-BZ count or loose phrasing? Affects
B_persist's gflat_acc term by up to 5×. For transverse channels
(`write_ibz_only=False`) n_q_disk = nq_full anyway. Need: pin
down the actual `n_q_disk` from a CrI3 run log.

**Q7. Per-call-site `fft_factor`.**
(My §7.4, Agent 1 §7.6, Agent 3 §7.Q2, Agent 4 §7.5.) Single
scalar `4.0` covers three FFT call sites. The accumulate-side
1-D FFT on length `n_rtot = 1.13M` plausibly needs 6–8×;
empirically my predicted `gflat_chunk_size ≈ 760` vs recipe 64 is
12× off. Agent 3's recipe: invoke `query_fft_peak_bytes` per call
site at planner time. Solves the problem cleanly.

**Q8. Cumulative `psi_Y_bc` reshard slabs across Python unroll.**
(Agent 1 §1c, R1 in my §2.) `compute_optimal_chunks._fft_moment`
counts this; gflat planner doesn't. Worth modelling explicitly in
the rewrite. At CrI3 the term is ~MB, but at larger r_chunk
(112k as in the gw_init docstring) it's 17 GB. Need: explicit
inclusion + a test at one large-r_chunk geometry to confirm the
term is real after future XLA upgrades.

**Q9. `psi_Y_full` aliasing into rank-5 slot.**
(My v1 §3a hand-wave, Agent 1 §2a "summed", Agent 3 §2.3
"aliasable", contradictory readings.) Two of us disagree on
whether XLA can free the shard_map input arg before the inner
GEMM completes. Need: HLO inspection at the
`c_q_from_psi_sm._local` boundary, lifetime annotations on the
input buffer.

**Q10. Auto-recalibration of `pair_density_slots` from HLO.**
(Reference §5.10 follow-up #2, Agent 3 §7.Q10.) Concrete fix:
build a startup probe that dumps `module_*.jit__kernel.memory-
usage-report.txt`, greps pair-density-shape lifetime slots,
fail-loud if it differs from source constant. Open: what's the
JAX 0.4.x API for triggering the dump?

## 5. Recommended next steps

Six concrete actions:

**S1.** Read the CrI3 6×6 80 Ry validation run log
(`runs/CrI3/M_6x6_80Ry_*/.../lorrax_D_*/...` per git status — the
2026-05-12 family) and capture the planner's log line for the
chunk choices: which planner produced `r_chunk = 12500`, which
stage bound, what HBM the run actually peaked at. Resolves Q1.

**S2.** Run `fit_zeta_to_h5` once on CrI3 6×6 80 Ry with
`XLA_FLAGS='--xla_dump_to=/tmp/hlo_cri3'`, grep for
`pair-density` or rank-5/rank-7 shape signatures, count distinct
preallocated-temp lifetime slots holding `c128[nk, ns², …]`-shape
buffers, and report the count alongside cuFFT scratch size in the
same class. Resolves Q2, Q9; partial resolution of Q3.

**S3.** Grep `psi_G_store.fetch_psi_rchunk`'s `to_rchunk` call
chain (`wfn_transforms.py:338, isdf_fitting.py:1167+`) in the same
HLO dump for the per-rank shape of the FFT-box allocation. If it
shows `(k_chunk, bc/P, ns, nx, ny, nz)`, the sharded formula
holds; if it shows `(k_chunk, bc, ns, nx, ny, nz)` per rank, the
"unsharded pathology" is actually "by-design local-dense."
Resolves Q3.

**S4.** Read `gw_jax.py` (or `gw_init.fit_zeta` and its outer
caller) and trace where the bispinor 4-channel loop lives: does
`meta.n_rmu` flip per channel before each `fit_zeta_to_h5`
call, or is the planner only invoked once on `n_rmu_C`?
Resolves Q5. Reading-only; ~30 minutes.

**S5.** Replace `gflat_memory_model.py`'s static `fft_box_factor`
with per-call-site `query_fft_peak_bytes` invocations:
- Peak A: `query_fft_peak_bytes` on `(nk, bpd_load, ns, *fft_grid)`.
- Peak C: same kernel, with `band_chunk` instead of `bpd_load`.
- Peak D: `(cs, *fft_grid)` 3D FFT.
At planner time each call costs one AOT lowering (~100 ms). Closes Q7.

**S6.** Add a planner unit test that asserts (at the **3-knob
user-frozen empirical CrI3 config** — `band_chunk = 16,
gflat_chunk_size = 64, psig_k_chunk_size = 6`) the planner's
auto-picked `r_chunk` lands within ±20% of 12,500. With S1's
result, this becomes pin-the-current-behaviour rather than
fix-then-verify. Includes a calibration row in the planner's log
that prints which of (compute_optimal_chunks, gflat_memory_model,
aot_chooser) made the decision, with the binding constraint.

Agent 2 round 2 done
