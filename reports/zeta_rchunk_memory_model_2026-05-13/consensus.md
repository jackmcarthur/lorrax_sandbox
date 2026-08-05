# Consensus — zeta-fit r-chunk memory model

Synthesised from four independent v1 derivations and four v2 critique-revisions
(`agent_{1..4}.md`, `agent_{1..4}_v2.md`, 2026-05-13). This document is the
team's final position. Round 3 is not planned; HLO and run-log evidence is
needed to close the remaining items.

## 1. TL;DR

The four agents converge on a four-peak (A pre-load, B CCT/factor, C
`fit_one_rchunk`, D `accumulate_rchunk_to_gflat`) model with a sum of
`B_persist` plus a single `W_pool` that the three transient peaks contend for.
The dominant linear-in-`r_chunk` term inside Peak C is
`α_zeta = pair_density_slots · 16 · nk · ns² · μ / P`, modulated by an
additive `psi_Y_full` cumulative slab and a separate FFT-box term whose
sharding is the most fragile assumption in the model.

The most load-bearing finding everyone agrees on:
**the empirical r_chunk ≈ 12 500 at CrI3 6×6 80 Ry is *not* an apples-to-apples
target.** Three of four chunk knobs are user-overridden in `cohsex.in`
(`band_chunk_size = 16`, `gflat_chunk_size = 64`, `psig_k_chunk_size = 6`);
only `r_chunk_size = 0` lets the planner pick. From-scratch derivations land
in the 20k–75k range; the gap is most plausibly absorbed by `F_fft` per-site
(≥ 5 at this geometry) and the implicit `target_utilization = 0.80` slack.

Two confirmed source bugs need fixing: `gflat_memory_model.py:184`
(`nk` typo for `nb_total`, plus wrong shard divisor) and the
`nb_total_chunker = nb_L + nb_R` doubling that shows up at `gw_init.py:588`.

User-actionable next steps in priority order: (1) capture the
`gflat_plan.format()` log line from any recent CrI3 run to settle which
planner produced the 12 500, (2) dump HLO `memory-usage-report.txt` from one
CrI3 `fit_one_rchunk` compile to resolve `pair_density_slots`, the FFT-box
sharding, and `psi_Y_full` aliasing in one shot, (3) fix the `:184` bug, (4)
thread per-call-site `query_fft_peak_bytes` through the planner.

## 2. The agreed model

### Notation

- `P = p_x · p_y` (mesh size). At CrI3 4×4, `p_x = p_y = 4`.
- `c128 = 16 B`. All bytes below are per-rank.
- `μ = meta.n_rmu_padded` (rounded up to a multiple of `P`).
- `nk = nq = meta.nk_tot`; `nq_disk` is the on-disk q-axis (IBZ when
  `write_ibz_only=True`, else `nk`).
- `ns = nspinor` (= 2 bispinor).
- `nb_F = nb_total = nb_L + nb_R` (with the `nb_L + nb_R` doubling noted
  in §3; see `gw_init.fit_zeta:588`).
- `n_rtot = nx · ny · nz`. `ngkmax` is the per-q ζ-sphere ceiling.
- `cr = r_chunk`, `bc = band_chunk`, `kc = psig_k_chunk_size`,
  `cs = gflat_chunk_size`, `n_bc = ⌈nb_F / bc⌉`.
- `S_pd = pair_density_slots` (default 3, MoS2 3×3 HLO-extracted).
- `F_fft = fft_factor` (default 4.0). Per-call-site values are needed; see
  `agent_3_v2.md §4` Q-3 / `agent_4_v2.md §4` OQ3.

### B_persist

The four persistent device buffers around the r-chunk loop, with the
**corrected per-axis sharding** (agent_2.md §3, agent_3.md §3.1,
agent_4.md §3a):

```
B_persist =  16 · nk · nb_F · ns · μ / p_y     # psi_rmu_Y
          +  16 · nk · μ · nb_F · ns / p_x     # psi_rmuT_X
          +  16 · nq · μ²        / P           # L_q  (charge: Cholesky;
                                               #        transverse: C_q passthrough + per-q LU)
          +  16 · nq_disk · μ · ngkmax / P     # gflat_acc
          (+ 16 · nq  for cct_trace_per_q, transverse only — negligible)
```

The two centroid copies (X-form on `p_x`, Y-form on `p_y`) are distinct
buffers, not views. At a balanced 4×4 mesh each costs ≈ 0.21 GB on CrI3
charge; the previous reference's "centroids one term" framing under-counts
by 2× (agent_3_v2.md §1.1, agent_4_v2.md §2 R1). At CrI3 6×6 80 Ry charge
the total is ~0.93–1.2 GB depending on `n_q_disk` (charge IBZ ≈ 7,
transverse `write_ibz_only=False` → 36). For transverse μ_L > 0,
`gflat_acc` is ~5× larger because the bispinor V_q orchestrator doesn't
yet support IBZ writes (agent_4.md §7.7).

### W_pool

```
W_pool = B · target_utilization − B_persist
```

`target_utilization = 0.80` (current `plan_gflat_chunks` default) is doing
implicit work: at CrI3 this subtracts ~12 GB from the physical 60 GB
ceiling, absorbing uncatalogued cuFFT scratch, NCCL buffers, and cuSolverMp
panels that none of the four agents could pin down with desk math
(agent_2_v2.md §1 "agent_2 §7.11", agent_3_v2.md §2 R8). It is an honest
empirical slack, not a principled term, and should be replaced by per-site
`query_fft_peak_bytes` measurements when those land.

### W_wfn (Peak C, FFT-box term — disputed)

The per-bc band-FFT box inside `to_rchunk`/`psi_G_store.fetch_psi_rchunk`,
shape `(nk_eff, bc, ns, nx, ny, nz)` with `nk_eff = min(nk, kc or nk)`:

```
W_wfn_sharded   = 16 · nk_eff · bc · ns · n_rtot · F_fft / P     (per-rank ideal)
W_wfn_unsharded = 16 · nk_eff · bc · ns · n_rtot · F_fft         (no /P — the CrI3 pathology)
```

The empirical observation at CrI3 80 Ry: `psig_k_chunk_size = 6` is required
to land under budget, and the box bytes match the unsharded formula at that
`kc` (~13.8 GB/rank). Whether this is "XLA refuses to shard the band axis"
(agent_1.md §1f, agent_2.md §3b, agent_4.md §5) or "the r-axis is per-rank
locally dense by construction" (agent_3.md §Q1) is the live dispute B-1
in §5 below. Both readings give the same numeric formula for the binding
case; the difference matters for the fix.

Aliasing with one freed pair-density slot is the **optimistic ceiling**
recipe (reference §5.5: `W_wfn ≤ W_pool / S_pd`). The conservative reading
is summed-not-aliased: `W_wfn + S_pd · α_zeta · cr ≤ W_pool` (`agent_1.md`
§2b, `agent_4_v2.md §3.d`). Without HLO confirmation the team's default is
the conservative form with `target_utilization = 0.80` carrying the slack.

### W_zeta (Peak C, dominant linear-in-cr term)

```
W_zeta = S_pd · 16 · nk · ns² · μ · cr / P                   (pair-density slots)
       + 16 · nk · nb_F · ns · cr / p_y                       (psi_Y_full, see below)
       + 16 · nq · μ · cr / P                                 (Z_q / Z_col, rank-3)
```

The three rank-5 "pair-density-shape" slots are `P_l_R_conj`, `P_r_R`, and
one XLA scratch held simultaneously across `gamma_double_contract`. The
rank-7 reshapes (`P_l_3d`, `P_r_3d`) are bitcast aliases of their parents,
not new buffers (agent_1.md §1c, agent_3_v2.md §2 R4). `S_pd = 3` was
HLO-extracted on MoS2 3×3; whether it scales at CrI3 6×6 80 Ry remains
open (dispute B-3 in §5).

`psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)` is a real catalog
entry that all four agents now agree on (agent_2.md §1 Tier 2, agent_3.md
§1.4a, agent_4_v2.md §1, agent_1_v2.md §2 first bullet). At CrI3 charge
it is ~3.7% of the pair-density-slot sum (agent_2.md §3a), but its
aliasing into a freed pair-density slot is unverified (dispute B-4).

### W_accum (Peak D, separate jit)

```
W_accum = 16 · nq_disk · μ · cr / P             (zeta_chunk carried in)
        + 16 · cs · n_rtot · F_fft_accum         (per-scan-iter FFT box, per-rank, no /P)
        + 16 · cs · ngkmax                       (gather contribution)
```

Peak D is a **separate XLA module** from `fit_one_rchunk`
(`zeta_chunk.block_until_ready()` at `isdf_fitting.py:2152`, agent_4.md
§2.b), so the overall device peak is
`B_persist + max(W_C, W_D)`, not their sum (agent_1.md §2, agent_2.md
end of §2). All four agents predict `cs ≈ several hundred` at empirical
geometries; the recipe's `cs = 64` is conservative or driven by an
`F_fft_accum` that is much larger than 4 for a 1-D `n_rtot ≈ 1.1 M`
batched FFT (agent_2.md §5, agent_3.md §5.4, agent_4_v2.md §2 row 6).

### r_chunk picker procedure

The team's algorithm (agent_4.md §4 has the cleanest pseudocode;
agent_1.md §4b and agent_3.md §4.1 agree on the steps):

1. **Compute `B_persist`** from the equation above with `μ = μ_chan`
   for the active channel. Raise informatively if
   `B_persist > 0.95 · B`.

2. **`W_pool = target_utilization · B − B_persist`.**

3. **Pick `r_chunk` first.** It is the single biggest performance lever
   and depends on no other knob:
   ```
   K1 = S_pd · 16 · nk · ns² · μ / P             (pair-density rank-5)
      + 16 · nk · nb_F · ns / p_y                 (psi_Y_full Y-sharded)
      + 16 · nq · μ / P                           (Z_q / Z_col)
   K0 = W_wfn(bc, kc)                              (if not aliased)
   cr_raw  = floor((W_pool − K0) / K1)
   cr      = clip(cr_raw, P, n_rtot)
   cr      = cr − (cr mod P)                       (sharding divisibility)
   cr      = max(cr, μ_chan)                       (per-iter overhead floor — disputed; §3)
   ```
   The `K1` decomposition tracks agent_3.md §5.3 and agent_1_v2.md §2 R2.

4. **Pick `band_chunk` and `psig_k_chunk_size`** subject to the W_wfn
   ceiling. Conservative form (no aliasing): `W_wfn ≤ W_pool − K1·cr`.
   Optimistic form (one freed slot): `W_wfn ≤ W_pool / S_pd`. Default
   to conservative until HLO confirms.
   - Set `kc = nk` first.
   - If at large `n_rtot · nk` the **unsharded formula** is used,
     halve `kc` until the unsharded box fits (agent_3.md §4.1 step 3,
     agent_4.md §4 gate `sharded_fft_box_holds`).
   - Then largest pow-2 `bc ≤ nb_F` such that `W_wfn(bc, kc)` fits.

5. **Pick `gflat_chunk_size` in the separate Peak D pool.** If
   one-shot `cs = ⌈n_q_disk · μ / P⌉` fits `W_accum ≤ W_pool`,
   prefer one-shot; else
   `cs = ⌊(W_pool − 16·n_q_disk·μ·cr/P) / (16·n_rtot·F_fft_accum + 16·ngkmax)⌋`.

6. **Per-channel re-pick for bispinor** (see below).

### Bispinor per-channel re-pick (settled)

`fit_zeta_to_h5` is called once per `vertex_mu_L ∈ {0, 1, 2, 3}`. Charge
uses `μ_C ≈ 1800` and Cholesky; transverse channels share `μ_T ≈ 1200`
and the per-q LU path with the `cct_trace_per_q` short-circuit.
`α_zeta ∝ μ`, so transverse can tolerate `cr ≈ μ_C / μ_T ≈ 1.5×` larger.
Three of four passes are transverse, so per-channel picking is worth
~30% on wall time per transverse pass.

Agent 1's §4c proposal ("size to charge once, reuse for transverse") was
overruled by agents 2, 3, 4 and accepted in `agent_1_v2.md §2 R6`. Plumb
`μ_chan` into the planner explicitly, call once per channel. Whether the
current code does this is open question C-3 in §6.

## 3. Confirmed bugs in current code

**Bug 1 — `gflat_memory_model.py:184`** (first flagged by agent_2.md §6,
confirmed by agent_3.md §6.1, agent_4_v2.md §3.c, agent_1_v2.md §2):

```python
"centroids_persist":
    2 * _bytes_c128(nk, ns, mu, nk, shard=p_xy),  # L+R approx
```

The fourth dim should be `nb_total` (= `nb_L + nb_R`), not `nk`. At CrI3
6×6 (nk=36, nb≈400) this under-counts by `nb/nk ≈ 11×`. The `shard=p_xy`
divisor is also wrong: each centroid copy is sharded on only one mesh
axis (`p_y` for `psi_rmu_Y`, `p_x` for `psi_rmuT_X`), not on `P`, so the
divisor over-credits the sharding savings by another 4× on a balanced
4×4 mesh. Proposed fix: replace with two separate terms,
`_bytes_c128(nk, nb_total, ns, mu, shard=p_y) + _bytes_c128(nk, mu, nb_total, ns, shard=p_x)`.

**Bug 2 — `nb_L + nb_R` doubling at `gw_init.fit_zeta:588`** (first
flagged by agent_2.md §5 Possibility 6, agent_2_v2.md §2 R-flag,
agent_3_v2.md §2 R5). The planner is handed
`nb_total_chunker = (b3−b0) + (b4−b1)`, which is roughly `2·nb_F` for
symmetric GW (overlapping ranges). This propagates into the `nb_F`-driven
centroid and `psi_Y_full` terms. Proposed fix: pass `max(nb_L, nb_R)`
or be explicit about which band-window axis is intended. Modest impact
on CrI3 charge (~0.4 GB) but flips sign on the `psi_Y_full`
"negligible" claim if `nb_F` is meant to be the sum.

Both are read-only flags here; KNOWN_SANDBOX_ERRORS.md filing is not
needed because these are LORRAX source bugs, not sandbox-infra bugs.

## 4. Disputed but tractable on code-read only

**Where does `r_chunk = 12 500` come from?** Two hypotheses:

- **H-A (agent_3.md §R3, agent_3_v2.md §3 D3):** `gw_init.py:617`
  unconditionally assigns `chunks['chunk_r'] = int(gflat_plan.r_chunk)`,
  so the gflat planner always wins. The 12 500 must therefore be the
  gflat planner's pick at this configuration. If true, the question
  reduces to "why does `plan_gflat_chunks` land at 12 500 instead of
  the 20k–75k that desk math predicts?" — most plausibly the
  `:184` centroid bug inflates B_persist enough to crush W_pool.

- **H-B (agent_1.md §7.9, agent_1_v2.md §3 D3, agent_2_v2.md §2 R2):**
  the `max_chunks = 64` floor in `plan_gflat_chunks` lower-bounds
  `r_chunk ≥ ⌈n_rtot / 64⌉ = 17 578` at CrI3 — so the gflat planner
  *cannot* output 12 500 unless `r_from_budget` is itself small. Agent
  1 §7.1 argues this implies the 12 500 came from `compute_optimal_chunks`
  (its 5-slot `_zct_moment` + `_fft_moment` with the missing `n_bc · α_psi_Y`
  term) or from the AOT chooser. Agent 3_v2 §3 D3 partially walks this back
  on re-reading lines 332–342: the `max(...)` only *upgrades* `r_from_budget`
  when budget is tight; it doesn't cap.

Resolution is two-minute: read the **`gflat_plan.format()` output** in
the lorrax log from any recent CrI3 6×6 80 Ry run (e.g.
`runs/MoS2/00_mos2_3x3_cohsex/A_*_cri3_*/lorrax.log`, or any of the
`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_*` directories in git status).
That single log line prints `(r_chunk, n_r_chunks, bottleneck, HWM,
peak breakdown)` and pins which stage bound. A careful re-read of
`gw_init.py:574–626` would also expose any path where
`_apply_aot_chunk_model` ends up writing `chunks['chunk_r']` before
the `:617` overwrite — but the current code is clear: gflat wins.

## 5. Live experimental disagreements (HLO-dependent)

These three items cannot be resolved by code reading alone; each
requires a fresh HLO `memory-usage-report.txt` from a current CrI3
6×6 80 Ry `fit_one_rchunk` compile.

**B-1 — Is the band-FFT box sharded, unsharded, or per-rank-locally-dense?**
- Hypothesis B-1a (agent_1.md §1f, agent_2.md §3b, agent_4.md §5):
  XLA replicates the band axis across ranks, so per-rank bytes are
  `16 · kc · bc · ns · n_rtot · F_fft` (no `/P`). This is the
  "unsharded pathology" of the reference report §5.8.
- Hypothesis B-1b (agent_3.md §Q1, partially agent_3_v2.md §1.1):
  the box is "per-rank-locally dense in the r-axis by construction"
  — every rank does its own full local cuFFT — so the bytes are right
  but the "factor of P" framing is misleading. The fix in B-1b is
  "shard the band axis at creation"; in B-1a it is "shard the
  *r-axis*-related buffer".
- Resolution: in
  `module_NNNN.jit__make_fit_one_rchunk_kernel.memory-usage-report.txt`,
  grep for the FFT-input thunk in `to_rchunk`. If the per-rank shape
  reads `c128[36, 16, 2, 75, 75, 200]` → B-1a (band replicated). If it
  reads `c128[36, 1, 2, 75, 75, 200]` → B-1b (band sharded, r dense).
  At `psig_k_chunk_size = 6, band_chunk = 16, P = 16`: B-1a → ~83 GB
  per rank at `kc = nk`, ~13.8 GB at `kc = 6`; B-1b → ~5.2 GB at `kc = nk`,
  ~0.86 GB at `kc = 6`. The empirical need for the `kc = 6` mitigation
  argues for B-1a (agent_1_v2.md §3 third bullet).

**B-3 — `pair_density_slots` at CrI3 scale: 3, or larger?**
- Hypothesis B-3a (current default, MoS2 HLO): `S_pd = 3` — the kernel
  body's static slot count under XLA BufferAssignment, geometry-invariant.
- Hypothesis B-3b (agent_4.md §5.b candidate 5, agent_2.md §7.1):
  cuFFT scratch on the rank-7 IFFT/FFT in `z_q_from_psi_sm` may force
  more concurrent scratch slots on a 75×75×200 box than on a small
  MoS2 box. `S_pd_effective` could be 5–8 at CrI3.
- Hypothesis B-3c (agent_1_v2.md §3 D2): the static slot count is
  invariant; the *size* of one slot grows with cuFFT scratch on a big
  box. Mechanistically this is a per-slot `F_fft` factor, not a slot
  count.
- Resolution: in the same HLO report, count distinct preallocated-temp
  slot lifetimes whose shape class matches
  `c128[nk, ns, ns, μ_loc, cr_loc]`. If count = 3 with one slot's bytes
  inflated by cuFFT, B-3c wins. If count > 3, B-3b wins.

**B-4 — Does `psi_Y_full` alias into a pair-density slot, or sum with them?**
- Hypothesis B-4a (agent_3.md §2.3): disjoint at shard_map entry,
  aliasable, drop the term.
- Hypothesis B-4b (agent_1.md §2a, agent_2_v2.md §3 D2): the input
  buffer is live during the einsum that allocates the first
  pair-density slot — they sum.
- Resolution: at the `c_q_from_psi_sm._local` / `z_q_from_psi_sm._local`
  shard_map boundary, look at the input buffer's last-use vs the rank-5
  output's allocation. Sum: ~3.7% extra on `α_zeta` at CrI3 charge
  (agent_2.md §3a) — small but real. Alias: drop from the model.

## 6. Universal open questions and prioritized experiments

Cross-cited across all four v2 reports. The order is the team's
recommended sequence — earlier items unlock later ones.

**C-1 (P1). Capture the CrI3 planner log.** Read
`gflat_plan.format()` output from any recent CrI3 6×6 80 Ry run
(see `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_*/lorrax.log` or
`runs/MoS2/00_mos2_3x3_cohsex/A_*_cri3_*/lorrax.log`). Pins
`(r_chunk, bottleneck stage, HWM, peak breakdown)` — settles whether
the 12 500 reproduces, which stage binds, what the planner thinks
W_pool is. **Resolves §4 H-A vs H-B and quantifies the bug-1 impact.**

**C-2 (P1). Dump one `fit_one_rchunk` HLO memory-usage-report.**
```bash
XLA_FLAGS="--xla_dump_to=/tmp/hlo_cri3 --xla_dump_hlo_pass_re=.*"
lxrun python -m gw.gw_jax -i cohsex.in   # one r-chunk and bail
grep -A 2 "preallocated-temp" /tmp/hlo_cri3/module_*.jit__make_fit_one_rchunk_kernel*.memory-usage-report.txt
```
Single-shot resolves disputes **B-1, B-3, B-4** by:
- counting pair-density-shape slots → B-3;
- reading FFT-box per-rank dims → B-1;
- checking the input-buffer lifetime at the shard_map boundary → B-4.

**C-3 (P1). Bispinor per-channel re-pick verification.** Read the
bispinor outer loop in `gw_jax.py` and confirm whether `meta.n_rmu`
flips between channels before each `fit_zeta_to_h5` call.
Trivial code-read (~30 min). If the planner is invoked once with
`μ_C`, transverse passes are over-budgeted by `(μ_C/μ_T)² ≈ 2.25×`
on W_zeta and lose ~30% wall time each. **Resolves the per-channel
design that all four agents flagged.**

**C-4 (P2). cuSolverMp scratch measurement.** At CrI3 distributed
scale, snapshot device memory across one `potrs` / `getrs` call.
Probably `O(μ²)`-class per rank, ~MB at current sizes; only matters
asymptotically. Agent 2 §3c gives a numerical bound; agent 3 §1.3
flags `k_factor ∈ [2, 4]` as the unknown. **Pins the magic constant
in §3.b's `W_solve` term.**

**C-5 (P2). Actual `ngkmax` from `zeta_cutoff_ry`-derived sphere at
CrI3 80 Ry.** `gw_init.py:596` uses `0.06 · n_rtot ≈ 67 500` as a
fallback; agents 1, 3, 4 vary their plug-ins between 30 000 and
67 500. Affects `B_persist` (gflat_acc) by up to 2× and matters for
the V_q pass. Hoist the sphere construction or thread `ngkmax_actual`
in. **Closes agent_1.md §7.7, agent_3.md §Q9 / Q11.**

**C-6 (P2). One-shot test of `r_chunk_size = 16` (= P).**
Agent 3 §Q4 argues the `r_chunk ≥ μ` floor in `compute_optimal_chunks`
and `gflat_memory_model.py:326` is folklore (a misreading of the
Σ_μν analogy); the real lower bound is the divisibility floor
`cr ≥ P`. Agent 1_v2 §3 D1 disagrees: the floor is an algorithmic
per-iter overhead bound, not a divisibility one. **Cheap smoke test:**
set `r_chunk_size = 16` in cohsex.in on MoS2 3×3, run; if completes
with reasonable wall time, drop the floor; if it crashes or thrashes,
document the algorithmic reason.

**C-7 (P3). Per-call-site `query_fft_peak_bytes`.** Replace the
single `fft_factor = 4.0` with three site-specific values from
`common.fft_helpers.query_fft_peak_bytes` calls at planner time
(Peak A loader, Peak C bc-FFT, Peak D accumulate FFT). One AOT
lowering per site at startup, ~100 ms total. Expected:
`F_fft_loader ≈ 4`, `F_fft_kshmap ≈ 4–5`, `F_fft_accum ≥ 6`. **Closes
the entire "why does my prediction miss by 2–4×" thread.**

Suggested order: C-1 → C-2 in parallel; C-3 in parallel (cheap
code-read); then C-7 to systematically deflate the F_fft uncertainty;
then C-4, C-5, C-6 in any order.

## 7. Implementation guidance

The team's consensus on the rewrite (agent_1.md §6, agent_2.md §6,
agent_3.md §6, agent_4.md §6, all four v2 critique sections):

**Keep.** `gflat_memory_model.plan_gflat_chunks`'s overall A/B/C/D
peak structure — it is the right shape and the right abstraction.
The aot_memory_model's per-kernel cost catalog provides the right
*calibration mechanism* (AlphaFit + DoE), even if its full closed-form
chooser is heavy for the maintenance footprint.

**Consolidate.** Retire `compute_optimal_chunks`'s 6-stage moment
inversion. Its `_zct_moment` counts 5 slots (legacy decomposed-chain
era) where the current monolithic shard_map has 3; its
`_fft_moment(n_bc=…)` *correctly* counts the cumulative
`n_bc · α_psi_Y_bc · cr` term that `gflat_memory_model` is missing;
preserve that piece of math as input to the rewritten `K1`. Keep
`query_fft_peak_bytes` integration. Drop `_apply_aot_chunk_model` as
a shadow path unless the AOT chooser is the planner.

**Add.** (i) Per-channel `μ_chan` plumbing so the planner is called
once per `vertex_mu_L` (agent_3.md §6.3 sketch, agent_4.md §4); (ii)
`shard_holds: bool` gate for the FFT box, with conservative default
at large `n_rtot · nk · bc · ns · 16` (agent_3.md §4.3, agent_1.md
§4d); (iii) a `W_vq` peak so `vq_g_chunk_size` falls out of the same
chunker rather than `_pick_g_chunk(ngkmax)`'s 4096 cap (agent_2.md
§7.9, agent_3.md §4.1.7); (iv) an HLO-calibration regression test
that fails loudly when `pair_density_slots` drifts (agent_3_v2.md §5
N6, agent_2_v2.md §5 S6); (v) the §3 bug fixes.

**Remove.** The 50/50 `W_wfn` vs `W_zeta` split in current
`plan_gflat_chunks:297` (hand-tuned, not principled — replace with
the conservative summed form pending HLO of dispute B-1). The
opaque `max_chunks = 64` floor unless C-1 shows it binds.

Reference v2 sections for proposed structure:
`agent_4.md §4` (cleanest pseudocode), `agent_3.md §6.3` (per-channel
API sketch), `agent_2_v2.md §5 S5` (per-site fft_factor).

## 8. Pointers

- **`agent_1.md`** (941 lines) — most thorough tensor catalog with
  bc-unroll cumulative-live cost, explicit "5b candidate explanations"
  for the 12 500 mystery, and the `max_chunks = 64` floor observation.
- **`agent_1_v2.md`** — Agent 1's revisions: accepts `psi_Y_full` as
  a catalog entry, accepts the `:184` source bug, holds line on the
  `cr ≥ μ` floor (D1) and on B-1a (band axis replicated).
- **`agent_2.md`** — most surgical, every tensor pinned to a line
  number. First to catch the `gflat_memory_model.py:184` `nk`-typo and
  the `nb_L + nb_R` doubling at `gw_init.py:588`. `target_utilization`
  framing.
- **`agent_2_v2.md`** — accepts agent 1's bc-unroll term, agent 4's
  cohsex-override observation; corrects own R3 reading and lays out
  the cuSolverMp vs shard_map fallback branch.
- **`agent_3.md`** — cleanest closed-form derivation
  (`K0 + K1 · cr` decomposition in §5.3), per-channel `μ` plumbing
  sketch, F_fft back-fit at ≈ 5. Has a known ~50× arithmetic error
  in §5.1 B_persist (corrected in v2).
- **`agent_3_v2.md`** — corrects the v1 arithmetic; settles the
  `max_chunks` floor disagreement on close re-read of
  `gflat_memory_model.py:332-342`; promotes per-channel
  re-picking as agreed-on.
- **`agent_4.md`** — cleanest pseudocode (§4), the `cohsex.in`-override
  observation in §7.10 that explains the 12 500 comparison is not
  apples-to-apples, MoS2 sanity check.
- **`agent_4_v2.md`** — consolidated 15-item OQ table cross-referenced
  to all four agents, six prioritised next-steps, holds line on
  per-channel re-pick against agent 1's "size to charge once".
