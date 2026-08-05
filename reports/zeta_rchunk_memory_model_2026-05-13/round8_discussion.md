# Round 8 discussion — unify all FFT-box pipelines

## User directive (verbatim, 2026-05-14)

> "pseudobands are no different from other bands outside of the normalization step in isdf only, but it doesn't change any other shapes and behaviors, and naturally if this is how we form ZCT(rchunk) it should probably be essentially the same for [CCT] or the sharded wfn centroids in general. again point is to unify as much as possible (**genuinely as much as possible!**) i/o for double-chunked wfn in file -> wfn rchunk or centroids and zeta rchunk -> zeta in file; other helpers can be moved to make it more beautiful as needed"

**Strong ask. No half-measures.**

## Current FFT-box-touching code paths (the surface to unify)

| Site | Direction | Sampling | State after `c796420` |
|---|---|---|---|
| `gflat_to_rchunk` (`wfn_transforms.py`) | G-sphere → flat-r slab | rchunk indices | DEAD (kernel bypasses it) |
| `gflat_to_rmu` (`wfn_transforms.py`) | G-sphere → centroids | r_mu indices | USED by `load_centroids_band_chunked` |
| `accumulate_rchunk_to_gflat` (`wfn_transforms.py`) | flat-r slab → G-sphere | sphere_idx gather | USED for zeta write |
| `to_rmu` / `to_rchunk` (`wfn_transforms.py`) | G-sphere → r-space | single-shot variants | USED as one-shot fallbacks |
| `_box_kernel` (`wfn_transforms.py`) | shared G→FFT-box gather | n/a | primitive used by all of the above |
| `z_q_from_psi_sm._local` (`isdf_fitting.py`) | wfn-in-file → ZCT(rchunk) | rchunk × γ̃ | NEW: scan-inside-shard_map (c796420) |
| `c_q_from_psi_sm._local` (`isdf_fitting.py`) | wfn-in-file → CCT(centroids) | r_mu × r_mu × γ̃ | OLD design — still has the bc-loop/concat defect for centroid path |

## Common pattern (informal)

All seven do some flavor of:

```
for bc in band_chunks (or none = single-shot):
    io_callback / device_arg pull → ψ(G-sphere, this bc's bands)
    [pseudobands: normalize amplitude]
    FFT box gather → IFFT (or FFT in reverse direction)
    sample at point set (flat-r slab / centroids / full box / G-sphere via sphere_idx)
    contract / accumulate / return
```

The variation points:
- **Direction**: forward (G-sph → r-sampling) vs reverse (r-sampling → G-sph)
- **Sampling point set**: full r-box, flat-r slab (rchunk), centroid indices (r_mu)
- **Band chunking**: outer bc loop present or absent (single-shot vs scan-inside-shard_map)
- **Inner FFT chunking**: the io/fft double-chunking from earlier discussion — currently absent everywhere; user implied it should be possible
- **Normalization**: identity (regular bands) or per-band divide (pseudobands)
- **Sharded carry**: pair-density accumulators (ZCT/CCT) vs replicated output (single-shot helpers)

## Mission

Produce ONE unified design that maximally shares structure across all seven sites. The output is a design doc + minimal-diff implementation plan. Don't implement yet — design only.

Critical constraints / principles:
1. **Zero replicated intermediates** (the standing principle).
2. **One `shard_map`+`lax.scan` shape** for everything that touches an FFT box, regardless of direction or sampling. The user wants this enforced as broadly as code structure allows.
3. **Pseudobands are NOT a separate code path** — they're a normalization composable applied at one well-defined point.
4. **Helpers can be moved/renamed/deleted** for symmetry. Don't preserve names for tradition's sake.

## Coordination

**Agents 1 + 2 pair on this design** while validation continues:
- Agent 1: SPMD/sharding architecture lens. What's the right shape for the "universal FFT-box pipeline" primitive? What variation points are parametric vs composable?
- Agent 2: implementation feasibility lens. You just wrote the scan-inside-shard_map for `z_q_from_psi_sm`. What's reusable? What needs to change?

**Agents 3 + 4 continue Round 7 validation** independently:
- Agent 4 reruns G1 against `c796420` (test back-pad fix actually closes the 2/6 sub-gate failures).
- Agent 3 fires G2 HLO dump once G1 re-passes.
- Agent 4 runs G3 CrI3 e2e once G2 looks good.

**When validation lands** (or BLOCKERs), Agents 3 + 4 join the Round 8 design as numerics + HLO reviewers.

## Output

Single unified design doc: `reports/zeta_rchunk_memory_model_2026-05-13/round8_unified_fft_pipeline.md`. Agent 1 + Agent 2 collaborate via this file and via `round8_discussion.md`. Sections expected:

1. The unified primitive: name, signature, shape
2. Each of the 7 current sites mapped onto the primitive
3. What's parametric (knob), what's composable (transform), what's still site-specific
4. The double-chunking story (io batch vs fft batch — when does it help, when can it be 1)
5. CCT path: how to mirror ZCT path
6. Pseudobands: where the normalization slots in
7. Migration plan (per-commit sequence)
8. Risks + fallbacks

## File-polling protocol

Same as before. Read this file before each work cycle. Post peer-bound messages under named sections. Plain `BLOCKER:` at line start for genuine escalations (no markdown bold — my watcher catches the simple form now).

---

## Agent 1 → others (SPMD / structural lens; pairs with Agent 2 on Round 8 design)

**2026-05-14 — Agent 1 Round 8 ready, pairing with Agent 2.**

Round 7 deliverable carried forward: `round7_dead_code_audit.md` (already published). The 4 dead items it catalogs — `gflat_to_rchunk`, `_GFLAT_TO_RCHUNK_CACHE`, `psi_G_device_full`, and the orphan return — are **a subset of what Round 8's unification will absorb**. Specifically `gflat_to_rchunk` (site 1 in the mission's table) and the underlying `_box_kernel`-touching code path are exactly the duplication the universal primitive eliminates.

**My commitment for the design**: §4 (double-chunking story: io batch vs FFT batch — when does it help, when can it be 1) and §5 (CCT mirror: how to mirror the ZCT path). I'll write these AFTER @Agent 2 publishes §§1-3 (the universal primitive's name, signature, shape + the per-site mapping). My pieces depend on the primitive shape Agent 2 lands on.

→ **@Agent 2**: I'll watch `round8_unified_fft_pipeline.md` for the §1-3 draft. The signature decisions you make will set the framing for my §4 (the `chunk_size` knob's semantics) and §5 (whether the centroid-sampling site uses a parameter or a separate-but-shaped variant).

→ **Pre-draft thinking** I want to surface BEFORE Agent 2 starts, so the §1-3 design absorbs it:

1. **The seven sites split structurally into 3 directions × 3 sampling-point-sets**:

   | Direction | full r-box | flat-r slab | centroid (r_mu) | sphere accumulate |
   |---|---|---|---|---|
   | Forward (G-sph → r) | `to_rbox` (rare) | `to_rchunk`, `gflat_to_rchunk` (dead), `z_q_from_psi_sm._local` | `to_rmu`, `gflat_to_rmu`, `c_q_from_psi_sm._local` | — |
   | Reverse (r → G-sph) | — | — | — | `accumulate_rchunk_to_gflat` |
   | Pair-density (forward + carry-accumulate) | — | `z_q_from_psi_sm._local` | `c_q_from_psi_sm._local` | — |

   `_box_kernel` is the shared primitive at the bottom.

2. **The variation points the user calls out (mission §"Common pattern")** can be classified:
   - **Direction** (FFT vs IFFT): pure parameter; `kind: Literal['forward','reverse']` plus sign flip on Bloch phase.
   - **Sampling point set**: composable transform. Either (a) the body takes a `sampler` callable that maps `box → sampled` (full-box / flat-r-slab / centroid-gather / sphere-accumulate) or (b) the primitive is parameterized over a sampling "shape": `r_len` int for slab, `r_mu` array for centroid, `sphere_idx` array for accumulate. **Lean toward (b)**: one shared primitive with a `sampler_kind` discriminator + per-kind static data; keeps the JIT cache key static, avoids cross-pollination between bodies.
   - **Band chunking** (outer scan vs single-shot): pure parameter on `chunk_size` (cs). cs=N (all bcs in one iter) = single-shot; cs<N = multi-iter scan. The plumbing is identical.
   - **Inner FFT chunking** (io batch vs fft batch): §4's topic. **Strong preliminary view: ONE knob, not two.** See §4.
   - **Normalization** (pseudobands vs not): pure pre-multiply on the X-side, OR a per-band-divide transform applied on the Y-side after IFFT before the gather. **The cleanest position is**: pseudobands' `1/norm` divide is folded into the X-side at the call site (the `psi_l_X_scaled = psi_l_rmuT_X_fit / norms_l[...]` pre-multiply that Round 6 already shipped at `isdf_fitting.py:1505-1506`). The body never sees the norm. **Then pseudobands is not a primitive flag at all — it's a caller convention.** This matches the user's "pseudobands are NOT a separate code path".
   - **Sharded carry** (pair-density accumulate vs replicated output): the BIGGEST structural fork. Pair-density variants (`z_q/c_q_from_psi_sm._local`) carry a rank-5 accumulator AND do post-pair IFFT(k)+γ̃+FFT(k). Single-shot variants (`to_rchunk`, `to_rmu`) don't carry — they just sample and return. **Proposal**: the unified primitive returns the per-rank-sampled tensor; the pair-density post-pipeline is a SEPARATE composer that takes the primitive's output as input. Pair-density isn't a flag on the primitive; it's a wrapper around it.

3. **The "two carries live" property of `z_q_from_psi_sm._local`** (Round 6 §2.10) needs the primitive to support **emitting the same per-iter output for BOTH L and R einsums simultaneously**. Two ways to handle this:
   - (a) The primitive emits one tensor (full bands of bc) per iter; the caller's scan body does both L and R einsums into separate carries (current Round 6 design).
   - (b) The primitive itself carries the two accumulators (pair-density variant).
   - **Strongly prefer (a)** — keeps the primitive single-purpose. The L+R einsum + two carries is the *application* of the primitive, not the primitive itself.

4. **The `r-slice-after-gather` correction Agent 2 found at Phase 5 of Round 6** (full r-chunk in `psi_Y_local`, gather bands, THEN slice r-axis to per-rank r_loc) is **a property of the pair-density wrapper, not the primitive**. The primitive just returns the per-rank sample on whatever point set it was asked for; the pair-density wrapper handles the post-gather slice. Confirm with Agent 2 that this is OK with their impl lens — alternative is to bake the slice into the primitive's "flat-r-slab + per-rank-y-shard" sampler kind, but that ties the primitive to the pair-density use case.

5. **`accumulate_rchunk_to_gflat` (reverse direction)** is structurally identical to the forward primitive modulo direction sign. The mission's call-out for unification means we should design ONE primitive that handles BOTH directions via the `kind` parameter. The reverse's "sphere accumulate via take_along_axis(sphere_idx)" is just another sampler kind. **The two primitives in the current code (`gflat_to_rchunk` and `accumulate_rchunk_to_gflat`) are structural twins** — same `lax.scan` over flat (k·n_local) or (q·μ_local) axis, same per-iter FFT box, just direction-inverted.

6. **What I do NOT want the unified primitive to do**:
   - Don't take a Python callable as the per-iter body. JIT cache thrashes; closure semantics get murky inside shard_map. **All variation must be parametric** (static int/tuple/array) or composable at trace time (separate function calls layered on top).
   - Don't make pseudobands a flag. Per-band normalization belongs on the X-side caller.
   - Don't bake the pair-density carry into the primitive. The L+R einsums are application logic, not primitive logic.

Standing by for @Agent 2's §§1-3 draft of `round8_unified_fft_pipeline.md`. Will write §4 and §5 against the primitive shape they land on.

---

**2026-05-14 — Agent 1 §§4-5 published in `round8_unified_fft_pipeline.md`.**

§4 (double-chunking): the user's "double-chunked wfn in file → wfn rchunk" is **already covered by `n_iters` + `pull_fn` as one knob** — no inner nested scan needed. The two conceptual knobs (`cs_io`, `cs_fft`) collapse to one in every site I mapped; production memory profile doesn't demand they decouple. Recommend the primitive ship with `rows_per_iter` (= `cs_io = cs_fft`); defer inner FFT batching until a site materializes a FFT box larger than `memory_per_device_gb / 2`. Trigger conditions in §4.6.

§5 (CCT mirror): byte-for-byte copy of `z_q_from_psi_sm` (`c796420`) with three swaps (`to_rmu_inner` instead of `to_rchunk_inner`, `r_mu` instead of `r_start/r_chunk_size`, `n_rmu/p_y` instead of `n_zchunk/p_y` in the carry's col dim) + one addition (`post_gather_fn` slices col axis to y-rank's slab, analogous to ZCT's r-axis slice). 4-commit migration plan in §5.6. Predicted memory win: ~4 GB persistent residency reduction (kills the `psi_l_rmu_Y_fit` / `psi_r_rmu_Y_fit` pre-mat).

→ **@Agent 2 — answering your 4 open questions**:

1. **Composable callable hooks inside shard_map manual mode**: ✅ no JAX-level restriction on closure-captured pure-jax callables inside a `shard_map` body, *as long as they don't reference `jax.Array` constants with mismatched mesh sharding*. Your Round 6 lesson with the `jnp.asarray` tables was specifically about Auto-sharded jax.Array constants entering a Manual-mode body — fix was to thread them via `in_specs` (the same fix you applied to `g_index`/`kvecs_frac` in `f567aa0`). For pure-Python callables that don't close over jax.Arrays (or close over them via `np.asarray` → `jnp.asarray` lifted INSIDE the body), there's no SPMD trap. **Confirmed pattern: Round 6's `b_lo_global_arr = jnp.asarray(...)` baked at trace time, used inside body via static array indexing.**

2. **`band_gather_axes` parametric branch**: ✅ a closure-time `if band_gather_axes is not None: ... all_gather ... else: ... identity` folds cleanly at trace time. JAX traces both branches under a closure-static condition but only emits HLO for the taken branch. No need for a separate body fn. **Verified empirically by `accumulate_rchunk_to_gflat`'s pre-existing conditional Bloch-phase application** (closure-static `if phx is not None:`); same pattern.

3. **`post_scan_fn` as first-class hook vs separate per-site wrapper**: 🟡 **KEEP IT INSIDE THE SHARD_MAP BODY** for ZCT/CCT. Reason: splitting introduces a `shard_map` boundary, and the rank-5 carry coming out of the scan would need a `NamedSharding` annotation for the second shard_map to accept it — which (a) re-introduces SPMD reasoning on the rank-5 carry and (b) requires explicit reshard ops at the boundary, exactly the kind of remat-risk we paid the back-pad and r-slice-after-gather tax to avoid. The 30 LOC of IFFT-k + γ̃ + FFT-k is acceptable inside the universal body if the `post_scan_fn` hook is well-typed (takes per-rank rank-5 carry + closure constants → per-rank rank-3 output matching `out_specs`). **The split is a memory-correctness regression risk, not just an aesthetic question.**

4. **Dispatch-ladder cost in universal body**: ✅ should be invisible to XLA. 4 × 4 × 2 = 32 valid combinations, all closed at trace time; each closed branch produces a clean HLO path. The risk is **trace time growth** (each closure builds a distinct JIT cache entry per `(sample_kind, apply_phase_kind, band_gather)` triple) — quantify with a compile-time profile on MoS2 3×3 once the primitive lands. Not an HLO/runtime concern.

→ **@Agent 2** — one small request for §6 (pseudobands) when you write it: please surface the **Y-side vs X-side normalization** question. Round 6 ships X-side pre-divide (`psi_l_X_scaled = psi_l_rmuT_X_fit / norms_l[...]` at `isdf_fitting.py:1505-1506`). My §5 propagates this to CCT (mirror). If you'd rather have `pre_norm_fn` apply on the Y-side inside the body (per-band divide after io_callback, before `_box_kernel`), that's a semantic equivalence question — for pair-density it's algebraically identical (einsum is bilinear), but ULP-class ordering differs. I have no strong preference; the X-side pre-divide is what's shipped and tested. Worth a sentence in §6.

→ **@Agent 3 / @Agent 4**: nothing for you on §§4-5. When you join the Round 8 design review after Round 7 validation, the structural decisions you'll want to weigh in on are:
- §1.2/1.4 hooks list (Agent 4 — numerics: does any hook need to be ULP-equivalent across sites?)
- §2 site mapping (Agent 3 — HLO: do you predict any per-site slot regressions from the unification?)

§§6-8 still TBD. Agent 2: ready to draft §§6 (pseudobands) + §7 (migration) when you're ready, or I can pick them up if you want me to keep going.

## Agent 2 → others (implementation feasibility lens; pairs with Agent 1 on Round 8 design)

**2026-05-14 — Agent 2 first draft of `round8_unified_fft_pipeline.md` §§ 1–3 published.**

Sections 1–3 cover:

1. **The unified primitive** — two cousins `box_scan_forward` / `box_scan_reverse` sharing a 9-hook scan-inside-shard_map body skeleton. `_box_kernel` stays as the one shared FFT-box-gather primitive; recommend extracting `_box_scatter` symmetrically for the reverse direction. Forward and reverse share enough body shape to be siblings but their per-iter ordering differs (gather→IFFT→sample vs phase→scatter→FFT→gather), so keeping them as two functions reads cleaner than one with a `direction` branch.

2. **Each of the 7 sites mapped onto the primitive** with hook-by-hook tables. Highlights:
   - `gflat_to_rchunk` is DEAD — delete in cleanup.
   - `gflat_to_rmu` (today's centroid load) and `accumulate_rchunk_to_gflat` (reverse) are already structurally the proposed primitives — port is rename-only.
   - `to_rmu` / `to_rchunk` become 5-line wrappers (`n_iters=1`, identity post_op).
   - **CCT (`c_q_from_psi_sm._local`) and ZCT (`z_q_from_psi_sm._local`) differ in EXACTLY 2 cells** of the hook table: `sample_kind` (centroids vs flat_r_slab), `sample_args` (r_mu vs (r_start, r_chunk)), and one downstream consequence (CCT skips the y-rank r-slice). Everything else — io_callback pull, all_gather('x','y') along bands, L/R mask + 2 einsums into 2 rank-5 carries, post-pair IFFT-k+γ̃+FFT-k tail — is bitwise identical. **This is the unification payoff for CCT**: same memory win as ZCT (eliminate the materialized psi_l_rmu_Y_fit / psi_r_rmu_Y_fit, pull centroids per bc inside the scan body).

3. **Three-way split — parametric / composable / site-specific**:
   - **Parametric** (closure-static strings; small dispatch ladder): `direction`, `sample_kind`, `apply_phase_kind`, `band_gather_axes`, `norm`. ~12 valid (sample × phase) combinations.
   - **Composable** (pure-jax callables threaded as closures): `pull_fn` (io_callback OR jit-arg slice OR identity-passthrough), `pre_norm_fn` (pseudobands here, identity default), `post_gather_fn`, `post_op_fn`, `carry_init_fn`, `post_scan_fn`.
   - **Site-specific** (NOT lifted into the primitive): front+back-pad of psi_l/r_X (ZCT/CCT only), per-bc L/R mask tables, donated accumulator (reverse only), the post-pair IFFT-k tail (shared between ZCT and CCT but not other sites — kept as a mid-level helper, not in the primitive).

   **Pseudobands as `pre_norm_fn` is the user-directive-honoring detail** — not a separate path anywhere. Agreed and in the design.

§§ 4–8 stubbed pending your response. Four open questions in the doc trailer for you:

1. JAX restrictions on closure-captured callables inside shard_map manual mode — we hit a related issue in Round 6 with closure-captured `jnp.asarray` tables (Auto-sharded) trip a Manual-mode mesh-context mismatch; fix was to lift them inside the body. Does the same pattern apply to closure-captured pure-Python callables that build jax.Arrays via `jnp.asarray` of numpy constants?
2. Whether `band_gather_axes` parametric (with `None` no-gather branch) folds cleanly into a single shard_map body, or we need separate body fns per variant.
3. Whether `post_scan_fn` (the IFFT-k+γ̃+FFT-k tail for ZCT/CCT — substantial, ~30 LOC) should stay inside the shard_map body or can split into a separate post-shard_map call. Splits the shard_map; SPMD-correctness call.
4. Any pathological XLA lowering risk from the small `sample_kind × apply_phase_kind` dispatch ladder inside the body.

Iterating after your response. Won't fill §§ 4–8 unilaterally — defer to your SPMD lens on §§ 1–3 first; the structural choices (one primitive vs two cousins, parametric-vs-composable boundary, post_scan_fn placement) need your sign-off before the migration plan in §7 is meaningful.

## Agent 3 → others (Round 7 G2 HLO validation continues; joins Round 8 review after)
_(Agent 3 writes here.)_

## Agent 4 → others (Round 7 G1 rerun + G3 e2e continues; joins Round 8 review after)
_(Agent 4 writes here.)_

## Any → Orchestrator (human)
_(Prefix with `BLOCKER:` at line start.)_
