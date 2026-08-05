# Round 5 discussion — scan-inside-shard_map forward helper (unified plan)

## Mission

Produce **one unified plan** for replacing the current `gflat_to_rchunk` + concat-with-reshuffle integration with a **scan-inside-shard_map** design that:

1. Eliminates the block-cyclic → contiguous band-axis reshuffle (heavy cross-device communication).
2. Eliminates the `psi_G_device_full` lazy-property tracer-leak bug.
3. Eliminates the `('x','y')` → `'x'`/`'y'` reshard remat at the consumer boundary (~30 GiB at CrI3 80 Ry — Agent 1 Round 4 §4).
4. Stays within the agents' Round-3 `agent_2_structural_fix.md` §4c-d design that was DEFERRED (not the §4a-b flat-axis variant that was actually implemented).

Goal: a plan thorough enough that Round 6 implementation is mechanical — every challenge addressed, every edge case named, every validation gate specified.

## Working principle (reminder)

**Zero replicated intermediates or work in this pipeline.** Defects are bugs to FIX, not resources to budget. Don't propose a design that line-items a remat or a global gather.

## Background you all already have

- `agent_2_structural_fix.md` §4 — original scan-inside-shard_map sketch (deferred during Round 3 implementation in favor of the now-buggy flat-axis design).
- `round4_improvements.md` (Agent 1 R4) — the 30 GiB double-materialization-of-`psi_Y_full` finding that motivates this round.
- `round4_memory_model_state.md` (Agent 3 R4) — the planner's accuracy + bookkeeping fixes.
- `psi_G_store.py:298-314` — the current buggy concat reshuffle + tracer-leaking lazy cache.
- `solve_zeta` lines 1119-1141 — the prior-art comment about WhileOp/SPMD trap that originally scared us off scan-with-sharded-carry. Why doesn't that apply here? (Hint: shard_map's `out_specs` boundary.)

## File-polling protocol

You read this file at the start of each work cycle. Write peer-bound messages under the named sections below. **Iterate to convergence** — don't just publish your initial analysis and walk away. If another agent's reasoning has implications for your lens, respond. The synthesis is in `round5_unified_plan.md` (Agent 4 owns; others contribute).

---

## Agent 1 → others (SPMD / shard_map / mesh-semantics lens)

**First-cut SPMD analysis published**: `round5_agent_1_spmd.md` (9 sections, ~600 lines). Headline conclusions:

1. ✅ **Scan-inside-shard_map is SPMD-safe by construction.** The WhileOp/SPMD trap from `solve_zeta` doesn't apply because shard_map's body is the *manual partition primitive* — SPMD partitioner has already stopped at the `in_specs`/`out_specs` boundary. The scan inside lowers to a WhileOp **within one rank's program**, with rank-local carries and no sharding annotations for SPMD to inflate. In-repo precedent: `accumulate_rchunk_to_gflat` does exactly this with a rank-3 carry and Round 4 HLO shows it works (FFT box aliased single-slot).

2. ⚠️ **One correctness gotcha I want everyone to internalize before implementation**: the body's accumulator `r_loc` dim must equal `n_zchunk / p_y`, NOT the full r-chunk extent. Why: `out_specs=P(None, 'x', 'y')` requires the per-rank output to be `(nq, mu_loc, r_chunk/p_y)`. Each rank pulls only its 1/p_y r-slab via per-rank `r0_local = r0 + axis_index('y') * r_loc`. The FFT is still local (FFT axes replicated by `in_spec`); only the output slice is per-rank. **Naive implementation will mis-size the carry.** Detailed in §3 of my doc.

3. ⚠️ **Novel composition risk**: io_callback inside scan inside shard_map — JAX supports each independently but I haven't seen the three composed in this repo. **Reproducer FIRST, kernel rewrite SECOND.** §1 risk 5 + §9.

4. ✅ **Mask approach over slice approach** for L/R per-bc band windows. Pointwise `jnp.where` on a rank-local band axis — no SPMD reshape. ~50% einsum FLOPS overhead is acceptable for a structural fix.

5. ✅ **`psi_l_X` per-bc slicing is purely local** — band axis is replicated under `P(None, 'x', None, None)`, so `dynamic_slice_in_dim` on it doesn't trigger any cross-rank op. Watch out: the tables index into psi_l_X's L window (not the full band axis); easy to get the offset wrong.

6. ✅ **Post-pair pipeline tail is byte-identical to today's `z_q_from_psi_sm._local`** — only the front of the body changes.

→ **Agent 2**: see my §8 questions on `ordered=True` semantics and `bpd_max` closure timing for `_slice_local_tile_bc`. Worth aligning before I bless the io_callback details.

→ **Agent 3**: my prediction for the HLO after this lands — §8. If you predict differently from the buffer-assignment lens, please push back; we want both lenses agreeing before Round 6 starts.

→ **Agent 4**: §3 (out_spec gotcha) and §9 (open risks) are the load-bearing pieces — please make them prominent in the unified plan. The scan/shard_map invariants in §7 are the SPMD-side acceptance gate. Numerics-side I think you and I agree: scan-sum is at most ULP-class drift vs the global einsum; rtol=1e-10/atol=1e-12 is the right gate.

---

**Round 2 response — Agent 3 caught two of my errors. Retracting + correcting.**

@Agent 3 is right on both counts:

1. **Carry size**: I had `μ_loc = 376/p_x_p_y = 94`, but `psi_l_X`'s spec is `P(None, 'x', None, None)` so μ is sharded on `'x'` only (p_x=4) → **μ_loc = 376, not 94**. Carry shape is `c128[36, 2, 18412, 376, 2]` = **14.85 GiB per accumulator**. Two carries simultaneously live = 30 GiB. I'm NOT proposing a sharding change (that would require reorganizing the upstream centroid loader); we accept the 14.85 GiB carry. My §3 of `round5_agent_1_spmd.md` is wrong and I'll update it.

2. **r_loc**: same error. r on `'y'` (p_y=4) → `r_loc = r_chunk / p_y = 73648/4 = 18412`, not 4603.

**But — there's a second issue I now see that compounds Agent 3's FFT-box concern**: the host tiles are **band-flat-sharded across the full ('x','y') mesh** (`_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)`). So when io_callback returns this rank's bc tile, the rank only has `bpd_per_bc = bc_size / P` bands of that bc (e.g., CrI3 bc_size=16, P=16 → **1 band per rank per bc**). But the einsum `'kmna, knbr → karmb'` contracts over the band axis, and `psi_l_X` has bands REPLICATED (under `P(None, 'x', None, None)`). **Mismatch.** The rank's einsum would only sum 1/P of the bands → partial result. To get the correct sum, we need either:
  - (a) `jax.lax.all_gather(psi_Y_bc_local, axis_name=('x','y'), axis=1, tiled=True)` before the einsum — per-iter gather of small data (~80 MB/rank/iter for CrI3), or
  - (b) `jax.lax.psum(P_l_partial, axis_name=('x','y'))` after the einsum — reduce of large data (~15 GiB carry/rank) on the carry.

(a) is dramatically cheaper than (b): per-iter ~80 MB gather × n_bc ≈ 800 MB total vs one 15 GiB reduce. Also (a) keeps the carry pure-local; (b) makes the carry effectively replicated and brings back the SPMD-trap-like inflation. **(a) is the right call.**

**This is the missing piece in the original `agent_2_structural_fix.md` §4c sketch** — it didn't address the band-sharding gap because it implicitly assumed bands replicated everywhere on host. They aren't. Adding the explicit per-iter all_gather is necessary.

3. **FFT-box-inside-scan-body — Agent 3's 80 GB number is per-rank with FULL bands**, which the rank doesn't have. Per rank, only `bpd_per_bc = 1` band for CrI3. The per-rank FFT box is `c128[nk·bpd_per_bc, ns, n_rtot] × cuFFT_scratch ≈ 36·1·2·1.125M·16 = 1.3 GB raw × 4 = 5.2 GB`. So Agent 3's option-2 flat-axis cs=8 (1.15 GB) is similar to bc-scan (5.2 GB). Either works; bc-scan is cleaner because the all_gather aligns with bc-iter boundaries.

4. **Both-carries-live vs serialized**: with the 14.85 GiB carry, total memory at "both live" = ~45 GiB (similar to today). With serialized two-scan design (build P_l_acc fully, IFFT/conj, free, build P_r_acc, IFFT, contract), only one 15 GiB carry lives at a time → total ~30 GiB. Cost: 2× host transfers (still small in absolute terms). **Recommend serialized design**; the 15 GiB savings is more valuable than the simplicity loss.

I'll rewrite §2 with this revised design (bc-scan, per-iter all_gather across ('x','y') axis=bands, serialized L-then-R scans). Will land in the unified plan within the next work cycle.

---

**Round 3 response — accepting Agent 3's self-retraction; converging on the final design.**

@Agent 3 — your `n_rmu = 376` correction restores μ_loc = 94 and carry = 3.71 GiB per side. So my v1 carry estimate (2 × 999 MB on a 16-way mesh) was off by a factor of 4× on μ_loc but otherwise the right order of magnitude. **Carry per side = 3.71 GiB, both live = 7.42 GiB.**

**Implications I want to fold in**:

1. **Drop the serialized-L-then-R recommendation.** At 14.85 GiB per carry it would save ~15 GiB. At 3.71 GiB per carry it saves <4 GiB during the scan only, and the γ̃-contract step at the end needs both P_l_R_conj and P_r_R simultaneously anyway (your catch — peak is 7.42 GiB regardless). **Interleaved single scan + both accumulators is now the cleaner choice** — simpler code, no 2× host transfer.

2. **All_gather order**: ✅ agreed with your IFFT-then-all_gather ordering. Gathering full bands BEFORE IFFT would blow the FFT box to 80 GB per rank. The right sequence is: pull 1/P bands → IFFT on per-rank bands → r-slice to r_loc → all_gather across ('x','y') → einsum with full bands.

3. **Option A (outer-bc-scan) vs Option B (flat-(k·b))**: @Agent 4, I want to push back on the Option B preference. Option B mirrors `accumulate_rchunk_to_gflat`'s flat-axis idiom, which works in the REVERSE direction because the FFT is local per μ-row (no cross-rank dependency in the body). In FORWARD direction, the einsum CONTRACTS over bands → per-iter cross-rank gather (or psum) is unavoidable under band-sharded host tiles. Option B doesn't have a natural alignment for that gather: flat (k·b) rows on different ranks belong to different (k, b) combinations and don't all-gather cleanly. Option A (bc-aligned) has the all_gather aligned with bc-iter boundaries — one bc's worth of bands gathered per iter, simple and bounded. **Recommend Option A.**

   If Option B is structurally tried, it requires either:
   - psum on the 7.42 GiB carry at scan end (slower, brings SPMD-trap-class risk back via "replicated carry equivalent").
   - or re-organizing host tiles to NOT be band-flat-sharded (e.g., k-sharded) — out of scope for this round.

4. **Confirmed peak analysis**: under interleaved single-scan + Option A:
   - Per-iter scan transient (aliased): FFT box ~5 GiB + per-iter small buffers ~400 MB ≈ **5.5 GiB**.
   - Both carries (live across scan): 2 × 3.71 GiB = **7.42 GiB**.
   - Post-pair pipeline (P_l_R_conj + P_r_R at γ̃ contract): 2 × 3.71 GiB = **7.42 GiB** (carries freed by this point).
   - Output ~1 GiB, parameters/constants ~0.5 GiB.
   - **Total preallocated-temp peak ≈ 14-15 GiB**, matches your prediction.

→ **@Agent 4**: §2 of the plan currently says "29.7 GiB carry" and §4 says "14.85 GiB per side" — both based on Agent 3's withdrawn n_rmu=1504. Please update §2.3, §2.9, §2.10, and §4.1 with the corrected `μ_loc=94, carry=3.71 GiB per side`. I'll also push corrections to my §2.9/§2.10 directly.

→ **@Agent 4**: drop my "Option A vs B" §4.3 question — Option A is the right answer per the band-shard mismatch reasoning above. Recast §4.3 as "interleaved single scan over bcs with per-iter all_gather" being the chosen design.

5. **`r_loc` gotcha capture confirmed**: §2.3 of the plan has `r_loc = n_zchunk // p_y` and `r0_local = r_start_dyn + axis_index('y') * r_loc` — correct. ✅

6. **Reproducer-first** still mandatory: the all_gather + io_callback inside scan inside shard_map is doubly-novel (per my §1 risk 5). 30-line CPU smoke test before kernel rewrite.

**Status**: Agent 1 round 5 ready, pending the carry-size correction landing in §2/§4 of the plan. I will push that edit myself now to unblock A4.

---

**Round 4 closer (after Agent 2 published).**

Edits landed in `round5_unified_plan.md`:

1. **§2.3** carry corrected to `mu_loc=94, r_loc=18412, 3.71 GiB per side, 7.42 GiB both live`. Iteration log inline.
2. **§2.10** rewritten to recommend **single interleaved scan with both accumulators**, dropping the v2 serialized-L-then-R recommendation. At 3.71 GiB per side, serialization's savings shrink to <4 GiB; γ̃ contract peaks at 7.4 GiB either way (Agent 3 v2 catch). Both-live is simpler.
3. **§6.5 rewritten** from Option B (flat-(k·b)) to Option A (bc-aligned scan + per-iter `all_gather`). Reason: forward direction CONTRACTS over bands → Option B would require a 7.42 GiB `psum` on the carry; Option A's per-iter ~340 MB all_gather is dramatically cheaper. Slicer API is bc-indexed.
4. **§2.8 checklist** updated with corrected carry size, all_gather row, and IFFT-first-then-gather row.

→ **@Agent 2**: ✅ confirming your Q on all_gather-inside-scan-inside-shard_map — manual mode collective using the shard_map's axis_name; scan's WhileOp body is rank-local; gather is a per-iter rank-aware collective. Documented composition (Agent 2 §1.3 in_repo prior art `cholesky_2d.py:167-194`). Should compose cleanly — the smoke test (§1 of your post) is the verification. Also confirming your slicer API spec: bc-indexed `_slice_local_tile_bc(x, y, bc_idx_traced) → (nk, bpd_per_bc, ns, ngkmax)` is correct; matches what I wrote in §6.5 of the plan.

→ **@Agent 2**: your **§5 H2D throughput optimization** ("return only `bpd_per_bc · ngkmax` raw bytes from io_callback, pad device-side via `jnp.pad`") is a real ~16× H2D win at CrI3 scale. Worth flagging in §6.5 of the plan as a Round-7 follow-up; for Round 6 keep the padded return for simplicity.

→ **@Agent 4**: §4 of the plan still has Agent 3 v1 numbers (38-48 GiB total, 14.85 GiB per carry). After Agent 3 v2 retraction the numbers are: 3.71 GiB per side carry, ~13-15 GiB total preallocated-temp. Please update §4.1 to reflect Agent 3 v2.

**Agent 1 round 5 ready** — sign-off provided. Awaiting Agent 4's final pass.

## Agent 2 → others (io_callback / host-device / host-tile lifecycle lens)

**2026-05-13 — Agent 2 first cut (incorporates A1 round-2 + A3 first cut).** SPMD safety, carry sizing, all_gather necessity all settled with A1+A3. I add four pieces from the io_callback / host-tile lens.

### 1. `io_callback` × `lax.scan` × `shard_map` × `lax.all_gather` — feasibility: **YES (smoke-test gated)**

A1 round-2 §2 escalated the design: in addition to `io_callback` inside `scan` inside `shard_map`, we now need `lax.all_gather` inside the same scan body (per-iter band-axis gather to align with replicated `psi_l_X`). Four primitives nested.

Three supporting observations:

- **JAX docs**: each primitive (`io_callback`, `lax.scan`, `shard_map`, `lax.all_gather`) is documented to compose. `all_gather` inside `shard_map` is a *manual collective* (axis_name from the shard_map mesh) — that's the intended composition. It doesn't go through SPMD partitioner.
- **In-tree io_callback-inside-scan prior art**: `src/common/progress.py:scan_progress` (line 149) — io_callback inside `lax.cond` inside `lax.scan` with `ordered=True`. Library helper, currently uncalled, but the API is exercised.
- **In-tree shard_map+io_callback prior art**: `psi_G_store.py:_pull_full` and the per-bc helpers (`5cadd4b`) — production today.
- **In-tree shard_map+all_gather prior art**: `accumulate_rchunk_to_gflat` does scan-inside-shard_map; `solve_zeta` and `_reshard_zeta_r_XY_to_mu_XY` use manual `all_gather` inside `shard_map`. Each pair is verified.

The composition (all four nested) is the missing piece. ~85 % confidence (down from 90 % when only three were nested); the all_gather inside scan is slightly more constrained but I haven't found a documented restriction.

**Smoke test (Round-6 prereq, ~50 LOC, ~5 min CPU + 5 min GPU):**

```python
# tests/test_io_callback_scan_in_shard_map.py
def test_io_callback_scan_shard_map_all_gather_smoke():
    """All four primitives nested.  Verify on 1×1 mesh first (smoke), then
    2×2 GPU mesh (correctness vs global einsum reference)."""
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), axis_names=('x', 'y'))
    n_bc, ngkmax, bpd_max, ns, nk = 3, 7, 4, 2, 5
    rng = np.random.default_rng(0)
    host_tile = (rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax))
                 + 1j*rng.standard_normal((n_bc, nk, bpd_max, ns, ngkmax)))

    def host_slice(x, y, bc):
        return host_tile[int(bc)]                   # (nk, bpd_max, ns, ngkmax)
    out_sds = jax.ShapeDtypeStruct((nk, bpd_max, ns, ngkmax), jnp.complex128)

    @partial(shard_map, mesh=mesh,
             in_specs=(), out_specs=P(None, None), check_rep=False)
    def _local():
        x = jax.lax.axis_index('x'); y = jax.lax.axis_index('y')
        def body(carry, bc_idx):
            slab = io_callback(host_slice, out_sds, x, y, bc_idx, ordered=False)
            slab_full = lax.all_gather(slab, axis_name=('x', 'y'), axis=1, tiled=True)
            return carry + slab_full.sum(), None
        carry, _ = lax.scan(body, jnp.complex128(0), jnp.arange(n_bc))
        return jnp.broadcast_to(carry, (1, 1))

    out = np.asarray(_local())
    np.testing.assert_allclose(out[0, 0], host_tile.sum())
```

If it passes on 1×1 CPU + 4-rank GPU, Path B is unblocked. If it fails on the all_gather-inside-scan composition specifically, fall back to Path A (preserves the leak fix only, but loses the ~30 GiB memory win).

**Why solve_zeta's WhileOp/SPMD trap doesn't apply** (agreeing with A1 §1): inside `shard_map`, SPMD has already done its work at the `in_specs/out_specs` boundary. The scan inside operates on rank-local data with no global sharding annotation. The `all_gather` is a *manual* collective using the mesh axis name; it doesn't go through SPMD. The WhileOp lives in one rank's program; SPMD doesn't see it.

### 2. Host-tile lifecycle + restored `_slice_local_tile_bc`

Restore the slicer (committed `cdd0fba`, deleted `5cadd4b`):

```python
def _slice_local_tile_bc(self, x_idx, y_idx, bc_idx) -> np.ndarray:
    """Per-rank host-tile slice for one bc, padded to (nk, _bpd_max, ns, ngkmax).

    bc_idx is a TRACED int32 scalar (resolved to Python int inside the host fn).
    The full bc spans all ranks via _PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None);
    rank r holds bands [r·bpd_per_bc, (r+1)·bpd_per_bc) of the bc.  Short final
    bc is zero-filled to bpd_max so the scan body sees a static return shape
    every iter.

    Pad rows hold zeros; consumer's per-iter all_gather + L/R band-mask makes
    them mathematically inert.
    """
```

State on `PsiGStore` for Path B:

| Field | Lives | Path-B disposition |
|---|---|---|
| `_host_tiles[(x,y)]` | per-rank numpy, bc-stacked | **keep** — read by io_callback every iter |
| `_bpd_max` | int, set at `__init__` (was in `cdd0fba`, removed in `5cadd4b`) | **restore** |
| `_bc_band_offsets` | tuple, set at `__init__` | unchanged |
| `_g_index_dev` / `_kvecs_frac_dev` | replicated jax.Array, set once via `jax.device_put` | **keep**, captured via closure in `_kernel`. Concrete arrays — no tracer hazard |
| `_psi_G_device_full` | lazy jax.Array cache | **DELETE** — Path B never materializes the full tile on device |
| `psi_G_device_full` property | computes via per-bc concat | **DELETE** |
| `g_index` / `kvecs_frac` properties | thin accessors | **keep** |

### 3. `ordered=True` semantics (answers A1 §8 q1)

My reading of `jax/_src/callback.py`:

- **Per-rank ordering** is what `ordered=True` provides: within a single rank's traced program, ordered callbacks fire in HLO appearance order (which inside `lax.scan(unroll=1)` is iter 0 first, then 1, ...).
- **Cross-rank ordering** is NOT enforced. Each rank has its own host queue. Ordered callbacks on rank 0 don't synchronize against rank 1 — exactly what we want for shard_map.
- **Recommendation: `ordered=False`** for the per-iter tile pull. No cross-iter dependency exists (each iter consumes its own bc_idx → its own slab); `ordered=False` lets XLA dispatch host calls async and pipeline the next iter's H2D against the current iter's compute.
- **Caveat**: `lax.scan(unroll=1)` already gives sequential body execution at runtime regardless of `ordered=`. So `ordered=False` can't reorder anything *meaningful* — it just permits XLA to overlap.

### 4. `bpd_max` closure timing (answers A1 §8 q2)

**YES**, `_bpd_max` is closure-time static. `io_callback` requires `out_sds` static at trace time (JAX uses it to size the device-side buffer); the scan's body output shape must be uniform across iters.

Concrete: at `PsiGStore.__init__`, compute `_bpd_max = max(bpd_per_bc)` once. Close into both:
- `out_sds = jax.ShapeDtypeStruct((nk, _bpd_max, ns, ngkmax), c128)` — io_callback static shape.
- `_slice_local_tile_bc` body — pads its return to `(nk, _bpd_max, ns, ngkmax)` regardless of which bc the traced index resolves to.

Pad rows hold zeros. After per-iter all_gather (A1 round-2 §2), the L/R band-mask zeros out pad rows; they contribute nothing.

### 5. Latency vs throughput (answers A3's payload question + sizes A1 round-2's all_gather)

**io_callback per-iter payload** at CrI3 6×6 80 Ry on 4×4 mesh:
- Per-call: `nk · bpd_max · ns · ngkmax · 16` ≈ 36 · 16 · 2 · 70k · 16 ≈ 1.3 GB / rank / call. (The pad inflates the effective byte count vs the actual data, since per rank we own only 1/P of the bc's bands ≈ 80 MB raw — but the io_callback transfers the full padded slab.)
- 320 calls × 1.3 GB ≈ 416 GB H2D per channel per rank. At ~25 GB/s: ~16 s / channel.
- × 4 channels = 64 s total H2D. Python overhead per call: 1–5 ms × 1280 ≈ 2–6 s.

(A3's 14 s number used 1.1 GB; my 16 s uses 1.3 GB — agrees modulo bpd_max rounding.)

**Optimization opportunity (Round-6 implementation note)**: the io_callback could return only the rank's actual `bpd_per_bc · ngkmax · 16` ≈ 80 MB instead of the padded `bpd_max · ngkmax · 16` ≈ 1.3 GB. That cuts H2D 16×. The pad would happen device-side via `jnp.pad` after the io_callback. Easy to skip on first pass since the CrI3 ratio is already small; flagged as a follow-up.

**all_gather per-iter cost** (A1 round-2 §2 design):
- Pre-gather: `(nk, bpd_per_bc, ns, r_loc)` ≈ 36 · 1 · 2 · 18412 · 16 ≈ 21 MB / rank / iter.
- Post-gather: `(nk, bpd_max, ns, r_loc)` ≈ 340 MB / rank.
- Each rank receives `(P-1) · 21 MB ≈ 315 MB` over the mesh. At NVLink ~100 GB/s: ~3 ms / iter / rank.
- 320 iters × 3 ms ≈ 1 s / channel. Trivial.

**Vs current Path A**: identical io_callback call count (320 per channel) and byte total. Path B's only difference is XLA gets to schedule dispatch (vs Python's sequential loop in `psi_G_device_full`).

**Verdict**: Path-B's I/O cost ≤ Path-A's, plus cheap all_gather. ~15–30 GiB memory savings (depending on serialized-L-then-R per A1 round-2 §4) is pure win. Open: profile whether XLA actually pipelines io_callback against compute — Round-6 measurement.

### 6. Withdrawn: my draft "FFT-box-inside-body" pushback

In an earlier draft I argued the per-iter FFT box would be too big (cited A3's 80 GB number) and recommended an inner k-scan. **A1 round-2 §3 corrects this**: per-rank FFT box is only `c128[nk · bpd_per_bc, ns, n_rtot] · cuFFT_scratch` ≈ 36 · 1 · 2 · 1.125M · 16 · 4 ≈ 5.2 GB. Per rank. Because host tiles are band-flat-sharded across the full mesh, each rank only owns 1/P of the bc's bands. **No nested k-scan needed.** Withdraw.

### 7. Path A vs Path B — decision matrix

| Dimension | A (pre-pull driver-jit-arg) | B (io_callback in scan in shard_map + per-iter all_gather) |
|---|---|---|
| Tracer leak | ✓ | ✓ |
| Concat reshuffle | ✗ | ✓ |
| Consumer-boundary remat (~30 GiB) | ✗ | ✓ |
| Implementation cost | ~50 LOC | ~250 LOC |
| Risk | low | medium (4-primitive composition smoke test §1) |
| Memory savings (CrI3 80 Ry) | ~0 | ~15–30 GiB |

**Decision: Path B**, gated on the §1 smoke test passing.

### 8. `begin_rchunk` / `end_rchunk` audit

| Field | Set | Cleared | Path-B status |
|---|---|---|---|
| `_host_tiles` | `_populate_from_loader` | `_clear_tiles` (RereadPsiGStore.end_rchunk only) | **must be valid for kernel duration** — io_callback reads per scan iter. Reread mode's `end_rchunk` runs **after** `block_until_ready` (`isdf_fitting.py:2168-2172` `finally:`), so async io_callbacks finish before tiles are freed. ✓ |
| `_g_index_dev` / `_kvecs_frac_dev` | once | NEVER | concrete `jax.device_put` arrays. ✓ |
| `_psi_G_device_full` | property | `_clear_tiles` | **DELETE** in Path B. |

The async-callback / host-tile lifetime contract already exists in production (Reread mode); Path B inherits it.

---

### Cross-references

→ **Agent 1**: agree with your round-2 corrections (carry size, all_gather requirement, serialized L-then-R). Withdrew my §6 pushback. My §3/§4 above answer your §8 q1 (`ordered=False` recommended; ordered is per-rank only) and q2 (`bpd_max` closure-time static; io_callback REQUIRES static `out_sds`).

  Question back: is `lax.all_gather(..., axis_name=('x','y'), tiled=True)` a pure manual collective inside the shard_map body, or does it need special wrapping because we're also inside `lax.scan`? My read: just a manual collective — scan's body lowers to a WhileOp body, and any rank-local op (including `all_gather` against the shard_map's axis name) lowers correctly. Confirm if your SPMD lens agrees.

→ **Agent 3**: answer to your payload question is in §5 (we agree on ~14–16 s host-side wall time). Your "FFT-box-inside-body" framing is now resolved by A1 round-2 §3 — per-rank box is 5.2 GB, not 80 GB. Updated HLO prediction request: with serialized L-then-R scans (A1 round-2 §4), the HWM should be ~30 GiB (one 14.85 GiB carry live at a time + 5.2 GiB FFT box + ~3.7 GiB post-pair scratch + ~1 GiB other). Does your buffer-assignment lens predict the same when XLA sees two serialized scans in the body?

→ **Agent 4** (synthesis): when you skeleton `round5_unified_plan.md`, my deliverables for the plan are §2 (host-tile lifecycle), §6 (io_callback `ordered=False` / `unroll=1` constraints / `bpd_max` closure timing), and the smoke test as a Round-6 prerequisite. The structural design (bc-by-bc outer scan, per-iter all_gather, serialized L-then-R) is now A1's piece — I defer to A1's round-2 §2 rewrite for the body sketch.

**Agent 2 round 5 ready** (will iterate as A1 / A3 / A4 respond).

---

**Round 2 — answering Agent 4's four asks + folding A1 v3 (interleaved scan + 7.42 GiB carry) consensus.**

§3 of the unified plan is now populated with full content. Direct answers to A4's four asks:

1. **Slicer API** — confirmed with one shape clarification:
   - Signature: `_slice_local_tile_bc(self, x_idx, y_idx, bc_idx) → np.ndarray`.
   - Return shape: **`(nk, _bpd_max, ns, ngkmax)`** with `np.zeros` for short-bc pad rows. Static = `bpd_max`, NOT `bpd_per_bc[bc]`.
   - Reason: `lax.scan` requires uniform body output shape across iters. If the final bc has fewer bands than the others (`bpd_per_bc[-1] < bpd_max`), the scan body's `out_sds` must be the max so every iter sees the same shape. Pad with zeros; consumer's L/R band-mask zeros them out (math-neutral).
   - A4 §6.5 / §2.10 use "bpd_per_bc" — please change to `_bpd_max` if you literally meant per-bc dynamic shape (the scan body forces static).

2. **`ordered=True` semantics** — within-rank only, NOT cross-rank.
   - Each rank has its own host queue. `ordered` callbacks on rank r don't synchronize against rank r' on the same axis.
   - For our use: **`ordered=False`** is correct (per §3.4 of the plan). `lax.scan(unroll=1)` gives sequential per-rank execution at runtime regardless; `ordered=False` permits XLA to overlap async H2D dispatch with device compute.

3. **Host-tile lifetime during scan** — confirmed (plan §3.10). The `RereadPsiGStore.end_rchunk` `finally:` clause at `isdf_fitting.py:2168-2172` runs **after** `block_until_ready`, so all in-flight io_callbacks complete before tiles are freed. This contract already protects today's per-bc fetch (which fires N io_callbacks per kernel call); Path-B inherits unchanged. No new lifecycle invariant.

4. **Path A fallback** — your §7.2 "driver-level Python bc-loop with donated jit per bc" is **a stronger fallback** than my §3.11 ("driver-level pre-pull + jit arg, single dispatch"). Donation chain gives sequential memory reuse à la `solve_zeta`'s pattern that originally informed Path D's design. **Adopt your §7.2 in the plan**; my §3.11 should be re-described as "Path A-2: pre-pull + single dispatch (simpler but worse memory profile than §7.2)". I'll update §3.11 to flag §7.2 as the preferred fallback after you confirm.

**n_rmu confirmation** — A3 line 392 cited the HLO param `c128[36, 376, 376]` for L_q, confirming `meta.n_rmu_padded = 376` (NOT 1504 as gw.out's "ISDF basis" line suggests — that's centroids before the Cholesky-for-mesh-divisibility round-up; the kernel sees the post-round-up value). 7.42 GiB carry stands. A4 v3 + A1 v3 (interleaved scan, both accumulators per iter) is correct.

**Sign-off**:
- ✓ Slicer API (with `_bpd_max` shape clarification — A4 §6.5 should say `_bpd_max`, not `bpd_per_bc`).
- ✓ `ordered=False` + within-rank ordering only.
- ✓ Host-tile lifetime preserved by existing `RereadPsiGStore.end_rchunk` `finally:` contract.
- ✓ Path A fallback: A4 §7.2 (driver-level Python bc-loop with donated jit per bc) is the preferred fallback; my §3.11 demoted to "Path A-2 (simpler but worse memory profile)".
- ✓ Carry size 7.42 GiB confirmed via A3's HLO param read.

**Agent 2 round 5 ready (full sign-off).**

## Agent 3 → others (XLA / HLO / BufferAssignment prediction lens)

**Agent 3 first cut — HLO prediction + critical disagreement with Agent 1 on carry size.** Full writeup in `round5_agent_3_hlo.md` and contribution to `round5_unified_plan.md`.

Key headline first: **the carry size depends entirely on a sharding choice that hasn't been made yet.** Agent 1's "2 GB total carry" only holds if μ is sharded on the combined `('x','y')` axis (μ_loc = 1504/16 = 94). Under today's `P(None, 'x', None, None)` for `psi_l_X` (μ on `'x'` only, μ_loc = 1504/4 = 376), the carry is **14.85 GiB per accumulator × 2 = 29.70 GiB.** That's still a 15 GiB savings vs today's 44.56 GiB temp pool — but it's not the dramatic 2 GiB Agent 1 predicts.

### HLO predictions (new design under TODAY's sharding)

| Item | Predicted per-rank | Note |
|---|---|---|
| Carry `P_l_acc` (lives all scan iters) | 14.85 GiB | `c128[36, 2, 18412, 376, 2]`, r sharded on `'y'` p_y=4, μ sharded on `'x'` p_x=4 |
| Carry `P_r_acc` | 14.85 GiB | Same shape |
| FFT box inside scan body (aliased across iters) | **5–15 GiB** | `c128[nk·bpd_max, ns, n_rtot]` per rank, depends on whether nk is replicated and whether we add an inner k-chunk scan |
| psi_G_bc per-iter (aliased) | ~1.1 GiB | `c128[36, 16, 2, 59990]` from io_callback |
| Post-pair scratch (IFFT/γ̃/FFT chain — unchanged from today) | ~3.7 GiB | Reused for Z output |
| Parameters + constants + output | ~0.4 GiB | L_q, R_L, R_R |
| **Predicted total preallocated** | **~38–48 GiB** | vs today's 48.63 GiB |

### Critical: the FFT-box-inside-scan-body is NOT trivially small

Agent 1 §8 predicts "ONE FFT-box-class slot per scan (aliased across bcs)" — that's structurally correct (XLA scan-internal allocator will alias). But what's the size of that one slot?

At CrI3 80 Ry with `band_chunk=16`, `nk=36` (replicated under `P(None, 'x', None, None)`), `ns=2`, `n_rtot=1.125M`:
- Per-iter FFT box (full, unsharded across mesh): `c128[36 · 16, 2, 1.125M]` × cuFFT scratch factor.
- Bare bytes: 36 · 16 · 2 · 1.125M · 16 = **20.7 GB per rank** before cuFFT scratch. With factor 4: **~80 GB**. **NOT FEASIBLE.**

Two ways out:
1. **Inner k-chunk scan** (agent_2_structural_fix.md §4e bullet): nest a second `lax.scan` over k_chunk=6 (or similar) inside the bc-scan. Per-iter box drops to `c128[6·16, 2, 1.125M] × 4 = 13 GB`. Workable, complexity penalty.
2. **Flat-axis scan with `chunk_size`** (mirrors `accumulate_rchunk_to_gflat` exactly): scan over the flat (k·b) axis with `cs = 8` or 16. Per-iter box `c128[cs, ns, n_rtot] × 4 = 1.15 GB` at cs=8. Simpler structure, same idiom as the reference.

**This is the load-bearing design decision that the plan must make explicit.** Agent 2's §4c sketch uses outer-bc-scan; that alone won't fit without nesting. I lean toward the flat-axis pattern (option 2) because it's the same idiom that's already verified in the reference helper.

### `psi_Y_full` materialization — confirmed GONE

In the new design `psi_Y_full` never exists as a whole array — only per-bc/per-iter `psi_Y_bc` lives inside the scan body, contracted directly into the accumulator. Today's HLO shows 30 GiB of double-materialization (Agent 1 R4 §4 finding); this is fully eliminated. ✅

### Remat boundary — confirmed GONE

The cause of the involuntary remat is the slice + reshard at the helper/consumer boundary (band-flat `→` r-sharded). In the new design, the boundary doesn't exist: the post-pair pipeline (`IFFT → γ̃ → FFT`) lives in the same shard_map body and consumes the carry directly. No reshard, no copy operation, no remat. ✅

### cuFFT batching answer

Per-iter FFT batch is `cs · ns` (flat-axis option) or `k_chunk · band_chunk · ns` (nested option). At cs=8 + ns=2 = 16 FFTs of size 75·75·200 — well above the cuFFT efficiency floor (~32 is amortized, 16 is acceptable). At nested-option (6·16·2 = 192 FFTs) — comfortably batched. Either is fine; flat-axis option has slightly smaller batch per call but more calls per scan.

### Compile-time prediction

The new kernel adds: a `lax.scan` (or two nested scans) inside the shard_map body, with `io_callback` per iter. Each `lax.scan` lowers to a single `WhileOp` (not unrolled at this n_bc≈20). Compile time should grow **<2× vs today** — same number of XLA modules, similar structure to `accumulate_rchunk_to_gflat`. Concern only if XLA decides to unroll the scan (e.g., if n_bc is small *and* `unroll` is set). Don't pass `unroll=` to scan.

### Open questions for peers

→ **Agent 1**: your §2 "999 MB per carry" assumes `μ_loc = 94`, requiring μ sharded on full `('x','y')`. Today's `psi_l_X` is `P(None, 'x', None, None)` — μ on `'x'` only. **Are you proposing a sharding change?** If yes, the plan needs to add that as an explicit decision with a re-sharding cost analysis (psi_l_X today is X-sharded; switching to `('x','y')`-sharded means the upstream centroid loader must produce it that way). If no, the carry is 14.85 GiB per side, 29.70 GiB total — still a win, just not 16×.

→ **Agent 1**: same question for `r_loc = 4603`. Today's r_chunk per call = 73648 (n_chunks=16). Sharding on `'y'` p_y=4 → r_loc = 18412. r_loc = 4603 only works if r is sharded on the full `('x','y')` mesh (p_xy=16) — but that conflicts with `out_spec = P(None, 'x', 'y')`. Did you mean to assume `out_spec = P(None, None, ('x','y'))` or a much smaller per-call r_chunk?

→ **Agent 2**: how big is the per-iter `psi_G_bc` payload the host actually transfers? At `c128[36, 16, 2, 59990]` ≈ 1.1 GB per call, 20 iters per r-chunk × 16 r-chunks = 350 GB of host→device bandwidth per fit_zeta. At PCIe 4.0 ~25 GB/s that's 14 s overhead per fit. Worth knowing if this is a wall-clock concern.

→ **Agent 4**: the unified plan needs to surface the "FFT-box-inside-body" sizing decision (flat-axis cs vs nested k-scan) as a primary design choice. It determines whether we hit 38 GiB or 48 GiB or worse.

Will iterate after seeing your responses.

---

**Round 2 — Agent 3 retraction + refined analysis after Agent 1's response.**

I owe the team a correction myself. I was using `n_rmu = 1504` (from gw.out's "ISDF basis: 1504 centroids") but the **actual `n_rmu` inside this kernel is 376** — verified from the HLO param `c128[36, 376, 376]` for `L_q` and `c128[36, 376, 160, 2]` for `psi_r_X`. So `μ_loc = 376/p_x = 94` under today's `P(None, 'x', None, None)` — Agent 1's number was right, mine was wrong. Apologies for the noise.

**Corrected per-rank carry under TODAY's sharding (no upstream changes):**
- `P_l_acc`: c128[36, 2, 18412, 94, 2] = 36·2·18412·94·2·16 = **3.98 GB ≈ 3.71 GiB**
- Two carries simultaneously live = **7.42 GiB**

But Agent 1 then raised the killer issue: host tiles are `_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)` — bands sharded on FULL mesh, 1 band per rank per bc at CrI3. The einsum requires all 16 bands per rank, so per-iter we need either all_gather or psum.

### Agreement on Agent 1's design refinements

**(1) Per-iter all_gather over `('x','y')` on bands** ✅ I concur. Size analysis:
- Pre-IFFT per-iter on rank: `c128[nk=36, bpd_per_bc=1, ns=2, ngkmax=59990]` = 80 MB (one band per rank).
- **Order matters**: do the IFFT BEFORE the all_gather, not after. Per-rank IFFT on its 1 band → `c128[36, 1, 2, n_rtot=1.125M]` ≈ 1.3 GB raw, cuFFT scratch ~5 GiB. Then slice to `r_loc` → `c128[36, 1, 2, 18412]` ≈ 21 MB. THEN all_gather across `('x','y')` tiled on band axis → `c128[36, 16, 2, 18412]` ≈ 340 MB per rank.
- Alternative order (all_gather first): all_gather `psi_G_bc` over bands → `c128[36, 16, 2, 59990]` ≈ 1.3 GB per rank — then IFFT on full bands → 80 GB. **Reject this order.**
- **Recommend**: IFFT-then-slice-then-all_gather. Per-iter peak transient is just the FFT box (~5 GiB cuFFT), all_gather payload tiny.

⚠️ **One subtlety**: an all_gather across the FULL mesh inside scan body breaks "no replicated intermediates" in the strictest sense — the gathered bands are replicated 1× (one copy per rank). But the gathered tensor (340 MB) is much smaller than the FFT box already present, so it's effectively absorbed into existing scan-internal slots. I'd flag this in the unified plan as "intentional replication of a small ~340 MB per-iter buffer, justified by the alternative being a 14.85 GiB psum on the carry."

**(2) Serialized L-then-R scans** ✅ Concur in principle but want to verify peak. The proposed flow:
1. Scan over bc → produce `P_l` (carry alive 3.71 GiB, but lives during scan)
2. IFFT(k) on P_l → `P_l_R`, conj → `P_l_R_conj` (3.71 GiB)
3. Free P_l intermediates
4. Scan over bc → produce `P_r` (carry 3.71 GiB)
5. IFFT(k) on P_r → `P_r_R` (3.71 GiB)
6. γ̃ double contract: needs both P_l_R_conj AND P_r_R simultaneously → peak = 7.42 GiB
7. Reduce to Z_R, FFT(k) → Z_q

**Peak memory in serialized design: γ̃ contract = both carries-class buffers live simultaneously = 7.42 GiB**. Same as the "both live" interleaved design. Serialization helps the scan body's working set (1 carry vs 2 during scan), but the γ̃ contract is unavoidably 2-buffer.

**However**: with serialized scans, the FFT box per iter is the same in both scans — XLA can alias them across the two scans because their lifetimes are disjoint. So serialization saves ~5 GiB of FFT box pile-up if XLA can't otherwise alias them.

Net: serialized vs interleaved, similar peak memory but serialized has slightly cleaner aliasing story.

### Updated HLO prediction (Agent 1's serialized + all_gather design)

| Item | Per-rank | Status |
|---|---|---|
| One live carry during scan (`P_l_acc` OR `P_r_acc`) | 3.71 GiB | Carry |
| Per-iter FFT box (aliased) | ~5 GiB (cuFFT scratch incl.) | Scan transient, ALIASED |
| Per-iter all_gather result | ~340 MB | Aliased |
| Per-iter psi_G_bc fetch | 80 MB | Aliased |
| Peak at γ̃ contract: P_l_R_conj + P_r_R | 7.42 GiB | Two buffers concurrent |
| Output Z_q | 0.93 GiB | live-out (corrected: c128[36, 94, 18412] = 0.99 GB) |
| Parameters + constants | ~0.3 GiB | |
| **Predicted total preallocated-temp** | **~13–15 GiB** | vs today's 48.63 GiB |

**A 3× memory reduction, end-to-end.** Strong improvement, well below the 28 GiB per-A100 nominal budget (and the 60 GiB cohsex.in setting).

### Still load-bearing decisions for the plan

1. **All_gather inside scan body**: cleanest implementation is `jax.lax.all_gather(..., axis_name=('x','y'), axis=1, tiled=True)`. Confirm XLA handles this correctly in a shard_map manual-mode body. Reproducer needed.
2. **Serialized vs interleaved scans**: I lean serialized for clearer aliasing.
3. **FFT-box sizing**: with bpd_per_bc=1 per rank, FFT box per iter = ~1.3 GB raw, ~5 GiB with cuFFT scratch. No need for inner k-scan or flat-axis trick — the band sharding inherently gives us per-rank-small batch.
4. **psi_l_X bands replicated but bc-windowed**: `psi_l_X` per-rank has FULL nb_l bands of L (since bands are replicated on `P(None, 'x', None, None)`). We slice it per-bc with `dynamic_slice_in_dim`. The bc's band window has bpd_max bands at FULL extent (not bpd_per_bc=1) — so psi_l_X_bc has bpd_max=16 bands. The einsum contracts these against the all_gathered psi_Y_bc with bpd_max=16 bands. Symmetric. ✅

### Open questions remaining

→ **Agent 1**: please update your §3 of `round5_agent_1_spmd.md` with the corrected per-rank carry math (3.71 GiB each, not 999 MB). Also surface the all_gather in §2 since it's now load-bearing.

→ **Agent 2**: with per-iter all_gather payload ~340 MB and IFFT FFT-box ~5 GiB per rank-iter, the host→device transfer per io_callback drops to 80 MB. Total transfer per fit_zeta: ~25 GB × n_r_chunks = 400 GB. Still substantial but tractable at PCIe 4.0 (~16 s overhead per fit).

→ **Agent 4**: the unified plan's §2.3 mu_loc/r_loc math (376/4=94, n_zchunk/4=4603) was based on r_chunk=18412 per-call. With today's r_chunk=73648 per call, r_loc=18412 instead. The plan should clarify whether the new design changes r_chunk in the planner (to 18412 per call, n_chunks=64) OR keeps 73648 per call (n_chunks=16, but r_loc=18412). Either works for the design; the planner choice affects wall-clock not memory.

**Agent 3 round 5 ready** — with the caveat that I want to see Agent 2's io_callback details before final sign-off. Will fold corrections into `round5_agent_3_hlo.md` and the unified plan §4.

---

**Round 3 — Agent 3 final response after Agent 1 v3 + plan synthesis.**

Concur with Agent 1 v3 §2.10: **interleaved single scan**, not serialized. My v2 serialized recommendation was driven by the (incorrect) 14.85 GiB per-carry estimate. With the corrected 3.71 GiB per-carry, the γ̃ contract peak (`P_l_R_conj` + `P_r_R` simultaneously = 7.42 GiB) dominates regardless of serialization — serialization savings shrink to <4 GiB during the scan body, not worth the structural complexity. Dropping the serialized recommendation.

**Updated `round5_unified_plan.md` §4 with final converged numbers**:
- §4.1: per-rank slots, ~13–15 GiB total (interleaved scan).
- §4.1a: host-band-sharding + per-iter all_gather rationale.
- §4.1b: phase-by-phase memory walk.
- §4.3: FFT-box-sizing RESOLVED — no chunking needed thanks to host sharding.
- §4.5–4.6: cuFFT batching + compile-time predictions.
- §4.7: planner-side `_peak_C_fit_one_rchunk` term replacement.
- §4.8: seven HLO acceptance gates for G2.
- §4.9: reproducer-first risk for the novel composition.

**Three-round cross-correction trail** (for the record):
1. R1: Agent 1 said 999 MB carry (wrong: assumed full-mesh μ sharding). I said 14.85 GiB (wrong: assumed `n_rmu = 1504`).
2. R2: HLO verification → `n_rmu = 376` in this kernel. Agent 1 caught the host-band-sharding (`bpd_per_bc = 1`) which compounds the FFT-box question and motivates the per-iter all_gather.
3. R3: With `bpd_per_bc = 1`, the per-rank carry is actually 3.71 GiB under today's sharding; FFT box naturally ~5 GiB (no chunking needed); interleaved single scan cleaner than serialized.

**Converged numbers**: ~13–15 GiB per rank total, **~3× memory reduction** vs today's 48.63 GiB. ~46 GiB of replicated/rematerialized intermediates eliminated; zero remat warnings expected at G2.

Agent 3 round 5 ready.

## Agent 4 → others (numerics / correctness / validation lead + synthesis)

**2026-05-13 — Agent 4 live for Round 5 (synthesis owner).**

Workflow: I'll read `agent_2_structural_fix.md` §4c-d (the deferred
scan-inside-shard_map design), `psi_G_store.py:298-314` (current
buggy concat reshuffle + tracer-leaking lazy cache), and the
`solve_zeta` lines 1119-1141 prior-art comment.  Then I'll publish
my numerics/correctness lens here as a first cut.  I'll skeleton
`round5_unified_plan.md` with §1, §5–§8 (the parts I own + the
shell), leaving §2, §3, §4 for you three to fill in via this
discussion.

Bit-identity stance, ahead of analysis: scan-with-sum is **not**
bit-equal to a global einsum — at most rtol = 1e-12 (one rounding
ULP per partial sum).  I'll cross-check against
`accumulate_rchunk_to_gflat`'s test scaffold to confirm that
tolerance is what the existing acceptance gate uses; if so we
mirror.

Iteration commitment: I'll respond to each of your messages within
my work cycle and won't declare the plan final until all three
"ready" markers land here.  If we disagree on a structural choice,
flag it explicitly in your section and I'll surface it as an open
question in the plan rather than picking unilaterally.

## Any → Orchestrator (human)
_(Prefix with `BLOCKER:` if you need an immediate answer.)_

---

**2026-05-13 — Agent 4 response (post A1 + A3 first cuts).**

Folded both your sections into `round5_unified_plan.md`:

- **§2 populated with A1's SPMD analysis** (carry design, in_specs /
  out_specs, mask approach, post-pair tail unchanged, SPMD safety
  checklist).  §2.3 surfaces the per-rank `r_loc = n_zchunk / p_y`
  correctness gotcha prominently; §1 has it as a load-bearing
  callout too.
- **§4 populated with A3's HLO predictions** including the
  carry-size reconciliation between you two (§4.2: under TODAY's
  `P(None, 'x', None, None)` for psi_l_X, carry = 14.85 GiB × 2 =
  29.7 GiB on CrI3, NOT 2 GiB).  μ-sharding change to `('x','y')`
  is out-of-scope per §8 (would shrink 4× to 7.4 GiB).
- **§4.3 surfaces the FFT-box-inside-scan sizing decision** —
  Option A (outer bc + nested k-scan) vs Option B (flat (k·b) axis,
  mirrors `accumulate_rchunk_to_gflat`).  Agent 3 + I lean **Option
  B** for in-repo precedent and single-scan simplicity.  Agent 1:
  please weigh in.
- **§5.3.6 added** for the divisibility precondition (`n_zchunk %
  p_y == 0`).  Agent 3: please confirm planner enforces this;
  today's enforcement on `n_rmu_padded ≡ ∏ p_a` is centroid-loader-
  side, and I want to know whether the r-chunk axis has equivalent
  guarantees in `gw_init.fit_zeta`'s planner pick.
- **§6.5 has the flat-(k·b) decode pseudocode**.  Agent 2: this is
  where your `_slice_local_tile_bc` slicer API must agree with the
  body's decode of `row_idx → (k_row, bc_row)`.  Please spec the
  slicer to take a flat row-batch (e.g. `(x_idx, y_idx,
  row_idx_traced, cs)`), not the §4a-style bc_idx-only API, if you
  concur with Option B.
- **§8 lists the μ-sharding change as future scope.**
- **§7 has io_callback-in-scan-in-shard_map novelty as Risk 7.1**;
  will spec the reproducer-first concretely once Agent 2 publishes
  §3.

**@Agent 1**: confirm that the `r_loc` gotcha is captured correctly
in §2.3 of the unified plan (`r_loc = n_zchunk // p_y`, `r0_local =
r_start_dyn + axis_index('y') * r_loc`).  Confirm your sketch's
outer-bc-only iteration was a simplification, and that you can swap
to flat-(k·b) iteration (Option B) without disturbing the SPMD
analysis (the mask approach generalizes to per-row masks gathered
by `row_idx → bc_row → mask_table` — same SPMD-safety properties).

**@Agent 3**: confirm I've captured your HLO predictions correctly
in §4 (especially the 38–48 GiB total and the cs=8 → 1.15 GB
per-iter FFT box).  Numerics gate is rtol=1e-10/atol=1e-12 (matches
`accumulate_rchunk_to_gflat` scaffold) — please confirm that's
also your acceptance threshold for G3 (Σ matches lorrax_A baseline
at production scale).

**@Agent 2**: pending your section.  Provisional asks:
- Slicer API should take `(x_idx, y_idx, row_idx_traced, cs)`
  returning `(cs, ns, ngkmax)` padded — see §6.5.  Confirm or
  push back on the API shape.
- `ordered=True` semantics: A1 §8.1 asks "within-rank only or
  cross-rank?"  We need within-rank only.
- `psi_G_device_full` lazy-cache deletion confirmed in §3?
  (Tracer-leak source per the round5 mission.)
- Host→device bandwidth (A3's ~350 GB transfer over full
  fit_zeta, ~14 s at PCIe 4.0): include in §3 lifecycle / bandwidth
  or surface as a separate profiling task.

**Status marker**: **Agent 4 partial-ready — §1, §2 (per A1), §4
(per A3), §5, §6 (§6.5 dependent on A2), §7, §8 all populated.
Waiting on A2 §3 + final sign-offs from all three.**  Won't print
the sentinel until all three "ready" markers land here.

---

**2026-05-13 — Agent 4 response (post A1↔A3 convergence to 13–15 GiB).**

Plan §1 + §4 updates landed.  Confirming the converged design:

- **Carry**: 3.71 GiB per side × 2 = 7.42 GiB.  μ_loc=94 (n_rmu=376
  in this kernel, not 1504); r_loc=18412.  §2.3 and §4.2 both
  reflect this.
- **Iteration**: single interleaved scan over bc, both `P_l_acc` /
  `P_r_acc` updated per iter.  §2.10 (A1 v3) is the binding sketch;
  §4.1b updated to match (was "serialized"; corrected to interleaved
  — peak is at γ̃ contract anyway, serialization buys nothing once
  the carry is small).
- **All_gather**: per-iter across `('x','y')` axis=bands,
  `tiled=True`, **AFTER** IFFT-and-slice.  Order matters
  (gather-first → 80 GB FFT box).  §2.9 has the load-bearing
  description; §4.1a in §4 too; §1 now lists this as one of three
  load-bearing structural requirements alongside the `r_loc` gotcha
  and the `out_spec=P(None,'x','y')` boundary.
- **Option A vs B**: §4.3 already collapsed to "no chunking needed
  — host-side band sharding does the work".  §6.5 has the
  converged outer-bc-scan body sketch (Agent 1 v3 push).
- **Slicer API**: `(x_idx, y_idx, bc_idx_traced)` returning `(nk,
  bpd_per_bc, ns, ngkmax)` — bc-indexed, NOT flat-(k·b) row-batch
  as I provisionally asked.  Reverting my §6.5 ask; Agent 2's §3
  spec should match the §6.5 / §2.10 body's call.
- **Predicted peak**: **~13-15 GiB per rank** at CrI3 6×6 80 Ry —
  ~3× memory reduction vs today's 48.63 GiB.  ~4 substantive
  preallocated-temp slots.
- **μ-sharding change** (§8): downgraded — under today's 7.42 GiB
  carry, the saving is 5.56 GiB not 22 GiB.  Not urgent; file as
  R7+ if profiling shows the carry is the bottleneck.

**Remaining numerics-side note (please review §5.2)**: tolerance
`rtol=1e-10, atol=1e-12` matches `accumulate_rchunk_to_gflat`
scaffold and is well above the predicted ULP-class drift (~6e-15
relative).  No change vs my earlier draft; both A1 and A3 confirmed.

**Status markers**:
- ✅ **Agent 1 round 5 ready** (line 111).
- ✅ **Agent 3 round 5 ready** (line 243), modulo Agent 2 sign-off.
- ⏳ **Agent 2 round 5 — not yet published** in their `Agent 2 →
  others` section (line 113 still placeholder).

**I will declare "Agent 4 round 5 unified plan done" only after
Agent 2 publishes their §3 contribution and signs off.**  Three
specific things I need from Agent 2 to lock in the plan:

1. **Slicer API confirmation**: `_slice_local_tile_bc(x_idx, y_idx,
   bc_idx_traced) → (nk, bpd_per_bc, ns, ngkmax)` static return
   shape.  Specifically: the slicer must use `np.zeros` (NOT
   `np.empty`) for pad rows — masked pad-bands rely on zeros to
   stay math-neutral (§5.3.3).
2. **`ordered=True` semantics**: within-rank only, or cross-rank?
   A1 §8.1 needs the former.  If the latter, the plan needs a
   different sequencing strategy.
3. **Host-tile lifetime during scan**: the scan body fires
   `io_callback` once per bc iter; tiles must survive across all
   iters.  Today's `RereadPsiGStore.end_rchunk` fires after the
   jit completes — confirm this lifecycle holds for the new
   scan-internal callback.
4. **Path A fallback** (Plan §7.1): if `io_callback`-in-`lax.scan`-
   in-`shard_map` proves non-functional or pathologically slow at
   the 30-line reproducer stage, what's the next-best design that
   still avoids `psi_Y_full`?  My provisional answer in §7.2 was
   "driver-level Python bc-loop with donated jit per bc" — would
   appreciate Agent 2's read on host-tile lifecycle implications.

Once you publish, I'll fold §3 into the plan, re-verify nothing
broke at §6.5 / §6.1 / §7, and post the sentinel.

---

**2026-05-13 — Agent 4 final sign-off + plan locked.**

All three peer "ready" markers confirmed:

- ✅ **Agent 1**: lines 111 + 130.  SPMD analysis, all_gather
  requirement, interleaved single scan, three load-bearing
  structural points in §1 of the plan.
- ✅ **Agent 2**: line 295.  io_callback / host-tile lifecycle,
  §3.1–§3.12 of the plan, §3.6 smoke test as Round-6 prereq,
  `np.zeros` allocator mandate for the slicer, `ordered=False`
  recommendation, host-tile lifetime audit confirming the
  Reread-mode `finally:` contract.
- ✅ **Agent 3**: lines 424 + 449.  HLO predictions converged at
  ~13–15 GiB/rank, seven G2 acceptance gates in §4.8, planner
  bookkeeping coordinated with Round 4 T3.

**Plan status**: FINAL.  Banner flipped at the top of
`round5_unified_plan.md`.  Round-6 implementer treats it as
authoritative; cross-references the three standalone analyses for
depth.

**Folded into the plan since my last update**:

- **§6.1 rewritten** to match Agent 2's §3.9 state table — `_bpd_max`
  restored (not "drop if added"), `_psi_G_device_full` deleted,
  `_slice_local_tile_bc` returns `np.zeros`-padded (not
  `np.empty`).  Aligns with §5.3.3's pad-zeros math-neutrality
  argument.
- **§7.1 rewritten** to surface the four-primitive composition risk
  and Agent 2's §3.6 smoke test as the Round-6 prerequisite.
- **§7.3 tightened** (`np.zeros` mandate now §6.1-binding, audit
  during PR review).
- **§7.4 resolved** by Agent 2 §3.10 (host-tile lifetime contract
  inherited from Reread mode's `finally:` block at
  `isdf_fitting.py:2168-2172`).
- **§1 headline + status banner** updated with the converged 13–15
  GiB / 3× memory reduction prediction and the smoke-test gate.

**Cross-references**: Round-6 implementer should read in this order:
1. §1 + §3 load-bearing requirements callout (3 items).
2. §3.6 smoke test (RUN FIRST; nothing proceeds without it).
3. §2.10 binding body sketch + §6.5 expanded sketch with closure
   context.
4. §5.4 G1/G2/G3 gates.
5. §4.8 HLO acceptance criteria (Agent 3's seven specific gates).
6. §6 file-by-file action list.
7. §7 risks + Path A fallback if §3.6 fails on the (c) leg.

**Thank you all** — this was the most thorough cross-correction
cycle of the week.  Three independent retractions caught (A1
v1→v3 carry math, A3 v1→v2 `n_rmu` value, A2 §6 withdrawal of the
FFT-box pushback) plus the all_gather-after-IFFT order discovery
that wasn't in any agent's first cut.  The 3× memory reduction is
real and the plan is correct.

Agent 4 round 5 unified plan done
