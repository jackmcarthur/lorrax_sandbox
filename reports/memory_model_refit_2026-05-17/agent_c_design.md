# Joint `(r_chunk, gflat_chunk_size, band_chunk)` picker — design

## 1. Key technical finding — C and D transients do NOT coexist

Inspecting `isdf_fitting.py:2442–2496`:

```python
with timing.section("zeta_fit.chunk.fit_one_rchunk"):
    zeta_chunk = fit_one_rchunk(...)
    zeta_chunk.block_until_ready()           # ← C transients fully released here
# ... fused-jit returns; P_pair slots / FFT box freed by XLA ...
with timing.section("zeta_fit.chunk.h5_write"):
    gflat_acc = accumulate_rchunk_to_gflat(rchunk=zeta_chunk, ...)
    del zeta_chunk
```

`fit_one_rchunk` is a fused jit whose entire transient slot table is released the moment XLA returns; `accumulate_rchunk_to_gflat` is a SEPARATE jit that allocates its FFT-box transient afresh. Persistent that straddles both:
- centroids (L+R, full nb, μ-sharded)
- L_q (μ × μ, q-replicated)
- gflat_acc (per-rank μ-flat ζ accumulator)
- zeta_chunk (output of C, input to D — sized (n_q_disk, μ_padded, r_chunk) / p_xy c128)

Everything else is in either C's lifetime or D's lifetime, never both.

**Implication:** the true HWM is `persistent + max(C_transient, D_transient)`. The planner's `max(A, B, C, D)` already gives this **provided each peak's `_peak_X` dict correctly enumerates its simultaneously-live tensors.** Joint optimization buys **nothing** for HWM reduction; shrinking gflat_chunk_size frees no headroom for r_chunk.

## 2. Why r_chunk is leftover-on-the-table

The actual problems are:

(a) **`target_utilization = 0.80` is too tight.** Empirics: r=24576 ran at HWM=66 GB on a 70 GB budget = 94% utilization. The planner refuses to go past 80% (= 56 GB).

(b) **`α_C` is under-counted.** Model predicts r=36k fits at util=0.94; empirics OOM at r=28.6k. Either `pair_density_slots=3` should be 4-5, or there's a Peak C term (the in-loop FFT box, or Z_q intermediate) that's misattributed to "absorbed into slots" but actually concurrent. Agent A's audit should resolve.

(c) **`D_accumulate` over-counts persistent state.** The reported D bottleneck includes the gflat_acc + zeta_chunk persistent terms that are also in C's persistent base. This isn't strictly a bug for `max(C, D)` correctness, but it makes the D number alarming.

## 3. Algorithm — `plan_gflat_chunks_v2`

```python
def plan_gflat_chunks_v2(meta, mesh, budget_gb,
                         target_utilization=0.94,     # raised from 0.80
                         bc_floor_factor=4):
    # ---- persistent footprint (live across all phases) ----------------
    persistent = (
        2 * c128(nk, ns, mu, nb_total, shard=p_xy)        # centroids L+R
        +     c128(nq, mu, mu,        shard=p_xy)         # L_q
        +     c128(nq_disk, mu, ngkmax, shard=p_xy)       # gflat_acc
    )
    B = target_utilization * budget_gb * 1e9
    H = B - persistent                                     # shared headroom

    # Step 1.  r_chunk — sized by Peak C transient ONLY.
    α_C = (pair_density_slots * c128(nk, ns, ns, mu, shard=p_xy)
           +                    c128(nq_disk, mu,    shard=p_xy))   # +zeta_chunk
    r_chunk = min(n_rtot, int(H / α_C))
    r_chunk = round_down(r_chunk, p_xy)
    r_chunk = max(r_chunk, mu)

    # Step 2.  gflat_chunk_size — sized by Peak D transient ONLY.
    fft_per_row = n_rtot * 16 * fft_box_factor
    H_D = H - c128(nq_disk, mu, r_chunk, shard=p_xy)
    cs_one_shot = ceil(nq_disk * mu / p_xy)
    if cs_one_shot * fft_per_row <= H_D:
        gflat_chunk_size = None              # one-shot
    else:
        gflat_chunk_size = max(bc_floor_factor, int(H_D / fft_per_row))

    # Step 3.  band_chunk — sized by Peak A transient ONLY.
    per_unit_bc = c128(nk, ns, n_rtot, shard=p_xy) * fft_box_factor
    band_chunk = round_pow2_down(min(nb_total, int(H / per_unit_bc)))
    band_chunk = max(band_chunk, p_xy)
    band_chunk = ceil_to_multiple(band_chunk, p_xy)
    band_chunk = max(band_chunk, bc_floor_factor * p_xy)

    return GFlatChunkPlan(...)
```

**Picker order (reversed from current):** r_chunk first → gflat_chunk_size second → band_chunk third.

**Floors:**
- `r_chunk ≥ mu` (work-amortization floor — Σ_μν output dominates per-chunk overhead below μ)
- `r_chunk` rounded down to a multiple of `p_xy` (sharding divisor)
- `gflat_chunk_size ≥ 4` (cuFFT plan-amortization floor; the per-iter body is tiny relative to per-chunk solve)
- `band_chunk ≥ p_xy` and a multiple thereof; `≥ 4·p_xy` for plan amortization

**Three independent constraints** sharing `H`:

```
α_C · r_chunk                                ≤ H − ε_C
cs · fft_per_row + zeta_chunk(r_chunk)       ≤ H
per_unit_bc · band_chunk + centroid_out      ≤ H
```

No coupling. Each variable saturates its own constraint.

## 4. Sanity check on 80 Ry CrI3 (16-GPU, 4×4 mesh, 70 GB budget)

Inputs: `nk=36, ns=2, mu=1520, nb=150, n_rtot=1.125M, ngkmax=59990, nq=nq_disk=36, p_xy=16, c128=16, fft_factor=4, slots=3, util=0.94`.

| term | formula | GB/dev |
|---|---|---|
| centroids L+R | `2·36·2·1520·150·16/16/1e9` | 32.83 |
| L_q | `36·1520²·16/16/1e9` | 5.20 |
| gflat_acc | `36·1520·59990·16/16/1e9` | 3.28 |
| **persistent** | | **41.31** |
| budget B = 0.94 × 70 | | 65.80 |
| **headroom H** | | **24.49** |

**Pick `r_chunk`:** `α_C = (3·36·4·1520·16 + 36·1520·16)/16 ≈ 6.77e5 B/r-row` → `r_chunk ≤ 24.49e9/6.77e5 ≈ 36,180` → mult-of-16 → **r_chunk = 36,176**. But empirics show r=28,672 OOMs. **The model under-predicts α_C** — Agent A must calibrate.

**Pick `gflat_chunk_size`** (assuming r_chunk corrected to ~24576): `zeta_chunk_carry = 21.5 GB`, `H_D = 24.49 − 21.5 = 3.0 GB`, `cs_one_shot · fft_per_row = 246 GB ≫ 3 GB` → `gflat_chunk_size ≈ 40`. Empirical optimum was 360 → the new picker is much smaller. Harmless since D is no longer binding.

**Pick `band_chunk`:** `per_unit_bc = 0.324 GB` → `bc_cap ≈ 75` → pow2-down → 64. Empirical was 32. 64 is fine (saturates Peak A budget; halves bc-loop iters).

**Predicted HWM** ≈ 41 (persistent) + 24.5 (C transient) = 65.8 GB ≈ empirical 66 GB. ✓

## 5. One-line summary

**Not a deep refactor:** the current sequential picker is structurally correct (transients don't coexist, so `max(C, D, A, B)` already does the right thing) — the fix is:
- (a) Move `r_chunk` pick before `band_chunk`
- (b) Raise `target_utilization` from 0.80 to ~0.94
- (c) Calibrate `α_C` against actual slot count (Agent A)
- (d) Make `gflat_chunk_size`'s headroom be `H − zeta_chunk_carry(r_chunk)` instead of double-counting persistent into D's bottleneck call

Roughly **30 LOC** of edits inside `plan_gflat_chunks`, no signature change, no caller changes.
