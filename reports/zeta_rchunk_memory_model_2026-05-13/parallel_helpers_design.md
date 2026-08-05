# Parallel-helpers design — unified forward `ψ(G) → ψ(rchunk)`

**Status**: v1 draft for Agent 2 (implementer). Read alongside
`PATH_D_PICKUP.md` (§0 is load-bearing) and `agent_2_structural_fix.md`
(prior Path D sketch — this doc supersedes its §4a-e framing with a
cleaner "structural twin of `accumulate_rchunk_to_gflat`" target).

## 0. Principle reminder

> Zero replicated intermediates or repeated work in `fit_one_rchunk`.

The pipeline needs **two exactly parallel `shard_map` helpers** —
structurally identical except for I/O endpoints. The reverse already
exists clean (`common.wfn_transforms.accumulate_rchunk_to_gflat`,
`wfn_transforms.py:468`). This doc specifies the forward that mirrors
it.

If at any point we find ourselves writing a Python-unrolled k-chunk or
bc-loop inside this helper or its caller, we've drifted off the
principle — re-read §0 of `PATH_D_PICKUP.md`.

## 1. The symmetry table

|                  | **Forward (new)**                              | **Reverse (existing)** `accumulate_rchunk_to_gflat`     |
|------------------|------------------------------------------------|----------------------------------------------------------|
| Direction        | `ψ_{n,k}(G) → ψ_{n,k}(rchunk)`                  | `ζ_{q,μ}(rchunk) → ζ_{q,μ}(G)`                           |
| Input shape      | `(nk, nb_total, ns, ngkmax)`                    | `(n_q, n_rmu_padded, r_len)`                              |
| Input spec       | `P(None, ('x','y'), None, None)`                | `P(None, ('x','y'), None)`                                |
| Output shape     | `(nk, nb_total, ns, r_len)`                     | `(n_q, n_rmu_padded, ngkmax)`                             |
| Output spec      | `P(None, ('x','y'), None, None)`                | `P(None, ('x','y'), None)`                                |
| Flat scan axis   | `N = nk · nb_local`  (row = `(k, n)` pair)      | `N = n_q · n_mu_local`  (row = `(q, μ)` pair)             |
| Per-iter body    | gather `g_index[k_row]` → `_box_kernel` (scatter sphere → FFT box) → **IFFT** → reshape → `dynamic_slice` flat-r slab → **Bloch phase on slice (+sign)** → write to output slab | `dynamic_slice` rows → **Bloch phase on slice (−sign)** → pad-to-box (`dynamic_update_slice`) → reshape → **FFT** → gather `sphere_idx[q_row]` → accumulate into output |
| Chunk knob       | `chunk_size: int` rows along flat `N` axis      | same                                                       |
| Divisibility     | none — flat `N` is zero-padded to `n_chunks·cs` | same                                                       |
| FFT box transient | `c128[cs, ns, nx, ny, nz]` per scan iter, aliased across iters | `c128[cs, nx, ny, nz]` per scan iter, aliased across iters |
| Caching key      | shapes + shardings + `chunk_size` + g-index hash + kvecs shape | shapes + shardings + `chunk_size` + sphere-id + qvec shape |
| Donation         | none (output is a fresh slab)                   | `gflat_acc` donated; ζ_G is in-place add                  |

The forward has one more inner-axis (`ns` for spin) than the reverse.
Its FFT box is `ns ×` bigger per iter. That's the only intrinsic shape
asymmetry; pick `chunk_size` accordingly.

## 2. Proposed forward signature

```python
def gflat_to_rchunk(
    psi_G: jax.Array,                                   # (nk, nb_total, ns, ngkmax) c128
    g_index: np.ndarray | jax.Array,                    # (nk, ngkmax) int32, flat-box indices
    *,
    mesh: Mesh,
    fft_grid: Sequence[int],                            # (nx, ny, nz)
    r0,                                                 # int or traced scalar — flat-r start
    r_len: int,                                         # static slab length
    kvecs_frac: np.ndarray | jax.Array | None = None,   # (nk, 3) float64; None = no phase
    norm: str = "ortho",
    chunk_size: int | None = None,                      # default one-shot
) -> jax.Array:                                         # (nk, nb_total, ns, r_len) c128
```

Inside one `shard_map` over `('x','y')`:

```python
in_spec  = P(None, ('x', 'y'), None, None)
out_spec = P(None, ('x', 'y'), None, None)

@partial(shard_map, mesh=mesh, in_specs=(in_spec, ..., P()), out_specs=out_spec, check_rep=False)
def _kernel(psi_, g_index_, r0_, kvecs_):
    # Per-rank: (nk, nb_local, ns, ngkmax) — bands flat-sharded over ('x','y').
    nk_, nb_local_, ns_, _ = psi_.shape
    N = nk_ * nb_local_
    psi_flat   = psi_.reshape(N, ns_, ngkmax)              # row = (k, n) pair
    if pad_N:
        psi_flat = jnp.pad(psi_flat, ((0, pad_N), (0, 0), (0, 0)))

    out_flat = jnp.zeros((N + pad_N, ns_, r_len_i), dtype=psi_.dtype)

    # Loop-invariant per-iter setup (decode r0_ once, like the reverse)
    if phx is not None:
        r_idx_slab = r0_ + jnp.arange(r_len_i, dtype=jnp.int32)
        # ... rx_slab, ry_slab, rz_slab (same as reverse)

    def body(out, i):
        i0   = i * cs
        sub  = jax.lax.dynamic_slice_in_dim(psi_flat, i0, cs, axis=0)     # (cs, ns, ngkmax)
        k_row = jnp.clip((i0 + jnp.arange(cs)) // nb_local_, 0, nk_ - 1)  # (cs,)

        # Scatter sphere → box (mirror of reverse's take_along_axis).
        # _box_kernel handles the (cs, ns, ngkmax) → (cs, ns, nx, ny, nz)
        # scatter via per-row g_index[k_row].
        g_per_row = g_index_c[k_row]                                    # (cs, ngkmax) int32
        box = _box_kernel_flat(sub, g_per_row, ngkmax=ngkmax)           # (cs, ns, nx, ny, nz)
        rb  = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)
        rb_flat = rb.reshape(cs, ns_, n_rtot)
        slab = jax.lax.dynamic_slice_in_dim(rb_flat, r0_, r_len_i, axis=-1)
        if phx is not None:
            # Per-k Bloch phase exp(+2πi k·r) on the (cs, r_len) slab.
            # Per-axis tables phx/phy/phz pre-built; gather per row by k_row.
            phx_q = phx[k_row][:, rx_slab]   # (cs, r_len)
            phy_q = phy[k_row][:, ry_slab]
            phz_q = phz[k_row][:, rz_slab]
            slab = slab * (phx_q * phy_q * phz_q)[:, None, :]
        return jax.lax.dynamic_update_slice_in_dim(out, slab, i0, axis=0), None

    out_flat, _ = jax.lax.scan(body, out_flat, jnp.arange(n_chunks, dtype=jnp.int32))
    if pad_N:
        out_flat = out_flat[:N]
    return out_flat.reshape(nk_, nb_local_, ns_, r_len_i)
```

The body is the structural mirror of `accumulate_rchunk_to_gflat`'s
body:
- Forward gather is `g_index[k_row]` and a scatter (`_box_kernel_flat`)
  in place of reverse's `sphere_idx[q_row]` + `take_along_axis`.
- Forward applies Bloch phase **after** IFFT slice (sign `+1`); reverse
  applies Bloch phase **before** FFT pad (sign `−1`).

### Why `_box_kernel_flat` instead of today's `_box_kernel`

Today's `_box_kernel` (in `wfn_transforms.py`) takes shape
`(..., ngkmax)` with `g_index` of shape `(..., ngkmax)` — the "..."
broadcasts. We need per-row `g_index[k_row]` of shape `(cs, ngkmax)`
applied to `psi_flat[cs, ns, ngkmax]`. Two options:

1. **Reuse `_box_kernel`** unchanged: pass `psi_flat` reshaped as
   `(cs, ns, ngkmax)` and `g_index_per_row` as `(cs, ngkmax)` — the
   existing broadcast contract (in the bands-replicated forward case
   it's `(nk, nb, ns, ngkmax) @ (nk, ngkmax)`) should handle this. **Verify
   first** by tracing on synth shapes; if it doesn't (because today it
   assumes only one leading "k" axis with `g_index` indexed by it),
   add a small `_box_kernel_flat(rows, g_index_per_row, ngkmax)`
   that's identical but documents the contract `(cs, ns, ngkmax) +
   (cs, ngkmax) → (cs, ns, nx, ny, nz)`. **Preferred:** reuse, no new
   helper. (Agent 2: check this on a 20-line trace test first.)

2. New helper. Avoid unless (1) fails.

### Why the output is `(nk, nb_total, ns, r_len)`, not flat

Output reshape happens outside the scan, after the `out_flat[:N]`
truncation. The flat → `(nk_, nb_local_, ns, r_len)` reshape on the
per-rank tensor is a bitcast (no data motion). Caller sees the same
`(nk, nb_total, ns, r_len)` global shape with `P(None, ('x','y'), None,
None)` sharding.

### Caching pattern

Identical to `accumulate_rchunk_to_gflat`:

```python
key = (
    tuple(int(s) for s in psi_G.shape),
    fft_grid_t, r_len_i, int(ngkmax),
    norm, kvecs_shape, cs, n_chunks, pad_N,
    g_index_id,                  # hash(g_index.tobytes())
    _sharding_key(psi_G),
)
fn = _GFLAT_TO_RCHUNK_CACHE.get(key)
```

## 3. Symmetry-breakers (acknowledged)

These are the places where the two helpers cannot be identical. List
them here explicitly so the implementer knows they're features, not
oversights:

1. **I/O endpoint.** `accumulate_rchunk_to_gflat`'s input
   (`rchunk`) is **already on device**, sharded by the caller.
   `gflat_to_rchunk`'s input lives on the host (`PsiGStore._host_tiles`).
   The forward helper *itself* takes ψ(G) on device — keeping it pure
   and symmetric with the reverse. The host-to-device transfer is the
   caller's job: `PsiGStore` gains one new method, **`pull_psi_g_full()`**,
   that does exactly the existing `_pull` shard_map+io_callback pattern
   (`psi_G_store.py:332-345`) but for the full host tile, not a
   per-bc slice.

   Per-rank footprint: `nk · nb_local · ns · ngkmax · 16` bytes. For
   CrI3 80 Ry 6×6 charge on 16 GPUs: `36 · 25 · 2 · 70k · 16 ≈ 5 GB/rank`.
   Same byte count as the current per-bc fetch × n_bc — no inflation
   versus today's concat. **The concat is what we're eliminating, not
   the per-rank ψ(G) residency.**

2. **Bloch phase sign.** Forward uses `+1` (post-IFFT, `exp(+2πi k·r)`).
   Reverse uses `−1` (pre-FFT, `exp(−2πi q·r)`). Both call
   `apply_bloch_phase_on_slice(..., sign=±1)` — already a single
   source of truth in `wfn_transforms.py`. The body sketch above
   inlines the same per-axis decompose to avoid an extra shape
   broadcast, but `apply_bloch_phase_on_slice` is fine if the
   implementer prefers reuse over the inline.

3. **FFT direction.** Forward `ifftn`, reverse `fftn`. Same `norm`
   keyword; today's `to_rchunk` calls with `norm="ortho"`,
   `accumulate_rchunk_to_gflat` with `norm="backward"`. The new
   forward must preserve `"ortho"` to match the legacy
   `get_sharded_wfns_rchunk_slice` 1/√N convention (see
   `psi_G_store.py:288-290`).

4. **Sphere endpoint.** Forward **scatters** from G-sphere into FFT
   box via `_box_kernel`'s `dynamic_update_slice`/scatter equivalent.
   Reverse **gathers** from FFT box onto G-sphere via
   `take_along_axis(G, sphere_idx[q_row], ...)`. The shape of the
   "sphere index" tensor is `(nk, ngkmax)` (forward, `g_index`) vs
   `(n_q, ngkmax)` (reverse, `sphere_idx`); the role is identical —
   per-row sphere index for the chunk's rows.

5. **Donation.** Reverse donates `gflat_acc` (in-place add).
   Forward writes to a fresh `out_flat`; no donation. (Could donate a
   pre-allocated output later if the caller wants to reuse buffer
   across rchunks, but that's out of scope for the structural fix.)

That's it. Everything else — the flat-axis chunking, the scan over
chunks, the per-iter aliased FFT box, the caching, the per-row
gather table for the sphere index, the loop-invariant slab-cell
decode — is **identical between the two helpers**.

## 4. Integration into `_make_fit_one_rchunk_kernel._kernel`

Today (`isdf_fitting.py:1268-1286`):

```python
_b0 = int(band_range_full[0])
_l_lo = int(band_range_left[0]) - _b0
_l_hi = int(band_range_left[1]) - _b0
_r_lo = int(band_range_right[0]) - _b0
_r_hi = int(band_range_right[1]) - _b0
psi_Y_parts = []
for bc_range in band_chunk_ranges:                     # Python unroll  ← principle violation
    psi_Y_parts.append(psi_G_store.fetch_psi_rchunk(   # N copies of FFT-box transient in HLO
        bc_range, r_start_dyn, actual_n_rchunk))
psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)      # concat doubles peak
del psi_Y_parts
psi_l_Y_sm = (psi_Y_full[:, _l_lo:_l_hi, :, :] / norms_l[None, :, None, None])
psi_r_Y_sm = (psi_Y_full[:, _r_lo:_r_hi, :, :] / norms_r[None, :, None, None])
del psi_Y_full
```

After:

```python
_b0 = int(band_range_full[0])
_l_lo = int(band_range_left[0]) - _b0
_l_hi = int(band_range_left[1]) - _b0
_r_lo = int(band_range_right[0]) - _b0
_r_hi = int(band_range_right[1]) - _b0

psi_G_full = psi_G_store.pull_psi_g_full()             # NEW: one io_callback, all bcs
psi_Y_full = gflat_to_rchunk(                          # NEW: one shard_map, scan over (nk·nb_local)
    psi_G_full, psi_G_store.g_index,
    mesh=mesh_xy, fft_grid=meta.fft_grid,
    r0=r_start_dyn, r_len=actual_n_rchunk,
    kvecs_frac=psi_G_store.kvecs_frac,
    norm="ortho",
    chunk_size=cfg.memory.gflat_to_rchunk_chunk_size,  # new knob, default = one-shot
)
psi_l_Y_sm = (psi_Y_full[:, _l_lo:_l_hi, :, :] / norms_l[None, :, None, None])
psi_r_Y_sm = (psi_Y_full[:, _r_lo:_r_hi, :, :] / norms_r[None, :, None, None])
del psi_Y_full
```

The `bc_loop` and the `concatenate` are gone. The chunk knob lives on
the new helper, not on the kernel.

`pull_psi_g_full()` is a one-liner extraction from the existing
`fetch_psi_rchunk` body (`psi_G_store.py:339-345`) — same shard_map +
io_callback, but the host-side `_slice_local_tile` returns the full
tile (no `[b_lo:b_hi]` slice), and no `to_rchunk` call after.

### `c_q_from_psi_sm` callers (`isdf_fitting.py:1661, 1667`)

Same pattern in `fit_centroids_kernel` (the CCT analog). Audit there
in pass 2; structurally identical refactor. **Agent 2: do `_kernel` (ζ
path) first, validate, then mirror in the CCT path** — same diff
shape, lower stakes (CCT is cheaper).

### `norms` divide

Today's `psi_Y_full[:, _l_lo:_l_hi, :, :] / norms_l[...]` happens
after the concat. After the refactor, it still happens after
`gflat_to_rchunk` returns — no change to where norms live. We could
fold the divide into `psi_l_rmuT_X_fit` / `psi_r_rmuT_X_fit`
pre-multiplied at construction time (cleaner: divide constants once),
but that's an orthogonal optimization — out of scope here.

## 5. Memory model after the refactor

Per-rank, inside the new shard_map body:

| Buffer                                              | Lifetime          | Bytes (CrI3 80 Ry 6×6 charge, 16 GPUs) |
|-----------------------------------------------------|-------------------|----------------------------------------|
| `psi_flat` `(N, ns, ngkmax)`                        | scan-resident     | `N · ns · ngkmax · 16` ≈ 5 GB           |
| `out_flat` `(N, ns, r_len)`                         | scan-resident     | `N · ns · r_len · 16` ≈ varies w/ r_len |
| `box` `(cs, ns, nx, ny, nz)` (FFT box)              | per-iter, aliased | **1 slot** ≈ `cs · ns · n_rtot · 16`    |
| cuFFT scratch                                       | per-iter, aliased | ~1× FFT box                              |

**The single FFT-box slot is the structural fix.** XLA's scan-internal
allocator aliases the box across iters; the slot count for that
buffer-class drops from `n_bc · n_kchunk` (today: 58 in the HLO
findings) to **1** (plus maybe 1-2 fusion-adjacent slots; the prior-art
on `accumulate_rchunk_to_gflat` HLO settled at 2-3 total).

Counterfactual without the refactor: today the bc-loop is unrolled in
the jit, so the FFT box appears `n_bc = N_bc · n_kchunk` times in the
HLO as concurrent live slots. That's the 200 GB peak from the
`agent_1_hlo_verify.md` findings.

## 6. Validation plan

### 6a. CPU bit-identity vs current `to_rchunk` + concat

Smallest possible: synthetic ψ(G) at MoS2 3×3 scale.

```python
# Reference (today's path):
psi_ref_parts = []
for bc in band_chunk_ranges:
    psi_ref_parts.append(to_rchunk(
        psi_G_full[:, b_lo[bc]:b_hi[bc], ...], g_index, fft_grid,
        r_start, r_len, kvecs_frac=kvecs_frac, norm="ortho", mesh=mesh))
psi_ref = jnp.concatenate(psi_ref_parts, axis=1)

# New (one call):
psi_new = gflat_to_rchunk(
    psi_G_full, g_index, mesh=mesh, fft_grid=fft_grid,
    r0=r_start, r_len=r_len, kvecs_frac=kvecs_frac, norm="ortho",
    chunk_size=None)  # one-shot

assert jnp.allclose(psi_ref, psi_new, rtol=1e-10, atol=1e-12)
```

Repeat with `chunk_size = 1`, `chunk_size = 3`, `chunk_size = N+1`
(padded), `chunk_size = N` (one-shot, no scan). All bit-identical
modulo scan-summation order — but the body has no accumulator, so
order is irrelevant; **expect ULP-class differences only, well under
`atol=1e-12`**.

Reuse the `tests/test_wfn_transforms.py` scaffolding Agent 2 already
added for `to_rchunk_inner`. Three new tests:
- `test_gflat_to_rchunk_no_phase`
- `test_gflat_to_rchunk_with_phase`
- `test_gflat_to_rchunk_chunked_matches_oneshot`

### 6b. HLO dump test — slot count predictions

Drop the kernel into `runs/CrI3/M_6x6_80Ry_2026-05-07/`-class config
with `XLA_FLAGS='--xla_dump_to=/path --xla_dump_hlo_pass_re=.*'`.

Predictions, by buffer class in the `module_*.jit_*_kernel.memory-usage-report.txt`:

| Buffer                                         | Before (lorrax_A HLO dump) | After (Path D) |
|-----------------------------------------------|----------------------------|----------------|
| `c128[k_chunk, bc, ns, nx, ny, nz]` FFT box    | **58 concurrent slots**    | **1 slot**     |
| Concat staging `c128[nk, nb_total, ns, r_chunk]` | 1 (eager concat result)  | absent (out_flat is the only such buffer) |
| Total preallocated-temp                        | ~200 GiB                   | ~10-30 GiB     |

**Pass criterion**: FFT-box slot count = 1 ± 2 fusion-adjacent slots.
**Fail criterion**: > 5 such slots — the scan didn't alias.

### 6c. End-to-end ζ-fit on MoS2 3×3

Run `fit_zeta` on the existing MoS2 3×3 cohsex reference. Compare
`eqp0.dat` to lorrax_A baseline (bit-identical within existing
COHSEX-vs-LORRAX tolerance, `rtol≈1e-8` on Σ).

### 6d. CrI3 6×6 80 Ry — the OOM-was-the-bug killer

The 2nd HLO dump at `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_k1_2026-05-13/`
shows the original OOM. After Path D:
- `nvidia-smi memory.used` at fit_zeta peak: from 196 GiB (OOM) to
  expected 10-30 GiB.
- Run at `psig_k_chunk_size=0` (no k-chunking — Path D makes it
  unnecessary). Verify it doesn't OOM.

This is the validation that the structural fix worked at scale, not
just at unit-test scale.

## 7. Open questions for Agent 2 to surface (write back to `parallel_helpers_discussion.md` if hit)

1. **`_box_kernel` broadcast contract.** Does the existing
   `_box_kernel(psi[cs, ns, ngkmax], g_index_per_row[cs, ngkmax])`
   broadcast correctly, or does it require explicit reshape? If the
   latter, decide whether to add a `_box_kernel_flat` variant or just
   inline a `jnp.zeros + dynamic_update_slice` scatter in the body.
2. **`PsiGStore.pull_psi_g_full()` placement.** Should it live as a
   public method, or as a `@property`-style cached
   `psi_G_full_device`? Once-per-`begin_rchunk` semantics matter — we
   don't want to re-pull on every fit_zeta call within the same
   r-chunk window. Suggest property, lazily computed, invalidated by
   `end_rchunk`.
3. **`gflat_to_rchunk_chunk_size` default.** One-shot (`cs = N`) is
   the cleanest default — it produces a single-iter scan that XLA
   folds away, giving the same fused HLO as a non-scan body. Add the
   config knob only if a deployment needs to cap the FFT box smaller
   than that. Suggest leaving the knob unwired in `gw_config.py` until
   the CrI3 80 Ry HLO dump confirms whether one-shot is feasible at
   scale.
4. **CCT path (`c_q_from_psi_sm`).** Does the existing `psi_l_Y` /
   `psi_r_Y` for CCT come from `to_rmu` or `to_rchunk`? Different
   helper — symmetric refactor in a separate pass, not Path D scope.

## 8. Non-scope (deferred follow-ups)

These are the other principle violations called out in
`PATH_D_PICKUP.md` §0. Not in this design's scope, but flag them so
the Agent-2 commit doesn't accidentally regress them:

- `solve_zeta` q-batch Python loop (`isdf_fitting.py:1119-1141`) —
  separate structural defect, separate follow-up.
- Peak A centroid-load FFT box (`gflat_memory_model._peak_A_centroid_load`)
  — single-slot but unsharded; same principle violation, separate fix.
- Planner's `band_fft_pool` term — mark for removal **in the same
  commit that lands `gflat_to_rchunk`**, since it codifies a defect
  that Path D eliminates.

---

Agent 1 design v1 done
