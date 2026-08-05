# Audit — `accumulate_rchunk_to_gflat` as a "clean reverse-direction reference"

**File:** `sources/lorrax_A/src/common/wfn_transforms.py:468-673`
**Branch read:** `sources/lorrax_A` (read-only; verified the function is on the
trunk version Agents 1/2 are mirroring).
**Auditor:** Restart Agent 3, 2026-05-13.

## 1. Verdict

**Clean, with two minor caveats that do not violate the principle.** The function
is a faithful instance of the "single shard_map containing a `lax.scan` whose
body holds all per-iter transients" pattern. Per-rank locality holds for every
intermediate buffer that scales with `r_chunk · n_rtot`. There is no Python-
unrolled inner loop, no `with_sharding_constraint` inside the shard_map body,
no cross-rank collective in the body, and the per-iter FFT box is structurally
aliasable by XLA's scan-internal allocator. Agents 1 and 2 can safely emulate
the structure.

The two caveats (both small / acceptable / documented below):
- Closure-captured **phase tables** (`phx/phy/phz`) and **sphere index table**
  (`sphere_c`) are replicated across every rank by shard_map's closure
  semantics. They are *inputs*, not *intermediates*, and they are tiny
  (`O(n_q · (nx+ny+nz))` and `O(n_q · ngkmax)` respectively). Not a defect by
  the principle (the principle targets replicated *intermediates / work*).
- A `jnp.pad` allocation on `acc_flat` (line 623) executed when
  `chunk_size` does not exactly divide `N` materialises a transient
  larger-than-donation buffer once per call. The default `chunk_size=None`
  (one-shot) takes `cs=N` and `pad_N=0`, bypassing this branch entirely.
  Counts as a budgeted single-shot allocation when used, not as scan slot
  pile-up.

## 2. Evidence — line-by-line

### 2.1 One shard_map over `('x','y')`

`wfn_transforms.py:613-616`:

```
@partial(shard_map, mesh=mesh,
         in_specs=(in_spec, in_spec, P()),
         out_specs=out_spec,
         check_rep=False)
def _kernel(rch_, acc_, r0_):
```

with `in_spec = out_spec = P(None, ('x','y'), None)` (lines 610-611). Both
`rchunk` and `gflat_acc` are sharded **only** on the μ-flat axis; the q and
r-axes are replicated, but that replication is structural to the data
contract (every rank gets the same q list and the full r-chunk slab — q is
small, r-chunk slab is per-call chunked by the caller). The scalar `r0_`
enters fully replicated (`P()`). One shard_map. ✓

### 2.2 `lax.scan` is inside the body

`wfn_transforms.py:663-664`:

```
acc_flat, _ = jax.lax.scan(
    body, acc_flat, jnp.arange(n_chunks, dtype=jnp.int32))
```

This sits inside `_kernel`, the shard_map'd function. **The scan is inside,
not outside.** No Python `for chunk in ...:` exists anywhere — the only Python
unrolling is at trace time for the static cache key. ✓

### 2.3 No `with_sharding_constraint` inside the body

Grepped `_kernel` (lines 617-667). Zero occurrences of `with_sharding_constraint`,
`jax.lax.with_sharding_constraint`, or any `psum` / `all_gather` /
`all_to_all` / `pmean`. The WhileOp/SPMD trap that bit Path D scaffolding
elsewhere does not apply here. ✓

### 2.4 No Python loops inside the body

`body` (lines 639-661) is a pure expression-level function — `dynamic_slice_in_dim`,
arithmetic, conditional `if phx is not None:` resolved at trace time (Python
`if`, not `lax.cond`, on a closure-time bool — this is correct because
`phx` is `None`-or-not at trace time, not data-dependent), `jnp.fft.fftn`,
`take_along_axis`, `dynamic_update_slice_in_dim`. No `for`/`while`. ✓

### 2.5 Per-iter buffers are scan-internal carry-irrelevant ⇒ aliasable

The body produces:
- `sub` (cs, r_len_i)         (line 641)
- `q_row` (cs,)                (line 642-643)
- `phx_q`/`phy_q`/`phz_q` (cs, r_len)   (lines 649-651)
- `buf` (cs, n_rtot)           (line 653 — fresh `jnp.zeros`)
- `box` (cs, nx, ny, nz)       (line 655 — reshape, alias)
- `G` (cs, n_rtot)             (line 656 — FFT result)
- `contrib` (cs, ngkmax)       (lines 657-658)
- `acc_sub` (cs, ngkmax)       (line 659)

**None of these are part of the scan carry.** The carry is only `acc_flat`
(plus the auxiliary `None` ys). XLA's scan-internal allocator is allowed to,
and on this body should, alias `buf`/`box`/`G`/`contrib`/`acc_sub` to a
single set of slots reused across iterations. This is the load-bearing
mechanism behind the principle's "O(1) slots not O(n_chunks)" guarantee.

The FFT-box buffer in particular (`(cs, nx, ny, nz)` c128 — the heavy slot)
is a fresh `jnp.zeros` on line 653 followed by a reshape; XLA's scan pass
recognises this idiom and assigns a single transient slot. ✓

### 2.6 All intermediates are per-rank-local

After the shard_map enters, `rch_` and `acc_` arrive at per-rank shape
`(n_q, n_mu_local, r_len)` / `(n_q, n_mu_local, ngkmax)`. The reshape to
`(N, *)` on lines 619-620 is local. The FFT axes are the spatial (x,y,z),
and the data is full-extent on those axes (they're replicated in the
input contract, and the FFT computation is local cuFFT — the docstring
even calls this out, line 508-510). The `take_along_axis` gather on
line 657-658 reads `G` (per-iter, per-rank) and `sphere_c[q_row]`
(closure const + small gather). No buffer scales with `n_rmu_padded` —
only `n_mu_local`. ✓

### 2.7 Donation works in the default path

`fn = jax.jit(_kernel, donate_argnums=(1,))` (line 669) donates the
`gflat_acc` argument. In the default case (`chunk_size=None ⇒ cs=N`,
`n_chunks=1`, `pad_N=0`), the `jnp.pad` branches on lines 621-623 are
skipped, the reshape on line 620 is in-place, and the donation
threads through. ✓ (caveat in §3 about non-default `chunk_size`).

### 2.8 No cross-rank collectives

Grepped the body for `psum`, `all_gather`, `all_to_all`, `ppermute`,
`pmean`, `reduce_scatter`. Zero hits. The docstring claim "No cross-rank
collectives in the body" is accurate. ✓

### 2.9 Predicted slot behaviour (no GPU dump available on login node)

For a typical config with `cs · n_rtot` ≪ device memory and `n_chunks > 1`:
- **1 carry slot** for `acc_flat`: `(N + pad_N, ngkmax)` c128.
- **1 per-iter slot** (aliased across iterations) for the FFT box:
  `(cs, n_rtot)` c128 ≡ `(cs, nx, ny, nz)`.
- **Small per-iter slots** for `sub` / `phx_q` / `q_row` / `contrib`
  (all linear in `cs`, fold into the same alias group or live in tiny
  separate slots).

Expected total preallocated transient FFT-box pile-up: **O(1) — one slot
of size `cs · n_rtot · 16 B`**, regardless of `n_chunks`. This is exactly
the contrast against the Python-unrolled bc-loop pattern documented in
`PATH_D_PICKUP.md §0` (which gives `N_BC × S_fft` slots).

Without a GPU HLO dump I cannot show the slot table directly, but the
code structure is consistent with the lorrax/lorrax_B test history cited
in memory `[[feedback_path_d_scaffolding_pattern]]`: "Python-unrolled
inner loop in jit ⇒ N× unsharded slots; fix is scan-INSIDE-shard_map".
This function uses the *fix* pattern. The slot pileup mode is structurally
absent.

## 3. Caveats (not defects)

### 3.1 Closure-replicated phase / sphere tables

`phx`, `phy`, `phz` (shape `(n_q, n*)`) and `sphere_c` (shape `(n_q, ngkmax)`)
are built at cache-construction time outside `_kernel` (lines 598-605) and
captured by closure. Inside the shard_map'd body they are treated as
replicated arrays — every rank holds a full copy.

This is acceptable under the principle because:
- They are *inputs* (constants in the kernel), not *intermediates of the
  procedure*.
- They are small: phase table total ≈ `n_q · (nx+ny+nz) · 16 B`; for
  CrI3 6×6 80 Ry with `n_q=36`, `nx≈120`: ~210 KB. `sphere_c` for
  `ngkmax ≈ few × 10⁴`: a few MB.
- Sharding them on `('x','y')` would require an `all_gather` inside the
  body — strictly worse.

If a future profiler shows these constants are getting baked into HLO
constants rather than data parameters (which sometimes inflates binary
size), promoting them to runtime-replicated `P()` arguments would be a
hygiene improvement, but not load-bearing for memory.

### 3.2 `jnp.pad` on non-divisible `chunk_size`

Lines 621-623:

```
if pad_N:
    rch_flat = jnp.pad(rch_flat, ((0, pad_N), (0, 0)))
    acc_flat = jnp.pad(acc_flat, ((0, pad_N), (0, 0)))
```

When the user passes a `chunk_size` that doesn't exactly divide `N`,
`pad_N > 0` and `jnp.pad` returns a freshly-allocated buffer larger than
the donated `acc_`. XLA may still alias the storage in some cases, but
in general the original donation is no longer in-place after pad. A
temporary buffer of size `(N + pad_N) · ngkmax · 16 B` exists alongside
the donated buffer until the function returns.

This is a one-shot transient (not multiplied by `n_chunks`), so it does
not violate the slot-pileup principle. It does mean callers should
prefer `chunk_size` values that exactly divide `N` (or accept the
small overhead).

A defensive improvement would be to clip the final write to the live
range (`acc_flat[:N]`) without the up-front pad — e.g. pass `N` into the
scan and conditionally write only `min(cs, N - i0)` rows per iter via a
mask. This is a refinement, not a fix to a current bug.

## 4. Found defects

**None at the level of the principle.** The function is the cleanest
reverse-direction kernel I can point Agents 1/2 at as a reference.

## 5. Recommendations for the parallel forward helper

Things Agents 1/2 **should emulate** from this function:

1. **Single shard_map over `('x','y')` with `check_rep=False`.** The forward
   helper's input contract (μ-flat sharded on `n_rmu_padded`) is the same;
   the output (pair-density acc) is also μ-flat sharded. No reason to
   change.
2. **`lax.scan` inside the body, not outside.** This is the principle.
   Carry only what must persist across iterations (the accumulator);
   keep every transient inside the body so XLA's scan allocator can
   alias it.
3. **Decode loop-invariant indices outside the scan body.** Lines 632-637
   pre-compute `rx_slab`/`ry_slab`/`rz_slab` once because `r0_` is
   loop-invariant within the scan. The forward helper's analogue: any
   per-q-only or per-r-slab-only tables (phase tables, slab indices)
   should be hoisted out of the body and gathered per-iter.
4. **Bake phase / sphere tables as closure constants at cache-key time.**
   Cheaper than passing them through the call site, and the hash-based
   cache key (line 586-594) keeps recompiles correct when the table
   changes.
5. **Pure `jnp.zeros`-then-`dynamic_update_slice` for fresh-per-iter
   buffers.** Lines 653-654. This is the idiom that XLA's scan pass
   recognises for slot aliasing. Avoid `lax.full` with carry-dependent
   shapes, avoid `jnp.empty` (not a thing in jax anyway), avoid
   donating the buffer into the body (donation only makes sense for
   carry).
6. **`donate_argnums` on the accumulator only.** Line 669. The
   `gflat_acc` analogue in the forward kernel is the pair-density
   accumulator; donate it for in-place update.

Things to **avoid carrying over** from this function:

1. **The `jnp.pad` on non-divisible `chunk_size`** (caveat 3.2). For the
   forward helper, prefer to either (a) enforce `chunk_size | N` at the
   planner level, or (b) use a masked partial write on the last
   iteration. Either is cleaner than `pad → scan → unpad`.
2. **Don't blindly mirror `check_rep=False` without verifying.** This
   function disables rep checking because the q-and-r axes are
   replicated by data contract and the scan body's outputs would
   otherwise fail the replication check. If the forward helper's
   output sharding differs (e.g., the pair-density accumulator is
   sharded on a different inner axis), re-evaluate.
3. **Don't copy the closure-bake pattern for tables that should be
   sharded.** `sphere_c` and `phx/phy/phz` are tiny and inherently
   q-indexed (no μ dimension), so replication is fine. If the forward
   helper has a table indexed by μ (e.g., a centroid coordinate table
   `(n_rmu_padded, 3)`), shard it on `('x','y')`, don't bake it as a
   closure constant.

Things Agents 1/2 should **specifically port over the same way**:

- The "decode flat-r → (rx,ry,rz) on the slab only" trick
  (lines 633-637) is exactly the right move for the forward helper too,
  if it has a slab analog. It avoids materialising the full
  `(n_q, nx, ny, nz)` Bloch phase box.
- The `q_row = (i0 + arange(cs)) // n_mu_local` clipped-to-`n_q-1`
  pattern (lines 642-643) plus the zero-padded data semantics ensures
  pad rows contribute zero — emulate this if your forward helper has
  the same flat-(q · μ) chunking.

## 6. Bottom line

`accumulate_rchunk_to_gflat` adheres to the zero-replicated-intermediates
principle. Use it as the reference. The two minor caveats above are worth
documenting in any commit message that says "modelled on
`accumulate_rchunk_to_gflat`", but they do not undermine the structural
pattern Agents 1 and 2 are mirroring.

Agent 3 audit done
