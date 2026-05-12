# Handoff: lorrax_A state for the padding-refactor agent — 2026-05-08

Branch: **`agent/v_q_bispinor`** on lorrax_A.  HEAD = `5a3b0af`.

## SlabIO already does logical/physical padding on both sides

`SlabIO.write_slab` / `read_slab` accept `shape=` (physical, in-memory)
and `valid_shape=` (logical, on-disk).  Both directions work; verified
empirically.

```python
# Write padded-in-memory → unpadded-on-disk
io.create_dataset('zeta', shape=(LOGICAL,), ...)
io.write_slab('zeta', A_padded, valid_shape=(LOGICAL,))
# disk has LOGICAL; padded tail of A_padded dropped on write.

# Read unpadded-on-disk → padded-in-memory
A_padded = io.read_slab('zeta',
                        shape=(PADDED,), valid_shape=(LOGICAL,),
                        partition_spec=P(...))
# returned array has PADDED shape, LOGICAL prefix from disk,
# trailing PADDED−LOGICAL zeros.  Divisibility check runs against
# PADDED, so a product spec like P(('x','y')) with PADDED = 672
# on a 4×4 mesh passes (672/16 = 42 ✓).
```

The on-disk extent is always LOGICAL — files round-trip across any
process count.  Existing tests: `tests/test_slab_io_ffi_contract.py`
and `src/common/phdf5_padded_slab_test.py`.

A 5-line adapter to take a `pad_meta` object directly (instead of
`shape=` + `valid_shape=`) is trivial if you want the API tighter.

## Don't touch (real fixes, not over-corrections)

In `_FfiBackend`: the `Path → str` cast in `SlabIO.__init__`, the
`_drain_pending()` call at the top of `create_dataset` and
`_ds_id(readonly=True)` (collective HDF5 ops were racing with the
async writer thread → MPI_File_set_view trap), and the h5py
shape/dtype introspect in `read_slab` when `shape=` is omitted.
`runtime/padding.py::pad_shape_to_mesh` is the shape arithmetic;
build the in-memory array helpers on top.

## Gotchas

* **`meta.n_rmu_jax` is set but rarely consulted in kernels.**  ζ-fit
  uses `meta.n_rmu` (logical) throughout; works because every
  sharding on the μ axis there is single-mesh-axis (668/4 = 167).
  The product-divisibility constraint only bites at the V_q tile
  read.  Renaming to `n_rmu_pad` or dropping in favour of on-demand
  computation is your call.

* **CCT singularity hazard.**  If you pad ψ with zeros at the ζ-fit
  boundary, CCT acquires zero rows/cols at the padded block →
  `jnp.linalg.solve` (the `vertex_mu_L ≠ 0` LU branch) propagates
  NaN.  ζ-fit doesn't need padding (single-axis sharding is fine);
  don't pad it.  If you must, ridge-regularise the padded diagonal
  first.

* **V_q tile kernel** reads ζ at `P(None, None, ('x','y'))` — product
  sharding is the boundary that fails on undivisible n_rmu.  Your
  read-side helper lands a padded array here.

* **Pre-existing post-V_q failures unrelated to padding:**
  `do_screened=true` → W solver writes W0_qmunu rank-8 (legacy 8-D)
  into a rank-3 placeholder; `x_only=true` → head resolver finds no
  eps source.  Both surface with `bispinor=false` too — ignore them
  while testing your padding work.

## Smoke test bed

`runs/MoS2/00_mos2_3x3_cohsex/A_bispinor_smoke_2026-05-08/cohsex.in`
uses `centroids_frac_656_current.txt` (hand-truncated from 668 to a
mesh-divisible 656).  Switch to `centroids_frac_668_current.txt` to
reproduce the canonical failure your helpers should fix.

Correctness signals on 656 baseline: V_qmunu_TT diagonal traces
~10¹² (three distinct values, MoS2 hexagonal), TT off-diagonal
~10¹⁰, CC ~10⁸.  Hold as regression target.
