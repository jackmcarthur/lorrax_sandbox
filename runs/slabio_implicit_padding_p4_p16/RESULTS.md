# Implicit padding in SlabIO — 2026-08-04

Implements `docs/architecture/decisions.md` 2026-08-04, "Padding is SlabIO's
business, not the caller's".  Branch `fix/slab-io-audit`, worktree
`/work2/08271/jackmc/frontera/wt-slabio`, on top of the 2026-08-04 audit
(8050e34, dccb02e).  Artifacts under `/scratch2/08271/jackmc/slabio_padding/`.

Commits: **61500ac** (decisions.md, file-only pick of dd831fb), **98afe58**
(the change), **94a0a75** (divisibility measured; read-path regression fixed),
**ca9bad5b** (gate correction).

Jobs: **7888641** build, **7888644** first gate (caught two defects in the
change), **7888650** P=4, **7888651** P=16 + pytest, **7888657**/**7888660**
pytest and its A/B.

`.so`: `/work2/08271/jackmc/frontera/lorrax_ffi_unified/build_host_IMPLICITPAD/
liblorrax_ffi_host.so`, sha256 `51c69680d078bf85`, 475 exported symbols,
exported-symbol diff vs `build_host_SLABIO` EMPTY.  `CANONICAL.md` repointed.

## 1. The contract now

A caller states LOGICAL shapes and nothing else:

    io.create_dataset("A", shape=(n_q, n_mu, n_G), dtype=c128)
    io.write_slab("A", A_padded)
    B = io.read_slab("A", shape=(n_q, n_mu_pad, n_G), dtype=c128,
                     mesh=mesh, partition_spec=spec)

The extent that reaches the file is derived once, in
`_slab_io_ffi._derive_valid_shape`, as `min(operand, dataset - offset)` per
dim.  That is exactly the arithmetic each call site used to do by hand.

The derivation needs the dataset's extent to be a REPLICATED quantity, which
is what `_DatasetGeometry` records: every entry comes from the shape passed
to `create_dataset` (SPMD by contract), the `global_shape` an auto-creating
`write_slab` used, or a metadata read every rank performs.  The record is
authoritative only because `ensure_dataset` now REFUSES a shape/dtype change
— without that, `H5Dopen` of a differently-shaped dataset would leave the
record describing geometry the file does not have.

`valid_shape` survives as the ragged-chunk OVERRIDE: a chunk buffer whose
tail is genuinely not part of this write, which SlabIO cannot derive because
both extents are legitimate.  An override that overruns the dataset refuses,
on every rank, in Python, before any collective is entered.  `global_shape`
is create-only and refuses when it contradicts an existing dataset.

## 2. The five items the audit left for the owner

| # | Item | Resolution |
|---|---|---|
| 1 | `ensure_dataset` shape/dtype check | ADDED (`context.cc check_existing_geometry`), on both the cached-hid and freshly-opened paths.  Refuses naming both shapes.  It does newly refuse a `mode='a'` rerun at a different mu. |
| 2 | `create_dataset`-on-existing backend divergence | UNIFIED on reuse-if-identical / refuse-otherwise, one shared `_refuse_geometry_change`.  The allgather delete-and-recreate is gone.  Its verdict is computed on rank 0 and BROADCAST, because only rank 0 holds a handle. |
| 3 | `_raise_if_error` clears the error, `close()` skips `H5Fclose` | FIXED.  `_FfiBackend.close()` records, announces, completes `H5Fclose` + the close barrier, then re-raises.  `AsyncDispatcher.close()` joins the worker in a `finally`.  `_MpiHostBackend.close()` closes the file in a `finally` before raising its divergence. |
| 4 | `ctx->pinned_buf` race | DOCUMENTED PRECISELY in `ctx.h`: the two producers, what orders them and what does not, the SlabIO-level drain that closes the window for every SlabIO caller, the direct-`ffi.io` case it does not close, and a prohibition on adding another sync entry point without the redesign.  No C++ redesign. |
| 5 | Host backend's missing divisibility check | MEASURED, and the answer inverted the plan — see §4. |

## 3. Certification

`tests/multi_device/phdf5_padded_rank_write.py` gains `implicit` and `nodiv`,
and the five accept-path cases now issue the IMPLICIT write, so every one of
them exercises the default.

P=4 (job 7888650) and P=16 (job 7888651), one case per process launch, each
under an external timeout, per-rank stdout AND stderr to separate files:

| case | P=4 | P=16 |
|---|---|---|
| repro / control / exact / read_pad | rc=0 on 4/4 | rc=0 on 16/16 |
| implicit | rc=0 on 4/4 | rc=0 on 16/16 |
| nodiv | rc=0, 4/4 refuse by name | rc=0, 16/16 refuse by name |
| oob_write / oob_read | rc=0, 4/4 refuse | rc=0, 16/16 refuse |
| probe rt / raw / oob_write / oob_read | rc=0 on 4/4 | rc=0 on 16/16 |

### 3.1 The 3x3 writer x reader matrix — written and read with NO padding argument

`slabio_probe.py case=rt`, all three backends, two request shapes each
(logical/replicated and PADDED/mu-sharded), plus a serial-h5py readback:
**18 of 18 checks at `0.000e+00`** at P=4 (mu=5, mu_padded=8, 1 wholly-padded
rank) and at P=16 (mu=17, mu_padded=32, 7 wholly-padded ranks).  `replica_dup`
and the fill-time contract both PASS.

### 3.2 BYTE-IDENTITY — the gate that proves the default is right

The same padded buffer, same dataset, two spellings:

* explicit: `create_dataset(shape=logical)` + `write_slab(A, offset=0,
  global_shape=logical, valid_shape=logical)`
* implicit: `create_dataset(shape=logical)` + `write_slab(A)`

| P | mu / mu_padded | explicit sha256 | implicit sha256 | identical |
|---|---|---|---|---|
| 4 | 5 / 8 | `c12e54ab4a217c1a` | `c12e54ab4a217c1a` | YES |
| 16 | 17 / 32 | `33af697ff8fc5d6c` | `33af697ff8fc5d6c` | YES |

The READ side too: `read_slab(shape=padded, valid_shape=logical)` and
`read_slab(shape=padded)` return byte-identical per-rank blocks, on every
rank, and both match the padded reference.

Unit twins in `tests/test_file_io.py`
(`test_allgather_implicit_pad_matches_explicit_valid_shape`,
`test_create_dataset_refuses_a_geometry_change`, four
`_normalize_valid_shape` derivation cases): 39 passed (job 7888651).

## 4. What the measurement changed

**Divisibility does not need SlabIO to pad, because JAX will not let the
violating case exist.**  98afe58 implemented an internal pad-and-trim on both
directions, on the reading that "SlabIO pads internally if the backend needs
it".  The `nodiv` gate case then measured (jax 0.9.1) that `jax.device_put` of
a `(shape, spec)` pair whose sharded dim does not divide its mesh-axis product
raises `IndivisibleError`.  So:

* SlabIO is never HANDED a non-divisibly-sharded operand on the write path —
  there is nothing to pad;
* it could not RETURN one on the read path either, so reading a rounded-up
  extent and trimming only replaces SlabIO's message with JAX's, at the same
  refusal.

Both padding paths were deleted (94a0a75).  What remains is
`_validate_block_divisible`, restated as what it actually is: an early,
numbered restatement of JAX's own constraint that names the rounded-up extent
to ask for instead.  This also settles item 5 — `_MpiHostBackend`'s absent
check is not a capability difference, because the case it would accept cannot
be expressed.  Refusal text, verbatim, on 16 of 16 ranks:

    read_slab 'zeta_like': dimension 1 size 17 is not divisible by its
    mesh-axis product 16, so the array you asked for cannot be sharded this
    way — JAX itself refuses to build it (IndivisibleError).  Ask for
    dimension 1 at 32 instead: SlabIO fills what the dataset covers and
    zeroes the rest, and you state nothing else.

## 5. Two defects the gate caught in the change itself

Both from job 7888644, both fixed before the certifying run:

1. `_FfiBackend.read_slab` had been made to introspect the dataset with serial
   h5py unconditionally.  On a handle still open for collective MPI-IO
   writing the superblock is not durable and h5py fails with
   `OSError: file signature not found` — probe/raw, rc=1 on 4 of 4 ranks.
   `_dataset_geom` now answers from the handle's own record and only
   introspects for a dataset the handle did not create.
2. The `oob_write` gate case passed a `global_shape` that contradicts the
   dataset, so it refused for THAT reason and no longer reached the
   valid_shape-overrun bounds test it exists to gate.  Argument dropped; the
   refusal is now `valid slab exceeds dataset extent (dim 1: 1+17>17)`.

## 6. Call-site sweep

| file | `valid_shape=` | `global_shape=` | other |
|---|---|---|---|
| `file_io/zeta_loader.py` | 6 | — | `valid_mu` removed from 5 signatures |
| `file_io/tagged_arrays.py` | 2 | 2 | |
| `file_io/sigma_output.py` | — | 4 | |
| `gw/isdf_fitting.py` | 1 | 1 | |
| `gw/v_q_bispinor.py` | 6 | 3 | 3 dead `*_padded_shape` locals |
| `gw/v_q_g_flat.py` | — | — | `valid_mu=` at its one consumer |
| **total** | **15** | **10** | |

Nothing changes meaning: every removed argument restated an extent SlabIO now
derives, and §3.2 is the proof at the byte level.

## 7. Not covered by the ruling — decided here, flagged for the owner

1. **The bounds refusal moved from C++ to Python.**  An explicit `valid_shape`
   that overruns is now caught by `_normalize_valid_shape` against the
   dataset's own extent, before any collective is entered — strictly earlier
   and strictly safer than the C++ test, and still rank-invariant.  The C++
   tests in `write_ffi.cc` / `read_ffi.cc` are unchanged and remain the
   backstop for direct `ffi.io` users; they are no longer reachable through
   `SlabIO`.  Their certification (job 7888525) still applies to this `.so`
   because those two files are byte-identical to `build_host_SLABIO`.
2. **The DEFAULT clips; only an OVERRIDE refuses.**  `write_slab(A)` where `A`
   overhangs the dataset writes the prefix and drops the rest, because that is
   indistinguishable from the pad-row case the ruling is about.  A caller who
   means "all of this must land" states `valid_shape` and gets a refusal.
3. **`global_shape` became create-only.**  The ruling does not mention it, but
   leaving it as a second, caller-supplied extent alongside the dataset's own
   would re-open the same hazard.  It now refuses when it contradicts the
   dataset.
4. **`valid_mu` deleted from `ZetaLoader`.**  It was the loader's own
   restatement of the header μ count.  For a legacy G-flat file whose on-disk
   μ extent EXCEEDS the header count, the derived clip reads those rows from
   disk instead of zero-filling them in the reader.  They are exact zeros by
   writer construction (L_q's pad block is identity), so the values are
   identical; the shapes are unchanged.  Flagged because it is the one place
   where "what the file says" and "what the header says" can differ.
5. **Restart-path interaction, for the agent on `gw/gw_init.py`.**
   `create_dataset` / `ensure_dataset` now refuse a shape or dtype change on
   an existing dataset instead of writing into the old geometry.  The normal
   flow is unaffected — `isdf_fitting` replaces the ζ inode unconditionally
   before the fit, and `write_restart_state_to_h5(mode='w')` replaces the
   tensors inode — but any restart path that REUSES an existing file at a
   different mu will now refuse by name where it previously wrote a prefix.
   That is the behaviour change the ruling explicitly accepts.

## 8. Known-not-a-regression

`tests/test_restart_pad_roundtrip.py` fails (2 tests) when it shares a pytest
process with `test_sanity_gates_jax.py` or `test_contract_bands.py`, which set
`XLA_FLAGS=--xla_force_host_platform_device_count=4` at import time.  A/B on
the SAME combined command, job 7888660: 2 failed on both `ca9bad5b` (136
passed) and base `dccb02e` (130 passed); the file passes ALONE on both.
Registered in `KNOWN_LORRAX_ISSUES.md` as a harness trap.
