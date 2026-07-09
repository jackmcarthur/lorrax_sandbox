# Group: src/ffi/phdf5/ — parallel-HDF5 FFI (Python side)

Deep-read notes for the GW refactor map, 2026-07-01. Repo root:
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. Grep scope for callers:
`src/`, `tests/`, `tools/`, `scripts/` (plus a whole-repo sweep for the slab wrappers).

Package contents: `__init__.py` (31 loc), `context.py` (113 loc), `write.py` (136 loc),
`read.py` (438 loc). Sibling non-Python assets: `cpp/{context.cc, read_ffi.cc,
write_ffi.cc, ctx.h, phdf5_interface.h}` (the actual MPI-IO/H5D handlers),
`ARCHITECTURE.md` (design doc — two-layer async write, pooled d2h events, CPU sibling
`file_io/_slab_io_mpi_host.py`), `scripts/{stage_cray.sh, stage_openmpi.sh}`.

Layering picture (important for the refactor):

```
gw_jax / isdf / wfn users
   │
   ├── file_io/slab_io.py        (public N-D SlabIO; dispatches on use_ffi_io)
   │       └── file_io/_slab_io_ffi.py   ── uses ffi_read_call / ffi_write_call
   │                                        + open_file / close_file
   ├── file_io/wfn_loader.py (phdf5 backend) ── uses open_file/close_file
   │                                        + read_kchunk_union_sharded
   └── ffi/phdf5/  (THIS PACKAGE: thin jax.ffi.ffi_call wrappers + file lifecycle)
           └── ffi/common/ffi_loader.py  (ctypes: lrx_phdf5_open/close/
                                          ensure_dataset/open_dataset_ro/init_mpi)
                   └── liblorrax .so  (cpp/: H5Fopen w/ MPI-IO plists, H5Dread/H5Dwrite)
```

No LorraxConfig flags or cohsex.in keys are consumed anywhere in this package.
Backend selection (`use_ffi_io`) lives in `file_io/slab_io.py`, not here.

---

## src/ffi/phdf5/__init__.py (31 loc)

Re-exports the advertised public API: `open_file`, `close_file`,
`write_sharded_slab`, `read_sharded_slab`. Docstring shows a usage example.

**Staleness**: the two slab functions it advertises (`write_sharded_slab`,
`read_sharded_slab`) have **zero external callers** (see dead suspects below); actual
consumers import `open_file`/`close_file` from the package and the low-level
`ffi_read_call`/`ffi_write_call`/`read_kchunk_union_sharded` directly from the
submodules. The `__init__` docstring documents an API surface nobody uses.

---

## src/ffi/phdf5/context.py (113 loc)

Python-side file lifecycle for the FFI. Collective open/close of a parallel-HDF5
file; the handle is the int64 address of a C++ `PhdfCtx` struct. Per-process
path→handle cache with a threading.Lock; atexit sweeper closes leaked handles.

| Function | Lines | Role |
|---|---|---|
| `validate_mesh_2d(mesh)` | 29–33 | Assert mesh axes are exactly ('x','y'); return (p,q). Callers: `write.write_sharded_slab`, `read.read_sharded_slab` (both intra-package only). Exported in `__all__` but never imported outside the package (grep across src/tests/tools/scripts: 0 external hits). |
| `_MODE_FLAGS` | 36 | `{"w":0, "a":1, "r":2}` — must match the C++ enum in `context.cc` / `ffi_loader.phdf5_open` docstring (0=truncate, 1=append-or-create, 2=read-only). Magic-constant coupling across the ctypes boundary. |
| `open_file(path, *, mesh, mode="w")` | 39–77 | Collective H5Fcreate/H5Fopen via `ffi_loader.phdf5_open(path, p, q, rank, world, mode_flag)`. Validates `p*q == jax.process_count()`. Caches per path. Callers: `file_io/wfn_loader.py:286` (probe import), `:624` (`self._phdf5_ctx = open_file(self._path, mesh=self._mesh, mode="r")`), `file_io/_slab_io_ffi.py:337,362`. |
| `close_file(path_or_handle)` | 80–94 | Collective close. Accepts path or int handle; for an int handle it linearly scans `_FILE_CTXS` to evict matching path entries. Callers: `file_io/wfn_loader.py:249–250`, `file_io/_slab_io_ffi.py:766`. |
| `_atexit_close_all()` | 97–110 | atexit hook: best-effort `phdf5_close` on all cached handles, exceptions swallowed. |

**Weird / notable**
- **`open_file` cache ignores `mode` on hit** (lines 68–70): `open_file(p, mode="w")`
  followed by `open_file(p, mode="r")` silently returns the *write-mode* handle.
  No mode check on the cache hit path. Hypothesis: fine in practice because each
  file is opened once per process life, but a latent trap for a refactor that
  interleaves write-then-read of the same path.
- The int64 handle is a raw C++ struct address round-tripped through Python ints and
  baked into jitted closures as a compile-time attr (see read.py cache note below).
- atexit close runs "on every process" but MPI may already be finalized at
  interpreter exit; the bare `except Exception: pass` hides that.

**I/O**: opens/creates arbitrary `.h5` files on shared FS (Lustre/GPFS/NFS) with
MPI-IO parallel property lists. No dataset knowledge at this layer.

---

## src/ffi/phdf5/write.py (136 loc)

Sharded-slab FFI writer: thin `shard_map` wrapper around the XLA custom-call target
`"lorrax_phdf5_write"` (C++ `write_ffi.cc`). Each rank writes its local shard to an
HDF5 hyperslab via collective MPI-IO — no gather through rank 0. Docstring itself
says the preferred public entry point is `file_io.slab_io`.

| Function | Lines | Role |
|---|---|---|
| `ffi_write_call(A_local, offset_base, valid_shape, *, ctx_handle, ds_id, mesh_shape, axis_count_per_dim, axis_flat)` | 34–70 | Low-level FFI call for one rank's shard, for use *inside* a shard_map body. Returns a `(1,) int32` token; `jax.ffi.ffi_call("lorrax_phdf5_write", token_spec, has_side_effect=True)`. `offset_base`/`valid_shape` are `(ndim,) int64` traced **buffers** (not Attrs) so one compile serves all chunk offsets — comment cites a measured ~400 ms XLA recompile per chunk otherwise (reports/zeta_offset_runtime_2026-04-19/). `ctx_handle`, `ds_id`, `mesh_shape`, `axis_count_per_dim`, `axis_flat` are compile-time Attrs. Caller: `file_io/_slab_io_ffi.py:311–314` (SlabIO FFI backend N-D write path). |
| `write_sharded_slab(fh, ds_name, A, *, mesh, global_shape=None, valid_shape=None)` | 73–136 | 2-D convenience wrapper: `phdf5_ensure_dataset(fh, ds_name, (n_rows,n_cols), dtype)` then `shard_map(_per_rank, in_specs=(P('x','y'), P(), P()), out_specs=P(), check_rep=False)` with `mesh_shape=(p,q)`, `axis_count_per_dim=(1,1)`, `axis_flat=(0,1)`. Raises NotImplementedError for ndim != 2. **Zero external callers** (grep whole repo: only defs/docstrings inside this package). |

**Padding contract** (comment at lines 32–33): `A_local` is the physical equal-block
shard (padded so global physical shape divides (p,q)); `valid_shape` is the logical
global slab prefix the C++ clips against; `global_shape` is the full on-disk dataset
extent. Three nested shapes validated at lines 102–112.

**Boundary arrays**: `A_local` device→(cudaMemcpyAsync D2H inside handler, per
ARCHITECTURE.md); output token `(1,) int32` replicated. Datatypes limited to
`ffi_loader._DTYPE_TAG` = {f32, f64, i32, i64, c64, c128}.

**Weird / notable**
- Underlying C++ handler is N-D, but this wrapper hard-blocks N-D
  (`NotImplementedError` line 90) and punts to `file_io.slab_io` — evidence the
  wrapper is a vestigial first-generation entry point.
- `check_rep=False` everywhere (standard for FFI side-effect ops, but means no
  replication checking).

---

## src/ffi/phdf5/read.py (438 loc)

Sharded-slab FFI readers. Three C++ handlers (`cpp/read_ffi.cc`), three low-level
call wrappers, three high-level shard_map builders. Key design: per-call offsets and
counts are runtime int64 buffers so a single compiled module re-dispatches across
chunks (same anti-recompile trick as write.py).

FFI targets: `"lorrax_phdf5_read"` (one H5Dread of one rectangle),
`"lorrax_phdf5_read_kchunk"` (n_kchunk sequential H5Dreads of same-shape windows,
one handler invocation), `"lorrax_phdf5_read_kchunk_union"` (ONE H5Dread of n_kchunk
disjoint windows via `H5S_SELECT_OR` compound hyperslab).

| Function | Lines | Role |
|---|---|---|
| `_encode_sharding_axes(mesh, file_partition_spec, ndim_file)` | 67–78 | Returns `(axis_count_per_dim, axis_flat)` for the C++ handler by delegating to `file_io._slab_io_ffi._sharding_to_axis_info` (lazy import; comment: "exactly one source of truth"). **Upward layering dependency**: low-level ffi package imports from higher-level file_io. |
| `_insert_at(seq, axis, value)` | 81–85 | Tuple insert helper. |
| `_register_and_open_dataset(fh, ds_name)` | 88–90 | `ffi_loader.phdf5_open_dataset_ro` (collective H5Dopen), returns ds_id (hid_t as int64). |
| `ffi_read_call(out_struct, offset_base, valid_shape, *, ctx_handle, ds_id, mesh_shape, axis_count_per_dim, axis_flat)` | 98–125 | Single-hyperslab read. offset/valid_shape are `(ndim,) int64` runtime buffers. Caller: `file_io/_slab_io_ffi.py:286–289` (SlabIO read path). Note: no `has_side_effect` (reads treated as pure). |
| `ffi_read_kchunk_call(out_struct, offset_base, *, ..., n_kchunk)` | 128–151 | Sequential-reads kchunk; `offset_base` is `(n_kchunk, ndim_file) int64`; per-k window shapes identical, only file offset varies. Callers: only `read_kchunk_sharded` (below) — **transitively dead**. |
| `ffi_read_kchunk_union_call(out_struct, offset_base, count_base, *, ..., n_kchunk, kchunk_axis)` | 154–184 | Compound-hyperslab kchunk; `offset_base` and `count_base` both `(n_kchunk, ndim_file) int64`; `kchunk_axis` Attr tells C++ where the k axis sits in the output so memspace row-major iteration matches filespace. Caller: `_read_kchunk_union_sharded_cached`. |
| `read_sharded_slab(fh, ds_name, *, global_shape, dtype, mesh)` | 190–229 | 2-D whole-dataset read → `P('x','y')`-sharded array (returns array, not closure). Requires global shape divisible by (p,q). **Zero external callers** (grep whole repo). |
| `read_kchunk_sharded(fh, ds_name, *, n_kchunk, file_global_shape, per_rank_file_shape, dtype, mesh, file_partition_spec)` | 232–306 | Builds jitted `f(offset_base) → (n_kchunk, *per_rank_file_shape)` closure; n_kchunk sequential H5Dreads for possibly-overlapping same-shape windows (ngkmax slabs at variable-ngk WFN files). Leading n_kchunk axis replicated; file dims sharded per spec. **Zero external callers** (grep whole repo) — superseded by the union variant. |
| `read_kchunk_union_sharded(fh, ds_name, *, n_kchunk, kchunk_axis, file_global_shape, per_rank_file_shape, dtype, mesh, file_partition_spec, count_partition_spec=None)` | 309–345 | Public wrapper: normalises args to hashable tuples, defaults `count_partition_spec=P()` (replicated counts) or accepts `P(('x','y'), None)` for per-rank clamped counts (mnband not divisible by world — see `tests/test_wfn_loader_phdf5_clamp.py`, `wfn_loader._build_phdf5_clamped_counts`). Caller: `file_io/wfn_loader.py:663,755` (`WfnLoader._phdf5_build`, reading `"wfns/coeffs"` with `file_global_shape=(nbands, ns, ngktot, 2)`, `per_rank_file_shape=(bands_per_rank, ns, ngkmax, 2)`, `kchunk_axis=2`, `file_partition_spec=P(('x','y'), None, None, None)`, `count_partition_spec=P(('x','y'), None)`). |
| `_read_kchunk_union_sharded_cached(...)` | 348–438 | `@functools.lru_cache(maxsize=None)` impl. Builds jitted `f(offset_base, count_base)` shard_map closure issuing ONE H5Dread via `H5S_SELECT_OR`. Correctness preconditions documented but **not runtime-checked**: (1) per-k rectangles pairwise disjoint, (2) sorted ascending in row-major file order (caller must argsort offsets and permute back — wfn_loader does this via `np.unique` + `searchsorted`). Output shape = per_rank_file_shape with n_kchunk inserted at kchunk_axis; partition spec gets `None` inserted there. `check_rep=False`. |

**Boundary arrays** (union path, the hot one — WFN load):
- in: `offset_base (n_kchunk, 4) int64` replicated; `count_base` either replicated
  or `(world*n_kchunk, 4)` global sharded `P(('x','y'), None)` → per-rank
  `(n_kchunk, 4)` slice.
- out (per rank): `(bands_per_rank, ns, n_reads, ngkmax, 2) f64` sharded
  `P(('x','y'), None, None, None, None)` — re/im packed pairs, unpacked to c128
  downstream (`common/gvec_fft_box.py:83` docstring references this format).

**I/O**: reads arbitrary HDF5 datasets by name through the open PhdfCtx. Concrete
datasets in current use: `wfns/coeffs` in BGW-style `WFN.h5` (via wfn_loader);
SlabIO-managed datasets (`zeta_q_G (n_q_disk, n_rmu, ngkmax)`, `V_qmunu`, etc.) go
through `file_io/_slab_io_ffi.py`, which uses only the low-level `ffi_read_call`/
`ffi_write_call` from this package, not the high-level wrappers.

**Weird / notable**
- **lru_cache keyed on the raw int handle** (`fh`) with `maxsize=None`
  (line 348): if a file is closed and another `PhdfCtx` is later malloc'd at the
  same address (or the same path is re-opened giving a new ds_id), a stale cached
  closure with a baked-in old `ds_id` compile-time Attr would be silently reused.
  No invalidation hook from `close_file`. Also holds `Mesh` objects alive forever.
- **Upward import** `from file_io._slab_io_ffi import _sharding_to_axis_info`
  (line 76) — the "single source of truth" for the sharding→axis encoding lives in
  the *higher* layer; refactor should move the encoder down into ffi/phdf5 (or a
  shared module) rather than keep the inversion.
- Reads have no `has_side_effect=True` and no token ordering vs. writes;
  read-after-write ordering relies entirely on caller-side Python sequencing plus
  the write Future semantics in the C++ layer (documented in ARCHITECTURE.md).
- Docstring-only correctness preconditions on the union read (disjointness,
  ascending order) — a wrong `count_base` gives silently garbled data, not an error.

---

## Dead suspects (grep evidence)

Whole-repo grep
`grep -rn --include='*.py' -E 'read_sharded_slab|write_sharded_slab|read_kchunk_sharded\b|ffi_read_kchunk_call\b' .`
returns hits **only inside `src/ffi/phdf5/`** (defs, `__all__`, docstrings). Also
grepped `src tests tools scripts` for every `__all__` name individually.

1. `read.read_sharded_slab` — zero callers anywhere (incl. tests/active, tests/archive).
2. `write.write_sharded_slab` — zero callers; docstring itself defers to slab_io.
3. `read.read_kchunk_sharded` — zero callers; superseded by the union variant.
4. `read.ffi_read_kchunk_call` + C++ target `lorrax_phdf5_read_kchunk` — only caller
   is dead (3), so transitively dead down through `cpp/read_ffi.cc`'s kchunk handler.
5. `context.validate_mesh_2d` in `__all__` — used only intra-package; export is stale.

## Redundancy suspects

1. `read_sharded_slab`/`write_sharded_slab` vs `file_io/slab_io` +
   `_slab_io_ffi`: classic parallel old/new path. slab_io builds its own shard_map
   closures around the same `ffi_read_call`/`ffi_write_call` primitives; the
   package-level 2-D wrappers are the abandoned first-gen API still advertised by
   `__init__.py`.
2. `read_kchunk_sharded` vs `read_kchunk_union_sharded`: same job (n_kchunk windows
   of an N-D dataset), sequential-reads vs compound-hyperslab implementations; only
   union survives in callers. The overlap-tolerant sequential variant may still be
   *conceptually* needed for overlapping windows (union requires disjointness), but
   no live code exercises it.

## Refactor deletions on offer

Deleting items 1–4 above (and the matching C++ kchunk handler) leaves the package
as: `context.py` lifecycle + three live primitives (`ffi_read_call`,
`ffi_write_call`, `ffi_read_kchunk_union_call` + its cached builder), with
`__init__.py` re-exporting what callers actually use. Moving
`_sharding_to_axis_info` down from `file_io/_slab_io_ffi` fixes the layering
inversion at the same time.
