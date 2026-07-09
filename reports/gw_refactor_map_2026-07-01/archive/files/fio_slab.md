# File group: SlabIO (sharded-slab HDF5 I/O)

Files: `src/file_io/slab_io.py` (374 loc), `src/file_io/_slab_io_ffi.py` (791 loc),
`src/file_io/_slab_io_mpi_host.py` (421 loc), `src/file_io/_slab_io_allgather.py` (305 loc).

Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. Deep-read 2026-07-01.

Architecture: `slab_io.SlabIO` is a facade over three interchangeable backends
selected by `gw.gw_config.SlabIOBackend`:

| enum | module | mechanism | platform |
|---|---|---|---|
| `H5PY_ALLGATHER` | `_slab_io_allgather` | process_allgather → rank-0 serial h5py | any (default/fallback) |
| `PHDF5_FFI` | `_slab_io_ffi` | collective MPI-IO via `ffi.phdf5` C++ FFI (cudaMemcpy D2H) | GPU |
| `PHDF5_HOST` | `_slab_io_mpi_host` | parallel h5py + mpi4py, per-rank hyperslab | CPU |

The auto-router lives in `gw_config.LorraxConfig.from_input_file` (cohsex.in key
`use_ffi_io=true` → PHDF5_FFI on GPU, PHDF5_HOST on CPU with mpi4py+h5py-parallel,
H5PY_ALLGATHER otherwise; see gw_config.py:976-1034).

Common contract across all backends: N-D dataset writes/reads at `offset` with
`global_shape` (dataset extent) and `valid_shape` (logical prefix inside a
physical slab padded for even mesh sharding; padded tail never touches disk and
is zero-filled on read).

---

## src/file_io/slab_io.py (374 loc)

**Purpose.** Public facade: `SlabIO` context manager + single-shot free functions
`write_slab` / `read_slab` / `accumulate_slab`. Replaces the historical ad-hoc
`process_allgather` → rank-0 h5py patterns. No physics; pure dispatch.

**Category.** I/O: unified sharded-array HDF5 facade.

### Functions

| name | lines | role |
|---|---|---|
| `_normalize_slab_backend(backend, use_ffi_io)` | 61-89 | Resolve `(backend enum \| legacy use_ffi_io bool \| None)` → `SlabIOBackend`. Legacy `use_ffi_io=True` maps to PHDF5_FFI ("Anyone still calling SlabIO directly with use_ffi_io=True ... intends the GPU FFI"); `None`/False → H5PY_ALLGATHER. Imports `SlabIOBackend` from `gw.gw_config` inside the function "to avoid circular import at module load". |
| `SlabIO.__init__` | 124-163 | Coerces path to str (FFI C bindings `.encode()` fail on PosixPath), normalizes backend, requires `mesh` for both PHDF5 variants, lazily imports the chosen backend class. Sets derived bool `self.use_ffi_io = backend in (PHDF5_FFI, PHDF5_HOST)` used only for `read_slab` branch dispatch. |
| `SlabIO.__enter__/__exit__/close` | 166-173 | Context manager; close delegates to backend. |
| `SlabIO.create_dataset(name, shape, dtype, chunks, attrs)` | 176-196 | Pre-create dataset. Docstring documents the padding policy and points at `runtime.padding` "agent/padding-refactor branch". |
| `SlabIO.write_attr(name, value)` | 198-204 | Small rank-0-only dataset (e.g. `omega_ev`); skips allgather. |
| `SlabIO.write_slab(name, A, offset, global_shape, valid_shape, dtype, chunks, k_chunk_size)` | 209-243 | Hyperslab write; sharding inferred from `A.sharding` on FFI path; `k_chunk_size` is an allgather-only streaming knob (legacy sigma_output k_chunk pattern), ignored by FFI. |
| `SlabIO.read_slab(name, shape, dtype, offset, valid_shape, mesh, partition_spec, as_numpy)` | 248-288 | Hyperslab read. FFI/host branch returns sharded array with `partition_spec` on `mesh`; allgather branch returns replicated array or plain ndarray (`as_numpy=True` fast path skipping H2D+D2H). Final `as_numpy` coercion at 286-288 via `jax.device_get`. |
| `SlabIO.accumulate_slab(name, A, offset)` | 290-305 | `dset[off:off+A.shape] += A` RMW. Docstring: "Used by the Σ_c(ω) stream-mode accumulator in ppm_sigma" — but see dead_suspects. |
| free `write_slab(path, ds_name, A, ...)` | 311-336 | open+create+write+close one-shot. |
| free `read_slab(path, ds_name, ...)` | 339-358 | open+read+close one-shot. |
| free `accumulate_slab(path, ds_name, A, ...)` | 361-374 | open+RMW+close one-shot. |

### Entry points / callers (grep over src, tests, tools, scripts)

`SlabIO` class (the only name actually imported anywhere):
- `src/gw/gw_init.py:919` (zeta write/read orchestration; four-channel bispinor ζ files at :1006-1064, plain ζ at :1071,1095)
- `src/gw/v_q_bispinor.py:319,528,775` (V_qmunu tile writer + `_VqReader`)
- `src/common/isdf_fitting.py:2291` (zeta_q.h5 writer in the ζ-fit loop)
- `src/file_io/tagged_arrays.py:46,108`; `src/file_io/zeta_loader.py:65`; `src/file_io/kin_ion.py:14`; `src/file_io/sigma_output.py:282`; `src/file_io/zeta_reader.py:58`
- `tests/test_slab_io_ffi_contract.py:9` (allgather-mode roundtrip at :98-106)

Free functions `write_slab`/`read_slab`/`accumulate_slab`: **zero callers** (see dead_suspects).
`file_io/__init__.py` does not re-export any slab_io name.

### Flags consumed
- `config.backend.slab_io` (`SlabIOBackend`, derived from cohsex.in `use_ffi_io`) — passed in by every production caller.
- Legacy kwarg `use_ffi_io: bool | None` — still threaded through `cohsex_sigma.py`, `v_q_bispinor.py`, `sigma_x_bispinor.py` signatures.

### I/O
Generic HDF5 (any dataset name). Actual files written through this facade by
callers: `zeta_q.h5` (+3 transverse ζ files), `V_qmunu.h5` tiles, sigma ω-grid
h5 (`sigma_output.write_sigma_omega_h5`), kin_ion h5, tagged arrays, restart state.

### Suspects
- **dead**: free functions `write_slab` (311), `read_slab` (339), `accumulate_slab` (361). Grepped `from file_io.slab_io import`, `slab_io.write_slab(|read_slab(|accumulate_slab(`, `import write_slab|read_slab|accumulate_slab` across src/tests/tools/scripts — only `SlabIO` is imported anywhere. Also `SlabIO.accumulate_slab` method: grep `\.accumulate_slab(` finds only slab_io-internal definitions/delegations; the claimed consumer `ppm_sigma` stream mode uses an in-memory `_accumulate_kij_stream` (ppm_sigma.py:1633) instead. Entire accumulate path (facade + all 3 backend impls) appears unreached.
- **weird**: `file_io` package imports from `gw.gw_config` (lines 72, 133) — inverted layering, dodged with function-local imports; a refactor should move `SlabIOBackend` down or out of `gw`.
- **weird**: dual selector state — `self.backend` enum plus derived `self.use_ffi_io` bool (lines 148-149) kept "for branch dispatch" in `read_slab`; two sources of truth for one decision.
- **redundancy**: legacy `use_ffi_io` bool accepted on all 4 public entry points in parallel with `backend` enum (documented back-compat shim, but it's the classic old/new parallel path).

---

## src/file_io/_slab_io_ffi.py (791 loc)

**Purpose.** GPU backend: each rank writes/reads its own hyperslab via the
`ffi.phdf5` C++ XLA-FFI (collective MPI-IO, cudaMemcpyAsync D2H inside C++).
Adds a Lustre prestripe, a module-level compiled-shard_map cache, and an async
single-worker write-dispatch thread so the Python main thread can build the next
chunk while H5Dwrite is in flight.

**Category.** I/O + distributed-linalg FFI glue: parallel-HDF5 write/read backend.

### Module-level helpers

| name | lines | role |
|---|---|---|
| `_lustre_prestripe(path, stripe_count=16, stripe_size="4M")` | 31-69 | Rank-0-only `lfs setstripe` pre-creation; Cray MPICH silently drops `striping_factor` hints on pscratch (dir default 1×1 MiB). Removes existing file first. Best-effort no-op if `lfs` missing; bare `except Exception: pass`. Measured: 32 MB/s/rank → expected ~500 MB/s/rank on Si 10³ zeta_q.h5. Reused by `_slab_io_mpi_host` and `isdf_fitting.py:2294`. |
| `_sharding_to_axis_info(sharding, ndim)` | 75-115 | Encode NamedSharding as `(axis_count_per_dim, axis_flat)` int tuples for FFI attrs; handles str/tuple/None spec entries; validates axis names. |
| `_replicated_sharding(mesh, ndim)` | 118-120 | All-None PartitionSpec helper. |
| `_replicated_i64_vector(values, mesh)` | 123-135 | Small int64 control buffer explicitly `device_put` replicated — comment records a real bug: an implicitly-placed offset buffer "arrived in C++ with dimensions permuted in the real CrI3 driver". |
| `_normalize_slab_request(op, name, offset, slab_shape, global_shape, check_bounds)` | 138-179 | Validate/default `(offset, slab_shape, global_shape)`; rank match, non-negativity, optional `off+shape<=gshape` bounds. Shared with both other backends and unit-tested in `tests/test_slab_io_ffi_contract.py`. |
| `_normalize_valid_shape(op, name, valid_shape, slab_shape, offset, global_shape)` | 184-232 | Validate logical prefix `valid_shape <= slab_shape`, and `off+vshape <= gshape` when gshape given. |
| `_validate_block_divisible(op, name, shape, axis_count_per_dim, axis_flat, mesh_shape)` | 235-258 | Reject sharded dims not divisible by the product of their mesh axis sizes (physical slab must equal-block shard; ragged handled by valid_shape clipping in C++). |
| `_get_read_sm(mesh, partition_spec, *, ds_id, ctx_handle, mesh_shape, axis_count_per_dim, axis_flat, out_struct)` | 280-301 | `functools.lru_cache(maxsize=None)` factory returning `jax.jit(shard_map(_per_rank))` around `ffi.phdf5.read.ffi_read_call`; `in_specs=(P(), P())`, `out_specs=partition_spec`, `check_rep=False`. Cache keyed on FFI signature so the closure `id()` is stable (vs functools.partial defeating JAX trace cache). |
| `_get_write_sm(mesh, in_specs, *, ds_id, ctx_handle, ..., no_jit)` | 304-326 | Same for `ffi.phdf5.write.ffi_write_call`; `in_specs=(in_specs, P(), P())`, `out_specs=P()`. `no_jit` (env `LORRAX_WRITE_NO_JIT=1`) skips jit — diagnostic for a jit-argument-retention buffer leak. |

### `_FfiBackend` (332-791)

| method | lines | role |
|---|---|---|
| `__init__(path, mesh, mode)` | 335-418 | Lazy imports `ffi.phdf5.open_file/close_file`, `ffi.common.ffi_loader`. mode='w': rank-0 prestripe (env `LORRAX_PHDF5_STRIPE_COUNT` default 16, `LORRAX_PHDF5_STRIPE_SIZE_FS` default "4M") + `sync_global_devices` barrier. Opens FFI file handle `self.fh`. Starts daemon writer thread on a `queue.Queue(maxsize=2)` — bound chosen from measurements (K=0 unbounded OOMs: 12.91→22.48+ GB; K=2 flat 16.47 GB; K=4 18.50 GB same throughput; Si 4×4×4 60Ry/2400c). One FIFO worker per backend = same dispatch order on all ranks (MPI collective rendezvous requirement); rationale: XLA `ffi::Future` still blocks the jit caller, so async must be Python-side (reports/session_2026-04-18_async_probe). |
| `create_dataset` | 421-455 | `_drain_pending()` first (collective `phdf5_ensure_dataset` H5Dcreate on the same MPI handle as in-flight H5Dwrite trips `MPI_File_set_view: Invalid datatype`), then registers `ds_id`. **`chunks`/`attrs` are silently a no-op** (warns): FFI has no collective set-chunks; caller must pre-create with h5py if chunking needed. |
| `write_attr` | 458-462 | Appends to `_deferred_attrs`; flushed at `close()` (concurrent serial h5py + MPI-IO would corrupt HDF5 metadata). |
| `_dispatch_loop` | 466-487 | Worker: pop task, run, stash first exception in `_dispatch_error`, decrement `_dispatch_pending` under condition var. `None` sentinel exits. |
| `_drain_pending` | 489-497 | Wait until pending==0, re-raise stashed error (error surfaces at next drain, not at enqueue). |
| `_introspect_dataset` | 500-523 | h5py metadata read of `(shape, dtype)`, cached in lazily-created `self._introspect_cache` (via getattr — created outside `__init__`). Relies on `HDF5_USE_FILE_LOCKING=FALSE` process-wide. |
| `_ds_id(name, readonly)` | 525-538 | Cached ds_id lookup; on miss drains then `phdf5_open_dataset_ro` (collective); raises if not readonly and unregistered. |
| `write_slab` | 543-623 | Coerce to jax.Array; if not on `self.mesh`, `device_put` replicated. Derive axis info from `A.sharding`; normalize request with `check_bounds=False` (bounds enforced against gshape inside `_normalize_valid_shape` instead); `_validate_block_divisible`. Auto-`phdf5_ensure_dataset` if unregistered (no drain here, unlike `create_dataset` — safe only because ensure of a *new* name?). Optional `LORRAX_FFI_DEBUG_SHARDS` stderr dump. Builds cached `sm = _get_write_sm(...)` (offset/valid_shape are RUNTIME replicated i64 vectors → one compile per (file, ds, sharding)), enqueues `_task = sm(A, offset_arr, valid_shape_arr).block_until_ready()` on the worker. Task closure holds `A` alive on GPU until written (the reason for maxsize=2). |
| `read_slab` | 628-705 | Synchronous. Introspects shape/dtype from file if omitted. Default fully-replicated `partition_spec`; computes per-rank `local_shape` by dividing each sharded dim by its mesh-axis product; `out_struct = ShapeDtypeStruct(local, dtype)`; cached `_get_read_sm`; blocks until ready. C++ reads `valid_shape` prefix, zero-fills padded tail. |
| `accumulate_slab` | 708-727 | `existing = self.read_slab(...)` at A's spec; `write_slab(name, existing + A)`. Comment says "collective read-modify-write". No caller (see slab_io.py dead_suspects). |
| `close` | 730-791 | Drain → sentinel → join worker → `close_file(fh)` (order matters: in-flight jit could hold ctx_handle). Per-stage rank-0 timing prints gated by `LORRAX_PHDF5_CLOSE_VERBOSE` (default ON: "1" unless "0"). Then rank-0 reopens with serial h5py to write `_deferred_attrs` (delete-then-create), barrier `slab_io_ffi_close_attrs`. |

### Callers / entry points
- Instantiated only by `slab_io.SlabIO.__init__` (line 154-155).
- `_normalize_slab_request` / `_normalize_valid_shape` imported by `_slab_io_allgather.py:28`, `_slab_io_mpi_host.py:51-55`, and (with `_validate_block_divisible`) `tests/test_slab_io_ffi_contract.py:4`.
- `_lustre_prestripe` imported by `_slab_io_mpi_host.py:52` and `src/common/isdf_fitting.py:2294`.
- `_sharding_to_axis_info` imported by `src/ffi/phdf5/read.py:76` ("one source of truth").
- `src/gw/aot_memory_model/kernels/slab_write.py` models `_FfiBackend.write_slab` cost analytically (doc reference).
- `src/common/async_io.py` documents copying "the same threading discipline" (doc reference to `_slab_io_ffi.py:330-339`).

### Flags consumed (env)
`LORRAX_PHDF5_STRIPE_COUNT` (16), `LORRAX_PHDF5_STRIPE_SIZE_FS` ("4M"),
`LORRAX_FFI_DEBUG_SHARDS`, `LORRAX_WRITE_NO_JIT`, `LORRAX_PHDF5_CLOSE_VERBOSE` (default verbose).
Config: reached only via `SlabIOBackend.PHDF5_FFI` routing (cohsex.in `use_ffi_io`).

### Cross-module deps
`ffi.phdf5` (open_file, close_file, read.ffi_read_call, write.ffi_write_call),
`ffi.common.ffi_loader` (phdf5_ensure_dataset, phdf5_open_dataset_ro), h5py,
jax shard_map/multihost_utils, `lfs` binary via subprocess.

### Key boundary arrays
- `A`: N-D jax.Array, NamedSharding on caller's mesh (typically `mesh_xy`), device-resident; worker-thread jit does D2H inside C++ FFI.
- `offset_arr`, `valid_shape_arr`: ndim×int64, explicitly replicated (`P()`).
- read output: per-rank `ShapeDtypeStruct(local_shape)` assembled to `partition_spec` on `mesh`.

### Suspects
- **weird (potential RMW/read-after-write hazard)**: `read_slab` does not `_drain_pending()` when the ds_id is already cached (`_ds_id` drains only on cache miss, 525-538). A read of a dataset with queued-but-unflushed writes from the same backend instance could read stale data. `accumulate_slab` inherits this. Today's usage (write-only files closed before reading) masks it.
- **weird**: module-level `lru_cache(maxsize=None)` on `_get_read_sm`/`_get_write_sm` keyed on `ctx_handle`/`ds_id` ints — cache entries (and their compiled executables) are never evicted across file opens; comment acknowledges coalescing would need C++ changes. Long multi-file runs accumulate compiles.
- **weird**: `create_dataset` silently no-ops `chunks`/`attrs` (only a warning, 451-455) — semantic divergence from the other two backends, which honor them.
- **weird**: `_introspect_cache` created lazily via `getattr` (511-514) instead of in `__init__` — cheap, but a foot-gun for anyone adding threads.
- **weird**: duplicated separator comment lines 464-465 (`# ---` twice) — harmless copy-paste residue.
- **weird**: magic numbers with recorded provenance — queue `maxsize=2` (measured table in comment, 399-407), stripe 16×4M default (bandwidth measurement in `_lustre_prestripe` docstring).
- **redundancy**: none internal; but this file is the template two other backends deliberately mirror (see below).

---

## src/file_io/_slab_io_mpi_host.py (421 loc)

**Purpose.** CPU-backend equivalent of `_FfiBackend`: per-rank hyperslab writes
via h5py(parallel, driver="mpio") + mpi4py, no allgather, no rank-0 bottleneck.
Deliberately synchronous (no worker thread): no D2H to overlap on CPU XLA, and
Cray MPICH `MPI_THREAD_SINGLE` deadlocks if MPI-IO runs on a worker thread and
H5Fclose on main.

**Category.** I/O: parallel-HDF5 CPU backend.

### Functions

| name | lines | role |
|---|---|---|
| `_rank0()` | 58-59 | `jax.process_index() == 0`. |
| `_barrier(tag)` | 62-69 | `sync_global_devices` with blanket `except Exception: pass` fallback for single-process. |
| `_local_shard_and_global_offset(A)` | 72-104 | Return `(np local shard, global offset)` from `A.addressable_shards[0].index` (tuple of slices). Hard error if ≠1 addressable shard per process (multi-device-per-process not the LORRAX mesh-xy regime). Replicated axes → `slice(None)` → offset 0 (all ranks redundantly write identical bytes; "semantically correct" under independent MPI-IO). |
| `_clip_shard_to_valid(local, shard_offset, slab_offset, valid_shape)` | 107-149 | Intersect this rank's shard with the valid (non-padded) region; returns `(clipped_local, dataset_offset)` or `None` if the shard is entirely padded tail. Per-axis: `write_end = min(shard_end, valid_end)`; keeps `local[0:keep]` prefix. |

### `_MpiHostBackend` (153-421)

| method | lines | role |
|---|---|---|
| `__init__` | 161-192 | Lazy `from mpi4py import MPI` (inits MPI); same rank-0 prestripe + barrier as FFI (same env vars); `h5py.File(path, mode, driver="mpio", comm=COMM_WORLD)`; `_deferred_attrs` list. |
| `create_dataset` | 195-222 | Collective H5Dcreate via h5py; **idempotent-return if name already exists** (mode 'a' respect); honors `chunks` and `attrs` (all ranks write identical attr bytes). |
| `write_attr` | 224-227 | Defer to close (same interleave hazard rationale as FFI). |
| `write_slab` | 230-283 | Same normalize helpers as FFI; ensure dataset; `_local_shard_and_global_offset` → `_clip_shard_to_valid` → `dset[slc] = arr` (independent MPI-IO write, no collective sync at write time; rank with fully-padded shard just returns). `k_chunk_size` accepted and ignored (`noqa: ARG002`). |
| `read_slab` | 286-377 | Builds a **zero-filled proto array of the full global shape** via `device_put(jnp.zeros(read_shape), sharding)` just to ask JAX "what slab does this rank own?" (`proto.addressable_shards[0].index`); reads that hyperslab from file into prefix of `host_local` (padded tail stays zero); assembles via `jax.make_array_from_single_device_arrays`. `as_numpy=True` returns the local host block only. |
| `accumulate_slab` | 380-403 | read + add + write back. Docstring: "Drains before/after to keep the read + write rounds non-interleaved" — **there is no drain; writes are synchronous here** (stale copy of the FFI docstring). No caller. |
| `close` | 406-421 | Flush `_deferred_attrs`: rank-0 value `comm.bcast` to all, collective `create_dataset(data=arr)` per attr (skipped if name exists); collective `self._fh.close()`. |

### Callers / entry points
Instantiated only by `slab_io.SlabIO.__init__` (line 160-161) when
`SlabIOBackend.PHDF5_HOST` (gw_config auto-routes `use_ffi_io=true` on CPU here).
No direct imports elsewhere (grepped `_slab_io_mpi_host` across src/tests/tools/scripts:
only slab_io.py). No dedicated unit test found (test_slab_io_ffi_contract.py covers
shared helpers + allgather roundtrip only).

### Flags
Env `LORRAX_PHDF5_STRIPE_COUNT` / `LORRAX_PHDF5_STRIPE_SIZE_FS` (same as FFI).
Config: `SlabIOBackend.PHDF5_HOST`.

### Cross-module deps
mpi4py, h5py(parallel, mpio driver), `_slab_io_ffi` (`_lustre_prestripe`,
`_normalize_slab_request`, `_normalize_valid_shape`), jax multihost_utils.

### Suspects
- **weird**: stale docstring in `accumulate_slab` (388-391) claims drains that don't exist — copy-paste from the FFI backend.
- **weird**: `read_slab` allocates a full-global-shape zeros proto array on device merely to introspect the shard layout (337-341) — O(global) memory + fill for a metadata question; then a second `np.zeros(local_shape)` "shape carrier" is passed to `_clip_shard_to_valid` (350-353) whose data is discarded, and `keep_shape` (357-359) re-derives the clip arithmetic a third way. Comment calls it "the cheap way"; it's cheap in code, not memory.
- **weird**: `create_dataset` idempotency diverges from the allgather backend, which **deletes and recreates** an existing dataset (`_slab_io_allgather.py:114-115`) — same facade call, different overwrite semantics per backend.
- **weird**: h5py dtype line 211 `jnp.dtype(dtype) if not isinstance(dtype, np.dtype) else dtype` — mixed jnp/np dtype juggling.
- **redundancy (by design, documented)**: whole class deliberately parallels `_FfiBackend` minus the writer thread; `_rank0`/`_barrier` are copy-pasted identically in `_slab_io_allgather.py:31-39`.

---

## src/file_io/_slab_io_allgather.py (305 loc)

**Purpose.** Default/fallback backend: gather the full array to every process via
`multihost_utils.process_allgather(tiled=True)`, rank-0 writes with serial h5py;
reads are per-rank independent h5py handles. Works anywhere h5py+jax exist; slow
at scale (rank-0 disk bandwidth + full-array gather memory) but byte-identical
output to the parallel paths.

**Category.** I/O: serial-fallback HDF5 backend.

### Functions

| name | lines | role |
|---|---|---|
| `_rank0()` / `_barrier(tag)` | 31-39 | Identical to `_slab_io_mpi_host` copies. |
| `_to_host(A)` | 42-75 | Fully-replicated host ndarray regardless of sharding. Dispatch: numpy → as-is; single-process → `device_get`; multi-process fully-replicated → `A.addressable_data(0)` (skips gather AND avoids the Path-(D) `(world*N0, *rest)` stacking bug of `process_allgather(tiled=True)` on fully-addressable inputs); else `process_allgather(tiled=True)` (Path B, identity-jit to `P()`, returns exactly `A.shape`). References `reports/.../PROCESS_ALLGATHER_DESIGN_REVIEW_2026-05-20.md`. |

### `_AllgatherBackend` (78-305)

| method | lines | role |
|---|---|---|
| `__init__(path, mode)` | 86-101 | abspath; rank-0 `makedirs`; barrier; rank-0-only `h5py.File` (`self._file`); others hold `None` and no-op except barriers. Separate cached per-rank read handle `self._read_file` (reopening per read was an 80% time hit in the V_q compute loop). |
| `create_dataset` | 104-123 | Rank-0: **delete existing then create** with chunks/attrs; barrier. |
| `write_attr` | 126-143 | Rank-0 immediate write (delete-then-create), handles ndarray / python scalar / JAX value; barrier. (Unlike PHDF5 backends there's no deferral — serial h5py can't corrupt anything.) |
| `write_slab` | 148-207 | Normalize (shared FFI helpers, `check_bounds=False`); `host = _to_host(A)` (every rank gathers the FULL padded slab); rank-0 creates dataset if absent and writes `valid_shape` prefix at `offset`. `k_chunk_size`: stream the rank-0 write along **axis 1** in chunks (legacy sigma k_chunk pattern for large-ω writes; index gymnastics at 193-204 building per-chunk src/dst slicers). Barrier. |
| `read_slab` | 212-270 | Every process reads the hyperslab independently through its own cached 'r' handle (historic LORRAX pattern; with `HDF5_USE_FILE_LOCKING=FALSE`, N independent readers beat rank-0-read + `broadcast_one_to_all` for the many small V_q reads). Zero-fill embed of `valid_shape` prefix into `shape`. `as_numpy=True` fast path returns host ndarray (skips forced H2D+D2H). With `mesh` + `partition_spec`: every rank loads full slab then `device_put(NamedSharding(mesh, spec))` — JAX scatters at device_put time (keeps the unified V_q tile driver backend-agnostic). Else `jnp.asarray(host)` replicated. |
| `accumulate_slab` | 273-295 | `_to_host(A)`; rank-0 reads existing slab, adds, writes back; barrier. No caller. |
| `close` | 298-305 | Close read handle + rank-0 write handle; barrier. |

### Callers / entry points
- Instantiated by `slab_io.SlabIO.__init__` (line 163) as the default backend; also implicitly used by `tests/test_slab_io_ffi_contract.py:98-106` roundtrip.
- `_to_host` imported directly by `src/gw/gw_init.py:1121` (as `_gather_to_host`, for the ζ_μ(G=0) writeback) and `src/common/isdf_fitting.py:2691`.
- `src/solvers/davidson.py:50` and `src/bse/bse_davidson_helpers.py:44` say they "mirror the pattern used by `_slab_io_allgather._to_host`" — i.e. reimplement it locally rather than import (see redundancy).

### Flags
None directly (no env vars, no config reads). Selected by `SlabIOBackend.H5PY_ALLGATHER`.

### Cross-module deps
h5py, jax multihost_utils, `_slab_io_ffi` (`_normalize_slab_request`, `_normalize_valid_shape`).

### Suspects
- **redundancy**: `_to_host` logic duplicated (per their own docstrings) in `solvers/davidson.py` and `bse/bse_davidson_helpers.py` instead of imported — three copies of subtle process_allgather Path-(B)/(D) dispatch that was tricky enough to need a design review doc. Also `_rank0`/`_barrier` duplicated verbatim vs `_slab_io_mpi_host.py`.
- **weird**: `k_chunk_size` chunks hard-codedly along axis 1 (188-204) — meaningful only for the (nk, nω, ...) sigma layout it was lifted from; silently a plain write for other layouts.
- **weird**: `create_dataset` delete-and-recreate vs mpi_host's idempotent-return (cross-backend semantic divergence; see mpi_host notes).
- **weird**: `write_slab` gathers the full global slab to EVERY rank (not just rank 0) — `_to_host` is inherently all-ranks; memory cost is world_size × slab, the documented reason this is the fallback path.
- **dead**: `accumulate_slab` (273) — no callers anywhere (see slab_io.py section).

---

## Group-level observations for the refactor

1. **Three-way backend mirror**: the valid_shape/offset/padding contract is shared
   (good — single source in `_slab_io_ffi`), but overwrite semantics
   (`create_dataset` on existing name), chunks/attrs honoring, and attr-write
   timing (deferred vs immediate) silently differ per backend.
2. **Layering inversion**: `file_io` ← `gw.gw_config` for the `SlabIOBackend` enum,
   worked around with function-local imports in two places.
3. **Dead surface**: the entire `accumulate_slab` feature (facade method, 3 backend
   impls, free function) and the two other free functions are unreached; the
   docstring's claimed consumer (ppm_sigma stream mode) went a different way.
4. **Legacy `use_ffi_io` bool** still threaded through cohsex_sigma / v_q_bispinor /
   sigma_x_bispinor signatures in parallel with the enum.
5. FFI backend's cached-ds_id read path skips the pending-write drain — latent
   read-after-write hazard if any future caller reads a dataset it just wrote
   through the same open backend.
