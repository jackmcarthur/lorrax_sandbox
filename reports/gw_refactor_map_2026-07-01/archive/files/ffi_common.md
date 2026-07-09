# ffi/common group — deep-read notes (gw refactor map, 2026-07-01)

Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`
Files: `src/ffi/common/ffi_loader.py` (411 loc), `src/ffi/common/broadcast.py` (68 loc).
Sibling non-Python: `src/ffi/common/cpp/` (api.cc, ffi_helpers.h, scalapack_descriptor.h, CMakeLists.txt, build.sh, in_container.sh, run_shifter.sh, select_gpu.sh) — the C shared object these files load. `src/ffi/common/__init__.py` is a one-line docstring, no re-exports.

---

## src/ffi/common/ffi_loader.py

### Purpose
Locate `liblorrax_ffi*.so`, load it with `ctypes.CDLL(..., RTLD_GLOBAL)`, declare ctypes argtypes/restypes for all `lrx_*` C entry points, and register every XLA-FFI handler symbol with `jax.ffi.register_ffi_target(..., platform="CUDA")` on first `get_lib()` call. Also provides thin Pythonic wrappers around the `lrx_*` lifecycle/utility functions (error-buffer handling, int conversion) so downstream FFI modules don't rewrite ctypes plumbing. Pure infrastructure — no physics, no arrays beyond raw pointers/handles.

### Category
distributed linalg FFI — loader/registration hub (shared by cusolvermp, cusolvermg, cublasmp, slate, phdf5 FFI backends).

### Module-level data
| Name | Lines | Role |
|---|---|---|
| `_LIB` | 35 | module singleton `Optional[ctypes.CDLL]` |
| `_FFI_TARGET_SYMBOLS` | 38–57 | dict: XLA target name → C handler symbol. 16 entries: `lorrax_cusolvermp_eigh→EighMpFfi`, `lorrax_cusolvermp_batched_potrf→CusolverMpBatchedPotrfFfi`, `..._potrs`, `..._solve_lu`, `lorrax_cublasmp_batched_gemm→CublasMpBatchedGemmFfi`, `..._w_solve`, `lorrax_cusolvermg_eigh_f64→EighMgF64`, `lorrax_phdf5_write→PhdfWriteFfi`, `lorrax_phdf5_read→PhdfReadFfi`, `..._read_kchunk`, `..._read_kchunk_union`, `lorrax_slate_eigh→SlateEighFfi`, `..._potrf`, `..._trsm`, `..._batched_potrf`, `..._batched_trsm`. Every target name has a consumer (verified by grep, table below). |
| `_ERR_CAP` | 60 | 512-byte error buffer size for all `lrx_*` wrappers (must match/exceed what the .so writes) |
| `_DTYPE_TAG` | 312–319 | numpy dtype name → int tag: float32=1, float64=2, int32=3, int64=4, complex64=5, complex128=6. Comment says it matches `xla::ffi::DataType`; hardcoded, must stay in sync with `cpp/api.cc`. |
| `__all__` | 33 | `["get_lib"]` only — but many other public functions here are imported by callers (see below); `__all__` is stale/undersized. |

### Function table
| Function | Lines | Role | Callers (grep evidence across src/tests/tools/scripts) |
|---|---|---|---|
| `_candidate_paths()` | 63–81 | search order: `$LORRAX_FFI_SO` env, `src/ffi/common/cpp/build/liblorrax_ffi*.so`, then every dir on `sys.path` glob `liblorrax_ffi*.so`; dedup preserving order | `_locate_so` only |
| `_locate_so()` | 84–93 | first existing candidate; FileNotFoundError with hint "bash src/ffi/common/cpp/build.sh" listing searched paths | `get_lib` only |
| `_set_argtypes(lib)` | 96–192 | declares ctypes signatures for: `lrx_nccl_unique_id_bytes`, `lrx_fill_nccl_unique_id(void*, char*, int)`, `lrx_create_cusolvermp_context(rank, world_size, nccl_uid_addr, uid_nbytes, p, q, grid_layout_col_major, ctx_out*, err, err_cap)`, `lrx_destroy_cusolvermp_context(i64)`, `lrx_smoke_allreduce_sum(i64 ctx, void* dev_ptr, int n)`, `lrx_version_info(int*, int*, int*)`, `lrx_phdf5_open(path, p, q, rank, world, mode_flag, ctx_out*, err, cap)`, `lrx_phdf5_close(i64)`, `lrx_phdf5_init_mpi()`, `lrx_phdf5_ensure_dataset(ctx, name, i64* shape, ndim, dtype_tag, ds_id_out*, err, cap)`, `lrx_phdf5_open_dataset_ro(ctx, name, ds_id_out*, err, cap)`, `lrx_slate_context_create(rank, world, p, q, err, cap)→i64`, `lrx_slate_subrow_context_create(rank, world, Px, Py, err, cap)→i64`, `lrx_slate_context_destroy(i64)`, `lrx_slate_init_mpi()` | `get_lib` only |
| `_register_ffi_targets(lib)` | 195–206 | for each `_FFI_TARGET_SYMBOLS` entry: `jax.ffi.register_ffi_target(name, jax.ffi.pycapsule(fn), platform="CUDA")`; swallows exceptions whose string contains "already registered" | `get_lib` only |
| `get_lib()` | 209–219 | idempotent singleton load + argtypes + FFI registration; RTLD_GLOBAL so the .so's deps (NCCL/MPI/HDF5) resolve for later dlopens | `src/ffi/cusolvermp/{context,batched,eigh}.py`, `src/ffi/cusolvermg/eigh.py`, `src/ffi/cublasmp/batched.py`, `src/ffi/slate/{cholesky,trsm,context,eigh,batched}.py`, `src/ffi/phdf5/{read,write,context}.py`, `src/common/slate_{cholesky_trsm,trsm_isolated,eigh,batched}_test.py`, `src/common/slate_chol_trsm_bench.py`, `src/file_io/wfn_loader.py:284`, `src/file_io/_slab_io_ffi.py:338` |
| `_check_err(rc, err_buf)` | 225–228 | rc!=0 → RuntimeError with decoded err buffer | internal wrappers only |
| `nccl_unique_id_bytes()` | 231–232 | size of ncclUniqueId | `src/ffi/cusolvermp/context.py:82` |
| `fill_nccl_unique_id(addr)` | 235–238 | rank-0 fills ncclUniqueId into caller-owned buffer at raw address `addr` (host uint8 numpy buffer) | `src/ffi/cusolvermp/context.py:65` |
| `create_cusolvermp_context(rank, world_size, uid_addr, uid_nbytes, p, q, grid_layout_col_major=True)` | 241–258 | creates cal/cusolverMp context over p×q device grid; returns opaque int64 handle | `src/ffi/cusolvermp/context.py:87` |
| `destroy_cusolvermp_context(ctx)` | 261–262 | teardown | `src/ffi/cusolvermp/context.py:124` |
| `smoke_allreduce_sum(ctx, device_ptr, nelems)` | 265–267 | NCCL allreduce smoke test over a raw device pointer | **NONE** (grep `smoke_allreduce_sum` across whole repo *.py: only this def) — dead suspect (diagnostic) |
| `version_info()` | 270–276 | dict {cuda_runtime, cuda_driver, nccl} versions | **NONE** (grep `version_info` across whole repo *.py: only this def) — dead suspect (diagnostic) |
| `phdf5_open(path, p, q, rank, world_size, mode_flag)` | 280–296 | collective open/create of parallel-HDF5 file; mode_flag 0='w' truncate, 1='a', 2='r'; returns int64 ctx | `src/ffi/phdf5/context.py:71` |
| `phdf5_close(ctx)` | 299–300 | collective close | `src/ffi/phdf5/context.py:94,107` |
| `phdf5_init_mpi()` | 303–308 | eager `MPI_Init_thread(MPI_THREAD_MULTIPLE)` so first open_file avoids ~400 ms init in hot path; collective, idempotent | `src/gw/gw_driver_helpers.py:156-157` |
| `phdf5_ensure_dataset(ctx, ds_name, shape, dtype_name)` | 322–347 | collective create-or-open N-D dataset; shape = seq of ints → C int64[]; dtype via `_DTYPE_TAG`; returns hid_t as int64 | `src/ffi/phdf5/write.py:115`, `src/file_io/_slab_io_ffi.py:437,577` |
| `phdf5_open_dataset_ro(ctx, ds_name)` | 350–362 | collective H5Dopen read-only; returns hid_t as int64 | `src/ffi/phdf5/read.py:90`, `src/file_io/_slab_io_ffi.py:533` |
| `create_slate_context(rank, world_size, p, q)` | 366–380 | collective SLATE ctx over p×q grid; dups MPI_COMM_WORLD for SLATE; handle==0 → RuntimeError | `src/ffi/slate/context.py:79` |
| `create_slate_subrow_context(rank, world_size, Px, Py)` | 383–400 | SLATE sub-comm: MPI_COMM_WORLD split color=x_rank key=y_rank → one comm of size Py per X-row; for batched ops on (Nbatch, N, N) distributed `P('x', None, 'y')` | `src/ffi/slate/context.py:103` |
| `destroy_slate_context(ctx)` | 403–404 | teardown | `src/ffi/slate/context.py:133,139` |
| `slate_init_mpi()` | 407–411 | eager MPI_THREAD_MULTIPLE init wrapper | **NONE** — the five `src/common/slate_*_test.py` / `slate_chol_trsm_bench.py` files bypass it and call `get_lib().lrx_slate_init_mpi()` on the raw CDLL directly. Dead wrapper / redundancy suspect. |

### Flags / env consumed
- `LORRAX_FFI_SO` env var (absolute path override for the .so).
- No LorraxConfig / cohsex.in keys.

### Arrays crossing the boundary
None as arrays — only raw addresses (`buf.ctypes.data` for ncclUniqueId host buffer, raw device pointer in `smoke_allreduce_sum`) and opaque int64 handles (contexts, hid_t). Actual device-buffer traffic goes through the registered XLA FFI targets called via `jax.ffi.ffi_call` in the sibling modules, not here.

### I/O
None directly (the phdf5 wrappers *manage handles* for parallel-HDF5 files; actual reads/writes of e.g. `V_qmunu.h5` / WFN slabs happen in `src/ffi/phdf5/{read,write}.py` and `src/file_io/_slab_io_ffi.py`). Reads the shared object `liblorrax_ffi*.so` from disk.

### Weird code
1. Lines 12–14 (docstring): claims XLA FFI handlers are "``EighF64``, ``EighC128``" — stale; actual symbols are `EighMpFfi` etc. (dict at 38–57). Hypothesis: docstring predates the one-handler-per-routine dtype-dispatch refactor noted in the dict's own comment (lines 39–41).
2. Line 33: `__all__ = ["get_lib"]` while ~15 other public functions are the real API imported elsewhere. Cosmetic but misleading.
3. Lines 204–206: exception filtering by substring `"already registered"` — fragile string-match on JAX error text; would re-raise if JAX rewords the message, silently pass on any error containing that phrase.
4. Line 215: `RTLD_GLOBAL` — deliberate (symbol resolution for NCCL/MPI/HDF5 chains) but a classic source of cross-library symbol-clash bugs; worth a comment, has none.
5. `_DTYPE_TAG` (312–319) hardcoded to `xla::ffi::DataType` integers — silent breakage if the C side enum shifts; no shared header/codegen.
6. Asymmetric error conventions: cusolvermp/phdf5 wrappers use `rc + err_buf` (`_check_err`), slate context creators use `handle==0 + err_buf`. Two error protocols for the same .so.

### Dead suspects
- `smoke_allreduce_sum` — grep `smoke_allreduce_sum` over all *.py under repo: only ffi_loader.py. (Diagnostic; possibly invoked ad hoc.)
- `version_info` — grep `version_info` over all *.py under repo: only ffi_loader.py.
- `slate_init_mpi` (the Python wrapper) — grep `slate_init_mpi(` finds only test files doing `get_lib().lrx_slate_init_mpi()` directly.

### Redundancy suspects
- `slate_init_mpi()` vs direct `get_lib().lrx_slate_init_mpi()` in 5 files under `src/common/` — parallel old/new call paths for the same C symbol. Consolidate on the wrapper (or delete it).
- `phdf5_init_mpi()` and `slate_init_mpi()` both exist to do "eager MPI_THREAD_MULTIPLE init"; `src/ffi/slate/context.py:8-10` docs note phdf5's version already suffices. Two Python entry points for one job.

---

## src/ffi/common/broadcast.py

### Purpose
Single function `broadcast_bytes(buf, *, key, timeout_ms=60000)`: byte-exact rank-0→all broadcast of a uint8 numpy buffer via the JAX distributed runtime's KV store (hex-encoded string set/blocking-get). Exists because `jax.experimental.multihost_utils.broadcast_one_to_all` silently promotes uint8→uint64 under `jax_enable_x64=True`, scrambling opaque payloads like ncclUniqueId. Used to ship the cuSOLVERMp ncclUniqueId from rank 0 before collectives start.

### Category
distributed linalg FFI — multi-process bootstrap utility.

### Function table
| Function | Lines | Role | Callers |
|---|---|---|---|
| `broadcast_bytes(buf, *, key, timeout_ms=60_000)` | 29–68 | dtype check (uint8 only, else TypeError); single-process → pass-through of input; else rank 0 does `client.key_value_set(key, buf.tobytes().hex())`, all ranks `blocking_key_value_get(key, timeout_ms)`, `bytes.fromhex`, length check vs `buf.size`, return fresh uint8 copy | `src/ffi/cusolvermp/context.py:32,66` (broadcasting ncclUniqueId, key like `'lorrax_ffi/cusolvermp/nccl_uid/v0'`). Grep across src/tests/tools/scripts: only this one caller. |

### Flags / env / arrays
- No config flags. `key` must be unique per call site (KV-store keys are never deleted — see weird code).
- Boundary array: `buf` uint8, host numpy, tiny (ncclUniqueId ~128 B); returned copy equal on every rank.

### I/O
None (network KV store of the JAX distributed runtime; no files).

### Weird code
1. Line 58: `from jax._src.distributed import global_state` — private JAX API; breaks on JAX internal reorganizations. Hypothesis: no public accessor for the distributed client existed when written.
2. Lines 61–63: hex string encoding doubles payload size — fine for 128-B ncclUniqueId, poor for anything large; also KV keys are set-once and never deleted, so re-broadcasting under the same key returns the *stale first* payload (docstring mandates distinct keys but nothing enforces versioning; the `/v0` suffix convention in the caller is the manual workaround).
3. Rank ordering assumption: non-zero ranks call `blocking_key_value_get` possibly before rank 0's `set`; correctness relies entirely on the 60 s `timeout_ms` blocking semantics.

### Dead suspects
None — single public function, one live caller.

### Redundancy suspects
None within the file. Ecosystem-level: it deliberately duplicates `multihost_utils.broadcast_one_to_all` semantics for the uint8 case (justified in docstring; keep, but re-check whether newer JAX fixed the promotion bug).

---

## Cross-module dependency summary (this group)
- Depended on by: `src/ffi/cusolvermp/*`, `src/ffi/cusolvermg/eigh.py`, `src/ffi/cublasmp/batched.py`, `src/ffi/slate/*`, `src/ffi/phdf5/*`, `src/gw/gw_driver_helpers.py`, `src/file_io/wfn_loader.py`, `src/file_io/_slab_io_ffi.py`, `src/common/slate_*_test.py`, `src/common/slate_chol_trsm_bench.py`.
- Depends on: stdlib ctypes/os/sys/pathlib, `jax.ffi` (loader); numpy + `jax._src.distributed` (broadcast).
- FFI-target consumer map (grep for the target-name string): every one of the 16 registered targets is used by exactly one sibling module (cusolvermp/eigh.py, cusolvermp/batched.py ×3, cublasmp/batched.py ×2, cusolvermg/eigh.py, phdf5/write.py, phdf5/read.py ×3, slate/eigh.py, slate/cholesky.py, slate/trsm.py, slate/batched.py ×2).
