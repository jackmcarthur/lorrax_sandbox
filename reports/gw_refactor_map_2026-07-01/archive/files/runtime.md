# src/runtime/ — deep-read notes (gw_refactor_map_2026-07-01)

Group: `src/runtime/aot_memory.py` (519 loc), `src/runtime/padding.py` (333 loc),
`src/runtime/__init__.py` (214 loc). All paths relative to
`/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

No physics equations anywhere in this package — it is pure resource-management /
process-bootstrap infrastructure. No cohsex.in keys or LorraxConfig flags are consumed
directly by any of these files (env vars only; see per-file tables).

---

## 1. src/runtime/aot_memory.py (519 lines)

### Purpose
Ahead-of-time GPU peak-memory prediction for jit-compiled kernels that contain FFTs.
XLA's `compiled.memory_analysis()` misses cuFFT plan workspace (allocated lazily by
jaxlib's FFT thunk); this module parses the lowered HLO text for `fft(...)` ops,
rebuilds each plan via ctypes against the *exact* `libcufft.so` jaxlib has dlopen'd
(found via `/proc/self/maps`), and queries `cufftMakePlanMany` with
`cufftSetAutoAllocation(plan, 0)` for the workspace size. Category: **resource mgmt:
memory planner (exact AOT query flavor)**.

### Function table

| Name | Lines | Role |
|---|---|---|
| `FftSpec` (frozen dataclass) | 53–76 | Plan identity: `rank`, `transform_shape` (row-major, = XLA `fft_length`), `batch` (product of leading non-transform dims), `dtype` ('c128'/'c64'/'f64'/'f32'), `fft_type` ('FFT'/'IFFT'/'RFFT'/'IRFFT'). Hashable → used as lru_cache key. |
| `AotPeakBreakdown` (frozen dataclass) | 79–100 | Result: `compiled_peak` (XLA-visible bytes), `cufft_scratch` (**max** over distinct plans, not sum — only one plan live at a time), `total = compiled_peak + cufft_scratch`, `fft_specs` tuple for diagnostics. |
| `HloFftParseError` | 103–110 | Loud-failure exception when HLO contains `" fft("` but regex matched nothing (XLA format drift detection). |
| `CufftQueryError` | 113–114 | cuFFT ctypes call failure / no libcufft found. |
| `_FFT_OP_RE` (module regex) | 138–157 | Verbose regex matching both HLO printout flavors (typed-operand and bare-operand). Captures LHS output `dtype`, `shape`, `fft_type`, `fft_length`. |
| `parse_fft_specs_from_hlo(hlo_text)` | 160–227 | Walk HLO dump → `list[FftSpec]`. Validates rank ∈ {1,2,3}; for C2C enforces trailing output dims == fft_length (skipped for RFFT/IRFFT half-spectrum convention). `batch = ∏ op_shape[:-rank]`. Raises `HloFftParseError` if `" fft("` present but zero matches. |
| cufftType constants | 235–244 | `_CUFFT_R2C=0x2A, _CUFFT_C2R=0x2C, _CUFFT_C2C=0x29, _CUFFT_D2Z=0x6A, _CUFFT_Z2D=0x6C, _CUFFT_Z2Z=0x69`, `_CUFFT_SUCCESS=0` (from cuda/include/cufft.h). |
| `_cufft_type_for(dtype, fft_type)` | 247–274 | (dtype, op kind) → cuFFT type code. c128→Z2Z/D2Z/Z2D, c64→C2C/R2C/C2R, f64/f32 RFFT only. Raises `CufftQueryError` otherwise (no half precision). |
| `_jax_cufft_handle()` | 277–366 | `@functools.lru_cache(maxsize=1)`. Forces `jax.devices("gpu")`, scans `/proc/self/maps` for first path whose basename starts with `libcufft.so`, `ctypes.CDLL(..., RTLD_GLOBAL)`, binds ABI signatures for `cufftCreate/Destroy/SetAutoAllocation/MakePlanMany`. Raises `CufftQueryError` on CPU-only builds. |
| `_query_one_plan_workspace_bytes(spec)` | 374–446 | `@functools.lru_cache(maxsize=512)`. Builds the plan with auto-allocation off; `idist/odist = ∏ n` for C2C; RFFT out_dist and IRFFT in_dist use the half-spectrum `n[-1]//2 + 1` factor (mirrors jaxlib fft_thunk.cc convention). Guards `batch < 2**31` (plain int-batch API; `cufftMakePlanMany64` not implemented). `cufftDestroy` in `finally`. |
| `query_cufft_workspace_bytes(specs)` | 449–464 | Dedup specs via `set`, return **max** of per-plan workspaces (0 if empty). NOTE: docstring first line says "Sum ... across distinct FftSpecs" but the body of the docstring and the code return max. |
| `aot_kernel_peak_bytes(compiled)` | 472–519 | **Public entry.** `compiled_peak = temp + argument + output − alias` from `compiled.memory_analysis()`; parses `compiled.as_text()`; if FFT specs found, queries cuFFT scratch but **demotes `CufftQueryError` to `cufft_scratch=0`** (chooser still works on CPU sandboxes). Parser errors still propagate loudly. |

### Entry points and callers (grepped src/, tests/, tools/, scripts/)
- `aot_kernel_peak_bytes` ← `src/gw/v_q_tile.py:220-221` (V_q full-kernel AOT peak, cached in `_v_q_full_kernel_aot_cache`; verbose print gated by env `LORRAX_V_Q_AOT_VERBOSE`, process 0 only); ← `tests/test_aot_memory.py:120`.
- `parse_fft_specs_from_hlo`, `FftSpec`, `HloFftParseError` ← `tests/test_aot_memory.py` (lines 33, 55, 74, 88).
- `_query_one_plan_workspace_bytes` (private) ← `tests/test_aot_memory.py:189, 214`; ← `scripts/profiling/aot_cufft_sanity.py:50-51`.
- `query_cufft_workspace_bytes`, `AotPeakBreakdown`, `CufftQueryError` — no external callers by name (used internally / returned as value).

### Key arrays / boundary data
None cross this boundary as arrays; input is a `jax.stages.Compiled` object, output is a
plain dataclass of ints. All work is host-side (ctypes + regex); allocates no GPU memory
beyond the plan descriptor.

### Flags consumed
None directly. (`LORRAX_V_Q_AOT_VERBOSE` is consumed by the *caller* v_q_tile.py, not here.)

### I/O
Reads `/proc/self/maps` (Linux procfs). No LORRAX file formats read/written.

### Relationship to `src/gw/aot_memory_model/`
Distinct subsystem with an overlapping name and mission: `gw/aot_memory_model/`
(chooser.py, core.py, kernels/*, fitted JSON artifacts under
`aot_memory_model/artifacts/`) is a *fitted/regressed* memory model used by
`gw_init.py:431` for chunk-size choice; `runtime/aot_memory.py` is an *exact*
compile-and-query path used by v_q_tile. The v_q_tile comment ("no slope+intercept
guess") explicitly positions it as superseding the fitted approach for that kernel.
Refactor-map relevance: two parallel memory-prediction systems coexist.

### Suspects
- weird_code: `query_cufft_workspace_bytes` docstring/name-vs-behavior mismatch
  ("Sum" in the summary line; code and later docstring text say max). Hypothesis:
  implementation changed from sum to max, first line never updated.
- weird_code: magic cuFFT hex type codes (0x2A etc.) and the `/proc/self/maps`
  library-discovery trick — both deliberate and documented, but Linux/CUDA-only.
- weird_code: intentional asymmetry — parser failures raise loudly, cuFFT-query
  failures silently demote to 0 scratch (documented design choice at lines 504–512;
  means a broken ctypes path could silently reintroduce the under-prediction the
  module exists to fix, on GPU systems where libcufft *should* be present).
- dead_suspects: none. All public names have callers (grep above).
- redundancy_suspects: conceptual overlap with `src/gw/aot_memory_model/` (see above).

---

## 2. src/runtime/padding.py (333 lines)

### Purpose
Single-source-of-truth helpers for the padding contract on sharded arrays: in-memory
arrays may be zero-padded so an axis extent divides the mesh(-product) sharding divisor,
while files on disk keep the logical (unpadded) extent so they can be re-read on any
process count. Pure shape arithmetic + jit-safe jnp pad/slice helpers + a `PadAxis`
metadata record bridging to SlabIO's `valid_shape=` argument. Category: **resource
mgmt / distributed-sharding shape machinery**.

### Function table

| Name | Lines | Role |
|---|---|---|
| `PadAxis` (frozen dataclass) | 41–73 | Per-axis pad record: `axis`, `logical`, `padded`, `mesh_axes` (single-axis `('x',)` or product `('x','y')`). Properties `pad_size`, `is_padded`. |
| `pad_shape_to_mesh(shape, partition_spec, mesh)` | 76–104 | Pure shape arithmetic: for each dim with non-None spec entry, round extent up to divisibility by ∏ mesh.shape[ax] over the entry's axes. Docstring example: mesh {'x':4,'y':4}, `P(None,None,('x','y'))`, (9,60,668) → (9,60,672). |
| `logical_shape_from_padded(padded, logical)` | 107–125 | "Trivial validator": asserts rank match and `logical[d] <= padded[d]`, returns `logical` unchanged. Self-described as "mainly a documentation hook". |
| `round_up_to_mesh_product(n, mesh)` | 128–144 | Round scalar n up to ∏ over ALL mesh axis sizes (worst-case divisor). Used for e.g. `Meta.n_rmu_padded` before shardings exist. |
| `_spec_axes(spec_entry)` | 152–158 | Normalize PartitionSpec entry → tuple of axis names or None. |
| `_spec_divisor(spec_entry, mesh)` | 161–169 | Effective divisor for a spec entry (1 for None). |
| `pad_array_to_mesh(arr, partition_spec, mesh)` | 172–236 | Zero-pad (`jnp.pad`, trailing side) each under-divisible sharded axis, then `jax.lax.with_sharding_constraint(out, NamedSharding(mesh, partition_spec))`. Returns `(padded_array, tuple[PadAxis,...])`. Lazy `import jax` inside body. Jit-safe; raises ValueError if spec rank > array rank. |
| `unpad_array_from_mesh(arr_padded, pad_meta, *, target_partition_spec=None, mesh=None)` | 239–302 | Inverse: `jax.lax.slice` (static limits) back to logical extents; optional WSC reshard to `target_partition_spec` (e.g. drop product layout to single-axis `P(None,'x',None)`). Validates PadAxis.axis range and logical ≤ padded. |
| `valid_shape_from_pad_meta(padded_shape, pad_meta)` | 305–322 | Build SlabIO `valid_shape=` tuple: padded shape with padded axes replaced by logical extents. |
| `__all__` | 325–333 | All 7 public names exported. |

### Entry points and callers (grepped src/, tests/, tools/, scripts/)
- `round_up_to_mesh_product` ← `src/common/load_wfns.py:360,418` (`n_rmu_padded = round_up_to_mesh_product(n_rmu, mesh_xy)`); ← `tests/test_padding.py`. **Only production caller in the whole module.**
- `pad_shape_to_mesh`, `pad_array_to_mesh`, `unpad_array_from_mesh`, `valid_shape_from_pad_meta`, `PadAxis` ← `tests/test_padding.py` ONLY (comprehensive tests at lines 27–244). Zero production callers.
- `logical_shape_from_padded` ← NOTHING. Zero callers anywhere including tests (grep across src/, tests/, tools/, scripts/ for the name returned only its definition).
- `src/file_io/slab_io.py:191` references `runtime.padding` in a *comment*: "(see ``runtime.padding`` in the agent/padding-refactor branch)" — i.e., the array-level helpers are scaffolding landed ahead of a padding refactor that has not been wired into production paths on this checkout.

### Key arrays crossing the boundary
- `pad_array_to_mesh`: any jax/numpy array (device, sharded via WSC after pad); pad zone always zero-filled so downstream contractions along the padded axis (docstring cites "einsums in V_q tile, V·χ in W solve") see no contribution from pad rows.
- No einsums in this file.

### Flags consumed
None (no env vars, no config keys).

### I/O
None directly; `valid_shape_from_pad_meta` exists purely to feed
`SlabIO.write_slab(..., valid_shape=...)` / `read_slab(..., shape=padded, valid_shape=...)`.

### Suspects
- dead_suspects: `logical_shape_from_padded` — zero callers found (grepped
  `logical_shape_from_padded` across src, tests, tools, scripts; only hits are its
  definition and `__all__`). Self-admits to being "mainly a documentation hook".
- dead_suspects (production-dead, test-only): `pad_shape_to_mesh`,
  `pad_array_to_mesh`, `unpad_array_from_mesh`, `valid_shape_from_pad_meta`, `PadAxis`
  — only callers are `tests/test_padding.py`; slab_io.py comment says the consuming
  refactor lives on branch `agent/padding-refactor`, not on this checkout. Not cruft
  per se — pre-landed scaffolding — but on main only `round_up_to_mesh_product` earns
  its keep.
- redundancy_suspects: none within the file; check whether ad-hoc padding logic still
  exists at JIT boundaries in gw drivers that this module was meant to replace
  (outside this group's scope).
- weird_code: hardcoded port-free; only oddity is lazy `import jax` inside
  `pad_array_to_mesh` / `unpad_array_from_mesh` (deliberate — keeps module importable
  before `set_default_env()`; consistent with runtime/__init__ pattern).

---

## 3. src/runtime/__init__.py (214 lines)

### Purpose
Centralized JAX process bootstrap: env-var defaults (`JAX_ENABLE_X64`,
`JAX_PLATFORMS`), idempotent `jax.distributed.initialize()` with the SLURM/Cray-MPICH
argument pattern that works on Perlmutter (explicit `local_device_ids` from
CUDA_VISIBLE_DEVICES; explicit coordinator fallback), CPU fallback for GPU-less
sandboxes, and NCCL communicator warmup. Replaced five drifting per-driver copies
(per its own docstring). Category: **runtime bootstrap / distributed init**.

### Function table

| Name | Lines | Role |
|---|---|---|
| `_DISTRIBUTED_SENTINEL` | 38 | `"_LORRAX_JAX_DISTRIBUTED_DONE"` — env-var (not module-global) sentinel so idempotence survives re-imports (`python -m gw.gw_jax` → `gw_init` re-imports `gw.gw_jax`). |
| `__all__` | 40–44 | `set_default_env`, `init_jax_distributed`, `fallback_to_cpu_if_no_gpu_backend`. **`nccl_warmup` is defined and used but missing from `__all__`.** |
| `set_default_env(*, platform="gpu")` | 47–62 | Must run BEFORE `import jax`. `setdefault("JAX_ENABLE_X64","1")`; gpu → `setdefault("JAX_PLATFORMS","cuda,cpu")`; cpu → hard-set `JAX_PLATFORMS=cpu`; else ValueError. |
| `_resolve_proc_count()` | 65–71 | `JAX_PROCESS_COUNT` → `JAX_NUM_PROCESSES` → `SLURM_NTASKS` → 1. |
| `_resolve_proc_id()` | 74–78 | `JAX_PROCESS_INDEX` → `SLURM_PROCID` → 0. |
| `_resolve_coordinator_address()` | 81–106 | `JAX_COORDINATOR_ADDRESS` wins; else first host of `SLURM_NODELIST` via `scontrol show hostnames` subprocess + hardcoded port `12355`; final fallback `SLURMD_NODENAME`/`HOSTNAME`/`localhost` + `:12355`. Broad `except Exception: pass` around the subprocess. |
| `init_jax_distributed()` | 109–152 | Idempotent via env sentinel. proc_count ≤ 1 → set sentinel, return (no jax import). Else: `n_local` = count of entries in CUDA_VISIBLE_DEVICES; try `jax.distributed.initialize(local_device_ids=list(range(n_local)))` (bare `except Exception: pass` on failure); fall back to explicit `(coordinator_address, num_processes, process_id)` form. Rationale documented: Cray MPICH runs each rank with `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`, and no-args initialize hangs in topology exchange. |
| `nccl_warmup(mesh_xy)` | 155–195 | Pre-initialize the three NCCL communicator patterns the mesh uses (full-mesh psum, per-'x'-axis psum, per-'y'-axis psum) by firing `jax.jit(jnp.sum)` on tiny f64 arrays sharded `P(*axis_names)` and `P(ax)` per axis. Moves ~1–2 s/communicator `ncclCommInitRank` cost off timed sections (docstring cites a 1.9 s `all-reduce-start` inside `jit(_mean)` in the sigma phase). No-op when `jax.process_count() <= 1`. Routes through implicit-reduction path because `jax.lax.psum` isn't callable from top-level jit. Docstring's replica-group examples (`{{0,1,2,3}}` etc.) assume a 2×2 mesh; code is general over `mesh_xy.axis_names`. |
| `fallback_to_cpu_if_no_gpu_backend()` | 198–214 | `jax.devices()`; on `RuntimeError` containing `"Unknown backend: 'gpu'"`, pop `JAX_PLATFORM_NAME` and force `JAX_PLATFORMS=cpu` so the caller's next `jax.devices()` succeeds; other RuntimeErrors re-raised. |

### Entry points and callers (grepped src/, tests/, tools/, scripts/)
- `set_default_env` ← `src/gw/gw_jax.py:1-2`, `src/psp/run_nscf.py:23-24`, `src/psp/run_sternheimer.py:48-49`, `src/centroid/kmeans_cli.py:10-11`, `src/ffi/cusolvermp/profile_batched.py:35-37`, `src/psp/tests/test_sternheimer_jvp.py:41-42`.
- `init_jax_distributed` ← `src/gw/gw_jax.py:13-14`, `src/psp/run_nscf.py:34-35`, `src/psp/run_sternheimer.py:62-63`, `src/centroid/kmeans_cli.py:21-22`, `src/bandstructure/htransform.py:9-10`, `src/bse/bse_jax.py:11-12`, `src/bse/absorption_haydock.py:25-26`, `src/ffi/cusolvermp/profile_batched.py:268`.
- `fallback_to_cpu_if_no_gpu_backend` ← `src/gw/gw_jax.py:15`, `src/ffi/cusolvermp/profile_batched.py:269`.
- `nccl_warmup` ← `src/gw/gw_driver_helpers.py:148-152` (inside `_timing.section("nccl_warmup")`).
- Back-compat alias: `src/gw/gw_jax.py:19` `_maybe_init_jax_distributed = init_jax_distributed` ("legacy name" shim).

### Flags / env vars consumed
`JAX_ENABLE_X64`, `JAX_PLATFORMS`, `JAX_PLATFORM_NAME`, `JAX_PROCESS_COUNT`,
`JAX_NUM_PROCESSES`, `JAX_PROCESS_INDEX`, `JAX_COORDINATOR_ADDRESS`,
`SLURM_NTASKS`, `SLURM_PROCID`, `SLURM_NODELIST`, `SLURMD_NODENAME`, `HOSTNAME`,
`CUDA_VISIBLE_DEVICES`, `_LORRAX_JAX_DISTRIBUTED_DONE` (read+written).
No LorraxConfig / cohsex.in keys.

### Key arrays crossing the boundary
Only `nccl_warmup`'s throwaway warm arrays: shape `mesh.shape` (full mesh, spec
`P(*axis_names)`) and `(n_ax,)` per axis (spec `P(ax)`), f64, device-placed via
`jax.device_put`, reduced by `jax.jit(jnp.sum)` and discarded.

### I/O
None (spawns `scontrol show hostnames` subprocess; hardcoded coordinator port 12355).

### Suspects
- redundancy_suspects: `src/common/cusolvermp_eigh_test.py:43-66` still carries its
  own private copy of `_maybe_init_jax_distributed` (same `_LORRAX_JAX_DISTRIBUTED_DONE`
  sentinel, slightly different logic: `local_device_ids=[0]` only when exactly one GPU
  visible, and it sets the sentinel even after total failure). This is exactly the
  drifting-copy pattern the module docstring says it eliminated — one straggler remains.
- redundancy_suspects: `src/gw/gw_jax.py:19` legacy alias
  `_maybe_init_jax_distributed = init_jax_distributed` — back-compat shim, candidate
  for deletion in the refactor once external references are gone.
- weird_code: `nccl_warmup` defined here and imported by gw_driver_helpers but absent
  from `__all__` (lines 40–44). Hypothesis: added after `__all__` was written; harmless
  (explicit import works) but inconsistent.
- weird_code: sentinel is an **env var**, so it is inherited by child processes — a
  subprocess that should initialize its own jax.distributed would silently skip it.
  Deliberate for the re-import case; a latent footgun for any fork/spawn usage.
- weird_code: two bare `except Exception: pass` blocks (first-attempt
  `jax.distributed.initialize`, line 144; `scontrol` subprocess, line 101) — masks
  genuine init errors and retries down a different path; documented as intentional
  fallback ladders.
- weird_code: hardcoded coordinator port `12355` (lines 100, 106) — collides if two
  independent multi-process LORRAX jobs share a node without setting
  `JAX_COORDINATOR_ADDRESS`.
- weird_code: `fallback_to_cpu_if_no_gpu_backend` mutates `JAX_PLATFORMS` *after*
  `import jax` — works only because the backend is initialized lazily on the next
  `jax.devices()` call; ordering-sensitive.
- dead_suspects: none — all four functions have production callers (grep above).
