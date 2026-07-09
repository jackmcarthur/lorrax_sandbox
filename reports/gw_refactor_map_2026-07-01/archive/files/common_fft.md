# Refactor map — group: common FFT / G-box / k-q plumbing

Files (all relative to `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`):
- `src/common/fft_helpers.py` (432 LOC)
- `src/common/gvec_fft_box.py` (132 LOC)
- `src/common/kq_mapping.py` (122 LOC)

Grep scope for all caller claims: `src/`, `tests/`, `tools/`, `scripts/` (`grep -rn --include="*.py"` on function names and module names). `tools/` contains only 3 profiling scripts, `scripts/` only `checks/` + `profiling/`; neither imports these modules.

---

## 1. `src/common/fft_helpers.py` (432 LOC)

### Purpose
Factory functions for sharding-aware FFTs used everywhere in the pipeline (GW, BSE, ISDF fitting, bandstructure), plus an AOT-compile-based FFT workspace query for the memory planner and a block-size chooser for the 2D distributed Cholesky. All FFTs here are "device-local": FFT axes must be replicated; only batch axes may be sharded. Category: **infrastructure: sharded-FFT primitives (+ a memory-planner probe and a linalg block chooser that arguably don't belong here)**.

### Function table

| Function | Lines | Role |
|---|---|---|
| `query_fft_peak_bytes(*, input_shape, fft_axes, sharding, dtype=c128)` | 37–108 | AOT-compiles the exact `jnp.fft.fftn` XLA would emit (via `make_jittable_local_fftn_3d`), reads `compiled.memory_analysis()`, returns per-rank peak bytes = `temp + argument + output − alias`. Caches in module-global `_fft_workspace_cache` keyed by (shape, axes, spec-str, dtype, mesh names+shape). On compile failure falls back to `3 × data_size / n_devices`. Compiler option `xla_gpu_memory_limit_slop_factor: 10000` (magic). |
| `compute_block_size_for_2d_cholesky(n_rmu, Pr, Pc)` | 111–159 | Block-size chooser for 2D blocked Cholesky: constraints `n_rmu % block == 0`, `J % Pr == 0`, `J % Pc == 0` with `J = n_rmu/block`. Tries `J = lcm(Pr,Pc)`, then multiples 2..19, then brute-force descent. Not FFT-related at all — placement is odd. |
| `_normalize_local_fft_axes(rank, axes)` | 168–174 | Negative-axis normalization + uniqueness/bounds checks. Private. |
| `_validate_local_fft_specs(in_spec, out_spec, axes)` | 177–191 | Enforces equal-rank in/out specs and that every FFT axis is replicated (`None`) in both. Private. |
| `_make_jittable_local_fft(mesh, in_spec, out_spec, *, fft_kind, norm, axes)` | 194–262 | Builds a jit-compatible FFT out of per-axis `custom_partitioning`-wrapped 1D FFTs (sequential over axes). IFFT implemented as `conj(fft(conj(x)))/N` (norm-dependent scaling: backward /N, ortho /sqrt(N), forward none) — presumably to reuse the forward-FFT plan. `sharding_rule="...i -> ...i"`. `del mesh` at line 205: the mesh param is ignored (supplied by callback at trace time). Private. |
| `make_jittable_local_ifftn_3d(mesh, in_spec, out_spec, *, norm=None, axes=(-3,-2,-1))` | 265–282 | Public wrapper, `fft_kind="ifftn"`. |
| `make_jittable_local_fftn_3d(...)` | 285–302 | Public wrapper, `fft_kind="fftn"`. |
| `make_sharded_ifftn_3d(mesh, in_spec, out_spec, *, norm=None, axes=(-3,-2,-1))` | 304–327 | shard_map wrapper around plain `jnp.fft.ifftn` on local shard — single 3D cuFFT plan. NEWER family; comments in `bse_lanczos.py:175` call the jittable form "older". Note: signature lines use TAB indentation (305–310). |
| `make_sharded_fftn_3d(...)` | 329–345 | Forward counterpart. Same tab weirdness. |
| `make_flat_k_fft(mesh, kgrid, spec, *, kind, norm='ortho', out_spec=None)` | 359–406 | Flat-k convention wrapper: `(nk, *trail) -> reshape (nkx,nky,nkz,*trail) -> with_sharding_constraint -> local 3D FFT over axes (0,1,2) -> reshape back (nk,*trail)`. `spec` is the PartitionSpec of the 3-D form; leading 3 k-axes must be `None`. Internally dispatches to `make_sharded_{i,}fftn_3d` (single 3D cuFFT plan; comment cites ~1 s savings on Si 4×4×4 BSE 200-iter Lanczos vs the per-axis custom_partitioning form). |
| `make_flat_k_ifftn(mesh, kgrid, spec, *, norm='ortho', out_spec=None)` | 409–419 | Trivial `kind='ifftn'` wrapper. |
| `make_flat_k_fftn(...)` | 422–432 | Trivial `kind='fftn'` wrapper. |

### Entry points and callers (grep evidence)
- `query_fft_peak_bytes` <- `src/gw/gw_init.py:343,351` (chunk chooser stage-cost model), `src/gw/gflat_memory_model.py:485,492` (only when `use_query_fft_peak_bytes=True`, off by default per `gflat_memory_model.py:697`).
- `compute_block_size_for_2d_cholesky` <- `src/common/isdf_fitting.py:137,1132` (only caller).
- `make_jittable_local_fftn_3d` <- internal only (`query_fft_peak_bytes`, fft_helpers.py:76). No external callers.
- `make_jittable_local_ifftn_3d` <- `src/gw/aot_memory_model/kernels/load_psi_rchunk.py:136,144` (only caller).
- `make_sharded_ifftn_3d` <- `src/bse/bse_simple.py:72`, `src/bse/bse_lanczos.py:178`, `src/bse/davidson_absorption.py:123`, `src/bse/absorption_haydock.py:210`, `src/bse/bse_ring_comm.py:404,603`, `src/bse/test_davidson_bse.py:107`, `src/common/load_wfns.py:13`, `src/common/wfn_transforms.py:284`, `tests/archive/test_chunked_wfn_loading.py:419` (re-import via load_wfns).
- `make_sharded_fftn_3d` <- `src/bse/bse_simple.py:74`, `src/bse/bse_ring_comm.py:406,605`, `src/gw/compute_vcoul.py:242`, `src/gw/v_q_tile.py:299,646`, `src/gw/aot_memory_model/kernels/vq_mu_chunk.py:73`, `src/common/wfn_transforms.py:287`, `src/file_io/zeta_reader.py:392`.
- `make_flat_k_fft` <- internal only (via the two wrappers); referenced by name in comments (`gw_init.py:877`, `compute_vcoul.py:1113`, `gflat_memory_model.py:264`).
- `make_flat_k_ifftn` <- `src/bandstructure/htransform.py:388`, `src/gw/cohsex_sigma.py:77,79`, `src/gw/ppm_sigma.py:521,523`, `src/common/isdf_fitting.py:135`, `src/gw/aot_memory_model/kernels/{zct_lr.py:56, chi0_tau_step.py:151, cct_lr.py:52, sigma_kij.py:121,123}`.
- `make_flat_k_fftn` <- `src/gw/cohsex_sigma.py:78`, `src/gw/ppm_sigma.py:522`, `src/gw/w_isdf.py:89-91` (Gv, Gc, chi0 FFTs), `src/common/isdf_fitting.py:136`, same 4 aot_memory_model kernels.

### Physics / equations
None directly — pure FFT plumbing. The flat-k convention ("flatten kx/ky/kz except inside the FFT", header comment lines 348–356) is the k-space convolution backbone of w_isdf chi0, ppm_sigma, cohsex_sigma static COHSEX, and isdf_fitting CCT/ZCT.

### Flags consumed
None from LorraxConfig / cohsex.in directly. `gflat_memory_model` exposes `use_query_fft_peak_bytes` (Python kwarg, default False) that gates the call into this module.

### Key arrays crossing the boundary
- Flat-k FFT: `(nk, *trail)` device arrays, complex128; trail typically `(n_mu, nb)` or `(n_mu, n_mu)`; 3-D form `(nkx,nky,nkz,*trail)` with k-axes replicated, trail axes shardable per `spec`.
- Sharded 3D FFT: any `(..., nx, ny, nz)` with last-3 replicated, batch axes sharded (e.g. `P(None, ('x','y'), None, None, None)` in v_q_tile, compute_vcoul).
- `query_fft_peak_bytes` takes the UNSHARDED full shape (see `gw_init.py:346` comment) and returns an int (host).

### I/O
None (no files read/written).

### Dead suspects
- None strictly dead, but `make_jittable_local_fftn_3d` has zero external callers (grep on name across src/tests/tools/scripts: only fft_helpers.py:76 internal + a "NOT this one" comment at v_q_tile.py:270), and `make_jittable_local_ifftn_3d` has exactly one caller (aot_memory_model mock kernel). The jittable family is on its way out.

### Redundancy suspects
- **Two parallel local-FFT families**: `make_jittable_local_*` (custom_partitioning, per-axis 1D FFTs) vs `make_sharded_*` (shard_map, single 3D plan). Same correctness contract (FFT axes replicated). `bse_lanczos.py:175` explicitly calls the jittable form "older". Production has migrated to shard_map except (a) `query_fft_peak_bytes` internally and (b) `aot_memory_model/kernels/load_psi_rchunk.py`. Candidate for consolidation — but note `v_q_tile.py:268-270` warns the two compile *different HLO*, so the planner probe must match whichever family production uses.
- `make_flat_k_ifftn` / `make_flat_k_fftn` are 2-line wrappers over `make_flat_k_fft(kind=...)` — harmless but three names for one function.
- `compute_block_size_for_2d_cholesky` lives in an FFT module with one caller in `isdf_fitting` — misplaced, should move next to its caller in a refactor.

### Weird code
- `fft_helpers.py:87-99` — the AOT-compile fallback comment says "Logged so the caller notices" but there is **no logging statement**; failure is silent, returning a 3× estimate.
- `fft_helpers.py:84` — magic compiler option `{"xla_gpu_memory_limit_slop_factor": 10000}`.
- `fft_helpers.py:68-79` — `query_fft_peak_bytes` measures via `make_jittable_local_fftn_3d`, but most production FFTs now go through `make_sharded_*` (shard_map). If the two families really compile different HLO (per v_q_tile.py:270 comment), the measured peak may not match production kernels that use the shard_map family.
- `fft_helpers.py:213-218` — IFFT via `conj(fft(conj(x)))` with manual norm scaling instead of `jnp.fft.ifft`; hypothesis: forward-plan reuse or custom_partitioning constraint. Convention-sensitive; verify norm behavior if touched.
- `fft_helpers.py:205` — `del mesh`: the `mesh` parameter of the jittable family is dead; API asymmetry vs the sharded family where mesh is load-bearing.
- `fft_helpers.py:251` — `sharding_rule="...i -> ...i"` written verbatim; declares only the last axis for a per-axis 1D FFT wrapper.
- `fft_helpers.py:305-310, 330-335` — TAB-indented signature lines in `make_sharded_*` (rest of file uses spaces).
- Default `norm` differs between families: flat-k wrappers default `'ortho'`, 3D wrappers default `None` (= 'backward'). aot kernels for ZCT/CCT use `norm="forward"` while chi0/sigma use `"ortho"` — convention spread across call sites.

---

## 2. `src/common/gvec_fft_box.py` (132 LOC)

### Purpose
Sparse-ψ(G) → dense FFT-box scatter, implemented as a precomputed gather table: for each k, map every FFT-box cell `(nx,ny,nz)` to the G-coefficient index within that k's slab (sentinel `ngkmax` → gather zero). One `jnp.take` fills all boxes; no scatter, no per-k loop. Category: **infrastructure: wavefunction G→r-box layout transform (precompute + kernel)**. Companion doc: `src/common/GVEC_FFT_BOX_GATHER.md`.

### Function table

| Function | Lines | Role |
|---|---|---|
| `build_g_index_for_fft_box(gvecs_per_k, fft_grid, ngkmax)` | 29–66 | Host-side (numpy) precompute, once per WFN. For each k: `wrapped = gvecs % fft_grid`; `g_index[k, wx, wy, wz] = g_local`; empty cells = `ngkmax` sentinel. Returns `(nk, nx, ny, nz) int32`. Raises if any `ngk[k] > ngkmax`. |
| `make_fft_box_kernel(mesh, nk, ngkmax, nb_padded, nspinor, fft_grid)` | 69–132 | Builds a jitted shard_map kernel `(cnk_slab, g_index) → psi_G_box`. Input `cnk_slab (nb_padded, ns, nk, ngkmax, 2)` f64 re/im-packed, band-sharded over combined `('x','y')`; `g_index` replicated. Recombines complex, appends a zero slot on the G-axis, flat-gathers via `flat_index = k*(ngkmax+1) + g_index`, transposes to `(nk, nb_padded, ns, nx, ny, nz)` c128 out spec `P(None, ('x','y'), None, None, None, None)`. `check_rep=False`. Requires `nb_padded % world_size == 0`. **Zero callers.** |

### Entry points and callers (grep evidence)
- `build_g_index_for_fft_box` <- `src/file_io/wfn_loader.py:428,438` (`WfnLoader.box_index()`, cached per (k-set, fft_grid); `box_index_dev` caches the device_put — the sphere-idx replicated-leak fix). Only caller.
- `make_fft_box_kernel` <- **NONE**. Grep for `make_fft_box_kernel` across src/tests/tools/scripts: only its own definition/`__all__`/docstring and a comment reference in `src/common/wfn_transforms.py:186` ("Algorithm (matches `common/gvec_fft_box.make_fft_box_kernel` but on WfnLoader's c128 layout)").

### Physics / equations
Layout only: `psi_box[k, b, s, nx, ny, nz] = c[k, b, s, g]` where `gvecs[k][g] mod fft_grid == (nx,ny,nz)`, else 0. This is the standard G-sphere → FFT-box embedding preceding `ifftn` to get ψ(r).

### Flags consumed
None.

### Key arrays
- `g_index (nk, nx, ny, nz) int32` — host numpy at build; replicated on device via `wfn_loader.box_index_dev` (0.16 GB/rank for the documented leak case).
- `cnk_slab (nb_padded, ns, nk, ngkmax, 2) f64` band-sharded `P(('x','y'), None, None, None, None)` — documented as the output of `read_kchunk_union_sharded` with `kchunk_axis=2` (that consumer path no longer exists at this kernel).
- Output `psi_G_box (nk, nb_padded, ns, nx, ny, nz) c128` sharded `P(None, ('x','y'), ...)`.

### I/O
None directly (docstring references the WFN read path that feeds it).

### Dead suspects
- `make_fft_box_kernel` (lines 69–132): zero call sites. Its algorithm was duplicated into `src/common/wfn_transforms.py:_box_kernel` (lines ~195–224, explicitly commented as "matches ... but on WfnLoader's c128 layout", pure-jax, no shard_map, `mode='clip'` take). The half of this module that survives is `build_g_index_for_fft_box`; the kernel half is superseded.

### Redundancy suspects
- `make_fft_box_kernel` vs `wfn_transforms._box_kernel`: same flat-gather algorithm, different layouts (`(nb,ns,nk,ngkmax,2)` f64 shard_map vs `(nk,nb,ns,ngkmax)` c128 pure-jax) — classic old/new parallel path; delete the dead one in the refactor.

### Weird code
- `check_rep=False` at line 131 (standard for shard_map with replicated outputs, but silences a safety check).
- Sentinel-as-`ngkmax` + zero-slot-append convention is shared by three sites (here, `wfn_transforms._box_kernel`, and `wfn_loader.box_index` docs) — a convention that must move in lockstep.
- Header cites `GVEC_FFT_BOX_GATHER.md` without a path; actual location is `src/common/GVEC_FFT_BOX_GATHER.md`.

---

## 3. `src/common/kq_mapping.py` (122 LOC)

### Purpose
Single source of truth for k → k−q plumbing (index lookup + umklapp G-shift + wrap phase) shared by the Sternheimer χ-column solver and the SOS finite-q matrix elements. Created to de-duplicate inlined `round((k−q)−k_kmq)` / `exp(±2πi G_wrap·r)` code in two pipelines. Category: **symmetry/BZ machinery: k−q index + umklapp phase helpers**.

### Function table

| Function | Lines | Role / equation |
|---|---|---|
| `kminq_idx_for_iq(sym, iq_red)` | 43–51 | `(nk_full,) int32` with `kvec_full[idx[ik]] = k_ik − q`. Thin wrapper over `sym.kq_map[:, iq_red]` (canonical SymMaps lookup; convention: full-BZ ik × reduced-BZ iq). Host numpy. |
| `umklapp_G_wrap(kvec_full, kvec_kmq_full, qvec)` | 54–70 | `G_wrap = round((k − q) − k_kmq)`, all crystal coords, `qvec` the signed representative in [−½,½)³; returns int32; vmap/jit safe. |
| `umklapp_phase_box_batched(G_wrap, fft_grid, sign=+1.0)` | 73–104 | jitted (static `fft_grid`): phase `e^{i·sign·2π·G_wrap·r}` on the FFT box, `r = (fx,fy,fz)` fractional grid; batched over leading axis via vmap. Returns `(nk, nx, ny, nz) c128`. `sign=+1` = naive→wrapped gauge; `−1` unwraps. |
| `gather_kminq_box(psi_full, kminq_idx)` | 107–114 | `psi_full[kminq_idx]` — trivial fancy-index alias "kept here for the symmetry of the API". **Zero callers.** |

### Entry points and callers (grep evidence)
- `kminq_idx_for_iq` <- `src/psp/run_sternheimer.py:1337-1338`, `src/psp/get_dipole_mtxels.py:310,389`.
- `umklapp_G_wrap` <- `src/psp/run_sternheimer.py:957-958`. Also **imported** at `src/psp/get_dipole_mtxels.py:310` but never called there — that file recomputes `G_wrap_np = np.round((kvec_k_np - qvec) - kvec_kmq_np)` inline at line 400 (numpy per-k scalar loop).
- `umklapp_phase_box_batched` <- `src/psp/run_sternheimer.py:957,959-960` (both signs).
- `gather_kminq_box` <- **NONE** (grep across src/tests/tools/scripts hits only this module).

### Physics / equations
- Umklapp: for `k − q` folded back into the BZ as canonical `k_kmq`, the residual reciprocal-lattice shift is `G_wrap = round((k − q) − k_kmq)`; the cell-periodic parts relate by `u_{k−q}(r) = e^{−2πi G_wrap·r} u_{k_kmq}(r)` (sign handled by the `sign` argument; +1 wrap / −1 unwrap per this module's convention).

### Flags consumed
None.

### Key arrays
- `kq_map (nk_full, nq_red)` int, host, inside SymMaps.
- `G_wrap (nk, 3) int32` device (traced under vmap in Sternheimer).
- Phase stacks `(nk, nx, ny, nz) c128` device — one per sign in run_sternheimer (two full FFT-box-sized stacks resident).

### I/O
None.

### Dead suspects
- `gather_kminq_box` (lines 107–114): zero callers; self-described as kept only for API symmetry.

### Redundancy suspects
- `get_dipole_mtxels.py:400` re-inlines the exact `round((k−q)−k_kmq)` formula this module was created to centralize (and carries the dead import of `umklapp_G_wrap` at line 310). Violates the module's own "single source of truth" charter — fold in during refactor.
- Header (lines 8–9) claims the SOS path is "consumed by `common.chi_sos`"; `src/common/chi_sos.py` exists but contains no reference to kq_mapping/umklapp — consumption is indirect at best; docstring may be stale.

### Weird code
- `sign: float = +1.0` (line 77) — a float ±1 flag baked into a jitted function's traced arithmetic rather than a static bool; convention (+1 wrap / −1 unwrap) documented but easy to invert silently.
- `functools.partial(jax.jit, static_argnames=('fft_grid',))` on `umklapp_phase_box_batched`: module-import-time jit; `sign` is a traced arg so both signs share one trace (fine, just noting).
