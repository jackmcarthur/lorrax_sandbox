# Deep-read notes: src/common/load_wfns.py, src/common/async_io.py

Group: common wavefunction-loading facade + async I/O infrastructure.
Repo root: /pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D. Read-only audit, 2026-07-01.

---

## 1. src/common/load_wfns.py (535 LOC)

### Purpose

Legacy-facade module for pulling wavefunctions from `WFN.h5` onto GPU in the three
representations the rest of LORRAX consumes: full FFT box, contiguous r-chunk slabs,
and centroid-sampled ψ(r_μ). Every function body has already been migrated to thin
wrappers over `file_io.wfn_loader.WfnLoader` + `common.wfn_transforms.{to_box,
to_rchunk, gflat_to_rmu}`; the module survives purely for caller-API back-compat
(`sym` params are `del`'d, `use_phdf5` is `del`'d). `file_io/wfn_loader.py:45`
states the plan explicitly: "P4: migrate consumers; delete SymMaps unfold helpers +
load_wfns helpers" — i.e. this whole module is slated for deletion in the refactor.

Category: **I/O facade: WFN loading / representation transforms (deprecated shim layer)**.

### Module-level imports

- `from . import Meta` (common/__init__), `from . import timing`, `from .fft_helpers import make_sharded_ifftn_3d` (imported but **unused in this file** — only referenced by `tests/archive/test_chunked_wfn_loading.py:419` which imports it *from* load_wfns).
- jax/jnp, Mesh/NamedSharding/PartitionSpec, shard_map (shard_map import unused now).
- Lazy in-function imports: `common.wfn_transforms.{to_box,to_rchunk,gflat_to_rmu}`, `runtime.padding.round_up_to_mesh_product`.

### Function table

#### `load_kpoint_fftbox(wfn, sym, meta, k_idx, nb)` — lines 17–43

- Role: single k-point ψ into FFT box on GPU, shape `(nb, nspinor, nx, ny, nz)` (~0.55 GiB for 12×12 grid per docstring).
- Body: `wfn` is actually a `WfnLoader` (despite legacy name); `loader.load(bands=(0,nb), k=[k_idx], sharding=None)` → `(1, nb, nspinor_wfn, ngkmax)`; zero-pads spinor axis if `meta.nspinor > loader.nspinor` (bispinor pad); builds an inline trivial 1×1 `Mesh(axis_names=('x','y'))` so `to_box` runs the mesh-required path; returns `psi_box[0]`.
- `sym` is `del`'d (line 26) — kept for caller back-compat.
- IMPORTANT semantic trap (documented in `src/psp/dev_status.md:107-113`): `k_idx` is an **UNFOLDED** full-BZ index, not an irreducible index; misuse gives ~1–5 mRy errors at k≥3.
- Physics: none beyond the plane-wave scatter ψ_nk(G) → FFT box (rotation/τ-phase/TR handled inside WfnLoader.load).
- Callers (grep across src, tests, tools, scripts):
  - `src/gw/kin_ion_io.py:194`
  - `src/psp/scf_potential.py:151`
  - `src/psp/dft_operators.py:1113`
  - `src/psp/run_sternheimer.py:820` (imported at :66)
  - `src/psp/tests/test_sternheimer_jvp.py:87`
  - `src/psp/archive/charge_density.py:106`
- Flags: `meta.nspinor`, `meta.fft_grid`.

#### `get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange, nspinor=2)` — lines 46–115

- Role: band energies expanded IBZ→full BZ plus a per-band weighting heuristic used by the ζ-fit least squares.
- Body: reads `wfn.energies[0, :, band_lo:band_hi]` (host numpy, deliberately NOT jnp — big warning block lines 60–72 says jnp here caused ~16 standalone pjit compiles, trimmed in commit 31b5961 2026-04-18; "Do NOT 'fix' this back to jnp"); expands via `sym.irr_idx_k`; if `band_hi > nbnd` in file, pads with sentinel = `max(all energies) + 1.0` Ry (line 88) so f_n=0 for padded bands while keeping PPM resolvent `1/(ω - e + iη)` finite-safe.
- Weight heuristic (lines 106–113):
  - valence: `w = 1/sqrt(max(E_max − e, 1e-12))`
  - conduction: `w = 1/sqrt(max(e − E_min, 1e-12))`
  - selected by `e <= efermi`; normalized by global max; bands inside the sigma window forced to exactly 1.0; then `np.repeat(weights, nspinor, axis=1)` → shape `(nk_full, nb*nspinor)`.
- Returns `(jnp enk (nk_full, nb), jnp weights (nk_full, nb*nspinor))`.
- Note: here `sym` IS used (`sym.irr_idx_k`) — unlike the other functions.
- Callers:
  - `src/gw/gw_jax.py:381,488,728,749` (import :25)
  - `src/gw/ppm_pipeline.py:130,186` (import :26)
  - `src/gw/sc_iteration.py:153`
  - `src/gw/gw_init.py:1159,1260` (imports :1156,1192)
  - `src/bandstructure/htransform.py:424` (via wrapper `load_wfns_and_enk_for_sigma`)
  - Re-exported by `src/common/__init__.py:2` (`from .load_wfns import get_enk_bandrange`).
- Flags: none directly; uses `wfn.efermi`, `wfn.energies`.

#### `read_Gvecs_to_devices(wfn, sym, bandrange, meta, bispinor, mesh_xy, k_range=None)` — lines 118–169

- Role: full-BZ (or k-range) ψ FFT boxes on a 2-D device mesh, band-sharded. Returns `(global_psi_Gtot, nb_logical)`; `global_psi_Gtot` shape `(nk, nb_padded, nspinor, nx, ny, nz)` sharded `P(None, ('x','y'), None, None, None, None)`.
- Body: `loader.load(bands, k, sharding=P(None,('x','y'),None,None), bispinor=...)` (sym unfold, τ-phase, TR conjugation, spinor rotation, band pad/shard, bispinor lift all inside WfnLoader.load); spinor-pad if `meta.nspinor > (4 if bispinor else loader.nspinor)`; `to_box(psi_G_flat, loader.box_index(k), meta.fft_grid, mesh=mesh_xy)`.
- `sym` `del`'d line 144.
- Docstring (line 127) claims the legacy body is "kept available as `read_Gvecs_to_devices_legacy`" — **that function does not exist anywhere in the repo** (grep `read_Gvecs_to_devices_legacy` over src/tests/tools/scripts hits only this docstring). Stale doc.
- Docstring memory note (lines 138–142): FFT box kept only for back-compat; g_flat is ~6–11% of box size; planned PsiGStore→PsiGCache rewrite ("next P4c sub-step") will make the GW hot loop consume g_flat + `to_rchunk` instead.
- Callers:
  - `src/psp/get_DFT_mtxels.py:869` (imports :59,:70 — two import sites, relative and absolute)
  - `src/psp/get_dipole_mtxels.py:541` (import :28)
  - `src/psp/tests/test_dft_hamiltonian.py:137`
  - `tests/archive/test_chunked_wfn_loading.py:104,117,408`
  - `src/centroid/pivoted_cholesky.py:758,780,818` are comment/doc mentions only (its code calls `load_centroids_band_chunked`).
- Flags: `meta.nspinor`, `meta.fft_grid`.

#### `iter_psi_rchunk_bandwise(wfn, sym, meta, mesh_xy, band_range, r_start, r_end, bispinor, band_chunk_size=16, k_chunk_size=0, band_chunk_ranges=None)` — lines 182–285

- Role: generator yielding `(bc_range, psi_bc_Y)` one band chunk at a time; each `psi_bc_Y` is `(nk, nb_chunk, ns, r_end-r_start)` sharded `P(None, None, None, 'y')` — r-chunk slab of ψ for streaming pair-density accumulation (docstring: caller does `P += einsum(ψ_L_bc, ψ_R_bc)` so only one band chunk's shard is live).
- Body:
  - `band_chunk_ranges` param lets caller align chunks with left/right pair-density endpoints (no out-of-range einsums dispatch); default = contiguous `band_chunk_size` chunks.
  - JIT'd shape-memoised zero allocator `_zeros_Y(shape)` (lines 230–239) with `out_shardings=P(None,None,None,'y')` — avoids a replicated top-level `jnp.zeros` on every device.
  - all-k path: `loader.load(..., sharding=P(None,('x','y'),None,None))` → `to_rchunk(psi_G_flat, g_index_full, meta.fft_grid, r_start, n_rchunk, mesh, norm="ortho", kvecs_frac=...)` → `with_sharding_constraint` to Y.
  - k-chunked path (`k_chunk_size>0`): pre-alloc `_zeros_Y((nk_tot, nb_chunk, nspinor, n_rchunk))`, fill with `.at[k0:k1].set(...)` per k batch.
  - `kvecs_frac_full = sym_loader.kvecs_asints / kgrid` — fractional k for the e^{ikr} phase in `to_rchunk`.
  - Uses **private accessor** `loader._ensure_sym()` (line 245).
- `sym` `del`'d line 210. Docstring notes retired params `cached_gspace`/`kvecs_frac`/`use_phdf5` were dropped in migration.
- Callers: **only** `src/bandstructure/htransform.py:162` (import :25). `src/gw/aot_memory_model/kernels/load_psi_rchunk.py:45,108` reference it in docs/model comments only (the AOT memory model mimics it, doesn't call it).
- Flags: `meta.nk_tot`, `meta.nspinor`, `meta.fft_grid`, `meta.kgrid`.

#### `load_centroids_band_chunked(wfn, sym, meta, centroid_indices, bispinor, mesh_xy, band_range, band_chunk_size=64, k_chunk_size=None, *, use_phdf5=False)` — lines 293–535

- Role: ψ sampled at ISDF centroids for a band window. Returns
  - `psi_rmu_Y`: `(nk, nb, ns, n_rmu_padded)` sharded `P(None, None, None, 'y')`
  - `psi_rmuT_X`: `(nk, n_rmu_padded, nb, ns)` = `conj(transpose(0,3,1,2))`, sharded `P(None, 'x', None, None)`.
- Body (the only function here with real logic left):
  1. `del use_phdf5` (line 357 — dead param, "WfnLoader backend='auto' picks phdf5") and `del sym` (line 358).
  2. Chunk-size planning (lines 382–410): per-iter FFT-box budget. Magic constants: `peak_copies = 4 if n_devices==1 else 9` ("conservative XLA scratch multiplier... covers IFFT scratch + IFFT output"), default `gpu_mem_bytes = 36e9` overridden by `meta.memory_per_device_gb * 1e9`; `cs_budget = gpu_mem // (nspinor * n_rtot * 16 * peak_copies)`; legacy `band_chunk_size`/`k_chunk_size` hints translated into flat row count `cs` (rows of the flat `nk·nb_local` axis per scan iter). Long comment cites Defect 3 of `zeta_rchunk_memory_model_2026-05-13/defect_catalog.md` (old path materialised unsharded `c128[nk, band_chunk, ns, nx, ny, nz]` = "Peak A" in `gw/gflat_memory_model.py`).
  3. Canonical device index (line 432): `g_index_full = loader.box_index_dev(k="full_bz", mesh=mesh_xy)` — Round-6 fix; passing host numpy previously produced 3 duplicate device buffers of the same `(nk,nx,ny,nz)` index (measured live count 3 at `pre_rchunk_loop`, agent_l Round-5 §2).
  4. Past-mnband cap (lines 447–478): `b_end_in_file = min(b_end, loader.nbands)`; without cap `WfnLoader.load` raises at `file_io/wfn_loader.py:678` when world_size rounds user band count past file extent (concrete example in comment: CrI3 6×6 30Ry SOC mnband=86, world_size=16 ⇒ b_id_4=96). Zero-pads band axis back to `nb_total` with sharding re-constrained.
  5. `gflat_to_rmu(psi_G_flat, g_index_full, centroid_idx_np, mesh, fft_grid, kvecs_frac, norm="ortho", chunk_size=cs)` — one shard_map + lax.scan; FFT box per-rank-local, band-sharded on `('x','y')`, aliased across scan iters.
  6. `_reshard_all` jit (lines 498–509): pad n_rmu → `round_up_to_mesh_product(n_rmu, mesh_xy)`; two-step reshard `P(None,('x','y'),None,None)` → staging `P(None,'y',None,None)` → `P(None,None,None,'y')` (single all-to-all on band axis before the all-to-all onto n_rmu axis); `psi_rmuT = conj(transpose(0,3,1,2))` constrained to `P(None,'x',None,None)`.
  7. Slice off loader band-pad rows; zero user-band-pad rows `[meta.b_id_4_user − b_start, nb_total)` (contract from `common/meta.py:100-117`).
- n_rmu divisibility note (docstring lines 349–355): each output shards n_rmu by a single mesh axis so n_rmu need only divide one axis size (668/4=167 OK); the `('x','y')` product-shard gap is closed downstream by SlabIO auto-pad (`file_io.slab_io.create_dataset`).
- Physics: representation change only — ψ_nk(G) → ψ_nk(r_μ) at ISDF centroids via masked IFFT + e^{ik·r} phase (inside `gflat_to_rmu`); no equation beyond FT.
- Callers:
  - `src/common/isdf_fitting.py:139` (module-level import; ζ-fit pipeline)
  - `src/gw/gw_init.py:796, 1221` (imports :766, :1189)
  - `src/centroid/pivoted_cholesky.py:959, 975` (import :857; 2-D Gram build, left/right windows)
  - `src/bandstructure/htransform.py:87` (import :25)
  - Doc/comment references: `bandstructure/bse_setup.py:15,164`, `centroid/kmeans_cli.py:150`, `gw/sigma_x_bispinor.py:158`, `gw/v_q_bispinor.py:832`, `gw/wavefunction_bundle.py:205`, `file_io/tagged_arrays.py:204`, `common/psi_G_store.py:208`, `common/meta.py:105`, `gw/gflat_memory_model.py:192-343`, `tests/test_wfn_transforms.py:332,343`.
- Flags: `meta.memory_per_device_gb` (from cohsex.in `memory_per_device_gb`), `meta.nspinor`, `meta.nk_tot`, `meta.fft_grid`, `meta.kgrid`, `meta.b_id_4_user`; dead kwarg `use_phdf5`.
- Uses timing sections: `load_centroids.loader_load`, `load_centroids.gflat_to_rmu`, `load_centroids.reshard`.

### I/O

No direct file I/O in this module. All reads go through the passed-in `WfnLoader`
(reads `WFN.h5`: `wfns/coeffs`, `mf_header` energies (`wfn.energies[0,:,b]`),
`wfn.efermi`, G-vector/box index tables; backend auto-picks phdf5-FFI vs eager h5py).
Writes nothing.

### Suspects

Dead:
- `make_sharded_ifftn_3d` import (line 12–14): unused inside the module; only external ref is `tests/archive/test_chunked_wfn_loading.py:419` importing it *through* load_wfns. Grep: `make_sharded_ifftn_3d` in src/tests/tools/scripts.
- `read_Gvecs_to_devices_legacy` (docstring line 127): promised but nonexistent — grep over src/tests/tools/scripts hits only the docstring.
- Whole module is a shim scheduled for removal per `file_io/wfn_loader.py:45` ("P4: ... delete SymMaps unfold helpers + load_wfns helpers").

Redundancy:
- `load_kpoint_fftbox` vs `read_Gvecs_to_devices`: two parallel routes to the same FFT-box representation (single-k unfolded-index vs full-BZ batch); `psp/dev_status.md:107-113` documents that mixing them causes mRy-level errors. Classic fetch_X/fetch_X_dyn shape.
- `iter_psi_rchunk_bandwise` and `load_centroids_band_chunked` are both thin adapters over `WfnLoader.load` + one `wfn_transforms` kernel; the vestigial `sym`/`use_phdf5`/trivial-mesh scaffolding exists only so old call signatures survive.
- `band_chunk_size`+`k_chunk_size` knobs in `load_centroids_band_chunked` are legacy hints re-derived into a single `cs` — two parameters expressing one quantity.

Weird:
- line 88: sentinel `max(energies) + 1.0` Ry pad for file-short band windows (deliberate, documented).
- lines 106–112: 1/sqrt band-weight heuristic with `1e-12` floor and hard 1.0 inside sigma window — undocumented provenance for the exponent choice.
- lines 60–72: ALL-CAPS "do not convert to jnp" guard block (intentional, commit 31b5961).
- lines 387–389: `peak_copies = 4 if n_devices == 1 else 9`, `gpu_mem_bytes = 36e9` magic constants.
- lines 245, 433: `loader._ensure_sym()` — private-attr reach-in across module boundary.
- line 29 etc.: parameter named `wfn` but must be a `WfnLoader`; `load_centroids_band_chunked` docstring (line 320) still says "wfn: WFNReader" — stale after migration.
- `del sym` ×3, `del use_phdf5` — back-compat ballast the refactor should drop.

---

## 2. src/common/async_io.py (128 LOC)

### Purpose

`AsyncDispatcher`: a single daemon worker thread draining a bounded `queue.Queue`
of zero-arg tasks, with FIFO order, back-pressure (default maxsize=2), and
stash-and-re-raise error semantics. Written to generalise the `_dispatch_loop`
pattern from `file_io/_slab_io_ffi.py` so one threading discipline could serve both
async HDF5 writes (SlabIO) and async ψ reads (`AsyncWfnReader`). Single-worker (not
a pool) because HDF5 MPI-IO file handles are not multi-thread safe (out-of-order
calls trip `MPI_File_set_view: Invalid datatype`).

Category: **infrastructure: threading/async I/O utility (currently orphaned)**.

### Class table

#### `AsyncDispatcher(name, maxsize=2)` — lines 34–125

- `__init__` (58–67): bounded queue, pending counter + Condition, daemon thread on `_loop`.
- `submit(task)` (69–76): raise if closed; re-raise stashed worker error; increment pending; `queue.put` (blocks at maxsize → back-pressure).
- `drain()` (78–82): wait pending==0, then re-raise stashed error.
- `pending` property (84–87).
- `close()` (89–95): idempotent; drain, poison-pill `None`, join.
- `__enter__`/`__exit__` (97–102): context manager.
- `_raise_if_error` (104–109): pop-and-raise stashed exception.
- `_loop` (111–125): get task; `None` → return; run; stash first `BaseException`; decrement pending + `notify_all` in `finally`.
- No physics, no flags, no arrays. Rationale for maxsize=2 cites measurement at `_slab_io_ffi.py:330-339` (K=2 same throughput as K=4, saves 2 chunk buffers).

### I/O

None directly (pure threading). Intended substrate for HDF5 write (SlabIO H5Dwrite
tasks) and read (`WfnLoader.load` prefetch) paths.

### Callers / dead-code analysis

Grep `AsyncDispatcher|async_io|AsyncWfnReader` across src, tests, tools, scripts:
- Sole importer: `src/file_io/wfn_loader.py:1214` inside `AsyncWfnReader` (defined :1179).
- `AsyncWfnReader` itself is **never instantiated anywhere**: the only other hit is a
  comment at `src/common/psi_G_store.py:195` — "an AsyncWfnReader (file_io/wfn_loader.py)
  is available" (i.e. explicitly not wired in).
- Meanwhile `file_io/_slab_io_ffi.py` still runs its **own inline** `_dispatch_loop`
  (thread launch :414, loop body :466) and was never migrated onto AsyncDispatcher.

So: dead_suspect (entire module — its only consumer chain AsyncDispatcher →
AsyncWfnReader terminates in a comment) AND redundancy_suspect (parallel
implementation of the same single-worker/bounded-queue/poison-pill pattern living
in `_slab_io_ffi.py`). The refactor should either migrate SlabIO's `_dispatch_loop`
onto `AsyncDispatcher` and actually wire `AsyncWfnReader` into `psi_G_store`, or
delete both `async_io.py` and `AsyncWfnReader`.

### Weird code

- `except BaseException` with only-first-error stashing (line 118): later task
  exceptions after the first are silently dropped until the first is re-raised.
- `submit` after a worker error still increments pending before the queue put only
  if `_raise_if_error` passes — fine — but `close()` calls `drain()` which re-raises,
  leaving the object not-closed and the worker alive on error paths (minor).
