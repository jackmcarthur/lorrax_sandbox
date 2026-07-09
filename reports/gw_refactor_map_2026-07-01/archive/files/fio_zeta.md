# file_io ζ group — deep-read notes (refactor map 2026-07-01)

Files: `src/file_io/zeta_loader.py` (621 loc), `src/file_io/zeta_reader.py` (426 loc),
`src/file_io/isdf_header.py` (296 loc). Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

Grep scope used for all caller evidence: `src/`, `tests/`, `tools/`, `scripts/` (tools/ and
scripts/ produced zero hits for every symbol in this group; only src + tests are cited below).

---

## 1. `src/file_io/zeta_loader.py` — `ZetaLoader`

### Purpose
New-style, `WfnLoader`-shaped single-entry-point reader for `zeta_q.h5` (ISDF ζ interpolation
vectors). One `.load(q=, mu=, layout=, sharding=)` call replaces the offset/count slab methods of
the legacy `ZetaReader`, adds symbolic q ranges (`'ibz'` / `'full_bz'`) and an in-loader IBZ→full-BZ
symmetry unfold. Header attribute surface is a 1:1 mirror of `ZetaReader` so it is a drop-in source.

Category: **I/O: zeta_q reader (new API, migration target)** — with a slice of symmetry machinery
embedded (`_unfold_q_full_bz`).

### Entry points and callers
- `ZetaLoader` (class) exported via `src/file_io/__init__.py:15`.
  - Callers found: `tests/test_zeta_loader.py` (7 instantiations), `tests/test_compute_all_V_q_g_flat.py`
    (3 instantiations), `tests/test_compute_V_q_bispinor_g_flat.py:235` (charge-ζ loader in one test).
  - **Zero production instantiations**: grep `ZetaLoader(` across `src/` (excluding zeta_loader.py)
    returns nothing. Production code (`gw/gw_init.py`) instantiates `ZetaReader` exclusively.
  - Duck-typed acceptance: `gw/v_q_g_flat.py:232-256` `_make_read_all_ibz` accepts either object
    (`has_load = callable(getattr(zeta_loader,'load',None))`), with comment "ZetaLoader (test bench)
    and ZetaReader (production driver)". `gw/v_q_bispinor.py:484-485` annotates loader args as
    "ZetaReader/Loader".

### Function table

| Function | Lines | Role |
|---|---|---|
| `ZetaLoader.__init__` | 81–190 | Open file once; `read_mf_header_from_file` + `read_isdf_header_from_file`; probe dataset shape (`zeta_q` for r_space, `zeta_q_G` for G_flat); mirror ~35 mf_header attrs + isdf_header attrs onto self; classify `q_layout = 'full_bz' if n_q_on_disk == prod(kgrid) else 'ibz'`; hold a `SlabIO(self._path, mode=mode, mesh=mesh)` open for the loader lifetime. Note: `backend` argument is validated but **not forwarded to SlabIO** (comment at 184–188 admits it is "currently advisory"). |
| `_auto_pick_backend` | 193–197 | `'phdf5'` iff `mesh is not None and jax.process_count() > 1`, else `'eager'`. |
| `close` / `__enter__` / `__exit__` | 200–210 | SlabIO lifecycle. |
| `n_rtot` (property) | 215–218 | `nx*ny*nz` from fft_grid. |
| `slab_io` (property) | 221–226 | Escape hatch to raw `read_slab('zeta_q', ...)` "during migration". No callers found by grep of `.slab_io` outside the two reader files + gw_config (unrelated `config.backend.slab_io`). |
| `load` | 231–351 | The contract. Resolves q (`_resolve_q`) and μ (`_resolve_mu`); default sharding `P(None, None, ('x','y'))` for r_space, `P(None, ('x','y'), None)` for G_flat. Disk-native G_flat → `_read_g_flat_disk` (raises `NotImplementedError` for `layout='r_space'` and for `q='full_bz'` on G-flat disk). r-space: `_read_r_space`, optional `_unfold_q_full_bz`, and for `layout='G_flat'` calls `zeta_reader._do_disk_to_G` (requires `qvec_frac` + `sphere_idx`). |
| `_resolve_q` | 356–382 | `'ibz'`→all disk rows; `'full_bz'`→all rows + `need_unfold` flag if disk is IBZ; explicit int sequence validated to `[0, n_q_on_disk)`. |
| `_resolve_mu` | 384–398 | `(lo,hi)` / slice(step-1 only) / None → half-open μ range against layout-aware `n_rmu_disk`. |
| `_ensure_sym` | 400–409 | Lazy `common.symmetry_maps.SymMaps(self)` cached on instance (works because attr surface mimics WFNReader). |
| `_full_bz_unfold_tables` | 411–457 | Builds `(full_to_irr_idx, full_to_irr_sym, r_perm, mu_perm)` from `sym.irr_idx_q` / `sym.sym_idx_q` + `centroid.orbit_syms.compute_rgrid_sym_perm` / `compute_centroid_sym_perm`. **Loud `NotImplementedError` if any full-BZ q needs a TR-augmented sym index (`>= ntran`)** — this is the TRS-guard corresponding to the trs-blind-sym-bug fix; ψ-side TR (`ζ→ζ*`) not yet wired. Comment at 442–444 asserts writer/reader row-order agreement citing `isdf_fitting.py:1689`. |
| `_unfold_q_full_bz` | 459–519 | Physics: eq. 3 of `reports/zeta_ibz_2026-05-11/report.md`: `ζ_full[q, r_new, μ_new] = ζ_ibz[i(q), r_perm[s(q), r_new], inv_mu_perm[s(q), μ_new]]`, no τ-phase (contracts out inside the V_q bilinear). `inv_mu = np.argsort(mu_perm, axis=-1)`. Partial-μ unfold raises `NotImplementedError` (perm can mix in/out of window). jit with `out_shardings=NamedSharding(mesh, partition_spec)`; body uses `jnp.take_along_axis` twice (r-axis then μ-axis) after `z[idx_j]` parent-row gather. |
| `_read_g_flat_disk` | 521–563 | `read_slab('zeta_q_G', shape=(Q, μ, ngkmax), dtype=complex128, offset=(q,μ,0))`. Contiguous q → single slab; else per-row read + `jnp.concatenate` (host-slow diagnostic path). |
| `_read_r_space` | 565–611 | Same pattern for `'zeta_q'` shape `(Q, n_rtot, μ)`, offset `(q, 0, μ)`. |
| `_is_contiguous` | 618–621 | `all(diff == 1)`. |

### Arrays crossing the boundary
- r_space output: `(Q, n_rtot, n_rmu)` complex128, device, sharded `P(None, None, ('x','y'))`.
- G_flat disk output: `(Q, μ, ngkmax_zeta)` complex128, sharded `P(None, ('x','y'), None)`; pad
  slots `j ≥ ngk[q]` zero by writer construction.
- G_flat-from-r_space output: `(Q, μ/p_prod, n_G_sph)` via `_do_disk_to_G` (see zeta_reader).
- Host-side tables in unfold: `full_to_irr_idx/sym (n_q_full,)`, `r_perm (n_sym, n_rtot)`,
  `inv_mu (n_sym, n_rmu)` int32, moved to device as jit constants.

### Flags / config consumed
None directly. No `LorraxConfig` / cohsex.in keys. `backend=` kwarg exists but see weird_code.

### I/O
Reads `zeta_q.h5` (HDF5): groups `mf_header` (full BGW WFN.h5-style header), `isdf_header`
(see file 3), dataset `zeta_q` `(n_q_disk, n_rtot, n_rmu)` complex128 or `zeta_q_G`
`(n_q_disk, n_rmu, ngkmax)` complex128. Writes nothing.

### Suspects
dead:
- `ZetaLoader` itself has **no production callers** (grep `ZetaLoader(` in src excluding its own
  file: zero hits; only tests). It exists as the intended post-refactor API but production
  (`gw_init.py:1004, 1092`) still constructs `ZetaReader`.
- `slab_io` property (221): no callers found (grepped `.slab_io` in src/tests/tools/scripts;
  only hits are `config.backend.slab_io`, a different thing).
- `backend='eager'|'phdf5'` machinery incl. `_auto_pick_backend` (193): the chosen backend is
  never used — SlabIO is constructed without a backend kwarg and "autoselects"; the loader's own
  comment (184–188) calls the parameter "advisory". Dead knob.

redundancy:
- Whole-class parallel to `ZetaReader` — the classic old/new dual path this codebase's rules ban.
  ~70 lines of attribute mirroring in `__init__` (117–166) are copy-pasted verbatim from
  `zeta_reader.py:85–133`. `_read_r_space`/`_read_g_flat_disk` duplicate `read_zeta_r_slab`/
  `read_zeta_G_slab` semantics. `v_q_g_flat._make_read_all_ibz` carries duck-typing shims to
  tolerate both. Refactor should collapse to one reader.
- `_unfold_q_full_bz` duplicates unfold intent with the production post-V_q unfold
  (`gw.v_q_tile._unfold_v_q_ibz_to_full`, referenced in zeta_reader docstring line 34) — two
  places in the codebase can expand IBZ→full-BZ, in different objects (ζ vs V_q).

weird:
- Line 291–292 comment "The current writer always produces G-flat on disk" contradicts
  `isdf_header.py` (default layout `'r_space'`, "legacy default") and the code path right below
  it that handles r-space disk; likely stale comment from a mid-migration state.
- Hard-coded source line citation `isdf_fitting.py:1689` (line 444) — brittle cross-reference;
  isdf_fitting's zeta writer is now near line 2246+.
- TRS `NotImplementedError` (433–440): intentional loud guard for the known TRS-blind-sym-bug
  family; unfolding with TR-requiring wedges is unsupported here (workaround text: regenerate
  IBZ with TR off or unfold post-V_q).
- `_ensure_sym` builds `SymMaps(self)` by passing the loader itself as a fake WFNReader —
  works only because the attribute mirror is complete; a silent contract.

---

## 2. `src/file_io/zeta_reader.py` — `ZetaReader` + `_do_disk_to_G`

### Purpose
Production ("reader of record after C3", per `gw_init.py:1089`) reader for `zeta_q.h5`, shaped
like `WFNReader`. Two data methods: legacy r-space slab read, and the G-flat read that either
reads the on-disk `zeta_q_G` directly or performs Bloch-phase + 3D-FFT + G-sphere gather so the
V_q kernel becomes a pure v(K)-multiply + einsum. `_do_disk_to_G` is the shared, module-level
jit-cached r→G transform also used by `ZetaLoader`.

Category: **I/O: zeta_q reader (production, V_q stage input)**.

### Entry points and callers
- `ZetaReader(path, mesh=, backend=, mode=)`:
  - `src/gw/gw_init.py:1004–1012` (bispinor path: 4 readers — charge + 3 transverse μ_L files,
    `backend=cfg.backend.slab_io`), `gw_init.py:1092–1094` (charge V_q path, feeds
    `compute_all_V_q`).
  - Tests: `tests/test_zeta_reader.py`, `tests/test_compute_V_q_bispinor_g_flat.py:149-152, 251-254`.
- `read_zeta_G_slab` <- `gw/v_q_tile.py:1303` (ζ_L) and `:1369` (ζ_R) in the V_q hot loop;
  `gw/v_q_g_flat.py:256` (duck-typed `_make_read_all_ibz`); `tests/test_zeta_reader.py:193, 229`.
- `read_zeta_r_slab` <- **no external callers** (grep across src/tests/tools/scripts excluding
  zeta_reader.py: zero hits). Called only internally by `read_zeta_G_slab`'s legacy r_space branch
  (line 333).
- `_do_disk_to_G` <- `file_io/zeta_loader.py:343-346`; referenced in comments at
  `gw/compute_vcoul.py:1011`, `gw/v_q_bispinor.py:345`, `common/wfn_transforms.py:1256`.

### Function table

| Function | Lines | Role |
|---|---|---|
| `ZetaReader.__init__` | 67–166 | `read_mf_header(path)` + `read_isdf_header(path)`; mirror mf attrs (85–119, identical block to ZetaLoader) and isdf attrs (121–133); probe disk dataset shape directly via h5py (comment 137–139: SlabIO is "write-or-collective-read", so shape metadata is read out-of-band); layout-aware axis bookkeeping — G_flat: `(n_q, n_rmu, n_G_sph)` with `n_rtot_disk` synthesized from fft_grid "for compatibility"; r_space: `(n_q, n_rtot, n_rmu)`, `n_G_sph_disk=None`. Then `SlabIO(path, mode=mode, mesh=mesh, backend=backend)` — **unlike ZetaLoader, backend IS forwarded**. |
| `close` / `__enter__` / `__exit__` | 168–178 | Lifecycle; file held open across reads to amortise FFI open cost (docstring cites `isdf_fitting.py:1656` "historical motivation" — another brittle line ref). |
| `slab_io` (property) | 183–188 | Raw handle escape hatch "while the rest of the V_q stack is migrating to G-flat". No external callers found. |
| `n_rtot` (property) | 190–193 | fft_grid product. |
| `read_zeta_r_slab` | 198–228 | `SlabIO.read_slab('zeta_q', shape=(Q, n_rtot, μ), valid_shape=(Q, n_rtot, valid_mu), complex128, offset=(q,0,μ), partition_spec=P(None,None,('x','y')))`. Trailing μ pad zero-filled via `valid_mu`. |
| `read_zeta_G_slab` | 233–343 | Two branches. (a) disk `zeta_layout=='G_flat'`: direct `read_slab('zeta_q_G', (Q, μ, ngkmax), offset=(q,μ,0), P(None,('x','y'),None))`; `qvec_batch_frac` **ignored** (phase already baked in by writer); if caller passed a `sphere_idx` whose size ≠ `n_G_sph_disk` → `NotImplementedError` ("per-q → shared-sphere scatter via gvec_components not yet wired"; comment 302–309 explains why a single shared `jnp.take` would be per-q wrong on the per-q disk sphere). (b) legacy r_space disk: `read_zeta_r_slab` then `_do_disk_to_G`. Physics of (b): `ζ_G(q,μ,G) = FFT_r[ e^{-2πi q·r} ζ_{q,μ}(r) ]` gathered onto the G-sphere. |
| `_do_disk_to_G` | 354–423 | Module-level so the jit caches across reader instances. Cache dict `_disk_to_G_cache` keyed `(id(mesh_xy), Q, mu_total, nx, ny, nz, n_G_sph, id(sphere_idx))`. Body (`_f`, 403–418): transpose `(0,2,1)` with sharding constraint `P(None,('x','y'),None)` → reshape `(Q, μ, nx, ny, nz)` → `common.wfn_transforms.apply_bloch_phase(z5, qvec_frac, (nx,ny,nz), sign=-1)` (separable per-q phase, scratch `Q·(nx+ny+nz)` instead of legacy 4-D `Q·nx·ny·nz` phase_batch, per docstring 272–277) → `common.fft_helpers.make_sharded_fftn_3d` (μ-sharded 3D FFT) → flatten to `(Q, μ, n_rtot)` → `jnp.take(box, sphere_idx, axis=-1)` sphere gather. in_shardings `(P(None,None,('x','y')), P(None,None))`, out `P(None,('x','y'),None)`. |

### Arrays crossing the boundary
- `read_zeta_r_slab` → `(q_count, n_rtot, mu_count)` complex128 device, `P(None,None,('x','y'))`.
- `read_zeta_G_slab` → `(q_count, μ_per_rank·p_prod?, n_G_sph|ngkmax)` complex128,
  `P(None,('x','y'),None)`. G-flat disk case returns raw padded slab.
- Inputs: `qvec_batch_frac (Q,3)` fractional q in kgrid units (BGW wrapped to (−nk/2, nk/2] ÷ kgrid);
  `sphere_idx (n_G_sph,)` int32 flat-FFT indices (or None = full box).

### Flags / config consumed
No direct config reads. `backend` receives `cfg.backend.slab_io` (a `SlabIOBackend` enum from
`gw_config.py:625`) at the `gw_init.py` call sites — so slab-IO backend choice is the one config
surface flowing through this file.

### I/O
Reads `zeta_q.h5` (same schema as ZetaLoader: `mf_header`, `isdf_header`, `zeta_q` or `zeta_q_G`).
Writes nothing.

### Suspects
dead:
- `read_zeta_r_slab` as public API: zero external callers (grep evidence above); only reached via
  the legacy branch of `read_zeta_G_slab`, which itself is dead if the writer "always produces
  G-flat" (per zeta_loader.py:291 comment). If that comment is accurate, the whole
  r_space branch + `_do_disk_to_G` production path is legacy-only (still exercised via
  ZetaLoader in tests).
- `slab_io` property: no external callers found.

redundancy:
- Full parallel with `ZetaLoader` (see file 1). The ~50-line mf_header attribute mirror block is
  duplicated verbatim in both classes; a shared mixin/helper or a single class would remove it.
- `n_rtot` property duplicated in both classes; `_zeta_dataset_name` selection logic duplicated.

weird:
- `_disk_to_G_cache` keyed on `id(mesh_xy)` and `id(sphere_idx)` (line 384): if a mesh or
  sphere_idx array is garbage-collected and a new object reuses the id, a stale compiled fn with
  the *old* baked-in `sphere_idx` closure would be served (sphere_idx is captured by closure, not
  passed as an argument). Also unbounded cache growth across differing shapes.
- G-flat branch: `sphere_idx` acceptance check compares only **sizes** (`n_G_sph != n_G_sph_disk`,
  line 322); equal-sized but different-content spheres would silently pass and be interpreted as
  the disk's per-q sphere.
- `qvec_batch_frac` silently ignored on the G-flat-disk branch (documented at 299–301 but the
  signature still requires it) — API asymmetry that duck-typed callers must know about.
- `backend=None` untyped parameter (line 71) with no docstring entry.
- Brittle line-number cross-references in docstrings (`isdf_fitting.py:1656`).

---

## 3. `src/file_io/isdf_header.py` — `IsdfHeader` schema + read/write

### Purpose
Defines the ζ-specific metadata group `isdf_header` inside `zeta_q.h5`: which density/Lorentz
vertex the ζ was fit for, centroid positions, completion flag, and the on-disk layout descriptor
(`'r_space'` vs `'G_flat'` with per-q G-sphere tables). Deliberately minimal — everything
derivable from `mf_header` (symmetry, k-grid, FFT grid) is rebuilt at read time via `SymMaps` /
`centroid.orbit_syms`, not stored.

Category: **I/O: zeta_q header schema (read+write)**.

### Entry points and callers
- `IsdfHeader` / `IsdfHeader.build` <- writer: `common/isdf_fitting.py:2293, 2328` (in
  `fit_zeta_to_h5`); tests: `test_mf_isdf_header_roundtrip.py`, `test_per_q_sphere.py:181, 202, 213`,
  `test_zeta_loader.py:31, 109, 138`, `test_zeta_reader.py:37, 143`,
  `test_compute_all_V_q_g_flat.py:83`, `test_compute_V_q_bispinor_g_flat.py:81`.
- `write_isdf_header` <- `common/isdf_fitting.py:2343`; same test files.
- `read_isdf_header` <- `gw/gw_init.py:988` (layout detection: `read_isdf_header(zeta_h5_path).zeta_layout`);
  tests.
- `read_isdf_header_from_file` <- `file_io/zeta_loader.py:107` only.
- `mark_zeta_done` <- `common/isdf_fitting.py:2715` (after final ζ chunk drains);
  `tests/test_zeta_loader.py:80, 83`.

### Function table

| Function | Lines | Role |
|---|---|---|
| `IsdfHeader` (frozen dataclass) | 54–104 | Fields: `density` str ('scalar' \| 'current' \| 'mu_L=<int>' \| 'unknown'); `vertex_mu_L` int (0 = charge γ̃⁰ = I; 1,2,3 = transverse γ̃ⁱ = αⁱ — the bispinor four-density vertices); `r_mu_fft_idx (n_rmu,3)` int32 (primary centroid representation, sym-orbit closure checked against it); `r_mu_crystal (n_rmu,3)` f64 (= idx/FFTgrid, redundant, human-readable); `zeta_is_done` bool (default True for in-memory/test construction); `zeta_layout` str; G-flat extras `gvec_components (n_q,3,ngkmax)` int32 (pad sentinel `(-nx/2,-ny/2,-nz/2)`), `ngk_per_q (n_q,)` int32, `zeta_cutoff_ry` float. Properties: `n_rmu` (96–98), `ngkmax` (100–104, None for r_space). |
| `IsdfHeader.build` | 106–178 | Validating constructor: layout whitelist; `r_mu_fft_idx` shape check; derives `r_mu_crystal = idx / fft_grid`; `zeta_is_done` defaults **False** here (writer path — flipped later by `mark_zeta_done`); G_flat requires all three metadata fields, with shape/consistency checks incl. `max(ngk_per_q) ≤ ngkmax`. |
| `_read_group` | 185–212 | Reads group with legacy defaults: missing `zeta_is_done` → True ("legacy files were written atomically at end-of-fit"), missing `zeta_layout` → 'r_space', G-flat fields optional. |
| `_decode_str` | 215–218 | bytes→utf-8. |
| `read_isdf_header` | 221–224 | Path-opening wrapper. |
| `read_isdf_header_from_file` | 227–229 | Open-handle variant. |
| `write_isdf_header` | 236–268 | Creates `isdf_header` group; **refuses to overwrite** an existing group (caller must delete first). Datasets: `density`, `vertex_mu_L`, `zeta_is_done`, `zeta_layout`, `centroids/r_mu_fft_idx`, `centroids/r_mu_crystal`, and conditionally `gvec_components`, `ngk` (note: field `ngk_per_q` stored under dataset name **`ngk`**), `zeta_cutoff_ry`. |
| `mark_zeta_done` | 271–287 | Idempotent flip of `isdf_header/zeta_is_done` to True (creates dataset if missing); called by writer after last chunk H5Dwrite drains. |

### Flags / config consumed
None.

### I/O
Reads/writes the `isdf_header` group of `zeta_q.h5` (HDF5). Full dataset list in the
`write_isdf_header` row above. `mode='a'` append alongside the pre-copied `mf_header`.

### Suspects
dead:
- `zeta_is_done` **as a restart guard is written but never consumed in production**: grep for
  `zeta_is_done` across src (excluding file_io) hits only the writer comment/call in
  `isdf_fitting.py:2709-2715`; both readers surface it as an attribute (`self.zeta_is_done`) but
  no gw/ code branches on it — only `tests/test_zeta_loader.py` reads it. The advertised
  "restart paths check this flag" (docstring lines 25–32) does not currently exist.

redundancy:
- `r_mu_crystal` is stored on disk despite being exactly `r_mu_fft_idx / fft_grid`
  (self-acknowledged "redundant" at line 24) — harmless but duplicated truth.
- `read_isdf_header` vs `read_isdf_header_from_file` — thin path/handle pair (acceptable, but
  the `fetch_X`/`fetch_X_from_file` shape; ZetaReader uses the path form and reopens the file it
  is about to open again for the shape probe, ZetaLoader uses the handle form in one open).

weird:
- Dataset-name/field-name mismatch: field `ngk_per_q` ⇄ HDF5 dataset `ngk`, colliding conceptually
  with mf_header's per-k `ngk`; easy to confuse when grepping file schemas.
- Docstring drift: module docstring (line 13–15) lists density ∈ {'scalar','current','unknown'};
  the dataclass comment (line 56) adds `'mu_L=<int>'`. One of them is stale.
- Pad-sentinel magic value `(-nx/2, -ny/2, -nz/2)` for gvec_components pad slots (lines 37, 85) —
  convention shared with the WFN.h5-style writer; consumers must know not to treat it as a real
  Miller index.
- `zeta_is_done` default asymmetry: dataclass default `True`, `build()` default `False` —
  intentional (test vs writer ergonomics) but a footgun if someone constructs `IsdfHeader(...)`
  directly on the writer path.

---

## Cross-file observations for the refactor map

1. **ZetaLoader/ZetaReader is an unfinished old→new migration.** ZetaReader is the production
   reader (gw_init, v_q_tile hot loop); ZetaLoader is the intended replacement but is only used
   in tests, and `v_q_g_flat._make_read_all_ibz` carries a duck-typing shim for both. This is
   precisely the parallel-path pattern the sandbox rules prohibit; the refactor should pick one
   surface (presumably `.load`) and delete the other, including the twice-duplicated ~50-line
   mf_header attribute mirror.
2. **The in-loader IBZ→full-BZ ζ unfold (`_unfold_q_full_bz`) is test-only**; production unfolds
   post-V_q (`gw.v_q_tile._unfold_v_q_ibz_to_full`). Two sym-unfold implementations for
   ζ-derived objects = candidate for the "one canonical sym-action helper" consolidation.
3. **Per-q disk sphere vs shared consumer sphere scatter is unimplemented** in both readers
   (NotImplementedError guards at zeta_loader.py:299–311 and zeta_reader.py:322–329); the
   gvec_components table exists on disk for exactly this and is consumed instead by
   `gw/compute_vcoul.py` (lines 743, 758) on the V_q side.
4. **zeta_is_done restart guard is write-only** — either wire the consumer or delete the flag.
