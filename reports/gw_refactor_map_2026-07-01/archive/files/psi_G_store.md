# src/common/psi_G_store.py — deep-read notes (gw_refactor_map_2026-07-01)

LOC: 475. Read fully 2026-07-01. All line numbers = file as of lorrax_D main HEAD (216d4193 branch state).

## Purpose

Host-resident ψ(G-flat) cache with per-band-chunk `io_callback` slicing, feeding the
r-chunk ISDF ζ-fit pipeline. Replaces the legacy g_box host cache (P4c rewrite): stores
ψ per rank as `(nk, nb_local, ns, ngkmax)` complex128 (~6–11 % of the FFT-box size,
~14× smaller than g_box) and lets consumers pull one band-chunk (bc) per `lax.scan`
iteration inside `shard_map`, so the FFT box is never a persistent device buffer.

Category: **resource mgmt: host ψ(G) cache / io_callback data-staging for ISDF ζ fit**.
No physics equations live here — it is pure data layout, sharding, padding, and
lifecycle plumbing. The FFT/Bloch-phase math it stages inputs for lives in
`common/wfn_transforms.py` (`gflat_to_rchunk`) and `common/isdf_fitting.py`
(`z_q_from_psi_sm` / `c_q_from_psi_sm`).

## Module-level items

| Item | Lines | Role |
|---|---|---|
| module docstring | 1–41 | P4c design note: g_flat host cache + on-demand to_rchunk; lifecycle modes; WfnLoader `backend='auto'` collapse of legacy reader adapters |
| imports | 42–52 | **`partial` (l.44), `io_callback` (l.50), `shard_map` (l.51) are imported but never used in this file** — leftovers from the pre-rewrite version where the fetch kernel lived here. `io_callback`/`shard_map` are now used only by consumers in isdf_fitting.py. |
| `_PSI_G_FLAT_SPEC = P(None, ('x','y'), None, None)` | 58 | Sharding contract for the post-io_callback ψ(G-flat) tile: `(nk, nb, ns, ngkmax)`, band axis flat-sharded over mesh axes (x,y). Consumed (as a documentation import, `# noqa: F401`) by `isdf_fitting.py:535`. The same spec is retyped inline as `sharding_spec` at l.203 of this file rather than referencing the constant. |

## Functions / classes

### `_zero_user_band_pad_in_shard(shard_data, *, bc_range, shard_band_slice, user_band_stop)` — lines 61–92
- Role: zero the band rows of one per-device shard whose *global* band index ≥ `user_band_stop` (= `meta.b_id_4_user`, the user's requested nband before mesh-divisibility rounding to `b_id_4`). Mirrors the centroid loader's contract ("The centroid loader zeros them after extraction; this helper applies the same contract").
- Math: pad-band ψ rows must be exactly 0 so they are math-neutral in the pair-density einsum downstream.
- Details: requires contiguous band slice (`step != 1` → ValueError, l.81–83); computes `local_global_bands = b0 + arange(s0, s1)`; returns input unchanged if no pad owned (no copy), else copies then zeros (`out[:, pad_mask, :, :] = 0.0`).
- Callers: `PsiGStore._populate_from_loader` (l.273); unit tests `tests/test_psi_g_store.py:11,28,41`.

### `_mesh_device_coords(mesh)` — lines 95–101
- Role: build `{id(device): (x_idx, y_idx)}` for every device in the 2-D mesh; used to map `addressable_shards` back to host-tile keys.
- Callers: `PsiGStore.__init__` (l.166); test stub `tests/test_psi_g_store.py:75`.

### class `PsiGStore` — lines 104–403

#### `__init__(self, *, loader, mesh_xy, band_chunk_ranges, meta, bispinor=False)` — 122–175
- Stores loader (a `file_io.wfn_loader.WfnLoader`), 2-D mesh, bc ranges, `meta`, bispinor flag.
- Layout math: `p = mesh.shape['x'] * mesh.shape['y']`; per-bc local band count `bpd_per_bc = (b_hi - b_lo) // p`; `_bpd_max = max(bpd_per_bc)` (the static io_callback out-shape pad, restored in "Round 6 Phase 2", originally commit `cdd0fba`, deleted in `5cadd4b` per comment l.150–157); `_bc_band_offsets` = cumulative offsets → per-rank tile band axis is bc-stacked; `_per_rank_shape = (nk, nb_local, ns, ngkmax)`.
- `_host_tiles: dict[(x,y) -> np.ndarray]` host numpy tiles; `_g_index_dev` / `_kvecs_frac_dev` lazily staged device arrays.
- dtype hard-wired `jnp.complex128` / `np.complex128`.

#### `_populate_from_loader(self)` — 180–300
- Role: one `loader.load(bands=bc, k="full_bz", sharding=P(None,('x','y'),None,None), bispinor=...)` per band-chunk; `block_until_ready`; then per-shard host copy into `_host_tiles[(x,y)][:, b_lo:b_hi]` after `_zero_user_band_pad_in_shard`.
- Past-EOF band-pad handling (l.205–267): when `world_size` rounds `b_id_4` past the file's `mnband` (concrete case documented: CrI3 6×6 30 Ry SOC, mnband=86, world_size=16 ⇒ b_id_4=96), caps the load at `bc_end_in_file = min(bc_end, file_nbands)`; if a whole bc is past EOF, writes zeros to the tile span directly and skips the loader call (which would reject `b_hi > nbands` at `file_io/wfn_loader.py:678`); if partially past, zero-pads on device via `jnp.concatenate` + `with_sharding_constraint` (l.256–267).
- Timing sections: `psi_G_store.populate.loader_load`, `psi_G_store.populate.shard_to_host` (via `common.timing`).
- Dead-end note (l.195–201): AsyncWfnReader depth-2 prefetch was tried; xprof showed H2D/compute `overlap_frac = 0.000` at MoS2 3×3 scale; synchronous path kept.
- Staging (l.284–300): once per store, stages `_g_index_dev = loader.box_index_dev(k="full_bz", mesh=...)` — the loader-level dedupe is explicitly to fix the agent_h §3 Finding 3 leak (4 fit channels × 0.16 GB/rank replicated g_index ≈ 1.3 GB/rank wasted); and `_kvecs_frac_dev = sym.kvecs_asints / kgrid` replicated `(nk,3)` f64 via `device_put(..., P(None,None))`. Uses the loader's *private* `_ensure_sym()` (l.295).
- Callers: `HostPsiGStore.__init__` (once), `RereadPsiGStore.begin_rchunk` (per r-chunk).

#### `_clear_tiles(self)` — 302–303
- `self._host_tiles.clear()`. Called by `close()` and `RereadPsiGStore.end_rchunk`. Note it does NOT drop `_g_index_dev`/`_kvecs_frac_dev` (device buffers persist; g_index is deduped at the loader anyway).

#### `_slice_local_tile_bc(self, x_idx, y_idx, bc_idx)` — 327–364 (plus design comment block 305–325)
- THE hot entry point: per-iter host-tile slicer invoked from inside consumers' `io_callback` within a `lax.scan` body under `shard_map`.
- Returns `(nk, _bpd_max, ns, ngkmax)` c128: first `bpd_per_bc[bc]` rows are real data, rest zero. `np.zeros` deliberately (NOT `np.empty`) — comment l.322–325: pad rows must be EXACTLY zero or garbage "could still pollute IFFT precision" even though the L/R band mask zeros them at the einsum.
- Args are `jax.lax.axis_index('x')/('y')` int32 scalars + traced bc_idx, resolved to Python ints host-side. Bounds check on bc (l.354–357).
- Lifetime contract (docstring l.346–351): host tiles must outlive the enclosing jit; `RereadPsiGStore.end_rchunk` runs after `block_until_ready` in `isdf_fitting.py`'s `finally:` clause, so async callbacks complete before tiles are freed.
- Callers (grep across src/tests/tools/scripts): `common/isdf_fitting.py:645` (the io_callback host fn inside `z_q_from_psi_sm`/`c_q_from_psi_sm` streaming pipeline; also referenced at :517, :1724, :1831); AOT memory-model stub replicates its signature at `src/gw/aot_memory_model/kernels/fit_one_rchunk.py:112` (`_AotStubPsiGStore`); tests `tests/test_psi_g_store.py:95,128,148`.

#### `g_index` property — 366–377
- Replicated `(nk_tot, nx, ny, nz)` int32 box-index device tensor; raises RuntimeError if accessed before staging. Consumer: `isdf_fitting.py:856` (`psi_G_store.g_index` passed as jit arg into the pair-pipeline shard_map kernel, feeding `gflat_to_rchunk`).

#### `kvecs_frac` property — 379–386
- Replicated `(nk_tot, 3)` f64 fractional k-vectors (Bloch phase ingredient). Consumer: `isdf_fitting.py:856`.

#### `begin_rchunk(r_start, r_end)` / `end_rchunk()` — 391–399
- Lifecycle hooks called by the Python r-chunk driver (`isdf_fitting.py:2561` / `:2596`). Base-class no-ops (host_cache mode). `r_start`/`r_end` args unused even by the override.

#### `close()` — 401–403
- `_clear_tiles()` only. Docstring says "drop the loader reference" but the code never does `self.loader = None` — docstring/code mismatch. Caller: `isdf_fitting.py:2720`.

### class `HostPsiGStore(PsiGStore)` — 405–427
- `__init__` (415–424): populate once at construction, print resident GB on rank 0. Footprint math in docstring: CrI3 80 Ry 6×6, ngkmax≈70k, 1000 bands, 4 spinor, 16 GPU ⇒ ~28 GB/process (g_box form ≈ 400 GB, won't fit).
- `_per_rank_shape_bytes()` (426–427): `prod(shape) * 16`.
- Callers: `build_psi_G_store` only.

### class `RereadPsiGStore(PsiGStore)` — 430–437
- `begin_rchunk` → `_populate_from_loader()`; `end_rchunk` → `_clear_tiles()`. Zero persistent residency between r-chunks.
- Callers: `build_psi_G_store` only.

### `build_psi_G_store(*, wfn, sym, mesh_xy, meta, band_chunk_ranges, bispinor=False, mode="host_cache")` — 440–475
- Factory. `mode` values `"host_cache"` / `"file_reread"` map to the two subclasses; anything else → ValueError.
- **`sym` param is dead**: kept "for caller-API back-compat but is ignored" (`del sym`, l.461) — loader builds its own SymMaps lazily.
- `loader = wfn` — reuses the caller's top-level WfnLoader (opening a second one would re-slurp coefficients into host RAM).
- Sole caller: `common/isdf_fitting.py:2433–2438` (`fit_zeta_to_h5` r-chunk driver), with `mode=gspace_mode`.

## Entry points and call graph

- `build_psi_G_store` <- `common/isdf_fitting.py:2433` (only production caller).
- `PsiGStore._slice_local_tile_bc` <- `common/isdf_fitting.py:645` (io_callback host fn), stubbed by `gw/aot_memory_model/kernels/fit_one_rchunk.py:_AotStubPsiGStore:112`, tested in `tests/test_psi_g_store.py`.
- `.g_index`, `.kvecs_frac` <- `common/isdf_fitting.py:856`.
- `.begin_rchunk`/`.end_rchunk` <- `common/isdf_fitting.py:2561/2596`; also stub no-ops in `fit_one_rchunk.py:108–109`.
- `.close` <- `common/isdf_fitting.py:2720`.
- `_zero_user_band_pad_in_shard`, `_mesh_device_coords`, `PsiGStore` <- `tests/test_psi_g_store.py`.
- `_PSI_G_FLAT_SPEC` <- `common/isdf_fitting.py:535` (F401 documentation import only).
- Grep scope used: `grep -rn "psi_G_store|PsiGStore|build_psi_G_store|_slice_local_tile_bc|begin_rchunk|end_rchunk|_PSI_G_FLAT_SPEC|_zero_user_band_pad_in_shard|_mesh_device_coords" src tests tools scripts`.

## Flags consumed

- No direct LorraxConfig access in-file. `mode` comes from cohsex.in key `gspace_mode` (values `host_cache` | `file_reread`), parsed at `gw/gw_config.py:1035` into `GspaceIO` enum (gw_config.py:104, default `"host_cache"` at :218, legacy `.gspace_mode` view at :824–826), threaded through `isdf_fitting.fit_zeta_to_h5(gspace_mode=...)` (:1853) to `build_psi_G_store(mode=...)`. Documented in `docs/theory/isdf-zeta-vq.md:143–145`.

## I/O

- No direct file I/O. All reads delegated to `file_io.wfn_loader.WfnLoader.load()` (WFN.h5 wavefunction coefficients, eager-h5py or FFI-phdf5 backend chosen by `backend='auto'`) and `loader.box_index_dev()` (g_index from the WFN gspace group). Writes nothing.

## Key arrays crossing boundaries

| Array | Shape / dtype | Residency | Sharding |
|---|---|---|---|
| `_host_tiles[(x,y)]` | `(nk, nb_local, ns, ngkmax)` c128 | host numpy, one per locally-addressable device | band axis bc-stacked, flat-sharded over (x,y) implicitly by construction |
| `_slice_local_tile_bc` output | `(nk, _bpd_max, ns, ngkmax)` c128 | host → device via consumer's io_callback | per-rank slab; downstream `all_gather` along band axis inside shard_map |
| `psi_G_bc` (transient in populate) | `(nk, bc_width, ns, ngkmax)` c128 | device, freed per bc (`del` l.280) | `P(None, ('x','y'), None, None)` |
| `g_index` | `(nk_tot, nx, ny, nz)` int32 | device, deduped at loader level | replicated |
| `kvecs_frac` | `(nk_tot, 3)` f64 | device | replicated `P(None, None)` |

## Suspects

### dead_suspects
- Unused imports `partial` (l.44), `io_callback` (l.50), `shard_map` (l.51): grepped the file body — none of the three identifiers appears after the import lines. Leftovers from the pre-P4c version where the fetch kernel lived in this module.
- `sym` parameter of `build_psi_G_store` (l.444, `del sym` l.461): explicitly dead, kept for caller-API back-compat; the single caller (isdf_fitting.py:2434-2438) still passes it. Refactor: drop from both ends.
- `begin_rchunk(r_start, r_end)` arguments: neither the base no-op nor `RereadPsiGStore`'s override uses `r_start`/`r_end` — the reread repopulates ALL bands for every r-chunk regardless of the window. Signature suggests a per-rchunk partial read that was never (or no longer) implemented.

### redundancy_suspects
- `_PSI_G_FLAT_SPEC` (l.58) vs inline `sharding_spec = P(None, ('x','y'), None, None)` (l.203): same spec written twice in the same file; the exported constant is only ever imported for documentation (`# noqa: F401` in isdf_fitting.py:535).
- `_AotStubPsiGStore` in `gw/aot_memory_model/kernels/fit_one_rchunk.py:72` duplicates the PsiGStore layout math (`bpd_per_bc`, bc-stacked offsets, `_bpd_max` padding, comment l.92 "PsiGStore layout (host tile bc-stacked over P ranks)") — a parallel reimplementation that must be kept in sync by hand; a refactor could have the stub subclass PsiGStore the way `tests/test_psi_g_store.py:_FakePsiGStore` does.
- `HostPsiGStore` vs `RereadPsiGStore`: not redundant per se (documented lifecycle modes behind cohsex.in `gspace_mode`), but `RereadPsiGStore` is a full-repopulate-per-rchunk path whose distinct value vs host_cache is only the residency window; worth checking run configs for whether `file_reread` is ever exercised outside tests.

### weird_code
- l.150–157 & 305–325: archaeology comments citing raw commit hashes (`cdd0fba` added, `5cadd4b` deleted, "Round 6 Phase 2 restoration", "Round 5 unified plan §3.2/§6.5") — a helper that was deleted for a "(now-buggy) flat-axis `psi_G_device_full` path" then restored. Signals prior churn; the referenced `psi_G_device_full` path no longer exists in this file.
- l.322–325 + l.362: `np.zeros` (not `np.empty`) is load-bearing for correctness — pad rows must be exactly zero because the L/R mask "could still pollute IFFT precision"; easy to break in a refactor chasing the allocation cost.
- l.229–246: whole-bc-past-EOF branch writes zeros into `[:, b_lo:b_hi]` where the comment simultaneously claims `b_hi - b_lo == 0` for such bcs ("bpd_per_bc[bc] == 0 since nb_total < p") yet the code guards `if b_hi - b_lo > 0` and zero-fills anyway; also the comment "host_tile entries already pre-zero via np.empty" is self-contradictory (np.empty does NOT zero) — the explicit zero-fill is what actually protects correctness. Hypothesis: comment rot from two rounds of edits to the pad path.
- l.277: `getattr(self.meta, "b_id_4_user", self.meta.b_id_4)` — soft fallback for a Meta attribute that may not exist on older Meta objects; silent behavior fork.
- l.295: `sym = self.loader._ensure_sym()` — reaches into WfnLoader's private API from outside the class.
- l.401–403: `close()` docstring promises "drop the loader reference" but code only clears tiles; loader (and its host coefficient cache) stays referenced.
- l.195–201: dead-experiment note (AsyncWfnReader prefetch, overlap_frac 0.000) — the AsyncWfnReader presumably still exists in `file_io/wfn_loader.py` with no production caller from here.
- l.422–424: rank-0 `print` for the resident-GB banner instead of the logging/timing infra used elsewhere in the file.
- Hard-wired complex128 (l.165, 193) — no single-precision option; relevant if the refactor considers mixed precision.

## Cross-module deps

- `common/isdf_fitting.py` (sole production consumer; drives lifecycle, io_callback, g_index/kvecs_frac).
- `file_io/wfn_loader.py` (`WfnLoader.load`, `.box_index_dev`, `.ngkmax`, `.nbands`, private `._ensure_sym`; loader-level g_index dedupe at wfn_loader.py:225–229 exists specifically for this store).
- `common/timing` (sections).
- `common/wfn_transforms.py` (consumer of the staged g_index via gflat_to_rchunk; comment cross-refs at wfn_transforms.py:88,108,551).
- `gw/gw_config.py` (`GspaceIO` enum / `gspace_mode` flag), `gw/aot_memory_model/kernels/fit_one_rchunk.py` (stub twin), `gw/gflat_memory_model.py` (memory-model comments referencing the historical g_index leak).
- Tests: `tests/test_psi_g_store.py`, `tests/test_band_chunk_size_floor.py` (layout invariant), `tests/test_planner_refit_2026-05-17.py` (leak regression note).
