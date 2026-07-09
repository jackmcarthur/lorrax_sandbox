# src/file_io/wfn_loader.py — deep-read notes (2026-07-01)

LOC: 1260. Category: **I/O: ψ(G) wavefunction loader with built-in symmetry unfold + bispinor lift** (I/O + symmetry-machinery hybrid).

## Purpose

Single entry point for loading ψ(G) from `WFN.h5` (BGW HDF5 format). Explicitly designed
(per module docstring P-roadmap, P1–P5) to replace the legacy mess of
`{WFNReader + PhdfWfnReader + SymMaps.get_cnk_fullzone[_batch] + SymMaps.get_gvecs_kfull +
load_wfns.read_Gvecs_to_devices + load_kpoint_fftbox}`. Returns ψ in **G-flat layout**
`(n_k, nb_padded, nspinor, ngkmax)` complex128; FFT-box / r-space transforms live downstream
in `common/wfn_transforms.py`. Two backends: `eager` (host h5py + numpy unfold + device_put)
and `phdf5` (collective parallel-HDF5 FFI read + on-device shard_map unfold). P2 contract:
both backends byte-identical for the same request. P5 (delete WFNReader + PhdfWfnReader)
appears COMPLETE — `PhdfWfnReader` no longer exists anywhere in src; `file_io/__init__.py:22`
aliases `WFNReader = WfnLoader` for back-compat.

## Entry points (grep evidence across src/, tests/, tools/, scripts/)

`file_io/__init__.py` exports `WfnLoader` and aliases `WFNReader = WfnLoader`, so ~30 modules
that do `from file_io import WfnLoader as WFNReader` are really constructing this class
(mostly as a metadata-only header accessor).

- **`WfnLoader.__init__`** <- gw/gw_jax.py:136 (`wfn = WFNReader(config.paths.wfn_file, mesh=mesh_xy)` — THE production top-level construction), centroid/current_density.py:144, centroid/charge_density.py:95, centroid/pivoted_cholesky.py:114, psp/get_DFT_mtxels.py:44, file_io/qp_wfn.py:160, common/wfn_loader_backend_parity_test.py:124-125, tests/test_wfn_loader_eager.py (many), tests/test_wfn_transforms.py:43,51. Metadata-only `WFNReader` alias users: bandstructure/htransform.py, centroid/kmeans_cli.py, bse/bse_io.py (6 sites), gw/compute_vcoul_0d.py:192, gw/kin_ion_io.py:26, common/symmetry_maps.py:16, common/symmetry_test.py:7, gw/gw_jax.py:21, etc.
- **`load`** <- common/psi_G_store.py:~250 (`_populate_from_loader`, k="full_bz", sharding=P(None,('x','y'),None,None), bispinor flag), common/load_wfns.py:30 (single-k), centroid/charge_density.py:96 (k="ibz"), centroid/pivoted_cholesky.py:115 (k="ibz"), common/wfn_loader_backend_parity_test.py:134-143, tests/test_wfn_loader_eager.py.
- **`bands` (iterator)** <- ONLY tests/test_wfn_loader_eager.py:157. No production caller.
- **`gvecs`** <- centroid/current_density.py:145, psp/dft_operators.py:154, psp/get_DFT_mtxels.py:48, file_io/qp_wfn.py:163, parity test, tests.
- **`ngk_valid`** <- centroid/current_density.py:146, psp/dft_operators.py:155, psp/get_DFT_mtxels.py:52, file_io/qp_wfn.py:164, tests/test_wfn_transforms.py:98,466.
- **`get_gvec_nk` (deprecated shim)** <- gw/compute_vcoul_0d.py:209, gw/vcoul.py:171 (and a commented line 75), scripts/checks/w_from_eps0_0d_check.py:227, tests/test_wfn_transforms.py:461, psp/tests/test_dft_hamiltonian.py:125 (via build_vnl_setup needing it).
- **`box_index`** <- common/wfn_transforms.py (via docstrings + `to_box` g_index arg), self (`box_index_dev`).
- **`box_index_dev`** <- common/psi_G_store.py:292, common/load_wfns.py:432; referenced in gw/gflat_memory_model.py:182-199 (memory-model accounting).
- **`AsyncWfnReader`** <- NO production or test instantiation (grep `AsyncWfnReader(` hits only its own docstring line 1193). psi_G_store.py:195-201 comment: "was tried here … overlap_frac = 0.000 … Keep the synchronous path". Dead-by-decision.

## File-level constants / module functions

| Name | Lines | Role |
|---|---|---|
| `KSpec` | 68 | Type alias `Sequence[int] \| Literal["ibz","full_bz"]` |
| `_build_phdf5_clamped_counts` | 71-118 | Per-rank `(world*n_reads, 4)` int64 hyperslab counts table for the phdf5 kchunk-union read; clamps band-axis count to `max(0, min(bands_per_rank, mnband_file - (b_lo + r*bands_per_rank)))` so tail ranks past EOF read 0 bands (H5Dread would otherwise fail). Rank flattening `coord_x = r // p_y, coord_y = r % p_y`. Called by `_phdf5_build` and tests/test_wfn_loader_phdf5_clamp.py. |
| `_HALFALPHA` | 1035 | `0.00364867628215` = α/2 (fine-structure constant, Hartree units). DUPLICATED literal in common/bispinor_init.py:30. |
| `_get_bispinor_lift_jit` | 1038-1073 | `lru_cache`d jit factory keyed on output sharding. Physics: σ·(k+G) lower-component lift. p_cart = `(G + k) @ bvec`; σ·p = `[[pz, px-i·py],[px+i·py, -pz]]`; einsum VERBATIM: `psi_S = halfalpha * jnp.einsum("kgij,kbjg->kbig", sdp, psi_2[:, :, 0:2, :])`; output `concatenate([psi_2, psi_S], axis=2)` → 4-spinor. |
| `_bispinor_lift_kernel` | 1076-1091 | Thin wrapper dispatching to the cached jit. psi_2 (n_k, nb, 2, ngkmax) c128; gvecs (n_k, ngkmax, 3) f64; kvecs (n_k,3); bvec (3,3). |
| `_phdf5_unfold_kernel` | 1106-1172 | `lru_cache`d factory building the jit(shard_map) that converts the FFI's re/im-packed IBZ union read to G-flat ψ. Steps: re/im→c128; `jnp.take(cnk, position_in_reads, axis=2)` (IBZ-union → requested k-set); if unfold: `jnp.where(tr_mask, conj(cnk), cnk)` then `cnk * phase_per_k[None,None,:,:]` then spinor rotation einsum VERBATIM: `cnk = jnp.einsum("kac,bckg->bakg", U_per_k, cnk)`; transpose `(bpr, ns, n_k, ngkmax) → (n_k, bpr, ns, ngkmax)`. in_specs cnk `P(('x','y'),None,None,None,None)`; out_specs `P(None, ('x','y'), None, None)`; `check_rep=False`. |
| `AsyncWfnReader` | 1179-1257 | Background daemon-thread prefetch wrapper over `WfnLoader.load` via `common.async_io.AsyncDispatcher`; `submit/get/drain/close`; worker calls `block_until_ready` before publishing. ZERO callers. |

## WfnLoader methods

| Method | Lines | Role |
|---|---|---|
| `__init__` | 125-235 | Opens h5 file; auto backend pick; copies ALL MfHeader fields onto self (drop-in WFNReader surface: nkpts, nbands, nspinor, ngk, kpoints, energies, occs, sym_matrices, translations, fft_grid, bvec, …). Derived: `atom_crys = einsum('ij,kj->ki', inv(avec).T, atom_positions)`; `nelec = max(ifmax)` (fallback: count occs>0.5); vbm/cbm/efermi from energies. Eager state: slurps `wfns/coeffs` (nb, ns, ngktot, 2) f64 and `wfns/gvecs` (ngktot, 3) into host RAM; builds `_kpt_starts` = cumsum(ngk). Caches: `_gvecs_cache`, `_ngk_valid_cache`, `_gvecs_dev_cache` (device g_index dedup — fixes agent_h §3 Finding 3 replicated-buffer leak, was ~1.3 GB/rank). |
| `close/__enter__/__exit__/__del__` | 238-262 | Closes h5 + phdf5 FFI ctx; broad `except Exception: pass`. |
| `_auto_pick_backend` | 267-289 | eager unless (mesh AND process_count>1 AND ffi lib loadable) → phdf5. |
| `_ensure_sym` / `_sym_wfn_stub` | 294-319 | Lazy SymMaps construction from a `types.SimpleNamespace` stub (avoids circular loader ref). Stub recomputes atom_crys einsum (duplicate of __init__). |
| `_resolve_k` | 321-336 | 'ibz' → (arange(nkpts), unfold=False); 'full_bz' → (arange(sym.nk_tot), True); explicit list = **full-BZ indices**, unfold=True. |
| `_k_cache_key` | 338-341 | ('ibz',) / ('full_bz',) / ('list', tuple). |
| `gvecs` | 346-382 | (n_k, ngkmax, 3) int32 zero-padded G-lists. Unfold path inlines former `sym.get_gvecs_kfull`: `g_rot = einsum('ij,kj->ki', sym_krep, k_gvecs) - Gkk` where Gkk = `sym._get_umklapp_vector(self, nk, sym_idx, kbar, sym_krep)` (BGW umklapp). Cached. |
| `ngk_valid` | 384-397 | per-k logical ngk; unfold path maps through `sym.irr_idx_k`. |
| `get_gvec_nk` | 399-408 | Deprecated single-k IBZ G-list shim "for one release" (vcoul/qp_wfn callers). |
| `box_index` | 413-442 | (n_k, nx, ny, nz) int32 gather table, sentinel=ngkmax for empty cells; delegates to `common.gvec_fft_box.build_g_index_for_fft_box`; strips pad rows first so zero-padded gvecs don't clobber the Γ slot. |
| `box_index_dev` | 444-506 | Cached device_put of box_index, key `("box_index_dev", k_key, id(mesh))`; default sharding `NamedSharding(mesh, P(None,None,None,None))` REPLICATED; passes bare numpy to device_put to avoid all-reduce broadcast. Fixes sphere-idx leak. |
| `_default_sharding` | 511-543 | Returns (NamedSharding\|None, p_band). Default when multi-device: `P(None, ("x","y"), None, None)` band-sharded (GW production layout). p_band = product of mesh axis sizes on band dim (for padding). |
| `_pad_to` | 545-548 | Round n up to multiple. |
| `_ensure_phdf5_static` | 556-646 | Once-per-loader device staging of unfold tables. Per-full-BZ-k: ibz_per_full, sym_idx_per_full, tr_mask (`sym_idx >= ntran`), spinor rotation `U_per`: spatial rows `U_spinor[s]`; TRS rows einsum VERBATIM `np.einsum('ij,kjl->kil', _I_SIGMA_Y, np.conj(U_per_spatial))` (T = iσ_y K rule, refs reports/trs_sym_audit_2026-05-14 Sites #5-#7); τ-phase VERBATIM `phase[nk, :ngk_k] = np.exp(-1j * rotated.astype(np.float64) @ tau)` with `rotated = (sym.sym_mats_k[s] @ g_bar.T).T`, SAME formula for spatial+TRS rows (TRS: sym_mats_k[s] = -S_spatial ⇒ phase = conj of spatial; combined with downstream conj gives per-element rule ψ_full = (iσ_y·conj(U))·conj(ψ_kbar)·conj(phase_spatial)). Notes pre-PR3 TRS-rows-phase=1 bug (non-symmorphic non-inversion bispinor). Opens phdf5 ctx via `ffi.phdf5.open_file(path, mesh, "r")`. All tables REPLICATED. |
| `_phdf5_build` | 648-790 | Collective read + on-device unfold. Computes union of IBZ sources (`np.unique`), `position_in_reads = searchsorted(...)`; hyperslab offsets `[b_lo, 0, kpt_starts[ibz], 0]` on dataset layout (mnband, nspinor, ngktot, 2) f64, kchunk_axis=2; per-rank clamped counts (see helper); calls `ffi.phdf5.read.read_kchunk_union_sharded(ctx, "wfns/coeffs", ..., file_partition_spec=P(("x","y"),None,None,None), count_partition_spec=P(("x","y"),None))`; per-rank read result (bands_per_rank, ns, n_reads, ngkmax, 2) f64; unfold kernel; final `with_sharding_constraint(psi, out_sharding)`. Raises if nb_padded % world != 0 ("loader bug") or b_lo >= mnband. |
| `load` | 793-884 | THE public read. Validates bispinor needs nspinor==2; band range in [0, nbands). Resolves k-set, default sharding, pads nb to p_band. phdf5 branch: build → optional bispinor lift on device. eager branch: `_eager_build` host numpy → optional lift via jnp → device_put(named_sharding) or plain jnp.asarray. Padding contract: band-pad rows zero; G-pad rows zero. |
| `bands` | 889-906 | Band-chunk iterator yielding ((bc_lo,bc_hi), psi). Test-only caller. |
| `_apply_bispinor_lift` | 911-953 | ψ 2-spinor → 4-spinor, physics ψ_S = (α/2) σ·(k+G) ψ_L. kvec table: unfold ? `sym.unfolded_kpts[k_idxs]` : `self.kpoints[k_idxs]`. Matches legacy `common.bispinor_init.get_small_psi_component` byte-for-byte, k-vectorised. |
| `_eager_build` | 958-1021 | Host slab compose. Non-unfold: slice `_coeffs_raw[b_lo:b_hi, :, start:end, :]`, re+1j·im. Unfold: per full-BZ k, delegate to `common.symmetry_maps.unfold_psi(cnk, sym_idx=..., g_kbar=..., sym_mats_k=..., translations=..., U_spinor_spatial=...)`; matches legacy SymMaps.get_cnk_fullzone_batch byte-for-byte. |

## Cross-module deps

- `file_io.mf_header.read_mf_header_from_file` (header parse)
- `common.symmetry_maps` (SymMaps, unfold_psi, `_I_SIGMA_Y`, `_get_umklapp_vector` — a PRIVATE member accessed cross-module at line 377)
- `common.gvec_fft_box.build_g_index_for_fft_box`
- `ffi.phdf5` (open_file, close_file, read.read_kchunk_union_sharded), `ffi.common.ffi_loader.get_lib`
- `common.async_io.AsyncDispatcher` (AsyncWfnReader only)
- jax shard_map / NamedSharding / Mesh

## I/O

- **Reads** `WFN.h5` (BGW HDF5 wavefunction): mf_header groups (via mf_header.py), `wfns/coeffs` (mnband, nspinor, ngktot, 2) float64, `wfns/gvecs` (ngktot, 3) int. Eager backend slurps both fully into host RAM at construction; phdf5 backend re-opens the same file through the collective FFI for hyperslab reads (h5py handle kept open for the loader lifetime regardless).
- **Writes**: nothing.

## Flags consumed

None. No LorraxConfig / cohsex.in keys are read here; backend/mesh/bands/k/sharding/bispinor arrive as constructor + method arguments (gw_jax.py passes `config.paths.wfn_file` and mesh).

## Key arrays crossing the boundary

- `psi` out: `(n_k, nb_padded, nspinor_out, ngkmax)` c128; device; default sharding `P(None, ('x','y'), None, None)` (band-sharded on the 2-D GW mesh) or replicated/host-eager.
- `_coeffs_raw`: `(mnband, ns, ngktot, 2)` f64 HOST (eager backend, full-file resident — the big host cache).
- `box_index_dev`: `(n_k, nx, ny, nz)` int32 REPLICATED device, deduplicated per (k, id(mesh)).
- phdf5 static tables: U_per (nk_full,2,2) c128, phase (nk_full, ngkmax) c128, ibz/sym idx (nk_full,) i32 — all REPLICATED.
- phdf5 read tile: per-rank `(bands_per_rank, ns, n_reads, ngkmax, 2)` f64 sharded `P(('x','y'),None,None,None,None)`.

## Dead suspects

1. **`AsyncWfnReader` (lines 1179-1257)** — grep `AsyncWfnReader(` over src/tests/tools/scripts hits only its own docstring (line 1193). psi_G_store.py:195-201 explicitly documents trying it and reverting (xprof overlap_frac = 0.000). Entire class is dead-by-decision, kept "until … the broader async-reader story comes back". `common/async_io.py`'s docstring cites it as a motivating consumer.
2. **`WfnLoader.bands` iterator (889-906)** — only caller is tests/test_wfn_loader_eager.py:157 (grep `\.bands(` across src/tests/tools/scripts). Advertised in the docstring as "band-chunked iterator for GW driver loops" but the GW driver (psi_G_store) calls `load` per chunk directly.
3. **`get_gvec_nk` (399-408)** — NOT dead (vcoul.py:171, compute_vcoul_0d.py:209, scripts/checks/w_from_eps0_0d_check.py:227) but self-declared deprecated "for one release"; refactor should sweep those three callers.

## Redundancy suspects

1. **Bispinor lift duplicated**: `_get_bispinor_lift_jit`/`_apply_bispinor_lift` here vs `common.bispinor_init.get_small_psi_component` (same math, same magic constant `0.00364867628215` typed twice). The legacy function's only remaining callers are tests (tests/test_wfn_loader_eager.py:170,194 — used as the parity reference). Classic parallel-old/new-path per the P4/P5 plan; bispinor_init version deletable once tests reference the loader.
2. **`atom_crys` einsum computed twice** — `__init__` line 190 and `_sym_wfn_stub` line 315, identical `einsum("ij,kj->ki", inv(avec).T, atom_positions)`.
3. **`__all__` defined twice** — line 65 `["WfnLoader"]` and line 1260 `["WfnLoader", "AsyncWfnReader"]`; second silently overrides the first; the first is stale.
4. `gvecs()` unfold body "inlines the former sym.get_gvecs_kfull" (comment 364-368) — that legacy helper IS deleted from symmetry_maps.py (only tombstone comments remain), so this one is resolved, not redundant. Noted for the map.

## Weird code

1. **Stale hardcoded line references "wfn_loader.py:678"** — the load() bounds check now sits at lines 829-832, but line 740 in this file ("rejects upstream at lines 678-681") and cross-module comments (common/load_wfns.py:453, common/psi_G_store.py:213,243) all cite :678. Hypothesis: comments written pre-edit and never re-anchored; will mislead a refactorer.
2. **Magic constant `_HALFALPHA = 0.00364867628215`** (line 1035) — α/2 hard-coded to 11 digits, duplicated verbatim in common/bispinor_init.py:30; no shared constants module.
3. **`id(mesh)` as cache key** (box_index_dev line 491, `_gvecs_dev_cache`) — comment argues mesh outlives loader in production; if a mesh were GC'd and a new one reused the id, a stale (wrong-mesh) device buffer would be returned. Also `_phdf5_unfold_kernel` lru_cache keys on the Mesh object itself (hash), and `_get_bispinor_lift_jit` lru_cache on NamedSharding — unbounded caches keyed on device objects.
4. **No-op `elif bispinor: pass` branch** in load() lines 856-860 — "Rebuild the sharding for the post-lift shape" comment followed by `pass  # NamedSharding doesn't care about exact dim sizes`. Dead branch kept as documentation; also `ns_out = 4 if bispinor else nspinor` at 852 is computed and never used afterward (only the comment path references it).
5. **TRS phase sign gymnastics** (_ensure_phdf5_static 596-620): deliberately uses the SPATIAL formula `exp(-i (sym_mats_k[s]·G_kbar)·τ)` for TRS rows too, relying on `sym_mats_k[s] = -S_spatial` so it evaluates to the conjugate, which then combines with the kernel's `where(tr_mask, conj, ...)` to give the right per-element rule. Correct per trs_sym_audit_2026-05-14 but a two-site conspiracy (phase table + unfold kernel) that breaks if either half is touched alone.
6. **Cross-module private access**: `gvecs()` line 377 calls `sym._get_umklapp_vector(self, ...)` — a leading-underscore SymMaps method, passing the loader itself as the wfn arg.
7. **Broad exception swallowing in close()/`__del__`** (lines 238-262): `except Exception: pass` on both h5 close and FFI close; `__del__` invoking FFI teardown at interpreter shutdown is fragile.
8. **Eager backend always slurps the whole coeffs dataset at __init__** (line 211) even for the ~25 metadata-only `WFNReader` alias users (bse_io, kin_ion_io, kmeans_cli, htransform, …). For a big WFN.h5 that's a full-file host-RAM copy just to read header fields. Hypothesis: acceptable back-compat crutch from the P5 alias (`file_io/__init__.py:17-21` says "a follow-up commit will sweep them"), but it's a real memory-behavior landmine for the refactor.
9. **Explicit k-list means full-BZ indices, always unfold=True** (_resolve_k line 336) — no way to request an explicit subset of raw IBZ k's; a caller passing IBZ indices as a list silently gets sym-unfolded data.
10. Commented-out legacy call at gw/vcoul.py:75 (`# gq = wfn.get_gvec_nk(iq)...`) — external, but part of the get_gvec_nk deprecation debt trail.
