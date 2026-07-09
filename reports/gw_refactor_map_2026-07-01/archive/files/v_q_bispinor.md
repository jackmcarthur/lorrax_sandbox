# src/gw/v_q_bispinor.py — deep-read notes (893 LOC)

Refactor-map deep read, 2026-07-01. Repo: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

## Purpose

Bispinor V_q^{μ_L,ν_L} orchestrator. The bispinor pair density carries a Lorentz
4-vector index from the Pauli decomposition
`n_iσ,jσ′(r) = Σ_{μ_L} ζ_{μ_L,a}(r) ⟨σ|τ^{μ_L}|σ′⟩ C^a_ij` (τ^0 = I, τ^{1,2,3} = Pauli).
In Coulomb gauge the kernel is block-structured:

```
V^{μ_L,ν_L}_q(μ,ν) = Σ_K ζ̄_{μ_L,μ}(K) · t^{μ_L,ν_L}(K) · v(K) · ζ_{ν_L,ν}(K)
t^{0,0} = 1 ;  t^{i,j} = δ_ij − K̂_i K̂_j (transverse projector) ;  t^{0,i} = t^{i,0} = 0
```

Of 16 (μ_L,ν_L) blocks: 6 zero by gauge (never computed), 7 unique kernel calls
(CC + 3 TT diag + 3 TT off-diag upper), 3 Hermitian-redundant (reconstructed by
the reader as `V[j,i] = conj(swapaxes(V[i,j], -1, -2))`). Each tile reuses the
scalar charge-only V_q kernel; tiles stream to per-tile HDF5 datasets so peak
GPU memory ≈ one scalar V_q tile.

Category: **physics: bare-Coulomb V_q stage (bispinor/Breit channel), orchestration + I/O layout**.

## Module-level constants

| Name | Lines | Content |
|---|---|---|
| `UNIQUE_TILES` | 57–61 | 7 tuples: (0,0), (1,1),(2,2),(3,3), (1,2),(1,3),(2,3) |
| `ZERO_TILES` | 64–66 | frozenset of (0,i) and (i,0), i ∈ {1,2,3} — gauge-zero |
| `HERMITIAN_PAIRS` | 70–74 | {(2,1):(1,2), (3,1):(1,3), (3,2):(2,3)} |
| `V_QMUNU_FORMAT` | 76 | `"bispinor_lorentz_v1"` format tag |

Consumers of constants (grep evidence): `tests/test_v_q_bispinor_helpers.py`
(exhaustive bookkeeping tests), `tests/test_compute_V_q_bispinor_g_flat.py`,
comment reference in `tests/test_R_proper_cri3.py:23`.

## Function-by-function

### `tile_dataset_name(mu_L, nu_L)` — lines 79–88
Stable per-tile HDF5 dataset name: `V_qmunu_CC` for (0,0), else `V_qmunu_TT_{mu}{nu}`.
Callers: internal (both orchestrators, reader), `gw_init.py:947` (import),
`tests/test_compute_V_q_bispinor_g_flat.py:33,177`, `tests/test_v_q_bispinor_helpers.py:69-75`.

### `_make_K_cart_on_sphere_fn(sphere_idx, fft_grid, bvec)` — lines 96–119
Geometry helper for the LEGACY shared-sphere path. Builds
`K_cart(q,G) = (G + q) · b` (cartesian Bohr⁻¹) for G on the Coulomb sphere;
returns `jax.vmap` of `get_K_cart: (3,) → (n_G_sph, 3)`, lifted to `(Q, n_G_sph, 3)`.
Uses `fft_integer_axes` from `gw.compute_vcoul`. Indexes flat G-int table by
`sphere_idx` (None ⇒ full grid).
Callers: `compute_V_q_bispinor_to_h5` (line 332); `tests/test_v_q_bispinor_helpers.py:156`.

### `_make_v_per_G_for_tile(*, base_v_per_G_fn, K_cart_batch_fn, mu_L, nu_L, eps_K2=1e-30)` — lines 127–174
LEGACY path per-G weight closure. CC tile: identity wrapper around
`base_v_per_G_fn`. TT tiles: multiplies v(K) by transverse projector
`t = 1 − K̂_i² (i==j)` or `t = −K̂_i K̂_j (i≠j)`, with
`K̂_i K̂_j = K_i K_j / max(|K|², eps_K2)`. Guard `eps_K2 = 1e-30` keeps the K=0
slot finite (gauge-singular limit). Raises ValueError for spatial indices
outside {1,2,3}. Docstring notes: q=0 head handled via `g0_acc` only on the CC
tile; transverse head captured in body integral.
Shapes crossing: input `qvec_np_batch (Q,3)` f64; `v (Q, n_G_sph)` c128;
`K_cart (Q, n_G_sph, 3)` f64; output `(Q, n_G_sph)` c128 (device, inside jit
downstream in v_q_tile).
Callers: `compute_V_q_bispinor_to_h5` (line 383); `tests/test_v_q_bispinor_helpers.py:81,96,117,144`.

### `_make_per_q_v_builder_for_tile(*, mu_L, nu_L, bvec, cell_volume, sys_dim, vcoul_cutoff_ry, bdot=None, eps_K2=1e-30)` — lines 186–236
G-FLAT path equivalent of the above ("Math is the same as `_make_v_per_G_for_tile`" —
docstring's own admission of duplication; the difference is the per-q G-sphere
comes from the ζ writer's `gvec_components` table). Returns
`builder(q_irr_frac, gvec_components) → (n_q, ngkmax) c128`. Calls
`compute_v_q_per_G` (gw.compute_vcoul) for the bare `v(q+G)` — that function
guards `denom_zero` so v(K=0)=0. Einsum (verbatim, line 229):

```
K_cart = np.einsum('ba,qbg->qag', bvec_f, qG_frac)
```

with `qG_frac = q_irr_frac[:, :, None] + gvec_components` (f64, host numpy —
this builder runs on host, unlike the legacy jax closure). Then
`K2 = sum(K_cart², axis=1)`, `K2_safe = where(K2 > 1e-30, K2, 1.0)`,
`Khat_ij = K_cart[:, i] * K_cart[:, j] / K2_safe`, `t = (1−Khat_ij)` or `−Khat_ij`.
Callers: `compute_V_q_bispinor_g_flat_to_h5` (line 607). No test coverage found
by grep for this helper by name in tests (covered indirectly via the g-flat
orchestrator test).

### `compute_V_q_bispinor_to_h5(...)` — lines 244–475  [LEGACY r-space orchestrator]
Streams the 7 unique tiles to HDF5 via the legacy `compute_V_q_tile` primitive
(shared Coulomb sphere, in-kernel FFT era). Per tile:
- Picks `zeta_L/zeta_R` SlabIO handles (charge `zeta_C_io` for μ_L=0, else
  `zeta_T_ios[μ_L−1]`); `same_zeta = (mu_L == nu_L)` ⇒ passes `zeta_R_io=None`.
- `wants_g0 = (0,0)` only; BGW overlay (`bgw_v_grid_overlay_fn`) applied only to CC.
- Calls `compute_V_q_tile(...)` from `gw.v_q_tile` with `chooser_choice=None`,
  `budget_bytes` for the chooser, `timing_label=f"V_q_bispinor[{mu_L},{nu_L}]"`.
- Writes each tile immediately via a fresh per-tile `SlabIO(mode='a')`
  (per-tile open kept as "defense-in-depth against MPI-IO datatype-cache state").
  Writes LOGICAL shape `(n_q_total, n_rmu_L, n_rmu_R)` via `valid_shape`,
  stripping the padded μ extent (`V_acc` comes back padded to mesh product);
  docstring cross-refs the band-axis seam at `common/meta.py:99`.
- CC also writes `V_qmunu_CC_g0 (n_q_total, n_rmu_C)`.
- Post-loop rank-0 h5py write of `v_qmunu_format` dataset (np.bytes_) + JSON
  attrs `unique_tiles`, `zero_tiles`, `hermitian_pairs`; then
  `multihost_utils.sync_global_devices("v_q_bispinor_tile_layout_meta")` in a
  bare try/except.

Key arrays: `V_acc (n_q_total, n_rmu_L_pad, n_rmu_R_pad)` c128 device sharded;
`g0_acc (n_q_total, n_rmu_pad)` c128. Inputs are SlabIO handles (host-backed
streaming reads inside `compute_V_q_tile`).

NO IBZ path: this legacy orchestrator has no `sym`/`centroid_idx`/`use_ibz`
plumbing and no Lorentz mixing — full-BZ only.

Callers: `src/gw/gw_init.py:1056` (dispatch branch when
`read_isdf_header(zeta_h5_path).zeta_layout != 'G_flat'`). No test exercises it:
`tests/test_v_q_bispinor_orchestrator.py` is referenced by
`src/lorrax.egg-info/SOURCES.txt:293` and by the comment at `gw_init.py:926`
but DOES NOT EXIST in `tests/` (verified by ls).

### `compute_V_q_bispinor_g_flat_to_h5(...)` — lines 482–747  [CURRENT G-flat orchestrator]
Same 7-tile loop but via `gw.v_q_g_flat._compute_V_q_g_flat_one_tile` (per-q,
G-chunked contraction; loaders are `ZetaReader` objects, not SlabIO). Extras
over the legacy version:
- **IBZ cascade** (`use_ibz`, `sym`, `centroid_C_idx`, `centroid_T_idx`):
  `_ibz_tables_for(centroid_idx, label)` (nested fn, lines 561–573) calls
  `gw.v_q_g_flat._resolve_ibz_q_list` per centroid set to check orbit closure;
  on failure logs loudly and falls back to full-BZ for that channel's tiles.
  CC tiles gate on the charge closure, TT tiles on the transverse closure —
  independent fallbacks. NOTE: the returned `tables` (`_ibz_C`, `_ibz_T`) are
  discarded; only the `_ok` booleans are used, and the per-tile kernel
  re-resolves the tables internally from `sym=_tile_sym` /
  `centroid_indices=_tile_cent` — the closure check work is done twice.
- **BGW vcoul overlay** (CC only): `_v_builder_with_bgw` (lines 618–630) wraps
  the builder; per q it calls `bgw_v_grid_fn(tuple(q))`, flattens the FFT box
  as `flat = (m0%nx)*ny*nz + (m1%ny)*nz + (m2%nz)`, then
  `v[qi] = np.where(v_at != 0.0, v_at, v[qi])` — i.e. BGW value substituted
  wherever nonzero, LORRAX value kept where BGW grid holds exact 0.0 (zero used
  as a "not present" sentinel).
- **Write-time Lorentz mixing on the TT block** (lines 587, 682–728): when the
  transverse IBZ cascade is active, the 6 unique TT tiles are BUFFERED
  (`tt_buffer: dict[(i,j)] → jax.Array`) instead of streamed; the 3
  Hermitian-redundant lower tiles are synthesized
  (`tt_full_in[(j,i)] = jnp.conj(jnp.swapaxes(tt_buffer[(i,j)], -1, -2))`) as
  inputs; then `common.symmetry_maps.unfold_v_q_bispinor_lorentz(tt_full_in,
  sym_idx=sym.sym_idx_q, R_proper_table=sym.R_proper, mesh_xy)` applies the 3×3
  rotation mixing across tiles; only the 6 upper-triangle outputs are written.
  CC never mixes (γ̃^0 = I invariant). Design note in-code: write-time mixing
  chosen over read-time so `BispinorVqReader.get_tile` stays per-tile.
  The gflat memory model accounts for this buffer at
  `gw/gflat_memory_model.py:610` (`_peak_E_v_q_bispinor_buffer`, citing
  `gw/v_q_bispinor.py:587-728`).
- When IBZ-T is off, TT tiles stream straight to disk per tile (identical
  SlabIO block to the legacy orchestrator), preserving free-between-tiles
  memory behaviour.
- Same rank-0 metadata footer + `sync_global_devices("v_q_bispinor_g_flat_tile_layout_meta")`.

Docstring contradiction (see weird_code): the docstring (lines 519–526) says
bispinor ζ files are always full-BZ "so we don't pass sym/centroid_indices",
but the signature/body DO accept and use them (IBZ cascade added later; stale
docstring).

Callers: `src/gw/gw_init.py:1014` (dispatch when `zeta_layout == 'G_flat'`,
which is what `common/isdf_fitting.py:2317` always writes now);
`tests/test_compute_V_q_bispinor_g_flat.py:154,256`.

Flag plumbing at the gw_init call site (flags consumed indirectly):
`cfg.bispinor`, `cfg.memory.vq_g_chunk_size` → `g_chunk`,
`cfg.backend.slab_io` → `backend`, `bare_coulomb_cutoff` (via
`vcoul_cutoff_ry`), `cfg.paths.centroids_file_current` (transverse centroid
reload), env `LORRAX_FORCE_FULL_BZ` → `use_ibz` gate,
`cfg.head.mc_average_vcoul_body` (legacy branch only, via
`make_v_munu_chunked_kernel`).

### `class BispinorVqReader` — lines 755–893
Uniform read interface over all 16 blocks of `v_q_bispinor.h5` (opens both
legacy and g-flat outputs — same on-disk layout).

| Method | Lines | Role |
|---|---|---|
| `__init__(filename, mesh_xy, backend, use_ffi_io)` | 773–805 | Opens a `SlabIO(mode='r')` (kept open; `self._io.__enter__()` called immediately); reads scalar metadata (`v_qmunu_format`, `kgrid`, `n_rmu_C`, `n_rmu_T`, `n_q_total`) via a plain per-rank h5py handle ("broadcast overhead would dominate" for few-byte reads); raises ValueError on format-tag mismatch. |
| `__enter__` / `__exit__` | 807–811 | Context manager delegating to the SlabIO. |
| `_zero_tile(mu_L, nu_L)` | 813–821 | Materialises sharded zeros `(n_q_total, n_L_pad, n_R_pad)` c128 with `NamedSharding(mesh, P(None,'x','y'))` for gauge-zero tiles. |
| `_tile_shape` | 823–826 | Logical shape lookup (n_rmu_C vs n_rmu_T per axis). |
| `_padded_shape_LR(n_L, n_R)` | 828–843 | Rounds n_L, n_R up to `mesh.shape['x'] * mesh.shape['y']` (product, not per-axis) — mirrors write-side `_round_up_to_mesh` at `v_q_tile.py:1116-1118` and matches ψ-side μ extent from `load_centroids_band_chunked`, so Σ^B's V tile broadcasts against ψ with no further padding. |
| `get_tile(mu_L, nu_L)` | 845–875 | Range-check; ZERO_TILES → `_zero_tile`; HERMITIAN_PAIRS → read companion via `SlabIO.read_slab(..., shape=padded, valid_shape=logical, partition_spec=P(None,'x','y'))` then `jnp.conj(jnp.swapaxes(V, -1, -2))` (Hermitian: V[j,i](q,μ,ν) = V[i,j](q,ν,μ)*); else direct read. Returns sharded jax c128 `(n_q, n_L_pad, n_R_pad)`. |
| `get_g0_CC()` | 877–889 | Reads `V_qmunu_CC_g0` with spec `P(None,'x')`; `except Exception: return None` (bare catch used as "dataset absent" probe — masks real I/O errors). |
| `filename` property | 891–893 | Path accessor. |

Callers: `gw_init.py:1070-1073` (`get_tile(0,0)` + `get_g0_CC` to recover the
scalar CC V_q/G0 for the downstream restart-state writer);
`sigma_x_bispinor.py:52,176,181` (`get_tile(i, j)` loop over 9 transverse
tiles for Σ_X^B); `tests/test_compute_V_q_bispinor_g_flat.py:32`.

## I/O

**Writes** (both orchestrators; identical layout, read interchangeably):
`v_q_bispinor.h5` (path chosen by gw_init: `<zeta_dir>/v_q_bispinor.h5`), HDF5
via SlabIO (PHDF5/FFI or h5py-allgather backend) + a rank-0 h5py footer.

- attrs/scalar datasets: `kgrid` int64[3], `n_rmu_C`, `n_rmu_T`, `n_q_total`
  int64 (via `SlabIO.write_attr` — stored as datasets), `v_qmunu_format`
  fixed-bytes dataset = `"bispinor_lorentz_v1"`, file attrs `unique_tiles` /
  `zero_tiles` / `hermitian_pairs` (JSON strings).
- datasets: `V_qmunu_CC (n_q_total, n_rmu_C, n_rmu_C)` c128;
  `V_qmunu_CC_g0 (n_q_total, n_rmu_C)` c128;
  `V_qmunu_TT_{11,22,33,12,13,23} (n_q_total, n_rmu_T, n_rmu_T)` c128.

**Reads**: same file (BispinorVqReader). Inputs read upstream (by the loaders
passed in, not this module directly): `zeta_q.h5` (charge ζ) and
`zeta_q_mu{1,2,3}.h5` (transverse ζ), via SlabIO (legacy r-space layout) or
ZetaReader (G-flat layout, dataset `zeta_q_G`).

## Cross-module dependencies

- `gw.v_q_tile.compute_V_q_tile` (legacy tile kernel)
- `gw.v_q_g_flat._compute_V_q_g_flat_one_tile`, `gw.v_q_g_flat._resolve_ibz_q_list` (private-underscore imports across modules)
- `gw.compute_vcoul.fft_integer_axes`, `gw.compute_vcoul.compute_v_q_per_G`
- `common.symmetry_maps.unfold_v_q_bispinor_lorentz`
- `file_io.slab_io.SlabIO`; h5py (footer/metadata)
- Callers: `gw.gw_init` (dispatch + CC readback), `gw.sigma_x_bispinor` (reader), `gw.gflat_memory_model` (memory accounting mirrors the tt_buffer)

## Dead suspects

- `compute_V_q_bispinor_to_h5` (legacy r-space orchestrator, lines 244–475):
  only caller is the `gw_init.py:1056` fallback branch for
  `zeta_layout == 'r_space'`, but `common/isdf_fitting.py:2317` is the only
  writer call and it always passes `zeta_layout='G_flat'` — the legacy branch
  is reachable only for pre-G-flat ζ files left on disk. Its named test
  (`tests/test_v_q_bispinor_orchestrator.py`, cited in `gw_init.py:926` and
  `lorrax.egg-info/SOURCES.txt:293`) no longer exists (ls-verified). Greps:
  `grep -rn compute_V_q_bispinor_to_h5 src tests tools scripts` → gw_init only.
  With it die `_make_K_cart_on_sphere_fn` and `_make_v_per_G_for_tile`
  (helpers-tests aside).
- Unused imports: `ExitStack` (line 44) and `Iterable` (line 46) — grep shows
  no other occurrence in the file. ExitStack is a leftover from the refactor
  that produced the `if True:` shim (line 363).
- `print_fn` parameter of `compute_V_q_bispinor_to_h5` (line 258): accepted,
  never referenced in the function body (grep: `print_fn` occurs in this file
  only at 258, 501, 569, 634 — the latter two are in the g-flat function).

## Redundancy suspects

- Legacy vs G-flat orchestrator pair (`compute_V_q_bispinor_to_h5` /
  `compute_V_q_bispinor_g_flat_to_h5`) — the classic parallel old/new path;
  dispatch in gw_init on on-disk `zeta_layout`, writer only emits G_flat.
- `_make_v_per_G_for_tile` vs `_make_per_q_v_builder_for_tile` — same
  projector math, jax-sphere vs numpy-per-q implementations; the latter's
  docstring says "Math is the same as `_make_v_per_G_for_tile`".
- Tile-write SlabIO block (create_dataset + write_slab with
  padded→logical `valid_shape`) copy-pasted 3×: legacy loop (425–446), g-flat
  straight-stream branch (660–681), g-flat post-mix branch (715–726).
- Metadata footer (rank-0 h5py `v_qmunu_format` + 3 JSON attrs +
  sync_global_devices) duplicated verbatim in both orchestrators (459–473 and
  732–746).
- `_ibz_tables_for` computes full IBZ tables via `_resolve_ibz_q_list` just to
  extract the `_ok` boolean; the same tables are recomputed inside
  `_compute_V_q_g_flat_one_tile` for every tile that uses IBZ.

## Weird code

- Line 363 `if True:` — vestigial indentation shim (removed ExitStack context;
  the unused `ExitStack` import corroborates).
- `eps_K2 = 1e-30` (twice, lines 133 and 192) — magic guard for the
  gauge-singular K→0 transverse projector denominator; sits alongside the
  claim that `compute_v_q_per_G` already zeroes v(K=0), so the product is zero
  regardless of t (g-flat), whereas in the legacy path the guard is
  load-bearing for jit/AD.
- BGW overlay sentinel (line 629): `v[qi] = np.where(v_at != 0.0, v_at, v[qi])`
  — exact float `0.0` on the BGW grid means "slot not present"; a physically
  zero BGW v(q+G) would be silently replaced by the LORRAX value.
- Stale docstring in `compute_V_q_bispinor_g_flat_to_h5` (lines 519–526):
  claims bispinor ζ files are always full-BZ and "we don't pass sym /
  centroid_indices", directly contradicted by the IBZ-cascade parameters and
  the `_tile_sym`/`_tile_cent` plumbing below it.
- `get_g0_CC` (line 888): bare `except Exception: return None` used as a
  dataset-existence probe; a genuine read failure (corrupt file, MPI-IO error)
  is indistinguishable from "no head written".
- Bare `try/except Exception: pass` around both
  `multihost_utils.sync_global_devices` calls (469–473, 742–746) — swallows
  sync failures silently.
- The Lorentz-mixing branch (lines 693–728) is the machinery implicated in the
  known bispinor-TT in-plane unfold bug (CrI3 C3 gate: z-tile exact, in-plane
  Σ^B ~23% off vs full-BZ truth; see memory `project_bispinor_tt_noncovariance`).
  Not re-verified here — the mixing itself lives in
  `common.symmetry_maps.unfold_v_q_bispinor_lorentz`; this module supplies the
  Hermitian-synthesized full 3×3 input set and writes only the upper triangle.
- Legacy orchestrator comment (lines 390–395): for TT diagonal tiles the
  auto-allocated `g0_acc` is computed and then discarded ("allocates a g0
  buffer we won't write") — wasted work by design, noted in-code.
- Per-tile SlabIO open/close "defense-in-depth against MPI-IO datatype-cache
  state" (comment lines 351–355) — acknowledged as probably unnecessary after
  the create_dataset drain fix, kept anyway.
