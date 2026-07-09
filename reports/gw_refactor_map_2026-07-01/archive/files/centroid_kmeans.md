# Centroid k-means group — deep-read notes (refactor map 2026-07-01)

Files: `src/centroid/kmeans_isdf.py` (995 loc), `src/centroid/kmeans_cli.py` (427 loc),
`src/centroid/kmeans_plot.py` (87 loc), `src/centroid/centroid_io.py` (55 loc).
All paths relative to `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.

This is the ISDF **sampling-point selection** stage: pick N_c real-space FFT-grid points
(centroids) weighted by ρ(r) (or the Gordon Pauli-current j² for the bispinor transverse
channels), optionally symmetry-orbit-aware, optionally pruned by pivoted Cholesky on
pair-density Gram matrices. Output is a text file of fractional coordinates consumed by
`gw.gw_jax` via `file_io/centroids.py:load_centroids`.

---

## src/centroid/kmeans_isdf.py (995 lines)

**Purpose.** Density-weighted Lloyd k-means with exact periodic-boundary (min-image)
distances under the lattice metric G = avec·avecᵀ, JAX-jitted, single-device and
shard_map-sharded variants, plus a symmetry-orbit-aware variant where centroids are orbit
*representatives*. Algorithm module only — no file I/O, no matplotlib.

**Category:** preprocessing tool (ISDF sampling-point selection, numerical/geometric, no physics equations beyond the metric quadratic form).

Sets `jax_enable_x64=True` at import. Constant `BOHR_TO_ANG = 0.529177210544` (line 36).

### Function table

| Function | Lines | Role |
|---|---|---|
| `build_min_image_offsets(metric_tensor, search_radius=1)` | 43–79 | Host/numpy startup: greedy set-cover over `{-R..R}³` integer offsets so that `min_n (df+n)ᵀG(df+n)` over df∈[-0.5,0.5]³ (11³ probe grid) matches brute force. Returns (M,3) int32, identity first. M=1 cubic, 5 hex-2D, 7 FCC. Raises RuntimeError if R too small. Einsum verbatim: `np.einsum("koi,ij,koj->ko", df_all, G, df_all)` (line 60). Callers: `weighted_kmeans_jax` (line 831), tests/test_kmeans_sharded.py (lines 517, 530, 547, 578). |
| `_quadform_G(dx,dy,dz,g00..g22)` | 82–86 | Fused d² = δᵀGδ, avoids K=3 GEMM. Used by every distance kernel. |
| `pbc_distance_sq_single(positions_frac, centroid_frac, metric, offsets)` | 89–111 | jit. (P,) min-image d² to ONE centroid; fori_loop over offsets. Callers: `_orbit_distance_sq` (line 711, i.e. kmeans++ init path) and tests (558, 678). |
| `assign_labels_chunked(positions, centroids, metric, n_c, c_block=32, offsets)` | 118–176 | jit(static n_c, c_block). (P,) int32 nearest-centroid labels. lax.scan over C-chunks of `c_block` (NaN-pads centroids to a multiple; NaN propagates through the offset fori_loop, masked to inf once before argmin so padding never wins). Peak buffer (P, c_block). Callers: `kmeans_update_step`, `_kmeans_step_shardmap_body`, `weighted_kmeans_jax` final labeling (line 900), tests. |
| `_orbit_d2_chunk(positions, image_chunk, metric, offsets)` | 183–200 | (P, c_block) min-image d² to a block of centroid *images*; same fori-over-offsets pattern (copy of assign_labels body's distance part). |
| `_orbit_d2_per_point(positions, image_per_point, metric, offsets)` | 203–218 | (P,) variant, one image per point (tie-mask pass). |
| `assign_labels_orbit_chunked(positions, reps, metric, n_rep, c_block, offsets, Rinv, tau, tie_tol=1e-10)` | 221–314 | jit(static n_rep, c_block). Orbit-aware assignment: pass 1 fori(sym)∘fori(offset) orbit distance per (point, rep-chunk) with image `rep_chunk @ Rinv[s].T + tau[s]`; pass 2 rebuilds (P, n_sym) bool tie mask for the winning rep only (`d2_s <= local_d + tie_tol`). Returns (labels, best_d2, tie_mask). Defaults Rinv=identity[None] ⇒ reduces to the plain version. Callers: `_orbit_step_shardmap_body`, `weighted_kmeans_jax` final labeling (line 894). |
| `_min_image_delta(delta, metric, offsets)` | 321–341 | (...,3) minimising lattice image of a displacement vector (keeps the vector, not just d²). |
| `_local_update_accumulators(positions, centroids, rho, labels, n_c, metric, offsets)` | 344–353 | Lloyd accumulators via two `jax.ops.segment_sum`: Σ_p ρ_p·δ_p and Σ_p ρ_p per centroid. O(P+C) memory. |
| `_orbit_local_update_accumulators(positions, reps, rho, labels, tie_mask, n_rep, metric, offsets, R, Rinv, tau)` | 356–398 | Orbit accumulator: for each tied sym op s, δ_image = min-image(x_p − (r_μ@Rinvᵀ+τ)), fold-back **δ_rep = δ_image @ R[s].T**, weight w_s = ρ_p / n_tied(p) on tied ops. Projects onto stabiliser-invariant subspace at special positions. fori over n_sym, two segment_sums each. |
| `_finalize_update(centroids, sum_wd, sum_w, metric, offsets)` | 401–413 | new = (c + Σwδ/max(Σw,1e-10)) mod 1; movement² = `jnp.einsum('ci,ij,cj->c', move, metric_tensor, move)` (line 412) with min-image move. |
| `_orbit_finalize_update(reps, sum_wd, sum_w, metric, offsets, Rinv, tau)` | 416–438 | Same + vmap'd canonicalisation of each new rep (`_canonicalize_rep`) so reps don't hop between orbit members; movement² old-rep→canonical-new-rep, einsum verbatim `jnp.einsum('ci,ij,cj->c', move, metric, move)` (line 437). |
| `kmeans_update_step(positions, centroids, rho, metric, n_c, c_block, offsets)` | 441–465 | jit. One Lloyd iteration, single device. Callers: **tests only** (test_kmeans_sharded.py 86, 137, 385). NOT called by the driver. |
| `_kmeans_step_shardmap_body(...)` | 472–486 | One Lloyd step inside shard_map: local assign + accumulate, `lax.psum` on (C,3) and (C,) accumulators, finalize. Used by `make_sharded_kmeans_update` and `make_sharded_lloyd_loop`. |
| `_orbit_step_shardmap_body(...)` | 489–507 | Orbit sibling; same psum pattern. |
| `make_sharded_kmeans_update(mesh, n_c, c_block, mesh_axis='x', offsets)` | 510–553 | Builds jitted single-step sharded update. Docstring: "kept for direct tests / reuse". Callers: **tests only** (test_kmeans_sharded.py 147, 394). |
| `make_sharded_lloyd_loop(mesh, n_c, max_steps, tolerance, offsets, c_block, mesh_axis)` | 560–620 | Full Lloyd loop as ONE `lax.while_loop` inside ONE `shard_map` — zero host syncs per iteration. in_specs: positions (P-sharded on mesh_axis), centroids/metric replicated, rho sharded. Returns (centroids, steps, max movement²). tol/offsets closed over. Caller: `weighted_kmeans_jax`. |
| `make_sharded_orbit_lloyd_loop(mesh, n_rep, max_steps, tolerance, offsets, R, Rinv, tau, ...)` | 623–684 | Orbit sibling, identical sharding; reps + (R,Rinv,τ int32/f64) replicated. Caller: `weighted_kmeans_jax`. |
| `_gumbel_sample_argmax(log_weights, key)` | 691–698 | One categorical draw via Gumbel-max; u clamped to (tiny, 1−tiny). |
| `_orbit_distance_sq(positions, rep, metric, offsets, Rinv, tau)` | 701–715 | (P,) d² to the ORBIT of one rep; image = `(rep @ Rinv[s].T + tau[s]) % 1.0`; fori over n_sym calling `pbc_distance_sq_single`. n_sym=1 identity reduces to the plain single-centroid distance. |
| `_canonicalize_rep(rep, Rinv, tau)` | 718–722 | Lex-smallest orbit member; thin wrapper importing `orbit_syms.canonicalize_orbit` (deferred import inside function body). |
| `kmeans_pp_init(positions, rho, metric, n_c, key, offsets, Rinv, tau)` | 725–778 | jit(static n_c). Density-weighted k-means++: first ~ Categorical(ρ), then Categorical(D²·ρ) with D² = orbit distance when Rinv/τ given; new centroids canonicalised in orbit mode. fori_loop over n_c draws (O(n_c·P)). Callers: `weighted_kmeans_jax`, tests (200, 201, 234, 286). |
| `_pick_c_block(n_c, preferred=32)` | 785–788 | min(preferred, n_c). |
| `weighted_kmeans_jax(avec, rho, N_c=10, max_steps=200, tolerance=5e-3, seed=0, *, mesh, mesh_axis='x', init_method='kpp', R, Rinv, tau)` | 791–905 | Driver. metric = avec@avecᵀ (avec in Å from the CLI, so `tolerance=5e-3` is in Å); builds offsets table; builds fractional (P,3) grid positions from rho.shape; kpp or density-weighted-random init; device_puts with NamedSharding; dispatches orbit vs plain lloyd factory; final label assignment. Returns (labels, centroids, steps, max_movement_sq). Callers: `centroid.kmeans_cli.main` (line 293), tests (468, 482, 640), tests/archive/test_kmeans.py (line 24 — stale, see below). |
| `snap_centroids_to_grid(centroids_frac, fft_grid, deduplicate=True)` | 912–931 | Host: round to nearest FFT-grid index mod grid; optional np.unique dedup. Returns (indices, frac, n_dups_removed). Callers: kmeans_cli (310, 323, 329), tests. |
| `ensure_unique_centroids(centroids_frac, fft_grid, rho=None)` | 934–962 | Host: snap; duplicates redistributed onto highest-ρ unoccupied grid points via flat-index set arithmetic; drops overflow with a warning. Caller: kmeans_cli (335) — only in the non-orbit dedup branch. |
| `_decide_init_method(n_c, n_rtot, threshold=0.10)` | 977–984 | kpp → random when N_c > 0.10·n_rtot (kpp is O(N_c·P) and unnecessary in the dense regime). Callers: kmeans_cli (217), tests (414–435). |
| `_warn_dense_grid_regime(m_cand, n_c, n_rtot, threshold=0.25)` | 987–995 | Warning string when N_c > 0.25·n_rtot: "pivoted Cholesky on the full grid would be the right algorithm here but is not implemented". Note: `m_cand` parameter is **unused** in the body. Callers: kmeans_cli (220), tests (442, 448). |

### Key arrays / boundary
- `positions` (P,3) f64 fractional, P-sharded on mesh_axis; `rho_flat` (P,) f64 sharded;
  `centroids/reps` (N_c,3) f64 replicated; `metric` (3,3) replicated; offsets (M,3) int32
  closure constant; R/Rinv (n_sym,3,3) int32, tau (n_sym,3) f64 replicated.
- Everything device-resident during the Lloyd while_loop; output centroids pulled to host by the CLI.

### Flags consumed
None directly (no LorraxConfig / cohsex.in). `jax_enable_x64` forced at import.

### Suspects
- **dead / test-only:** `kmeans_update_step` and `make_sharded_kmeans_update` have zero
  production callers — grep for their names across src/, tests/, tools/, scripts/ hits only
  `tests/test_kmeans_sharded.py` and the module itself; docstring admits "kept for direct
  tests / reuse". They duplicate `_kmeans_step_shardmap_body` + `_finalize_update`
  composition already exercised by the while-loop path.
- **redundancy:** classic parallel old/new path pairs — plain vs orbit variants of five
  routines (`assign_labels_chunked`/`assign_labels_orbit_chunked`,
  `_local_update_accumulators`/`_orbit_local_update_accumulators`,
  `_finalize_update`/`_orbit_finalize_update`, `_kmeans_step_shardmap_body`/`_orbit_step_shardmap_body`,
  `make_sharded_lloyd_loop`/`make_sharded_orbit_lloyd_loop`). The code itself notes the
  orbit path reduces to the plain one at n_sym=1 identity (`_orbit_distance_sq` docstring,
  lines 703–706); the plain path could be deleted (identity sym) at the cost of a small
  fori(1) overhead. Also `_orbit_d2_chunk` duplicates the distance body of
  `assign_labels_chunked`, and `_orbit_d2_per_point`/`pbc_distance_sq_single` are
  near-identical (per-point vs single-centroid broadcasting).
- **weird:**
  - line 36 `BOHR_TO_ANG = 0.529177210544` — CODATA-2022 value; k-means runs in Å (CLI
    converts avec), so `tolerance=5e-3` is Å-denominated; a Bohr-metric caller would get a
    ~2× different effective tolerance.
  - line 114 `_DEFAULT_C_BLOCK = 32` magic ("~30 MB peak per axis at P=1e6 in fp32" — but
    the module forces x64, so the comment's fp32 estimate is off 2×).
  - NaN-as-sentinel padding (lines 137, 261): relies on NaN propagating through
    subtract/quadform/minimum then a single isnan→inf mask; correct but subtle.
  - `tie_tol=1e-10` (line 231) absolute tie tolerance in Å² — scale-dependent.
  - `_finalize_update` line 406: `1e-10` floor on Σw despite the `> 0` where-guard
    (belt-and-braces double guard).
  - `_warn_dense_grid_regime` takes `m_cand` and never uses it (lines 987–995).
  - Driver comment (lines 878–882) explains why a compile/exec split idiom is *not* used —
    dead-idiom commentary referencing `chi.compile / chi.exec in gw_jax`.
  - `_canonicalize_rep` does a function-body relative import of `orbit_syms` (line 721),
    presumably to dodge a circular import; called inside jit via vmap.

---

## src/centroid/kmeans_cli.py (427 lines)

**Purpose.** CLI driver: `python -m centroid.kmeans_cli N_C [opts]` (also console script
`lorrax-centroids = "centroid.kmeans_cli:main"` in pyproject.toml:35). Loads WFN.h5 + charge
(or Pauli-current) density, builds the device mesh, runs `weighted_kmeans_jax`, snaps/unfolds
to the FFT grid, optionally prunes via pivoted Cholesky, writes `centroids_frac_{N}{suffix}.txt`.

**Category:** preprocessing tool / CLI driver (ISDF centroid generation stage).

### Function table

| Function | Lines | Role |
|---|---|---|
| `build_parser()` | 42–133 | argparse. Flags: `N_c` (positional, default 400), `--seed`, `--plot`, `--plot-zoom`, `--no-shard`, `--force-shard`, `--rho-source {auto,qe_save,wfn_ibz}`, `--rho-power` (ρ^α weight; Gersho: centroid density ∝ ρ^(3α/5), α=5/3 ⇒ ρ¹, α=10/3 ⇒ ρ² ≈ the \|ψ_v\|²\|ψ_c\|² target), `--qe-save`, `--oversample` (default 1.5; 1.0 disables pruning), `--prune-n-val`, `--prune-n-cond`, `--prune-window {v_x_c,v_x_vc,vc_x_vc}` (default v_x_vc), `--orbit`, `--no-orbit`, `--use-phdf5`, `--density-mode {scalar,current}`, `--out-suffix`. |
| `_build_mesh(args, n_points)` | 146–181 | 2-D device mesh ('x','y') via most-square factorisation ("same recipe as gw_jax._build_mesh"). Single device ⇒ 1×1 2-D mesh (one downstream codepath). Auto-gate: falls back to single device when P/n_shards < `_P_PER_SHARD_MIN = 100_000` (line 140, NCCL-latency floor measured on Si 4×4×4) UNLESS `--force-shard` or multi-host (multi-host fallback would deadlock the distributed shutdown barrier — lines 166–177). |
| `main()` | 188–423 | Sequence: `runtime.set_default_env()` **before any jax import** (lines 10–11), `runtime.init_jax_distributed()` (line 22), jax compile cache; read `WFN.h5` via `file_io.WfnLoader`; `common.symmetry_maps.SymMaps(wfn)`; init-method + dense-grid heuristics; density = `charge_density.get_charge_density(wfn, sym, source, save_dir)` or `current_density.build_current_density(wfn, sym, n_occ)` for `--density-mode current` (n_occ = nelec for nspinor=2, nelec/2 for nspinor=1, lines 227–229); FFT-grid consistency check ρ.shape == wfn.fft_grid, error mentions "ecutrho = 4·ecutwfc" (lines 240–245); avec_ang = wfn.avec·alat·BOHR_TO_ANG; ρ ← max(ρ,0)^α if `--rho-power≠1` (QE iFFT negative-noise clip, lines 252–257); mesh; orbit decision: `args.orbit or (wfn.ntran > 1 and not args.no_orbit)` (line 265) — **orbit mode is the DEFAULT whenever WFN has >1 sym op**; orbit mode divides the kmeans target by n_sym (M_cand_orbit = ceil(M_cand/n_sym), lines 276–288, ValueError if <1); run kmeans; snap/unfold; prune; write; timing report (rank 0); optional plot. Returns 0. |

### Snap/unfold detail (lines 303–342)
Orbit path: snap reps to grid FIRST (`deduplicate=False`), then
`orbit_syms.unfold_orbit_unique_with_id(reps_snapped, Rinv, tau)` — comment block lines
304–320: fp64 sym images of off-grid reps would round inconsistently; snap-then-unfold
guarantees on-grid orbit closure "because R is integer and τ × fft_grid is integer for
grid-commensurate τ" (the τ-commensurability assumption is asserted only in prose).
Convention comment: "Pass Rinv = inv(mtrx). BGW r-action is r' = Rinv·r + τ; matches
compute_centroid_sym_perm and validate_atomic_symmetries. No-op vs forward S on symmorphic
systems (CrI3, MoS2); critical for Si Fd-3m." Non-orbit path: snap with dedup; duplicates
redistributed by `ensure_unique_centroids` onto highest-ρ free grid points.

### Prune detail (lines 344–390)
If oversample>1 and n_unique>N_c: `pivoted_cholesky.prune_candidates_by_pivoted_cholesky`.
Orbit mode targets ORBITS: `n_orbit_keep = max(1, ceil(N_c · n_orbits / n_unique))` (line
350) so the kept Σorbit_size ≈ N_c. Band-window resolution: `_n_val_eff = --prune-n-val or
wfn.nelec`; `_n_cond_eff = --prune-n-cond or min(n_val, nbands − n_val)`. Windows:
`v_x_vc` (default) left=(0,n_val) right=(0,n_val+n_cond) — covers |ψ_v|² + v×c;
`vc_x_vc` square σ-window Gram; `v_x_c` legacy, passes `n_val=`/`n_cond=` kwargs instead of
`band_range_left/right` (two different kwarg protocols into the same pruner — see suspects).
Note `--prune-n-val` default = wfn.nelec — for nspinor=1 that counts electrons not bands
(nelec vs nelec/2 asymmetry handled for `--density-mode current` n_occ at line 229 but NOT
for the prune default; whether the pruner internally halves is out of scope for this file —
flagged as a cross-check item).

### I/O
- Reads: `WFN.h5` (cwd, via `file_io.WfnLoader`); QE `<prefix>.save` charge density
  (auto-detected `qe/scf/*.save` or `qe/nscf/*.save`, or `--qe-save`) through
  `centroid.charge_density.get_charge_density`; optionally G-space ψ through the phdf5 FFI
  loader (`--use-phdf5`) inside the pruner.
- Writes: `centroids_frac_{n_unique}{suffix}.txt` (np.savetxt, `%.6f`, `# `-comment header
  of 4 lines: grid/count, `density: scalar|current`, weight description, intended channels
  γ̃^0 vs γ̃^{1,2,3}), suffix `''`/`'_current'` by density-mode unless `--out-suffix`.
- Optional: `kmeans_centroids.png` via kmeans_plot.

### Cross-module deps
`runtime` (set_default_env, init_jax_distributed), `file_io.WfnLoader`,
`common.symmetry_maps`, `common.timing`, `common.jax_compile_cache`,
`centroid.charge_density`, `centroid.current_density`, `centroid.kmeans_isdf`,
`centroid.orbit_syms` (build_real_space_syms, unfold_orbit_unique_with_id),
`centroid.pivoted_cholesky`, `centroid.kmeans_plot`.

### Callers of main / module
- `python -m centroid.kmeans_cli` — documented in docs/quickstart.md:42,
  docs/architecture/codebase.md:97,466,478,621, docs/installation/perlmutter.md:36,
  docs/theory/physics.md:69,834 (`lxpre` runs it), skills. Console script pyproject.toml:35.
- Referenced in error strings: src/gw/gw_init.py:762, src/common/isdf_fitting.py:2069,
  src/gw/v_q_bispinor.py:555,571, src/gw/compute_vcoul.py:925.
- No test imports kmeans_cli itself (heuristics tested via kmeans_isdf).

### Suspects
- **weird / stale flag name:** parser defines `--orbit` / `--no-orbit` (lines 101–107), but
  `src/gw/v_q_bispinor.py:571` ("Regenerate centroids with ``kmeans_cli --orbit-aware``"),
  `v_q_bispinor.py:555` ("--orbit-aware run") and `docs/theory/symmetry.md:505` instruct a
  nonexistent `--orbit-aware` flag. Following that advice verbatim → argparse error.
- **weird:** `_P_PER_SHARD_MIN = 100_000` magic (empirical, documented); orbit-by-default
  gating on `wfn.ntran > 1` (line 265) means behavior silently changes with the WFN's sym
  content; two kwarg protocols into `prune_candidates_by_pivoted_cholesky`
  (`band_range_left/right` vs legacy `n_val/n_cond`, lines 369–382) — parallel old/new
  path inside one call site; prune n_val default = `wfn.nelec` without an nspinor
  adjustment (contrast line 229); `n_orbit_keep` proportional heuristic (line 350) only
  approximates N_c output count.
- **dead:** none found at file level.

---

## src/centroid/kmeans_plot.py (87 lines)

**Purpose.** Matplotlib visualisation helpers, split out so `kmeans_isdf` stays free of
matplotlib/scipy. Forces `MPLBACKEND=Agg` before import (shifter image lacks Qt/Tk).

**Category:** diagnostic/plot script.

| Function | Lines | Role |
|---|---|---|
| `interpolate_density(rho_np, zoom_factors=(1,1,1))` | 19–25 | Bicubic (`scipy.ndimage.zoom`, order=3) upsample for plotting only; identity if zoom=1. Caller: kmeans_cli:421 (under `--plot`). |
| `plot_density_and_centroids(wfn, rho_np, centroids_frac, out="kmeans_centroids.png", threshold_frac=0.05)` | 28–87 | 3D scatter of grid points with ρ > 0.05·max (colored `np.log(np.abs(rho_at_pts) - 0.9*threshold)`, plasma, alpha=0.09), red-star centroids, unit-cell wireframe; frac→cart via `@ wfn.avec` (note: **wfn.avec is in alat units here, not Å** — the CLI converts avec for kmeans but passes raw `wfn` to the plot; axes are unlabeled-unit, cosmetically fine). Saves PNG at dpi=150. Caller: kmeans_cli:422. |

### I/O
Writes `kmeans_centroids.png`. No reads.

### Suspects
- **dead-ish:** only reachable via `kmeans_cli --plot` (default off). Zero test/tool callers
  (grep `plot_density_and_centroids|interpolate_density|kmeans_plot` over src/tests/tools/
  scripts hits only kmeans_cli).
- **weird:** color transform `np.log(np.abs(rho_at_pts) - 0.9 * threshold)` (line 61) —
  magic 0.9; safe only because the mask guarantees rho > threshold so the argument
  ≥ 0.1·threshold, but `abs()` before subtracting is vestigial; alpha=0.09 / s=20 /
  s=100 cosmetic magics; `Axes3D` import marked `# noqa: F401` (needed side-effect on old
  matplotlib, no-op on new).

---

## src/centroid/centroid_io.py (55 lines)

**Purpose.** Header-aware reader for `centroids_frac_*.txt`: parses the `# density: ...`
comment line (written by kmeans_cli post-2026-05-02) so the bispinor pipeline can validate/
route charge vs current centroid files, returning `CentroidFile(coords (n,3) f64, density
∈ {"scalar","current","unknown"})`.

**Category:** I/O: centroid-file reader (currently orphaned).

| Item | Lines | Role |
|---|---|---|
| `DensityLabel` | 22 | Literal["scalar","current","unknown"]. |
| `CentroidFile` | 25–27 | NamedTuple (coords, density). |
| `_DENSITY_RE` | 30 | `^#\s*density:\s*(\S+)`, IGNORECASE. |
| `read_centroids(path)` | 33–52 | Scans leading `#` comment lines for the density label (pre-2026-05-02 files → "unknown"), then `np.loadtxt`. `__all__` exports all three. |

### I/O
Reads `centroids_frac_*.txt` (text, `#` header with `density:` line).

### Suspects
- **DEAD:** `read_centroids` / `CentroidFile` have **zero callers**. Grep evidence:
  `grep -rn "read_centroids\|centroid_io\|CentroidFile" src tests tools scripts` → only the
  definition file plus one docstring cross-reference in `src/file_io/isdf_header.py:15`
  (":class:`centroid.centroid_io.CentroidFile`" in prose). No test file imports it either
  (repo-wide grep). The production loader is `src/file_io/centroids.py:load_centroids`
  (plain `np.loadtxt`, ignores the density header) — used by `gw.gw_init` even for the
  bispinor `centroids_file_current` path (gw_init.py:766–775), i.e. the very consumer this
  module's docstring says it exists for ("gw_jax in bispinor mode") trusts config/filename
  instead of the header.
- **REDUNDANCY:** duplicates `file_io/centroids.py:load_centroids` reading (both do
  loadtxt of the same file format; this one adds the density label, that one adds grid-index
  snapping). Two centroid readers in two packages; a refactor should merge them (single
  source of truth per the no-redundancy rule) — either wire the density validation into
  `file_io.centroids` or delete `centroid_io.py`.

---

## Cross-file summary

- Production call chain: `kmeans_cli.main` → (`charge_density.get_charge_density` |
  `current_density.build_current_density`) → `weighted_kmeans_jax` →
  (`make_sharded_[orbit_]lloyd_loop`) → snap/unfold (`orbit_syms`) →
  `pivoted_cholesky.prune_candidates_by_pivoted_cholesky` → savetxt.
  Downstream consumer of the txt output: `file_io/centroids.py:load_centroids` ←
  `gw.gw_init` / `gw_config` default `"centroids_file": "centroids_frac.txt"` (gw_config.py:160).
- `tests/archive/test_kmeans.py:24` calls `weighted_kmeans_jax(avec_jax, rho_jax, N_k=5,
  seed=0)` — stale keyword `N_k` (signature is `N_c`) and missing required `mesh` kwarg;
  would TypeError if ever run. Archived-dead.
- Orbit machinery correctness anchor: image action `r' = r @ Rinv[s].T + tau[s]`
  (BGW inverse-rotation convention), fold-back `δ_rep = δ_image @ R[s].T` — matches the
  memory note that ψ/ζ/V_q sym actions must all route through one convention.
