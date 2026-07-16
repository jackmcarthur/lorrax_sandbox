# recover-or-generate V/W for BSE — design scout

Design scout, NO code. Base: `agent/bse-cleanup` @ origin/main c7a30ff (worktree
`sources/worktrees/lorrax_D_bse_cleanup`). GW source re-read at this HEAD.
Owner intent: BSE should preferentially **recover** `V_qmunu`/`W0_qmunu` (+ eqp)
from the GW restart h5; if absent, **generate** them by calling the same gw/
functions, with a loud banner and a DFT-energy fallback when no GW eqp is present.

## 1. Current call chain (the producers, verbatim from gw_jax.main)

The entire "produce V_qmunu + W0_qmunu" span is `gw_jax.main()` lines **187–287**
— it is already a clean, linear prefix that stops right before the Σ/QP half:

```
config = LorraxConfig.from_input_file(input)              # gw_jax:181 (canonical parser)
mesh_xy = _build_mesh()                                    # :188  x×y, gx=largest div ≤ √ndev
_setup_runtime(config, mesh_xy)                            # :189  nccl + phdf5 + compile cache
wfn  = WFNReader(config.paths.wfn_file, mesh=mesh_xy)      # :200
sym  = symmetry_maps.SymMaps(wfn)                          # :201
_, centroid_indices, n_rmu = load_centroids(...)           # :202
tensors_filename = tmp/isdf_tensors_{n_rmu}.h5             # :205
meta = Meta.from_system(...); band_slices = ...            # :211-216
enk_dft = get_enk_bandrange(...)                           # :221  (DFT, Ry)
head_resolver = HeadResolver(config, input_dir, wfn, sym, meta)   # :229
bgw_v_grid_fn = build_bgw_v_grid_fn(...)                   # :233  (None unless use_bgw_vcoul)

isdf = prepare_isdf_and_wavefunctions(cfg=config, wfn, sym, meta, # gw_init.py:537
          centroid_indices, band_slices, mesh_xy, tmp_dir,
          tensors_filename, print0, bgw_v_grid_fn)
   └─ (a) V_qmunu: plan_gflat_chunks → load_centroids_band_chunked → fit_zeta
          → compute_V_q  (gw_init.py:545-716; = the (a) path in the task)
      writes restart mode="w": V_qmunu(flat-q) + G0_mu_nu + enk_full + W0
          placeholder(W0_ready=False) + kgrid; appends psi_full_y (mode="a")
V_q = isdf.V_qmunu
if do_screened:                                            # gw_jax:266
    quad,e_ref = build_static_quadrature(wfns, config.minimax_config)
    requests   = screening_requests_for(mode, config)   # SC keeps only "static"
    W_by_role  = compute_screening(wfns, V_q, requests, quad, e_ref,
                     sym, centroid_indices, config, meta, mesh_xy)   # screening.py
persist_w0_and_head(W_by_role.get("static", V_q),          # gw_output.py:176
        tensors_filename, head_resolver, config, meta, mesh_xy)
   └─ (b) writes W0_qmunu(flat-q, W0_ready=True) + vhead + whead[ω] + omega_grid
```

Everything from `static_head_terms`/`compute_sigma_xc` onward (gw_jax:296→end) is
Σ/QP and is **not** needed to produce V/W. `eqp0.dat`/`eqp1.dat` are written far
downstream by `gw_output.write_results` (via `compute_eqp_diag`) — **not** into the
restart h5. The restart h5 carries only `enk_full` (DFT); confirmed in
`tagged_arrays.read_restart_state_from_h5` (returns V_qmunu, S_qmunu, psi_full_y,
enk_full, V0_noG0, G0 — no eqp).

## 2. What is reusable as-is vs entangled with the driver

| Producer | Reusable as a library call? | Entanglement |
|---|---|---|
| `prepare_isdf_and_wavefunctions` (gw_init:537) | **Yes, already library-shaped**: pure kwargs `(cfg, wfn, sym, meta, centroid_indices, band_slices, mesh_xy, tmp_dir, tensors_filename, print0, bgw_v_grid_fn)`, returns `SimpleNamespace`. No globals, no argparse, no driver state. | needs a fully-built `LorraxConfig` (reads `cfg.memory/.bispinor/.backend/.restart`) + `Meta` + `BandSlices` + `mesh_xy`. |
| `compute_screening` (screening.py) | **Yes**: pure kwargs `(wfns, V_q, requests, quad, e_ref, sym, centroid_indices, config, meta, mesh_xy)`. | needs `wfns` from the isdf bundle + `quad/e_ref` from `build_static_quadrature`. |
| `persist_w0_and_head` (gw_output:176) | **Yes**: keyword-only, self-guards (`no-op unless config.do_screened and file exists`). | needs `head_resolver` (HeadResolver) + `W_by_role["static"]`. |
| `build_static_quadrature`, `screening_requests_for`, `HeadResolver`, `_build_mesh`, `_setup_runtime`, `build_bgw_v_grid_fn` | Yes — all pure/keyword. | `_build_mesh`/`_setup_runtime` are gw_jax module-private (leading `_`); the orchestration that wires them is inlined in `main()`. |

Conclusion: the **functions** are already modular; the only thing *not* factored
out is the ~100-line **orchestration prefix** that constructs mesh/wfn/sym/meta/
head_resolver and threads them. Duplicating that prefix inside `bse/` would
violate no-redundancy. The honest fix is to extract it once.

## 3. Proposed seam

**One extraction in gw/, one thin caller in bse/. No physics in bse/.**

**(A) `gw_init.py` (or new `gw/gw_restart.py`) — `build_gw_restart(config, *, print_fn=print, require_screened=None) -> SimpleNamespace`**
Lift `gw_jax.main()` lines 187–287 verbatim (mesh → runtime → wfn/sym/centroids →
meta/band_slices → enk_dft → head_resolver → bgw_v_grid_fn →
prepare_isdf_and_wavefunctions → optional quadrature+compute_screening →
persist_w0_and_head). Returns the context bundle already in scope there
(`config, mesh_xy, wfn, sym, meta, band_slices, wfns, wfns_transverse, V_q,
W_by_role, head_resolver, enk_dft, quad, e_ref, tensors_filename`).

**(B) `gw_jax.main()` shrinks** to: build `config`, `ctx = build_gw_restart(config)`,
then run the Σ/QP half off `ctx.*`. Behaviour-preserving faithful extract — net
~0 logic, and it directly realizes MAP §7-step-1's e2e gate (run gw_jax → BSE on
the restart it just wrote).

**(C) `bse/` driver — `produce_restart_if_missing(input_file, tensors_filename, *, need_screened, print_fn)`**
Contains ZERO kernel logic — existence probe + banner + delegate:
```
present, w0_ready = _restart_status(tensors_filename)   # h5: has V_qmunu? W0_qmunu.attrs['W0_ready']?
if present and (w0_ready or not need_screened):
    return                                               # RECOVER: bse_io loader reads it (§5)
_warn_no_restart_banner(print_fn, tensors_filename, need_screened)   # loud multi-line
cfg = LorraxConfig.from_input_file(input_file, print_fn=print_fn)    # canonical config (mandate #1)
build_gw_restart(cfg, print_fn=print_fn, require_screened=need_screened)   # GENERATE via SAME gw fns
```
`_restart_status` is a ~10-line `h5py` peek (grep-verifiable, no device work).
This is the ONLY new gw→bse import beyond `head_correction` — the intended
library-call direction (mandate #1), not a layer.

**Where it hooks**: at the top of the BSE driver, *before* `create_mesh_2d()` and
before any `load_bse_data_from_restart_sharded` call (bse_jax `_preview_lanczos`,
bse_feast.main, absorption_haydock). One call site, guarded once.

## 4. Warning UX + DFT fallback

Loud multi-line banner (reuse `gw_output.print_banner` style — a `print_fn`
box, not `logging`), fired in two places:

*No restart → generating:*
```
========================================================================
  !!  BSE: no GW restart at tmp/isdf_tensors_{n_rmu}.h5
  !!  Generating V_qmunu + W0_qmunu now by running the GW screening
  !!  pipeline (kmeans → fit_zeta → V_q → compute_screening → W0).
  !!  This does NOT compute quasiparticle energies: the BSE diagonal
  !!  will use DFT (enk_full) eigenvalues unless a BGW eqp1.dat is
  !!  supplied via --eqp.  For GW-quality peaks, run gw_jax to
  !!  completion first, or pass --eqp eqp1.dat.
========================================================================
```
*Restart present but no GW eqp (recover path, in the eqp-resolution block):*
same box, one line: `using DFT energies (enk_full); pass --eqp for GW energies.`

**DFT fallback is already the default**: `bse_io` slices the diagonal from
`enk_full` (DFT, Ry) and only overwrites it when `--eqp` unfolds `eqp1.dat`
(`apply_eqp_corrections`, SymMaps IBZ→full). The fallback needs **no new data
path** — only the banner when the eqp source is absent. Owner's phrase "GW
energies if present in file" maps to: (i) `--eqp eqp1.dat` sibling written by a
completed GW run, or (ii) an optional future `eqp` dataset in the restart h5
(§7 open Q). Either way, absence ⇒ banner ⇒ `enk_full`.

## 5. Composition with the parallel loader repair (flat-q)

The RECOVER branch **defers every on-disk read to the repaired loader** — it calls
`load_bse_data_from_restart_sharded` / `_load_ring_subset`, which are exactly the
functions the flat-q normalization work is fixing (B3 8-D-only V reader, B4
missing-`kgrid` W reader, B5 pre-shim head injection). This seam touches **no**
on-disk layout and adds **no** parsing; it is layout-agnostic by construction.
The GENERATE branch writes through the canonical writers (`write_restart_state_to_h5`,
`write_w0_qmunu_to_h5`), which already emit flat-q. So ordering: **loader repair
(B3-B5) is a hard prerequisite** — `produce_restart_if_missing` is inert value
until the recover path can read a fresh restart. The two land as: loader repair
first (MAP §7-step-1), this seam on top (same step, after normalization).

## 6. What blocks it

1. **Loader repair (B3/B4/B5)** — recover path is dead against fresh restarts
   until flat-q normalization lands. Hard dependency.
2. **eqp is not in the restart h5** — only `enk_full` (DFT) is. "Recover eqp from
   the restart file" needs a decision (§7).
3. **gw_jax.main() extraction** is a behaviour-preserving refactor that touches
   the driver — must be its own commit, gated by the fresh-restart e2e gate
   (which is *this* seam). Not a fork; a single-source move.
4. **BSE config duality** — GENERATE forces `LorraxConfig.from_input_file`
   (prepare_isdf needs `cfg.memory/.bispinor/.backend`). RECOVER still reads
   cohsex.in through bse_io's private `_parse_head_overrides`/`_parse_wfn_path`.
   Unifying BSE onto `gw_config.read_lorrax_input` is **larger than this scout's
   change** → recorded in CLEANUP_LOG "Deferred consolidations", not half-done here.
5. **Mesh double-init** — `_build_mesh` (gw) and `create_mesh_2d` (bse) produce
   the *same* x×y shape (largest divisor ≤ √ndev), but GENERATE must run *before*
   the BSE mesh/`init_jax_distributed` so there is exactly one mesh + one runtime
   init per process. Hence the call site at the very top of the BSE driver
   (§3C). Unifying the two mesh builders is a deferred consolidation.

## 7. LOC estimate

| Change | LOC |
|---|---|
| `build_gw_restart` extraction in gw_init/gw_restart (lift 187-287) | +130 moved, main() −130 → ~+20 net (bundle namespace) |
| `gw_jax.main()` rewire onto `ctx.*` | ~−100 (shrinks) |
| `produce_restart_if_missing` + `_restart_status` (bse) | ~45 |
| `_warn_no_restart_banner` | ~15 |
| DFT-fallback banner in eqp-resolution block | ~10 |
| **Genuinely new logic** | **~70–90** (rest is net-neutral move) |

## 8. Open questions (for Jack)

1. **eqp source of truth**: reuse sibling `eqp1.dat` via existing `--eqp`
   (no new persistence, matches current BSE), OR have `write_results` also stash
   an `eqp` dataset into `isdf_tensors_*.h5` so recover is self-contained
   ("from the GW restart file", owner's words)? Scout leans (a) + optional (b).
2. **GENERATE depth**: stop at `persist_w0_and_head` (V/W only, DFT diagonal —
   proposed), or run the full `gw_jax.main()` to also produce eqp? Full-QP
   generation is a heavier opt-in; propose gating it behind an explicit flag.
3. **`build_gw_restart` home**: extend `gw_init.py` vs new `gw/gw_restart.py`?
   New module reads cleaner (gw_init is ISDF-only today) but adds a file.
4. **Screened-vs-not**: `need_screened` should derive from the BSE kernel choice
   (`--bse` ⇒ need W0_ready; `--rpa` ⇒ V only). Confirm the C1 default-routing
   fix lands first so `need_screened` is unambiguous.
5. Confirm the mesh/runtime single-init ordering (call site above BSE mesh build)
   is acceptable vs passing a pre-built mesh into `build_gw_restart`.
