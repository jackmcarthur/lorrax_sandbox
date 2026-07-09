# LORRAX GW — refactor MAP (the spine)

> **This is the PLAN. For live progress (what's done / deferred / cancelled,
> which branch, what's on origin), read `STATUS.md` first** — updated 2026-07-03.
> Quick state: gate-first ✅ (now incl. first **BGW-anchored 3D COHSEX gate**), delete-pass ✅,
> single-source 🟡 (parser+vcoul+**load_wfns facade DELETED**+reader boilerplate done;
> eqp math + zeta reader still dup), memory-planner 🟡 (aot package
> DELETED; two in-tree planners still stacked), **v_q_tile DELETED (r-space V_q
> subsystem gone; V_q is now G-flat-only)**, **A2 ζ-fit stage MOVED common/→gw/ and
> SPLIT into a reusable `isdf/` core library**, IBZ symmetry unification not started.

> **UPDATED 2026-07-03** (branch `agent/memplanner-cleanup`, latest `bb04399`):
> Second reconciliation pass. Deltas since the 2026-07-02 note (all verified against
> `git show`): (1) **§4 #2 DONE** — `common/isdf_fitting.py` was moved to `gw/` (`c9fb0e2`)
> then its CORE primitives extracted to a NEW top-level `isdf/` package (`isdf/core.py`,
> `dfb6b90`); `gw/isdf_fitting.py` is now a 1030-L thin GW orchestrator consuming
> `isdf.core`. (2) **§4 #10 DONE** — `common/load_wfns.py` **deleted** (`bb04399`); its 5
> ψ-loading helpers relocated to `common/wfn_transforms.py` (now 1876 L), 22 sites
> repointed. (3) **§4 #4 mostly DONE** — the bispinor lift moved out to
> `bispinor_init.lift_to_4spinor` (`842c3c6`), the ψ-unfold rule to
> `symmetry_maps.trs_augment_U`/`tau_phase_row` (`a697abc`), and the eager whole-array
> slurp made lazy (latent-OOM fix). (4) **New 3D gate** — Si 4×4×4 no-SOC COHSEX,
> BGW-anchored, `tests/regression/si_cohsex_debug/` (`d55c4cb`) — closes gate-prereq #2.
> (5) Reader boilerplate single-sourced into `mf_header`/`isdf_header` binders + `_slab_io_ffi`
> (`94cc354`). (6) 4 container-JAX failures conditionally skip-marked; suite = **242 passed /
> 24 skipped / 0 failed**; 3 golden gates wired into `skills/checkpoint/SKILL.md` (`25067f4`).
> **Still deferred:** `zeta_loader`/`zeta_reader` merge (§4 #8 — genuine backend + valid_mu
> divergence, needs a padded-μ fixture; see `archive/READER_CLEANUP_PLAN.md` step 4).

> **UPDATED 2026-07-02** (branch `agent/memplanner-cleanup`, base `e7b6c7d`):
> This doc has been reconciled against the merged refactor branch. Every §4 target
> now carries a `status as of 2026-07-02` marker. Headline deltas since the doc was
> written: the whole `gw/aot_memory_model/` package (~8.7k L incl. DoE artifacts) is
> **deleted**; the r-space **`gw/v_q_tile.py` (1662 L) is deleted** and V_q is now a
> single G-flat path; 5 verified-dead modules removed; the cohsex.in parser and
> `gw/vcoul.py` were single-sourced; `_rotate_psi` collapsed 4→2; and the e2e crash
> was fixed with GN-PPM + IBZ-vs-full-BZ gates added and the reference re-frozen.

Synthesized 2026-07-01 from a 67-file agent read pass + 1049 adversarial verdicts
+ two independent teleological sortings (`archive/_raw_sorts.json`) + `archive/FLAGS.md` + `archive/GATE_AUDIT.md`.
Checkout: `sources/lorrax_D`. Per-file detail lives in `archive/files/*.md`.

This document is the map a model should load to orient before touching GW code:
(1) the dataflow spine, (2) the concern×stage matrix, (3) the category interaction
cross-map, (4) where the current file layout fights the taxonomy — the ranked refactor targets.

---

## 0. The final teleology (merged from both sortings)

Two independent sorters produced ~26 categories that agree almost cell-for-cell.
Merged into three teleological tiers plus config/diagnostics/dead:

**Tier A — Pipeline stages** (dataflow order; "a new stage kernel goes here"):
`A0` Preprocessing (input producers) · `A1` WFN ingest · `A2` ζ-fit (ISDF) ·
`A3` V_q (bare Coulomb) · `A4` χ₀/W screening · `A5` Σ static (X/COHSEX) ·
`A6` Σ dynamic (PPM Σc) · `A7` QP solve / eqp / SC (QSGW) · `A8` Output writers

**Tier B — Variant axes** (cut *across* A-stages; "a new physics option toggles here"):
`B1` Frequency treatment (minimax/PPM engine) · `B2` Coulomb truncation + q→0 head/wing (3D/2D/0D) ·
`B3` Bispinor / Breit spin algebra · `B4` IBZ symmetry & BZ mapping

**Tier C — Infrastructure** (physics-free, serves any stage; "reusable machinery"):
`C1` Layout/sharding/padding contracts · `C2` Sharded FFT & G↔r transforms ·
`C3` Distributed dense linear algebra (FFI + in-tree) · `C4` Parallel-HDF5 slab I/O ·
`C5` Per-format readers/writers · `C6` Memory planner & chunk choosing ·
`C7` Host caches & compile cache · `C8` Runtime bootstrap ·
`C9` Generic iterative solvers · `C10` DFT/pseudopotential operator stack (shared by A0)

**Tier D — Config & flag surface** · **Tier E — Diagnostics/instrumentation/benches** ·
**Tier F — Archived / dead**

Full category→file membership: `archive/_raw_sorts.json`. Full feature enumeration: `archive/FEATURES.md`.

---

## 1. Pipeline dataflow spine

Boundary arrays are what cross between stages; residency H=host, D=device(sharded on XY mesh).

```
A0 PREPROCESSING (standalone CLIs, run once; produce the files gw_jax consumes)
   centroid.kmeans_cli ──▶ centroids_frac_*.txt        (orbit-aware weighted k-means + pivoted-Cholesky)
   gw.kin_ion_io       ──▶ kin_ion.h5   (T+V_loc+V_NL) │ psp.get_dipole_mtxels ──▶ dipole.h5 (velocity mtxel, q→0 head)
   psp.run_sternheimer ──▶ sternheimer.h5 (opt: q→0 head w/o SOS)
   psp.run_nscf / pseudobands ──▶ WFN.h5 (opt: no-QE path)
        │
        ▼   boundary: WFN.h5, centroids txt, kin_ion.h5, dipole.h5   [H on disk]
A1 WFN INGEST                          file_io/wfn_loader.WfnLoader  (ψ-loading helpers now in common/wfn_transforms; load_wfns facade DELETED 2026-07-03)
        │  build Wavefunctions bundle (4 copies: ψ(G), ψ(r)-slab, ψ-centroid, occ)
        │  [B3 lift → bispinor_init.lift_to_4spinor · B4 ψ-unfold → symmetry_maps.trs_augment_U/tau_phase_row · slurp now lazy]
        ▼   boundary: Wavefunctions bundle, Meta   [ψ(G) H via PsiGStore/io_callback; ψ(r) D]
A2 ζ-FIT (ISDF)                        gw/isdf_fitting.fit_zeta_to_h5 (orchestrator) ◀── driven by gw_init
        │      └── consumes isdf/core.py (NEW reusable ISDF mini-library: pair_density, c_q_from_psi_sm,
        │          factor_c_q, solve_zeta, fit_one_rchunk, …; GW + future direct-BSE are both consumers)
        │  pair density M_μ = Σ ψ*ψ at centroids → CCT/ZCT metric → distributed Cholesky/LU solve
        │  → G-flat accumulate → zeta_q.h5
        ▼   boundary: zeta_q.h5  [ζ_q(μ,G) D, sharded μ on x]   (+isdf_header/mf_header groups)
A3 V_q (bare Coulomb)                  compute_vcoul.compute_all_V_q (dispatcher, G-flat-only since 2026-07-02)
        │    ├─charge──▶ v_q_g_flat.compute_all_V_q_g_flat        (r-space v_q_tile path DELETED)
        │    └─bispinor▶ v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5
        │  V_q(μ,ν) = Σ_G ζ*_μ v(q+G) ζ_ν ,  g0_μ(q)      [B2 supplies v(q+G); B4 unfold via _unfold_g0_ibz_to_full, now in v_q_g_flat]
        ▼   boundary: V_qmunu.h5 or in-mem V_q, g0   [D, sharded μν on xy]
A4 χ₀/W SCREENING                      gw/w_isdf.solve_w   (only if do_screened)
        │  χ₀(q,μ,ν) = Σ minimax-τ  G_v G_c  (B1 supplies τ nodes) → Dyson W=(I−Vχ₀)⁻¹V
        │  q→0 head/wing from dipole/epshead (B2)
        ▼   boundary: W_q(μ,ν)  [D]  ·  or PPM params (B,Ω) if use_ppm_sigma
A5 Σ STATIC                            gw/cohsex_sigma (+ sigma_x_bispinor for Σ^B)
        │  Σ_X = −Σ G_occ V ; SX−X, COH (static COHSEX) ; bispinor transverse Σ^B
        ▼   boundary: Σ_μν → project to Σ_ij band basis (wavefunction_bundle.project)
A6 Σ DYNAMIC (if use_ppm_sigma)        gw/ppm_sigma + ppm_pipeline
        │  fit 2-pt PPM(GN/HL) from W(0)/W(probe) → Σc(k,ω) on real-ω grid via τ-quadrature windows
        ▼   boundary: sigma_mnk.h5 / Σc(ω)  [D or streamed H5]
A7 QP SOLVE / eqp / SC                 gw/sc_iteration + sigma_dispatch + qsgw_utils + eqp_bgw + degen_average
        │  Z-factor, eqp0/eqp1 linearization ; QSGW: H_qp build, diagonal Σ(E) fixed point, Hermitise
        ▼   boundary: E_qp, Σ^xc
A8 OUTPUT                              gw/gw_output + eqp_bgw + file_io/sigma_output + qp_wfn
             ──▶ eqp0.dat, eqp1.dat, sigma_diag.dat, sigma.h5, qp_wfn.h5
```

Driver that sequences A1–A8: `gw/gw_jax.py:main` (routes by `compute_mode` ∈ {cohsex, gn_ppm, …}).
Stage sequencing + chunk planning + restart lives in `gw/gw_init.py` (still overloaded — see §4;
now 1169 L, down from 1328 after the aot-planner deletion but not yet split).

---

## 2. Concern × stage matrix

Each cell: **where the variant/infra concern hooks the stage** (file:function or mechanism).
Blank = concern does not touch that stage. This matrix is the refactor's real target — a
concern smeared across many stages via *different* mechanisms is the disorder.

| Concern ↓ / Stage → | A2 ζ-fit | A3 V_q | A4 χ₀/W | A5 Σ_X | A6 Σc dyn | A7 eqp/SC |
|---|---|---|---|---|---|---|
| **B1 freq treatment** | — | — | `w_isdf.build_*_quadrature` ← `minimax`,`minimax_screening` | — | `ppm_sigma` fit+eval; `minimax_screening.fit_*_ppm` | — |
| **B2 Coulomb trunc + head** | — | `compute_vcoul`(v(q+G)); `coulomb/{bulk,slab,box}` | `head_correction`, `chi_from_dipole`, `vcoul.wcoul0` | — | `ppm_pipeline._inject_analytic_head` | — |
| **B3 bispinor/Breit** | `isdf.core`/`isdf_fitting` γ̃ 4-channel; lift → `bispinor_init.lift_to_4spinor` (moved out of `wfn_loader`) | `v_q_bispinor.compute_V_q_bispinor_g_flat_to_h5` (G-flat; 7-tile r-space path DELETED) | **absent (unbuilt)** | `sigma_x_bispinor` Σ^B | drops Σ^B (known) | — |
| **B4 IBZ symmetry** | `orbit_syms` centroid orbits | `v_q_g_flat._unfold_g0_ibz_to_full` + `_resolve_ibz_q_list` (v_q_tile unfold DELETED) | unfold via same tables | ψ-unfold → `symmetry_maps.trs_augment_U`/`tau_phase_row` (moved out of `wfn_loader`) | — | — |
| **C1 layout/padding** | bundle spec | μν PartitionSpec | χ₀/W PartitionSpec | Σ spec | ω,k,m,n spec | — |
| **C6 memory planner** | `gw_init.compute_optimal_chunks` + `gflat_memory_model` (2 in-tree, was 3; `aot_memory_model` package DELETED) | (was `v_q_tile._choose_v_q_chunks`; DELETED with tile) | request planner | ~~`aot sigma_kij` cost~~ (deleted) | window sizing | — |
| **C4/C5 I/O** | `zeta_reader`/`zeta_loader` (still 2, deferred §4#8), `slab_io`; header boilerplate → `mf_header`/`isdf_header` binders + `_slab_io_ffi` (single-sourced 2026-07-03) | `read_bgw_vcoul` | — | — | `sigma_mnk.h5` streamed | `sigma_output`,`eqp_bgw`,`qp_wfn` |
| **C3 dist. linalg** | Cholesky/LU (`cholesky_2d`,`ffi/*`) | — | Dyson solve (`ffi/cusolvermp`,`cublasmp`,`slate`) | — | — | eigh (QSGW Hermitise) |

Symmetry unfolding (B4) and the memory planner (C6) are the two most-smeared concerns:
each appears in ≥4 stages through **different helpers**. Those two drive the refactor.
_Update 2026-07-02:_ C6 lost its worst offender — the dead-by-clobber `aot_memory_model`
package is deleted, and the r-space `v_q_tile._choose_v_q_chunks` chooser went with the tile;
but `gw_init.compute_optimal_chunks` and `gflat_memory_model` are **still two parallel planners**,
so C6 is not yet single-homed. B4 unification (one `SymMaps` table + one sym-action helper)
is **untouched** — the helpers just moved (v_q_tile's copy died), they were not unified.

---

## 3. Category interaction cross-map (couplings + mechanism)

Refactor order follows these edges: you cannot move a node until its inbound contracts are stable.

| A ↔ B | Mechanism (the coupling to preserve) |
|---|---|
| A2 ζ-fit → A3/A4/A5 | **zeta_q.h5 on-disk G-layout**: per-q sphere {G:|q+G|²≤cut} ordered by flat-FFT index. `coulomb_sphere` defines it (its r-space `compute_bare_coulomb_sphere_idx` was DELETED with the tile subsystem); readers validate by *size only* (latent bug, §5). |
| C1 layout ↔ all A-stages | Module-level `PartitionSpec` constants + the 4-copy `Wavefunctions` bundle. Every kernel assumes them; changing a spec is a cross-stage edit. |
| B4 sym ↔ A2/A3 | IBZ-cascade **gate**: activates only when centroid orbit closure holds (`v_q_g_flat._resolve_ibz_q_list`). One canonical `SymMaps` table *should* serve ψ/ζ/V_q — today there are ≥4 parallel unfold helpers. |
| B1 freq ↔ A4/A6 | `minimax` emits (τ,α) node sets; `w_isdf` and `ppm_sigma` both consume. `w_isdf` also *hosts* quadrature builders (misplaced, §4). |
| B2 head ↔ A4/A6 | head source resolved from `dipole.h5`/epshead; **streamed Σc path silently skips head injection** (§5). |
| C6 planner ↔ A2/A3/A4/A5 | Planner picks static chunk sizes *before* jit; kernels must match. ~~`gw_init` stacks 3 planners; `aot_memory_model.chooser` is dead-by-clobber~~ **(2026-07-02: aot package deleted; still 2 planners — `gw_init.compute_optimal_chunks` + `gflat_memory_model`)** (§5). |
| C3 FFI ↔ A2/A4 | shard_map wrappers over .so; **stale-cache & layout bugs** in `cholesky_2d`, `slate/eigh`, `compute_vcoul` cache key (§5). |
| C4 slab I/O ↔ A2/A6 | `SlabIO` moves sharded arrays to HDF5 under one padded-memory/logical-disk contract; 3 interchangeable backends. |
| D config ↔ everything | `LorraxConfig` (frozen) threaded through all stages; **shadow env-flag surface** (`LORRAX_SC_*`, `LORRAX_V_Q_*`) bypasses it (§5). |

---

## 4. Where the file layout fights the taxonomy — ranked refactor targets

Both sorters independently flagged these as multi-category files ("misfits"). Ranked by
(smearing severity × blast radius). These are the move-and-split targets, in order.
**Status legend: ✅ DONE · 🟡 PARTIAL · ⬜ OPEN — markers carry their own dates; unmarked = `status as of 2026-07-02`.**

1. **`gw/gw_init.py` (was 1328 L → now 1169 L)** — hosts A2 orchestration + **stacked C6 memory planners** + restart I/O + cutoff resolution. The repo's worst multi-category file. Split: planners→C6, restart→C5, keep stage sequencing.
   🟡 **PARTIAL** *(status 2026-07-02)*: the dead `aot_memory_model` planner is gone, but `compute_optimal_chunks` (its own in-file memory model) **and** the gflat planner are still both here; restart I/O + cutoff resolution not extracted. The stage split is **not done**.
2. **`common/isdf_fitting.py` (was 2744 L)** — an entire A2 pipeline stage living in `common/`, with embedded C6 memory probes, 6 hand-keyed jit caches, C3 FFI solves. Belongs in `gw/` as the ζ-fit stage; extract the caches/probes.
   ✅ **DONE** *(status 2026-07-03)*: moved `common/`→`gw/` as a pure move (`c9fb0e2`), dead `_mem_report` probe dropped + jit caches docstringed (`62ce45e`), then the CORE primitives extracted to a NEW top-level **`isdf/` package** (`isdf/core.py` 1733 L + `isdf/__init__.py`; 7 public + private helpers + the 6 jit caches, byte-identical — `dfb6b90`). `gw/isdf_fitting.py` is now a **1030-L thin GW orchestrator** (`fit_zeta_to_h5` + `mem_probe`/`_nvsmi_used_mb_local_gpu`) importing from `isdf.core`. The isdf/ library is a reusable ISDF mini-lib — GW and a future direct-BSE path are both consumers. (The C6 probes stay in the orchestrator by design, not the pure-math core.)
3. **`gw/v_q_tile.py` (1662 L)** — A3 physics kernel + embedded **3-tier C6 chooser** + **two hand-rolled B4 IBZ unfold helpers**. Three categories; split all three out.
   ✅ **DONE — by DELETION, not split** *(status 2026-07-02)*: the entire r-space V_q tile subsystem is deleted (`v_q_tile.py` + `v_q_bispinor.compute_V_q_bispinor_to_h5` + 2 factories + `compute_vcoul.make_v_munu_chunked_kernel` + the FFT-era trio + the r-space tail of `compute_all_V_q` + `coulomb_sphere.compute_bare_coulomb_sphere_idx` + the `gw_init` legacy else-branch). V_q is now G-flat-only; the one surviving B4 helper `_unfold_g0_ibz_to_full` was relocated to `v_q_g_flat`. `compute_all_V_q` now raises on non-G_flat.
4. **`file_io/wfn_loader.py` (was 1260 L → now 1206 L)** — C5 reader hosting **B3 bispinor lift** + **B4 ψ symmetry unfold** + eager whole-array host slurp (latent OOM, §5). Reader should read; lift→B3, unfold→B4.
   🟡 **PARTIAL — de-physics mostly done** *(status 2026-07-03)*: **B3 lift moved out** → `bispinor_init.lift_to_4spinor` + shared `HALFALPHA` (`842c3c6`; loader keeps only the jit-cache-by-sharding wrapper). **B4 ψ-unfold moved out** → `symmetry_maps.trs_augment_U`/`tau_phase_row`, and `_get_umklapp_vector` promoted to public `get_umklapp_vector` (`a697abc`). **Eager slurp made lazy** — `__init__` no longer slurps whole ψ arrays; eager backend hyperslabs per call (latent phdf5 OOM fixed). `_rotate_psi` collapsed 4→2 in `wavefunction_bundle.py` (`466040a`). *Remaining:* the loader is still 1206 L; residual physics-flavored transform glue and the broader wfn_transforms consolidation not fully finished.
5. **`gw/eqp_bgw.py` + `gw_output.py` + `gw_jax`** — QP-linearization math **duplicated ~4×** across these; A7 physics fused with A8 BGW text writer. Single-source the eqp0/eqp1/Z math.
   ⬜ **OPEN** *(status 2026-07-02)*: not single-sourced. `gw_jax` imports `compute_eqp_diag`/`compute_z_factor_from_omega_grid` from `eqp_bgw` (one shared path) but the ~4× duplication called out here was not consolidated in this branch.
6. **`gw/gw_driver_helpers.py` (285 L)** — grab-bag: D config mirror (`PPMSigmaRuntimeOptions`, 23-field) + C8 runtime init + A3 BGW-grid glue. Fan out to the three homes.
   ⬜ **OPEN** *(status 2026-07-02)*: unchanged.
7. **`psp/get_DFT_mtxels.py` (926 L)** — C10 kernel library + stale debug driver + **third copy of the cohsex.in parser**. Parser→D (single source), driver→delete.
   🟡 **PARTIAL** *(status 2026-07-02)*: parser **consolidated** — the duplicate is deleted and consumers now import `read_lorrax_input as read_cohsex_input` from `gw.gw_config` (single source of truth). The C10-vs-driver split and the stale debug driver deletion are **still open**.
8. **`file_io/zeta_loader.py` vs `zeta_reader.py`** — unfinished old/new C5 reader migration; duplicated mf_header mirrors + duck-typing shim in `v_q_g_flat`. Finish the migration, delete the old.
   🟡 **OPEN — boilerplate de-duped, merge DEFERRED** *(status 2026-07-03)*: the shared `mf_header`/`isdf_header` mirrors were single-sourced into binders (`94cc354`, both readers now bind rather than re-declare fields). But the two readers still both exist (`zeta_loader.py` 575 L + `zeta_reader.py` 381 L) and `v_q_g_flat` still duck-types between them (`_make_read_all_ibz`). The full merge is **deliberately deferred** — genuine divergence in backend semantics + `valid_mu` zero-fill needs a padded-μ fixture to land safely. See `archive/READER_CLEANUP_PLAN.md` step 4. *(2026-07-08: the padded-μ gate now exists — `tests/test_mu_pad_invariance.py` — so this is UNBLOCKED; see NEXT_TARGETS #7.)* **Top remaining single-source target.**
9. **`w_isdf.py:build_*_quadrature`** — B1 frequency code hosted inside the A4 stage file. Move to the minimax engine.
   ⬜ **OPEN** *(status 2026-07-02)*: `build_static_quadrature` / `build_imag_quadrature` / `build_real_quadrature` still live in `w_isdf.py`.
10. **`common/load_wfns.py` (was 535 L)** — deprecated C5 facade, "P4-slated for deletion", still on the import path of gw_jax/gw_init/ppm_pipeline/sc_iteration. Cut over to `WfnLoader`, delete.
    ✅ **DONE** *(status 2026-07-03)*: **file DELETED** (`bb04399`). Its 5 ψ-loading helpers (incl. `load_centroids_band_chunked`, `get_enk_bandrange`) relocated to `common/wfn_transforms.py` (`04ff0a9`, now 1876 L); 22 call-sites repointed (`ab5d11f`); `common.get_enk_bandrange` stays public via the `__init__` re-export sourced from wfn_transforms. Verified: bispinor COHSEX e2e eqp0/sigma_diag bit-identical. (Residual `load_wfns` mentions in code are stale comments/docstrings + egg-info, not live imports.)

Fully-dead / archived (Tier F — evidence in `archive/DEAD_CODE.md`). *Status 2026-07-02:*
- ✅ **DELETED**: `psp/archive/{projector_pipeline,build_projectors}.py`, `solvers/cg_posdef.py`,
  `centroid/centroid_io.py`, `common/chi_sos.py`, `aot_memory_model/` (whole package incl. `chooser.py`).
- 🟡 **KEPT (reduced, not deleted)**: `gw/vcoul.py` — was to be deleted as a superseded shim, instead
  **stripped 234→68 L** (2 live helpers retained).
- ⬜ **STILL PRESENT**: `gw/experimental/head_wing_schur.py` (zero prod callers) — not yet removed.
- 🔄 **NO LONGER DEAD**: `common/bispinor_init.py` (was flagged test-oracle-only) is now a **live B3 home** —
  it hosts `lift_to_4spinor` + shared `HALFALPHA`, consumed by `wfn_loader` (`842c3c6`). Keep. *(2026-07-03)*

---

## 5. Gate reality (from archive/GATE_AUDIT.md) — **substantially improved 2026-07-02**

**The two `e7b6c7d` failures are FIXED and new gates landed** *(status 2026-07-02)*:
- ✅ **Crash fixed** (`86349a0`): the one-shot QP-WFN dump no longer crashes on full-BZ Σ.
  `gw_jax.py` now guards `debug.write_wfn_h5 and not self_consistent` and, when Σ lives on
  `nk_full > wfn.nkpts`, **skips the dump with a message** instead of asserting. The eqp compare
  now runs on IBZ fixtures.
- ✅ **Reference re-frozen** (`143dd99`): the COHSEX gate reference (`tests/regression/cohsex_debug/eqp_ref.dat`)
  was re-frozen from current main, absorbing the plateau-shaped drift so the gate is green and
  meaningful again.
- ✅ **New gates**: GN-PPM regression gate (`e7646e1`, parametrized cohsex + gnppm; `6dbb3b4` fixed a
  1-GPU/1×1-mesh GN-PPM crash to make it runnable) and **`test_ibz_full_bz_equivalence`** (`1479162`,
  1-GPU MoS₂, catches wrong IBZ→full-BZ unfolds — the exact class of bug §5/B4 warned about).
  `LORRAX_FORCE_FULL_BZ` is the dev/gate seam that drives the full-BZ leg (NOT a user-facing flag).
- ✅ **First BGW-anchored 3D gate** (`d55c4cb`, *2026-07-03*): **Si 4×4×4 no-SOC COHSEX**,
  `tests/regression/si_cohsex_debug/` + `si_cohsex_3d` case — native body + BGW head scalars,
  anchored to ~few-meV, atol 1e-3. **This closes gate-prereq #2 (3D COHSEX)** — the sys_dim=3 /
  truncation-off path now has a value gate, not just the 2D MoS₂ fixtures.
- ✅ **"Red means red"** (`25067f4`, *2026-07-03*): 4 container-JAX env failures conditionally
  skip-marked, 3 golden gates wired into `skills/checkpoint/SKILL.md`; suite = **242 passed /
  24 skipped / 0 failed** (no green-washing of real failures).
- ✅ **Two correctness fixes** (`61ae4b8`, *2026-07-03*): cohsex.in parser now strips inline `#`
  comments; `wfn_writer` nspinor=1 occupation factor-of-2 fixed (`nelec//2`).

**Still-open coverage gaps** *(updated 2026-07-03)*: value gates now cover static-COHSEX **and**
GN-PPM eqp on a 2D/charge/IBZ MoS₂ fixture, IBZ-vs-full-BZ equivalence, **and Si 3D COHSEX**. Still
**ungated end-to-end**: HL-PPM/real-axis, bispinor beyond unit bookkeeping, 3D **GN-PPM**/head-off,
self-consistency, all multi-GPU.

Coverage: the *original* end-to-end value gate was static-COHSEX eqp on a 2D/charge/IBZ/1-GPU MoS₂
fixture (now joined by GN-PPM + IBZ-equivalence, above). Still with **no dedicated test**:
`w_isdf.solve_w`, `cohsex_sigma`/`ppm_sigma` kernels, `sc_iteration`, `sigma_dispatch`,
`compute_vcoul` slab truncation, `degen_average`.

**⇒ Gate prerequisites #1 (GN-PPM e2e) + #2 (3D COHSEX, `d55c4cb`) + #4 (IBZ vs full-BZ) + crash-fix
+ re-freeze are now ALL DONE** — the stage-move gate wall is essentially in place for the
charge/IBZ path across 2D+3D COHSEX and GN-PPM. **Remaining coverage** is the ungated surfaces above
(HL/real-axis, bispinor value, head-off, SC, multi-GPU). See `archive/GATE_AUDIT.md §4` — seeds exist under
`runs/`. A stage-by-stage move-and-delete is only safe once each stage has a gate that a
wrong-but-plausible refactor would fail. *(The ζ-fit stage move in §4 #2 landed under exactly this
protection — all 4 e2e gates green + a ULP bit-identity guard.)*

---

## 6. Recommended attack order

0. ✅ **Gate first** — DONE (2026-07-03): e2e crash fixed, reference re-frozen, GN-PPM + IBZ-vs-full-BZ gates added, **+ Si 3D COHSEX BGW-anchored gate** (`d55c4cb`). The 3D-COHSEX gate prereq is now closed. §5.
1. 🟡 **Delete pass** (Tier F, no logic moves) — MOSTLY DONE: `aot_memory_model` package, `v_q_tile` subsystem, archive/{projector_pipeline,build_projectors}, cg_posdef, centroid_io, chi_sos all deleted; vcoul stripped (not deleted); **`common/load_wfns.py` facade deleted** (`bb04399`). *Leftover:* `experimental/head_wing_schur.py` still present (`bispinor_init.py` is now live B3, keep). ~16.4k L deleted total.
2. 🟡 **Single-source the duplicates** — PARTIAL: cohsex.in parser ×N→1 ✅; `_rotate_psi` 4→2 ✅; vcoul single-sourced ✅; **load_wfns facade→wfn_transforms ✅ (#10)**; **reader mf_header/isdf_header boilerplate→binders ✅** (`94cc354`). *Still open:* eqp math ×4→1 (#5), zeta reader old/new→1 (#8, deferred — needs padded-μ fixture).
3. ⬜ **Extract the two smeared concerns** — NOT STARTED: C6 planner still 2-in-tree (gw_init `compute_optimal_chunks` + gflat); B4 symmetry unification (one `SymMaps` table + one sym-action helper) untouched — the v_q_tile unfold copy died with the tile but no unification landed. Matches `feedback_unified_sym_action`. *(Note: the ψ-unfold move to `symmetry_maps.trs_augment_U`/`tau_phase_row` in §4#4 consolidates the ψ side toward `SymMaps` — a down-payment on this step, not the full unification.)*
4. 🟡 **Stage-by-stage move** — STARTED: **isdf_fitting→gw/ + split into `isdf/` core ✅ (#2)**; wfn_loader de-physics 🟡 mostly done (#4 — B3 lift + B4 ψ-unfold moved out, slurp lazy); w_isdf quadrature→minimax ⬜ (#9). (v_q_tile "split" is moot — deleted.)
5. ⬜ **Cross-cutting last**: head-correction seams, streamed-Σc head injection, shadow env-flag surface fold into `LorraxConfig`.

Suspected bugs (74, `archive/DEAD_CODE.md` top) are **not** refactor work — triage separately; several
(streamed-head skip, negative-Ω² head sign flip, stale vcoul cache key, wfn_writer factor-of-2,
`gw_jax:237` latent NameError) are live-wrong-answer paths worth a physics look before moving code.
