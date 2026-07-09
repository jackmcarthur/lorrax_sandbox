# psp / pseudopotential group — refactor-map notes (2026-07-01)

Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. All paths relative.
Caller greps were run over `src/`, `tests/`, `tools/` (absent), `scripts/` (absent) —
only `src/` and `tests/` exist; `src/psp/tests/` and `src/psp/archive/` included.

Big picture of the group: UPF XML → dataclass (`upf/`), dataclass → clean numpy
per-species bundle (`species.py`, `pseudos.py`), radial → q-space Hankel tables
(`radial_tables.py`, `radial/radial_jax.py`), angular/spin-orbit algebra + V_loc(G)
builder (`radial/build_projectors_qe.py`, `radial/solid_harmonics.py`). Consumers:
`psp/vnl_ops.py`, `psp/ionic_gspace.py`, `psp/dft_operators.py`, `psp/get_DFT_mtxels.py`,
`psp/get_dipole_mtxels.py`, `psp/run_sternheimer.py`, `psp/run_nscf.py`, `psp/kpm_dos.py`,
`gw/kin_ion_io.py`.

**Group-level headline for the refactor:** there are THREE parallel radial-table
stacks — (1) `psp/radial_tables.py` (new, SpeciesData-based, batched-JAX Hankel),
(2) `psp/radial/radial_jax.py` `make_*_table` numpy/RadialTable path (used by
`build_projectors_qe` + `dft_operators`), and (3) `psp/archive/build_projectors.py`
(archived copy). The old real-spherical-harmonics + per-row projector construction in
`build_projectors_qe.py` (~lines 100–360, 690–895) is mostly only reachable from
`psp/archive/projector_pipeline.py`; production uses `solid_harmonics.py` +
`vnl_ops`/`dft_operators` instead. Only `build_E_blocks_full` and
`build_local_ionic_potential_on_G_total` in `build_projectors_qe.py` are live in
production.

---

## src/psp/upf/load_upf.py (179 loc)

**Purpose.** Parse a UPF v2.0.1 pseudopotential XML file into xsdata dataclasses
(`psp.upf.upf_model_2_0_1.Upf`). Pre-processes the XML (renames `PP_BETA.n` tags →
`PP_BETA` + `index=n` attribute via a temp-file rewrite) and annotates each beta with
FR `jjj`/`lll` pulled from `PP_SPIN_ORB/PP_RELBETA.n`. Also has a debug dataclass-tree
printer and a `__main__` block.

**Category.** I/O: UPF pseudopotential reader.

**Functions.**
- `detect_version(upf_path)` — sniff root XML tag; 'UPF' → '2.0.1', else '0.99'. Called only inside `load_upf` (no external callers).
- `load_upf(upf_path)` — main entry; XML preprocess + xsdata parse + jjj/lll annotation. Callers: `psp/pseudos.py:load_pseudopotentials`, `psp/get_DFT_mtxels.py` (two import variants, lines 56/67), `tests/archive/plot_upf_projectors.py`, `tests/archive/plot_vloc.py` (both via stale path `psp.load_upf`).
- `_brief_value`, `_print_dataclass_tree`, `print_dataclass_tree` — debug pretty-printer. Zero external callers.
- `main(argv)` — CLI; but the argv handling is commented out and the path is HARDCODED to `tests/cohsex_prod/Mo_ONCV_PBE_FR-1.0.upf` (line 161). Debug leftover.

**I/O.** Reads: `*.upf` / `*.UPF` XML (UPF v2.0.1 schema; datasets PP_HEADER, PP_MESH/PP_R/PP_RAB, PP_LOCAL, PP_NLCC, PP_NONLOCAL/PP_BETA.n/PP_DIJ, PP_SPIN_ORB/PP_RELBETA.n). Writes a NamedTemporaryFile `.upf` copy (deleted afterward).

**Dead suspects.**
- `print_dataclass_tree` / `_print_dataclass_tree` / `_brief_value`: grep `print_dataclass_tree` over src/tests → only this file. Debug-only.
- 0.99 branch of `load_upf` (lines 96–106): imports `from psp import upf_model` — **module does not exist** (`src/psp/upf_model.py` absent, verified). Any non-2.0.1 UPF → ImportError. Dead AND broken.
- `main()` — only `__main__`; hardcoded path makes it useless as a CLI.

**Weird code.**
- Line 161: hardcoded `"tests/cohsex_prod/Mo_ONCV_PBE_FR-1.0.upf"` shadowing argv; usage check commented out (158–160). Hypothesis: debugging session leftover, never reverted.
- Lines 34–95: blanket `try/except Exception: pass` nesting (5 levels) around preprocessing and jjj annotation; a malformed PP_RELBETA silently yields projectors without `jjj`, which downstream defaults to j=l+1/2 (see build_projectors_qe/species) — silent physics error path.
- Temp-file round-trip to rename tags instead of in-memory parse; works but odd.

**Redundancy.** `get_DFT_mtxels.py` re-imports load_upf both relatively and absolutely (lines 56/67) — path-hack duplication in the caller, not here.

---

## src/psp/upf/normalize.py (57 loc)

**Purpose.** Post-parse normalization: recursively walk the UPF dataclass tree and
convert any string field literally named `value` containing whitespace-separated
numbers into `np.ndarray` (via `np.fromstring`). Preserves dynamically attached attrs
(`jjj`, `lll`).

**Category.** I/O: UPF post-parse normalizer.

**Functions.**
- `normalize_dataclass(obj)` — the entry. Callers: `psp/pseudos.py:load_pseudopotentials`, `psp/get_DFT_mtxels.py` (55/66), `tests/archive/plot_upf_projectors.py`, `tests/archive/plot_vloc.py`, `psp/upf/load_upf.py:main`.
- `_convert_value_field_if_numeric` — internal.

**I/O.** None directly (operates on parsed objects).

**Weird code.**
- Line 15: `np.fromstring(text, sep=' ')` — deprecated NumPy API (DeprecationWarning; slated for removal). Should be `np.fromstring`→`np.array(text.split(), dtype=float)` or `np.loadtxt`. Silent failure path if it starts raising.
- Lines 47–56: re-attaches non-field attrs after `dataclasses.replace` with try/except — needed because load_upf smuggles `jjj/lll` as dynamic attrs. Fragile contract between the two files.

**Dead/redundancy.** None.

---

## src/psp/upf/upf_model_2_0_1/qe_pp_2_0_1.py (1807 loc)

**Purpose.** xsdata-GENERATED dataclass model of the QE UPF v2.0.1 XML schema
(from `src/psp/qe_pp-2.0.1.xsd`). ~45 dataclasses/enums (PpHeader, PpMesh, PpR, PpRab,
PpLocal, PpNlcc, PpNonlocal, UpfPpBetaN, PpDij, PpSpinOrb, UpfPpRelbetaN, PAW/GIPAW
sections, root `Upf`). Pure data schema; no logic.

**Category.** I/O: generated XML schema bindings (vendored/generated code — exclude from refactor except regeneration).

**Entry points.** All classes re-exported by `upf_model_2_0_1/__init__.py`; consumed as `model.Upf` in `load_upf.py` (only external consumer). Individual class names (PpHeader etc.) are only touched via attribute access on parsed objects, not imported anywhere else (grep `from psp.upf.upf_model_2_0_1` → only `load_upf.py` + package `__init__`).

**I/O.** Defines the UPF v2.0.1 format shape; no file access itself.

**Dead suspects.** Most PAW/GIPAW/semilocal classes (PpPaw, PpGipaw*, PpSemilocal, PpMultipoles, PpAugmentation, UpfPpQijlNNN…) are never populated by the NC pseudos LORRAX uses — but they're generated code; harmless.

**Weird code.** None (generated). Note `UpfPpBetaN` Meta name is `PP_BETA` (not `PP_BETA.n`) — that's why load_upf does the tag-rename preprocess.

---

## src/psp/species.py (133 loc)

**Purpose.** One clean pass from parsed UPF objects → `SpeciesData` numpy bundle
(radial grid r/rab, vloc_r, NLCC ρ_core, β(r)/r projectors with l/j, D-matrix,
nspinor). Also maps crystal atoms → species indices with padded tau arrays. This is
the "new path" front-end feeding `radial_tables.build_all_tables`.

**Category.** preprocessing: UPF → per-species numpy bundle (physics setup).

**Functions.**
- `SpeciesData` dataclass — the bundle.
- `extract_species(pseudos, nspinor)` <- `psp/vnl_ops.py:108` (build_vnl_setup), `psp/ionic_gspace.py:153`.
- `build_atom_species_map(crystal, species_list)` <- same two callers (`vnl_ops.py:137`, `ionic_gspace.py:169`).

**I/O.** None directly (consumes load_upf output).

**Weird code.**
- Line 85–86: β(r=0) set to 0.0 via `np.where(r>0, raw/safe_r, 0.0)`, whereas `build_projectors_qe.py` (lines 407–411, 877–882) extrapolates `beta_r[0] = beta_r[1]`. Two different r→0 conventions for the same quantity in the same package. Probably immaterial (r² weight kills it) but a convention fork.
- Line 88–89: `proj_j` defaults to `l + 0.5` when `total_angular_momentum` attr is missing — silently wrong for the j=l−1/2 channel if load_upf's jjj annotation failed (see load_upf's swallow-all except). NOTE it reads attr `total_angular_momentum` while load_upf attaches `jjj` — works only because normalize preserves... actually `UpfPpBetaN` schema itself has `total_angular_momentum`; `jjj` is the fallback annotation used by build_projectors_qe instead. Two attribute-name conventions for j.

**Redundancy.** Projector extraction (β/r, l, j, dij reshape) duplicates the ad-hoc extraction inside `build_projectors_qe._form_factors_qe` / `precompute_projector_tables` / `build_E_blocks_full`, which grope the raw UPF object with getattr chains. SpeciesData is the clean successor; the getattr path survives for the archive pipeline + build_E_blocks_full.

---

## src/psp/pseudos.py (123 loc)

**Purpose.** Minimal pseudopotential entry point: periodic table symbol→Z, directory
scan loading all `*.upf` via load_upf+normalize, atom→pseudo assignment
(`AtomPP`), and a structure/PSP summary printer. Extracted from get_DFT_mtxels.

**Category.** I/O: pseudopotential loading / lookup utility.

**Functions.**
- `symbol_to_Z(symbol)` <- `psp/species.py`, `psp/get_DFT_mtxels.py:183` (as `_symbol_to_Z`), `psp/operator_checks.py:86`, `psp/archive/charge_density.py` (via get_DFT_mtxels re-export).
- `_symbol_to_Z` — backwards-compat alias, re-exported through get_DFT_mtxels; archive/charge_density imports it from there.
- `load_pseudopotentials(work_dir)` <- `gw/kin_ion_io.py:118,123`, `psp/run_sternheimer.py:1116`, `psp/kpm_dos.py:338`, `psp/get_DFT_mtxels.py:876`, `psp/get_dipole_mtxels.py:545,550`, `psp/run_nscf.py:38`, `src/psp/tests/test_dft_hamiltonian.py:40`, doc-referenced in `scf_potential.py`/`operator_checks.py`.
- `AtomPP` dataclass; `build_atom_pp_assignments(...)` <- `gw/kin_ion_io.py:139`, `get_DFT_mtxels.py:455,641`, `src/psp/tests/test_dft_hamiltonian.py:78`.
- `print_atomic_structure(wfn, pseudos)` <- `get_DFT_mtxels.py:879`, `get_dipole_mtxels.py:556`.

**I/O.** Reads directory glob `*.upf`/`*.UPF` (delegates parse to load_upf). Attaches `_source_path` attr to each pseudo.

**Weird code.**
- Lines 61–68: per-file `try/except Exception` prints a warning and continues — a corrupt pseudo silently disappears from the dict; downstream `z_to_pseudo.get(Z)` then yields `AtomPP(pseudo=None)` which most consumers don't guard.
- `_SYMBOLS` table is fine; alias `_symbol_to_Z` kept for backwards compat (line 37) — mini parallel-name cruft.

**Dead suspects.** None (all four public functions have callers).

---

## src/psp/radial_tables.py (228 loc)

**Purpose.** "New path" radial→q Hankel table builder: SpeciesData → uniform-q tables
for V_loc^SR (erf-subtracted), NLCC core charge, β form factors F_l(q), and the raw
H_{l+1} integral for the analytic dF/dq. `build_all_tables` batches per unique l on
GPU via `radial_jax.spherical_hankel_table_batch_jax`; the standalone numpy functions
at the top are the reference/scalar versions.

**Category.** physics: pseudopotential radial→G-space tables (setup stage).

**Functions.**
- `hankel_l(l, r, f_r, q, rab)` — numpy Simpson Hankel core. **Zero callers outside this file** (grep `hankel_l` over src/tests → only radial_tables.py itself). Only used by the four thin wrappers below.
- `vloc_sr_table(sp, q)` — **zero callers anywhere** (grep `vloc_sr_table` → nothing outside file).
- `core_charge_table(sp, q)` — **zero callers anywhere**.
- `projector_table(sp, ip, q)` — **zero callers anywhere** (careful grep excluding `make_projector_table`).
- `projector_deriv_table(sp, ip, q)` — imported by `vnl_ops.py:28` as `_projector_deriv_table` but **never called in vnl_ops** (grep `_projector_deriv_table` in vnl_ops → import line only; the analytic-deriv logic was inlined using `build_all_tables` deriv_tables + q^l division at vnl_ops:171–173). Dead import; big docstring is the only live documentation of the dG_l/dq formula though.
- `alpha_z(sp, vol)` <- `psp/ionic_gspace.py:154,176` (G=0 local-potential term).
- `_simpson_weights(n_r)` — internal to build_all_tables.
- `build_all_tables(species_list, q_max, n_q=4000)` <- `psp/vnl_ops.py:109,136` and `psp/ionic_gspace.py:154,168`. Returns dict {q, dq, vloc, nlcc, has_vloc, has_nlcc, proj_tables, deriv_tables}.

**I/O.** None (numpy/JAX arrays in-memory).

**Dead suspects.** `hankel_l`, `vloc_sr_table`, `core_charge_table`, `projector_table` are a self-contained reference implementation with no callers; `projector_deriv_table` reachable only via a dead import in vnl_ops. Evidence: grepped each name over src/, tests/ — no hits outside the file except the vnl_ops import line. They duplicate `build_all_tables` internals; keep at most one as documented reference.

**Redundancy suspects.**
- Simpson weights implemented THREE times just in this file (`hankel_l` inline, `alpha_z` inline, `_simpson_weights`) plus a fourth in `radial_jax.radial_weights(scheme="simpson_rab")`.
- `vloc_sr_table`/erf-SR integrand (lines 50–56) duplicated inside `build_all_tables` (177–181) and again in `radial_jax.make_local_sr_table` (306–323).
- Whole-file overlap with `radial/radial_jax.py`: two independent Hankel-table stacks (see radial_jax notes).
- `alpha_z` duplicates the inline alphaZ computation in `build_projectors_qe.build_local_ionic_potential_on_G_total` (lines 512–524).

**Weird code.**
- `n_q=4000` default, magic `max(q_max, 1e-8)`; `deriv_tables` intentionally returns the UNSCALED H_{l+1} — the q^l division contract is documented in the docstring but enforced only by convention in `vnl_ops` (cross-file coupling, easy to misuse).
- `e2 = 2.0` (Ry units e²) repeated as a bare literal here and in 3 other files.

---

## src/psp/radial/radial_jax.py (352 loc)

**Purpose.** Declared "single backend for UPF radial data": uniform-q `RadialTable`
dataclass with linear interp (numpy + jitted JAX twins), integration-weight schemes,
FD table derivative, custom JAX spherical Bessel j_l (Miller backward recurrence, safe
in jit), JAX Hankel tabulators (single + batched), a scipy CPU Hankel, and `make_*`
table constructors (projector F_l, V_loc SR, NLCC).

**Category.** physics/numerics: radial transform kernel library.

**Functions & callers.**
- `RadialTable` (+`__call__`, `.derivative()`) <- `psp/dft_operators.py:768,793,836`, `build_projectors_qe.py` (494, 773, 851…), `archive/projector_pipeline.py`.
- `make_uniform_q_grid` <- `build_projectors_qe.py` (400, 492, 871), `archive/charge_density.py:260`.
- `radial_weights` <- `build_projectors_qe.py` (341, 374, 401, 872), `archive/charge_density.py:258`.
- `differentiate_uniform_table` <- `dft_operators.py:769,803`, `vnl_ops.py:26`.
- `interp_uniform_np` — **zero external callers**; used only by `RadialTable.__call__` in-module.
- `interp_uniform_jax` <- `dft_operators.py` (895, 916–917, 978–982), `ionic_gspace.py:25,101`.
- `spherical_jn_jax(l, x)` <- `archive/projector_pipeline.py:17` only (production Hankel goes through `spherical_hankel_table_batch_jax` which calls it in-module).
- `spherical_jn_deriv_jax` <- `archive/projector_pipeline.py:16,141` **only** (archive-only).
- `spherical_hankel_table_jax` (single-f) — **zero callers anywhere** (grep → only definition + `__all__`). Superseded by the batch version.
- `spherical_hankel_table_batch_jax` <- `psp/radial_tables.py:155,182,205,214`.
- `_spherical_hankel_table_np` — no external callers (in `__all__` though); used in-module by the three `make_*` constructors.
- `make_projector_table` <- `build_projectors_qe.py:342,412,883`.
- `make_local_sr_table` <- `build_projectors_qe.py:375`.
- `make_core_charge_table` <- `psp/archive/charge_density.py:261` **only** (archive-only; live NLCC goes through `radial_tables.build_all_tables` nlcc row).

**I/O.** None.

**Dead suspects.**
- `spherical_hankel_table_jax` — zero callers (grep over src/tests).
- `interp_uniform_np` — only internal via RadialTable.__call__ (fine, but the standalone export is unused).
- `spherical_jn_deriv_jax`, `make_core_charge_table` — archive-only callers.
- `RadialTable.derivative()` — grep `\.derivative(` → only dft_operators via `differentiate_uniform_table` directly, not through the method; method appears uncalled (grep `.derivative(` hits none in src outside definition).

**Redundancy suspects.**
- np/jax twin interps (`interp_uniform_np`/`interp_uniform_jax`) — acknowledged in docstring as intentional, still a mirrored pair to keep in sync.
- THREE Hankel implementations across the package: `_spherical_hankel_table_np` (scipy), `spherical_hankel_table_jax`(+batch), and `radial_tables.hankel_l` (numpy Simpson). The `make_*` constructors here parallel `radial_tables.{vloc_sr,core_charge,projector}_table` one-for-one — the classic old/new parallel-path pattern this codebase's rules forbid.
- `radial_weights(scheme='simpson_rab')` duplicates `radial_tables._simpson_weights` logic (with a Python loop instead of slicing).

**Weird code.**
- `spherical_jn_jax`: Miller recurrence magic constants `small = |x|<0.1`, `l_start = max(l+30, 80)`; Taylor fallbacks per l. Standard technique, but untested edge (grep tests → none reference it).
- `radial_weights('rab')` returns raw rab (trapezoid-in-log-grid ≈ rectangle rule) while the new path uses simpson*rab — quadrature-order difference between the two V_loc/projector paths; could cause tiny numeric diffs between paths.
- `make_uniform_q_grid`: `max(q_max, 1e-8)` magic.

---

## src/psp/radial/solid_harmonics.py (110 loc)

**Purpose.** QE-convention solid harmonics S_lm = r^l Y_lm as pure Cartesian
polynomials (l ≤ 3), autodiff-friendly, no singularities. Two forms: per-l
`solid_harmonics_jax` (Python branch on l) and branch-free padded
`all_solid_harmonics` for lax.scan/jit.

**Category.** physics: angular-momentum kernel (V_NL / dipole stage).

**Functions.**
- `solid_harmonics_jax(l, K_cart)` <- `psp/vnl_ops.py:27,445` (jvp for dV_NL/dk), `psp/dft_operators.py:766,897,920,926,985,987`.
- `all_solid_harmonics(K_cart, l_max=3)` <- `psp/vnl_ops.py:357,365,411,419`.

**I/O.** None.

**Dead suspects.** None.

**Redundancy suspects.**
- The two functions duplicate the identical polynomial coefficient tables (l=0..3 written out twice in one 110-line file). Also overlaps in purpose with `build_projectors_qe.qe_real_sph_harmonics` (`_ylmr2_qe` recurrence, numpy) and `qe_real_sph_harmonics_with_grad` — FOUR real-harmonic implementations in the package (two live, two archive-leaning).

**Weird code.**
- Sign flips `-c*x, -c*y` etc. — QE ylmr2 phase convention (matches `_ylmr2_qe`); intentional, but undocumented in-file beyond "QE ordering"; a refactor unifying with sph_harm conventions must preserve these.
- l>3 raises NotImplementedError (fine for ONCVPSP, would break for f-projector pseudos).

---

## src/psp/radial/build_projectors_qe.py (914 loc)

**Purpose.** Spin-orbit (FR) projector algebra following QE / Dal Corso & Mosca Conte
PRB 87, 115112: complex↔real harmonic rotations U^ℓ, Clebsch–Gordan α^{σ,ℓ,j},
spin-block tensors f^{σσ'} and assembled D-weighted E-blocks per ℓ (`build_E_blocks_full`,
the live V_NL spin structure); plus the live total-V_loc(r)-on-FFT-grid builder
(`build_local_ionic_potential_on_G_total`, with 2D slab truncation); plus a legacy
per-row projector construction path (real harmonics with grads, form-factor tables)
now used only by `psp/archive/projector_pipeline.py`.

**Category.** physics: spin-orbit projector algebra + V_loc builder (mixed live/legacy).

**Functions (grouped).**
Live production:
- `build_E_blocks_full(pseudo)` <- `psp/vnl_ops.py:25,153`, archive/projector_pipeline. Assembles (2,2,n_beta_l·(2ℓ+1), …) E-blocks from PP_DIJ × f-blocks per j-group.
- `build_local_ionic_potential_on_G_total(assignments, species_groups, fft_grid, bdot, cell_volume, bvec, blat, truncation_2d)` <- `gw/kin_ion_io.py:33,158`, `psp/get_DFT_mtxels.py:73,548,701` (doc refs in dft_operators/operator_checks/ionic_gspace). SR (erf-subtracted, RadialTable) + LR Gaussian-damped Coulomb tail + optional QE Coul_cut_2D factor + alpha-Z at G=0, IFFT to real space.
- Internal algebra: `m_to_idx`, `mj_grid`, `U_complex_from_real`, `alpha_sigma_lj`, `U_sigma_lj`, `f_blocks_lj` — all internal-only (zero callers outside this file; used by build_E_blocks_full chain).
Legacy (archive-only or dead):
- `qe_real_sph_harmonics(l, vectors)` / `_ylmr2_qe(lmax, vectors)` <- archive/projector_pipeline only.
- `qe_real_sph_harmonics_with_grad(l, vectors)` <- archive/projector_pipeline only (250, 354).
- `compute_type_projectors_real(...)` <- archive/projector_pipeline:104 only.
- `precompute_projector_tables(...)` <- archive/projector_pipeline (82, 522) only.
- `precompute_projector_splines(...)` — compat wrapper, **zero callers anywhere**.
- `spherical_hankel_transform_l_np` — public alias, **zero callers** (archive/build_projectors.py has its OWN copy of a function with the same name).
- `_complex_sph_and_angular_partials(l, theta, phi)` — **zero callers even in-file** (defined line 195, never invoked; `qe_real_sph_harmonics_with_grad` uses the ylmr2 recurrence instead). Dead, ~60 lines incl. scipy sph_harm FD partials + l≤1 hand-coded fallback.
- `E_spin_blocks_for_atom_l`, `E_all_l_for_atom`, `real_harmonic_slot` — zero callers outside file; `E_all_l_for_atom` and `real_harmonic_slot` have zero callers, period (E_spin_blocks_for_atom_l called only by E_all_l_for_atom).
- `_form_factors_qe`, `_fourier_transform_vloc_qe`, `_spherical_hankel_transform_l`, `_real_m_index_pairs`, `_angles_from_cart_vectors`, `_spherical_frame` — internal to the above.
- Comment markers of previous pruning: lines 167, 417, 707 "Removed: qe_complex_sph_harmonics / build_species_rows_qe / per_type_beta_m_slots (unused)".

**I/O.** None directly (consumes parsed UPF objects; returns arrays).

**Cross-module deps.** `psp.radial.radial_jax` (RadialTable, make_local_sr_table, make_projector_table, make_uniform_q_grid, radial_weights); jax/jnp only in the V_loc IFFT tail.

**Dead suspects (evidence = grep of each name over src/, tests/, incl. src/psp/tests):**
- `_complex_sph_and_angular_partials` — no call site anywhere including this file.
- `precompute_projector_splines` — no callers.
- `spherical_hankel_transform_l_np` — no callers (archive has its own copy).
- `E_all_l_for_atom`, `E_spin_blocks_for_atom_l`, `real_harmonic_slot` — no external callers; E_all_l chain fully unreferenced.
- Archive-only block (`qe_real_sph_harmonics{,_with_grad}`, `compute_type_projectors_real`, `precompute_projector_tables`, `_form_factors_qe`) — dies if `src/psp/archive/` is dropped.

**Redundancy suspects.**
- β(r)/r extraction with r=0 patch duplicated at lines 405–411 and 875–882 (and a third convention in species.py).
- `_fourier_transform_vloc_qe` is a 1-call wrapper around `make_local_sr_table`; `_spherical_hankel_transform_l` a 1-call wrapper around `make_projector_table`; `precompute_projector_splines` a rename-wrapper of `precompute_projector_tables` — three layers of compat shims.
- alpha-Z at G=0 inline (512–524, with its own trapz fallback) duplicates `radial_tables.alpha_z`.
- `_ylmr2_qe` per-G Python loop vs vectorized recurrence in `qe_real_sph_harmonics_with_grad` — same recurrence written twice, once scalar once vectorized.
- Real-harmonics machinery overall vs `solid_harmonics.py` (production).

**Weird code.**
- Line 539: Gaussian damping `np.exp(-0.25 * G2)` — QE's hardcoded eta=1/2 Ewald-style screening for the LR tail; magic constant, must match SR erf(r) (unit width) — undocumented pairing with the `scipy_erf(r)` (width-1) in make_local_sr_table.
- Lines 501–526: alpha-Z under `try/except Exception: pass` — a failure silently leaves V(G=0) wrong (constant potential shift).
- Line 551: 2D truncation `lz = π / B[2,2]` assumes the third reciprocal vector is exactly ẑ-aligned (diagonal cell); silently wrong for tilted c-axis cells. Also `cutoff_2d` zeroed only at G=0 exactly.
- `getattr(beta, 'lll', getattr(beta, 'angular_momentum', 0))` / `getattr(beta, 'jjj', l+0.5)` pattern everywhere — the j=l+0.5 silent default is a physics landmine for j=l−1/2 channels when PP_SPIN_ORB annotation failed upstream.
- Docstring says "F_splines"/"spl(0)" though splines were replaced by linear tables — stale naming (`precompute_projector_*splines*`, comment at line 507–509 about spl(0)).
- `q_points = max(1024, 2*len(r))` vs `max(2048, 2*len(r))` — two different magic minimums (lines 399 vs 491/870) for essentially the same tables.
- `_ylmr2_qe` line 141: `phi = sign(gy)*π/2` when |gx|<eps — QE-faithful but fragile corner handling; eps=1e-9 magic.

---

## Cross-file redundancy map (for the refactor)

| Concern | Implementations |
|---|---|
| Hankel transform | `radial_tables.hankel_l` (np), `radial_jax._spherical_hankel_table_np` (scipy), `radial_jax.spherical_hankel_table_jax` (dead), `radial_jax.spherical_hankel_table_batch_jax` (live), `archive/build_projectors.spherical_hankel_transform_l_np` |
| V_loc SR table | `radial_tables.vloc_sr_table` (dead), `radial_tables.build_all_tables` inline (live), `radial_jax.make_local_sr_table` (live via build_projectors_qe) |
| NLCC table | `radial_tables.core_charge_table` (dead), `build_all_tables` inline (live), `radial_jax.make_core_charge_table` (archive-only) |
| β form factor | `radial_tables.projector_table` (dead), `build_all_tables` inline (live), `radial_jax.make_projector_table` (legacy path), archive copy |
| Simpson weights | `radial_tables.hankel_l` inline, `radial_tables._simpson_weights`, `radial_tables.alpha_z` inline, `radial_jax.radial_weights('simpson_rab')` |
| alpha-Z (G=0) | `radial_tables.alpha_z` (live via ionic_gspace), inline in `build_local_ionic_potential_on_G_total` (live via get_DFT_mtxels/kin_ion_io) |
| Real/solid harmonics | `solid_harmonics.solid_harmonics_jax` + `all_solid_harmonics` (live, duplicated tables), `build_projectors_qe._ylmr2_qe`/`qe_real_sph_harmonics{,_with_grad}` (archive), `_complex_sph_and_angular_partials` (dead) |
| β(r)/r at r=0 | species.py → 0.0; build_projectors_qe (×2 sites) → extrapolate [1] |
| UPF loading imports | `pseudos.load_pseudopotentials` (canonical) vs direct load_upf+normalize in `get_DFT_mtxels` dual-import blocks |

Refactor direction consistent with the SpeciesData/`build_all_tables` "new path":
delete `radial_tables` dead reference functions OR make them the single source and
have build_all_tables call them; collapse radial_jax `make_*` layer into it; drop
`src/psp/archive/` to kill ~500 lines of build_projectors_qe legacy surface; keep
only `build_E_blocks_full` + `build_local_ionic_potential_on_G_total` there (or fold
V_loc builder into ionic_gspace, which its docstring says was the plan).
