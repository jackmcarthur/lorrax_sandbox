# File group: kin_ion_io

## src/gw/kin_ion_io.py (220 LOC)

### Purpose

Standalone CLI preprocessing driver that computes the static one-body matrix
elements `T + V_loc + V_NL` (i.e. `H_DFT - V_xc - V_H`) for every k-point in
the full BZ and every band pair, and writes them to `kin_ion.h5`. It processes
one k-point at a time to bound GPU memory (the older "all-at-once" path was
removed 2026-05-04 per the module docstring). The output is a required input
of the main GW driver (`gw.gw_jax` loads it via
`file_io.kin_ion.load_kin_ion_submatrix`) and of `gw.eqp_bgw`.

Category: **preprocessing tool (I/O driver): kin+ion H matrix-element stage**.
It is a thin orchestration layer; the actual operator physics lives in
`psp.get_DFT_mtxels`, `psp.dft_operators`, `psp.vnl_ops`,
`psp.radial.build_projectors_qe`.

### Module-level environment setup (lines 14-18)

Before importing jax:
```
JAX_ENABLE_X64=1
JAX_PLATFORMS=cuda,cpu
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```
`setdefault`, so ambient environment wins. The `platform` allocator choice is
deliberate for low peak memory (synchronous cudaMalloc, no BFC arena) — this
differs from the main gw_jax driver's allocator posture.

### Function table

| Function | Lines | Role |
|---|---|---|
| `_resolve_against(path, base_dir)` | 39-40 | Join relative path against the input-file directory. Private helper; called only by `main` (line 91). |
| `get_kin_ion_k(wfn_k, Gk_crys, kvec, V_loc_r, vnl_setup, wfn, g_mask=None)` | 43-71 | Physics per k-point: `H_k[m,n] = <ψ_mk| T + V_loc + V_NL |ψ_nk>`. Composes three delegated kernels: `compute_kinetic_k` (T_mn = Σ_G ψ*_m(G) ½|k+G|²_bdot ψ_n(G), Ry via bdot metric), `compute_local_V_k` (V_mn = ∫ ψ*_m(r) V_loc(r) ψ_n(r) dr on the FFT box, normalized by cell_volume), and `vnl_matrix_from_kdata` (Kleinman-Bylander separable V_NL: Σ_lm,ij β projections with D_ij). Returns `(nb, nb)` complex device array. `V_NL_k = 0.0` scalar if `vnl_setup is None` (skip). |
| `main(argv=None)` | 74-216 | CLI driver; see walkthrough below. Returns 0. |
| `__main__` guard | 219-220 | `raise SystemExit(main())`. |

### main() walkthrough

1. **argparse** (75-83): `-i/--input` (cohsex.in, required), `-o/--output`
   (default `<input_dir>/kin_ion.h5`), `-n/--nb` (band-count override),
   `--sys_dim` (0/2/3, CLI > input file > default 3), `--pseudo_dir`
   (default = input-file dir).
2. **Input parse** (90-113): `read_cohsex_input` (from `psp.get_DFT_mtxels`).
   cohsex.in keys consumed: `wfn_file` (default "WFN.h5"), `sys_dim`
   (default 3), `nval` (default 5), `ncond` (default 5), `nband`
   (default 100), `bispinor` (default False). Note: this is the raw
   dict-style parser, NOT `gw.gw_config.LorraxConfig` — a parallel config
   path (see redundancy suspects).
3. **WFN + symmetry** (99-101): `WfnLoader` (aliased `WFNReader`) on wfn_path;
   `symmetry_maps.SymMaps(wfn)`;
   `Meta.from_system(wfn, sym, nval, ncond, nb_eff, 0, bispinor)` with
   `nb_eff = clamp(nb_req, 1, wfn.nbands)`.
4. **Pseudos** (117-126): `load_pseudopotentials(pseudo_dir)`; if empty,
   fallback probes `<input_dir>/../qe/scf` then `<input_dir>/../qe/nscf`
   (hard-coded sandbox run-directory layout).
5. **Validation** (129-134): `validate_operator_inputs(pseudos, wfn, sys_dim,
   caller="kin_ion_io")` from `psp.operator_checks` → raises if pseudos
   missing / sys_dim invalid; yields `ctx.truncation_2d` (2D slab vs 3D bulk
   Coulomb for the local ionic potential).
6. **Structure data** (137-153): `build_atom_pp_assignments(atom_crys,
   atom_types, pseudos)`; then groups atoms by species keyed on
   `id(ap.pseudo)` (object identity!) into `species_payload =
   [(pseudo, positions (na_s,3) float array)]`, skipping atoms with
   `ap.pseudo is None` silently.
7. **V_loc build** (156-171): `build_local_ionic_potential_on_G_total(
   assignments, species_groups, fft_grid=(nx,ny,nz), bdot, cell_volume,
   bvec, blat, truncation_2d)` → real-space local ionic potential
   `V_loc_r (nx,ny,nz)` float64, moved to device. k-independent; built once.
8. **V_NL setup** (173-182): `vnl_ops.build_vnl_setup(wfn, sym, meta,
   pseudos, nspinor=wfn.nspinor)` — unified projector tables shared across
   k-points. `None` if no pseudos (V_NL silently skipped in that case —
   but step 5 already raised on missing pseudos, so this branch is
   effectively unreachable-dead defensive code).
9. **Per-k loop** (186-200): host result buffer
   `kin_ion_all (nk_tot, nb_eff, nb_eff) complex128` (host, replicated,
   no sharding). Per ik:
   `wfn_k = load_kpoint_fftbox(wfn, sym, meta, ik, nb_eff)`
   (`(nb, nspinor, nx, ny, nz)` device),
   `kvec = sym.unfolded_kpts[ik]`,
   `Gk_crys, _ = generate_gvectors_k(ik, sym, wfn, meta)` (`(nG,3)` int),
   `H_k = get_kin_ion_k(...)`, copied to host with `np.asarray`, `del wfn_k`.
   Progress print every 16 k-points. `g_mask` parameter of `get_kin_ion_k`
   is never passed from main (always None here).
10. **HDF5 write** (204-212): see I/O below.
11. **Timing** (85, 99, 157, 193, 204, 215): `common.timing`
    `reset()/section()/report()` instrumentation throughout.

### Key arrays crossing boundaries

| Array | Shape | Residency |
|---|---|---|
| `wfn_k` | (nb, nspinor, nx, ny, nz) complex | device, per-k, freed each iter |
| `Gk_crys` | (nG, 3) int | host→kernel |
| `V_loc_r` | (nx, ny, nz) float64 | device, persistent across k loop |
| `vnl_setup` | VNLSetup (projector tables) | built once, persistent |
| `H_k` | (nb, nb) complex | device → host copy per k |
| `kin_ion_all` | (nk_tot, nb, nb) complex128 | host, single-process, unsharded |

No einsums appear in this file itself (all delegated to psp kernels).

### Entry points and callers

- `main` ← `python -m gw.kin_ion_io` only. Documented/invoked in:
  `AGENTS.md`, `docs/quickstart.md`, `docs/architecture/codebase.md`,
  `docs/installation/perlmutter.md`, `docs/theory/{overview,physics}.md`,
  `config/README.md`, sandbox `skills/build_inputs/SKILL.md` and
  `skills/execute_workflow/SKILL.md`. Grep across src/tests/tools/scripts
  found NO Python import of `gw.kin_ion_io` — CLI-only entry point.
- `get_kin_ion_k` ← only `main` in this same file (grep
  `get_kin_ion_k` over src, tests, tools, scripts: zero external hits).
  `src/psp/dft_operators.py:5` mentions kin_ion_io in a comment as an
  intended caller of its helpers, consistent.
- **Downstream consumers of the output file** `kin_ion.h5`:
  - `src/file_io/kin_ion.py::load_kin_ion_submatrix` ← `gw.gw_jax.main`
    (gw_jax.py:467) — the GW driver's static one-body input.
  - `src/gw/eqp_bgw.py` (lines 325, 381-386) reads `kin_ion.h5` directly
    with h5py for the BGW-style eqp0 assembly.
  - `src/gw/sc_iteration.py` consumes `kin_ion_dft` passed from gw_jax
    (shape-validated against the file).
- Config plumbing: `gw.gw_config` defines `kin_ion_file` (default
  `"kin_ion.h5"`, gw_config.py:166,515,909) — consumed by gw_jax's loader,
  NOT by kin_ion_io itself (which hard-codes the default output name).

### Flags / input keys consumed

- CLI: `-i/--input`, `-o/--output`, `-n/--nb`, `--sys_dim`, `--pseudo_dir`.
- cohsex.in keys (via `read_cohsex_input`): `wfn_file`, `sys_dim`, `nval`,
  `ncond`, `nband`, `bispinor`.
- Env (setdefault): `JAX_ENABLE_X64`, `JAX_PLATFORMS`,
  `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_PYTHON_CLIENT_ALLOCATOR`.

### I/O

- **Reads**: `WFN.h5` (BGW-format wavefunctions, via `file_io.WfnLoader`);
  `*.upf` pseudopotential files (via `psp.pseudos.load_pseudopotentials`,
  with `../qe/scf` and `../qe/nscf` fallback dirs); `cohsex.in` text input.
- **Writes**: `kin_ion.h5` — single dataset `kin_ion`
  `(nk_tot, nb_eff, nb_eff)` complex128 in Ry, with attrs `description`,
  `nk`, `nb`, `sys_dim`, `truncation_2d`, `pseudopotentials` (stringified
  list of keys).

### Dead suspects

- None hard-dead. `get_kin_ion_k`'s `g_mask` parameter: never supplied by
  any caller (only caller is `main`, which omits it → always None); the
  masking capability is exercised only through the underlying psp kernels'
  other callers. Evidence: grep `get_kin_ion_k` across
  src/tests/tools/scripts → only this file.
- Lines 173-174 `if pseudos:` guard around `build_vnl_setup` is effectively
  unreachable-false: `validate_operator_inputs` at line 129 already raises
  when pseudos is empty (per its "will raise if pseudos missing" comment),
  so `vnl_setup` can never end up None in practice.

### Redundancy suspects

- **species_tmp/species_payload grouping block (lines 142-153) is a
  copy-paste of `psp/get_DFT_mtxels.py` lines ~460-471 and again ~642-658**
  (three occurrences of the same group-atoms-by-`id(pseudo)` idiom across
  the codebase). Should be one shared helper in psp.
- **Pseudo-dir fallback probing (lines 119-126)** duplicates, with a
  different convention, the QE-save search logic in
  `src/centroid/charge_density.py:155-199` (ancestor-walk `qe/scf`,
  `qe/nscf` probing). Two independent path-guessing implementations.
- **Parallel config parsers**: this driver uses the legacy dict-style
  `read_cohsex_input` from `psp.get_DFT_mtxels`, while the main GW driver
  uses `gw.gw_config.LorraxConfig` (which itself carries a `kin_ion_file`
  key kin_ion_io never sees). Defaults can drift (e.g. nband=100 default
  here vs whatever gw_config defaults to).

### Weird code

- **Line 146: `key = id(ap.pseudo)`** — dict keyed on Python object
  identity to group atoms by species. Works because
  `load_pseudopotentials` returns one shared object per species, but is
  fragile under any future copy/deserialization of pseudo objects
  (silently splits a species into duplicates).
- **Lines 121-122: hard-coded `'../qe/scf'` / `'../qe/nscf'` fallbacks** —
  sandbox run-layout knowledge baked into library source.
- **Lines 14-18: env mutation at import time** (before jax import),
  including `XLA_PYTHON_CLIENT_ALLOCATOR=platform`; any code importing
  this module (none currently do) would get its allocator changed as a
  side effect.
- **Line 62: `V_NL_k = 0.0`** — float scalar broadcast-added to a complex
  matrix when vnl_setup is None; fine numerically, but means "V_NL skipped"
  is silent at the physics level (no attr in the output records it).
- **Lines 143-148: atoms with `ap.pseudo is None` skipped silently** in
  the species grouping (validation should have caught it earlier, but the
  silent `continue` masks partial-assignment bugs).
- **Docstring lines 6-8**: notes removal of the old all-at-once path on
  2026-05-04 — historical marker; the promised single path is indeed
  what's here (good, no parallel path remains).
- Default `nval=5`, `ncond=5`, `nband=100` magic fallbacks at lines
  103-105; nval/ncond only feed `Meta.from_system` and do not affect the
  computed matrix elements (all `nb_eff` bands are processed regardless).
