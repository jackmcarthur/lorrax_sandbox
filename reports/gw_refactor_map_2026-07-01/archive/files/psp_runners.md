# psp runners group — detailed notes (gw_refactor_map 2026-07-01)

Files: `src/psp/run_sternheimer.py`, `src/psp/run_nscf.py`, `src/psp/nscf_input.py`, `src/psp/kpm_dos.py`
Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`
Grep scope for callers: `src/`, `tests/`, `tools/`, `scripts/` (tools/ and scripts/ contain no hits for any of these modules).

---

## src/psp/run_sternheimer.py (1528 LOC)

**Purpose.** Standalone driver computing the G'=0 column of the static density-density
response χ_{G'0}(q, ω=0) by insulating Sternheimer linear solves (projected/level-shifted CG,
QE-DFPT-style α_pv shift), instead of sum-over-states. Optionally computes ∂χ/∂q via
`jax.linearize` (with a "unfrozen projector" k·p linearisation of U_val) and the exact
S-tensor S_ij = ∂²χ_{00}/∂q_i∂q_j at q=0 via a P_val-rotation formula that avoids the k·p²
Hessian. Single-GPU (whole ψ buffer on device 0); "multi-GPU sharding will land in a later
commit" per the module docstring.

**Category.** physics: Sternheimer χ head/wing column (DFPT-style linear response); also
diagnostic/research driver — not on the main gw_jax pipeline.

**Entry points / callers.**
- `main` <- CLI only: `python3 -m psp.run_sternheimer` (docstring, `src/psp/PERFORMANCE.md`). No importers.
- `run_sternheimer` <- `main` only.
- Test-only imports (`src/psp/tests/test_sternheimer_jvp.py`): `_psi_box_to_G_sphere`,
  `build_Gprime_list`, `build_sternheimer_source`, `chi_col_contrib_at_kvec_traced`,
  `make_density_perturbation`, `make_umklapp_phase`.
- Everything else is internal to the file.

**Function table.**
| Function | Role |
|---|---|
| `_batched_real_norm_host` | batched L2 norm helper — **DEAD, zero callers** |
| `_gather_box_at_G` | gather FFT-box values at signed integer G indices (mod wrap) |
| `build_sternheimer_source_preQ` | V_pert(r)·u(r) gathered on (k−q) sphere, pre-Q-projection (used by driver) |
| `build_sternheimer_source` | same + Q_{k−q} projection (only used by test; driver superseded it with preQ + inline Q) |
| `accumulate_chi_density` | Σ_v u(r)·conj(δu_naive(r)) with wrapped→naive gauge unwrap phase |
| `project_density_to_Gsphere` | FFT δn(r), gather at target G' list |
| `make_density_perturbation` | V_pert cell-periodic part ≡ 1 (density response) |
| `make_umklapp_phase` | e^{±iG_wrap·r} gauge phase (test-only; driver uses `common.kq_mapping.umklapp_phase_box_batched`) |
| `compute_kp_tangent_at_kvec` | ∂u/∂k via 3 batched Sternheimer solves (k·p first order) |
| `compute_kp2_tangent_at_kvec` | ∂²u/∂k² via 6 solves — **DEAD; own docstring in `compute_s_tensor_contrib_at_q0` calls it "a dead-end for S-tensor at q=0"** |
| `_q_project` | Q projector as free function — **DEAD, zero callers** |
| `compute_s_tensor_contrib_at_q0` | per-k (3,3) S-tensor contribution from k·p gradients through A⁻¹ |
| `build_sternheimer_op_at_kvec_traced` | rebuild SternheimerOp from a traced kvec (jvp-friendly: T_diag + vnl_Z via `_build_vnl_kdata_core`) |
| `chi_col_contrib_at_kvec_traced` | per-(k,q) χ column contribution as a pure JAX fn (q → χ_col); supports frozen/unfrozen projector, quadratic U_val ansatz, `sos_only` truncated SoS |
| `_load_unfolded_wfns` | load full-BZ ψ FFT-box buffer + expanded IBZ energies |
| `_psi_box_to_G_sphere` | gather box scatter back to G-sphere coefficient list |
| `build_Gprime_list` | ng_out lowest-|q+G'| integer G' vectors (head + wings ordering) |
| `_per_k_chi` | module-scope per-k χ kernel (shared jit cache across q) |
| `_make_chi_vmap_over_k` | vmap wrapper with Schur-dependent in_axes |
| `_build_stk_at_q` | jitted per-q stack builder (kminq gathers, umklapp phases, source b, TPA precond) |
| `_per_k_full_S`, `_make_s_vmap_over_k` | fused per-k kp+S-tensor kernel and its vmap |
| `run_sternheimer` | main driver: load WFN/SymMaps/Meta, V_scf from ρ_val full-BZ sum, H_k cache, α_pv, q-loop, optional derivatives/S-tensor, writes sternheimer.h5 |
| `main` | argparse CLI; reuses `psp.get_DFT_mtxels.read_cohsex_input` for `-i` files |

**Cross-module deps.** `runtime` (set_default_env, init_jax_distributed), `common` (Meta,
symmetry_maps.SymMaps, load_wfns.load_kpoint_fftbox, kq_mapping.{umklapp_G_wrap,
umklapp_phase_box_batched, kminq_idx_for_iq}, jax_compile_cache), `file_io.WfnLoader`,
`psp.dft_operators.{setup_H_k_from_kvec, compute_ngkmax}`, `psp.h_dft.make_apply_H`
(imported, apparently unused — see weird), `psp.pseudos.load_pseudopotentials`,
`psp.scf_potential.{build_dft_potentials, build_rho_val_from_wfn}`,
`psp.vnl_ops._build_vnl_kdata_core` (private-name import across modules),
`psp.get_DFT_mtxels.read_cohsex_input`, `solvers.projectors.make_Q_kminq` (unused import),
`solvers.sternheimer_precond.{compute_per_band_kinetic, tpa_preconditioner_diag}`,
`solvers.sternheimer_solve.{SternheimerOp, sternheimer_solve, _apply_A_inline,
cond_subspace_sos_solve}` (repeatedly re-imported inside functions).

**I/O.**
- Reads: `WFN.h5` (BGW-format via WfnLoader); UPF pseudopotentials from `pseudo_dir`;
  optional INI input (`sternheimer.in` / `cohsex.in`, parsed by `read_cohsex_input`, key
  `wfn_file` consumed).
- Writes: `sternheimer.h5` (h5py): root attrs (tol, max_iter, n_cond_bands, n_occ,
  truncation_2d, ngkmax, use_schur, with_derivatives, with_s_tensor, note), datasets
  `q_crys`, `iq_reduced`, optional `s_tensor_q0` (+`s_tensor_note` attr); per-q groups
  `q_{i}/` with `chi_col` (ng_out complex), `G_int` (ng_out,3), optional `dchi_col_dq`
  (ng_out,3), attrs `q_crys`, `iq_reduced`.

**Flags consumed.** CLI: `-i/--input`, `--wfn`, `--pseudo_dir`, `--n-cond-bands`,
`--iq-list`, `--ng-out`, `--tol`, `--max-iter`, `--truncation-2d`, `--with-derivatives`,
`--s-tensor`, `--sos-only`, `-o/--output`. Input-file key: `wfn_file`. Env: `KP2_DEBUG`
(debug prints in `compute_kp2_tangent_at_kvec` — itself dead).

**Dead suspects.**
- `_batched_real_norm_host` — grep for the name across src/tests/tools/scripts: only the
  definition (line 85).
- `_q_project` — grep: only the definition (line 490). Inline copies of the same expression
  are used instead.
- `compute_kp2_tangent_at_kvec` (+ its `KP2_DEBUG` block and the `U_val_hess_kp` parameter of
  `chi_col_contrib_at_kvec_traced`) — grep: no caller anywhere; no caller ever passes
  `U_val_hess_kp`; docstring of `compute_s_tensor_contrib_at_q0` explicitly says the kp² Hess
  machinery "is NOT used here — it was a dead-end for S-tensor at q=0".
- Unused imports: `solvers.projectors.make_Q_kminq` (line 73, never called in file),
  `psp.h_dft.make_apply_H` (line 69, no call site in file), `os` (top-level import at
  line 53 unused; kp2 re-imports `os as _os` locally).
- `build_sternheimer_source` and `make_umklapp_phase` — driver no longer uses them (only
  `src/psp/tests/test_sternheimer_jvp.py` does); production path uses
  `build_sternheimer_source_preQ` + inline Q, and `common.kq_mapping.umklapp_phase_box_batched`.

**Redundancy suspects.**
- Q-projector `x − U (U†x)` implemented inline **five** times (`_q_project`, `Q_of` in
  `compute_kp_tangent_at_kvec`, `Q_of` in `compute_kp2_tangent_at_kvec`, inline `b_eff` in
  `chi_col_contrib_at_kvec_traced`, `_q_apply` in `_build_stk_at_q`) while
  `solvers.projectors.make_Q_kminq` exists and is imported. Classic parallel-path cruft.
- `build_sternheimer_source` vs `build_sternheimer_source_preQ`: old/new pair; old kept only
  for the test.
- `make_umklapp_phase` duplicates `common.kq_mapping.umklapp_phase_box_batched` (single-G vs
  batched variants of the same phase; kq_mapping claims to be "single source of truth").
- `_gather_box_at_G` vs `_psi_box_to_G_sphere` vs the inline gather in
  `accumulate_chi_density` — three copies of mod-wrap gather/scatter index math.
- `compute_ngkmax` imported from `psp.dft_operators` here, but `run_nscf`/`kpm_dos` use the
  other definition in `psp.gvec_utils` — two module-level duplicates repo-wide (see below).
- The `SternheimerOp` "tile eps_v 3×, tile precond 3×" reconstruction appears twice
  (`compute_kp_tangent_at_kvec` op_b3 and `compute_s_tensor_contrib_at_q0` op_3), copy-paste.

**Weird code.**
- `compute_kp2_tangent_at_kvec` lines 474-479: `import os as _os` mid-function +
  `KP2_DEBUG` env-gated `print(float(tracer))` — would crash under jit tracing; eager-only
  debug leftovers inside dead code.
- `chi_col_contrib_at_kvec_traced` has 25+ parameters with three mutually-exclusive modes
  (frozen b vs `Vu_G_preQ`, linear vs quadratic `U_val_eff`, CG vs `sos_only`) — violates the
  "pass bundles" convention; refactor target.
- Repeated function-local `from solvers.sternheimer_solve import ...` (4 sites) despite a
  top-level import of the same module — circular-import scar or leftover.
- Magic normalisations: `prefactor_chi = 2·spin_factor·sqrt(N_grid)/(vol·nk)` vs
  `prefactor_st = 2·spin_factor/(vol·nk)` — the stray `sqrt(N_grid)` is the ortho-FFT
  convention factor, documented only in a comment at the V_pert_box branch.
- `alpha_pv = 2·(E_max − E_min)` (QE LR_Modules convention) — magic factor 2, documented.
- Docstring at line 1444 still says "projected MINRES" while the code/comment block at
  line ~1170 explains it was replaced by level-shifted CG ("MINRES knobs" in
  `run_sternheimer` docstring too) — stale naming.
- `out_h5` is opened at line 1184 before all heavy compute; a crash mid-q-loop leaves a
  truncated file and there is no try/finally.

---

## src/psp/run_nscf.py (531 LOC)

**Purpose.** LORRAX-native NSCF driver: builds V_scf from a QE `.save`, solves for nbnd KS
eigenstates per k-point with Davidson, writes a BGW-format `WFN.h5`; optionally (or
standalone from an existing WFN.h5) appends stochastic CJ-filtered Ritz "pseudobands" to
fill the high-energy spectrum → `WFN_pseudobands.h5`. Multi-process round-robin over
k-points with allgather-sum reduction; rank 0 writes.

**Category.** physics: DFT/NSCF stage (wavefunction generation feeding gw_jax); pipeline
driver.

**Entry points / callers.**
- `main` <- CLI only: `python3 -m psp.run_nscf` (used by sandbox skills:
  `skills/profiling_stack/SKILL.md`, run scripts under `runs/`). No Python importer of
  `run_nscf` found in src/tests/tools/scripts.
- `run_nscf(crystal, pseudos, kgrid, nbnd, ...)` <- `main` only.
- Historical note: `psp/scf_potential.py` docstring says its `build_dft_potentials` was
  "lifted from psp/run_nscf._build_potentials" — `_build_potentials` here is now a thin
  wrapper.

**Function table.**
| Function | Role |
|---|---|
| `_build_potentials` | ρ_val from QE .save + `scf_potential.build_dft_potentials` wrapper |
| `_setup_kgrid` | k-grid (MP reduce or override), master G list, ngkmax, per-k G-vecs |
| `_make_flat_matvec` | (batch,ns,ng) matvec → flat vector interface for pseudobands |
| `_write_pb_k` | reshape/reorder-to-QE and write one k's pseudobands via WFNWriter |
| `run_nscf` | main driver: Davidson stage (per-rank k loop, allgather, WFNWriter), pseudobands-only load path, pseudobands stage (v1/v2, k=0 calibrates windows) |
| `main` | argparse CLI, `nscf.in` parsing via `psp.nscf_input.read_nscf_input`, `--ref_wfn` k-point override |

**Cross-module deps.** `runtime`, `file_io.{CrystalData, WFNWriter}`,
`psp.pseudos.load_pseudopotentials`, `psp.h_dft.{setup_H_k_from_kvec, make_apply_H}`
(re-export of dft_operators), `psp.dft_precond.{make_dft_preconditioner, make_pw_init}`,
`psp.gvec_utils.{build_master_gvec_list, select_gvecs_for_k, compute_ngkmax, reorder_to_qe}`,
`psp.scf_potential.build_dft_potentials`, `psp.nscf_input.read_nscf_input`,
`solvers.davidson.{davidson, warmup_davidson_jit}`,
`solvers.pseudobands_v2.ritz_pseudobands_v2` / `solvers.pseudobands.ritz_pseudobands`,
`common.jax_compile_cache`, `jax.experimental.multihost_utils`.

**I/O.**
- Reads: QE `.save` directory (CrystalData: charge density, cell, ecut); UPF pseudos;
  optional `nscf.in` (INI); optional existing `WFN.h5` for pseudobands-only mode (datasets
  `mf_header/kpoints/{nrk,mnband,el,rk,w}`, `wfns/coeffs`); optional `--ref_wfn` WFN.h5
  (`mf_header/kpoints/{rk,w}`).
- Writes: `WFN.h5` (BGW mf_header + wfns, via WFNWriter, rank 0 only);
  `<output>_pseudobands.h5` (same format, nbnd = protected + pseudo).

**Flags consumed.** CLI: `-i/--input`, `--save`, `--pseudo_dir`, `--nbnd`, `--nk`,
`--nosym`, `-o/--output`, `--tol`, `--ref_wfn`, `--no-davidson`, `--pseudobands`,
`--pb-wfn`, `--pb-seed`. Input-file keys (`[nscf]`): save_dir, nbnd, kgrid, nosym, output,
tol, rho_from_wfn, wfn_file, pseudobands, pb_version, pb_k, pb_M_max, pb_F, pb_n_windows,
pb_n_prot.

**Dead suspects.** None internal — all helpers called. Module-level: `run_nscf` has no
programmatic caller (CLI-only), which is expected for a driver.

**Redundancy suspects.**
- `compute_ngkmax` exists in both `psp/gvec_utils.py:49` (used here and by kpm_dos) and
  `psp/dft_operators.py:186` (used by run_sternheimer) — repo has two definitions of the
  same routine.
- Pseudobands v1/v2 dual path (`pb_version` switch, two keyword spellings
  `Phi_dav/E_dav/n_prot` vs `Phi_det/E_det/n_protected`) — parallel old/new code path of
  exactly the flagged "fetch_X_dyn next to fetch_X" species.
- `setup_H_k_from_kvec` imported from `psp.h_dft` here but from `psp.dft_operators` in the
  two sibling drivers — inconsistent aliasing of the same function.

**Weird code.**
- Pseudobands-only load path (lines 299-322): `all_evecs` is allocated `(nk_file, ...)` but
  the fill loop writes only `all_evecs[0, ib, ...]` — for nk>1 all k>0 wavefunctions stay
  zero, then the pseudobands k-loop (`for ik in range(1, nk)`) reads them. Also
  `f["wfns/coeffs"][ib]` is the *concatenated-over-k* G axis in BGW layout, so even k=0 gets
  the wrong slice unless nk==1. Hypothesis: mode was written/tested only for a single
  k-point and silently breaks for multi-k pseudobands-only runs.
- `main`: `pb_wfn_input = inp.wfn_file if inp.rho_from_wfn else None` — the `rho_from_wfn`
  flag (documented as "charge density from WFN") is actually used to gate the pseudobands
  WFN input path; the ρ path never consults it (ρ always comes from the .save). Misnamed or
  vestigial flag.
- `tol = args.tol or inp.tol` etc. — `or`-based override breaks for legitimate falsy values
  (`--tol 0`, `--nbnd 0`); minor.
- Multi-process reduction gathers full `(n_proc, nk, nbnd, ns, ngkmax)` complex buffers to
  every rank then `.sum(axis=0)` over zero-padded copies — memory-heavy allgather-as-reduce
  idiom (fine at current scales, flagged for the refactor).
- Davidson JIT warmup loop `for m in range(nbnd, m_max + nbnd, nbnd): apply_H0(zeros(min(m, m_max), ...))`
  with magic `m_max = 4*nbnd` — pre-compiles the batched matvec at each block size Davidson
  will hit; opaque without knowing davidson's internals.

---

## src/psp/nscf_input.py (78 LOC)

**Purpose.** Tiny INI parser for the `[nscf]` input file: a frozen-default `NSCFInput`
dataclass plus `read_nscf_input()` which resolves relative paths against the input file's
directory.

**Category.** I/O: input-file parsing (config for run_nscf).

**Entry points / callers.**
- `read_nscf_input` <- `psp.run_nscf.main` (lazy import at run_nscf.py:469). Only caller
  repo-wide (grepped src/tests/tools/scripts).
- `NSCFInput` dataclass — consumed only via `read_nscf_input`'s return value.

**Function table.** `NSCFInput` (dataclass, 17 fields incl. 8 pb_* knobs);
`read_nscf_input(filename)` (configparser + path resolution).

**I/O.** Reads `nscf.in`-style INI file. Writes nothing.

**Flags consumed.** All `[nscf]` keys listed above under run_nscf.

**Dead suspects.** `rho_from_wfn`/`wfn_file` fields: parsed here, but the only consumer
(run_nscf.main) uses `rho_from_wfn` merely as a gate for `pb_wfn_input`; no code path ever
reads ρ from a WFN. Semantically dead / misnamed pair.

**Redundancy suspects.** Pattern-level: this is one of several per-driver ad-hoc INI parsers
(cf. `psp.get_DFT_mtxels.read_cohsex_input` used by run_sternheimer); a refactor could
unify. No intra-file redundancy.

**Weird code.** `resolve(sec.get("wfn_file", ""))` — resolving the empty-string default
yields the input directory itself as a "path"; harmless only because it's gated by
`rho_from_wfn`.

---

## src/psp/kpm_dos.py (353 LOC)

**Purpose.** Diagnostic tool: total electronic DOS of the KS Hamiltonian via the Kernel
Polynomial Method (Chebyshev moments + Jackson damping + stochastic Rademacher trace),
using the same potential/H_k setup as run_nscf, then plots to PDF. Docstring: "mirrors
bse_kpm.py as closely as possible".

**Category.** diagnostic/bench script (DOS sanity check on H_DFT).

**Entry points / callers.**
- `main` <- CLI only: `python3 -m psp.kpm_dos` (docstring usage line). No importer found.
- `run_kpm_dos` <- `main` only. NOTE: `bse/bse_feast.py:1203` and `bse/bse_pseudopoles.py:621`
  call `bse_kpm.run_kpm_dos` — a *different*, same-named function in `src/bse/bse_kpm.py:101`,
  not this one (confirmed by import lines).
- `make_dft_h_tilde`, `make_dft_random_vector`, `estimate_spectral_bounds_from_diag`,
  `_plot_dos` — internal only, zero external callers.

**Function table.** `make_dft_h_tilde` (rescaled matvec (H−c)/w for Chebyshev);
`make_dft_random_vector` (masked Rademacher vectors); `estimate_spectral_bounds_from_diag`
(E-range from h_diag, padding sentinel 1e10/1e9 filtered); `run_kpm_dos` (driver: potentials
→ k-grid → per-k H_k → moments → Jackson → reconstruct → plot, returns dict);
`_plot_dos` (matplotlib Agg PDF); `main` (argparse CLI).

**Cross-module deps.** `file_io.CrystalData`, `psp.pseudos`, `psp.ionic_gspace.build_ionic_and_core`,
`psp.dft_operators.{build_G_cart, compute_V_H_and_V_xc, build_V_scf, HamiltonianK,
setup_H_k_from_kvec}`, `psp.h_dft.make_apply_H`, `psp.gvec_utils.{build_master_gvec_list,
compute_ngkmax}`, `psp.vnl_ops.build_vnl_setup`, `solvers.chebyshev.{make_chebyshev_recurrence,
chebyshev_moments, jackson_coefficients, reconstruct_dos}`, `common.timing`.

**I/O.** Reads QE `.save` + UPF pseudos. Writes DOS plot PDF (`dft_dos_kpm.pdf` default).
No HDF5 output; results returned as a dict.

**Flags consumed.** CLI: `--save` (required), `--pseudo_dir`, `--nk`, `--nosym`,
`--n-moments`, `--n-random`, `--buffer`, `--seed`, `--plot`. Env: sets
`JAX_ENABLE_X64=1` directly (does NOT use `runtime.set_default_env` like the other two
drivers — drift from the runtime-bootstrap convention).

**Dead suspects.** None beyond CLI-only status of the whole module (no importer of
`psp.kpm_dos` anywhere; sole documented use is manual DOS inspection, e.g. for
pseudobands window calibration).

**Redundancy suspects.**
- Potential-construction block (lines 162-177: build_ionic_and_core → load_charge_density →
  build_G_cart → compute_V_H_and_V_xc → build_V_scf → build_vnl_setup) is an inline
  copy of what `psp.scf_potential.build_dft_potentials` now encapsulates — the lift-out that
  run_nscf and run_sternheimer migrated to never reached kpm_dos.
- `run_kpm_dos` name collides with `bse_kpm.run_kpm_dos`; the module docstring admits it
  "mirrors bse_kpm.py as closely as possible" — deliberate copy-paste sibling, refactor
  candidate for a shared KPM core.
- Uses `psp.gvec_utils.compute_ngkmax` (duplicate of `psp.dft_operators.compute_ngkmax`).
- `build_master_gvec_list` called but its result `G_master` is never used afterwards
  (line 186; only ngkmax is consumed) — vestigial call copied from `_setup_kgrid`.

**Weird code.**
- `HamiltonianK` imported (line 30) but never referenced — unused import.
- `G_master, _ = build_master_gvec_list(crystal)` result unused (see above).
- Padding sentinels `1e10` / threshold `1e9` in `estimate_spectral_bounds_from_diag` — magic
  numbers coupling to h_diag's padding convention in dft_operators.
- Env bootstrap divergence: `os.environ.setdefault("JAX_ENABLE_X64","1")` instead of
  `runtime.set_default_env()` + `init_jax_distributed()`; single-process only.
- Note "reserved" quirk: `x_dummy` warmup uses zeros — Chebyshev recurrence on a zero vector
  compiles fine but any NaN-poisoning semantics differ from real input; harmless.

---

## Cross-file summary for the refactor map

- Shared pattern: three drivers × (potentials → k-grid/ngkmax → per-k `setup_H_k_from_kvec`
  → per-k solver). `scf_potential.build_dft_potentials` unified 2 of 3; kpm_dos still inline.
- Two repo-wide duplicate `compute_ngkmax` definitions (gvec_utils vs dft_operators) and
  inconsistent import origin for `setup_H_k_from_kvec` (h_dft re-export vs dft_operators).
- run_sternheimer is the heaviest refactor target: 5 inline Q-projector copies, 3 dead
  functions, a 25-arg mega-function, and stale MINRES naming.
- run_nscf's pseudobands-only mode looks broken for nk>1 (all_evecs fill bug) — verify
  before relying on it.
