# LORRAX BSE — exhaustive feature catalog

The features a Berkeley-physics owner would name, sorted into a merged teleology, in the format
established by `reports/gw_refactor_map_2026-07-01/archive/FEATURES.md`. Each feature: **what**
(with physics) · **where** (files/functions) · **flags** · **interactions** · **refactor note**.
This is the document to load before adding a BSE feature — it says which category it belongs to
and what it must not break.

Legend for footprint: 🟢 clean (one home) · 🟡 smeared (see refactor note) · 🔴 has a suspected bug
or is broken/unreachable at HEAD.

> **Method.** Two independent agents sorted the same 19 per-file notes (`files/*.md`, authoritative)
> into tiered taxonomies; a third pass wrote a reader digest per file (purpose/entry-points/features,
> each claim grep-verified). This document merges the two sortings into one taxonomy and calls out
> every place they disagreed rather than silently picking a winner. Raw inputs are preserved verbatim:
> sortings in `_raw_sorts.json`, digest in `_digest.json`. No prior BSE MAP/FEATURES doc exists — this
> is the first cut, dated **2026-07-15**, checked out at `adc2197`/`e18d0e5` (files byte-identical
> between the two).

---

## How the two sortings disagreed

Both agents agree on the physics and the call-graph (verified independently by grep); they disagree
on **where the seams are**. Six disagreements are structural enough to affect this document's shape:

1. **Ingest/kernel/eigensolve numbering is offset by one.** Sorting 1 (S1) numbers ingest=A0,
   kernel=A1, eigensolve=A2 (one bucket for Lanczos+Davidson+FEAST+KPM), pseudopoles=A2b, spectra=A3,
   output=A4, driver=its own `A_driver`. Sorting 2 (S2) numbers ingest=A1, kernel=A2, and splits
   eigensolve into **three parallel A-tier stages** (A3a Lanczos, A3b Davidson, A3c FEAST), with KPM
   demoted to a Tier B variant axis; spectra=A4, output=A5, and there is no dedicated driver stage.
   **Resolution:** this doc adopts S2's granularity (three solver families are different enough in
   code, dependency, and maturity to deserve separate entries — matches how GW's own FEATURES.md
   splits similarly-sized stages) but keeps S1's `A_driver` as an explicit stage, because `bse_jax.py`
   really does fuse A+D+E+F content the way S1's note describes and that is worth naming.
2. **KPM-DOS: pipeline stage or variant axis?** S1 folds `bse_kpm.py` into the eigensolve pipeline
   stage (A2) with a note that it "straddles A2 and E." S2 makes it its own Tier B axis ("FEAST
   window-selection variant: KPM-derived windows"). **Resolution:** filed as B6 (a variant of *how
   FEAST picks windows*, not a solve stage in its own right — it never returns eigenpairs) with a
   pointer from A2c; this is closer to S2 but the disagreement is real and both framings are legitimate.
3. **Is pseudopole W-compression a pipeline stage or a W-source variant?** S1 gives `bse_pseudopoles.py`
   its own semi-independent Tier A stage (A2b, "the low-scaling real-axis endgame") — and *also* lists
   it under its own Tier B3 ("screened-interaction source") axis, i.e. S1 double-counts it. S2 places
   it only under Tier B ("W-source variant: pseudopole-compressed W_c(ω)"), never as its own A-stage.
   **Resolution:** filed as A3 (it is forward-looking enough, and physically distinct enough from the
   default matvec-time W lookup, to warrant a numbered stage — the GW map does the same for
   comparably-unwired-but-real stages like the Sternheimer head) *and* cross-referenced from B3.
4. **`bse_w_exact.py`: pipeline-adjacent or pure diagnostic?** S1 lists it as a *member* of the A2b
   pseudopole stage (grouped with the production-intent code). S2 places it in Tier E, explicitly
   contrasting it with the broken `bse_pseudopoles.py`: "functional-but-unused... it is functional,
   unlike bse_pseudopoles.py which is genuinely broken." Both agree it compiles and runs cleanly with
   zero external callers. **Resolution:** filed as E2 (diagnostic/validation tool) since "zero callers,
   own `__main__` only" is the E-tier definition used everywhere else in this doc; A3's entry cross-
   references it as the tool that would validate a repaired pseudopole path.
5. **`feast_ellipse_mixed_sweep.py`: diagnostic or dead?** S1 keeps it in Tier E grouped with the other
   three sweep scripts, noting in the same breath that it fails `py_compile` (IndentationError, line
   113) and is "literally unreachable." S2 puts it in **Tier F** for exactly that reason — an
   uncompilable file has no callers by construction. **Resolution:** Tier F (S2's placement) — a file
   that cannot be imported is dead, not merely stale; noted under both E1 and F for discoverability.
6. **Which file is the single worst offender?** S1 ranks `bse_jax.py` (626 L, A_driver + dead matvec
   trio + E diagnostics + D flags all in one file) #1 and `bse_ring_comm.py` #2. S2 ranks
   `bse_ring_comm.py` (996 L, A2/B2/B1/C1/E4/F all in one file) #1 and doesn't even place `bse_jax.py`
   above #4. **Resolution:** both are presented in "Worst misfits" below without forcing a single
   ranking — they are straddling different axis-counts (`bse_jax.py`: 4 tiers over 626 L;
   `bse_ring_comm.py`: 5 tiers over 996 L) and a reader should know both.

Two smaller disagreements are noted inline where they occur: whether `pseudopoles_sweep.py` is an
"in-file embedded diagnostic" (S1 — but it is a standalone file with its own `__main__`, which is an
internal inconsistency in S1's own framing) or a standalone sweep script grouped with its siblings (S2,
adopted here as E1); and whether the Tier B5 fine-grid/finite-Q axis belongs in Tier B (S1, because it
*is* conceptually a variant axis) or Tier E (S2, because the only artifact is a reference document, not
LORRAX code) — resolved by filing the concept as B5 with an explicit pointer to its E6 document, since
neither framing is wrong and this doc's Tier B otherwise only lists axes with real LORRAX members.

---

## Tier A — Pipeline stages

### A0. Ingest — restart-bundle loaders, EQP overrides, q=0 head injection 🟡
Turns the GW ISDF restart bundle (`tmp/isdf_tensors_*.h5`: flat-q `V_qmunu`/`W0_qmunu` (nq,μ,μ),
`psi_full_y` (nk,nb,nspinor,n_rμ), `enk_full`, `G0_mu_nu`, `vhead`/`whead`) into the ψ/ε/V/W tensors
every solver consumes. Two loader entry points — sharded (x,y)-mesh slab loader
(`load_bse_data_from_restart_sharded` + `_read_psi_mu_sharded`/`_read_vq0_sharded`/`_read_wq_sharded`)
and single-device ring loader (`_load_ring_subset`) — plus `resolve_n_occ` (band-window resolution,
v=0 deepest, LORRAX-internal ordering), `read_bgw_eqp`/`apply_eqp_corrections` (BGW `eqp1.dat`
substitution `enk_qp[k,b] = e_qp_ibz[irr_idx_k[k],b] / 13.6056980659`), and q=0 Coulomb-head
reinstatement as a rank-1 update `V_q0[μ,ν] += v_scalar·conj(G0[μ])·G0[ν]` (BGW mini-BZ 1/q² average).
`src/bse/bse_io.py:load_bse_data_from_restart_sharded` (358-536), `_load_ring_subset` (767-932),
`resolve_n_occ` (603-664), `read_bgw_eqp` (539-581), `apply_eqp_corrections` (698-753), head injection
at 463-513 (sharded) and 804-836 (ring), `_find_restart_file` (globs `tmp/isdf_tensors_*.h5`).
**Flags:** `--eqp`, `--n-occ`, `vhead`/`whead_0freq` (cohsex.in). **Interactions:** the single
highest-fan-in file in `bse/` — 10+ direct importers of the sharded loader and `_find_restart_file`
alone (every solver/diagnostic module imports both). `gw.head_correction.apply_q0_head_rank1{,_sharded}`
(bse_io.py:504,818) is the *only* `from gw.` import anywhere in `bse/`. **Refactor note:** the q=0
head-rank-1 injection is hand-copied once for the sharded loader (463-513) and again for the ring
loader (804-836) instead of being one shared helper called twice — a duplication smell flagged
elsewhere in the sandbox's "no redundancy" convention (memory `feedback_no_redundancy`). `BSEData`
(bse_io.py:18) is a dead dataclass — zero callers anywhere (Tier F).

### A1. Kernel build & matvec application (D + V − W) 🟡
"Kernel build" and "matvec" are the same step in this codebase — a `build_*` factory returns the
applied closure directly; there is no separate assembled-`H` object anywhere in the package (unlike
the prompt's canonical 6-slot dataflow template). `H·X = D·X + V·X − W·X` where
`D[b,c,v,k] = (ε_c[k,c] − ε_v[k,v])·X`, the exchange `V` contracts spin-traced ISDF pair amplitudes
`M[k,c,v,μ] = Σ_s ψ*_c[k,c,s,μ]ψ_v[k,v,s,μ]` through `V_q0[μ,ν]`, and the direct term `W` is an FFT
convolution `(1/√nk)Σ_k' W_{k−k'}[μ,ν]T[k']` over spinor-resolved pair-density slabs. Three
implementations of the identical math: `bse_serial.py` (single-device, the **correctness reference**
every other matvec is checked against — `bse_ring_comm.py:934-996`), `bse_simple.py`
(`build_bse_simple_matvec`, the **production default**, plain jit + einsums + `with_sharding_constraint`
on the (x,y) mesh, XLA-generated collectives), `bse_ring_comm.py` (`build_bse_ring_matvec` +
`build_bse_ring_matvec_full`, hand-rolled `lax.ppermute` rings / `all_gather` inside `shard_map`,
conduction+μ sharded on x, valence+ν on y). **Flags:** `--matvec-kind {simple,ring,gather}` (dispatch
at `bse_jax.py:446-459,622` → `bse_lanczos.py:155-164`). **Interactions:** see B2 (comm-strategy axis)
and B1 (TDA/full axis, `build_bse_ring_matvec_full` implements the two-block Casida structure
`[[A,B],[−B^H,−A^H]]`). **Refactor note:** deliberately kept triple for correctness cross-checking, but
this means adding a new kernel term (e.g. a Rytova-Keldysh correction) means touching `bse_serial` +
`bse_simple` + `bse_ring_comm` together — a genuine co-change cost, not accidental duplication.
`bse_serial.apply_bse_hamiltonian_single_device_jit` and `bse_serial.symmetrize_W_q` are dead within
this file (Tier F).

### A2. Eigensolve family
Three independently-maintained solver families over the A1 matvec, plus one window-selection feeder.

- **A2a. Lanczos / block-Lanczos (TDA-only) 🟡** — `solvers/lanczos.py` is physics-blind infra (four
  Hermitian lowest-eigenpair solvers: Python-loop reference, `fori_loop` jit with partial reorth,
  fixed-size block-jit, `lax.while_loop` convergence-driven block); `bse/bse_lanczos.py` wires the BSE
  physics and dispatches single-device (`solve_bse`) or 2D-mesh sharded (`solve_bse_sharded`, which
  also hosts the A2b Davidson dispatch). `solve_bse` (30-97), `solve_bse_sharded` (100-313), `_full_run`
  (246-304). **Refactor note:** `solvers.lanczos.block_lanczos_eig` (the non-jit block variant) is dead
  — gated behind `use_block=True` at `bse_lanczos.py:83`, and repo-wide grep for `use_block` finds only
  `test_bse.py:257` passing `False` (Tier F); the *jit* variants `block_lanczos_eig_jit`/
  `_jit_converged` are live and distinct.
- **A2b. Davidson (block, TDA-only) 🟢** — `solvers/davidson.py:davidson` is a shape-agnostic Hermitian
  block Davidson (QE cegterg-style: ellipsis-einsum Gram projection, Cholesky generalized eigh, CGS2
  re-orth, restart at m_max), reached via `bse_lanczos.py` (`--solver davidson`) or the standalone
  `davidson_absorption.py` CLI; `bse_davidson_helpers.py` supplies the lowest-ΔE initial subspace
  (`init_bse_subspace`) and diagonal preconditioner `1/(ΔE − λ + ε)` (`bse_diagonal_precond`).
  **Interactions:** `solvers/davidson.py` is shared *verbatim* with `psp/run_nscf.py`'s DFT NSCF
  diagonalization — a BSE-only Davidson change must not regress the DFT driver. Production evidence:
  `runs/Si/C_bse_davidson_profile_2026-04-29/` (100 eigvals in 18 iter / 209 s on 4×A100).
- **A2c. FEAST contour filter — DEFAULT solver 🟡** — `bse_feast.py` estimates spectral bounds via a
  short Lanczos run, builds eV energy windows (default/user/KPM-derived, see B6), approximates the
  spectral projector `P = (1/2πi)∮(zI−H)⁻¹dz` via ellipse-trapezoid or Zolotarev quadrature, filters
  through diagonal-preconditioned FGMRES shifted solves, and extracts eigenpairs by overlap-whitened
  Rayleigh-Ritz. Handles both TDA and full non-TDA (non-TDA is the CLI default). `estimate_spectral_
  bounds_sharded` (695-846), `run_feast_ritz` (463-684), `_rayleigh_ritz` (306-422), quadrature rules
  (858-973). **Refactor note:** despite its own docstring calling itself "setup utilities," `bse_jax.py`
  dispatches here whenever `--lanczos` is absent (`bse_jax.py:540-543`) — it *is* the production default
  path, a self-description/reality mismatch both sortings independently flagged. Zero pytest coverage
  (grep for "feast" under `tests/` is empty).
- **A2d. KPM Chebyshev DOS (window feeder) 🔴** — see B6 for its role as a FEAST/pseudopole
  window-selection variant. `run_kpm_dos` rescales `H_tilde = (H−c)/h`, runs the 3-term Chebyshev
  recurrence on Rademacher probe vectors, Jackson-damps, and reconstructs `ρ(E)` for equal-mass window
  placement. **Broken at HEAD:** the digest flags "the entire BSE KPM path is broken at HEAD by a
  phantom kwarg" (see `files/kpm_chebyshev_dos.md` for the exact signature mismatch).

### A3. W_c(ω) pseudopole compression 🔴 — lost wiring, NOT dead code
Windowed pseudopole compression of the correlation screened interaction `W_c(ω)` in the ISDF `r_μ`
basis: density-biased random seeds → FEAST contour filtering via sharded GMRES → Rayleigh-Ritz in a
brightness subspace → J-metric normalization for the non-Hermitian Casida structure
`S = [[A,B],[−B*,−A*]]` → anti-resonant pole doubling. Output feeds
`W_c[μ,ν](z) = Σ_p w_p d_p[μ]conj(d_p[ν])/(z − ω_p)` (`pseudopoles_eval.reconstruct_Wc_columns`) and a
`p_keep` convergence sweep (`pseudopoles_sweep.py`, Tier E). `bse_pseudopoles.py:run_pseudopoles`
(212-487), `write_pseudopoles_h5` (490-547), `main` (550-695, 3 window-selection modes). **This is the
low-scaling real-axis endgame the sandbox's GW map alludes to** — but it is **un-importable at HEAD**:
`bse_pseudopoles.py:31` imports `build_density_drive_operators`/`build_density_readout_operator_full`
from `bse_ring_comm.py`, neither of which exists there anymore (only `build_realspace_random_
transition_generator`/`build_density_snapshot_operator` do) → `ImportError` at module load. Two further
breaks even if the import were fixed: `bse_pseudopoles.py:239,248` pass `build_bse_ring_matvec(...,
v_couples_k=...)`, a kwarg the HEAD signature (`bse_ring_comm.py:340-347`) doesn't have (`TypeError`);
and `bse_pseudopoles.py:599` passes `load_bse_data_from_restart_sharded(..., use_nohead=...)`, a kwarg
the HEAD signature (`bse_io.py:358-368`) doesn't have (`TypeError`). All three losses trace to the
`src/isdf`/`bse_isdf` → `src/bse` package consolidation. **Per the sandbox's "parsed-but-unread ≠ dead
config" convention, this is restore-worthy wiring, not an archival candidate** — do not delete under a
zero-callers heuristic. Validation companion: E2 (`bse_w_exact.py`, functional).

### A4. Spectra — Haydock continued fraction, sum-over-states, BGW rebroadening 🟡
Converts a solved TDA BSE Hamiltonian into `ε₂(ω)`/JDOS/`.dat` files via two independent routes plus a
comparison tool (958 LOC over 4 files). **Eigenvector route** (`absorption_eigvecs.py`):
`ε₂^a(ω) = 16π²/(V_cell·N_k·n_spin·n_spinor) Σ_S |Σ_kcv A[S,k,c,v]d[a,k,c,v]|² (η/π)/((ω−E_S)²+η²)`,
dipole `d` sliced from `dipole.h5` (host numpy, no jax). **Haydock route** (`absorption_haydock.py`):
3-polarization block Lanczos on the sharded TDA matvec (`lax.scan`), backward continued fraction
`g(z) = 1/(z−a_0−b_0²/(z−a_1−…))`, `ε₂ = −pref·‖d‖²·Im g(ω+iη)/π`. `eigvals_to_eps2.py` re-broadens any
BGW-format `eigenvalues.dat` at arbitrary η/truncation for fair BGW-vs-LORRAX comparison — "the
canonical reproduce-BGW-absorption-at-custom-eta tool" per its own docstring. Prefactor verified
identical to BGW `BSE/absh.f90:46`. **Refactor note:**
`absorption_eigvecs.py:compute_oscillator_strengths` (line 50) is dead within an otherwise-live file —
`main()` recomputes oscillator strengths inline via `compute_eps2`'s return instead of calling it
(Tier F pocket inside a Tier-A file).

### A5. Output writers — BGW-format eigenvectors.h5 🔴
Two independent `eigenvectors.h5` writers with **different conventions** — a real bug-shaped smell, not
just duplication. `bse_io.write_eigenvectors_stream` (23-105, production, called from `bse_jax.py:335`
under `--write-eigs`) performs Ry→eV conversion and the valence-axis flip
(`file[0,i,k,c,v] = vec[k,c,nv−1−v]`, BGW `BSE/input_fi.f90:407` convention) at the write boundary.
`write_eigenvectors.py:write_eigenvectors_h5` (21-153, legacy, only reachable via the non-pytest
`test_bse.py` smoke script) does **neither**. A change to one convention silently desyncs the other.
**Refactor note:** `write_eigenvectors.py`'s only surviving caller is `test_bse.py`, itself a Tier-E
smoke script with zero asserts — this file is a de-facto dead-code candidate wearing an A5 badge; worth
deleting once `test_bse.py` is retired or repointed at the streaming writer.

### A_driver. CLI entry point / stage dispatcher 🟡🔴
`bse_jax.py` `__main__` (349-626) routes to A2a/A2b (`_preview_lanczos`, itself a 200-line sharded
Lanczos/Davidson mini-driver at 203-345) or A2c/A2d (`bse_feast.main`/`bse_kpm.main`, argv-string
forwarding — a list of CLI flag strings rebuilt and re-parsed, not a shared in-memory config object;
fragile to keep in sync when flags are renamed), applies A0 EQP overrides, and calls A5's streaming
writer. **This is the single worst-smeared file candidate in the package** (see "Worst misfits" below):
it also contains a fully **dead** file-local jit'd matvec trio (`apply_bse_hamiltonian`/`apply_V`/
`apply_W`/`apply_D`, lines 67-160, zero callers anywhere — Tier F, and dangerous because it sits at the
top of the file's `__all__`/docstring surface where a reader would mistake it for the live path, when
the real production matvecs live in three other files entirely), a Tier-E diagnostic demo
(`_main_random_demo`, `--debug-parallelism`), Tier-E ring-test/ring-check CLI plumbing
(`--ring-test`/`--ring-check`, delegating to `bse_ring_comm.ring_matvec_{smoke_test,correctness_check}`),
and the largest single argparse flag surface in the package (Tier D).

---

## Tier B — Variant axes (cut across A-stages)

### B1. TDA vs full (non-TDA/Casida) BSE 🟡
Cuts across A1 (`build_bse_ring_matvec_full`), A2c (full is the FEAST CLI *default*), A2d (KPM doubles
the trial-vector dimension for non-TDA), A3 (non-TDA RHS `[f,−f̄]`, J-metric pole normalization), and
`bse_w_exact.py` (matvec vs matvec_full selection). **Asymmetric axis, not a uniform toggle:** full/
non-TDA is the CLI default in `bse_feast.py` (the actual default solver path) while the entire
Lanczos/Davidson family (A2a/A2b) is **TDA-only, Hermitian-assumed** — the CLI hard-gates `--lanczos`
behind `--tda`. **Flags:** `--tda` (`bse_jax.py`).

### B2. Matvec communication strategy (serial / simple / ring) 🟢
Serial (`bse_serial.py`, correctness reference) / plain-jit (`bse_simple.py`, production default) /
hand-rolled ring+allgather (`bse_ring_comm.py`). Deliberately kept triple for correctness
cross-checking (`bse_ring_comm.py:934-996` checks against `bse_serial`). **Flags:**
`--matvec-kind {simple,ring,gather}`.

### B3. Screened-interaction (W) source 🔴
Default: `W0_qmunu` read straight from the GW ISDF restart, consumed via A0 loaders + FFT convolution
in the A1 matvecs — the **only source wired end-to-end today**. Alternates: `bse_w_exact.py` (exact,
shifted-GMRES Casida columns — functional but unused, E2) and A3's pseudopole-compressed representation
(**broken**, three independent wiring losses — see A3). Both alternates are validation/future-production
paths, not reachable in production.

### B4. Spin/bispinor pair-amplitude convention (embryonic — no runtime toggle yet) 🟡
`compute_pair_amplitude`/its inlined copies (`bse_serial.py`, `bse_ring_comm.py`, `bse_simple.py`,
`bse_lanczos.py`'s `M` tensor) all hardcode a spin sum
`M[k,c,v,μ] = Σ_s conj(ψ_c[k,c,s,μ])ψ_v[k,v,s,μ]`. **Unlike GW's genuine bispinor runtime toggle
(memory `project_lorrax_zeta_fit_architecture`'s B3), BSE has no bispinor mode at all** — this is a
latent axis, named here because it is exactly the seam (pair-amplitude construction) where a future
bispinor BSE lift would need to hook in (memory `feedback_minimal_signatures` / the GW map's own
warning that spin toggles belong at this seam, not scattered later). Only one of the two raw sortings
names this as its own axis (see disagreements list); kept because the underlying physics claim is
independently confirmed in both digests' matvec descriptions.

### B5. Fine-grid k-interpolation / finite-Q (future axis; unbuilt in LORRAX) 🟡
No LORRAX `bse/` code implements coarse-to-fine WFN interpolation or a finite-momentum-transfer
Hamiltonian; BerkeleyGW's `kernel.x`/`absorption.x` machinery is the only reference (see E6 for the
crib-sheet document). Conceptually this is a real variant axis (interpolated-fine-grid vs
single-grid-only BSE) even though today there is exactly one branch of it.

### B6. FEAST window-selection strategy (default / user / KPM-derived) 🟡
Three ways A2c picks its energy windows before running the contour filter: `build_default_windows_eV`
(`[0,2] eV` + `[2,E_max]`, collapsed if `E_max<2`), user-specified `--window1`/`--window2`, or
`--windows-kpm` → A2d's `bse_kpm.run_kpm_dos(...)["windows_ry"]` (equal-mass DOS quantile placement).
**Refactor note:** `bse_kpm.py` reaches into `bse_feast.py`'s non-underscored-but-treated-as-internal
symbols (`estimate_spectral_bounds_sharded`, `_parse_window_arg`) across the module boundary — a
private-symbol reach-in the digest calls out explicitly (`bse_kpm.py:28,159-166,140,361` →
`bse_feast.py:695-847,686,281`); `bse_pseudopoles.py` reaches into the same private surface for its own
window selection. This axis is filed as Tier B (S2's framing) rather than folded into A2 (S1's framing)
— see disagreements list, item 2.

---

## Tier C — Infrastructure

- **C1. 2D mesh + BSE sharding vocabulary** 🟡 — `create_mesh_2d`, `make_bse_shardings`
  (`bse_ring_comm.py`). The single most-imported infra symbol in the package (13 direct importers,
  confirmed by repo grep — essentially every other `bse/` solver file: `bse_jax`, `bse_simple`,
  `bse_lanczos`, `bse_feast` ×2, `bse_kpm`, `bse_pseudopoles`, `bse_w_exact`, `absorption_haydock`,
  `davidson_absorption`, `test_davidson_bse`) — yet it lives inside the same file as two live matvec
  factories (A1/B1), a B3 W-source probe-operator pair, two Tier-E diagnostic CLIs, and two fully dead
  functions (Tier F). **Refactor note:** highest-priority extraction target in the package — a change
  to the sharding spec here has the widest blast radius of anything in `bse/`, and it has no reason to
  live inside a matvec-kernel file.
- **C2. Generic iterative eigensolvers (matvec-blind)** 🟡 — `solvers/lanczos.py`
  (`simple_lanczos_eig`, `lanczos_eig_jit`, `block_lanczos_eig{,_jit,_jit_converged}`,
  `_build_block_tridiag`), `solvers/davidson.py` (`davidson`, `warmup_davidson_jit`,
  `_ritz_and_residuals`). `davidson.py` genuinely serves both `bse_lanczos.py` (A2b) and
  `psp/run_nscf.py` (DFT NSCF) — cross-pipeline infra, changes here risk both consumers.
  `simple_lanczos_eig` is also called by `solvers/dos.py:estimate_spectrum` (pseudobands/GW side, C5).
  **Refactor note:** `block_lanczos_eig` (non-jit) confirmed dead by grep (Tier F).
- **C3. Chebyshev/KPM DOS core + generic matrix-free DOS/window partitioning** 🟢 —
  `solvers/chebyshev.py` (`make_chebyshev_recurrence`, `chebyshev_moments`, `jackson_coefficients`,
  `reconstruct_dos`, `partition_windows`), `solvers/dos.py` (`compute_dos`, `estimate_spectrum`,
  `dos_weighted_windows`, `geometric_windows`, `compute_window_partition`). The single shared core
  correctly reused by A2d (`bse_kpm.py`), `psp/kpm_dos.py`, and `solvers/pseudobands{,_v2}.py` — a
  genuine single-source success, unlike A0's head-injection duplication.
- **C4. FEAST ellipse quadrature (generic contour-integral rule)** 🟢 —
  `solvers/quadrature.py:feast_ellipse_quadrature`. The only `solvers/` file whose *sole* production
  consumer is `bse/` (A2c wraps and conjugate-augments it for full-BSE); everything else in `solvers/`
  is shared with or exclusive to `psp/`.
- **C5. Generic spectral compression (pseudobands; operator-agnostic, no BSE consumer yet)** 🟢 —
  `solvers/pseudobands.py:ritz_pseudobands` (v1, hybrid stochastic + Chebyshev-Jackson-Ritz),
  `solvers/pseudobands_v2.py:ritz_pseudobands_v2` (v2, adds shifted-boundary quadratic
  partition-of-unity windows + Gauss-quadrature energy nodes). Physics-agnostic over a flat
  `dim = nspinor*ngkmax` vector space; the only production caller is `psp/run_nscf.py` (writes
  `WFN_pseudobands.h5`, BGW-parabands analogue). **Refactor note:** architecturally reusable for a
  future BSE trial-space compression (large nc/nv) — a placeholder note for scope creep, not a current
  BSE dependency. One sorting groups this with C6 (Sternheimer) as "colocated non-BSE infra"; the other
  gives it its own category — this doc keeps it separate because its KPM-windowing math is the same
  family as B6, an unexploited-but-real conceptual link C6's Sternheimer files don't share.
- **C6. Sternheimer/psp linear-response solver library (out of BSE scope)** 🟡 —
  `solvers/sternheimer_solve.py`, `sternheimer_precond.py`, `projectors.py`, `minres.py`. None of these
  four files serve BSE; all four serve only `psp/run_sternheimer.py`'s χ_{G'0}(q) head/wing column.
  `minres.py` is explicitly superseded ("CG avoids MINRES pseudo-convergence-NaN pitfall",
  `run_sternheimer.py:1174-1176`) with zero production callers — the strongest Tier-F candidate in this
  whole group, but it belongs to a different program (psp Sternheimer, not BSE) so its deletion is out
  of scope for a BSE-only refactor pass. Colocated in `solvers/` only by directory, not by dataflow.

---

## Tier D — Config & flag surface 🟡
Unlike `gw/` (single `LorraxConfig` + `gw_config.read_lorrax_input` threaded everywhere), BSE has **no
shared config object** — this is the BSE-side counterpart of the GW map's "shadow env-flag surface
bypasses LorraxConfig" finding, same disorder, different mechanism.

- **D1. BSE CLI flag surface** — 14 independent argparse surfaces, no shared config type:
  `bse_jax.py` (largest, 349-626), `bse_feast.py:main` (1073-1316), `bse_kpm.py:main`,
  `davidson_absorption.py:main`, `absorption_haydock.py:main`, `absorption_eigvecs.py:main`,
  `eigvals_to_eps2.py:_main`, `bse_pseudopoles.py:main`, `pseudopoles_eval.py:main`,
  `pseudopoles_sweep.py:main`, `bse_w_exact.py:main`, `write_eigenvectors.py:main`, `test_bse.py`,
  `test_davidson_bse.py`. `bse_jax.py`'s dispatch to `bse_feast`/`bse_kpm` is argv-string
  reconstruction, not an in-memory handoff (see A_driver).
- **D2. Ad-hoc cohsex.in consumption** — private per-file parsers in `bse_io.py`
  (`_parse_wfn_path`, `_parse_head_overrides`, `_find_restart_file`) bypass the canonical
  `gw_config.read_lorrax_input` used everywhere else in LORRAX. Only `wfn_file`, `vhead`,
  `whead_0freq` are ever read from `cohsex.in` by BSE.
- **D3. QP correction override surface (BGW eqp1.dat semantics) — genuinely single-sourced** 🟢 —
  `bse_io.apply_eqp_corrections`/`read_bgw_eqp`, consumed identically at `bse_jax.py:243`,
  `davidson_absorption.py:34`, `absorption_haydock.py:178`, `test_davidson_bse.py:77`. A positive
  counter-example inside an otherwise disorderly config tier — worth preserving as the pattern D1/D2
  should be refactored toward, not away from.

---

## Tier E — Diagnostics / instrumentation / benches 🟡

- **E1. FEAST parameter-exploration sweeps (dev-phase; stale)** — `feast_sweep.py`,
  `feast_zolo_sweep.py`, `feast_ellipse_mixed_sweep.py` (also Tier F — see disagreement #5),
  `bse_feast_dense_debug.py`, `pseudopoles_sweep.py`. Zero external callers; functionality absorbed
  into `bse_feast.main()`'s own `--quadrature`/`--rho-scale`/n_quad-schedule flags. Deleted as dead in
  `a0da0a5`, restored already-broken in `fe5e3e8` the same day (2026-04-29).
- **E2. Exact-W reference / pseudopole-sweep validation tools** — `bse_w_exact.py:main` (functional,
  clean imports, would validate a repaired A3 pseudopole path; zero external callers today),
  `pseudopoles_eval.py:main` (functional, pure numpy/h5py, only stale caller is an archived pre-rename
  test script). See disagreement #4 for the S1/S2 framing split on `bse_w_exact.py`.
- **E3. Manual smoke/comparison CLIs (argparse, not pytest-collected)** — `test_bse.py`,
  `test_davidson_bse.py`. `pyproject.toml testpaths=["tests"]` excludes `src/bse` entirely; the only
  pytest-collected BSE coverage anywhere is `tests/test_eqp_bgw.py::test_reader_skips_provenance_header`
  (a pure-numpy round-trip of `bse_io.read_bgw_eqp`, D3). Neither smoke script has a single assert.
- **E4. In-file diagnostic/demo entry points embedded in production modules** —
  `bse_jax.py:_main_random_demo` (163-200, `--debug-parallelism`),
  `bse_ring_comm.py:ring_matvec_smoke_test`/`ring_matvec_correctness_check` (853-996, `--ring-test`/
  `--ring-check`). The clearest A/E straddles — diagnostics that live physically inside otherwise-
  production files rather than a separate diagnostics module.
- **E5. Prior-knowledge / validated-results documentation ledger** — `STATUS.md` (validated Si 4×4×4
  8v×8c LORRAX-vs-BGW absorption match: eigenvalues ~3 meV at the ISDF floor, Σ|d|² = 2314.177
  machine-match, Haydock peak within 1.5%), `BGW_COMPARE.md` (six-convention comparison cookbook),
  `context/*.md` (8 files: physics spec, three generations of parallel-matvec design history, FEAST/
  pseudopole theory notes, the verbatim Henneke-2020 source conversion). All files ≥2 months stale;
  **5 load-bearing claims have rotted vs HEAD** (matvec default, V-as-W placeholder, TDA-only, sharding
  table, FEAST CLI/γ) and are flagged for verifier re-check before trusting them.
- **E6. External BGW fine-grid/finite-Q reference** — `files/bgw_fine_grid_reference.md`. Deep-read of
  BerkeleyGW's `kernel.x`/`absorption.x` coarse-to-fine machinery (dcc/dvv WFN expansion, head/wing/body
  kernel storage + interpolation, epsdiag ε⁻¹ tabulation, mini-BZ v(q) average, finite-Q, CSI clustered
  subsampling, non-TDA coupling-block interpolation) — the crib sheet for B5, a capability LORRAX does
  not implement. `BSE/fixwings` is called **zero times** from BGW's own BSE code (only `Sigma/`) —
  BSE does its own wing scaling inline, a fact worth knowing before assuming any shared-fixup helper.

---

## Tier F — Fully dead code
Zero callers anywhere in `src/tests/tools/scripts/docs`, confirmed by repo-wide grep in every case.

- `bse_jax.py:apply_bse_hamiltonian`/`apply_V`/`apply_W`/`apply_D` (67-160) — dead matvec trio; **the
  most dangerous instance in the package**, sitting at the top of the file's `__all__`/docstring
  surface where a skimming reader would mistake it for the live path (see A_driver).
- `bse_ring_comm.py:apply_bse_hamiltonian_ring`, `apply_W_ring` (the latter only ever called from the
  former — transitively dead).
- `bse_serial.py:apply_bse_hamiltonian_single_device_jit`, `symmetrize_W_q`.
- `bse_io.py:BSEData` (dataclass, definition-only).
- `bse_preconditioner.py:BSEPreconditionerTerms`, `_pair_amplitude`, `compute_v_diagonal`,
  `compute_w_diagonal`, `extract_w_q0`, `build_preconditioner_terms`, `build_shifted_preconditioner` —
  **7 of the file's 8 exports**; the 8th, `energy_diff_cv_k`, is load-bearing across `bse_feast.py`,
  `bse_jax.py`, `bse_serial.py` (A1/A2c). Worst dead-to-live ratio in the package: a 3-line function
  keeps an otherwise ~90%-dead 158-line file alive.
- `solvers/lanczos.py:block_lanczos_eig` (non-jit) — gated behind `use_block=True`, which nothing in
  the repo ever sets (only `test_bse.py:257` passing `False`); `block_lanczos_eig_jit`/
  `_jit_converged` are the live, distinct siblings.
- `feast_ellipse_mixed_sweep.py` (whole file) — uncompilable, `py_compile` `IndentationError` at line
  113; filed here per disagreement #5 (also referenced from E1 as a stale sweep script).
- `absorption_eigvecs.py:compute_oscillator_strengths` (line 50) — unreferenced; `main()` uses
  `compute_eps2`'s returned `f_Sa` instead.
- **Soft-dead, not yet delete-eligible:** `write_eigenvectors.py` (A5) — its only surviving caller is
  the Tier-E `test_bse.py` smoke script; worth deleting once that script is retired or repointed.

---

## Worst misfits (both rankings, unreconciled — see disagreement #6)

**S1's #1 / S2's #4 — `bse_jax.py` (626 L).** A_driver + a fully dead matvec trio sitting next to the
real dispatch logic + a Tier-E diagnostic demo + Tier-E ring-test/ring-check plumbing + the largest
argparse surface in the package (D1). Four tiers (A/D/E/F) in one file.

**S2's #1 / S1's #2 — `bse_ring_comm.py` (996 L).** Two live matvec factories (A1, TDA + non-TDA/B1),
the package's single most-imported infra symbols (C1, 13 importers), two diagnostic CLIs (E4), two
fully dead functions (F), and density-space probe operators serving only the B3 W-source alternates.
Five tiers (A/B/C/E/F) in one file — the single largest tier-count straddle in the package by either
sorting's count.

**Joint #3 — `bse_io.py` (932 L).** A0 ingest + A5 output writer + D2 (private ad-hoc cohsex.in
parsing) + F (dead `BSEData`) + the A0 head-injection duplication bug-smell, all in one file.

**Joint #4 — `bse_preconditioner.py` (158 L).** Worst dead-to-live ratio in the package (see Tier F).

**Joint #5 — `davidson_absorption.py`.** A second, largely-parallel end-to-end driver (load → Davidson
solve → dipole projection → BGW `eigenvalues_b{1,2,3}.dat` write) that substantially duplicates the
`--lanczos --solver davidson` route already reachable through `bse_jax.py`, with its own bespoke output
stage bolted on. Worth auditing whether this should collapse into `bse_jax`'s dispatch (one driver, one
output path) — the same "no redundancy in refactors" pattern the GW map flagged for
`eqp_bgw`/`gw_output`/`gw_jax` (memory `feedback_no_redundancy`).

**Named once each:** `bse_feast.py`'s "setup utilities" docstring vs. its reality as the default solver
path (A2c); the `solvers/` package's own `__init__.py` docstring claiming "no physics dependencies"
while half its files (davidson, chebyshev, dos, pseudobands) are consumed only by `psp/`, not `bse/` at
all (C2/C3/C5) — a BSE maintainer running `ls src/solvers/` sees a package that is roughly half foreign
territory with no subpackage boundary to signal it.

---

## Sources

- `_raw_sorts.json` — the two independent tier sortings verbatim, as given.
- `_digest.json` — the 19-file reader digest verbatim, as given (purpose/entry-points/features per
  file, every claim independently grep-verified against `adc2197`).
- `files/*.md` — the 19 per-file notes, authoritative for line numbers and quoted code.
