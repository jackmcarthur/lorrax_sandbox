# Exciton bandstructure pipeline — session worklog (agent A, 2026-07-17)

Branch: `agent/bse-exciton-bands` (worktree `sources/worktrees/lorrax_A_exciton_bands`),
based on `agent/bse-phase2` @ c4c349f (fast-forwarded from 0426efe after the
non-TDA solver landed; TDA-only scope here).

## Plan (tracked steps)

1. [x] Worktree + FFI .so copy + fast-forward to c4c349f.
2. [x] Required reading (arbitrary_q_bse §§1-2,12-13(+14 pending in-flight),
       F_SCHEME_NOTE, PRIMER §III.5, REFERENCE_arbitrary_q_vq.py + README,
       PHASE2_LOG recompile+sharding sections, JOINT_FINDINGS, finite_q_bse,
       AGENTS.md + CONVENTIONS).
3. [x] `src/bse/vq_interp.py` — production port of REFERENCE_arbitrary_q_vq
       (prepare_coarse sharded + FFI-eigh dispatch; host b26p LSQ; ONE jitted
       eval_vq, Q-dependent data as runtime args).
4. [x] `bandstructure/bse_setup.compute_wfns_fi` extension: explicit q_list +
       return_coeffs (the ~40-LOC §1 generalization).
5. [x] `src/bse/exciton_bands.py` — Q-path driver (K_POINTS crystal_b reuse,
       single-compile lax.scan of per-Q block-Lanczos over the stack matvec,
       .dat + PNG outputs).
6. [x] Gates: tests/test_bse_vq_interp.py (+ driver smoke), sharding assert,
       compile census.
7. [x] Refit mode (per-Q ζ refit ground truth via htransform full-r recon).
8. [x] Final run: MoS2 3×3 640-centroid fixture, Γ→M→K→Γ ≥30 pts interp +
       ~5 refit spot checks; census + timing + memory_analysis.
9. [x] PHASE2_LOG §"Exciton bandstructure pipeline", CHANGELOG, commits.

## Fixture decision (recorded)

The mandate names "MoS2 gnppm fixture". Its `zeta_q.h5` stores ζ at the 5 IBZ
q only (IBZ cascade active) — vq_interp training needs full-BZ ζ (9 q of the
3×3). The `runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native` fixture (640
centroids, same MoS2 3×3 system, nval=26/ncond=54/nband=80) has full-BZ ζ,
W0_ready=True, WFN.h5 + cohsex.in — and is the exact fixture the reference
impl's pinned acceptance numbers were measured on. Final run + acceptance
gates use it; IBZ-ζ unfolding for vq_interp is recorded as deferred work
(route: the ONE SymMaps sym-action, not a parallel helper).

## Convention decisions (recorded up front)

- Exciton momentum Q: electron leg shifted +Q (`|v k, c k+Q>`, LORRAX
  convention). Reference-metric arithmetic has conduction at k−q with tile
  V_q ⇒ the driver evaluates the exchange tile at q_tile = wrap(−Q).
  Gated at on-grid Q against the stored V_qmunu tile.
- Γ endpoint: exactly-Γ uses the production q=0 tile (stored V_q0 convention,
  head per loader); at finite Q the G=0 term is kept (energy_loss/full
  v(Q+G) — the §finite_q adjudication). Documented in the .dat header.
- htransform ψ are cell-periodic u at wrapped labels (fH_q periodic in q) —
  matches the torus convention of the stored grid ψ (no umklapp phases),
  same convention the finite-q W resolvent validated.
- Interpolated V_Q is Hermitized (0.5(V+V^H)) before entering the solver —
  commutes with the B-contraction, identical to the reference's exciton_evs
  Hermitization.

## Log

- 2026-07-17: worktree created; rebase onto c4c349f (non-TDA solver commit;
  bse_io/bse_lanczos deltas reviewed — TDA paths unchanged).
- 2026-07-17: fixture recon: gnppm zeta_q_G (5,399,1963) = IBZ-only;
  640-centroid cohsex fixture full-BZ (9,640,1963), W0_ready — adopted.
- 2026-07-17: 4h 1-node GPU alloc requested (lx-alloc-jackm tag), background.
- 2026-07-17 (alloc 56074608): smoke_vq PASS 19s — vq_interp port reproduces
  the reference e2e baseline to EVERY digit (B med 1.409e-2 / max 3.553e-2,
  exc 0.642/2.542 meV); jit-vs-host ~3e-16; census 1; P('x','y') asserted.
  Commit 1f16ea2.
- 2026-07-17: driver smoke 3-pt path PASS 36s total — ONE compile,
  warm 1.52 s/Q, temp 467 MiB; htransform@Γ gate Δε=0.000 meV, subspace
  overlap min-sval 0.943. smoke3pt.dat/png written. Γ-row cross-check vs
  production solve_bse_sharded launched (check_gamma.py).
- 2026-07-17/18 SOLVER BUG FOUND+FIXED (commit 196c30b): block-Lanczos
  Krylov exhaustion (bs·max_iter > n_flat=144) manufactured sub-spectrum
  ghosts 60-100 meV BELOW the dense ground state in EVERY solver run.
  Clamp at floor(n/bs) in solvers/lanczos.py; driver full-reorth default.
  Post-fix: production Lanczos == dense to 0.0000 meV; driver Γ row ==
  dense(htransform-ψ) exactly; htransform representation floor at Γ
  = 2.25 meV vs dense(stored-ψ) (Kramers-doublet rotations + edge mixing).
- 2026-07-18 REFIT convention chase (diag_refit*.py, disk logs in tasks):
  * refit mechanics green: Galerkin full-r residual 5.4e-15, Hermitian
    1e-16, my normal equations solved to 7e-12, rcond-insensitive below
    1e-6 (cond(C)=1.8e7).
  * stored-ζ phase convention derived + fixed in refit_vq:
    ZG_μ = e^{−2πi q·s_μ}·FFT(ζ̃_μ) — B null improved 10.9%→1.97% (q1).
  * REMAINING on-grid null gap ~2-3.7% B (htransform m-leg 2.0-2.9%,
    stored m-leg 3.7-5.1%): the stored fit satisfies NO variant of my
    normal equations (torus / umklapp / conj / q-sign all r≈15%, partly
    a sphere-truncation floor); C matches build_cq to 2e-3; producer
    kernels decode per-element to my equations. Allegiance test (which
    pair family the stored ζ FITS) in flight.
  * Fallback framing if unresolved: refit = self-consistent independent
    truth with a MEASURED on-grid systematic (~2-3% B) vs the stored-fit
    family; off-grid verdict = interp-vs-refit compared against that
    on-grid floor.
- 2026-07-18: allocation 56074608 lost (background salloc task killed);
  re-granted 56075918 (3.5h).
- 2026-07-18 FINAL: allegiance test → stored ζ fits the TORUS family
  (E 0.112 vs umklapp 0.123) — refit convention pinned; E_min(refit,
  full-grid ζ) = 8.5e-4 vs E_stored(band-limited) = 0.112 → the stored
  representation's expansion error is sphere-truncation-dominated; the
  2-3% B on-grid systematic is the truncation-sector realization, and it
  collapses at the exciton level (on-grid K: ≤0.13 meV / 8 states).
- 2026-07-18 FINAL RUN (alloc 56077214): 32-pt Γ→M→K→Γ interp + 5 refit
  rows in ONE scan compile (37 rows, warm 713 ms/Q, temp 514 MiB);
  interp-vs-refit off-grid ≪0.1 meV median, 1.85 meV worst state.
  exciton_bands_GMKG.{dat,png} final. smoke_refit PASS; new tests 4/4;
  full suite 233 passed / 1 fixed test.
