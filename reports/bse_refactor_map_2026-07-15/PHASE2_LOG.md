# BSE Phase-2 execution log

Branch `agent/bse-phase2` off `main` (6bd4dc9) in `sources/lorrax_A`. Four
self-contained commits: dense-reference gate → B1 dense-exchange fix →
trial-stack matvec → consumer wiring. All targeted 1-GPU checks green;
**full plain 1-GPU suite: pending orchestrator run**.

## Commits

| # | sha | title |
|---|-----|-------|
| 1 | `6d52999` | dense-reference kernel gate for the Q=0 BSE Hamiltonian |
| 2 | `d7b51a1` | B1 fix — dense (k-summed) Q=0 exchange in all live V paths |
| 3 | `11bab32` | trial-stack BSE matvec (scan-inside-shard_map, one T alive) |
| 4 | `5d3819f` | wire block-Lanczos + FEAST subspace onto the stack matvec |

## Commit 1 — dense-reference gate

`tests/test_bse_dense_reference.py` + session-scoped `bse_dense_state` fixture
(piggybacks `gnppm_session`; MoS2 3×3×1, 2v2c ⇒ N=nc·nv·nk=36; no second GW
run). Builds the explicit `⟨cvk|H|c'v'k'⟩` matrix from the same padded,
head-injected arrays (`bse_io._load_ring_subset`) the production matvecs consume.
Audit of the predecessor's draft: correct and faithful to the design; removed
one dead helper (`_kdiag`). Verified pre-B1 on 1 GPU: **3 passed** (W-only
positive control, serial/simple/ring — pins the convolution sign q=k−k′),
**7 xfailed** (strict) full-H/D+V/spectrum. §4 BGW `bsemat.h5` off-diagonal test
was **not** added — the Si data (`runs/Si/04_si_4x4x4_bse/00_bgw_bse/bsemat.h5`)
is absent, so it would only ever skip (deferred).

## Commit 2 — B1 dense exchange

The Q=0 exchange is DENSE in (k,k′): `⟨cvk|K^x|c'v'k'⟩ = (1/Nk) Σ_{μν}
conj(M_cvk(μ)) V_q0(μν) M_c'v'k'(ν)`, no δ_kk′ (VERDICT.md). Every matvec kept k
as a batch axis (`S[b,ν,k]`), computing only the (k,k) diagonal scaled 1/Nk.
Fixed, single-sourced, in the three live V paths (`bse_serial`, `bse_simple`
+ new `sh.S_k0` accessor, `bse_ring_comm.apply_V_ring` — which transitively
fixes the non-TDA B-block via `apply_V_ring_B`). The preconditioner diagonal
(`bse_feast.build_preconditioner_diagonal_sharded`) legitimately keeps the (k,k)
element and is untouched; the density-snapshot / transition-generator helpers
are not H-exchange paths and are untouched.

**Verification (1 GPU, MoS2 3×3, N=36):** all matvecs are now bit-exact to the
dense reference —

| matvec | full-H relerr | D+V relerr |
|--------|--------------|-----------|
| serial | 1.9e-15 | 8.9e-16 |
| simple | 2.2e-15 | 1.1e-15 |
| ring   | 2.3e-15 | 9.9e-16 |

Gate flipped: 10 passed (W-control ×3, full-H ×3, D+V ×3, spectrum).

### Eigenvalue shift (before = k-diagonal exchange, after = dense; gate fixture)

Both spectra are `eigvalsh` of the same reference builder — only the exchange
kernel's off-diagonal k-blocks differ — so this isolates the physical B1 shift.

| idx | before (Ry) | after (Ry) | Δ (meV) | | idx | before (Ry) | after (Ry) | Δ (meV) |
|----:|------------:|-----------:|--------:|-|----:|------------:|-----------:|--------:|
| 0 | 0.008253 | 0.008189 | −0.9 | | 10 | 0.100036 | 0.100035 | −0.0 |
| 1 | 0.008255 | 0.008317 | +0.8 | | 11 | 0.100038 | 0.100038 | −0.0 |
| 2 | 0.010900 | 0.010682 | −3.0 | | 12 | 0.119331 | 0.119308 | −0.3 |
| 3 | 0.010914 | 0.010763 | −2.1 | | 13 | 0.119359 | 0.119322 | −0.5 |
| 4 | 0.017570 | 0.017570 | −0.0 | | 14 | 0.121185 | 0.119497 | −23.0 |
| 5 | 0.017570 | 0.017571 | +0.0 | | 15 | 0.121494 | 0.119633 | −25.3 |
| 6 | 0.025485 | 0.024752 | −10.0 | | 16 | 0.121548 | 0.121485 | −0.9 |
| 7 | 0.025499 | 0.024896 | −8.2 | | 17 | 0.121893 | 0.121696 | −2.7 |
| 8 | 0.098299 | 0.098299 | −0.0 | | 18 | 0.122861 | 0.122829 | −0.4 |
| 9 | 0.098399 | 0.098382 | −0.2 | | 19 | 0.122893 | 0.122845 | −0.7 |

Exchange-sensitive states move up to ~25 meV; the lowest exciton pair splits a
few meV. Exchange-insensitive states (4/5, 8–13) are unchanged — expected: the
dense exchange only couples states with overlapping transition densities.

### Spectrum-gate note (solver conditioning)

Design (d) asked for an *iterative* lowest-4 check. Both single-vector and block
Lanczos are numerically fragile on this fixture: the q=0 head injection makes
the V/W ISDF tiles O(1e5) (`V_q0[0,0]≈2.3e5`, vhead=1655, whead=322) and they
near-cancel against D, so the Krylov solvers return ghost (repeated-lowest) or
below-λ_min Ritz values across every tested block config — a solver-conditioning
issue orthogonal to B1. The gate therefore **materialises** the corrected serial
matvec (one batched application to the identity basis) and compares its full
spectrum to the dense reference (relerr 2.6e-15, spectrum max|Δ|=3e-16) — a
robust, solver-independent proof. Flagged for follow-up (below).

## Commit 3 — trial-stack matvec

`src/bse/bse_stack_matvec.py`: `build_bse_stack_matvec(mesh, nkx,nky,nkz, *,
kernel='bse'|'rpa')`. The W (direct) term is ONE `shard_map` over `('x','y')`
whose body is a `lax.scan` over the trial axis, so XLA reuses the body scratch:
exactly one `T`-family (`μ_loc·ν_loc·ns²·nk`) alive regardless of `n_trials`.
Encode all_gathers (v over y, c over x); decode `psum_scatter`s (μ→scatter c on
x, ν→scatter v on y — no replicated `(c_full,v_full)` buffer survives). Exchange
is the batched B1 dense form (S,U k-free, kept outside the scan). W-tile seam is
the single line `U = fft_k(W_R·ifft_k(T))`. `fft_helpers` consolidation:
factored `local_ifftn3`/`local_fftn3` (one source); `make_sharded_*fftn_3d` now
wrap them and the scan body calls the kernel directly (shard_map cannot nest).

**Equality (1 GPU, n_trials=4, per trial):**

| kernel | vs dense H | vs simple matvec |
|--------|-----------|-----------------|
| bse (D+V−W) | 3.7e-15 | 2.4e-15 |
| rpa (D+V)   | 1.5e-15 | 1.2e-15 |

**Memory — `compiled.memory_analysis().temp_size_in_bytes`** (one-trial T bound
= `μ_pad²·ns²·nk·16 B` = 399²·2²·9·16 = **91.7 MB**):

| n_trials | stack temp | ring temp | bound×n_trials |
|---------:|-----------:|----------:|---------------:|
| 1 | 183.9 MB | 183.9 MB | 91.7 MB |
| 4 | **183.4 MB** | 734.1 MB | 366.8 MB |
| 8 | **183.4 MB** | 1467.7 MB | 733.6 MB |

**Verdict:** stack peak temp is FLAT in `n_trials` — the `n_trials` axis appears
on no intermediate. Peak ≈ 2× the one-T bound (the FFT `T_R`/`U_R` scratch, as
the design predicted ≈3×). The legacy ring temp is strictly linear. Gate
`test_bse_stack_matvec.py` asserts flatness (temp(8) < 1.25·temp(1)), the
single-T bound (temp < 5× one-T), and temp(8) < 0.5·ring(8).

## Commit 4 — consumer wiring

`build_bse_stack_matvec` is a bit-exact, dtype-adaptive drop-in (same 9-arg
signature; `1/√Nk` now follows the input dtype so it also serves the fp32 GMRES
path). Repointed:
- `bse_lanczos.solve_bse_sharded` — bs==1 Lanczos, bs>1 block-Lanczos, Davidson
  `apply_H`; dropped the `matvec_kind` selector and the unused
  `build_bse_ring_matvec` import.
- `bse_feast.run_feast_ritz` (TDA) — shifted-GMRES contour solves + Rayleigh-Ritz
  on the stack matvec; `_rayleigh_ritz` applies H to the filtered TDA subspace in
  ONE batched `jnp.concatenate` dispatch instead of a per-vector Python loop.
  Non-TDA keeps `build_bse_ring_matvec_full` (B2-blocked).

**Smoke (1 GPU, MoS2 3×3 restart):** `solve_bse_sharded` bs=1 lowest-4
`[0.008189, 0.008317, 0.010683, 0.010763]` == dense reference; bs=4 block path
finite; `run_feast_ritz` TDA GMRES+Ritz runs end-to-end (4 in-window Ritz
values). All consumer paths execute through the stack matvec.

## Deferred / follow-up

1. **Ring/gather/simple retirement** (NOTED, not executed — per mandate). Still
   live: `build_bse_ring_matvec` (used by `estimate_spectral_bounds_sharded` +
   the equality gates) and `bse_simple` (gates). Delete together with the
   `matvec_kind` data key once the spectral-bound Lanczos is repointed.
2. **Non-TDA (`build_bse_ring_matvec_full`)** — out of scope (B2 malformed
   B-encode unfixed). When B2 lands it should reuse `bse_stack_matvec._w_stack`
   encode/decode rather than its own.
3. **Solver conditioning** — single-vector + block Lanczos return ghost /
   below-λ_min Ritz values on the head-injected operator (V/W tiles ~1e5
   near-cancelling D). A conditioning/preconditioning fix (or a scaled inner
   product) is needed before the iterative solvers are trustworthy on
   head-injected BSE. Orthogonal to B1.
4. **BGW `bsemat.h5` off-diagonal exchange gate** (design §4) — needs
   `runs/Si/04_si_4x4x4_bse/00_bgw_bse/bsemat.h5` + a Si restart (absent). Add as
   an `extra` gate when the data exists.
5. **W(ω)/ladder seam** — `get_W_R(ω)`/`ladder_kernel` provider closures are the
   designed hook (scan body is byte-identical for a different `W_R`); not built.

## Reusable artifacts

- Persistent MoS2 3×3 restart + diagnostics under
  `/pscratch/sd/j/jackm/lorrax_sandbox/tmp_phase2/` (module-free srun+shifter
  runners; `bse_phase2_diag.py`, `bse_consumer_smoke.py`). Delete when done.
- The `extra`-marked `test_report_before_after_eigenvalues` regenerates the
  eigenvalue table from the gate fixture: `pytest -o addopts="" -s
  tests/test_bse_dense_reference.py -k before_after`.

## Orchestrator verification (post-executor, 2026-07-16)

1. **Full plain 1-GPU suite on agent/bse-phase2: 218 passed / 12 skipped /
   0 failed (7:13)** — all four golden gates + the new BSE gates (xfails
   flipped). `cleanup_verify/phase2_full_suite.log`.
2. **Multi-device closure of the memory-review finding** (evidence was 1x1-only):
   two-leg differential, same solver config on 1x1 vs 2x2 mesh (4 GPUs,
   production create_mesh_2d), MoS2 gnppm fixture:
   - bs=1: max|1x1 − 2x2| = **1.5e-16** (bit-level) — the stack matvec is
     device-count invariant under real partitioning.
   - bs=4: the two TRUE Ritz values (0.00818851, 0.00831683) are bit-identical
     across meshes; the other two entries are GHOST Ritz values
     (0.00829/0.00851 vs 0.00830/0.00859) that differ across meshes by ~7e-5 —
     the documented block_lanczos_eig ghost/transposed-beta/final-slot defect
     family (solver_program.md P1, pre-existing, NOT introduced by this
     branch; wiring did not change solver behavior). Reproduction numbers
     recorded here for the P1 fix gate.
3. Observed benign warning under the new wiring: "Some donated buffers were
   not usable: complex128[200,200,3,3,1]" (per-device W_q slab donation
   falls back to copy in some solve paths) — perf-only; fold into solver P1.

Artifacts: `cleanup_verify/bse_multidev_check.py` + `evals_1x1_bs*.txt`.

## W(0) resolvent cross-check (2026-07-16, agent/bse-phase2)

Owner-requested cross-validation: confirm the GW static screened Coulomb obeys
the Casida resolvent identity

    W(0) - v  =  v (0 - H_RPA)^{-1} v

in the ISDF centroid basis, reproducing the restart's `W0_qmunu - V_qmunu` q=0
tile to the GW numerical-integration noise.  Diagnostic: `src/bse/bse_w_exact.py
--compare-w0` (shifted GMRES on the non-TDA Casida via `bse_feast`).  Fixture:
the gnppm gate restart (MoS2 3x3, nspinor=2, nval=26/ncond=20, 399 centroids,
`W0_ready=True`), 1 GPU.  Run dir `runs/MoS2/A_bse_w0_resolvent_2026-07-16/`.

### What the resolvent path needed (three fixes on this lineage)

1. **Stale `data["W_R"]` (KeyError).**  `_apply_shifted_matvec` reads the 8th
   matvec arg `data["W_R"]`, which the restart loader never emits (only `W_q`).
   Single-sourced the `W_q -> W_R` conversion into `bse_feast.ensure_W_R` and
   repointed all four call sites (FEAST main + fp32, spectral-bound Lanczos,
   `bse_w_exact`); the 3 inline copies are gone.

2. **The kernel is RPA test-charge screening, not the exciton BSE.**  The
   non-TDA symplectic H that reproduces W is `[[D+V, V],[-V, -D-V]]` with the
   RING coupling `V = K^A = (1/Nk)<M_t|v|M_t'>` in BOTH blocks (density-density
   RPA bubble).  The existing `build_bse_ring_matvec_full` B-block used the
   excitonic `V_B` (conjugated pairing, Henneke 2-20) — the correct optical-BSE
   kernel, but a *different response*: it overshoots the q=0 tile by ~1.8x
   (measured: ratio 1.794, std 0.003).  Added a `screening=True` flag to that
   builder selecting the ring `K^A` B-block (single matvec, physics-flagged;
   asserts `include_W=False`).

3. **Symplectic combination (was zero at omega=0).**  The old `bse_w_exact` used
   `rhs=[f;f]` + readout `X+Y`, which gives IDENTICALLY 0 at omega=0 in the
   non-interacting limit (`1/(0-D)+1/(0+D)=0`).  Correct RPA density super-vertex
   is `rhs=[f;-f]` (same `f`, minus on the anti-resonant Y block) with readout
   `s=X+Y`, `w_c = v(M s)`.  Derivation: with `H=[[A,B],[-B,-A]]` the response
   operator is `diag(-1,1)(0-H)`, so `(0-H)^{-1}[f;-f]` yields `X=Y=-(A+B)^{-1}f`
   and `A+B = D+2K^A` — exactly the folded static RPA `-(1/Nk)M(D/2+K^A)^{-1}M`.
   Cross-checked bit-for-bit densely (below).

### Convention lock-in (dense, head-less q=0 tile, exact static 1/(e_c-e_v))

| dense construction | relerr vs disk W0-V | diag ratio |
|--------------------|--------------------:|-----------:|
| full RPA `(I-Vχ0)^{-1} Vχ0 V` | **2.34e-9** | 1.00000 |
| folded `-(1/Nk) M (D/2+K^A)^{-1} M†` | **2.16e-9** | 1.00000 |
| symplectic ring `B=K^A`, `[f;-f]`, `X+Y` | **2.16e-9** | 1.00000 |
| symplectic exciton `B=K^B` (old kernel) | 7.9e-1 | 1.79406 |
| symplectic `B=0` (TDA) | 5.1e0 | 6.12 |

Both `χ0 = -2/Nk Σ_{cvk} M conj(M)/(e_c-e_v)` convention and the RPA Dyson
`W=(I-Vχ0)^{-1}V` (matching `w_isdf.solve_w`) are confirmed by the 2.3e-9 match.
The exciton and TDA kernels do NOT reproduce W — the owner's "non-TDA, not TDA"
call is right, with the ring (not V_B) B-block.

### Sharded result — `bse_w_exact --compare-w0 --n-cols 8 --seed 7` (1 GPU)

chi0 window `n_val=26 n_cond=20` (full occ x cond, = GW `compute_screening`);
`gmres(max_iter=200, tol=1e-10)`; head-less bodies both sides.

| nu | \|\|(W0-V)_col\|\| | rel_err | max\|Δ\| | gmres_resid |
|-----:|------------:|---------:|---------:|------------:|
| 179 | 6.601e+07 | 2.157e-09 | 3.92e-02 | 2.37e-10 |
| 375 | 6.601e+07 | 2.157e-09 | 3.92e-02 | 2.37e-10 |
|  63 | 4.729e+07 | 2.349e-09 | 3.13e-02 | 3.83e-10 |
| 267 | 4.729e+07 | 2.349e-09 | 3.13e-02 | 3.82e-10 |
| 373 | 9.699e+06 | 2.173e-09 | 5.30e-03 | 3.90e-10 |
| 247 | 8.586e+06 | 2.408e-09 | 5.80e-03 | 4.31e-10 |
| 357 | 7.212e+06 | 2.303e-09 | 4.68e-03 | 3.97e-10 |
| 272 | 1.702e+07 | 1.924e-09 | 8.16e-03 | 2.50e-10 |

**max rel_err = 2.41e-9, median = 2.24e-9; max gmres_resid = 4.3e-10** (12
GMRES iters/column).

### Interpretation — does it close at the minimax-noise level?

**Yes.**  The resolvent uses the EXACT static denominator `1/(e_c-e_v)`; the disk
`W0_qmunu` is `W(0)` from `χ0(iω)` on the minimax Laplace nodes evaluated at ω=0.
So `rel_err` IS the GW minimax-quadrature error of `1/x` over `[E_gap, E_max]` —
here a very tight **~2e-9** (this fixture's energy range is modest).  The GMRES
residual (~3e-10) sits an order of magnitude below `rel_err`, so the closure is
quadrature-limited, NOT solver-limited: the two are cleanly separated, exactly as
the report format requires.  The identity `W0 = v(0-H_RPA)^{-1}v + v` holds.

Degenerate pairs (nu 179/375, 63/267 share norms) are symmetry-related centroids
— expected on the D3h MoS2 mesh.

Artifacts: `runs/MoS2/A_bse_w0_resolvent_2026-07-16/` (fixture restart, module-free
`lxrun_free.sh`, `explore_*.py`/`verify_sharded.py` dense-vs-sharded harnesses).
Gate: `tests/test_bse_w0_resolvent.py` (closure < 1e-6 on the gnppm fixture).

## W-column sharding (2026-07-16, agent/bse-phase2)

Sharding-quality upgrade of the W-column resolvent path so it emits a
device-resident `W(mu_X, nu_Y)` tile on the square processor grid and becomes a
single-sourced engine for the future Lanczos-chain `W(omega)` model.  Exemplar:
the Sigma_PPM reduce-scatter (`gw.ppm_tau_kernel._make_project_ri_reduce_scatter`,
which emits `sigma(m_X, n_Y)`).

**New single engine** `bse_w_exact.apply_screening_resolvent_block(G_zeta, z, ...)`
— three named stages that the Lanczos model reuses verbatim (docstring carries
the plug-in): (1) SEED zeta->pair (`gen`, batched over the whole probe block),
(2) SOLVE per-column-independent shifted GMRES via `lax.scan` over the probe
axis (one Krylov subspace alive; bit-identical to the old per-column Python loop
because each column's norm/LSQ reductions are global-per-single-column), (3)
PROJECT pair->zeta reduce-scatter -> `W(mu_X, nu_Y) = sh.V`.  `_resolve_wc_columns`
is now a thin unit-column wrapper over it; `--compare-w0` and the gate share the
one implementation.

**Before -> after layout.**

| stage | before | after |
|-------|--------|-------|
| probe columns | Python `for nu` loop, one GMRES dispatch + one host `device_get` per column | one batched seed + `scan` of GMRES + one batched projection (3 dispatches) |
| output | host `np.stack` -> `(nu, mu)` numpy (implicit all-gather to host) | device tile `W(mu_X, nu_Y) = P('x','y')`, no replicated `(mu, nu)` |
| projection | `snapshot` `psum('y')` -> `(nu, mu_X)` replicated on nu | `psum_scatter('y', scatter=nu)` fused reduce+scatter -> `(mu_X, nu_Y)` |

`build_density_snapshot_operator` gained a build-time `scatter_nu_on_y` flag
(default False keeps the `(b, mu_X)` contract that `bse_pseudopoles` relies on;
True is the W-tile reduce-scatter).  Gate now asserts `W_tile.sharding.spec ==
P('x','y')`.

**Pre-existing multi-device bug found + fixed (the seed boundary).**
`--compare-w0` had only ever run on 1 GPU.  On 2x2 the OLD code returned
`rel_err ~0.5` (max|Delta| ~ column norm) while GMRES still converged
(resid ~9e-11) — bit-identical to the new code on 2x2, proving the refactor is
value-faithful and the defect is upstream.  Cause:
`build_realspace_random_transition_generator` did the centroid (mu) contraction
as a LOCAL x-slice einsum with the conduction index pre-sliced to the x-rank's
block, so on px>1 each c-block received only its aligned mu-slice (~50% of the
sum).  Fix mirrors `apply_V_ring`: full c, local-mu partial, then
`psum_scatter('x', scatter=c)` to complete the mu sum across x AND scatter c —
a no-op at px=1 (values unchanged), correct at px>1.

**Validation (gnppm fixture, MoS2 3x3, nval=26/ncond=20, 399->400 centroids).**

| leg | max rel_err | median | max gmres_resid |
|-----|------------:|-------:|----------------:|
| 1x1 (6 cols) | 3.203e-9 | 2.253e-9 | 4.22e-10 |
| 2x2 (6 cols)  | 3.203e-9 | 2.253e-9 | 4.22e-10 |

Per-column bit-identical 1x1 vs 2x2 (179/375 -> 2.157e-9, 63 -> 2.349e-9,
253 -> 2.467e-9, 204 -> 3.203e-9, 337 -> 2.064e-9) — device-count invariant.
Overlapping columns match the pre-refactor 1-GPU baseline above (179/375
2.157e-9, 63 2.349e-9) bit-for-bit.  No OOM/memory blowup (2x2 resolve 19.8 s;
the seed's full-c transient equals the one `apply_V_ring` already carries every
matvec).  Gate `tests/test_bse_w0_resolvent.py` passes (33 s) with the added
PartitionSpec assertion; core BSE gates 14/14 (50 s).

Runners: module-free srun+shifter, 1 GPU `runs/MoS2/A_bse_w0_resolvent_2026-07-16/lxrun_free.sh`
(4-GPU = same with `--gres=gpu:4`, `--px 2 --py 2`).

## Symmetry-centroid degeneracy experiment (2026-07-16)

**Do orbit-closed (symmetry-obeying) centroids restore exact BSE degeneracies
on Si? NO.** Old/sym intra-manifold splitting ratio 1.004–1.018; sym splittings
remain 500–2000 μeV vs BGW ~2 μeV (the 4v4c doublet reproduces the historical
~485 μeV datum). The sym arm used FEWER centroids (792 orbit-closed vs 960
literal) with near-identical splittings — neither symmetry closure nor count is
the lever. **The symmetry violation enters downstream of centroid placement:
the ψ IBZ→full-BZ unfold and/or the ζ-fit are not symmetry-covariant** — this
is the deferred unified-sym-action / ψ-side Phase-2 work, now with a concrete
observable (Si manifold splitting) to gate it. Full table + setup:
`runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/report.md`. Side product: a
vectorised dense-H builder (bit-equal to `_build_dense_H`, rel-err 4.8e-17)
worth folding into the gate for larger windows.

### SUPERSEDED interpretation — symmetry-breaking root cause found (2026-07-16, diag pass)

The "ψ IBZ→full-BZ unfold / ζ-fit covariance" hypothesis recorded above is
**REFUTED by direct measurement** (runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/diag/FINDINGS.md):
energies are covariant to 0 μeV, ψ-at-centroids to 1e-15 (closed Γ multiplets
to 1e-9). **Root cause: band-window truncation of degenerate multiplets at
high-symmetry k** — at Γ the Si valence top (Γ₂₅′) and conduction bottom (Γ₁₅)
are 6-fold (nspinor=2); a 4v4c window keeps 4 of 6, making the transition
density non-covariant exactly there. The 518 μeV doublet split is a
near-cancellation of ±3000–4300 μeV contributions on the small-orbit stars
(Γ: −3252, star1: +4303; generic size-24 k: −208). Degenerate-CLOSED Γ window
[2,8)×[8,14) restores multiplets to ≤36 μeV (~100×). This is a property of any
fixed (nv,nc) BSE window, not of LORRAX — larger fixed windows just cut
different multiplets (6v6c/8v8c stay ~2000 μeV). The correct LORRAX-vs-BGW
degeneracy gate uses degenerate-closed windows (or manifold-averaged
comparisons); the April "BGW ~2 μeV" datum needs its window convention
re-established before being treated as a reference.

**Genuine LORRAX defect found en route** (subdominant here, real elsewhere):
V0/W0 ISDF tiles are ~3% non-covariant under the centroid permutation (worst
on nonsymmorphic ops; q=0 head injection worsens V0 to ~8%); contracts to
~1e-4 in kernel blocks. Filed for the tile/head-injection path (fits the
w_head_wings / GW-infra alignment work).

## W-column resolvent profiling (2026-07-16, agent/bse-phase2)

Performance audit + tuning of `apply_screening_resolvent_block` (and the
`--compare-w0` path), the future Lanczos-chain W(ω) engine.  Fixture: gnppm
restart (MoS2 3×3, nspinor=2, nval=26/ncond=20, 399→400 centroids), 8 probe
columns, z=0, GMRES(max_iter=200, tol=1e-10) = 11 iters/col.  All times are warm
min-of-N after warmup (`profile_resolvent.py`); COLD includes first-compile.

### Hot-spot table — BEFORE (warm min, % of stage sum)

| stage (op)          | 1×1 time | 1×1 % | 2×2 time | 2×2 % |
|---------------------|---------:|------:|---------:|------:|
| SEED (`gen`)        | 1.84 s   | 46.4  | 4.06 s   | 26.8  |
| SOLVE (scan-GMRES)  | 0.095 s  |  2.4  | 9.78 s   | 64.6  |
| PROJECT (`snapshot`)| 2.03 s   | 51.2  | 1.30 s   |  8.6  |
| **e2e (warm)**      | **4.62 s** |     | **18.44 s** |     |

**Root cause — the GMRES/matvec was never the bottleneck.**  `gen` and
`snapshot` were *bare* `shard_map`s: an eager `shard_map` call re-traces and
re-lowers to HLO **every call** (the trace is not memoized), so they cost
~2–4 s apiece while doing trivial compute.  The BSE ring matvec is already
`jax.jit`-wrapped, so SOLVE dispatches its cached executable (~1 ms/matvec at
1×1).  Isolated micro-benchmark (single call, 1×1):

| op        | bare      | jax.jit  | speedup |
|-----------|----------:|---------:|--------:|
| `gen`     | 2536 ms   | 0.83 ms  | **3050×** |
| `snapshot`| 2773 ms   | 2.60 ms  | **1069×** |

At 2×2 the picture shifts: the un-jitted boundaries still re-lower (SEED 4.06 s,
PROJECT 1.30 s), but SOLVE now dominates (64.6 %) because the ring matvec's
collectives (ppermute rings + psum_scatter over 4 GPUs) cost ~20 ms/call on this
tiny per-device problem — latency-bound, not compute-bound.

### Fix (commit) — jit the seed/project reshard boundaries

`build_realspace_random_transition_generator` and
`build_density_snapshot_operator` now return `jax.jit(shard_map, in_shardings=…,
out_shardings=…)` (same house style as the matvec's own `jax.jit`).  Caches the
compiled executable so repeated calls dispatch instead of re-lowering.  Single
source — also speeds the `bse_pseudopoles` FEAST-seed path that calls the same
builders.

### Hot-spot table — AFTER (warm min)

| stage (op)          | 1×1 time | (Δ) | 2×2 time | (Δ) |
|---------------------|---------:|----:|---------:|----:|
| SEED (`gen`)        | 0.0025 s | 740× | 0.020 s | 204× |
| SOLVE (scan-GMRES)  | 0.095 s  |  —  | ~11 s    |  —  |
| PROJECT (`snapshot`)| 0.0022 s | 920× | 0.0054 s| 241× |
| **e2e (warm)**      | **0.75 s** | **6.2×** | **12.07 s** | **1.53×** |

At 1×1 the resolvent is now ~40× faster in device work (stage-sum 0.10 s vs
3.96 s); e2e is 6.2× (a residual ~0.65 s host-orchestration gap remains, below).
At 2×2 the win is 1.53× — the fix removed the 5.36 s of SEED+PROJECT re-lowering;
what remains (SOLVE ~11 s, 99.8 %) is the shared ring matvec's collective cost,
out of scope for this pass (see below).

### Numerics faithfulness (adversarially checked)

- **Gate: 14 passed / 1 deselected** (`test_bse_w0_resolvent` +
  `test_bse_stack_matvec` + `test_bse_dense_reference`), `W_tile.sharding.spec ==
  P('x','y')` holds.  Per-column closure rel_err after = recorded before to all
  4 reported sig figs (e.g. 179→2.157e-9, 63→2.349e-9, 272→1.924e-9; max
  2.4078e-9), gmres_resid ~4e-10.
- **Device invariance survives at its true (pre-fix) level.**  The "1×1-vs-2×2
  bit-identical" property was never raw-bit-identical: the *bare* pre-fix tile
  already differed **5.64e-12** relative across meshes (inherent multi-device
  `psum` reduction order — at py>1 the ν-sum is split+recombined, a different fp
  association than the single-device einsum).  The jitted tile differs
  **6.31e-12** — the same level.  The claim always meant device-count-invariant
  *to reported precision*, which is preserved.
- **jit perturbation is XLA eager-vs-jit fp reassociation, not my shardings.**
  Plain `jax.jit(shard_map)` and the sharded-jit give *identical* results
  (gen 1.271e-13, snap 1.118e-8-abs ≈ 1e-15 rel vs bare — same at 1×1 and 2×2).
  End-to-end same-mesh change bare→jitted ≈ 8e-12 rel (1e-13 at the `gen` source,
  amplified through 11 GMRES iters) — ~400× below the 2.4e-9 physics closure and
  6 orders below the 1e-6 gate tolerance.  No numerics knob or precision changed.

### Deliberately NOT done (with reasons)

1. **Column-batched / block-GMRES matvec.**  The matvec IS underutilized —
   batch-scaling shows ~10× throughput headroom (1×1: b=1 1.06 ms → b=32 0.09
   ms/col; 2×2: b=1 20.7 ms → b=8 1.06 ms/col).  But (a) at 1×1 SOLVE is 2.4 % —
   irrelevant; (b) at 2×2 SOLVE dominates yet the lever is the *shared* ring
   matvec (FEAST, spectral-bound Lanczos, every BSE solver), whose collective
   latency a resolvent-local batch cannot touch without a per-column-reduction
   block-GMRES.  That rewrite risks the bit-level closure and violates
   no-redundancy: `gmres_solve_sharded_jit` is shared with FEAST and its global
   norms/LSQ are correct only per-single-column (TDA batch axis 0 vs non-TDA axis
   1 differ).  The design chose scan for "one Krylov subspace alive"; kept.
2. **Eliminate the per-column true-residual recompute** (one extra matvec/col).
   Measured 3.1 % of SOLVE at 1×1 (= 0.08 % of e2e), in the noise at 2×2.  It is
   a required diagnostic (gate asserts resid<1e-6); swapping it for the GMRES LS
   estimate would change a reported value and touch the shared FEAST return
   signature.  Not worth it.
3. **z as a runtime arg (compile-once across shifts).**  The Lanczos W(ω) model
   replaces stage-2 (scan-GMRES) with block-Lanczos + tiny per-ω tridiagonal
   solves; a z-generic GMRES optimizes a path the model won't use.  Stages 1/3
   (the reshard boundaries this commit jits) are what the model reuses verbatim.
4. **One jit over the whole `apply_screening_resolvent_block`** (would close the
   ~0.65 s/1×1 host-orchestration gap between e2e 0.75 s and stage-sum 0.10 s).
   Cross-stage fusion could perturb fp further; the per-boundary jit is the
   minimal, targeted change.  Candidate once the W(ω) model's numerics tolerance
   is fixed.

Artifacts (run dir `runs/MoS2/A_bse_w0_resolvent_2026-07-16/`): `profile_resolvent.py`
(stage/hotspot/batch-scaling harness), `probe_jit_shardmap.py` (bare-vs-jit
micro-bench + faithfulness), `probe_bitid.py` + `run_bitid.sh` (stash-bracketed
device-invariance baseline), `validate_after.py`, `prof_{1,2}gpu_{before,after}.log`,
`gate_after.log`.  Runners: module-free srun+shifter (`lxrun_free.sh`, 4-GPU =
`lxrun_free_4gpu.sh`).

## Finite-q W_q resolvent check (2026-07-16, agent/bse-phase2)

Owner-requested generalization of the W(0) resolvent to FINITE q: generate W_q
at the symmetry-reduced q-grid one at a time and validate each against the
restart's own `(W0_qmunu − V_qmunu)[q_flat]` tile.  **Done — all 5 IBZ q's on
the MoS2 gnppm fixture close at the GW minimax-quadrature floor.**

### Engine (single-source generalization, NOT a fork)

The finite-q RPA density response is the on-grid `|v k, c k+q⟩` pair basis — a
k-axis remap of the CONDUCTION slots + a V-tile swap.  The matvec, seed,
project, solver, and sharding are byte-identical to q=0.

- `common/symmetry_maps.kgrid_shift_map(nkx,nky,nkz,q_off)` — the ONE place the
  C-order `k+q` fold + umklapp-`G` arithmetic lives (pure numpy; gather ≡
  `jnp.roll`; unit-gated).
- `bse_io.load_bse_data_from_restart_sharded(..., load_v_full=True)` → the full
  `V_qmunu(μ,ν,nkx,nky,nkz)` tensor `data['V_q_full']` (default False keeps q=0
  byte-identical).
- `bse_w_exact.build_finite_q_data(data, q, mesh)` — roll `ψ_c`/`ε_c` by `+q` on
  the reshaped (nkx,nky,nkz) k-axis (`jnp.roll`), set `V_q0 = V_qmunu[q_flat]`.
  `q=(0,0,0)` is the identity.  `--compare-wq` loops `SymMaps.q_irr_kgrid_int`.

### Convention lock-in (derived + numerically validated)

Roll direction and the umklapp phase were determined by a dense χ0→W_q sweep
(`finite_q/dense_finite_q.py`) over sign × phase, each vs the disk tile:

| construction | q=(0,1,0) | q=(1,0,0) | q=(1,1,0) | q=(1,2,0) |
|---|---:|---:|---:|---:|
| roll `+q`, **no phase** | **1.3e-4** | **1.2e-4** | **1.1e-4** | **8.0e-5** |
| roll `−q`, no phase | 8.0e-1 | 9.1e-1 | 2.0e0 | 1.4e0 |
| roll `+q`, umklapp phase on | 6.3e-1 | 5.8e-1 | 2.1e0 | 1.7e0 |

**Roll conduction by `+q`, NO umklapp Bloch phase.**  Derivation: GW's χ0(q) is a
plain *periodic* FFT-convolution over k (`w_isdf._get_chi_minimax_kernel`:
`χ_q ∝ Σ_k Gc_k Gv*_{k+q}` with the RAW stored ψ at the wrapped index — the DFT
delta enforces `k+q mod N`, no phase).  Rolling `ψ_c`/`ε_c` by `+q` gives the
pair density `conj(ψ_c[k−q])ψ_v[k]` = that convolution (relabel k→k−q), and the
finite-q V tile `V_qmunu[q_flat]` KEEPS G=0 (`compute_vcoul` zeroes G=0 only at
q=0) with NO separate head.  The design-doc `exp(−2πi G_umk·s_μ)` phase applies
to a DIRECT-READ finite-Q BSE against a differently-built reference — it BREAKS
the match to this FFT-convolution-produced tile (0.6–3.2 vs 1e-8).

### Three GMRES defects the stiff finite-q tiles exposed (shared solver fix)

The finite-q V_q carries a large G=0 head ⇒ `cond(H)~1e8`.  The shared
`_get_gmres_solver` diverged (true resid O(1)) via three coupled defects, all
fixed (q=0 / FEAST unchanged; head-less tiles are well-conditioned):

1. **Normal-equations LSQ** `solve(HᴴH)` squared the condition to ~1e17 ≈ 1/eps
   → garbage `y` → false projected early-exit → `lstsq` (QR/SVD).
2. **Single Gram-Schmidt** lost orthogonality catastrophically (`||VᴴV−I|| → O(1)`
   by ~20 iters) → falsely tiny projected residual, rounding-dependent solve →
   added a DGKS **reorthogonalization** pass (`||VᴴV−I|| ≲ 1e-14`).
3. **Operator-blind solver cache** keyed on `(max_iter,tol,dtype)` but the solver
   closes over `matvec`/`data`, so the q-loop silently reused q=0's operator for
   every later q → key now includes `id(matvec)`/`id(data)` (refs held).

### Per-q closure — `bse_w_exact --compare-wq --n-cols 6` (MoS2 gnppm, 1 GPU)

chi0 window n_val=26 n_cond=20 (full occ × cond), N_μ=399, gmres(200, 1e-10),
head-less bodies; each q vs its OWN `(W0−V)[q_flat]`.

| iq | q (kgrid) | q_flat | max rel_err | median | max gmres_resid |
|---:|:---------:|-------:|------------:|-------:|----------------:|
| 0 | (0,0,0) | 0 | 2.349e-9 | 2.190e-9 | 4.31e-10 |
| 1 | (0,1,0) | 1 | 2.854e-8 | 2.484e-8 | 2.39e-10 |
| 2 | (1,0,0) | 3 | 2.905e-8 | 2.537e-8 | 3.22e-10 |
| 3 | (1,1,0) | 4 | 5.265e-8 | 4.871e-8 | 1.87e-10 |
| 4 | (1,2,0) | 5 | 3.146e-8 | 2.530e-8 | 2.46e-10 |

**max per-q rel_err = 5.3e-8**, all `gmres_resid ~2e-10` (≥100× below closure →
quadrature-limited, not solver-limited).  Finite-q floor sits ~10× above q=0's
2.3e-9 (the roll/exact-1/D vs minimax residual grows with the number of wrapped
k's) — still cleanly at the GW quadrature noise floor.  Identity
`W_q = v_q(0−H_RPA^q)⁻¹v_q + v_q` holds at every symmetry-reduced q.

### Validation

- Gate `tests/test_bse_w0_resolvent.py`: **3 passed** — W(0) closure (unchanged
  2.467e-9), `kgrid_shift_map` unit (permutation / k+q fold / G∈{0,1}³ / roll≡
  gather / q=0 identity), finite-q closure at the smallest nonzero q (<1e-6).
- BSE gates 16/16 (`test_bse_stack_matvec` + `test_bse_dense_reference` 13
  passed / 1 deselected).  **FEAST smoke green** (gnppm restart, W2–W4 Ritz
  1.75–3.32 eV finite) — the shared GMRES change is FEAST-safe.  GMRES is BSE-only
  (grep: consumers = FEAST + bse_w_exact), so the golden GW gates are unaffected.

Artifacts (run dir `runs/MoS2/A_bse_w0_resolvent_2026-07-16/finite_q/`):
`dense_finite_q.py` (convention sweep), `diag_gmres.py` / `gmres_lsq_test.py` /
`lstsq_pad_test.py` / `scan_vs_loop.py` (GMRES defect isolation),
`matvec_check.py` (sharded-vs-dense matvec 1e-16), module-free
`lxrun_freshcache.sh`.

### Finite-q addendum: full-suite confirmation (orchestrator)

The finite-q agent skipped the full suite on a blast-radius argument, but
kgrid_shift_map lives in GW-shared common/symmetry_maps.py — so the full plain
1-GPU suite was re-run at HEAD 6ca714b: **221 passed / 12 skipped / 0 failed**
(5:16), golden gates included. cleanup_verify/finite_q_full_suite.log.

## Audit follow-up: P3 pair-amp hoist + P5 donation drop + c64 flag (2026-07-16)

Three approved items from the matvec efficiency audit
(`archive/matvec_efficiency_audit/JOINT_FINDINGS.md` §1/§3/§4/§6). Landed on
`agent/bse-phase2` in `sources/lorrax_A` (module-free srun+shifter, 1 GPU A100).
Owner-excluded P2 (`apply_V_ring` rewrite) and P-NT (nt-aware dispatch) untouched.

### P3 — hoist the V-term pair amplitudes out of the per-iteration matvec

`M_X`(μ on x)/`M_Y`(ν on y) `= Σ_s conj(ψ_c)ψ_v` (`compute_pair_amplitude`) were
rebuilt inside EVERY matvec (`bse_stack_matvec:138,143`; the equivalent
`apply_V_ring` decode einsum) — the matvec is a per-iteration black-box jit whose
ψ args XLA cannot hoist across calls (audit P3 CONFIRMED). Now computed ONCE at
load (`bse_io.load_bse_data_from_restart_sharded` → `data["M_X"]/["M_Y"]`, the
single source) and threaded as matvec args across ALL sharded matvecs — stack,
simple, ring, ring-full — with a uniform 11-arg signature (append `M_X,M_Y` to the
existing 9). `apply_V_ring` now slices this rank's y-block out of the hoisted
full-v `M_X` (was: slice ψ_v then GEMM — same values, one fewer GEMM/iter);
`bse_feast` `_apply_shifted_matvec`/`_rayleigh_ritz`/`_build_gmres_data_fp32`,
`bse_lanczos`, `absorption_haydock`, `davidson_absorption`, `bse_kpm`,
`bse_pseudopoles` all carry M from `data`. Per-variant-unused args (`psi_c_Y` in
stack/simple, `M_Y` in the ring, `psi_v_X` where not the B-encode) are retained
for a calling convention shared by the FEAST GMRES/Ritz drivers — additive diff,
no `_apply_shifted_matvec` branch, no `apply_V_ring` structural change (keeps P2
conflict-free).

**Finite-q coupling (caught by the gate).** `bse_w_exact.build_finite_q_data`
rolls ψ_c/ε_c by +q; the hoisted M's shallow-copied from `data` were then STALE
(built from unshifted ψ_c) → wrong screening operator (GMRES converged, closure
rel_err 2.66 at q=(0,1,0)). Fixed: recompute `M_X`/`M_Y` from the ROLLED ψ_c in
`build_finite_q_data`. The `test_wq_resolvent_matches_restart_finite_q` gate flipped
red→green on that recompute.

**Memory.** Peak-neutral (both M's already lived inside every matvec call); only
the between-matvec floor rises by ~2·M/p (audit §2b). At the audit's inflated 1×1
regime `M_X = M_Y = 471 MB` each.

**Timing (before/after, warm min-of-50, 1 GPU c128, inflated nc48/nv48/ns2/nk16/μ800).**
The stack matvec == one block-Lanczos iteration; BEFORE recomputes M each call from
its ψ args (psi passed as jit ARGS → no constant-folding, faithful to the pre-hoist
matvec), AFTER receives the precomputed M's:

| nt | BEFORE min (ms) | AFTER min (ms) | Δ per matvec |
|---:|----------------:|---------------:|--------------|
| 1  | 15.11 | 13.67 | **+1.45 ms (+9.6%)** |
| 4  | 48.59 | 47.17 | **+1.42 ms (+2.9%)** |

The saving is a FIXED ~1.4 ms/matvec (M is X-independent — two 471 MB pair-amp
GEMMs removed from the hot loop), so it is 9.6% of the nt1 (single-vector / GMRES
contour / spectral-bound-Lanczos) matvec and 2.9% of the nt4 (block-Lanczos) matvec
— matches the audit's "~10% of matvec bytes, zero comms" prediction. `after` vs
`before` **relerr 7.6e-17 (nt1) / 4.1e-16 (nt4) = bit-identical** (machine ε).

### P5 — drop cosmetic `donate_argnums` (audit §3)

Removed the declined donations at `bse_lanczos:240` (W_q) and
`absorption_haydock:221` (W_q) — `W_R=ifft(W_q)` is a fresh buffer with no aliasable
same-shape output — and `bse_ring_comm` `apply_W_from_T`'s `donate T` (output `WX`
shape ≠ T shape). The last appears in BOTH ring builders
(`build_bse_ring_matvec` + `build_bse_ring_matvec_full`) → **4 sites** (the audit
listed 361/581 as one site; I dropped both — same cosmetic donation, keeping one
would be an inconsistent half-fix). All were declined (no fallback copy, §3). Full
1-GPU suite: the BSE "Some donated buffers were not usable" warnings are GONE (the
3 residual warnings are FFI `test_ffi_linalg_contract` cusolvermp, unrelated).

### Item 3 — c64 flag comment ONLY (audit §4)

Comment at the stack W-term dtype seam (`bse_stack_matvec._w_stack`, the `sqrt_nk`
line the whole W-term dtype inherits from): complex64 mixed precision would ~halve
the 655 MB T-tensor and every ~7 HBM round-trip (measured ~2× W-term bandwidth
lever), DELIBERATELY left at c128 per owner decision (2026-07-16), pointer to
JOINT_FINDINGS §4. No behavioral change; no c64 anywhere.

### Bit-identity / tolerance honesty

No BSE gate asserts bit-identity — the dense-reference and stack-matvec gates use
`relerr < 1e-9` between paths (stack vs dense vs simple; ring vs dense). No tolerance
was touched. The production path passes M as a runtime jit arg computed by the SAME
einsum as the old inside-matvec M → bit-identical; the direct before/after check is
relerr ~1e-16. (A 1e-12 "drift" in a v1 timing harness was a compile-time constant-
fold artifact from closing ψ over the jit — absent when ψ are real jit args.)

### Validation (module-free srun+shifter, 1 GPU A100, job 56012954)

- BSE gates: `test_bse_dense_reference` + `test_bse_stack_matvec` +
  `test_bse_w0_resolvent` — **all green** (incl. finite-q after the M recompute fix;
  W(0) closure 2.467e-9 unchanged).
- **Full plain 1-GPU suite: 221 passed / 12 skipped / 0 failed (5:22)** — identical
  pass count to the pre-change baseline; all four golden GW gates + the BSE gates.

## W(omega) Lanczos-chain model (2026-07-16, agent/bse-phase2, lorrax_A)

The feature the whole W-resolvent arc was preparing: full-frequency screened
Coulomb ``W_q(omega)`` from ONE structure-preserving block-Lanczos chain per q,
replacing the per-omega shifted-GMRES solves.  New module
``src/bse/w_omega_chain.py`` (builder + evaluator, plain arrays/functions, no new
class); CLI mode ``bse_w_exact --w-omega-chain``; gate
``tests/test_bse_w_omega_chain.py``.  The shifted-solve path (``--compare-w0`` /
``--compare-wq``) is KEPT as the validation ORACLE (gate reference / ground
truth), NOT a parallel production path.

### Structure-preserving reduction (the recurrence, per-element)

The RPA screening operator is the para-Hermitian ``H_RPA=[[A,B],[-B,-A]]`` with
``A=D+V``, ``B=V`` (ring kernel ``K^A``), ``D`` = diagonal transition energies
``eps_c-eps_v > 0``.  The screened column is
``W(z)-v = L (zI-H)^{-1} B_seed`` with ``B_seed_nu=[f_nu;-f_nu]``,
``f_nu=M^dag v e_nu`` (= SEED/``gen``) and ``L(x)=v M (x_X+x_Y)`` (= PROJECT/
``snapshot``).  In the (q=X+Y, p=X-Y) basis H acts as ``q'=(A-B)p``,
``p'=(A+B)q``; the seed is pure-p (q=0) and the readout reads q, so the 2N
symplectic resolvent collapses to an N-dim SYMMETRIC one (Casida ``Omega^2`` /
Shao et al. product structure):

    W(z) - v = 2 * Phi [ z^2 I - S ]^{-1} Phi^dag,
      S      = D^{1/2}(A+B)D^{1/2} = D^{1/2}(D+2V)D^{1/2}   (Hermitian, Euclidean),
      Phi    = v M D^{1/2},   Phi^dag e_nu = D^{1/2} f_nu   (the SEED, D^{1/2}-scaled).

``A-B=D`` is EXACT for screening, so this is exact, not an approximation
(numpy prototype ``proto_chain.py``: full-symplectic vs symmetric-reduced rel
5e-16; block-chain vs oracle machine-exact at full length).  Only ``z^2`` enters
=> ONE chain serves the whole complex plane.  ``S`` is applied through the
production matvec VERBATIM: ``(A+B)U = matvec([U;U])[X-block]`` (since
``H[U;U]=[(A+B)U;-(A+B)U]``), plus a ``D^{1/2}`` transition-diagonal scale — no
new kernel, no duplicated encode/decode, matvec call signature unchanged.

Symmetric block Lanczos on ``S`` (block width ``p=len(cols)``, seed
``B0[b]=D^{1/2} f_{cols[b]}``, block-QR ``B0=Q_0 R_0``): for ``j=0..m-1`` (blocks
``p x (c,v,k)``, inner product Euclidean over ``(c,v,k)`` completed by the mesh
allreduce)

    Wb = S Q_j ;  alpha_j = Q_j^dag Wb ;
    Wb = Wb - Q_j alpha_j - Q_{j-1} beta_{j-1}^dag ;
    (DGKS) Wb -= sum_{i<=j} Q_i (Q_i^dag Wb)   [full reorthogonalization, 2 passes] ;
    Q_{j+1} R = Wb (block-QR) => beta_j = R

=> block-tridiagonal ``T`` (diag ``alpha_j``, sub ``beta_j``, super
``beta_j^dag``).  Full reorth is mandatory (the head-injected/stiff tiles lose
orthogonality catastrophically otherwise — same lesson as the GMRES DGKS pass).
Block-QR uses the robust Gram/eigen route (tiny ``p x p`` ``G=W^dag W`` on host,
near-zero eigenvalues deflated) so degenerate/parallel probe columns and zero
pad columns cannot break the chain.

Evaluator (per omega, tiny; ``z=(omega+i eta)/Ry``, ``E=[R_0;0;..;0]``):
``C(z)=(z^2 I-T)^{-1} E`` (mp x p host solve) ;
``x(z)=sum_j Q_j C_j(z)`` (device einsum over the stored chain blocks) ;
``W(z)-v = 2 snapshot(D^{1/2} x(z))`` (reduce-scatter to the ``(mu_X,nu_Y)=sh.V``
tile).  No matvec, no GMRES per omega.  Chain blocks live in the pair basis
(stacked ``(m,p,c,v,k)`` on the ``sh.X_full`` spec); ``T``/``R_0`` are replicated
host numpy; the evaluator projects with the existing PROJECT machinery — single
source.  ``--chain-len`` is the accuracy knob (no new class); block width = probe
width (small, the ring matvec's optimal regime, keeping ``T`` small).

### Convergence vs chain length — q=0 (probe cols 179/375/337/253, p=4)

``rel_vs_oracle`` = max-column ``||W_chain(m)-W_oracle||/||W_oracle||``;
``rel_vs_disk`` (omega=0 only) = closure vs the on-disk ``(W0-V)`` tile.

| m   | omega=0 (=disk) | 2i eV  | 10i eV  | 1.5+0.1i eV |
|----:|----------------:|-------:|--------:|------------:|
| 8   | 1.10e-2 | 8.86e-3 | 8.10e-4 | 1.28e-2 |
| 16  | 2.79e-3 | 1.66e-3 | 6.73e-6 | 4.01e-3 |
| 32  | 2.17e-4 | 6.89e-5 | 3.85e-8 | 5.22e-4 |
| 64  | 6.54e-6 | 1.05e-6 | 9.58e-12| 5.47e-5 |
| 96  | 5.49e-7 | 4.95e-8 | 9.70e-12| 1.30e-5 |
| 120 | 1.95e-7 | 7.11e-9 | 9.70e-12| 4.49e-6 |

Monotone everywhere.  **omega=0 reproduces the disk ``(W0-V)`` closure THROUGH the
chain evaluator** (rel_vs_disk == rel_vs_oracle to all figures), driving toward
the GW minimax floor (~2.4e-9) as m grows.  The **imaginary axis** (the
GW-relevant axis, chi(i omega)) converges fastest and saturates at the oracle's
own GMRES residual (10i -> 9.7e-12 by m=64).  Real-axis-below-gap is monotone but
slowest (Krylov-standard for interior real omega with small eta).

### Convergence — finite q=(0,1,0) (p=4)

| m  | omega=0 | 3i eV  | 8i eV  | 1.0+0.15i eV |
|---:|--------:|-------:|-------:|-------------:|
| 16 | 1.61e-2 | 5.36e-3 | 3.44e-4 | 1.90e-2 |
| 32 | 2.47e-3 | 3.69e-4 | 5.50e-6 | 3.60e-3 |
| 48 | 4.85e-4 | 4.68e-5 | 1.00e-7 | 7.82e-4 |

Finite-q floor ~10x above q=0 (same behavior the oracle shows), still cleanly
convergent.  Finite-q reuses ``build_finite_q_data`` (rolled psi_c + per-q V tile)
unchanged.

### Timing / amortization (warm, 1 GPU) — the whole point

| case | chain build | per-omega oracle | per-omega chain eval | speedup/omega | break-even |
|------|------------:|-----------------:|---------------------:|--------------:|-----------:|
| q=0, m=48       | 2.23 s | 961 ms | 9.3 ms  | 103x | ~2.3 omega |
| q=0, m=120      | 11.1 s | 942 ms | 13.7 ms | 69x  | ~12 omega  |
| q=(0,1,0), m=48 | 2.18 s | 983 ms | 22.9 ms | 43x  | ~2.3 omega |

The chain is built ONCE (m matvecs); each extra frequency is 40-100x cheaper than
a fresh oracle solve (a full block-GMRES: ~11 iters x p cols).  **omega-count
break-even is ~2-12 frequencies** — far below any real GW/BSE frequency grid
(10-30 imaginary nodes + a real grid), so the model wins for any production
sweep.  Default ``--chain-len 32`` (~1e-6 imaginary / ~2e-4 static on this
fixture); raise for tighter (>=96 for the ~1e-9 static floor).

### Validation

- Gate ``tests/test_bse_w_omega_chain.py``: **2 passed** (46.8 s) — q=0 + smallest
  nonzero IBZ q, chain-vs-oracle at static/imaginary/real omega, monotonicity
  (m vs m/2 on the imaginary axis), and the ``W_tile.sharding.spec == P('x','y')``
  tile contract.
- Only-owned edits: ``bse_w_exact.py`` (CLI ``--w-omega-chain`` + harness),
  ``w_omega_chain.py`` (new).  ``apply_V_ring`` internals untouched (consumed as
  is); the matvec call signature is unchanged.
- Artifacts: run dir ``runs/MoS2/A_bse_w_omega_chain_2026-07-16/`` (``proto_chain.py``
  numpy math proof, module-free ``lxrun_free.sh``, ``manifest.yaml``, gate logs).
