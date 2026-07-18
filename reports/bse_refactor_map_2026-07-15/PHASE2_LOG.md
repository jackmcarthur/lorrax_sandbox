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

## Si BSE symmetry — Round 2: closed-window residual + tile-defect root cause (2026-07-16)

Round-2 diagnosis (read-only on source; scratch in
`runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/diag2/`, full trail in
diag2/FINDINGS2.md). Attribution verdict: **the Round-1 "genuine tile defect"
is the SAME degenerate-multiplet-cut mechanism as the BSE window, in the
SCREENING ISDF fit window, amplified by CCT conditioning — NOT a
fractional-translation-phase bug — and in the closed-window regime it IS the
remaining symmetry breaker.**

- **Closed-window residual is tile-driven.** Γ-on-site closed block [2,8)×[8,14):
  little-group-symmetrizing the q=0 tiles collapses the residual 36.39 → 0.000
  μeV (head off; 34.70 → 3.06 μeV with head — the 3.06 is raw-G0's own 8.6%).
  Full coupled BSE: genuine exciton multiplets split up to 15.4 μeV (raw) →
  <1 μeV (covariant), max eigenvalue shift 8.8 μeV. (V0_sym covariant to
  6.4e-16; production unfold_v_q roundtrip exact 0.0.) The 2031 μeV "8v8c
  manifold" is distinct excitons, NOT a broken multiplet.
- **Born in the ζ-fit, τ-blind.** Bisection ψ(1e-15)→CCT-input(0.4%)→ζ-head
  G0(8.6%)→V0(3.2%)→W0(3.0%); corr(V0,W0)=0.997 (W inherits V). 8.6% is at G=0
  (τ phase = 1) yet worst on a nonsymmorphic op ⇒ nonsymmorphic-worst is base
  rate (36/48), REFUTING the Round-1 phase hypothesis. V_q assembly is a
  faithful bilinear ζ-contraction (τ cancels leg-to-leg; per-element proof).
- **Root cause: screening band-window cut + CCT conditioning.** Production
  screening window [0,8)×[8,60) cuts band 59's multiplet; every
  degeneracy-closed conduction top gives the CCT covariant to 6e-10 vs 4.3e-3
  open. cond(CCT)=3.6e9 amplifies the seed 20×. Fix (design only, FINDINGS2
  Task 3): degeneracy-round the screening fit window (common/meta.py:99
  b_id_4_user=nband → gw_init band_range_right; today world_size-round-up
  only), via a helper in gw/degen_average.py. No solver change needed.
- **Degeneracy gate designed** (Γ-on-site closed-window eigvalsh on the gnppm
  session fixture, new tests/test_bse_degeneracy.py, two-tier μeV threshold);
  NOT committed (needs one-time fixture threshold calibration).

Supersedes the Round-1 "genuine tile defect (fractional-translation phase,
nonsymmorphic-worst)" note.

## P2 + P-NT — exchange comms-reduction + nt-aware dispatch (2026-07-16, agent/bse-comms-opt)

Two approved count-reduction items from the matvec audit (JOINT_FINDINGS §5-6,
trace_dossier §2c), in an ISOLATED worktree
`sources/worktrees/lorrax_A_comms_opt` off `agent/bse-phase2` HEAD 0ecc7d7.
Commits 4301434 (P2), ae2d1fc (P-NT). NOT merged into bse-phase2 (owner + main
coordinate after the W(ω) work lands).

### P2/C1 — apply_V_ring 6-collective ring exchange → shared GSPMD form
`apply_V_ring` computed the dense q=0 exchange with **6 collectives/apply** (2×py
band-ppermute rings + psum + psum_scatter); the non-TDA resolvent SOLVE calls it
4×/matvec = **24** (the ~20 ms/call floor). Replaced by ONE shared helper
`bse_ring_comm.bse_exchange_gspmd(X, M_enc, M_X, V_q0, sh, nk)` — the same
`S=Σ_kcv M_enc·X` / `U=V_q0·S` / `VX=Σ_M conj(M_X)·U` GSPMD form the stack matvec
already used, per-element identical to `apply_bse_hamiltonian_single_device`
(dense-reference gate). **Single source of truth**: `bse_stack_matvec` dropped its
inline copy and calls the helper; both ring matvecs (TDA + non-TDA full) call it.
`apply_V_ring` + the four `_apply_V_ring*` shard_map wrappers DELETED (dead). The
A/B blocks differ ONLY in the encode amplitude (`M_Y` for A / RPA-screening B;
`conj(M_Y)` for the optical coupling block). B2 was never a blocker: the resolvent
(`screening=True`) has no W-term, the B1 fix already made the B-block exchange
correct, and only the W-term B-encode (`encode_T_B`) stays ring-based (out of scope).

**Collectives / matvec (2×2, optimized HLO, start-side counts):**

| operator | before | after |
|---|---|---|
| resolvent (non-TDA screening, 4 sub-applies) | **40** (32 collective-permute + 4 all-reduce + 4 reduce-scatter) | **12** (8 all-reduce + 4 all-gather) |

The topology-blind ppermute band-rings are GONE. In the audit's per-apply logical
units this is 6→3 (2 all-reduce + 1 all-gather), i.e. resolvent **24→12** — NOT
24→8: the audit's "2 all-reduce/apply" undercounted the encode X all-gather; the
stack-style exchange compiles to 3 collectives/apply at HEAD.

### P-NT — nt-aware dispatch (bse_lanczos.solve_bse_sharded)
Routed every solve (incl. bs==1) through the trial-stack matvec — a measured ~1.5×
single-vector regression. Dispatch at the existing builder seam: **bs≤2 → ring,
bs≥3 (and Davidson's wide subspace) → stack** (crossover nt≈2-3, trace_dossier
§1/§4). Same 11-arg signature → pure builder swap; no new config surface
(`matvec_kind` stays retired). Other consumers already route sensibly (verified):
FEAST spectral-bound + Haydock use the ring (single-vector); FEAST GMRES contour
uses the stack (block).

### Validation (module-free srun+shifter, worktree PYTHONPATH, A100)
- **Gates GREEN**: `test_bse_dense_reference` + `test_bse_stack_matvec` +
  `test_bse_w0_resolvent` (incl. finite-q) — **16 passed / 1 deselected** (1 GPU).
- **Full plain 1-GPU suite: 221 passed / 12 skipped / 0 failed** (4:28) — identical
  to the pre-change baseline; no regressions.
- **Closure unchanged**: `validate_after` W0-V tile vs restart max rel_err =
  **2.4077e-9** at BOTH 1×1 and 2×2 (base 2.4078e-9) — device-count invariant.
  `--compare-wq` per-q closure = 7.9e-8 max (minimax floor), matching base.
- **Timing (2×2 warm)**: per resolvent matvec (min-of-20): b=1 **14.7→19.6 ms**,
  b=8 **1.94→1.51 ms/col**. `--compare-wq` `resolve_q` wall (5 IBZ q):
  before **148.25 s** / after **146.31 s**.

**Honest wall-time note.** The 40→12 count cut does NOT speed the SINGLE-COLUMN
(b=1) matvec on this latency-bound fixture — it is ~30% slower — because the 12
full-mesh all-reduce/all-gather cost more per barrier than 40 tiny point-to-point
ppermutes (the same crossover P-NT exploits: ppermute wins at nt≤2). It wins
BATCHED (b≥8) and is the correct topology-aware formulation for multi-node scale-out
(the audit's design lock against ppermute rings). The `--compare-wq` end-to-end wall
is GMRES-reorthogonalization-dominated (the matvec exchange is a minor fraction), so
the count cut is roughly wall-neutral at the current single-column resolvent; the
value is architectural (scale-out) + code (single source of truth, −62 net lines).

Artifacts: `tmp_comms_opt/` (probe_collectives.py, timing_matvec.py,
compare_wq_2x2_{before,after}.log, val_{1x1,2x2}_after.log, full_suite_after.log).

## Screening-window degeneracy fix + gate (2026-07-16, agent/screening-degeneracy-fix, lorrax_A)

Owner-approved Round-2 fix + gate (FINDINGS2 Task 3/4). Root-cause recap:
splitting a degenerate multiplet at the screening ISDF fit window top makes the
fitted ζ̃ — and every V_q/W_q tile — non-covariant under the crystal symmetry at
the high-symmetry k where the multiplet lives, amplified ~20× by cond(CCT)~1e10.

### The fix (commit b13bd4d)

`round_band_window_to_closed_shell(energies_kn_ry, b_hi, tol_ry, direction)` in
`gw/degen_average.py` — a boundary is degeneracy-closed when
`min_k(e[k,b]−e[k,b−1]) > tol` (BGW TOL_Degeneracy 1e-6 Ry); reuses the module's
existing contiguous-group logic, no parallel detector. `Meta.from_system` routes
`b_id_4_user` (→ `gw_init.fit_zeta band_range_right = (b1, b4)`) through it.

**Composition rule (documented in-code).** The physical top is rounded DOWN to
the closed shell FIRST; the world_size divisibility round-UP then pads with ZERO
bands (ψ=0, sentinel energies), NEVER real bands — so the padded window can never
re-cross a multiplet. `b_id_3 ≤ b_id_4` is a hard BandSlices invariant (a
reported QP band must sit inside the GF/screening band sum), so when the closed
shell falls below `b_id_3` the fix clamps at `b_id_3` and warns loudly rather
than reduce the σ OUTPUT set (whose identical exposure is flagged, not fixed).

Fix behaviour, `Meta.from_system` on the real Si WFN (nelec=8, nbands_file=62,
1 GPU):

| config | b_id_3 (σ top) | b_id_4 out | result |
|---|---:|---:|---|
| work_sym ncond=52 | 60 | 60 | CLAMP at 60 (closed shell 40 < b3); warn b3 exposure; 0 dropped |
| ncond=32 | 40 | 40 | round 60→40, **drop 20** |
| ncond=8 | 16 | 40 | round 60→40, **drop 20** |
| nband=40 (already closed) | 40 | 40 | no change, no warning |

### Golden-gate impact (Item B.2): NONE — no re-freeze

All four committed golden-gate fixtures leave the fix a no-op (their own WFN
energies, tol 1e-6 Ry):

| fixture | nband(b4) | b3 | gap at boundary b4 | closed_down(b4) | verdict |
|---|---:|---:|---:|---:|---|
| cohsex_debug | 40 | 30 | 870 meV | 40 | already closed → no change |
| gnppm_debug | 46 | 46 | 107 meV | 46 | already closed → no change |
| bispinor_debug | 32 | 30 | 1.035 meV | 32 | already closed → no change |
| si_cohsex_debug | 60 | 60 | 0 (cut) | 40 | closed shell < b3 → **clamp**, no change |

Full plain 1-GPU suite on `agent/screening-degeneracy-fix`: **224 passed / 12
skipped / 0 failed (4:52)** — all four golden gates + the new gate. No golden eqp
value shifts; no reference re-freeze needed.

### Si covariance validation (Item B.1)

before = `work_sym` (genuinely fix-OFF: restart built 2026-07-16 pre-fix,
ncond=52, screening `[8,60)` cut); after = `work_demo` (fix ON, ncond=32 → b4
60→40, screening `[8,40)` closed). Same 792 orbit-closed centroids/seed. q=0
covariance viol = `max_op ||T[α,α]−T|| / ||T||` under the 48-op centroid perm:

| quantity | before `[8,60)` | after `[8,40)` | ratio |
|---|---:|---:|---:|
| CCT C0 cov viol | 4.26e-3 | **7.04e-10** | ~6e6× |
| ζ-head G0 = ζ̃(G=0) | 8.64e-2 | **2.88e-7** | ~3e5× |
| V0 q=0 tile | 3.16e-2 | **7.49e-8** | ~4e5× |
| W0 q=0 tile | 3.01e-2 | **7.15e-8** | ~4e5× |
| cond(C0) | 3.59e9 | 2.01e10 | — |

The fix FULLY closes the q=0/ζ-fit covariance defect it targets — CCT to the
machine seed (7e-10), the q=0 tiles to ~1e-7 (the residual is the ~400×
CCT-conditioning amplification of the 7e-10 seed, still 5–6 orders below the 3%
cut-window level). (ncond differs 52→32 only to give the fix headroom below the
σ set; both work_sym legs are cut at 60 and both work_demo legs closed at 40, so
the comparison isolates window closure.) zeta_probe seed scan independently: the
same restart's every degeneracy-closed conduction top gives ~6–7e-10, only the
cut top breaks — the fix simply selects a closed top.

### Scope boundary — the full-BZ ALL-q BSE is a SEPARATE issue

`full_multiplet.py` (all 64 k, window `[0,8)×[8,16)`) is UNCHANGED by the fix:

| arm | max\|λraw−λcov\| | worst genuine-multiplet raw split |
|---|---:|---:|
| sym (fix OFF, 60-band) | 8.78 μeV | 15.41 μeV |
| demo (fix ON, 40-band) | 8.94 μeV | 15.77 μeV |

The screening-window fix closes the q=0 covariance (the exchange tile + the
ζ-fit), but the all-q BSE multiplet splitting is dominated by the finite-q
IBZ→BZ UNFOLD of the direct W_q tiles — the deferred TRS / unified-sym-action
Phase-2 work — which this fix does not touch. This REFINES FINDINGS2 Task 1(B)'s
hypothesis that "a fully covariant [q=0] set from the fix would close" the
full-BSE residual: the q=0 set IS now fully covariant (table above), yet the
finite-q residual persists at ~15 μeV, so it is a distinct defect. The gate is
correspondingly scoped to the q=0 Γ-on-site block.

### The gate (commit b3d1b01) + calibration (Item C)

`tests/test_bse_degeneracy.py`: over an auto-detected degeneracy-closed (nv,nc)
window at Γ, build `H_Γ = D+Kx−Kd` (numpy eigvalsh, (nc·nv)²) from the production
q=0 tiles and their little-group-symmetrized (exactly covariant) counterparts;
assert (1) `max|λraw−λcov| < TIGHT` (covariant tiles reproduce the production
spectrum) and (2) any covariant-spectrum multiplet has `cov_split < 1 μeV` and
`raw_split < TIGHT`. Piggybacks `gnppm_session` — no second GW run.

Calibration against the committed gnppm MoS2 fixture (ntran=2, auto window
nv=nc=4): little-group symmetrization moves V0/W0 by 2.80e-9 / 2.82e-9 and the
Γ-on-site spectrum by **4e-4 μeV** (tiles already covariant), and the
low-symmetry fixture has NO Γ exciton multiplets (invariant 2 vacuous there,
invariant 1 carries the gate — the design's "calibrate on what it has"). Two-tier
thresholds tied to the fix: TIGHT 5 μeV (active now that the fix lands with the
gate — 4 orders of margin) / LOOSE 50 μeV (the Si Γ-block raw floor from FINDINGS2
Task 1 a gross regression would cross). Gate passes standalone (24 s) and in the
full suite.

### Artifacts
`runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/`: `work_demo/` (fix-ON
ncond=32 restart), `fix_validate/` (`analyze_fixtures.py` golden-gate scan,
`meta_check.py`, `tile_cov.py`+`tile_cov.json`, `calibrate_gate2.py`+`calib.json`,
`full_suite.log`, `demo_gw.log`), `diag2/{zeta_probe,full_multiplet}_demo.json`.

## Per-q recompile elimination — one compiled engine serves all q (2026-07-17, agent/bse-phase2, lorrax_A)

`bse_w_exact --compare-wq` was paying a full XLA compile per q: ~5–6 s/q of
which the actual solve is tens of ms.  With static shapes there should be ONE
compile serving all q — "roll one wfn copy, use a different slice of V_q".  Root
cause and fix below; artifacts in `runs/MoS2/A_bse_w0_resolvent_2026-07-16/per_q_recompile/`
(`baseline_compare_wq.log`, `after_compare_wq.log`, `lxrun_census.sh`).

### It was CLOSURES, not unrolled loops

Confirmed by reading the loop structure: the GMRES inner iteration is already
`lax.while_loop` + `fori_loop` (Arnoldi / DGKS reorth), and the column loop is
already `lax.scan` over the probe axis — **no Python-unrolled loop anywhere**.
Every per-q recompile came from a fresh per-q Python closure becoming a distinct
jit-cache entry, with q-specific device arrays baked in as trace constants:

1. **The scan (the ~4.8 s cost).**  `_get_gmres_solver` keyed the solver cache
   on `(id(matvec), id(data), …)` and `_solve` **closed over** `data`.  The
   top-level `lax.scan` in `apply_screening_resolvent_block` therefore baked the
   q-specific operands (rolled ψ_c, ε_c, `V_qmunu[q]`, hoisted M) as scan
   constants → a new jaxpr and a fresh XLA compile per q.
2. **gen / snapshot (`_map`, 2/q).**  `_build_rpa_resolvent` was called **per q**,
   rebuilding the jitted seed/project shard_maps each time (new object → new
   compile), though they depend only on (mesh, k-grid, pad sizes) — not on q.
3. **The roll (`_roll_static`, 2/q).**  Device `jnp.roll` bakes the static
   q-offset into the program, so each q compiled a fresh roll.

### The fix (plumbing operands through the solver layer; NO physics change)

`bse_feast`: `matvec_operands(data)` returns the 10-tuple the ring matvec
consumes after `x`; `_apply_shifted_matvec` and the extracted module-level
`_gmres_solve_core` take that tuple as a **runtime argument** instead of closing
over `data`.  `_get_gmres_solver` now keys on `(id(matvec), max_iter, tol, dtype)`
— **id(data) dropped**.  Operator-identity safety is preserved: genuinely
different STRUCTURES (screening vs optical, TDA vs full) carry distinct `matvec`
objects → distinct engines (the id-keyed cache keys on the ENGINE, not per-q
closures).

`bse_w_exact`: new cached `_get_block_gmres_solver(matvec, sh, max_iter, tol,
dtype)` wraps the stage-2 per-column scan in **one** `jax.jit` (`_block`) whose
args are `(rhs, diag_h, z, operands)` — so it compiles once per operator
structure and every later q / omega is dispatch-only.  `z` is passed as a device
`complex128` scalar (not a Python complex), so a frequency sweep stays a runtime
arg too.  The compare-wq loop builds `matvec/gen/snapshot` **once** before the q
loop; per q it only rebuilds the small operand arrays + `diag_h`.  `jnp.roll` →
host `np.roll` at data-build (`_roll_k_axis_host`; arrays are ~tens of MB), so the
rolled array enters as DATA with no per-offset compile.  Bonus: the per-omega
oracle in `run_w_omega_chain_compare` (the chain's ground truth) inherits the
same one-compile engine — a ω-sweep no longer recompiles the scan per ω either.

Single-source preserved: one GMRES body (`_gmres_solve_core`) feeds both the
per-column FEAST path (`gmres_solve_sharded_jit`) and the block scan; no parallel
old/new path, no duplicated matvec kernel.

### Compile census — `--compare-wq --n-cols 6` (MoS2 gnppm, 1 GPU, JAX_LOG_COMPILES=1, cold cache)

| jitted fn | BEFORE (compiles) | AFTER (compiles) | note |
|:----------|------------------:|-----------------:|:-----|
| GMRES engine (`scan` → `_block`) | **5** (1/q, ~4.8 s ea) | **1** | the whole win |
| `_map` (gen SEED + snapshot PROJECT) | **10** (2/q) | **2** | built once |
| `_roll_static` (ψ_c + ε_c roll) | **10** (2/q) | **0** | host roll |
| one-time scalar ops (broadcast/transpose/…) | ~30 (q=0 setup) | ~30 (q=0 setup) | unchanged |

`_block` compiles exactly once, immediately before the iq=0 row; iq=1..4 emit no
further compiles.

### Timing — full 5-q loop (1 GPU A100, wall)

| section | BEFORE | AFTER |
|:--------|-------:|------:|
| `w_exact.resolve_q` (5 q) | 30.771 s | **4.604 s** |
| per-q solve: iq=0 (compile) | ~6.1 s | 3.264 s |
| per-q solve: iq=1..4 (warm) | ~6.1 s ea | 0.255 – 0.398 s |
| `w_exact.wq_build` (5 q) | — | 0.440 s |
| `w_exact.wq_compare` (5 q) | — | 0.002 s |
| Total recorded | 32.764 s | **6.553 s** |
| end-to-end run (incl. JAX init + load) | ~45 s | ~15 s |

New per-q sub-timers (`build[s]` / `solve[s]` columns + `wq_build` / `resolve_q`
/ `wq_compare` sections) are now printed by the compare-wq path on every run.

### Validation — closure IDENTICAL, gates green

Per-q rel_err **bit-identical** before ↔ after (the whole point — no physics
change):

| iq | q (kgrid) | max rel_err | median | max gmres_resid |
|---:|:---------:|------------:|-------:|----------------:|
| 0 | (0,0,0) | 3.203e-9 | 2.253e-9 | 4.22e-10 |
| 1 | (0,1,0) | 2.854e-8 | 2.444e-8 | 6.03e-10 |
| 2 | (1,0,0) | 2.905e-8 | 2.626e-8 | 3.48e-10 |
| 3 | (1,1,0) | 7.895e-8 | 4.871e-8 | 1.46e-10 |
| 4 | (1,2,0) | 2.820e-8 | 2.464e-8 | 3.77e-10 |

max per-q 7.895e-8, median 2.854e-8 (identical in both logs).  Gates on 1 GPU:
`test_bse_w0_resolvent` + `test_bse_w_omega_chain` + `test_bse_stack_matvec` +
`test_bse_dense_reference` = **18 passed / 1 deselected** (`gates_targeted.log`);
full plain 1-GPU suite green (`full_suite_1gpu.log`).  FEAST call site updated to
the operands signature (shared GMRES change is FEAST-safe).

## Non-TDA eigensolvers + solver P1 (2026-07-17, agent/bse-phase2, lorrax_A)

Full (non-TDA) optical BSE brought to the general lowest-eigenvalue solver
alongside TDA through the ONE `solve_bse_sharded` dispatch (`tda` toggle), with
the FIRST value validation of the coupling B-block WITH screened W — which
exposed and fixed a real operator bug.  Fixture: gnppm gate restart (MoS2 3x3,
nspinor=2, 2v2c => N=36), 1 GPU.  All numbers machine-exact vs dense.

### The bug: the non-TDA matvec computed the wrong (complex-spectrum) operator

`build_bse_ring_matvec_full(screening=False)` computed `H = [[A,B],[-B,-A]]`.
Materialising A, B from the matvec and comparing to the analytic dense build:

- **A is Hermitian** (||A-A^H||/||A|| = 3e-13); **B is complex-SYMMETRIC**
  (||B-B^T||/||B|| = 2e-13) and **NOT Hermitian** (||B-B^H||/||B|| = 1.55).
- The old `[[A,B],[-B,-A]]` has a **COMPLEX** spectrum (max|Im| = 1.4e-4 Ry) —
  not a physical BSE.  It survived because full-BSE-with-W was never
  value-validated.

The physical operator is the para-Hermitian **SHAO** form
`H = [[A, B], [-B*, -A*]]` (Onida-Reining-Rubio; Rohlfing-Louie) — REAL spectrum,
+-omega pairs (verified: `Sigma_x H* Sigma_x = -H` holds for SHAO, not for the
loose `[[A,B],[-B^H,-A^H]]` when A is complex-Hermitian).

**Fix** (`bse_ring_comm._antiresonant_row`, `screening=False`): the anti-resonant
row is `Y_out = -B* X - A* Y`, computed by reusing the SAME appliers on
conjugated inputs — `B* X = conj(_apply_B(conj X))`, `A* Y = conj(_apply_A(conj
Y))` (operator ingredients unchanged; D real) — no new kernel.  `screening=True`
(the RPA W(0)/W(omega) resolvent path) is byte-UNCHANGED (`[[A,B],[-B,-A]]`, B
Hermitian there; validated by the W(0) closure) — the branch is build-time on
`screening`.

**First validated non-TDA-with-W eigenvalues (Ry):**
`0.007534 0.007623 0.009319 0.009487 0.017128 0.017128`.

### Structure-preserving solver — `bse_nontda.py` (new; procedural, no class)

The design's clean `Omega^2 = eig((A-B)(A+B))` product form (Shao LAA 488;
(A+B)-metric Lanczos) requires (A+-B) Hermitian, i.e. real / Hermitian-B BSE.
The spinor B is complex-symmetric, so `(A-B)(A+B)` reduces the WRONG (code)
operator and is not (A+B)-self-adjoint (1.6e-2).  The correct structure-preserving
object for complex B is the **Hermitian-definite pencil**

    K z = omega Sigma z,  K = [[A,B],[B*,A*]] (Hermitian, PD: min eig 5e-3),
    Sigma = diag(I,-I);  lowest omega = extreme eig of Hermitian K^{-1/2} Sigma K^{-1/2}.

This is BGW's own regime (dense ScaLAPACK `BSE_NTDA_SOLVER_SSEIG`).
`solve_bse_nontda_sharded` dispatches on B: complex-symmetric -> definite pencil;
Hermitian -> product form.  Both solved densely for small windows (BGW-parity);
(A+-B) actions reuse the full matvec verbatim (`make_ab_appliers`:
`matvec([U;+-U])[X-block] = (A+-B)U`); the matrix-free FEAST-on-K / BSEPACK-real-
transform is the fine-grid follow-on (prototype validated).  Eigenvectors:
normalised (X, Y) pairs with **X^H X - Y^H Y = +1**, stacked `(n_eig, 2, nc, nv, nk)`.

- **Positive-definiteness of (A-B)/K checked/asserted** — an indefinite K (triplet/
  charge instability => imaginary excitations) raises a clear message, not hidden.
- **Dispatch**: `solve_bse_sharded(..., tda=False)` -> `bse_nontda`; `bse_jax`
  `_preview_lanczos(tda=...)` (drops the old "TDA only" SystemExit); non-TDA routes
  through the sharded loader even on 1 device.  No parallel solver stack.
- **Writer**: `write_eigenvectors_stream(use_tda=...)` now HONEST (was hardcoded 1);
  non-TDA writes X to `eigenvectors` + Y to `eigenvectors_coupling`.

### Solver P1 — block-Lanczos final-slot overwrite + beta-transpose

The recorded bs=4 "ghosts" are **Krylov-truncation variational bias** (Krylov < N),
NOT spurious duplicates: with full reorth and Krylov >= N, bs=4 recovers the exact
lowest-4 == dense (measured: bs=1 AND bs=4 both `0.00818851 0.00831683 0.01068212
0.0107629`; undersampled Krylov=20 gives the biased `0.00827/0.00838`).  The
**final-slot overwrite** (`Q_all.at[min(j+1, M-1)]` clobbers Q_{M-1} on the last
iter) corrupts eigenVECTORS (the last Krylov block), not eigenvalues.

Fixed in `solvers/lanczos.py`: (1) allocate **M+1 Krylov slots** and write Q_next
to slot j+1 (all 3 jit sites: `lanczos_eig_jit`, `block_lanczos_eig_jit`,
`block_lanczos_eig_jit_converged`), so the final block is retained for the
eigenvector reconstruction; (2) `beta_j = R` (not `R.T`) in the dead
`block_lanczos_eig` (the transpose came out conj(R)/R.T instead of R/R^H, masked
for eigenvalues by (T+T^H)/2 but wrong for the T->Q mapping).  Regression:
`test_block_lanczos_eigenvector_residual_p1` — at Krylov=N the block-Lanczos
eigenvector residual `||Hv-λv||/||v|| < 1e-6`.

### Gates (1 GPU)

- `tests/test_bse_nontda.py` (synthetic, CPU-runnable): product form (real BSE),
  definite pencil (complex A-Herm/B-sym), PD-check raises, P1 eigenvector residual,
  jit-variant shapes — **5 passed**.
- BSE gate set (`test_bse_dense_reference` incl. the 2 new `test_nontda_*` +
  `test_bse_stack_matvec` + `test_bse_w0_resolvent` + `test_bse_w_omega_chain`) =
  **20 passed / 1 deselected** (18 -> 20 = +2 non-TDA gates; TDA dense/stack
  bit-unchanged).  `runs/MoS2/A_bse_nontda_2026-07-17/bse_gates.log`.
- Full plain 1-GPU suite: **230 passed / 12 skipped / 25 deselected (5:08)** — the
  223 baseline + 7 new non-TDA/P1 tests, all four golden GW gates included, no
  regressions (`full_suite.log`; 4 warnings pre-existing: cusolvermp donation,
  WFN symmetry fallback).

Artifacts: `runs/MoS2/A_bse_nontda_2026-07-17/` (explore1-4, prototype_solver,
bse_nontda_draft, test_dense_nontda_standalone, repro_ghost*; module-free runner
reused from `../A_bse_w0_resolvent_2026-07-16/lxrun_free.sh`).

## Full-basis W_q resolvent run (owner-requested, 2026-07-17)

All 399 columns (identity probe block) × all 5 symmetry-reduced q, MoS2 gnppm
fixture, 1 GPU, post-recompile-fix engine. EVERY column of every tile validated
against the restart's (W0−V)[q_flat]:

| iq | q | n_cols | build[s] | solve[s] | max_rel | med_rel | max_resid | n_bad |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | (0,0,0) | 399 | 0.32 | 23.5 | 1.90e-7 | 2.33e-9 | 8.7e-10 | 0 |
| 1 | (0,1,0) | 399 | 0.08 | 26.3 | 4.95e-8 | 2.58e-8 | 6.0e-10 | 0 |
| 2 | (1,0,0) | 399 | 0.02 | 26.4 | 6.23e-8 | 2.78e-8 | 6.6e-10 | 0 |
| 3 | (1,1,0) | 399 | 0.07 | 28.9 | 1.07e-7 | 4.89e-8 | 4.4e-10 | 0 |
| 4 | (1,2,0) | 399 | 0.16 | 26.1 | 4.29e-8 | 2.62e-8 | 6.0e-10 | 0 |

**Grand total 135.7 s** (~26 s/q ≈ 65 ms/column; iq=0 carries the one-time
engine compile). Zero non-converged columns. This is the strongest W validation
to date: the full W_q tiles, every column, at the GW quadrature floor.
Artifacts: runs/MoS2/A_bse_w0_resolvent_2026-07-16/full_basis/ (committed 8ed3cdc0).

## Ridge-regularized ζ-fit A/B — Tikhonov by default? (2026-07-17, agent/bse-phase2-zeta-ridge, lorrax_A_ridge_wt)

Owner question: §12's cleaned-ζ (Tikhonov filter f_ε(λ)=λ/(λ²+ε²), i.e.
solve (C²+ε²I)ζ = CZ) was physically inert on BSE observables and improved
tile covariance — would generating ridge-ζ BY DEFAULT change production GW?

**Verdict: YES — 4–200 meV drifts on Σ at every tested ε. NOT
default-safe.** Shipped opt-in: cohsex.in `zeta_ridge_eps` (default 0.0 =
bit-identical; charge-only; bispinor+ridge rejected at parse — transverse
indefinite-CCT semantics are a flagged follow-up). Commit `5f23631` on
`agent/bse-phase2-zeta-ridge` (isolated worktree off `agent/bse-phase2`
HEAD; the parent branch carried uncommitted BSE work). ε_q = ε_rel ·
λ̂_max(C_q), deterministic power-iteration λ̂ (matches eigh to 4 digits on
both fixtures), no eigh, operator-only change — sharded/FFI solve structure
and all Cholesky backends untouched; RHS premultiply = one batched GEMM.

Arms (1 GPU A100, job 56071522, module-free srun+shifter, worktree
PYTHONPATH): MoS2 gnppm `runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/`
02_stock/03..05_ridge{1e-4,1e-5,1e-6} + 06_stock_repeat; Si COHSEX
`runs/Si/B_zeta_ridge_covariance_2026-07-17/work_{stock,r1em4,r1em5,r1em6,
stock_repeat}` (orbit-closed 792-centroid set). **Determinism floor exactly
0 meV** (stock reruns bit-identical) — all deltas are real ridge effects.
Analysis: `reports/zeta_ridge_ab_2026-07-17/{analyze_ridge_ab.py,
ab_results.json, analysis.log, determinism_check.py}`.

### Physical drift vs stock (meV)

| fixture / observable | ε 1e-4 | ε 1e-5 | ε 1e-6 |
|---|---|---|---|
| Si ΣTOT max (gap window b5–12) | 31.0 | 17.3 | 4.3 |
| Si ΣTOT MAE | 7.0 | 3.8 | 0.9 |
| MoS2 σ_X max (gap window b23–30) | 192 | 130 | 113 |
| MoS2 σ_X MAE (all bands) | 56 | 44 | 27 |
| MoS2 Re σ_C gap max | 481 | 501 | 498 |
| MoS2 eqp0 max (pole-dominated, 714/720 bands PPM-pathological) | 39 eV | 25 eV | 12 eV |

Si scales with ε (÷2–4/decade); MoS2 σ_X does NOT (~0.1–0.2 eV flat): the
2D fixture's Σ_X carries ~0.1 eV of weight in the deepest spectral tail
(λ/λmax < 1e-6) — junk modes are NOT Σ-inert. The §12 RELATIVE class
(~1e-3) transfers, but GW lacks the exciton-level cancellation, so 1e-3 of
a 30–40 eV Σ is 30–200 meV. V0/W0 tiles change 65–100% relF under ridge
(junk-dominated norms — the §12 gauge-artifact picture, reconfirmed).

### Covariance under the centroid permutation (diag2 ladder, max_rel)

| arm | Si G0 | Si V0 | Si W0 | Si ΔW | MoS2 G0 | MoS2 V0 | MoS2 W0 |
|---|---|---|---|---|---|---|---|
| stock | 8.6e-2 | 3.2e-2 | 3.0e-2 | 6.9e-2 | 2.7e-7 | 1.0e-7 | 1.5e-7 |
| 1e-4 | 3.2e-3 | 4.1e-3 | 4.3e-3 | 2.0e-3 | 1.3e-8 | 1.7e-8 | 1.2e-8 |
| 1e-5 | 7.2e-3 | 5.9e-3 | 6.0e-3 | 5.5e-3 | 6.2e-8 | 1.2e-7 | 7.5e-8 |
| 1e-6 | 1.0e-2 | 8.7e-3 | 8.5e-3 | 9.7e-3 | 8.6e-6 | 1.2e-5 | 1.0e-5 |

Prediction CONFIRMED on Si: the ~3% covariance defect → 3–4e-3 at ε=1e-4,
exactly the §12 cleaned-tile floor, monotone in ε. On the already-clean
MoS2, ε=1e-6 DEGRADES covariance 100× via cond(B)≈1/ε_rel²=1e12 roundoff.

### Conditioning seen by the solver (owner mental model)

cond(C): MoS2 1.1e8 (λmax 1.73e-2, λmin 1.5e-10), Si 3.8e7. cond(B) =
min(cond(C)², 1/ε_rel²) = 1e8/1e10/1e12 at ε_rel 1e-4/1e-5/1e-6 — below
ε_rel* = cond(C)^{-1/2} (≈1e-4 here) the ridge makes the solver's operator
WORSE-conditioned than stock. At 20k centroids cond(C) grows and ε_rel*
shrinks, but the physics drift binds first. Best experimental setting:
ε_rel ≈ cond(C)^{-1/2} — best covariance, cond(B) ≈ cond(C).

### Suite status

- Knob OFF: full 1-GPU suite untouched-green (`lorrax_A_ridge_wt/
  suite_off_solo.log`; incl. 8 new `test_zeta_ridge.py` gates: OFF
  bit-identity, eigh-reference filter match, λ̂ gate, padded-extent
  zeros, transverse loud-fail). An earlier concurrent-srun-step run OOMed
  the bispinor session (GPU sharing under --overlap, not code).
- Golden gates knob ON (deltas recorded, references NOT re-frozen;
  `reports/zeta_ridge_ab_2026-07-17/golden_on_work/`): cohsex 2D
  7.46 eV / 0.47 eV / 5.2 meV at ε 1e-4/1e-5/1e-6; si_cohsex_3d 27.7 /
  5.4 / 2.5 meV (frozen atol 1 meV); gnppm 42.6 / 26.6 / 8.6 eV
  (pole-adjacent Σc columns included). ALL gates fail their frozen atol at
  every ε.

### Rank/invertibility audit (read-only)

One full-rank assumption downstream of the tiles: the opt-in
`screening_solver=low_mem` fused Dyson (`w_isdf.py:294`) potrf's the BARE
V tile (`v = X X†`) — numerically-PSD-only V (stock junk tail or
ridge-attenuated) can NaN it; non-default, flag before any ridge use with
low_mem. Everything else is safe by construction: default Dyson LU is on
`A = I − Vχ` (w_isdf.py:265; A→I on V's null space), GN-PPM ratio is
masked (`minimax_screening.py:408` + ppm_invalid_mode), head injection is
scalar, Σ assembly and BSE loaders contract tiles without inversion, the
W(ω) chain solves at shifted z (`w_omega_chain.py:315`).

### Adoption requirements (if ever)

Re-freeze all Tier-1/Tier-2 references; accept degrading the si_cohsex_3d
BGW-parity anchor by 2.5–28 meV (BGW has no matching knob — parity loss is
permanent); transverse-channel design + device-invariance validation;
low_mem potrf guard; and a physical argument that ridge-ζ is more accurate
— the measured benefit is covariance hygiene (~10×), not accuracy.

## Exciton bandstructure pipeline (2026-07-17/18, agent/bse-exciton-bands, lorrax_A worktree)

Production exciton-bandstructure capability: a BSE V_Q-interpolation
backend (`src/bse/vq_interp.py` — the F-scheme + b26p production port) and
a Q-path driver (`src/bse/exciton_bands.py`) that reads the SAME
`K_POINTS crystal_b` block the htransform bandstructure driver consumes,
runs ONE compiled `lax.scan` of per-Q TDA solves over the whole path, and
emits `exciton_bands.dat` + a PNG.  Artifacts + disk logs:
`runs/MoS2/B_exciton_bands_2026-07-17/` (WORKLOG.md = full session record).
Branch `agent/bse-exciton-bands` (base c4c349f); commits 1f16ea2, 196c30b,
0060a52, + final.

### What landed (single-source seams honored)

| piece | where | note |
|---|---|---|
| vq_interp: prepare_coarse (Tik clean + SR/LR split, device P('x','y') tiles, eigh backend auto\|off\|cusolvermp\|slate) + host b26p LSQ + ONE jitted `eval_vq` (all Q-dependent data runtime args) + `refit_vq` ground truth | `src/bse/vq_interp.py` | port of `REFERENCE_arbitrary_q_vq.py`, arithmetic preserved |
| `compute_wfns_fi(q_list=…, return_coeffs=…)` | `bandstructure/bse_setup.py` | the §1 ~40-LOC arbitrary-Q generalization; no parallel function |
| `streaming_galerkin_solve(return_full_proj=…)` | `bandstructure/htransform.py` | full-r α-basis projector W_proj = L⁻¹diag(1/s)U^H (refit consumer) |
| Q-path driver: htransform conduction caches ψ_c(k+Q)/ε_c(k+Q); ONE stack matvec (conduction slots swapped); scan-of-block-Lanczos; .dat+.png | `src/bse/exciton_bands.py` | exchange tile at wrap(−Q); Γ = production q=0 tile; V Hermitized |
| **block-Lanczos Krylov-exhaustion clamp** | `src/solvers/lanczos.py` | see below — pre-existing solver bug, fix benefits every consumer |
| gates | `tests/test_bse_vq_interp.py`, `tests/test_exciton_bands.py` | acceptance vs reference thresholds; sharding assert; census; driver smoke |

### vq_interp port parity (MoS2 3×3 640-centroid fixture, 1 GPU)

The port reproduces the reference e2e baseline TO EVERY PRINTED DIGIT:
LOO B med 1.409e-2 / max 3.553e-2, exciton swap med 0.642 / max 2.542 meV
(reference log values identical); machine gates (recon 2.3e-16,
makeVq-vs-disk 1.3e-9, X^HX-vs-C 6.2e-11) and nulls (exact-stencil
1.7e-15, F-rebuild 6.0e-11) all green; jitted-vs-host evaluator parity
3-5e-16 at on- and off-grid Q; tile sharding P('x','y') asserted; ONE jit
cache entry serves every Q.  (smoke_vq.py log in the run dir.)

### Solver bug found + fixed: sub-spectrum Lanczos ghosts past Krylov exhaustion

With the 4v4c exciton window (n_flat = 144) and the requested Krylov
320 (bs 8 × 40 iter), the fixed-iteration block Lanczos ran past
exhaustion: the residual block collapses, QR of a ~zero block returns
junk directions, and the manufactured α/β blocks put Ritz values
ANYWHERE — measured 60-100 meV BELOW the dense ground state, different
garbage per code path (production solve_bse_sharded, the driver scan, the
htransform-ψ variant all disagreed below 0.179 eV while dense eigh said
the true minimum IS 0.179359 eV).  Fix: clamp max_iter at floor(n/bs) in
`block_lanczos_eig_jit` + the converged variant; driver defaults to FULL
reorthogonalisation (exciton windows are small; partial reorth at
saturation breeds ghosts; cost negligible).  Post-fix: solve_bse_sharded
== 144-dim dense eigh to 0.0000 meV; the driver Γ row == dense
(htransform-ψ operator) exactly.  Every earlier small-window BSE Lanczos
run on this lineage is suspect below its true ground state.

### htransform ψ-source floor (quantified, driver gate)

At Γ the driver row differs from dense(stored-ψ) by 2.25 meV — the
htransform representation floor at 640 centroids / rank 720 (Kramers-
doublet rotations + window-edge mixing; ε exact to 0.000 meV, conduction
subspace min-sval 0.943).  The driver prints Δε and the subspace overlap
at every Γ path point and hard-fails only on gross breakage.

### Per-Q ζ-refit ground truth (--vq-mode both): convention found, floor measured

* Stored-ζ phase convention derived + adopted (pinned by the on-grid
  null): `ZG_μ(G) = e^{−2πi q·s_μ}·FFT_r[ζ̃_μ](G)` — the centroid winding
  phase folded into the stored sphere (the same phase the F-scheme
  factors out).  Omitting it decorates V by e^{iq(s_μ−s_ν)}: 54% tile /
  11% B error.
* Pair-family allegiance measured: the stored ζ fits the TORUS pair
  family (u at wrapped labels, no umklapp phases) better than the
  umklapp/lab family (expansion error 0.112 vs 0.123) — refit convention
  = torus, matching build_cq and the producer kernels' per-element
  decode.
* Remaining on-grid refit-vs-stored gap 2.0-2.9% B (htransform m-leg;
  3.7-5.1% stored-m-leg): NOT a convention error and NOT regularization
  (rcond-insensitive 1e-6→1e-12 at B(vs ridge) 2e-8; cond(C)=1.8e7; all
  discrete conj/q-sign/frame variants refuted).  Measured provenance:
  the refit's full-grid ζ fits the physical pair family at expansion
  error E = 8.5e-4, while the stored (30-Ry sphere-band-limited) ζ sits
  at E = 0.112 — the stored representation's expansion error is
  dominated by its sphere truncation, and the refit-vs-stored V tile
  difference lives in that truncation-sector realization.  The 2-3% B
  systematic is stable across q and CALIBRATES the ground-truth
  comparisons below; at the EXCITON level it collapses (on-grid K spot
  check: ≤0.13 meV across 8 states).

### Compile census + timings (final 32-pt Γ→M→K→Γ, 1 GPU A100, JAX_LOG_COMPILES, cold cache)

| engine | compiles | note |
|---|---:|---|
| `solve_path` (scan of per-Q block-Lanczos) | **1** (26.3 s XLA) | the whole path + refit rows in ONE compile |
| `eval_vq` (arbitrary-Q tile) | **1** | every Q dispatch-only |
| `_q_batch` (htransform ψ(k+Q)) | **1** | batched q-list |
| `_clean_split` (offline prep, per sphere size) | 3 | per distinct ngk; offline-only |
| one-time small eager ops | ~135 | setup/host prep, O(1) in nQ |

Per-Q marginal compiles: **0** — verified again on the both-mode final
run at nQ = 37 rows (32 path + 5 refit rows in the SAME scan/compile:
`solve_path` cold 28.3 s incl. its one compile, warm 26.4 s =
713 ms/Q; temp 514 MiB, args 341 MiB).  Interp-only pipeline on a cold
census cache: solve warm 1.20 s/Q over 32 Q; one-time htransform setup
61 s + vq prep 170 s (offline, per fixture; 5-11 s warm).  Per-refit-Q
cost ~4.7 s (htransform q-list + fit + sphere contraction — the
compute-don't-interpolate mode is ~5× the scan's warm per-Q solve).

### The bandstructure (deliverable)

`runs/MoS2/B_exciton_bands_2026-07-17/exciton_bands_GMKG.{dat,png}` —
MoS2 3×3 COHSEX fixture (the gnppm fixture's ζ storage is IBZ-only, which
the vq trainer cannot consume — recorded; same system, W0_ready, full-BZ
ζ), 4v×4c TDA, lowest 8 states, Γ→M→K→Γ 32 points.  Physical features:
Γ ground state 0.1794 eV (== dense), the finite-Q G=0 exchange branch
visible as the Γ↔Q→0 convention step (documented in the .dat header),
smooth arcs with a finite-momentum minimum on the K→Γ segment
(0.174 eV — below Γ), both Γ endpoints bit-identical.  Interp-vs-refit
(ground truth) ΔE_S at spot-check path points, lowest 8 states (meV):

| iQ | s | where | ΔE₁ | ΔE₂ | ΔE₃ | ΔE₄ | ΔE₅ | ΔE₆ | ΔE₇ | ΔE₈ |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.95 | Γ→M, off-grid | 0.002 | 0.004 | 0.103 | 0.087 | 0.001 | 0.005 | 0.566 | 1.230 |
| 9 | 2.86 | M→K, off-grid | 0.000 | −0.000 | −0.013 | 0.052 | 0.029 | 0.000 | −0.000 | 0.003 |
| 12 | 3.81 | K (ON-grid null) | −0.000 | −0.001 | 0.011 | 0.125 | 0.106 | −0.001 | −0.001 | 0.030 |
| 18 | 6.01 | K→Γ min, off-grid | −0.001 | −0.391 | −0.007 | −1.847 | −0.009 | 0.009 | −0.012 | −0.388 |
| 26 | 8.72 | K→Γ, off-grid | 0.228 | 0.001 | 0.007 | 0.004 | 0.231 | 0.003 | 0.130 | 0.002 |

**The off-grid ground-truth verdict the interpolation program was
missing (§11.4/§12.6 caveat): at the exciton level the b26p/F-scheme
interpolation agrees with the per-Q compute-don't-interpolate refit to
≪0.1 meV median and 1.85 meV worst-state across off-grid Q — the same
scale as the on-grid spot check (≤0.13 meV) plus the refit↔stored
2-3% B systematic documented above.  No off-grid degradation is
observed on this fixture.**

### Deferred / follow-ups

1. IBZ-stored-ζ unfold for vq_interp training (route through the ONE
   SymMaps sym-action; unblocks the gnppm fixture proper).
2. eqp/QP-corrected exciton bands (htransform accepts an EQP override —
   plumb --eqp through the driver).
3. Refit-quality htransform m-leg at large |G_umk| paths beyond the first
   BZ ring (current paths stay inside).
4. 3D-bulk vq_interp (K_z-continuous LR basis — §13.5(4) unchanged).
5. Multi-GPU path runs (driver is sharding-clean by construction;
   validated 1×1 only — no 16-GPU gating).
