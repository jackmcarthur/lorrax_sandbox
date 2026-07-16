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
