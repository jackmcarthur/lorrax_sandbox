# Group: src/mixing/ — fixed-point acceleration (Anderson / CROP family)

Deep-read notes for the GW refactor map, 2026-07-01. Files:
- `src/mixing/acceleration.py` (917 LOC)
- `src/mixing/benchmark_synthetic.py` (317 LOC)
- (context) `src/mixing/__init__.py` (26 LOC) — re-exports `anderson_acceleration, crop, crop_anderson, rcrop, rcrop_anderson`. Notably does NOT export `rcrop_nojit`, the only function production actually uses.

Grep scope used everywhere below: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/{src,tests,tools,scripts}` for the function name and for `from mixing` / `import mixing` / `mixing.acceleration`.

**Headline finding: the ONLY production consumer of the entire package is `mixing.acceleration.rcrop_nojit`, imported at `src/gw/sc_iteration.py:523` and called at `sc_iteration.py:572` inside `_run_rcrop` (the QSGW/SC self-consistency accelerator, selected via env `LORRAX_SC_ACCEL=rcrop`, default). Every other public function is exercised only by the in-package synthetic benchmark, or by nothing at all. There are zero tests: `ls tests | grep -i "mix|crop|anderson|accel"` → no hits.**

---

## src/mixing/acceleration.py

### Purpose
JAX implementations of Anderson acceleration, CROP, rCROP, and CROP-Anderson for fixed-point iterations x_{k+1} = x_k + f(x_k), f(x*)=0, following Wan & Międlar "On the Convergence of CROP-Anderson Acceleration Method" (Algorithms 2.1, 2.2, 3.1). Intended use per module docstring: x ↔ vec(Σ_mnk) or vec(H_qp) in GW self-consistency, f(x) = Σ^out − Σ^in. All complex128, fixed-size circular history buffers, `jax.lax.while_loop` early stopping — except `rcrop_nojit`, a plain-Python clone used when `residual_fn` itself contains jitted pipelines (avoids treating the whole GW pipeline as a traced static callable).

Category: **numerics: SCF/QSGW mixing & acceleration library** (mostly dormant; one live entry point).

### Module-level side effect
Line 29: `os.environ.setdefault("JAX_ENABLE_X64", "1")` *before* importing jax — import-order-sensitive; harmless if gw_jax imported first, but a hidden global.

### Function table

| Function | Lines | Role | Callers (grep evidence) |
|---|---|---|---|
| `AccelerationResult` (NamedTuple) | 41–47 | Result container: x, residual_norms, iterations, converged | all drivers here; `sc_iteration._run_rcrop` consumes `.x`, `.iterations`, `.converged`, `.residual_norms` |
| `_qr_least_squares(F, r, ridge=1e-12)` | 55–84 | min_γ ‖r − Fγ‖₂ via QR + ridge on R; zero-column mask | `_anderson_core` only |
| `_solve_crop_alpha(Fw, filled_cols, m)` | 87–118 | OLD CROP constrained LS, window (n, m+2); docstring: "Deprecated: use _solve_crop_alpha_v2 instead" | `_crop_anderson_core` only (i.e. the deprecated solver is still the one CROP-Anderson uses) |
| `_solve_crop_alpha_v2(Fw, filled_cols)` | 121–170 | min_α ‖F_w α‖₂ s.t. Σα=1, via affine coords γ centered on trial: min ‖f_trial + (F_hist − f_trial)γ‖, α=[γ; 1−Σγ]. Ridge 1e-12, valid-col threshold 1e-14 | `_crop_core`, `_rcrop_hermitian_core`, `rcrop_nojit` |
| `_anderson_core` | 178–253 | jitted while_loop Anderson: min_γ ‖f_k − ΔF γ‖; x_{k+1} = x_k + f_k − (ΔX+ΔF)γ. Circular buffers dX,dF (n,m), roll-to-chronological + mask | `anderson_acceleration` only |
| `anderson_acceleration(residual_fn, x0, m=5, maxit=100, tol=1e-10)` | 256–283 | Public Anderson/Pulay/DIIS wrapper | `benchmark_synthetic.run_benchmark`; `__init__` re-export. **No production/test callers.** |
| `_crop_core(..., use_real_residual)` | 291–392 | jitted CROP/rCROP loop. Window X_w=[hist(m) | trial], F_w likewise (m+1 cols). x_{k+1}=X_w α; f_{k+1}=F_w α (CROP, control residual) or residual_fn(x_new) (rCROP), switched by `jax.lax.cond` | `crop`, `rcrop` |
| `crop(...)` | 395–423 | Public CROP (control residuals) | benchmark only |
| `rcrop(...)` | 426–454 | Public rCROP (real residuals) | benchmark only |
| `_crop_anderson_core(..., use_real_residual)` | 462–574 | jitted CROP-Anderson (Alg 3.1): convergence checked at trial step; returns trial vector on convergence. Window is (n, m+2): [hist(m) | current | trial]. Uses OLD `_solve_crop_alpha` | `crop_anderson`, `rcrop_anderson` |
| `crop_anderson(...)` | 577–605 | Public wrapper | **imported by benchmark and `__init__` but never CALLED anywhere** (benchmark's `run_benchmark` runs only Anderson/CROP/rCROP) |
| `rcrop_anderson(...)` | 608–635 | Public wrapper, real residuals | same: imported, never called |
| `hermitian_to_upper_flat(H)` | 643–658 | (..., n, n) Hermitian → (..., n(n+1)/2) via `triu_indices` | **zero callers anywhere** (grep across src/tests/tools/scripts: only its own definition) |
| `upper_flat_to_hermitian(H_flat, n)` | 661–685 | inverse; lower triangle = conj of upper | **zero callers** |
| `_rcrop_hermitian_core(residual_fn, x0, n_matrix, m, maxit, tol)` | 688–774 | jitted rCROP on upper-triangle-flattened Hermitian vectors. `n_matrix` static arg is **never used in the body** (dead parameter; no reconstruction ever happens). dtype hardwired complex128 | `rcrop_hermitian` only |
| `rcrop_hermitian(...)` | 777–809 | Public wrapper | **zero callers** (grep: only definition + core) |
| `rcrop_nojit(residual_fn, x0, m=5, maxit=100, tol=1e-10, print_fn=None)` | 812–917 | Plain-Python rCROP (no outer jit) "for use when residual_fn contains JIT'd code... avoids XLA constant folding issues". Same algorithm as `_crop_core(use_real_residual=True)` | **`src/gw/sc_iteration.py:523` (import), :572 (call) inside `_run_rcrop` — the ONE live production entry point.** Not exported by `__init__` (imported as `from mixing.acceleration import rcrop_nojit`). |

### Math implemented (for the record)
- Anderson: γ = argmin ‖f^(k) − ΔF γ‖₂; x^(k+1) = x^(k) + f^(k) − (ΔX + ΔF)γ.
- CROP: α = argmin ‖F_w α‖₂ s.t. Σα_i = 1; x^(k+1) = X_w α; control residual f^(k+1) = F_w α (CROP) vs recomputed f(x^(k+1)) (rCROP).
- Circular buffer discipline (all cores): `oldest_pos = (head − filled) % m`; `jnp.roll(buf, −oldest_pos, axis=1)`; mask `(arange(m) < filled)`.
- No einsums; the only contractions are `Xw @ alpha`, `dF_use @ gamma`, `Q.conj().T @ r` style matvecs.

### Production boundary (via sc_iteration._run_rcrop)
- x = `H_qp_dft` flattened: shape (nk·nb·nb,) complex128, **host** numpy → jnp; residual_fn wraps the full `gw_iteration_map` (whole GW pipeline per evaluation, 2 calls per rCROP iteration). Buffers Xhist/Fhist are (nk·nb·nb, m) on default device — for large nk·nb² this is an unsharded device-resident history, worth noting for the refactor.
- tol conversion at sc_iteration: `tol_resid = sqrt(n_elem) · tol_ev / RYD_TO_EV`.
- sc_iteration re-Hermitises inside residual_fn and again on the final x (rCROP linear combinations don't preserve Hermiticity) — which is exactly the job `rcrop_hermitian` + the upper-triangle helpers were written for and never wired up to.

### Flags consumed
- `acceleration.py` itself: only env `JAX_ENABLE_X64` (setdefault at import).
- Upstream selection (in gw_jax.py:536–550, not this file): env `LORRAX_SC_ACCEL` (default "rcrop"), `LORRAX_SC_DEPTH` (history m, default 5), `LORRAX_SC_MAX_ITER`, `LORRAX_SC_TOL_EV`, `LORRAX_SC_MIXING`; `LORRAX_SC_DUMP_DIR` in sc_iteration.py:430. No cohsex.in keys (gw_jax has a `TODO: plumb max_iter / tol_ev through config; env vars for now`).

### I/O
None. Pure in-memory numerics.

### Dead suspects (with grep evidence)
1. `rcrop_hermitian` / `_rcrop_hermitian_core` / `hermitian_to_upper_flat` / `upper_flat_to_hermitian` — grep for each name across src, tests, tools, scripts returns only their definitions in this file. Fully dead (~170 LOC). Ironically their purpose (Hermiticity-preserving rCROP) is re-implemented ad hoc in sc_iteration's residual_fn via full-matrix re-Hermitisation.
2. `crop_anderson` / `rcrop_anderson` / `_crop_anderson_core` / `_solve_crop_alpha` (old) — only "callers" are the `__init__.py` re-export and the benchmark's *import list*; `run_benchmark` never invokes them (only Anderson/CROP/rCROP loops at benchmark lines 148–163). Dead chain ~180 LOC including the explicitly-deprecated solver.
3. `anderson_acceleration`, `crop`, `rcrop` (+ `_anderson_core`, `_crop_core`, `_qr_least_squares`) — callers exist but ONLY in `benchmark_synthetic.py`. Benchmark-only, no production or test usage.
4. Net: of 917 LOC, production reaches `rcrop_nojit` + `_solve_crop_alpha_v2` + `AccelerationResult` ≈ 220 LOC.

### Redundancy suspects
1. **`rcrop_nojit` vs `_crop_core(use_real_residual=True)`** — line-for-line algorithm duplicate (trial step, roll/mask, `_solve_crop_alpha_v2`, update) in a jit and a no-jit variant. Classic parallel old/new path; the jit variant is the unused one.
2. **`_rcrop_hermitian_core` vs `_crop_core`** — near copy-paste with rCROP hardwired, complex128 hardwired, plus a dead `n_matrix` arg. Third copy of the same loop body.
3. **`_solve_crop_alpha` vs `_solve_crop_alpha_v2`** — explicit old/new pair, old marked deprecated in its own docstring but still called by `_crop_anderson_core`. Also duplicates the QR+ridge pattern of `_qr_least_squares` (three QR-solve variants total).

### Weird code
- `_qr_least_squares` (73–83): `valid_mask` computed from column norms, but QR is performed on the **unmasked** F including zero columns; only γ entries are zeroed post-hoc. Comment in `_solve_crop_alpha_v2` line 159 acknowledges: "This includes zero columns, but they won't affect the solution" — true only thanks to the 1e-12 ridge making R nonsingular. Magic constants: ridge 1e-12, zero-column threshold 1e-14.
- `_crop_anderson_core` x_trial_out threading (493, 500, 556, 573): body's returned `x_trial_out` slot is set to *this* iteration's `x_trial`, and on convergence `x_out = x_trial_out`; correct only because `skip_crop` also returns `x_trial` as `x_new`. Fragile double bookkeeping.
- `_rcrop_hermitian_core` (688): static arg `n_matrix` documented as "(to reconstruct Hermitian)" but never referenced in the body.
- `_crop_core` line 327: `head = jnp.int32(1 % m)` — subtle m=1 edge-case handling, comment present but easy to break.
- `rcrop_nojit` line 896: progress printed only for `it < 10` (magic cap); production passes `print_fn=None` anyway (sc_iteration prints its own RMS history), so the parameter is live-dead.
- Line 29: `os.environ.setdefault("JAX_ENABLE_X64", "1")` import side effect (also in benchmark line 21).
- gw_jax.py:535 upstream: `# TODO: plumb max_iter / tol_ev through config; env vars for now` — the whole SC-accel knob surface is env-var-only.

---

## src/mixing/benchmark_synthetic.py

### Purpose
Standalone diagnostic script reproducing Example 1 (Figures 3a/3b) of Wan & Międlar: convergence of Anderson/CROP/rCROP with m ∈ {1,2,4} on 100×100 tridiagonal (1,−4,1) and "seven-diagonal" linear systems Ax=b, b=e₁, fixed-point map g(x)=x+b−Ax. Prints iteration counts and saves semilogy residual plots. Forces CPU + x64.

Category: **diagnostic/bench script** (validation of the acceleration library against the paper).

### Entry points
- `main()` ← `if __name__ == "__main__"` only; docstring says `uv run python -m mixing.benchmark_synthetic`. Grep for `benchmark_synthetic` across the repo (py/md/toml/sh): only its own docstring. **No callers, no test, no skill references.**
- All other functions (`make_tridiag_A` 52–61, `make_sevendiag_A` 63–77, `build_problem` 80–101, `make_residual_fn` 104–115, `run_benchmark` 123–165, `plot_results` 168–235) are internal to `main()`.

### Function table

| Function | Lines | Role |
|---|---|---|
| `make_tridiag_A(n=100)` | 52–61 | diag(−4) ± offdiag(1), complex128; eigenvalues in [−6,−2] so ‖I−(−A)‖-type contraction works via f(x)=b−Ax |
| `make_sevendiag_A(n=100)` | 63–77 | builds diagonals at offsets −3..+3 with values (0,0,1,−4,1,1,1) |
| `build_problem(n, matrix_type)` | 80–101 | returns (A, b=e₁, x0=0) |
| `make_residual_fn(A, b)` | 104–115 | jitted f(x) = b − A@x |
| `run_benchmark(n, matrix_type, maxit, tol, m_values)` | 123–165 | loops Anderson/CROP/rCROP over m_values; prints per-run stats |
| `plot_results(results, title, save_path, show)` | 168–235 | semilogy residual plot; matplotlib; per-method linestyle map |
| `main()` | 238–313 | argparse (`--show`, `--outdir`); runs tridiag then sevendiag; saves `benchmark_tridiag.png`, `benchmark_sevendiag.png` |

### Cross-module deps
`mixing.acceleration` (anderson_acceleration, crop, rcrop actually used; crop_anderson, rcrop_anderson imported unused), jax, matplotlib.

### Flags consumed
CLI: `--show`, `--outdir`. Env set at import: `JAX_ENABLE_X64=1`, `JAX_PLATFORM_NAME=cpu`. No LorraxConfig / cohsex.in.

### I/O
Writes `<outdir>/benchmark_tridiag.png` and `<outdir>/benchmark_sevendiag.png` (matplotlib PNG, dpi=150) when `--outdir` given. Reads nothing.

### Dead suspects
- Entire file: no callers found (grep `benchmark_synthetic` repo-wide → only its own docstring). Keep-or-drop is a policy call for a paper-reproduction bench.
- Imports `crop_anderson`, `rcrop_anderson` (lines 41, 43) — never called; `plot_results` has a `"CROP-Anderson"` linestyle/marker branch (186, 193–195) that can never trigger.

### Weird code
- `make_sevendiag_A` (69): three of the "seven" diagonals are 0.0, so the matrix is actually 5-diagonal AND nonsymmetric (lower side has only offset −1 = 1; upper side has +1,+2,+3 = 1). Values copied verbatim from the paper's stencil "(0, 0, 1, −4, 1, 1, 1)"; the `enumerate` variable `offset` (71) is unused (uses `diag_idx − 3` instead). Hypothesis: literal transcription of the paper, intentional but visually alarming.
- `--show` backend probing loop (25–32) mutates matplotlib backend by trial over TkAgg/Qt5Agg/GTK3Agg before pyplot import — sys.argv sniffing before argparse.
- Docstring says it creates "Figure 3a" but main() runs both 3a and 3b.

---

## Refactor-map summary for this group

- Single live edge: `sc_iteration._run_rcrop → rcrop_nojit → _solve_crop_alpha_v2`. Everything else is benchmark-only or fully dead (~700 of 1234 LOC).
- Three copies of the CROP loop body and three QR-solve variants; consolidation target is one rCROP implementation (the no-jit one, since the residual is the whole GW pipeline) + one constrained LS solver.
- The dead Hermitian upper-triangle path duplicates (and would halve the memory of) the ad-hoc re-Hermitisation sc_iteration does today — either wire it up or delete it; don't keep both.
- Knob surface is env-vars only (`LORRAX_SC_*`); gw_jax TODO to plumb through cohsex.in/LorraxConfig.
