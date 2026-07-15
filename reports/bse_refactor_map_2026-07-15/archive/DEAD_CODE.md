# LORRAX BSE — dead code, suspected bugs, refactor targets

From 207 adversarial verdicts (`_raw_verdicts.json`), each a reader claim against
`src/bse/` + the shared `src/solvers/` family that a second, independent high-effort
verifier tried to disprove (enumerate all call mechanisms before "dead"; explicit
index/einsum math, numerical reproduction, or cross-check against BGW source before
"bug"). Verdict tally:

| verdict | n | meaning |
|---|---|---|
| confirmed-bug | 88 | **wrong output or crash, reachable** |
| not-a-bug-convention | 36 | leave alone (documented BGW-compat or spinor convention) |
| benign-cruft | 30 | harmless; opportunistic cleanup |
| refactor-target | 23 | consolidation targets, no wrong output |
| confirmed-dead | 20 | delete candidates |
| test-only | 8 | keep, isolate from prod path |
| unverifiable | 1 | both sides of the comparison are dead code; can't adjudicate |
| alive | 1 | reader was wrong; not dead, just permanently inert |

**Read order: §1 first**, specifically the exchange-kernel item — it is the single
highest-severity finding and touches every matvec implementation in the package.
Then the crash-class bugs (§1.2), which block whole CLI entry points outright. Then
the quieter correctness bugs (§1.3) and padding/shape footguns (§1.4). §2 is the
delete pass. §3 is test-only surface. §4 is consolidation debt. A short §5 records
conventions that must **not** be "fixed" by someone acting on this file without
re-reading the reasoning column in `_raw_verdicts.json`.

Per-file audit notes (evidence trails, full claim text) live in
`reports/bse_refactor_map_2026-07-15/archive/files/*.md`, one per slug.

---

## 1. Bugs (88 confirmed) — triage before moving code

### 1.1 The one that matters most: exchange kernel drops Σ_k′ in every implementation

**`apply_V` / `apply_V_ring` / the V-term in `bse_serial`, `bse_simple`, `bse_jax`
— the q=0 BSE exchange kernel is computed k-diagonal, not dense in (k,k′).**

Every live matvec implementation in the package computes the same wrong contraction.
Per element, all four give:

```
VX[b,c,v,k] = (1/Nk) Σ_μν  M*_cvk(μ) · V_q0(μ,ν) · Σ_{c′v′} M_c′v′k(ν) X[b,c′,v′,k]
```

— `k` is retained as a batch index end-to-end (present in both operands *and* the
output of every einsum, e.g. `'kcvN,bcvk->bNk'`), so the exchange term only ever
couples pairs at the **same** k. The physically correct BSE exchange (Rohlfing-Louie;
BGW's `bsex`/`mtxel_kernel`, which loops over **every** (ik,ikp) pair) is dense:

```
⟨cvk|K^x|c′v′k′⟩ = (1/Nk) Σ_μν  M*_cvk(μ) · V_q0(μ,ν) · M_c′v′k′(ν)      — no δ_kk′
```

Locations, all independently confirmed:
- `src/bse/bse_serial.py:62-64`
- `src/bse/bse_simple.py:101-131`
- `src/bse/bse_ring_comm.py:262-300` (`apply_V_ring`, the production ring path used
  by `build_bse_ring_matvec`, reached from `bse_feast.py`, `bse_lanczos.py`,
  `absorption_haydock.py`)
- `src/bse/bse_jax.py:108-121` (dead code, see §2, but implements the identical bug)

Three separate audits confirmed this independently and by different methods:
one by hand-deriving the Henneke-2020 reference formula from `src/bse/context/`
(Eq. 4-5 has an explicit outer `Σ_k′`), one by numerically reproducing
`apply_bse_hamiltonian_single_device` on a 2-k toy system and diffing against a
hand-built dense `Σ_k,k′` sum (bit-for-bit match to the buggy k-diagonal form,
substantial disagreement with the correct dense sum), and one by cross-checking
BerkeleyGW's `kernel_main.f90`/`mtxel_kernel.f90` directly (`do ik=1,qg%nf` calls
`mtxel_kernel` for every (ik,ikp) pair — BGW's exchange is dense, LORRAX's isn't).

**Why it passed the one existing validation gate**: the Si 4×4×4 SOC comparison
(STATUS.md, ~3 meV agreement) is exchange-insensitive at that scale — Si's lowest
exciton manifold is triplet-dominated and the G=0 exchange element is weak by
orthogonality. `--ring-check`/`ring_matvec_correctness_check`
(`bse_ring_comm.py:853-996`) cross-checks the ring matvec against the serial
matvec, which implements the *same* wrong formula — it structurally cannot catch
this. No independent dense-kernel reference exists anywhere in the tree.

**Failure scenario**: any exchange-dominated or 2D system (MoS2/CrI3 excitons,
singlet-triplet splitting, LO-TO/depolarization physics) gets a silently wrong
singlet spectrum computed with only the k-diagonal fraction of the true exchange.
This is a physics correctness bug in the validated production matvec, not a dead
path — it needs a dedicated gate (dense-build `M^†VM` on a 2-k toy, or a BGW
`spin_triplet` A/B diff) before any exchange-sensitive result from this package
should be trusted.

Fix direction: `S(ν) = Σ_k Σ_{c′,v′} M[k,c′,v′,ν] X[b,c′,v′,k]` (drop the k output
axis on the encode step, i.e. `'kcvN,bcvk->bN'`), then re-broadcast the same
k-independent `U(μ)` back to every k on decode.

Related, separate crash bug in the *non-TDA* (B-block) W-encode path — see §1.2.

### 1.2 Crash-class bugs — block a whole CLI entry point outright

These are reachable `TypeError`/`ImportError`/`KeyError`/`ValueError`/
`IndentationError`s, not subtle wrong-number bugs. Several trace back to the
same event: commit `a0da0a5` deleted a batch of BSE modules as "dead/abandoned",
and `fe5e3e8` restored them the same day without reconciling against the
`bse_ring_comm`/`bse_io` API that had moved on in between. That pattern — restore
without validation — explains most of this list.

- **`build_bse_ring_matvec`/`build_bse_ring_matvec_full` dropped the `v_couples_k`
  parameter; three callers still pass it.** `bse_kpm.py:120,128,137` and
  `bse_pseudopoles.py:231,239,248` compute `v_couples_k=bool(not include_W)` and
  pass it as a kwarg the builder signature no longer accepts — `TypeError` before
  any compute. This breaks **four** entry paths: `python -m bse.bse_kpm`,
  `bse_jax.py --kpm-dos` (delegates to `bse_kpm.main`), `bse_feast.py
  --windows-kpm`/`bse_pseudopoles.py --windows-kpm` (call `run_kpm_dos`), and
  `python -m bse.bse_pseudopoles` directly. The parameter was real at an ancestor
  revision (`81ca040`, forwarded to `apply_V_ring(couple_k=...)`, summing `S_total`
  over k — i.e. this WAS the fix for §1.1's missing Σ_k, at least for RPA) and was
  silently dropped when the module moved from `src/isdf/bse_isdf/` to `src/bse/`.
- **`bse_pseudopoles.py:31-32` `ImportError` at module import time** —
  `build_density_drive_operators`/`build_density_readout_operator_full` are
  imported from `.bse_ring_comm` but defined nowhere at HEAD. Even `--help` fails.
  Same restore-without-validation pattern; `a0da0a5`'s own commit message called
  this file "broken on main (import error)".
- **`bse_kpm.main`/`bse_pseudopoles.main` pass `use_nohead=args.nohead` to
  `load_bse_data_from_restart_sharded`, which has no such parameter** —
  `TypeError` on every invocation, `--nohead` or not (`bse_kpm.py:370`,
  `bse_pseudopoles.py:599`). `--nohead` itself is lost wiring for a headless-V/W
  A/B debugging knob (historically real at `81ca040`); the head-injection
  machinery it was meant to gate (`apply_q0_head_rank1_sharded`) still exists.
- **`bse_w_exact.main` crashes with `KeyError: 'W_R'` at the first GMRES solve in
  EVERY mode, including `--rpa`.** `_apply_shifted_matvec` unconditionally indexes
  `data['W_R']`, which no code path in `bse_w_exact.py` ever sets (every other
  FEAST/KPM/pseudopole consumer sets it post-load; `bse_w_exact.py` never does).
  One-line fix: `data['W_R'] = jnp.fft.ifftn(data['W_q'], axes=(2,3,4),
  norm='ortho')` before the first solve.
- **`_ring_sum_valence_second`/`_encode_T_B_gather` (non-TDA B-block W encode)
  crash at trace time** — `einsum("kvsM,bvksN->kvsM->bMNtsk", ...)` has an output
  subscript `t` present in NO input (`bse_ring_comm.py:183` and `:551`; reproduced
  directly: `numpy.einsum` raises `ValueError: ... subscript 't' which never
  appeared in an input`). Intended string is `"kvtM,bvksN->bMNtsk"`. Every
  `build_bse_ring_matvec_full(include_W=True)` matvec crashes — and non-TDA+W is
  the **documented default** for `bse_feast.py`, `bse_kpm.py`, `bse_w_exact.py`,
  `bse_pseudopoles.py` (all default `--tda` off, `include_W` on). Full-BSE-with-W
  has therefore never actually run in this codebase.
- **Restart-format drift — three sequential bugs, together meaning multi-GPU/
  single-device BSE only runs against legacy-era restart files at HEAD:**
  1. `_read_vq0_sharded` (`bse_io.py:233-234,250`) is hardwired to the legacy
     8-D `(1,npol,npol,nkx,nky,nkz,μ,ν)` layout (`dset.shape[6]`, 6 leading
     scalar indices) — current `gw_jax` writes flat-q 3-D `V_qmunu (nq,μ,μ)`.
     `IndexError` on every current-format restart. The sibling `_read_wq_sharded`
     got a layout shim; this reader never did.
  2. Once (1) is patched, `_read_wq_sharded`'s flat-q branch (`bse_io.py:309-317`)
     requires `dset.attrs['kgrid']`, but the writer stores `kgrid` as a top-level
     **dataset** (`f['kgrid']`, via `write_attr`'s "rank-0-only dataset"
     mechanism), never as an HDF5 attribute on the V/W dataset. `ValueError` —
     the outer loader already resolves `kgrid` correctly two frames up but never
     threads it into this call.
  3. `_load_ring_subset`'s q=0 head injection (`bse_io.py:828`,
     `apply_q0_head_rank1`) runs on the **raw** on-disk array *before* the
     8D/6D/3D layout shim, and indexes `V_qmunu.at[..., 0, 0, 0, :, :]` — 5
     explicit trailing indices, correct only for legacy 8-D/6-D layouts.
     `IndexError` on any current-format restart carrying `G0_mu_nu` +
     `vhead`/`whead`, i.e. every ordinary `do_screened` GW run.
- **`feast_ellipse_mixed_sweep.py` does not compile** — `IndentationError` at
  line 113 (`python3 -m py_compile` fails); a multi-line `print` is mis-indented
  relative to its enclosing `for` loop. Any import dies before argparse runs.
- **`feast_zolo_sweep.run_sweep`/`feast_ellipse_mixed_sweep.run_sweep` call
  `_get_feast_runner` with 7 positional args; the signature requires an 8th
  (`dtype`, no default)** — `TypeError` on the first sweep config in both files.
  Stale API drift from the `--gmres-fp32` refactor.
- **`feast_sweep.run_sweep` calls `run_feast_ritz` with no `quadrature=`/
  `lambda_min_eV`/`lambda_max_eV`; the function default is now
  `quadrature='zolotarev'`, which raises `ValueError('...requires spectral
  bounds')` immediately** — swallowed by a blanket `except Exception: print
  FAILED; continue`, so the whole sweep silently produces zero results. Fixing
  that alone exposes a second latent bug: `feast_sweep.py:208` reads
  `rr_data.n_physical`, a field `RitzResult` no longer has (removed when subspace
  truncation was replaced by overlap regularization) — `AttributeError`.
- **`test_bse.load_bse_data_from_restart`** (`test_bse.py:63-85`) reads the
  pre-format-v2 restart schema (`psi_l`/`psi_r`/`enk_l`/`enk_r`, 8-D `V_qmunu`);
  the canonical writer emits only `V_qmunu`(flat-q)/`psi_full_y`/`enk_full`.
  `KeyError: 'psi_l'` on every `isdf_tensors_*.h5` produced at HEAD. `python -m
  bse.test_bse` is a real, documented CLI entry point (`context/README.md`), not
  orphaned — it is simply broken against the current restart format.

### 1.3 Quieter correctness bugs — wrong numbers, not crashes

- **`lanczos_eig_jit`/`block_lanczos_eig_jit`/`block_lanczos_eig_jit_converged`
  overwrite the final Krylov basis slot.** `Q` is allocated with exactly
  `max_iter` columns; `q_{j+1}` is stored at `min(j+1, max_iter-1)`, so the last
  iteration clobbers the real `q_{max_iter-1}` with an extra, unused `q_max`.
  Eigenvalues (built from `T` alone) are unaffected; eigenvectors get a
  `sqrt(2)*|vecs_T[last,i]|`-sized error in an off-basis direction — worst for
  exactly the *unconverged* Ritz pairs the sandbox cares about for per-state
  oscillator strengths. Confirmed by numpy reimplementation + Rayleigh-residual
  comparison against a hand-fixed `Q`. (Reachable in production; the non-jit
  twins `simple_lanczos_eig`/`block_lanczos_eig` don't have this bug.)
- **`block_lanczos_eig` (non-jit) stores the block off-diagonal transposed** —
  `beta_j = R.T` where the Galerkin recurrence needs `R`. Confirmed by
  numpy reimplementation: buggy-code eigenvalues visibly differ from the true
  Galerkin projection on the same Krylov basis; substituting `R` (undoing the
  transpose) reproduces the true eigenvalues bit-identically. Wrong for
  `block_size>=2`. **Currently dead** (`use_block=True` has no caller/CLI
  plumbing — see §2), so no live-path impact until someone resurrects it; the
  jit twin (`beta_j = R` directly) is correct.
- **Zero-padded band slots are exact spurious eigenvectors, below the true
  spectrum, in both Lanczos and Davidson.** `_pad_axis_to_multiple` zero-pads
  ψ *and* ε to mesh multiples; a padded conduction slot has ψ_c=0 ⇒ the V and W
  contractions vanish identically on that slot ⇒ it is an *exact* eigenvector
  with eigenvalue `-eps_v[k,v]` (below the true gap under the QE
  positive-energy-reference convention). No masking exists anywhere in
  `solve_bse_sharded` or the matvecs. Only fires when the mesh doesn't evenly
  divide the requested band counts (all validated runs used divisible counts,
  e.g. 8/8 on a 2×2 mesh) — latent, not currently triggered, but a real trap for
  the next non-divisible run. The Davidson variant is asymmetric: a padded-*c*/
  real-*v* slot is excluded by the `flat > 1e-12` init filter, but a
  real-*c*/padded-*v* slot (positive ΔE) is NOT excluded and can surface among
  the lowest `n_eig` states.
- **FEAST random start vectors are dense over the same zero-padded band slots**
  (`bse_feast.py:558` etc.) — a padded-valence channel is an exact eigenchannel
  at the bare conduction energy, inside typical FEAST bulk windows; with no pad
  mask, non-divisible-mesh runs get spurious in-window Ritz values reported with
  zero residual.
- **`_galerkin_ritz`/`_galerkin_ritz_cj`/`_galerkin_ritz_dav` (pseudobands Ritz
  rotation, `solvers/pseudobands.py:241`, `pseudobands_v2.py:330,373`) use the
  ROWS of the `eigh` eigenvector matrix instead of the columns** —
  `jnp.einsum('kl,kd->ld', S.T, Q)` should be `jnp.einsum('kl,kd->ld', S, Q)`.
  `eigh` returns eigenvectors in columns; the code's `S.T`-based contraction
  uses `S[l,:]` (a row) as the coefficient vector for state `l`, which is not an
  eigenvector of a generic (non-symmetric-by-coincidence) `H_proj`. Confirmed
  numerically on a 5×5 random complex-Hermitian test matrix: `H @ S[l,:] ≠
  theta[l]*S[l,:]` (nonzero residual), while the column-based contraction is
  exact. Output stays an orthonormal basis of the right subspace, just
  mis-paired with its claimed energy — bounded error, consistent with existing
  gates passing, but every CJ-window and Davidson-overflow-window
  energy↔vector pairing is scrambled within the window.
- **`ritz_pseudobands_v2` Davidson-window (`0 < n_in <= k`) sort-pairing bug**
  (`pseudobands_v2.py:643-646,351-353,699-705`) — Ritz energies are padded with
  the window *mean*, Gauss nodes with the window *midpoint*; the two arrays are
  then independently argsorted and zipped. Whenever a real eigenvalue sits on
  the far side of the midpoint from its mean-based rank, the sort orders
  disagree and the real eigenstate gets paired with a zero weight while a
  zero-padded row gets weight 1 — silent loss of one state's spectral weight,
  worked example included in the audit (k=2, n_in=1, E=9.0 → real state
  deleted). Reachable in production: partially-filled windows are common at
  spectrum edges.
- **`_sternheimer_solve_jvp` (`solvers/sternheimer_solve.py:376`) has the wrong
  sign on the b-tangent channel of the custom JVP rule.** The primitive solves
  `A·x = -b`; implicit differentiation requires `A·ẋ = -ḃ - Ȧx`, but the rule
  computes `rhs_tangent = -(ḃ - Ȧx)`, flipping the sign on `ḃ`. Confirmed
  numerically on a degenerate diagonal operator: `jax.jvp` via the custom rule
  disagrees with finite-difference ground truth by O(1), while flipping the sign
  by hand matches FD to 1e-10. Every in-repo JVP-exercising test has `ḃ≡0` (the
  op-channel `Ȧ` is correct and is all that gets tested), so this survived
  undetected — fires on the documented unfrozen-source Sternheimer path, used
  today by a sandbox S-tensor benchmark script whose output is therefore
  suspect. Sternheimer/orbital-mag program, not the core BSE physics path, but
  lives in the shared `solvers/` family.
- **`bse_preconditioner.compute_w_diagonal` contracts `rho_v` on the wrong
  index** (`bse_preconditioner.py:99`): `einsum("kcm,mn,kvm->cvk", rho_c, W_q0,
  rho_v)` sums `rho_c` and `rho_v` on the SAME axis `m` and leaves `W_q0`'s
  second axis `n` unweighted-summed — O(1) wrong relative to the physical
  `Σ_μν rho_c(μ) W(μ,ν) rho_v(ν)`. The correct, sharded version exists right
  next door in `bse_feast.py:97-102`. **Currently dead code** (§2) — no live
  impact today, but any resurrection of this preconditioner path inherits the
  bug.
- **`davidson_absorption` slices the dipole matrix with the PADDED band counts
  (`nc_pad`/`nv_pad`) instead of the real `n_val`/`n_cond`**
  (`davidson_absorption.py:200-202`) — the eigenvector valence axis was sliced
  with the real `n_val` and zero-padded at the *end*, but the dipole slice uses
  `nv_pad` directly with no padding step, offsetting every oscillator-strength
  pairing by `(nv_pad - n_val)` valence bands whenever the mesh doesn't evenly
  divide `n_val`. The correct slice-then-pad helper
  (`build_dipole_vector_bse`) exists and is used correctly by
  `absorption_haydock.py` but not here. Eigenvalues unaffected.
- **`davidson_absorption` hardcodes `n_spinor=2` in its `.dat` header writer**
  (`davidson_absorption.py:210`) regardless of the actual WFN — `eigvals_to_eps2`
  divides eps2 by `n_spinor` from that header, so any genuine `nspinor=1`
  (non-SOC) system run through this path gets eps2 exactly 2× too small with no
  warning.
- **`run_haydock` writes output files from every MPI rank, no `process_index()`
  guard** (`absorption_haydock.py:265-295`) — under the documented multi-process
  launch, every rank concurrently opens the same `.dat`/`.h5` outputs in write
  mode. HDF5 lock/truncation race. Sibling route `davidson_absorption.py`
  already guards with `rank0`.
- **`slice_dipole_to_bse_window`/`build_dipole_vector_bse` silently zero-fill
  missing conduction rows** (`absorption_common.py:97-101,114-115`) when
  `dipole.h5` has fewer bands than requested — no shape check, no warning; the
  Haydock route completes with spectral weight silently missing from the top
  conduction bands, while the eigvec route instead crashes on shape mismatch for
  the identical input defect (inconsistent failure modes for one root cause).
- **`run_haydock`'s `--eqp` branch re-slices `enk_full` with the unclamped CLI
  `n_val`/`n_cond`**, ignoring the loader's own clamped copies
  (`absorption_haydock.py:180-183`); requesting more bands than available with
  `--eqp` raises `IndexError` after the loader already printed "using N".
- **Anti-resonant pole convention split between `bse_pseudopoles` (producer)
  and `pseudopoles_sweep`'s two reconstruction paths**: the producer and one
  sweep path double poles with `-evals_b` (no conjugation); the other sweep path
  uses `-omega.conj()`. Re-derived the Casida pairing algebraically: `-conj(Ω)`
  is the exact anti-resonant partner; `-Ω` only coincides when `Ω` happens to be
  real. Reachable only on legacy final-format pole files today (the current
  producer always also writes `H_w`/`C_w`, routing fresh files to the
  unaffected "intermediates" path), but a real internal inconsistency.
- **`pseudopoles_sweep._reconstruct_from_final` double-counts anti-resonant
  poles for non-TDA files** (stored `d_bright` is already doubled by the
  producer; the sweep re-doubles on top) — same legacy-file-only reachability
  caveat as above.
- **`pseudopoles_sweep._reconstruct_from_intermediates`'s empty-poles return
  path has the transposed shape** vs. its own normal-path return
  (`(n_mu,len(cols))` vs `(len(cols),n_mu)`) — `numpy` broadcast `ValueError`
  in the caller's error-metric norm whenever any sweep config yields zero
  in-window poles.
- **`feast_sweep.build_full_bse_matrix` silently discards the imaginary part of
  a complex Hamiltonian and symmetrizes with `H.T` instead of `H.conj().T`**
  (`feast_sweep.py:93,107,110`) — exact only under the real-TDA convention that
  holds for real-gauge Si; a genuinely complex spinor/SOC TDA Hamiltonian would
  get a silently wrong reference spectrum with no error raised, miscalibrating
  the whole sweep's error columns.
- **`run_kpm_dos`'s window-partition floor clamps at the non-interacting
  diagonal gap `e_min_ry` (`eps_c_min - eps_v_max`), not the true BSE lowest
  eigenvalue** (`bse_kpm.py:248-249`) — any bound exciton (binding energy pushes
  the true lowest eigenvalue below the diagonal gap) falls below every
  KPM-weighted FEAST/pseudopole window, so those solvers never target the
  physically most important excitonic states. The DOS grid itself extends
  lower; only the partition floor is wrong.
- **`geometric_windows` (`solvers/dos.py:344-358`) hangs (or infinite-loops
  downward) if `E_cross <= 0`** — `boundaries=[E_cross]; while
  boundaries[-1]<E_max: boundaries.append(boundaries[-1]*(1+F))` never
  advances from `0`, or diverges away from `E_max` if negative. Latent: the one
  current caller always passes a positive `eps_cross`, but the function is a
  public re-export with no guard.
- **`_GMRES_SOLVER_CACHE`/`_FEAST_RUNNER_CACHE` (`bse_feast.py:33-34` etc.)
  key on `(max_iter, tol, dtype, ...)` but the cached jitted closures capture
  `matvec`/`data` as Python constants** — a second call in the same process with
  an equal cache key but genuinely different physics (RPA vs BSE `include_W`,
  or a different dataset of matching shape) silently reuses the FIRST call's
  matvec and arrays. Dormant today (no in-repo driver varies kernel/dataset
  within one process) but a live landmine for any future cross-kernel sweep.
- **FEAST's TDA conjugate-symmetry accumulation `filt_i + 2·Re(w·y)` and the
  forced-real Ritz-vector projection assume an elementwise-real `H`**
  (`bse_feast.py:250-252,452-453`) — correct for real-gauge Si, silently wrong
  (drops the imaginary part of the filtered subspace) for a genuinely complex
  Hermitian TDA `H` (SOC spinor ψ, general k-gauge). The default non-TDA path
  (explicit conjugate contour nodes) is unaffected.
- **`bse_feast.main`'s window dispatch makes `--window1`/`--window2` silently
  unreachable whenever `--windows-kpm` is set** (`bse_feast.py:1199,1224`), and
  `bse_jax.py` appends `--windows-kpm` to the FEAST delegation argv
  **unconditionally** while also conditionally forwarding
  `--feast-window1`/`2` — so `bse_jax --feast-window1 3 4` silently solves
  KPM-derived windows instead of the literal window the user asked for.
- **`--eqp`/`--n-occ` are accepted by `bse_jax`'s top-level parser but dropped
  from both the FEAST and KPM delegation argv lists**, and neither
  `bse_feast.py` nor `bse_kpm.py` defines those flags at all — the *default*
  (no `--lanczos`) invocation of `bse_jax` with `--eqp` silently computes
  FEAST/KPM results on uncorrected DFT energies. `bse_jax.py`'s own
  `parse_known_args()` (unique in the package; every other BSE CLI uses strict
  `parse_args`) compounds this by silently swallowing typo'd flags.
- **`_pad_axis_to_multiple` returns the PRE-pad extent, not the padded one**
  (`bse_io.py:139-146`; `size = x.shape[axis]` captured before `jnp.pad`,
  returned unchanged in both branches) — callers bind it to
  `n_val_pad`/`n_cond_pad` and size Lanczos/FEAST trial vectors from it, so
  whenever a pad actually occurs (`n_val`/`n_cond` not a multiple of the mesh),
  the trial-vector shape is smaller than the true padded ψ/ε tensors → einsum
  shape mismatch at the first matvec. All validated runs used mesh-divisible
  band counts (pad==0, where the bug is invisible).
- **`_preview_lanczos`/`run_haydock`'s `--eqp` branches re-slice `enk_full`
  with the unclamped CLI `n_val`/`n_cond`** (`bse_jax.py:254-261`,
  `absorption_haydock.py:180-183`) rather than the loader's clamped copies —
  over-requesting bands with `--eqp` set either wraps to the wrong (high-lying)
  bands via negative fancy-indexing or raises `IndexError`, depending on the
  exact overshoot.
- **`write_eigenvectors_stream`'s dataset shape is sized from the caller's raw
  `n_val`/`n_cond`, while the eigenvectors it writes carry the solver's clamped/
  padded shape** (`bse_io.py:81-103`, caller `bse_jax.py:333-345`) — h5py
  shape-mismatch crash (loud, not silent) whenever the loader clamped
  (requested more bands than available) or padded (mesh non-divisibility).
- **`write_eigenvectors_h5` (the second, non-canonical eigenvectors.h5 writer)
  claims BGW format but writes energies in Ry (no eV conversion) and never
  flips the valence axis** (`write_eigenvectors.py:96,141`) — a BGW-convention
  consumer applying the documented flip/eV-scale to this file's output would
  get every state's valence character mirrored and energies 13.6× too small.
  Reachable via `python -m bse.write_eigenvectors` and `test_bse.py --write-
  eigenvectors`; redundant with, and inconsistent with, the compliant
  `bse_io.write_eigenvectors_stream`.
- **`absorption_haydock`'s `--n-spinor` default of `2`** feeds directly into
  the eps2 prefactor `16π²/(V·Nk·n_spin·n_spinor)`; a scalar-relativistic
  (`nspinor=1`) system run without an explicit `--n-spinor 1` gets eps2 exactly
  halved. One independent audit judged this correct-as-shipped (the only
  documented/validated cookbook is the SOC path, where `n_spinor=2` is right);
  a second, later audit flagged it as a live footgun for any future non-SOC
  invocation, since `dipole.h5` carries no `nspinor` attribute the CLI could
  infer from (unlike the sibling `absorption_eigvecs.py` route, which infers it
  from `eigenvectors.h5`'s `spin_kernel`). Both readings are defensible —
  flagged here rather than silently "fixed" in either direction.

### 1.4 Test/doc-audit findings escalated to bug severity

- **`src/bse/test_bse.py` + `test_davidson_bse.py` have zero pytest coverage and
  zero asserts** — filed originally as "test-only" but re-escalated because the
  *absence itself* is the mechanism by which several of the crash-class bugs
  above (the `v_couples_k` TypeError chain especially) shipped and stayed
  undetected. The entire pytest-collected BSE surface at HEAD is one
  `bse_io.read_bgw_eqp` round-trip test.
- **`test_davidson_bse.py`'s final density-comparison slice uses the raw CLI
  `n_val`/`n_cond` instead of the loader's capped values**, contradicting an
  explicit warning comment three lines above it in the same file — a real
  correctness footgun in the tool whose entire purpose is BGW-vs-LORRAX
  numerical validation, not merely stylistic cruft.

---

## 2. Confirmed-dead (20) — the delete pass

Zero live callers after each verifier checked direct imports, `__init__`
re-exports (`src/bse/__init__.py` is a bare docstring — nothing is re-exported),
`python -m` / argparse dispatch, tests, tools, scripts, docs, and the sandbox-wide
`runs/**/*.sh` + `skills/**/*.md`.

| Symbol | Location | Note |
|---|---|---|
| `apply_bse_hamiltonian`, `apply_D`, `apply_V`, `apply_W` (bse_jax-local) | `bse_jax.py:67-160` | Non-executable as written even if called (unbound `lax.psum` axis names outside `shard_map`, tracer ints hit `.reshape`). Implements the exact same k-diagonal exchange bug as §1.1 — do not resurrect without fixing that first. |
| `bse_serial.symmetrize_W_q` | `bse_serial.py:12-24` | Correct math (W(q)=W(-q)† enforcement); orphaned. H hermiticity currently relies entirely on GW-side W being symmetric with no in-package check. |
| `bse_serial.apply_bse_hamiltonian_single_device_jit` | `bse_serial.py:83-99` | Consumers jit the plain function themselves with their own static-shape args. |
| `bse_ring_comm.apply_bse_hamiltonian_ring` + `apply_W_ring` | `bse_ring_comm.py:303-337,191-225` | Monolithic single-`shard_map` predecessor of the builder pipeline; ~80 LOC. |
| `timed=True` branches of both ring-matvec builders | `bse_ring_comm.py:452-468,672-700` | No caller passes `timed=True`; the `common.timing`-instrumented un-jitted variant is unreachable. |
| `bse_io.BSEData` | `bse_io.py:18-20` | Empty `SimpleNamespace` subclass, never instantiated. |
| `solve_bse(use_block=True)` path + `solvers.lanczos.block_lanczos_eig` (non-jit) | `bse_lanczos.py:82-85`, `solvers/lanczos.py:28-135` | No CLI plumbing for `use_block`; carries the transposed-beta bug from §1.3. |
| `bse_preconditioner.py` — everything except `energy_diff_cv_k` (`BSEPreconditionerTerms`, `_pair_amplitude`, `compute_v_diagonal`, `compute_w_diagonal`, `extract_w_q0`, `build_preconditioner_terms`, `build_shifted_preconditioner`) | `bse_preconditioner.py:22-158` | Superseded by `bse_feast.build_preconditioner_diagonal_sharded`. Carries the `compute_w_diagonal` bug from §1.3. |
| `absorption_eigvecs.compute_oscillator_strengths` | `absorption_eigvecs.py:50-52` | `main()` gets `f_Sa` from `compute_eps2`'s own return instead. |
| `solvers/minres.py` — entire module (294 LOC) | `solvers/minres.py:1-294` | Superseded by level-shifted CG (α_pv makes the Sternheimer operator positive-definite); MINRES also had a documented pseudo-convergence-NaN pitfall. Only referenced by a `pytest.mark.extra`-gated (deselected by default) test file and frozen one-off diagnostic scripts. |
| `solvers/projectors.py` — `make_P_val`/`make_P_precond`/`make_P_rest`/`make_Q_kminq` | `solvers/projectors.py:54-108` | Test/diagnostic vocabulary; `make_Q_kminq`'s sole src/ reference is an unused import. |
| `pseudobands_v2._compute_window_moments` (grid version) | `pseudobands_v2.py:229-254` | Orphaned by a documented Gauss-weights rollback commit; only the discrete sibling is still called. |
| `pseudobands_v2._gauss_from_moments`'s Cholesky `L` | `pseudobands_v2.py:173-180` | Computed, never read; the Stieltjes recurrence below works directly on the moment arrays. |
| `pseudobands.py:394` `Y_cj = None` local | `pseudobands.py:394` | Copy-paste residue from the v2-style dict; v1's CJ branch reads `Y_all` instead. |
| `from functools import partial` | `pseudobands_v2.py:26` | Unused import. |
| `bse_jax.py` `--ring-timing` flag | `bse_jax.py:481` | Parsed, never read; its consumer `ring_matvec_timing` was deleted from `bse_ring_comm.py` in the same cleanup commit that should have removed this flag too. |
| `feast_sweep.py`, `feast_zolo_sweep.py`, `feast_ellipse_mixed_sweep.py`, `bse_feast_dense_debug.py` — **treat as a group, see §3** | `src/bse/` | Zero external callers; deleted-as-dead once (`a0da0a5`), restored already-broken (`fe5e3e8`). Three of the four also carry confirmed crash bugs (§1.2); listed here for completeness but tracked under Test-only below since `bse_feast_dense_debug.py` alone still runs. |
| `pseudopoles_eval.py`/`bse_pseudopoles.py` stale cross-references (`--write-kind` flag that doesn't exist; `build_density_readout_operator_full` import) | `pseudopoles_eval.py:28`, `bse_pseudopoles.py:32` | Same restore-without-validation rot class as the module import bug in §1.2. |

---

## 3. Test-only (8) — keep, isolate from prod path

- **`bse_ring_comm.ring_matvec_smoke_test`/`ring_matvec_correctness_check`**
  (`bse_ring_comm.py:853-996`) — CLI-diagnostic only (`bse_jax --ring-test/
  --ring-check`), print-only with no assert/threshold; exits 0 regardless of the
  printed relative error. Best pytest-ification candidate in the package (add a
  threshold + a tiny fixture) — but note it structurally cannot catch §1.1
  (both "reference" and "ring" sides implement the identical k-diagonal
  exchange formula).
- **Entire Lanczos family** — nothing under `tests/` (pytest's configured
  `testpaths`) imports `bse_lanczos` or `solvers.lanczos`; `src/bse/test_bse.py`
  and `test_davidson_bse.py` are manual `__main__` scripts outside `testpaths`,
  not pytest tests, despite the `test_` prefix. The §1.3 Lanczos bugs (final-slot
  overwrite, transposed block-beta) are both reproducible with a synthetic
  1-GPU/CPU Hermitian operator — the gap is coverage absence, not hardware.
- **`bse_feast_dense_debug.py`** (whole module) — the only member of the
  `feast_*` experiment group that still compiles/runs; self-contained numpy
  validation with zero LORRAX imports. Validates a superseded algorithmic
  variant (hard SVD truncation vs. production's overlap regularization).
- **`bse_kpm.py`/`solvers/chebyshev.py`/`solvers/dos.py` coverage** — zero
  pytest coverage of any of the three; directly explains why the `v_couples_k`
  TypeError (§1.2) shipped and stayed undetected.
- **`tests/archive/projects/test_isdf/sweep_12v12c_plot.py`** — imports the
  stale pre-rename `bse_isdf.pseudopoles_eval` (package renamed to `src/bse` in
  the consolidation); excluded from collection (`norecursedirs=["archive"]`).
- **`test_davidson_bse.py`'s reach into `solvers.davidson._to_host`** (a
  private symbol) — the only exerciser of `solvers/davidson.py`, itself with
  zero pytest coverage despite serving both BSE and DFT NSCF.
- **`context/README.md`'s `test_bse` usage section** — still accurately
  documents the harness's 8 CLI flags and profiling env var; not stale, just
  describing test-only surface.
- **`LORRAX_EXTRA_MU_PAD`** (`src/runtime/padding.py:36-64`) — explicitly
  test-only per its own docstring ("NEVER set this in production runs"); BSE
  inherits it transitively through `padded_mu_extent`.

---

## 4. Refactor targets (23)

The consolidation list — every entry below is confirmed real duplication or
architectural debt with **no** demonstrated wrong-output today (distinguish from
§1's bugs). Grouped by theme:

**Duplicated matvec/kernel math** (the same root cause behind the §1.1 exchange
bug's blast radius — one fix must touch every copy):
`bse_serial.py` / `bse_simple.py` / `bse_ring_comm.py` (ring + full) / dead
`bse_jax.py` module-level quartet all hand-roll the same D/V/W contractions;
`build_bse_ring_matvec` vs. `build_bse_ring_matvec_full` share ~90 LOC verbatim
(`_encode_T`, the sharded-FFT factory + its 14-line workaround comment,
`_apply_W_from_T`, `_apply_D_term`); `compute_pair_amplitude`/`_pair_amplitude`
defined independently 3× (`bse_jax.py:94`, `bse_serial.py:27`,
`bse_preconditioner.py:44`, the last one now dead).

**Duplicated restart I/O**: two `eigenvectors.h5` writers with divergent
conventions (`write_eigenvectors_stream`: eV+valence-flip, compliant;
`write_eigenvectors_h5`: Ry+no-flip, non-compliant — see §1.2); three restart
loaders (`load_bse_data_from_restart_sharded`, `_load_ring_subset`, and a
third private copy in `test_bse.py`) duplicating n_occ resolution and
band-clamp warnings near line-for-line; the 8D/6D/3D layout shim is
independently reimplemented three times with different completeness — one copy
(`_read_vq0_sharded`) got none at all, which is §1.2's restart-format bug.

**Duplicated small utilities**: `_create_mesh_xy` defined 5×
(`bse_feast.py`, `bse_pseudopoles.py`, `bse_w_exact.py`,
`feast_zolo_sweep.py`, `feast_ellipse_mixed_sweep.py`) plus a 6th distinct
mesh builder (`bse_ring_comm.create_mesh_2d`); `WindowSpec` dataclass
triplicated; the Lorentzian broadening kernel implemented 3× (`absorption_
common.lorentzian_broaden`, inlined in `absorption_haydock.jdos_from_dipole`,
re-declared in `eigvals_to_eps2.py`); `_gather_to_host`/`_to_host` quadruplicated
(`solvers/davidson.py`, `bse_davidson_helpers.py`, `davidson_absorption.py`,
the `file_io._slab_io_allgather` original); `delta_E`/`energy_diff_cv_k`
computed independently at 3 sites; the EQP-override band-slice block
copy-pasted 3× (`bse_jax.py`, `absorption_haydock.py`, `davidson_absorption.py`)
with a real divergence in one copy (§1.3); the ellipse-quadrature helper
reimplemented a 3rd time inline in `bse_feast_dense_debug.py`;
`_zolotarev_quadrature_custom` in `feast_zolo_sweep.py` duplicates
`bse_feast.feast_zolotarev_quadrature` (whose `rho_scale` knob already
upstreamed the duplicate's only reason to exist); `_match_eigenvalues`
implemented 3× across `feast_sweep.py`/`feast_zolo_sweep.py`/
`feast_ellipse_mixed_sweep.py`; two parallel `PseudobandsResult` dataclasses
(`pseudobands.py` v1 vs `pseudobands_v2.py`) with incompatible fields forcing
`pb_version` branching in the caller; `cohsex.in` re-parsed by two private BSE
line-parsers instead of the shared `gw_config` layer, silently ignoring any
key besides `wfn_file`/`vhead`/`whead_0freq`; `_reconstruct_from_intermediates`
in `pseudopoles_sweep.py` duplicates `run_pseudopoles`'s brightness→Ritz→J-norm
pipeline (~55 LOC) with the same hard-coded `1e-6` floor.

**Performance/architecture debt** (no wrong output, real cost): `W_R =
jnp.fft.ifftn` on sharded `W_q` computed via the plain (all-gather-forcing) FFT
up to 3× per `bse_feast.main()` run instead of the sharded-FFT helper the
codebase already has elsewhere; `run_kpm_dos`'s bounds phase independently
rebuilds the same fp32 tensor copies + W_R IFFT + matvec that
`estimate_spectral_bounds_sharded` rebuilds again internally (memory-doubling);
`make_chebyshev_recurrence` computes M+1 KPM moments with M matvecs where the
standard doubling identities would need ~M/2; the Krylov storage `Q`/`Q_all`
inside `solve_bse_sharded`'s outer jit carries no sharding constraint (XLA
likely replicates it per device — violates the project's zero-replicated-
intermediates convention); `solve_bse`'s inner matvec is a fresh `jax.jit`
closure per call, observed causing real recompilation cost in a production
profile; `absorption_haydock`'s hard `>=2-device` gate blocks the standard
1-GPU dev rig for no structural reason (`create_mesh_2d` produces a valid 1×1
mesh); `estimate_spectral_bounds_sharded` always uses the TDA-only matvec even
when `use_tda=False`, bounding the non-TDA spectrum off the wrong operator with
only a flat 5% buffer.

---

## 5. Conventions — do not "fix" these (36 verified, not bugs)

Recorded so a future refactor pass doesn't break intentional behavior while
cleaning up the items above. Full reasoning for each is in `_raw_verdicts.json`
(filter `verdict=="not-a-bug-convention"`); highlights:

- **`H = D + V - W`, no singlet factor of 2 on V.** Correct for the FR-bispinor
  (`nspinor=2`) production convention — the spin sum is already absorbed into
  the spin-traced pair amplitude `M_cv = Σ_s ψ*_{c,s}ψ_{v,s}`. Design docs in
  `context/*.md` carry the textbook spin-restricted `D+2V-W` for reference/
  derivation only; do not inject the 2.
- **BGW-compat valence-axis flip + Ry→eV conversion happen ONLY at the
  `eigenvectors.h5` file boundary** (`write_eigenvectors_stream` on write,
  `load_eigenvectors_h5` on read) — all in-memory LORRAX arrays use v=0=deepest-
  valence and Ry throughout. `STATUS.md` "Index ordering" documents this with
  BGW source line citations, independently re-verified against a live BGW
  checkout (`Common/evecs.f90:1982`, `BSE/input_fi.f90:407`,
  `BSE/haydock.f90:536`, `BSE/absh.f90:46` all confirmed to match).
  `absorption.inp`'s documented default `cell_average_cutoff=1e-12` is
  similarly incomplete-but-not-wrong: BGW's actual default is `auto`, which
  resolves to "always average" for untruncated 3D semiconductors — a future
  LORRAX fine-grid design that copies the doc's literal default would silently
  disable BGW's mini-BZ averaging.
- **Default `bse_jax` route is RPA (D+V, no W) unless `--bse` is passed**, and
  `--bse --lanczos` still defaults to full non-TDA unless `--tda` is passed —
  documented in the CLI help text and in every recommended `STATUS.md` command;
  a real footgun for a plain invocation, but working as designed, not a defect.
- **`use_nohead`/`--nohead` head-A/B debugging convention** — `--nohead` is
  currently broken wiring (§1.2), not evidence the underlying head-injection
  machinery (`apply_q0_head_rank1_sharded`) is dead; it's still live.
- **W is charge-channel only** (no spinor axes on `W_q0`/`W_R`); exchange `M`
  is spin-traced at the same k. Consistent with the documented bispinor
  screened-W roadmap — correct for `nspinor=2`, would need updating for a
  future `nspinor=1` scalar-relativistic path.
