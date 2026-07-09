# G0W0 vs self-consistent toggle — design + single-jit audit

**Date:** 2026-07-08 · **Branch read:** `agent/memplanner-cleanup` (lorrax_D, HEAD bb95bc3)
**Scope:** analysis only — no source edits. Empirical runs under
`reports/gw_refactor_map_2026-07-01/archive/g0w0_sc_toggle_audit/` (MoS2 fixtures, 1 A100,
JID 55674176, `JAX_LOG_COMPILES=1`).

Companion to NEXT_TARGETS TIER-0★ A (`sigma_at_dft_energies` authoritative-QP wiring)
and #11 (`LORRAX_SC_*` env → config promotion).

---

## 1. Current-state map — how QP energies are produced today

Two config axes exist: `compute_mode` (the Σ ansatz: `x_only | cohsex | gn_ppm |
hl_ppm`) and `self_consistent` (bool). A *third*, implicit axis — **how the QP
energies are extracted from Σ** — is currently smeared across the post-Σ seam of
`gw_jax.main` and is what this design makes explicit.

### 1a. One-shot flow (`self_consistent = false`)

`main()` always runs: ISDF ζ-fit → V_q → χ₀ → W (`src/gw/gw_jax.py:199-290`) →
static COHSEX Σ pass (`:358`, all modes — gives `sig_sx/sig_coh/sig_h/sig_x`) →
for dynamic modes the standalone PPM pipeline (`:411-437`,
`ppm_pipeline.compute_ppm_sigma_pipeline`). Then the post-Σ dispatch:

| mode | `sigma_total` for the eigh (`gw_jax.py:751`) | eqp0.dat / eqp1.dat |
|---|---|---|
| X_ONLY / COHSEX (static) | `sig_sx + sig_coh + sig_h` (`:718`) — no ω-dependence, nothing to solve | at-DFT Newton: `eqp0 = E_DFT + (kin+V_H+Σ_x+Σ_c−E_DFT)`, Z=1 ⇒ eqp1==eqp0 (`eqp_bgw.compute_eqp_diag`) |
| GN/HL-PPM, in-memory Σ_c(ω) | **diagonal on-shell fixed point** `E = h0 + ReΣ(E)` (`qsgw_utils.solve_diagonal_sigma_fixed_point`, called at `gw_jax.py:655`) → optional scissor for out-of-grid bands (`:678-701`, gated by `ppm.sigma_at_dft_extrapolate`) → QSGW-Hermitised `Σ_xc(E_sc)` (`build_qsgw_sigma_xc`, `:709`) → eigh | at-DFT interp + central-difference Z from the ω-grid (`gw_output.write_results` → `eqp_bgw.compute_z_factor_from_omega_grid`) |
| GN/HL-PPM, streamed (`KIJ_STREAM`) | **no QP solve at all** — `sigma_c_omega is None` so the `:628` branch is skipped; `sigma_total` falls through to the *static* `sig_sx+sig_coh+sig_h` (`:716-718`). eigh output is static-COHSEX, silently, in a PPM run | eqp0 correct (at-DFT diag read back from the kij h5); **eqp1 == eqp0 because `sigma_c_omega_diag_ev` is None ⇒ no Z** (`:798-804`). Known WS1/Bug-B-adjacent gap. |

Key observation: **the standard G0W0 numbers (Σ(E_DFT) + Z-linearization) are
already what eqp0.dat/eqp1.dat report in every one-shot mode.** The at-DFT
interpolation runs unconditionally in `ppm_pipeline._eval_sigma_c_at_dft_energies`
(step 4, `ppm_pipeline.py:205-272,421-431`). What the *fixed point* controls is
only the **evaluation energies fed to the QSGW-symmetrised Σ_xc** whose eigh
produces `E_qp_ry` / `U_qp` → `qp_wfn_rotations.h5`, `WFN_qp.h5`, and
`GWResults.E_qp_ry`. So today's "one-shot" is really *one-shot Σ + on-shell
eigenvalue self-consistency in the evaluation energy* — not textbook G0W0 — for
everything downstream of the eigh, while the text outputs are textbook G0W0.

The orphan flag: `sigma_at_dft_energies` is parsed into `PPMConfig`
(`gw_config.py:317,572,954`) and **never read** (grep: zero physics consumers; both
regression fixtures set it `= true` expecting it to do something). Its intended
meaning (NEXT_TARGETS TIER-0★ A, consensus WS6): make the at-DFT evaluation the
*authoritative* QP path — i.e. skip the fixed point and evaluate the QSGW build at
`E_DFT`, so eigh-family outputs are consistent with eqp0/eqp1.

### 1b. Self-consistent flow (`self_consistent = true`)

Entry at `gw_jax.py:475-624`. **Mode-agnostic** — despite the stale comments in
`gw_config.py:51` ("currently only COHSEX is wired") and `:190-191`, the SC loop
dispatches through `sigma_dispatch.compute_sigma_xc` which handles all four modes,
and **GN-PPM SC runs end-to-end today** (verified empirically, §4).

Per iteration (`sc_iteration.gw_iteration_map`):

1. `eigh(H_qp_dft)` → `(E_qp, U_qp, efermi)` — k-sharded eigh, kernel cached per mesh.
2. `rotate_wavefunctions(wfns_dft, U_qp, …)` — rotates the **original DFT bundle**
   (no cumulative drift) on the active window `band_slices.sigma`.
3. `screening.compute_screening` — re-solves **χ₀ → W at every frequency the mode
   needs** (static for COHSEX; static + probe for PPM). ζ / V_q are **NOT** refit —
   they ride in `SCInputs.V_q` unchanged. W is rebuilt every iteration.
4. `compute_sigma_xc` — V_H + Σ_x (+ SX/COH for COHSEX; + full PPM Σ_c(ω) pipeline +
   QSGW build at `e_qp_ev` for dynamic modes). Per-iteration sigma_mnk.h5 writes are
   suppressed (`write_sigma_omega_h5=False`); one write at convergence.
5. Rotate `(V_H+Σ_xc)` back to DFT basis (`_rotate_to_dft_basis`), add `kin_ion_dft`,
   apply the band partition (protected off-diagonals / in-range diagonals /
   scissor for out-of-range) → new carry `H_qp_dft`.

Energy feedback into iteration n+1 is **entirely through the carry `H_qp_dft`**:
next iteration's eigh regenerates (E_qp, U_qp); `e_qp_ev` (PPM evaluation energies)
and the rotated ψ both derive from it. Convergence = RMS ΔE of consecutive
`eigvalsh` (eV). Driver: `run_self_consistency` — `max_iter==1` fast path is exactly
one-shot G0W0 at E_DFT (initial state is `diag(E_DFT)`, so `eigh → U=I`).

**Env knobs (all slated for config promotion, NEXT_TARGETS #11):**

| knob | default | consumer |
|---|---|---|
| `LORRAX_SC_MAX_ITER` | 20 | `gw_jax.py:536` |
| `LORRAX_SC_TOL_EV` | 1e-4 | `:537` |
| `LORRAX_SC_ACCEL` | `rcrop` (`linear` = plain α-mixing) | `:538` |
| `LORRAX_SC_DEPTH` | 5 (rCROP history) | `:539` |
| `LORRAX_SC_MIXING` | 1.0 (linear only) | `:540` |
| `LORRAX_SC_DUMP_DIR` | unset (E-history npy dump) | `sc_iteration.py:430` |

Note: rCROP makes **two** `gw_iteration_map` calls per accelerator iteration
(trial + residual), so pipeline-call count = 2×`max_iter` there.

**Redundancy at SC startup:** `main()` unconditionally runs the full one-shot
χ₀→W→static-Σ pass (+ W0 restart persist + bare-X print) *before* entering the SC
branch; iteration 1 then recomputes χ₀/W/Σ from the same DFT wfns. The only things
the loop actually consumes from the pre-pass are `V_q`, `quad`/`e_ref`,
`static_head_terms`, `kin_ion` — the W and Σ built at `:224-390` are discarded
(their side effect: kernel caches are warm, see §3).

### 1c. What eqp0.dat / eqp1.dat mean today (must not silently change)

| | eqp0.dat | eqp1.dat |
|---|---|---|
| one-shot static | `E_DFT + (kin+V_H+Σ_x+Σ_c^static−E_DFT)` | == eqp0 (Z=1) |
| one-shot PPM (in-memory) | same formula with `Σ_c(E_DFT)` interpolated on the ω-grid | Z-linearized, central-difference Z (`dE=0.5 eV`, BGW spacing) |
| one-shot PPM (streamed) | same (from h5) | **== eqp0 (Z silently lost — no ω-diag)** |
| SC (any mode) | same *formula*, but the Σ pieces are the **converged** Σ rotated back to the DFT basis and still evaluated at `E_DFT` — i.e. "one more at-DFT Newton step from the SC fixed point", **not** the converged QP eigenvalues (those are `E_qp_ry` from the eigh of `state_final.H_qp_dft`) | same caveat |

So eqp0/eqp1 have a *stable formula* everywhere; only the provenance of Σ changes
under SC. The toggle below keeps this invariant.

*Side observation (verify separately, not part of this design):* in the static-mode
writer path (`gw_output.write_results:298-303`) the eqp0 Σ_c slot receives
`diag(Σ_COH)` while `sigma_x_diag_ev` is **bare** Σ_X — the screened part of
Σ_SX−Σ_X never enters eqp0.dat for COHSEX runs; the freq_debug table
(`gw_jax.py:908-913`) instead uses `Σ_c = Σ_SX+Σ_COH` *on top of* bare-X, which
double-counts X relative to the writer. The two static eqp0 recipes disagree with
each other; at most one can match BGW. Flagged for a follow-up gate check.

---

## 2. The toggle design

### 2a. Config surface: one new axis, `qp_solver`

Proposal: a **three-state `qp_solver` axis** that answers "how are the QP energies
determined from Σ", orthogonal to `compute_mode` ("what Σ is"):

```
qp_solver = one_shot_dft   # G0W0: Σ from DFT inputs, everything evaluated at E_DFT.
                           # No iteration of any kind. THE DEFAULT.
          = fixed_point    # one-shot Σ, diagonal on-shell solve E = h0 + ReΣ(E)
                           # for the QSGW-build evaluation energies (today's
                           # dynamic-mode behavior; eigenvalue-only, Σ never rebuilt)
          = self_consistent # full QSGW loop (sc_iteration): Σ rebuilt each
                           # iteration from rotated ψ + previous iteration's E
```

Why one axis rather than keeping `self_consistent: bool` + a new binary flag:
the three states are **mutually exclusive answers to the same physics question**
(each names a standard method: G0W0 / G0W0+on-shell / QSGW — "options map to
physics docs"), and today's smearing (bool + orphan flag + implicit streamed
fallthrough) is exactly the failure mode of two half-axes. A single enum also
gives the driver ONE dispatch point instead of the current three
(`:475 if self_consistent` / `:628 elif dynamic and sigma_c_omega` / `:679
sigma_at_dft_extrapolate` sub-gate).

Absorption / back-compat (all in the `qp_solver` resolution property, mirroring
how `compute_mode="auto"` absorbs `do_screened`/`use_ppm_sigma`):

- `qp_solver = auto` (default in `_DEFAULTS`) resolves:
  1. `self_consistent = true` → `SELF_CONSISTENT` (legacy key kept, parsed,
     deprecation-warned);
  2. else `sigma_at_dft_energies = true` → `ONE_SHOT_DFT` (the orphan flag becomes
     a live legacy alias — TIER-0★ A wiring — then deprecated);
  3. else → **`ONE_SHOT_DFT`** (the lead's requirement: G0W0 one-shot is the
     standard result and the default).
- `ppm.sigma_at_dft_extrapolate` stays: it is a *sub-knob of `fixed_point`*
  (scissor extrapolation for out-of-grid bands); ignored in the other two modes.
- `LORRAX_SC_*` env knobs promote to a new `SCConfig` group
  (`sc_max_iter`, `sc_tol_ev`, `sc_accelerator`, `sc_history_depth`, `sc_mixing`,
  optional `sc_dump_dir`) read only when `qp_solver = self_consistent` (#11 done in
  the same commit, mechanical).

**Default-change consequence (be explicit):** for dynamic modes the eigh-family
outputs (`E_qp_ry`, `qp_wfn_rotations.h5`, `WFN_qp.h5`) currently come from
`fixed_point`; making `one_shot_dft` the default changes them (eqp0/eqp1/sigma_diag
are untouched — they are at-DFT in all one-shot modes already). If the golden gates
hash any eigh-family output this needs a gate re-freeze; if they only check
eqp/sigma_diag text files it is ref-invariant. Either way it is a *documented*
default flip, same class as the WS6 `static_limit` default flip.

### 2b. Semantics per mode

| `qp_solver` | static modes (X_ONLY/COHSEX) | dynamic modes (GN/HL-PPM) | eqp0/eqp1 |
|---|---|---|---|
| `one_shot_dft` | identical to today (Σ static ⇒ Σ(E_DFT)=Σ; fixed point would be a no-op anyway) | skip `solve_diagonal_sigma_fixed_point`; `E_eval = E_DFT−E_F` feeds `build_qsgw_sigma_xc` → eigh consistent with eqp0 | unchanged meaning (at-DFT + Z) |
| `fixed_point` | **rejected at config validation** (no ω-grid to solve on — a silent no-op would blur the axis) | today's behavior verbatim, incl. `sigma_at_dft_extrapolate` scissor | unchanged |
| `self_consistent` | current SC-COHSEX loop | current SC-PPM loop (works today, §4) | unchanged formula; Σ provenance = converged (as today, §1c) — document in the file header comment |
| any, streamed Σ_c | n/a | `one_shot_dft`: fine (at-DFT diag comes from the kij h5). `fixed_point`/`self_consistent`: **reject at config validation** — today this pair silently degrades (§1a row 3); an error is strictly better | eqp1 Z-loss in streamed mode remains a separate WS1 fix |

### 2c. Implementation sketch (~30 lines of real change)

1. **`gw/gw_config.py`** (+15 lines):
   - `class QPSolver(str, enum.Enum): ONE_SHOT_DFT="one_shot_dft"; FIXED_POINT="fixed_point"; SELF_CONSISTENT="self_consistent"`.
   - `_DEFAULTS["qp_solver"] = "auto"`; add to `_NORMALIZE_STR`; parse into the
     top-level config (`qp_solver_raw`).
   - `@property qp_solver` — the auto-resolution + validation
     (`fixed_point` × static mode → ValueError; `fixed_point|self_consistent` ×
     `omega_accumulation=kij_stream` → ValueError).
   - `SCConfig` frozen group with the five promoted knobs; deprecation warnings for
     `self_consistent=`/`sigma_at_dft_energies=` in `read_lorrax_input` (keys still
     honored via auto-resolution).
2. **`gw/gw_jax.py`** (~10 lines changed at 3 dispatch points):
   - `:475` — `if config.qp_solver is QPSolver.SELF_CONSISTENT:` (replaces
     `config.self_consistent`); `_max_iter…_mixing` read from `config.sc`.
   - `:649-712` — the WS6 flip, same dispatch pattern as the `:673` scissor sibling:
     ```python
     if config.qp_solver is QPSolver.ONE_SHOT_DFT:
         E_sc_rel_ry = E_dft_rel_ry            # authoritative at-DFT (G0W0)
     else:  # FIXED_POINT
         E_sc_rel_ry, _, n_iter = solve_diagonal_sigma_fixed_point(...)
         ... scissor block unchanged ...
     ```
   - `GWResults(self_consistent=config.qp_solver is QPSolver.SELF_CONSISTENT)`.
3. **`gw/sc_iteration.py`** (2 lines): `LORRAX_SC_DUMP_DIR` → `config.sc.dump_dir`.
4. **`templates/cohsex.in` + `docs/docs_gwjax/COHSEX_INPUT.md`**: document
   `qp_solver` beside `compute_mode` (feeds NEXT_TARGETS #13).

Gate story: existing eqp gates cover `one_shot_dft` (they already assert at-DFT
numbers); add one small assertion that `qp_solver=fixed_point` on the GN-PPM
fixture reproduces today's frozen `qp_wfn_rotations.h5` eigenvalues, so the default
flip is provably a pure re-labeling.

---

## 3. Single-jit audit (empirical, `JAX_LOG_COMPILES=1`, 1 A100)

Method: SC forced to 3-4 iterations (`LORRAX_SC_TOL_EV=1e-10`, `LORRAX_SC_ACCEL=linear`,
α=1) on two fixtures; every `Finished XLA compilation` / `Finished tracing` log line
binned by SC-iteration boundary. Runs + parser under
`archive/g0w0_sc_toggle_audit/{run_cohsex_sc,run_gnppm_sc}/`.

### 3a. Compile counts per iteration

**SC-COHSEX (MoS2 3×3, WFNsmall, 60 centroids, nb_sigma=30):**

| segment | XLA compiles | compile s | retraces | wall |
|---|---|---|---|---|
| pre-SC init + one-shot pass | 245 | 5.40 | 570 | ~10 s |
| SC iter 1 | **36** | 1.55 | 68 | 2.1 s cold cache / 0.52 s warm |
| SC iter 2 | **0** | 0 | **0** | 0.02 s |
| SC iter 3 | **0** | 0 | **0** | 0.02 s |
| SC iter 4 (timestamped rerun) | **0** | 0 | **0** | 0.02 s |
| post-SC writer | 5 | 0.36 | 4 | ~0.4 s |

(Iteration wall of 0.02 s is fixture-tiny — nq=9, μ=60, nb=40 — the point is the
compile column, not the absolute wall.)

**SC-GN-PPM (MoS2 3×3, full WFN, 642 centroids, nb=80, 41-pt ω-grid):**

| segment | XLA compiles | compile s | retraces | wall |
|---|---|---|---|---|
| pre-SC init + one-shot pass | 272 | 17.95 | 672 | ~46 s |
| SC iter 1 | **79** | 7.11 | 157 | 15.5 s |
| SC iter 2 | **8** | 0.48 | 8 | 6.4 s |
| SC iter 3 (steady state) | **2** | 0.02 | 2 | 6.1 s |
| post-SC writer | 14 | 0.75 | 13 | — |

### 3b. Per-jit verdict for one SC iteration

| pipeline piece | jit(s) | iteration n+1 behavior | why |
|---|---|---|---|
| ζ-fit / V_q | (none in loop) | **reused data** — not recomputed | `SCInputs.V_q` constant; ζ never refit |
| eigh / eigvalsh (k-shard) | `_f` ×2 | **reuse** | module cache `_KSHARD_EIGH_CACHE[id(mesh)]` (`sc_iteration.py:209`) |
| ψ rotation | `xn/xr/yr/yn` accessors, `_rotate_psi_band{last,first}`, eager `dynamic_update_slice`/`scatter` | **reuse** | module-level jits + static-slice accessors; shapes fixed |
| χ₀ (per W request) | `minimax_tau_integrate_chi` | **reuse** (compiled 2× total in iter 1 for PPM: static-quad and probe-quad have different node counts ⇒ different `nodes.t` shape ⇒ two cache entries; both stable after iter 1) | factory cache `_chi_minimax_kernel_cache[(id(mesh),kgrid)]`; minimax node values enter as data |
| W solve | `_solve_w` | **reuse** | `_w_solve_cache[(id(mesh),nq,n_rmu)]`; χ₀ donated each call (fine) |
| static Σ kernels | `sigma_sx`, `sigma_coh`, `hartree` | **reuse** | `_cohsex_kernel_cache[(id(mesh),kgrid)]` |
| PPM pole fit | eager jnp chain in `fit_ppm` | **reuse** (dispatch cache) | value-only changes |
| PPM τ kernel | `_tau_kernel` (+AOT `precompile_sigma`) | **reuse**; the per-iteration `precompile_sigma` call re-*lowers* (host trace ~0.1 s) but hits the compile cache | `_sigma_tau_kernel_cache[(id(mesh),kgrid)]`; shape-invariant across branches/windows — window/node *count* changes only change dispatch count, never shapes |
| ω-projection + Σ_c(ω) accumulation | (none — **host numpy**) | n/a | `_TauAccumulator` async-D2H + `np.exp` ω-kernel (`ppm_accumulators.py:103`) — deliberate (τ-scan-on-device tried + reverted for perf) |
| Σ_c diag extraction | `_extract` | **RETRACE + RECOMPILE every iteration** | `@jax.jit` closure defined *inside* `extract_sigma_diag_replicated` (`qsgw_utils.py:165`) — fresh function object per call ⇒ fresh pjit cache |
| QSGW Σ_xc build | `_kernel` | **RETRACE + RECOMPILE every iteration** | same pattern inside `build_qsgw_sigma_xc` (`qsgw_utils.py:262`) |
| DFT-basis rotation, band partition | `_rotate_to_dft_basis`, `apply_band_partition` | **reuse** | module-level jits |
| minimax window construction | (host Remez solves) | recomputed every iteration | value-dependent on E_qp ranges; host-only cost |
| iter-1 one-off retraces | χ₀/W/Σ kernels once more | iter-1 only, stable from iter 2 | rotated-bundle arrays are jit outputs (committed shardings) vs the pre-pass's host-staged inputs — pjit keys on committedness, one extra specialization |

**Verdict:** the Σ pipeline is *not* one jit — it is a Python-orchestrated chain of
~10 cached jits + host accumulation — but the user's operative requirement
("one compilation for a single-shot calc covers everything needed in SC") is
**already met for COHSEX** (0 compiles, 0 retraces from iteration 2 on) and missed
for GN-PPM by **exactly two nested-scope jits** (`qsgw_utils.py:165` `_extract`,
`:262` `_kernel`) plus a handful of iter-2 second-touch specializations. Steady-state
recompile cost is small here (0.02-0.5 s vs 6.1 s/iter, <1-8%) *with a warm
persistent compile cache*; at production sizes `_kernel` traces over the full
`(nω,nk,nb,nb)` tensor and the retrace is pure waste on every iteration.

**Fix (cheap, mechanical):** hoist both closures to module scope (the index/weight
arrays are already runtime args in `_kernel`'s signature; `_extract` needs only the
mesh via the sharding constants — use the same `id(mesh)`-keyed factory-cache
pattern every other kernel in the codebase uses). After that, GN-PPM SC steady
state should also be 0 compiles / 0 retraces.

**What a literal single-jit-per-iteration would additionally need** (not
recommended as a near-term target):
(a) on-device ω-projection/accumulation — i.e. the previously-reverted τ-scan
(regressed sigma_ppm ~80% at MoS2 3×3, see `ppm_tau_kernel.py:74-82` deferred
notes); (b) frozen window/minimax node layouts (node counts are value-dependent
on the drifting E_qp range — would need max-node padding); (c) io_callback-wrapped
head-resolver/h5 seams; (d) traced efermi/E_qp (currently `float()` host pulls in
`gw_iteration_map`). The zero-retrace contract above delivers the same compile
economics without any of that.

### 3c. Compile time vs iteration time (quantified)

- GN-PPM: pre-SC one-shot 272 compiles / 17.9 s; SC iter 1 pays 7.1 s compile of a
  15.5 s wall (46%); steady iterations 6.1-6.4 s with ≤0.5 s compile. Σ_c(ω)
  execution dominates each iteration (`sigma.exec` 19.3 s over 3 iterations).
- COHSEX: iter 1 pays 1.55 s compile of ~2.1 s (cold disk cache; 0.52 s warm);
  iterations 2+ are compile-free at 0.02 s each on the fixture.
- The pre-SC one-shot pass is why iter 1 is not worse: it warms every kernel cache
  the loop reuses. If the §1b startup redundancy is removed for SC runs, keep the
  `precompile_*` AOT warmers (or accept the compiles moving into iter 1 — same
  total).

---

## 4. Gaps

1. **SC is wired for GN-PPM — the docs deny it.** `gw_config.py:51,190` say
   "currently only COHSEX". Empirically: SC-GN-PPM ran 3 full iterations
   end-to-end on 1 GPU (RC=0, converged writer output). The mode-agnostic
   `compute_sigma_xc` + `screening_requests_for` did their job. Fix the two
   comments in the toggle commit.
   *Caveat:* with `linear` α=1.0 both fixtures oscillate (COHSEX RMS 137→2.6→0.7 eV;
   GN-PPM 6.7→2.3→8.3 eV) — the documented 2-cycle Jacobian behavior that `rcrop`
   (the default accelerator) exists for; not a wiring defect, but SC remains
   physics-ungated (no golden SC gate exists — NEXT_TARGETS #11 note).
2. **Streamed (`kij_stream`) × anything-but-`one_shot_dft` silently degrades**
   (§1a row 3: static Σ in the eigh; eqp1 Z lost). The toggle turns the first
   into a config-validation error; the eqp1 Z-loss stays a WS1 item.
3. **SC startup redundancy:** the pre-SC one-shot W + static-Σ pass (~1.4 s COHSEX
   fixture, ~2.2 s + wasted W0-persist at GN-PPM fixture scale; grows with system)
   is discarded by the loop. A `qp_solver` dispatch before the `:358` Σ pass could
   skip it for SC runs — do it *after* the toggle lands, since it changes where
   iter-1 compiles are paid (§3c).
4. **`user's 'one compile covers single-shot AND SC'` for GN-PPM** requires only:
   (i) the two-line hoist of `_extract`/`_kernel` (§3b), (ii) leaving the kernel
   factory caches alone (they already key on mesh+kgrid, not on wfn identity), and
   (iii) NOT moving the ω-grid or band window between one-shot and SC configs
   (shape identity is what makes the caches hit). With (i)-(iii), a single-shot
   run compiles every kernel an SC run needs except the SC-only quartet
   (`_f` eigh ×2, rotation pair, `apply_band_partition`, `_rotate_to_dft_basis`)
   — ~1.5 s, paid once in iter 1.
5. eqp0 static-mode Σ_SX−Σ_X inconsistency between writer and freq_debug (§1c
   side observation) — needs a BGW-parity check before anyone trusts static
   eqp0.dat beyond gate-relative comparisons.

---

## 5. Empirical artifacts

- `archive/g0w0_sc_toggle_audit/run_cohsex_sc/cohsex_sc.log` — 3-iter SC-COHSEX,
  JAX_LOG_COMPILES (compile counts of §3a table 1); `cohsex_sc_ts.log` — 4-iter
  timestamped rerun (wall times).
- `archive/g0w0_sc_toggle_audit/run_gnppm_sc/gnppm_sc.log` — 3-iter SC-GN-PPM, timestamped
  + compile-logged (§3a table 2).
- `archive/g0w0_sc_toggle_audit/run_ts.py` — print-timestamping wrapper (analysis-only).
- Parser one-liners inline in the session; segment = lines between `SC iter N` prints.
