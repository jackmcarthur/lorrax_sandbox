# g0w0_sc_toggle_impl — implementation validation runs for the qp_solver toggle

Implements G0W0_SC_TOGGLE_DESIGN.md on lorrax_D `agent/memplanner-cleanup`
(base 620b501). All runs: MoS2 3×3 GN-PPM fixture (tests/regression/gnppm_debug),
1 A100, JID 55691706, fresh per-run JAX_COMPILATION_CACHE_DIR.

## Directories

- `run_oneshot_head/` — one-shot GN-PPM fixture at HEAD 620b501 (pre-toggle:
  eigh family = diagonal on-shell fixed point). Source of the frozen
  reference `head_frozen/` and of
  `tests/regression/gnppm_debug/eqp_rotations_fixedpoint_ref.npy`.
- `head_frozen/` — frozen HEAD outputs: qp_wfn_rotations.h5, sigma_diag,
  eqp0/eqp1.
- `run_oneshot_new/` — same fixture after the toggle (default
  `qp_solver = one_shot_dft`).
- `run_sc_head/` / `run_sc_after/` — SC-GN-PPM, 3 forced iterations
  (`LORRAX_SC_MAX_ITER=3 LORRAX_SC_TOL_EV=1e-10 LORRAX_SC_ACCEL=linear
  LORRAX_SC_MIXING=1.0`), JAX_LOG_COMPILES=1, before/after the qsgw_utils
  jit hoist.

## Results

### Default flip is a pure re-labeling

- `run_oneshot_new` vs `head_frozen`: sigma_diag_gnppm_test.dat, eqp0.dat,
  eqp1.dat **bit-identical except the timestamp header line** (every physics
  row byte-equal).
- eigh-family output (E_qp_nk_rydberg in qp_wfn_rotations.h5) moves as
  documented (design §2a "default-change consequence"):
  max|Δ| = 8.93e-2 Ry = 1.215 eV, mean |Δ| = 7.9e-3 Ry — the at-DFT vs
  on-shell evaluation-energy difference.
- `qp_solver = fixed_point` reproduces the frozen HEAD E_qp_nk_rydberg —
  asserted by the new pytest gate
  `test_gnppm_fixed_point_reproduces_frozen_qp_rotations`
  (frozen ref: eqp_rotations_fixedpoint_ref.npy, atol 1e-6 Ry).

### SC compile counts per iteration (JAX_LOG_COMPILES=1)

| segment | before (620b501) | after (hoist) |
|---|---|---|
| pre-SC init + one-shot pass | 272 compiles / 671 retraces | 272 / 671 |
| SC iter 1 | 91 / 169 | 91 / 169 |
| SC iter 2 | 8 / 8 | 6 / 6 (one-off eager multiply/convert_element_type second-touches) |
| SC iter 3 (steady state) | **2 / 2** (`_extract` + `_kernel`) | **0 / 0** |
| post-SC writer | 14 / 13 | 13 / 12 |

SC RMS ΔE trajectories bit-identical before/after
(6.683779 / 2.322456 / 8.250645 eV) — the hoist is pure compile economics.
The LORRAX_SC_* envs flow through the deprecated-override path (config
prints a note per env).

Big h5 artifacts (WFN.h5 links, sigma_mnk.h5, tmp/) removed after the runs;
logs retained.
