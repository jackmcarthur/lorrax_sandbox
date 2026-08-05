
**2026-05-13 — Agent 4 review-ready (Phase 1 prep).**

Refreshed on `round5_unified_plan.md` §5.  Numerics contract:
**`rtol=1e-10, atol=1e-12`** (NOT bit-equal — sum order differs;
ULP-class drift ~6e-15 relative at CrI3 nb_total=160, well inside
the gate).  Mirrors `accumulate_rchunk_to_gflat` test scaffold.

### G1 validation checklist (post Agent 2's commit)

**G1.1 — Unit test on synth WFN (CPU, no allocation needed)**

Pass criteria (each is a hard gate; all must hold):

- `z_q_from_psi_sm(new)` vs `z_q_from_psi_sm(prior body @ 5cadd4b)`
  on synth `psi_l_X / psi_r_X + psi_G_store` for charge channel
  (γ̃_L = γ̃_R = None).  Output `Z_q` shape `(nq, n_rmu, n_zchunk)`.
  Tolerance: **`np.testing.assert_allclose(rtol=1e-10, atol=1e-12)`**.
- `c_q_from_psi_sm(new)` vs `c_q_from_psi_sm(prior body)`. Same
  tolerance.  CCT runs with `n_col == n_rmu` (square centroid),
  not r-chunk extent — must verify the new design preserves this.
- **Edge case G1.1a — single bc** (`n_bc == 1`).  Plan §5.3.1.
- **Edge case G1.1b — short final bc** (`nb_total = bc·k + r`,
  `r ∈ {1, bc-1}`). Plan §5.3.3.  Pad-band zeros must be inert.
- **Edge case G1.1c — asymmetric L/R windows** (`nb_L != nb_R`,
  partial overlap with bcs that hit L-only, R-only, both, neither).
  Plan §5.3.2.
- **Edge case G1.1d — bispinor `vertex_mu_L = 1`** (γ̃ ≠ I path).
  Plan §5.3.5.  Tests `gamma_double_contract` with `(perm, phase)`
  tuples threaded through unchanged.
- **Edge case G1.1e — pseudobands** (`band_norms ≠ None`, with
  pre-multiply of `psi_l_X / psi_r_X` by `1/norms` before the
  kernel call).  Plan §5.3.4.

Failure mode handling:
- If max |Δ| ∈ [1e-12, 1e-10] **relative** → expected ULP-class
  drift, report it as the "new design's drift signature".
- If max |Δ| > 1e-10 **relative** → real divergence; STOP and
  post `BLOCKER:` with the failing element + suspected cause
  (mask bug, norm-order bug, γ̃-fold bug, all_gather axis-ordering
  per §2.9 tail).

**G1.2 — MoS2 3×3 end-to-end ζ-fit (uses lxattach for the alloc)**

Pass criteria:

- Full ζ-fit completes on lorrax_B (new) for the MoS2 3×3 charge
  channel.  Compare `zeta_q.h5` (or `zeta_q_gflat.h5` on the
  G-flat path) element-wise vs a lorrax_A `ff5873c` baseline (a
  pre-Path-D MoS2 3×3 run).  Baseline locations include
  `runs/MoS2/00_mos2_3x3_cohsex/{01_lorrax_gnppm,02_lorrax_xonly,
  A_bench_default_2026-04-22}/tmp/zeta_q.h5` — will pick the one
  with cohsex.in closest to the new run's settings.  Tolerance:
  **`atol=1e-12` per element** (matches the prompt's instruction).
- Reports the max |Δ| across the full ζ tensor `(nq, n_rmu,
  n_rtot)`.
- Optional: re-run `02_lorrax_xonly`'s eqp0.dat target and confirm
  Σ values match within COHSEX tolerance — that's the user-
  observable correctness check.

If G1.1 + G1.2 both pass: post **"Agent 4 G1 passed"** with the max
|Δ| signature and the run path.

### G3 validation checklist (post Agent 3's CrI3 HLO run)

Coordinate with **Agent 3** on the shared CrI3 6×6 80 Ry run:
Agent 3 dumps HLO + reads memory-usage-report; I read `gw.out` for
end-to-end progress + scan/SPMD warnings.  Pass criteria:

1. `Started zeta fitting at HH:MM:SS` line appears, followed by all
   16 r-chunks (look for `r-chunk 16 / 16` in the progress bar) +
   the remainder chunk completing.
2. **Tracer-leak gate**: the remainder chunk has a DIFFERENT
   `r_len` from the full r-chunks, forcing a re-jit; this is
   exactly the scenario Round-3's flat-axis path hit the tracer
   leak on, so the remainder completing cleanly is the load-bearing
   evidence the new design fixed it.
3. `grep -c 'Involuntary full rematerialization' gw.out` returns
   `0` (no remat warnings).  Today's `lorrax_B_path_d_hlo_2026-05-13/
   gw.out` had ~32 of them at the consumer boundary; new design
   has no such boundary so they MUST be gone.
4. `grep -c 'UnexpectedTracerError\|Traceback' gw.out` returns `0`
   for the kernel path (ignore the known downstream
   `qp_wfn_rotations.h5` shape-mismatch at write-out — that's an
   unrelated bug per the prompt).
5. `grep -c 'RESOURCE_EXHAUSTED' gw.out` returns `0`.
6. Agent 3 confirms total preallocated-temp ≤ 15 GiB (vs today's
   48.63 GiB) per §4.1's converged prediction.

If all six conditions hold: post **"Agent 4 G3 passed"**.

### G1 test scaffold location

No existing direct unit test for `c_q_from_psi_sm` /
`z_q_from_psi_sm` (`grep` came up empty across `tests/*.py`).  Will
write `tests/test_zq_from_psi_sm_bit_identity.py` once Agent 2
commits — small synthetic WFN (nk=4, nb_total=8, ns=2, fft 4·4·4,
n_rmu=4, n_zchunk=8), mock PsiGStore with `np.zeros`-padded host
tile, run side-by-side on a 1×1 mesh.

### Round-6 dependencies for me

- Phase 2 (G1): blocks on Agent 2's "round 6 done" commit.
- Phase 3 (G3): blocks on Agent 3's CrI3 dump invocation.  Will
  read `gw.out` from the same dir Agent 3 dumps HLO into.
- Phase 4 (sentinel): blocks on G0 (A2) + G1 (me) + G2 (A3) + G3
  (me) + A1 SPMD blessing.

Standing by.  Reading nothing in `sources/lorrax_B/src/` until
the commit lands (read-only constraint per prompt §"Constraints").
