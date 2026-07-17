# Primer-response prototype campaign — synthesis and adjudication

**Date** 2026-07-17 · **Scope** three parallel prototype constructions (C1/C2/C3) of the
ARBITRARY_Q_PRIMER_RESPONSE counterproposal, adjudicated against the measured refutations in
`arbitrary_q_bse.md` §3.2/§3.5, under the OWNER-GOVERNING metric (2026-07-17 pushback): error
under PHYSICAL pair-amplitude contractions — the gap-window exchange block `B = M^H V_Q M`
(spin-traced pair rows, top-3 valence x bottom-3 conduction x all k) and the TDA
exciton-eigenvalue swap shift — NOT tile Frobenius. Tile numbers are secondary diagnostics.

**Fixture** MoS2 3x3 (+ 6x6 for C2 four-tails/angles), charge channel, n_mu=640, 30 Ry
zeta-sphere, full-BZ 9 q. Per-construction writeups: `proto0_C1_primary_target_frame_transp.md`,
`proto1_C2_mechanism_stringent_variant.md`, `proto2_C3_control_owner_mandated_re_e.md`.
All numbers below grep-verified from the on-disk run logs (`proto0_a.log`,
`out_proto1_C2_loo_3x3.log`, `out_proto1_C2_fourtails_{3x3,6x6}.log`,
`out_proto1_C2_probe_angles{,_6x6}.log`, `out_proto1_ladder_wrap_ab.log`,
`proto2_out_c3_3x3.log`, `proto2_out_c3_diag.log`) — none taken from summaries or
notifications (per the 2026-07-17 phantom-table incident in KNOWN_SANDBOX_ERRORS.md).

---

## 1. Verdict table

"phys" = gap-window B-block relF, median over the 9 LOO q0 (max in parens); exciton = swap
shift of lowest-4 TDA states. Target = few-percent B / ~meV exciton.

| construction | null-test | phys on-grid | phys off-grid | phys rank-cut (on TRUE data) | tile diag | vs few-percent target | verdict |
|---|---|---|---|---|---|---|---|
| **C1** target-frame transported V^SR interp (Method A + Strategy B) | gates only — full LOO chain never ran (aborted at psi_full_y regauge gate) | — not landed | — not landed | — | — | no data | **INCOMPLETE** — conventions + gates landed (production disk-match 1.9e-15, alpha-invariance 1.8e-14, leverage bound 0.189<=1); headline pending one identified fix |
| **C2** global periodic frame + four-tails + transported-Phi interp (Method B + Strategy A) | **PASS** 4.4e-14 | 0.96 (5.3); exciton 36 meV | terminated by pre-agreed four-tails rule (mechanism dead at 3x3 AND 6x6) | r=480/320/160/80 → 1.3e-3 / 2.3e-3 / 9.5e-3 / 4.2e-2 | 4.4e2 | fails by ~30x; 200x WORSE than plain rankcut ladder | **NEGATIVE — clean kill** of the counterproposal by its own sec-10C criterion |
| **C3** bounded-spectrum rank-r solve on interpolated C/Z (§3.5 re-base) | **PASS** 6.6e-13 (B) / 1.9e-10 (tile); §3.5 continuity exact | best row 1.14–1.19 (7.8–8.4); exciton 18 meV — **in the §3.5 unwrapped-q convention** | — not run | κ1e6/1e4/1e2 → 7.6e-4 / 3.3e-3 / 5.1e-2; exciton 0.01 / 0.05 / 5.5 meV | 1.00 at window | as-run: fails by 20–50x | **NEGATIVE as scheme**; delivered the re-based bar + junk-inertness proof; its INTERP rows superseded by the wrapped-labeling re-score (below) |
| **derived: plain rankcut ingredient interp, production (BGW-wrapped) labeling** — the §3.5 ladder re-scored inside C2's null-tested harness | inherits C2 chain PASS (4.4e-14) + solve/to_sphere commutation 2.3e-14 (C3) | **rankcut 1e-4: 4.7e-3 (3.2e-2)**; 1e-2: 4.4e-2, exciton 5.4 meV; raw 0.26 (3.1e2) | **PENDING — the decisive test** (3x3-subgrid → 6x6 complement with truth) | same as C2 column (floor ~2–3e-3 at matched rank) | 1.00 (tile stays destroyed — and physically irrelevant) | **MEETS target ON-GRID (~0.5%)** | **OPTIMISTIC — the campaign's surviving candidate**, pending off-grid confirmation |

Si 4x4x4 negative control: **not run by any construction** (C1 aborted before its
`proto0_c_negctrl.py` stage; C2/C3 were MoS2-only). The overfitting alarm never fired because
no construction reached a claimable pass; the control transfers to the follow-up.

## 2. Apples-to-apples audit

**Null tests.** No prototype's null test failed, so no numbers are void on that criterion:

- C2: full chain (links + gauge + frames + dressing), no interp, r=640 → B relF 4.4e-14. PASS.
- C3: true-C/true-Z through the full solve+metric chain → B 6.6e-13, tile 1.9e-10
  (machine-level at cond 1.8e7), plus exact reproduction of the logged §3.5 tile/randpair
  columns (3.7e6/1.2e4/1.1e1/1.00 and 7.4e4/1.5e3/2.0e1/0.89). PASS.
- C1: never reached its LOO chain — its gate battery passed (17 identities at 1e-12..1e-16,
  incl. on-grid alpha-invariance of the SR/LR split across the full alpha ladder), but C1
  has **no headline numbers to accept or void**. Its report's §4 "BAR" paragraph quotes C3's
  table and therefore inherits C3's convention (next item).

**The one apples-to-apples defect found — and resolved: the q-labeling convention.**
C2 and C3 ran the same §3.5 interp ladder (same nR=7 stencil, same 9 LOO q0, same truth =
stored full-rank fit, same slab kernel, same 81-row metric window) but in different q-label
conventions:

- `zeta_q.h5:mf_header/rk` is UNWRAPPED (0, 1/3, 2/3) while the stored zeta spheres and
  coefficients follow the BGW wrap (2/3 ≡ −1/3). Production labeling is **wrapped** — proven
  twice independently: C1's gate rebuilds the disk `V_qmunu` at 1.87e-15 (all 9 q) only with
  wrapped q + blat-scaled bvec (`v_q_g_flat.py:232`, `gw_init.py:292`); C2's `make_Vq`
  matches disk at ≤1.3e-9 at all q wrapped and breaks at 0.6–0.8 unwrapped at the 5
  wrap-affected q's. C3's own diag confirms from the other side: its (unwrapped-built) tiles
  match disk only at the 4 wrap-unaffected q's {0,1,3,4}+Γ.
- The unwrapped continuation puts a spurious `e^{iG0·r}` on 5 of 9 training fields, which
  scrambles the interpolation stencil, and mis-assigns Coulomb weights (|q+G|² up to 53 Ry
  on a 30 Ry sphere). Controlled A/B, same q0=2, same rankcut-1e-4 solve, same truth
  (`proto1_ladder_wrap_ab.py`): **B relF 4.5e-3 wrapped vs 0.70 unwrapped — 155x.**

Consequences, applied throughout this report:

1. C3's INTERP rows stand as the faithful re-base of §3.5 **in §3.5's own convention**
   (that was its assigned deliverable; continuity exact) but are NOT production-physical
   numbers at 5/9 LOO points. The governing physical ladder is C2's wrapped re-score.
2. C3's TRUE-truncation rows are convention-robust — independently reproduced by C2's
   wrapped whitened-rank ladder at the same magnitudes (e.g. r≈286/320: 3.3e-3 vs 2.3e-3;
   r=81/80: 5.1e-2 vs 4.2e-2). The junk-inertness finding survives both conventions.
3. C3's diagnostic claim that the stored per-q fits carry TRS-non-covariant solver junk
   (tile TRS violations 0.5–3.8) is **VOID** — it was the wrap trap. With wrapped labels the
   stored-fit tiles TRS-pair at 1.0e-15 (C2 gate; C1 gate3 idem). The junk tail is inert
   AND symmetry-clean. The companion corollary — the full-rank tile is ~100% junk by
   Frobenius — survives in both conventions (wrapped: tile relF 1.00 at rankcut 1e-4 while
   B = 0.47%).

## 3. §3.5's no-window conclusion: **FALLS** under the physical metric

Two corrections were each necessary and only jointly sufficient:

| §3.5 ladder row | §3.5 verdict variables (tile / random-pair) | + physical metric only (C3, §3.5 labeling) | + physical metric AND production labeling (C2) |
|---|---|---|---|
| raw | 3.7e6 / 7.4e4 | B 1.4e3, exciton 65 meV | B 0.26 (3.1e2) |
| tikhonov 1e-6 | 9.9e5 / 2.9e4 | B 4.0e2 | B 5.7e-2 |
| rankcut 1e-4 | 1.1e1 / 2.1e1 | B 1.19, 18 meV | **B 4.7e-3 (3.2e-2)** |
| rankcut 1e-2 | 1.00 / 0.89 | B 1.14, 18 meV | B 4.4e-2, exciton 5.4 meV |
| zeta-direct (skip solve) | phys 0.17→4.87 growing with nR | — | B 7.0e-2 |

- Physical metric alone (C3): the failure softens ~10³ (3.7e6 → 1.2) but "no window"
  apparently persists — because 5/9 metric points were computed in a non-production Coulomb
  labeling with seam-scrambled training fields.
- Physical metric + wrapped labeling (C2): **a genuine regularization window exists** —
  rankcut ~1e-4 gives **0.47% median / 3.2% max** on-grid LOO while the tile is still 100%
  destroyed (relF 1.00), flat-ish through 1e-2 (4.4%), raw explodes. Exactly the scenario
  the owner flagged: the ill-conditioned tail lives in physically inert directions, and
  tile-Frobenius/generic-d contractions over-weighted junk (§3.5's "phys 0.89" random-pair
  column included — it was junk-weighted).
- Adjudication: the physical metric must be computed with the production Coulomb labeling
  (it is the observable the BSE consumes; the §3.5-convention "metric" weights channels no
  production run uses). Therefore **§3.5(3)'s "no regularization window" and §3.5(4b)'s
  "Z-interp dominates" FALL**. What survives of §3.5: the conditioning-dominated rows
  (raw / rankcut ≤1e-6 genuinely fail in every convention), the C_R-falloff premise, the
  master-zeta kill (§3.2, untouched), and the zeta-direct rejection (7% — better than its
  old story but still 15x worse than the rankcut ladder).
- Scope guard: this is **on-grid only**. §3.5(2) measured C_q off-grid interp ~30x worse
  than on-grid (4.0e-2 vs 1.3e-3 at 4x4). If that factor carries to the B-metric, off-grid
  could land ~10–15% — marginal. The 3x3-subgrid → 6x6-complement off-grid-with-truth test
  is the single decisive missing measurement.

**Owner pushback: CONFIRMED quantitatively, by both C2 and C3 and in both conventions.**
Truncating TRUE ingredients at κ1e6 destroys 90% of the tile Frobenius norm and moves the
gap-window B-block 7.6e-4 and excitons 0.01 meV. The exchange tile is physically
rank-compressible 640 → ~160–320 (1% → 0.2% B) — useful for storage/solve economy
independently of interpolation.

## 4. Mechanism story — does the half-inverse / parallel-transport thesis survive?

**No.** The counterproposal's central objects are measured absent, by its own criterion:

- **Four-tails (response sec-10C, its pre-agreed falsification):** transported Phi~_R does
  not decay toward C_R — it is *rougher than raw zeta*. 3x3 shells (|R| = 0/5.98/10.36
  Bohr): C_R 1.0/2.3e-2/6.7e-4; raw zeta_R 1.0/0.39/0.16; Phi_R (whitened, untransported)
  1.0/1.37/0.99; **Phi~_R (whitened + covariant transport + exact sewing) 1.0/1.89/1.60**.
  6x6 replicates to 26 Bohr: Phi~_R 1.63 at shell 1, plateau 0.18–0.51 vs C_R
  5.2e-4 → 4.1e-5. Whitening makes locality *worse*; transport makes it worse again.
- **Why:** the whitened frame bundle is eigenvector chaos of a gapless spectrum. C_q has no
  spectral gap anywhere (smooth decay, all 640 directions resolved, cond ~1.8e7), so the
  eigenframe rotates wildly between adjacent q even though C_q itself is smooth:
  adjacent-q whitened principal cosines sit AT the random-subspace floor without transport
  (median 0.098 vs floor 0.105 at 3x3; 0.054 vs 0.053 at 6x6) and reach only 0.20–0.33 with
  proven-covariant transport (a smooth bundle needs 1−O(dq²)); plaquette holonomy is at the
  random-unitary ceiling (30.5 vs 35.8), Wilson eigenphases pinned at ±π, and densification
  3x3 → 6x6 does not converge any of it. Exact seam sewing changes nothing (the collapse is
  physics, not bookkeeping); det-W winding = 0 (not topology). Benzi–Boito–Razouk decay
  never applied (its gap/κ hypotheses are exactly what the fixture violates — as the
  adjudication's rejected-list predicted).
- **End-to-end confirmation:** C2's transported-Phi interpolation lands B = 0.90–0.98 —
  equal to the old §3.5 rankcut-1e-2 bar, 200x worse than the plain rankcut ladder it was
  designed to beat, with excitons at 36 meV vs the ladder's 5.4 meV.
- **The correct core insight was elsewhere.** The smoothness that exists lives in the
  frame-free quadratic ingredients (C_q, Z_q — C_R decays, C_q LOO 0.13%), not in any
  frame, section, or half-inverse object. The winning scheme interpolates the ingredients
  and does one rank-truncated solve **in the target's own frame** — no frame is ever
  interpolated or transported, so the eigenvector chaos never enters, and the junk
  directions the truncation discards are physically inert (owner's point). C3's
  "non-interpolable physical-sector rotation" reading (370x above truncation floor at
  matched rank) was an artifact of the seam-scrambled training fields: in production
  labeling, interp error at matched rank (4.7e-3) is only ~1.5–2x the truncation floor
  (2.3–3.3e-3).
- **What survives of the response** (validated, reusable): the physical-action test
  variable (its sec-10E anticipated the owner's metric); the exact-target dressing algebra
  with bounded leverage (C1 gate: rowmax(l)=0.189 ≤ 1 in the response's flavor — the
  conjugate flavor breaks the bound at 2.77, a useful convention tell); fixed-rank spectral
  truncation as the regularization family (C1-C3 ladder design); "never re-dress with the
  full inverse"; and the four-diagnostic falsification battery itself, which killed its own
  parent cleanly and is worth keeping as methodology.

## 5. Next steps

### 5a. Optimistic branch — the surviving candidate (rankcut ingredient interp, production labeling)

1. **The decisive test (blocking everything else):** off-grid-with-truth — interpolate
   C/Z from the 3x3 subgrid of the 6x6 fixture to the 27 complement q's, rankcut solve,
   score B-block + exciton swap against the stored 6x6 truth, wrapped labeling throughout
   (`sphere_max|q+G|²−cutoff == 0` asserted). Harness exists: `proto1_C2_loo.py` machinery +
   `proto0_b_mos2_6x6.py` fixture wiring (fixtures `interp_study/mos2_6x6/lorrax/tmp/`).
   Also 6x6 on-grid LOO for the density trend, and the missing exciton number at rankcut
   1e-4 (only the 1e-2 row, 5.4 meV, is logged). Pass bar: few-percent B, ~meV exciton.
2. **Si 4x4x4 negative control** (`isdf_tensors_792.h5`,
   `runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16/work_sym/tmp/`): expected to FAIL
   (C_q LOO already 33% there); a pass = overfitting alarm on the harness.
3. **C1 closeout, re-pointed:** apply the one-line Bloch-phase fix
   (`prep.build_regauged_fields`: psi_full_y hypothesized = u·e^{2πik·r_μ}; k=0 matches at
   3.3e-16, diagonal-in-r phase unfixable by band rotation — consistent with C2's 29–40%
   span residual) and rerun `proto0_a` to close the gate. This settles the psi_full_y
   provenance trap (KNOWN_SANDBOX_ERRORS 2026-07-17 item 2), enables the exact-LR `F_p`
   channel and the absolute ISDF-fit-floor closure (C2's "fit floor 1.6" is currently
   uninterpretable). C1's transported-V^SR headline itself is deprioritized: its
   interpolable object lives on the bundle C2 measured dead (Vc~^SR_R tail 0.56–0.63 vs
   C_R 2.3e-2, alpha/taper-insensitive).
4. **Metric hygiene for sub-percent claims:** Kramers-clean windows (2v×2c or 4v×4c — the
   spec's 3v×3c splits doublets at both edges, flagged by C3), and the LR channel at true
   off-grid Q must be the analytic finite-α Gaussian over a fixed G-superset + taper
   (constraints (c)–(d); C1's C-4 note) — at stored-sphere q the sphere-limited LR is
   truth-consistent, off-grid it is not.
5. **Production mapping (owner sharding note), gated on the off-grid pass.** All dense ops
   are N_mu² objects sharded `P('x','y')` per the zeta-fit conventions (scan-inside-
   shard_map for the k-slab accumulations; C_R/Z_R stencil combination is trivially
   sharded). The distributed-linalg FFI entry points that apply (`src/ffi`):
   - `ffi/cusolvermp/eigh.py::distributed_eigh` — the Hermitian eigendecomposition of the
     interpolated `C_Q` (the rankcut solve's core op); CPU-host twin
     `ffi/slate/eigh.py::distributed_eigh` (host platform live since 2026-07-10).
   - `ffi/cublasmp/batched.py::batched_distributed_gemm` — reduced-inverse application
     `U_r (λ_r^{-1} U_r^H Z_Q)`, dressing/contraction GEMMs, per-k cross-Gram accumulations.
   - If a Tikhonov/full-solve rung is ever wanted: `ffi/cusolvermp/batched.py::
     batched_distributed_cholesky` + `batched_distributed_potrs` (LU fallback
     `batched_distributed_solve_lu`); SLATE twins `ffi/slate/cholesky.py::
     distributed_cholesky`, `ffi/slate/trsm.py::distributed_trsm`,
     `ffi/slate/batched.py::batched_distributed_{cholesky,trsm}`.
   Gates before production wiring: on-grid non-regression (V^SR+V^LR = V, bit-level vs
   stored tiles), wrap self-check everywhere q touches a sphere/phase/kernel, the
   physical-metric LOO at few-percent on 1-GPU MoS2 fixtures (no 16-GPU gating), and
   alpha-invariance on-grid.
6. Production default until the off-grid test passes: **per-Q ζ refit unchanged.** The
   rankcut-interp scheme is a measured few-percent *on-grid* fallback, nothing more yet.

### 5b. Negative branch — what PART III of the PRIMER should say (so this is never re-derived)

Add to `ARBITRARY_Q_PRIMER.md` PART III (as §III.5, with a pointer patch in §III.4):

1. **Frame/section-transport schemes are measured dead.** Any scheme that interpolates or
   transports eigenframe-carrying objects (whitened Phi = S R^H ζ, spectral projectors,
   C^{−1/2}-class half-inverse objects, global BZ-periodic frames, path-ordered links)
   fails because the C_q eigenframe is chaotic (gapless spectrum, adjacent-q whitened
   subspace overlaps at the random floor 0.098/0.054 at 3x3/6x6, holonomy at the random
   ceiling, transported R-tails rougher than raw zeta: 1.89/1.60 vs C_R 2.3e-2/6.7e-4),
   and this is invariant to exact sewing, covariant transport, gauge protocol, and grid
   densification. Cite `primer_response_study/` for the full battery. Do not re-attempt
   with a new frame convention; the failure is the frame concept.
2. **The verdict variable is the physical pair-amplitude contraction** (owner ruling
   2026-07-17): gap-window `B = M^H V_Q M` + exciton swap shift. The tile is ~100%
   inert junk by Frobenius (κ1e6 truncation: 90% of tile norm, 7.6e-4 of B, 0.01 meV);
   tile-Frobenius and random-d contractions are void as bars.
3. **§II.3(b)/§III.4(3)-(4) are superseded on-grid:** with the physical metric AND the
   BGW-wrapped q-labeling (the `mf_header/rk` unwrap trap, KNOWN_SANDBOX_ERRORS
   2026-07-17, worth 155x on the ladder), rankcut ~1e-4 ingredient interpolation achieves
   0.47% median / 3.2% max on-grid LOO. The C^{−1}-amplification/no-window result was a
   compound artifact (junk-weighted metric + seam-scrambled training fields). Ingredient-
   side interpolation is re-licensed pending the off-grid-with-truth test; per-Q refit
   remains the default until that lands.
4. **Fixture traps that any future study must respect:** the rk unwrap trap and the
   psi_full_y band-span trap (both in KNOWN_SANDBOX_ERRORS 2026-07-17). Constraints
   (a)–(d) unchanged; the LR channel stays analytic in closed form.

## 6. Caveats and open items

- The 0.5% headline is **on-grid LOO at 3x3 only**; off-grid and 6x6 physical numbers do
  not exist yet. Do not quote it as an off-grid capability.
- Exciton swap shift at the rankcut-1e-4 optimum not logged (5.4 meV at 1e-2 bounds it
  loosely); truth for all B-metrics is the stored full-rank fit (relative accuracy) — the
  absolute ISDF fit floor is open pending the psi_full_y fix.
- Si negative control never ran; window splits Kramers doublets (shared by both sides of
  every swap, so comparisons stand; sub-permille absolute claims need clean windows).
- C1's report §4 quotes C3's (unwrapped-convention) bar numbers; superseded by §3 here.
- Allocation JID 56052603 ended before C1's rerun; no standing allocation as of synthesis.

## 7. File index

- This report: `CAMPAIGN_REPORT.md` (synthesis; supersedes the per-construction bars).
- C1: `proto0_C1_primary_target_frame_transp.md`, `prep.py`, `proto0_{a,b,c}_*.py`,
  `proto0_run.sh`, `proto0_a.log` (gates PASS; aborted at regauge assert line 225).
- C2: `proto1_C2_mechanism_stringent_variant.md`, `proto1_prep.py`,
  `proto1_C2_{fourtails,probe_angles,loo}.py`, `proto1_ladder_wrap_ab.py`, logs/npz
  (`out_proto1_*`, `proto1_C2_*.npz`).
- C3: `proto2_C3_control_owner_mandated_re_e.md`, `proto2_c3{,_diag}.py`,
  `proto2_out_c3_{3x3,diag}.log`, `proto2_c3_3x3.npz` (SLURM step 56052603.11).
- Campaign summary appended to
  `reports/bse_refactor_map_2026-07-15/archive/designs/arbitrary_q_bse.md` §10.
