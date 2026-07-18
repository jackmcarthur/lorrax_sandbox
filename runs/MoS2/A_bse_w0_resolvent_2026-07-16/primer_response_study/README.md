# primer_response_study — SCRATCH LEDGER

**What this directory is.** The prototype battleground of the arbitrary-Q
interpolation campaign (2026-07-17): three counterproposal constructions
(C1/C2/C3), the off-grid follow-up, the owner-spec tile schemes (F-scheme),
and the compact-LR-representation study (b26p). The campaign's verdicts live
in `CAMPAIGN_REPORT.md` and `arbitrary_q_bse.md` §§10–13
(`reports/bse_refactor_map_2026-07-15/archive/designs/`); the presentable
summary is `F_SCHEME_NOTE.html` (same directory).

**Where the winning pipeline lives NOW.** `REFERENCE_arbitrary_q_vq.py` —
the single self-contained reference implementation (Tikhonov clean →
Gaussian SR/LR split → SR-tile stencil + global b26p LR fit → closed-form
assembly at any Q), with its acceptance run pinned to the §13 numbers and
`test_reference_e2e.py` as the fast 3×3 smoke test. Read that file first;
everything else here is development history.

**Nothing in this directory is deleted.** The scratch scripts are the
evidence base for the campaign's measured refutations (frame-death, literal
moments, SVD compression, wrap traps…) and the logs are the grep-verified
provenance for every number in the reports (phantom-table rule,
KNOWN_SANDBOX_ERRORS 2026-07-17). Superseded ≠ wrong: it means "do not
extend this file; extend the reference implementation".

---

## Current (maintained)

| file | role |
|---|---|
| `REFERENCE_arbitrary_q_vq.py` | THE reference implementation. Three-stage pipeline as clean procedural functions, per-element math in docstrings, copy-with-attribution from the scratch scripts. Modes: `acceptance` (6×6 LOO vs §13 pins), `transfer` (§13.3 3×3-fit → 6×6-deploy). |
| `test_reference_e2e.py` | E2E smoke test on MoS2 3×3: prepare → fit → evaluate at held-out q → compare, machine-level nulls + pinned accuracy thresholds, ~10 s compute. |
| `REFERENCE_acceptance_6x6.log`, `REFERENCE_transfer.log`, `test_reference_e2e.log` | Disk provenance for the reference-impl validation numbers (untracked, like all logs here). |
| `README.md` | This ledger. |
| `CAMPAIGN_REPORT.md` | Campaign synthesis + §8 follow-up (authoritative verdict record; the file index in its §7 predates §§12–13 — this ledger supersedes it as the directory map). |

## In flight (second polish agent — do not touch)

| family | role |
|---|---|
| `stress_*.py` + logs/npz | §14 stress/edge campaign on the winning pipeline (α budget re-allocation; cleaning-ε / fit-ridge / stencil robustness; Γ-edge; negative controls; Si-3D probe — see each docstring). Owned by the parallel polish session; results will land in `arbitrary_q_bse.md`. |

## Shared prep libraries (imported by the evidence scripts; the winning-pipeline
parts are absorbed into the reference impl — extend THAT, not these)

| file | role | status |
|---|---|---|
| `proto1_prep.py` | The campaign's validated loader + conventions module: wrapped-q Fixture, recon/to_sphere, slab kernel, order-robust `build_Cq`, gap-window pairs, truncR weights, self-check battery. The conventions notes in its docstring (torus labeling, V orientation, blat) are still the fullest write-up. | absorbed into reference (loader/kernels/metrics); keep for the C2/§11–13 scripts that import it |
| `offgrid_prep.py` | `fix_sphere_wrap` (half-boundary trap), condensed gate battery, ladder solver, batched exciton machinery, `SiOldFixture` (3D control). | absorbed into reference (wrapfix/gates/Hdir); Si loader NOT absorbed (3D variant is future work) |
| `tile_prep.py` | §12 TileStudy: Z-free cleaning identity, gset/v_on_set, T2 moments, F-channels, assembly variants A–F. | stages 1/3 absorbed into reference; moment machinery evidence-only (refuted, §12.2) |
| `lr_prep.py` | §13 fit machinery: LRSamples, per-Gz weighted LSQ with per-q normal blocks (ChannelFit), SVD compression, sample-matrix rank. | stage 2 absorbed into reference; SVD/fine-lattice parts evidence-only |
| `prep.py` | C1's regauged-field builder (psi_full_y provenance work; hit the band-span trap). | evidence-only |
| `proto0_run.sh`, `proto1_run.sh` | module-free srun+shifter wrappers (`JID=<jid> ./proto1_run.sh python3 -u <script>`). | current — used by the reference/test runs too |
| `proto2_jid.txt` | dead SLURM JID record (56052603) of the C3 runs. | evidence-only |

## C1 family — target-frame transported V^SR (construction INCOMPLETE)

`proto0_a_mos2_3x3.py`, `proto0_b_mos2_6x6.py`, `proto0_c_negctrl.py`,
`proto0_probe.py`, `proto0_probe2.py`, `proto0_inspect.py`, `proto0_a.log`,
writeup `proto0_C1_primary_target_frame_transp.md`.
Gate battery passed (17 identities incl. the production disk-match at
1.9e-15 — the proof that wrapped labels are production labeling); LOO chain
never ran (aborted at the psi_full_y regauge gate → the band-span trap,
KNOWN_SANDBOX_ERRORS). Evidence-only.

## C2 family — global periodic frame / four-tails (clean NEGATIVE) + the harness that survived

`proto1_C2_fourtails.py`, `proto1_C2_probe_angles.py`, `proto1_C2_loo.py`,
`proto1_ladder_wrap_ab.py`, `proto1_fitfloor_diag.py`, `proto1_fitfloor_v2.py`,
`proto1_gauge_probe.py`, `proto1_vdisk_allq.py`, `proto1_wfn_span_probe.py`,
`proto1_zcheck_circular.py`, logs `out_proto1_*`, npz `proto1_C2_*`,
writeup `proto1_C2_mechanism_stringent_variant.md`.
The frame-transport mechanism kill (four-tails, holonomy at random ceiling —
§10; later NARROWED by §12.3 operator theory: eigenframes lawless, subspaces
smooth) AND the two load-bearing positives: the wrapped-labeling A/B
(155× — `proto1_ladder_wrap_ab.py`) and the null-tested LOO harness that
every later stage reused. Evidence-only.

## C3 family — bounded-spectrum rank-r re-base (§3.5 re-scored)

`proto2_c3.py`, `proto2_c3_diag.py`, logs `proto2_out_c3_*`, npz,
writeup `proto2_C3_control_owner_mandated_re_e.md`.
Delivered the physical-metric re-base of §III.4 and the junk-inertness
proof; its interp rows were superseded by the wrapped re-score
(CAMPAIGN_REPORT §2–3). Evidence-only.

## offgrid family — §11 owner-redesigned off-grid tests

`offgrid_mos2.py` (6×6 on-grid LOO + withdrawn subgrid leg),
`offgrid_path.py` (Γ→x̂ rank-cut trajectories), `offgrid_path_htr.py`
(htransform swap-H(t) physical anchors), `offgrid_si.py` (Si 4×4×4 negative
control — correctly FAILS off-grid), logs + npz.
Results: §11 (6×6 LOO pass, path smoothness pass, Si control pass;
midpoint ζ-refit truth still PENDING — the one measurement gating any
off-grid capability claim). Evidence-only; the 6×6-LOO arithmetic these
scripts pioneered is what the reference impl's acceptance mode re-runs.

## tile family — §12 owner-spec F-scheme constructions

`tile_t1t2_mos2.py` (A–F assembly head-to-head), `tile_smooth_filter.py`
(operator-theory checks A/B: Tik-filter continuity, subspace-vs-eigenframe
re-audit), `tile_path.py` (Γ→x̂ path for the tile schemes),
`tile_wannier_pair.py` (Wannier-gauge constructive attempt — blocked by
conduction-window entanglement), logs + npz.
Origin of the F-scheme (winner) and the measured kills: cleaning-alone,
literal T2 moments, mixed-split D, pure-3D z-Taylor. Evidence-only.

## lr family — §13 compact-LR (b26p) study

`lr_singlevalued.py` (seam parity + q-fiber decomposition),
`lr_fiber_source.py` (the fiber IS the hard cut → Tikhonov gauge mandate),
`lr_basis_ladder.py` (the basis ladder: b26p headline, SVD/gto/hybrid
kills; `--tik` flag = the winning gauge), `lr_transfer.py` (3×3-fit →
6×6-deploy, zero downstream loss), `lr_pin.py` (literal-moment pinning
refuted a second way), logs + npz.
`lr_basis_ladder_6x6_tik.log` and `lr_transfer.log` carry the §13 pinned
numbers that `REFERENCE_arbitrary_q_vq.py acceptance/transfer` reproduce.
Evidence-only; do not extend the ladder — new rungs belong in a stress
script or the reference impl.

---

## Traps any future work here MUST respect

1. **rk unwrap trap** + **half-boundary sphere centers** — always load via
   the reference impl's `load_fixture` (sphere-derived wrap) or
   `offgrid_prep.fix_sphere_wrap`. Worth 155× on physical metrics.
2. **psi_full_y band-span trap** — no WFN-content truth from the restart's
   psi alone (KNOWN_SANDBOX_ERRORS 2026-07-17).
3. **Verdict variable** is the gap-window B + exciton swap, never tile
   Frobenius (§10 ruling).
4. **Do not re-attempt** (measured dead): eigenvector-frame transport,
   literal/pinned moments, SVD "learned multipoles", multi-width GTO
   ladders without conditioning treatment, hard-cut cleaning in the fit
   gauge, trio windows for gauge construction, pure-3D multipoles on slabs,
   3×3→6×6 cross-grid-class off-grid designs (§11.0).
