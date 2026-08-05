# Multipole GW: current state and restart assessment

Date: 2026-07-21

Scope addendum: the subsequent regulator-dressed MPA exploration explicitly does
not use the BSE W(omega) Lanczos chain. See
`reports/multipole/REGULATOR_DRESSED_MPA.md` for the distinction between ordinary
double-parallel MPA and a width-dependent model fitted to LORRAX's regulated
real-frequency W.

## Summary

The multipole initiative is still **design-only**. There is no `ComputeMode.MPA`,
MPA fit module, staged frequency cache, MPA head path, or MPA test in any current
LORRAX checkout/worktree. The reviewed July 6 plan is nevertheless directionally
sound: implement **MPA-W**, preserve the existing low-scaling CTSP self-energy
machinery, and defer MPA-Sigma/MPA-G until spectral functions are required.

The plan is not ready to execute linearly as written. It predates the July GW
refactor, the W(omega) block-Lanczos oracle, the HGL crossing-conditioning fix, and
the rank-revealing charge-zeta fix. It also leaves several correctness contracts
underspecified: how complex pole damping composes with the sigma regularization,
how failed/partial pole fits preserve Wc(0), how pole ordering avoids pathological
window widths, and how the q->0 head is sampled without silently reusing a static
or single-probe value.

**Recommendation for ordinary complex-sample MPA:** make the first checkpoint a
fit and sampling proof against selected MoS2 W(omega) columns before wiring a
production `ComputeMode.MPA`. **Recommendation for the subsequent RegMPA scope:**
do not use the W-chain; fit direct regulated real-frequency W samples and validate
the resulting regulated G-times-W contraction against a small explicit
band/pair-denominator sum. The two checkpoints test different mathematical models.

## Artifacts found

| Artifact | State | Use |
|---|---|---|
| `reports/multipole/MPA_IMPLEMENTATION_PLAN.md` | reviewed design, no code | Primary MPA-W architecture and equations |
| `reports/multipole/REGULATOR_DRESSED_MPA.md` | design note, no code | Alternative real-frequency, width-dependent regulated modal model; contrasts it with ordinary complex-sample MPA |
| `reports/multipole/Multipole-W-metals.md` | paper text | MPA-W sampling, metals/intraband extension |
| `reports/multipole/multipole-sigma-2025.md` | paper text | MPA-Sigma/MPA-G follow-on; correctly deferred |
| `sources/lorrax_A/docs/theory/physics.md` section 6.9 | current production theory | Single-pole GN dynamic-Sigma data flow |
| `sources/lorrax_A/docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` | partly stale names, useful equations | Current four-branch/window conventions |
| `sources/lorrax_A/docs/dev/archive/ctsp_revised.md` | archived design | Multi-pole CTSP derivation; not production code |
| `sources/lorrax_A/src/bse/w_omega_chain.py` | implemented on `agent/bse-phase2` | Arbitrary-complex-frequency W-body oracle for selected columns |

No other multipole implementation or active MPA branch was found. Generic
`PpMultipoles` symbols under `src/psp/` are UPF/PAW schema fields and unrelated.

## Current implementation surface

| Capability | Current state | Consequence for MPA |
|---|---|---|
| Dynamic Sigma | GN/HL PPM is split across `ppm_pipeline.py`, `ppm_sigma.py`, `ppm_windows.py`, `ppm_tau_kernel.py`, and `ppm_accumulators.py` | The old plan's `ppm_sigma.py` line map and proposed 150-line lift are stale; the split provides a cleaner reuse boundary |
| Screening requests | `screening_requests_for` returns static plus at most one probe; `compute_screening` rejects off-axis complex frequencies | MPA needs a separate staged producer, not a `W_by_role` dict containing all 2*n_p matrices |
| Off-axis chi0 | No production quadrature for `Re(z) != 0` and `Im(z) != 0`; `compute_chi0` and `precompile_chi0` cast `quad.alpha` to float64 | Complex weights are silently lost until both host folds are fixed and tested |
| Complex poles in Sigma | The tau kernel accepts complex arithmetic, but `_prepare_sigma_state` replaces Omega with `max(Re(Omega),0)` and windows compare that real array | The reviewed two-array contract (real window energy plus complex phase pole) is still required |
| q->0 head | `HeadResolver` accepts arbitrary complex z through `s_tensor`; `compute_S_omega` already vectorizes an omega array | Build one batched MPA head sampler; do not call the generic fallback blindly |
| Sharded disk I/O | `file_io.slab_io.SlabIO` is the canonical large-array path | MPA sample/pole caches must use SlabIO and logical-mu clipping, not ad hoc rank-0 HDF5 files |
| Full-frequency W oracle | BSE block-Lanczos chain evaluates body columns at arbitrary complex z and closes to stored W(0) | Use it to validate MPA sampling/fit on selected columns; it is not a drop-in dense-W producer |
| BGW FF reference | Only a Si `frequency_dependence 2` epsilon two-point experiment exists; there is no BGW full-frequency sigma baseline | A new immutable BGW variant is required before the final cross-code gate |

## What remains correct in the July 6 plan

1. **MPA-W is the right first scope.** LORRAX already outputs Sigma_c(omega) and
   linearized/on-shell QP quantities. MPA-Sigma and MPA-G add value mainly for
   analytic spectral functions and can remain separate.
2. **The pair-density and projection path is pole-count agnostic.** Each pole
   contribution can reuse the existing G-times-W time-domain contraction.
3. **Memory must be flat in pole count.** Samples and pole slabs should be staged;
   never retain all W(z_i) or all poles in device memory.
4. **GN is the n_p=1 anchor.** The fit and full Sigma path need independent n_p=1
   checks before multi-pole convergence is meaningful.
5. **The head is a separate scalar channel.** It must be fit on the same sample
   grid and added exactly once after the body pole sum.
6. **Start with a gapped one-shot system.** MoS2 3x3 is the portable first gate;
   metals, self-consistent MPA, and MPA-Sigma/G remain later work.

## Corrections needed before implementation

### 1. Establish one source baseline

There is no unified current branch. `main` is at `6bd4dc9`; `agent/bse-phase2`
contains the W(omega) chain; `agent/gw-ppm-sigma-reg` contains the HGL bandwidth
floor (`d011a36`); and `agent/gw-rank-truncation` contains that floor plus the
mesh-invariant/rank-revealing charge-zeta work (`23af6b9`). The GN fit-classification
fix (`218aeb8`) exists in `lorrax_D`, not `lorrax_A`.

Create an absolute-path `lorrax_A` worktree on `agent/mpa-w` from the stabilized GW
tip, after deciding which unmerged GW fixes are prerequisites. Do not work in the
currently dirty `sources/lorrax_A` BSE checkout. The W-chain can initially remain a
separate oracle rather than forcing a merge of the full BSE phase-2 branch.

### 2. Prototype the fit before driver plumbing

The proposed scaled Padé-in-x solve is plausible but not yet demonstrated on noisy
ISDF W samples. The first artifact should fit selected q/mu/nu elements exported
from `w_omega_chain` and score reconstruction at held-out complex frequencies.

Required fit contracts:

- scale x=z^2 and report matrix rank/condition/residual;
- use batched linear algebra over chunks (not a Python loop per element);
- sort poles per element by `(Re Omega, Im Omega)` before writing pole slabs, so a
  pole index has a coherent energy range and does not inflate every sigma window;
- constrain time ordering, then refit residues whenever poles move;
- test reconstructed W, not only recovered roots, because pole/residue labels are
  non-unique and near-canceling pole-zero pairs can look correct by root matching;
- verify static identity, held-out error, diagonal loss sign, and the relevant
  q/mu/nu conjugacy/Hermiticity relations.

The plan's sample-grid pseudocode still defaults to `im_near=0.1` and `im_far=1.0`
while labeling them Ry; the reviewed values are 0.2 and 2.0 Ry. Also, the general
double-parallel grid does not automatically produce the promised n_p=1 GN pair.
Special-case n_p=1 to exactly `{0, 2i Ry}` for the anchor.

### 3. Define residual and invalid-pole physics

The GN `ppm_invalid_mode=static_limit` implementation is element-level. An MPA fit
can have some usable poles and some repaired/dropped poles. The implementation must
define the residual

`Delta Wc(0) = Wc_sample(0) - Wc_from_retained_poles(0)`

and either reject/refit the whole element when it exceeds tolerance or add one
static residual contraction without double counting the retained poles. Reusing
GN's full-element static correction per failed pole would be wrong. The `2ry`
fallback has no meaningful multi-pole analogue and should be rejected in MPA mode.

### 4. Prove the complex-pole Sigma quadrature

The tau phase itself can carry complex Omega, but current window planning assumes
real pole energies and the HGL crossing path also applies a configurable/floored
regularization width. Complex MPA poles already contain intrinsic damping. Before
claiming that the existing driver is unchanged, compare one and several complex
poles against direct analytic denominator sums over a small synthetic band/pair
basis, across Laplace and crossing windows. This test must determine whether the
current HGL broadening composes correctly with Im(Omega), or whether the crossing
target/accumulator needs an MPA-specific adjustment.

The source-side change is still the reviewed two-array threading:

- `Omega_window = Re(Omega)` (real values for masks, min/max, and references);
- `Omega_phase = Omega` (complex values passed to `_build_W_t_q`);
- validity based on `abs(Omega)`, not `Re(Omega)`.

Merge the HGL conditioning floor before this work; otherwise existing crossing
ill-conditioning can swamp the MPA comparison.

### 5. Make Stage A a true staged producer

`screening_requests_for(MPA)` should return only the static request used by the
normal driver/restart path. A dedicated MPA screening stage should iterate samples,
compute chi0/W, and write each sharded Wc slab immediately. Adding 2*n_p normal
requests and then bypassing `compute_screening` is contradictory and would retain
all W matrices in `W_by_role`.

Use one provenance-rich cache file (or an equivalently structured cache directory)
through `SlabIO`, with sample frequencies, logical/padded dimensions, source commit,
fit settings, completion flags, and restart validation. Stage B should write
pole-major slabs so Stage C reads one pole at a time.

### 6. Give the head an MPA-specific sampler

The generic `HeadResolver` is unsafe as-is for a multi-sample fit:

- any nonzero frequency selects the single `whead_imfreq` override, so every MPA
  sample would silently receive the same value;
- `epshead` is static-only and deliberately reuses epshead(0) at nonzero omega.

For the initial implementation, require `wcoul0_source=s_tensor`, read `dipole.h5`
once, call vectorized `compute_S_omega` on the full MPA grid, fit the scalar head,
and sum its analytic per-pole contribution once. Explicit multi-frequency head
overrides can be designed later.

### 7. Generalize the post-refactor Sigma driver, not the old monolith

The stream file is now opened once in `compute_sigma_c_ppm_omega_grid`, while branch
execution is already isolated in `_run_sigma_branch`. Generalize the outer driver
to own one output sink and iterate pole slabs around the existing four branches.
Do not invoke the whole PPM pipeline per pole: that would truncate streamed output,
reapply invalid-static terms, refit/reapply the head, and repeat final writers.

The initial MPA mode should explicitly reject self-consistent QP mode and metals.
One-shot DFT and the post-hoc fixed-point QP solve can share the completed Sigma grid.
Bispinor support should be enabled only after a dedicated regression, even though
the charge-body contraction is structurally reusable.

### 8. Repair the validation plan

- The reference to positional column 8 in `sigma_freq_debug.dat` is stale. Use the
  header-driven parser in `skills/compare/SKILL.md` and select
  `sig_c(Edft).Re` by name.
- Add direct chi0(z) and W(z) checks before fitting; a good Sigma result can hide
  compensating fit/integration errors.
- Add the W-chain held-out-frequency oracle for selected body columns.
- Create a new MoS2 3x3 BGW `frequency_dependence 2` epsilon+sigma variant. The
  existing Si `03_bgw_full_freq_2pt` directory has only epsilon at 0 and 200 Ry
  and is not a full-frequency Sigma reference.
- Keep the n_p Cauchy plateau, 1-vs-4 GPU, head-on/head-off, and sum-rule gates.

## Revised implementation order

1. **Baseline:** create `agent/mpa-w` worktree from the stabilized GW branch; merge
   or port the required GN/HGL/zeta fixes; run current golden gates unchanged.
2. **Fit/sampling proof:** use the existing MoS2 W-chain fixture to export selected
   Wc(z) columns; implement a host prototype and compare n_p={1,2,4,8}, sampling
   layouts, and held-out errors. Decide the production rational solver here.
3. **Fit module + cache schema:** production chunked fit, pole sorting, constraints,
   residue refit, residual-static policy, SlabIO round-trip tests.
4. **Noncrossing Stage A:** implement and directly validate complex-target chi0
   quadratures; fix both complex-alpha casts; stage far-line plus below-gap samples.
5. **Complex-pole Stage C:** thread window/phase Omega arrays, generalize the outer
   Sigma accumulator, add the scalar MPA head, and prove analytic small-system parity.
6. **First end-to-end gate:** n_p=1 vs GN, then n_p convergence on MoS2 3x3; compare
   body W against the chain and total Sigma against a new BGW FF reference.
7. **Interior sampling only if required:** add the global complex-time crossing chi0
   kernel and repeat the sampling adequacy A/B. The paper-based expectation is that
   some interior near-line samples will be required, so this is likely production
   work even though it remains gated by evidence.
8. **Expansion:** 1-vs-4 GPU, Si 3D, then bispinor/FM CrI3; metals and SC MPA last.

## First checkpoint definition

The initiative is ready for production wiring when a standalone artifact shows:

- W-chain truth for selected MoS2 q/columns at the exact proposed sample and
  held-out grids;
- stable n_p={2,4,8} fits with reported condition/rank/constrained-pole fractions;
- Wc(0), diagonal loss sign, and conjugacy gates passing;
- an explicit answer on far+below-gap versus full double-parallel sampling;
- no dependence on exact pole ordering or root matching.

Until that checkpoint exists, the project is **planned and well motivated, but not
implementation-ready**.
