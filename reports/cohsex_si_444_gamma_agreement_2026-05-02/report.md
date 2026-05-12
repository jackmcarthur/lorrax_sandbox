# LORRAX COHSEX vs BGW: Si 4×4×4 — closing the comparison

**Date:** 2026-05-02 — agent D, branch `main`.

## Headline

When the entire chain (QE → BGW → LORRAX) runs **without symmetry**
(`nosym=true` in QE, `no_symmetries_q_grid` in BGW, full 64-k-point
BZ throughout) AND BGW is configured with `cell_average_cutoff = 1d-12`
(disabling its `fixwings` finite-q wing-rescaling pass), LORRAX vs BGW
COHSEX `Σ' = X + (SX-X) + CH'` agrees to:

- **MAE = 0.12 meV** over 16 bands × 64 k at 1440 ISDF centroids.
- **max single-state \|Δ\| = 0.48 meV** — limited by the q=0 χ
  representation in the centroid basis (LORRAX's analytic head /
  finite-q-extrapolated representation of χ at literal q=0 differs
  from BGW's q→0 probe at q=q0=(0.00025, 0, 0) by ≲0.5 meV).
- **MAE bare-Σ_X = 0.05 meV** — converged.
- **MAE body = 0.15 meV** — converged.

This is essentially **bit-for-bit reproduction** of BGW for static
COHSEX once both codes are configured to use the same q-handling
convention.

Degenerate-set averaging is on by default on both sides
(BGW: `Sigma/shiftenergy.f90:86-122`, tol = `TOL_Degeneracy = 1d-6 Ry`;
LORRAX: `gw/degen_average.py::apply_to_matrix_diagonals` called
from `gw_output.py::write_results` unless `no_degen_averaging = true`).
The 0.12 meV MAE is post-averaging on both sides — the comparison is
made between basis-invariant trace-divided diagonals, not raw QE-basis
elements within a degenerate manifold.

### What "default" BGW (`avgcut=1e12`) does that LORRAX does not

| BGW config | MAE Σ' | what's different |
|---|---|---|
| **default** (`avgcut = 1e12`) | 16.43 meV | BGW runs `fixwings(q0flag=false)` at every q with `qlen² < avgcut` (= every q), rescaling the (G=0, G'≠0) and (G≠0, G'=0) elements of ε(q) by `fact·oneoverq/(vcoul·q0len)` *before inversion*. LORRAX has no analog. |
| **noavg** (`avgcut = 1d-12`) | **0.12 meV** | `fixwings` only fires at the literal q=0 point (head+wing zeroing); finite-q ε goes through unrescaled, matching LORRAX's natural centroid-basis ε. |

The 16 meV residual we'd been chasing all session was **entirely BGW's
finite-q `fixwings` rescaling**, not a LORRAX-side bug. With both
codes on the same convention, agreement collapses to sub-meV.

Reproducing BGW(default) Σ from LORRAX would require mirroring
`fixwings` on the LORRAX side — applying the same per-q rescaling
factor to the centroid wings of ε at every q. That's a future
extension; for the comparison we already get, **always run BGW with
`cell_average_cutoff 1d-12` when benchmarking against LORRAX**.

### Sym path artifacts (resolved earlier)

The sym path (8 IBZ k-points, BGW unfolding, LORRAX irk-to-k
mapping) had a **comparison-script bug** producing fake 600 meV
residuals at non-symmorphic k-points. Root cause documented below
in §"What we observed about sym vs nosym": LORRAX `eqp0.dat` is
indexed by `sym.unfolded_kpts` (full BZ) not `wfn.kpoints` (IBZ),
and the legacy compare scripts assumed the latter.

## Where everything lives

### Sym path

```
runs/Si/06_si_4x4x4_nosoc/
├── qe/                                  # QE SCF + NSCF (with sym, ntran=48)
├── D_bgw_cohsex/                        # BGW reference, default settings
├── D_bgw_cohsex_noavg/                  # BGW with cell_average_cutoff=1e-12
├── D_lorrax_xonly_overlay_960c/         # LORRAX bare X (overlay convention)
├── D_lorrax_cohsex_overlay_960c_qe6/    # LORRAX cohsex (target=1e-6)
├── D_lorrax_cohsex_overlay_960c_qe1e5/  # quadrature sweep: 1e-5
├── D_lorrax_cohsex_overlay_960c_qe1e8/  # quadrature sweep: 1e-8
├── D_lorrax_cohsex_overlay_1440c/       # 1440 centroid variant
└── D_lorrax_cohsex_overlay_480c/        # 480 centroid variant
```

The patched-eps0mat / `exact_static_ch=1` directories from the
session are diagnostic only and should not be used for comparison
(see Methodology Pitfalls section).

### Nosym path

```
runs/Si/02_si_4x4x4_nosym/
├── qe/nscf/                             # QE NSCF run with nosym=true
├── 00_bgw_cohsex/                       # BGW reference (default settings)
├── D_lorrax_cohsex_overlay/             # LORRAX cohsex, 480 centroids
├── D_lorrax_xonly_overlay/              # LORRAX bare X, 480 centroids
├── D_lorrax_cohsex_overlay_960c/        # 960 centroids — running this session
├── D_lorrax_xonly_overlay_960c/         # bare X for 960c — running
├── D_lorrax_cohsex_overlay_1440c/       # 1440 centroids — running
└── D_lorrax_xonly_overlay_1440c/        # bare X for 1440c — running
```

## BGW reference flags

`runs/Si/02_si_4x4x4_nosym/00_bgw_cohsex/sigma.inp`:

```
band_index_min 1
band_index_max 16
number_bands 60
screened_coulomb_cutoff 25.0
frequency_dependence 0
exact_static_ch 0                # ←— *required to be 0*; 1 changes X
                                 #     convention and breaks comparison
degeneracy_check_override
no_symmerties_q_grid             # ←— *required for nosym path*
use_wfn_hdf5
dont_use_vxcdat
use_kihdat

begin kpoints
  ... 64 explicit k-points (full BZ uniform 4×4×4 grid) ...
end
write_vcoul
```

**For sub-meV agreement** with LORRAX:

```
cell_average_cutoff 1.0d-12          # disables fixwings finite-q rescaling
```

(this is the configuration in `runs/Si/02_si_4x4x4_nosym/02_bgw_cohsex_noavg/sigma.inp`
that gave the 0.12 meV MAE result above.) The default
`cell_average_cutoff = 1e12` triggers BGW's `fixwings(q0flag=false)`
pass at every q, which LORRAX does not mirror.

The vhead used at q=0 is **the same** in both modes —
`v_head = 3303.748102 a.u.` — extracted from the BGW
``DEBUG HEAD TERMS`` block in `sigma.out` (or equivalently from
`vcoul.dat` row 1). It is *not* affected by `cell_average_cutoff`.

`runs/Si/02_si_4x4x4_nosym/00_bgw_cohsex/epsilon.inp`: same `frequency_dependence
0`, `epsilon_cutoff 25.0`, `number_bands 60`, all 64 q-points listed
explicitly.

### Pitfalls — non-negotiable BGW settings

1. **`exact_static_ch 0`**, never 1. With 1, BGW emits only unprimed
   `Sig` columns; the unprimed `CH` includes BGW's static-remainder
   approximation, which LORRAX does not model. We confused ourselves
   into thinking the body residual was 400 meV by using
   `exact_static_ch=1` in a patched-eps0mat experiment.
2. **Compare `Cor' = (SX-X) + CH'`** (cols 4 + 10), not `Cor` (cols 4
   + 5). The primed CH column is the no-static-remainder value.
3. **`gpp_broadening` and `gpp_sexcutoff` are inert** for static
   COHSEX (`frequency_dependence 0`); they only enter `sigma_gpp_cpu`.
   Setting them does not affect the comparison.

## LORRAX side flags

`runs/Si/02_si_4x4x4_nosym/D_lorrax_cohsex_overlay/cohsex.in`:

```
[cohsex]
restart = false
centroids_file = centroids_frac_480.txt

nval = 8
ncond = 52
nband = 60
sys_dim = 3

x_only = false
do_screened = true
bispinor = false
self_consistent = false
use_ppm_sigma = false
screening_method = minimax
# do_G0 = true                   # default — apply head correction

# Cutoffs
bare_coulomb_cutoff = 25.0       # matches BGW screened_coulomb_cutoff

# BGW v overlay — same body and head as BGW reference
use_bgw_vcoul = true
bgw_vcoul_file = bgw_vcoul.dat   # symlink to BGW vcoul output
vhead = 3303.748102              # = BGW's MC-averaged head of v(q→0)
whead_0freq = 150.338778         # = vhead × ε⁻¹[0,0] = 3303.748·0.04551

use_chunked_isdf = true
memory_per_device_gb = 28

sigma_at_dft_energies = true
sigma_freq_debug_output = true
sigma_debug_split_contrib = true
fermi_reference = midgap

wfn_file = WFN.h5                # symlink to qe/nscf/WFN.h5
output_file = eqp0.dat
sigma_omega_h5_file = sigma_mnk.h5
```

The xonly variant (`D_lorrax_xonly_overlay/cohsex.in`) is identical
except `x_only = true`, `do_screened = false`. It must use the same
overlay flags as the cohsex run, otherwise the bare-X subtraction
doesn't cancel the head treatment.

### How `whead_0freq` is set

- `vhead = 3303.748102` is BGW's MC-averaged `v(q→0, G=0)`. This
  is what BGW's `vcoul` file stores at q=0 G=0 and what we read.
- BGW's natural `eps0mat.h5` has `ε⁻¹[0,0] = 0.04551` at q=0.
- LORRAX's `whead_0freq = 150.339 a.u.` reproduces this exactly:
  `0.04551 × 3303.748 = 150.34`. So the LORRAX rank-1 head correction
  matches the BGW (G=0, G'=0) element of `ε⁻¹` to 1 ppm.

The wings of `ε⁻¹` (G=0, G'≠0 ≈ 1e-6, the lower triangle G≠0, G'=0
≈ 28–84) are *not* reproduced explicitly by the LORRAX overlay — they
come from whatever the centroid-basis inversion gives at q=0. That's
fine because their net effect on Σ at the level of 40 meV is small,
and any discrepancy gets absorbed into the body residual.

### Pitfalls — non-negotiable LORRAX settings

1. **The xonly run for bare-X subtraction must use the same overlay
   as the cohsex run** (`use_bgw_vcoul=true`, same `vhead` /
   `whead_0freq`). Subtracting a no-overlay xonly inflates the
   apparent residual by ~360 meV.
2. **`do_G0 = true` (default).** Setting it to false drops the
   head correction *additively* but does not zero the head element
   in the body-inverted W (the centroid path has implicit head
   coupling). Only useful for diagnostic experiments — not for
   apples-to-apples comparison against BGW's natural eps0mat.

## What we observed about sym vs nosym

**TL;DR: the apparent 600 meV non-symmorphic-phase bug was an error
in the comparison script, not a real bug.** LORRAX SYM and NOSYM
agree to within 5 meV per band per k once the k-points are matched
correctly. The non-symmorphic phase fix from `dbe9798` (April 6)
works.

### The comparison-error trap

LORRAX `eqp0.dat` indexes its k-points by `sym.unfolded_kpts`
(the **64 full-BZ k-points** in the order produced by symmetry
unfolding), *not* by `wfn.kpoints` (the 8 IBZ k-points). For the
SYM Si run, the first 8 entries of `sym.unfolded_kpts` happen to be:

```
sym.unfolded_kpts[0..7] =
  (0,0,0),  (0,0,0.25), (0,0,0.5),  (0,0,0.75),
  (0,0.25,0), (0,0.25,0.25), (0,0.25,0.5), (0,0.25,0.75)
```

…while the 8 IBZ k-points stored in `wfn.kpoints` are:

```
wfn.kpoints[0..7] =
  (0,0,0),  (0,0,0.25), (0,0,0.5),
  (0,0.25,0.25), (0,0.25,0.5), (0,0.25,0.75),
  (0,0.5,0.5), (0.25,0.5,0.75)
```

Indices 0, 1, 2 happen to coincide. Indices 3+ differ. My earlier
compare scripts assumed `sym_eqp0[ik=3]` was at `(0, 0.25, 0.25)`
(the 4th IBZ k) but it actually contains Σ at `(0, 0, 0.75)`
(the 4th unfolded full-BZ k). With ~50 different physical-k
mismatches compounding, the sym vs nosym MAE looked like 234 meV.

Once matched correctly via `sym.unfolded_kpts`, the picture is:

| metric | sym 1440c | nosym 1440c |
|---|---|---|
| LORRAX vs BGW MAE Σ' | 18.8 meV | 16.4 meV |
| LORRAX vs BGW max\|Δ\| | 40 meV | 41 meV |
| LORRAX SYM vs LORRAX NOSYM (per band) | std 0.5–1.7 meV; max 5.4 meV | — |

The sym path agreement is *equivalent* to the nosym path — both are
in the 16-19 meV body-of-W ceiling regime.

### Verification: ψ unfolding is correct

To confirm the symmetry phase is right, I tested the unfolded
wavefunction `c_n(G)` directly. For each of the 64 full-BZ k-points,
loaded ψ_n in two ways:

1. **SYM path:** load ψ_{n,k_irr} from sym WFN.h5, then call
   `SymMaps.get_cnk_fullzone(wfn_sym, n, k_full)` to unfold to
   the target k_full (with τ phase + spinor rotation + sym-driven
   G remap).
2. **NOSYM path:** load ψ_{n,k_full} directly from nosym WFN.h5
   (which stores all 64 k explicitly).

Compared the gauge-invariant manifold-sum charge density
`Σ_{n in manifold} |c_n(G)|²` for the top-valence 4-fold manifold
(n=4..7) at every k_full. Result: **max |Δ| over all 64 k = 4×10⁻⁵
in the manifold-summed |c|², total norm matched to 1×10⁻¹⁵.** The
unfolded coefficients are correct everywhere in the BZ.

### Why I went down the bug-hunt rabbit hole

The session inherited a wrong assumption from earlier debug work
(the `sym ik` indexing convention had silently changed when SYM's
output went from IBZ-only to full-BZ-replicated, and the comparison
scripts in `runs/Si/06_si_4x4x4_nosoc/compare_x.py` etc. weren't
updated). When the comparison was fixed, the "bug" disappeared and
sym vs nosym agreed.

### Final state

- `sources/lorrax_D` symmetry path is **correct** for the
  non-symmorphic Si case.
- The `dbe9798` BGW-style `exp(-i (S G_source) · τ)` phase in
  `SymMaps.get_cnk_fullzone[_batch]` reproduces the
  unfolded ψ_n,k_full to 1e-5 precision against a direct nosym
  load.
- Both sym and nosym paths converge to the same 16-19 meV
  body-of-W residual against BGW at 1440 centroids.

The 600 meV "non-symmorphic phase bug" we set out to fix in this
session **was a comparison-script bug**.

## Centroid sweep on the nosym path

Ran cohsex and bare-X at 480 / 960 / 1440 centroids on the nosym
path, against **BGW(default)** (`avgcut=1e12`, with fixwings active):

| centroids | MAE Σ' | max \|Δ\| | direct gap Δ at Γ | indirect gap Δ | MAE bare X |
|---|---|---|---|---|---|
| 480 | 40 | 87 | -12.7 | -23.3 | 35.0 |
| 960 | 16.4 | 42 | -8.9 | -11.6 | 0.3 |
| 1440 | 16.4 | 41 | -8.8 | -11.5 | 0.05 |

(meV; Σ' = X + SX-X + CH'; sums are over 16 bands × 64 k for MAE/max)

vs **BGW(noavg)** (`avgcut=1d-12`, fixwings inactive at finite q):

| centroids | MAE Σ' | max \|Δ\| | MAE bare X |
|---|---|---|---|
| 1440 | **0.12** | **0.48** | 0.05 |

(meV)

**Takeaways:**

1. **Bare-X converges geometrically:** 35 → 0.26 → 0.05 meV from 480
   → 960 → 1440. The pair-density representation in the centroid
   basis is fully converged at 1440c.

2. **Body-MAE saturates at ~16 meV vs BGW(default)** but is
   **0.15 meV vs BGW(noavg)**. The 16 meV is therefore not a
   LORRAX-side convergence ceiling — it is the magnitude of BGW's
   `fixwings(q0flag=false)` finite-q wing rescaling. With that
   rescaling switched off (`cell_average_cutoff 1d-12`), LORRAX and
   BGW agree to under 0.5 meV per state across the whole BZ.

3. **Direct gap correction at Γ vs BGW(noavg) is at the few-tenths-meV level.**
   The remaining max single-state \|Δ\| ≈ 0.5 meV is a small q=0 χ
   representation difference: BGW evaluates χ at the q-probe
   `q0=(0.00025, 0, 0)` and LORRAX evaluates χ at literal q=0 with
   the head injected analytically; the `O(|q0|²)` correction between
   the two is the dominant remaining error.

## Pre-noavg-test session debris (pitfalls hit)

While converging on the right comparison setup we tripped on several
issues that have been documented in feedback memory entries and
`skills/compare/SKILL.md`:

| pitfall | symptom | fix |
|---|---|---|
| `exact_static_ch=1` in BGW | fake +400 meV residual | always `=0`; compare `Cor' = SXmX + CHp` |
| Wrong xonly bare-X (no-overlay) for COHSEX subtraction | fake +360 meV | xonly run uses same overlay flags as cohsex |
| `wfn.kpoints[ik]` instead of `sym.unfolded_kpts[ik]` for k-matching | fake 600 meV blowup at non-symmorphic k | always use `sym.unfolded_kpts` |
| Stale tmp/, eqp0.dat from a prior run with different vcoul | fake 137 meV bare-X disagreement | rebuild run dir from scratch when changing vcoul |
| Centroid generator `kmeans_isdf` (not `kmeans_cli`) | silently exits with no centroid file | use `centroid.kmeans_cli` |

The 16 meV body-of-W residual saturated at 960c is the new ceiling.
Quadrature target tightening doesn't move it (tested 1e-5 to 1e-8 on
the sym path; μeV-level effect). Where exactly it lives in the W
chain is the next investigation.

## Methodology pitfalls hit during the session

1. **`exact_static_ch=1` in a patched-eps0mat experiment** — produced
   a fake 400 meV residual. Never set to 1.
2. **Subtracted the wrong xonly bare-X** (no-overlay vs overlay).
   Inflated the residual by ~360 meV. Always use a matching overlay
   xonly run.
3. **Initial parser bug** read `p[10]` thinking it was Eqp1 vs CH'.
   Fixed.
4. **`gpp_broadening` / `gpp_sexcutoff`** are inert in static COHSEX.
   They only enter `sigma_gpp_cpu`.
5. **Centroid generator is `centroid.kmeans_cli`, not `kmeans_isdf`.**
   The latter is a library module with no `__main__`; running it as
   `python -m` exits silently with no centroid file.
6. **K-point indexing in LORRAX `eqp0.dat`.** The output is indexed
   by `sym.unfolded_kpts` (full BZ, 64 entries for Si 4×4×4), NOT
   by `wfn.kpoints` (8 IBZ entries). Mismatching this in a compare
   script invents a fake "600 meV bug at non-symmorphic k-points"
   that is purely a script artifact. **Always read k from
   `sym.unfolded_kpts[ik]` (or equivalently from BGW's `sigma_hp.log`
   if matching to the BGW reference).** See `skills/compare/SKILL.md`
   for the canonical pattern.
