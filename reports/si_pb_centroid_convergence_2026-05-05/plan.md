# Si pseudobands × ISDF centroids — convergence plan (publication-quality)

**System:** Si 2×2×2, ecutwfc=60 Ry, FR-PBE, nosym (8 full-BZ k-points).
**Branch:** `main` (consolidated `head-wing-fix`+ `hl-ppm` as of 2026-05-05).
**Goal:** Two convergence plots that establish the rate at which LORRAX-ISDF reproduces BGW Σ on a fixed pseudoband WFN as a function of (a) ISDF centroid count and (b) pseudoband band-count (truth → 4199 explicit bands).

## Data inputs already in the sandbox

| dir | n_band | what it is |
|---|---:|---|
| `01_bgw_pseudobands` | 4199 | parabands explicit (truth) |
| `02_bgw_pseudobands_50sl`  | 116 | 50 slices |
| `03_bgw_pseudobands_100sl` | 216 | 100 slices |
| `06_bgw_pseudobands_150sl_2spb` | 304 | 150 slices, 2 PB/slice |
| `07_bgw_pseudobands_100sl_3spb` | ~315 | 100 slices, 3 PB/slice |

Each carries:
- `WFN_pb.h5` (the pseudoband-compressed WFN that LORRAX will read)
- `sigma_hp.log` (BGW's reference Σ on that same compressed WFN — apples-to-apples comparison target)
- `eps0mat.h5`, `epsmat.h5`, `vcoul`, `kih.dat`, `vxc.dat`

Crucially **BGW's Σ is the comparison target for each variant**. LORRAX-using-`50sl` should match BGW-COHSEX-using-`50sl`, not BGW-using-`4199`. The convergence-to-CBS axis is `n_band` (how good the pseudoband approximation is), not the LORRAX-vs-BGW residual at that same `n_band`.

## Existing pivoted-Cholesky assay (the headwind)

`reports/pseudoband_pivoted_cholesky_assay_2026-04-18/` measured the trR/trG residual of the ISDF basis for parabands vs pseudobands at fixed σ window. At `n_cond=64`, `k=1000` centroids:
- parabands_4199: trR/trG = 6.7e-08 (ISDF basis essentially exact)
- pseudo_100sl_216: 2.1e-02 (1000× worse than parabands)
- pseudo_50sl_116:  2.6e-01 (1e6× worse)

So pseudobands have intrinsically poor ISDF representation — the eigenstates are spatial random sums of high-energy bands and don't reduce well to centroid product states. The convergence plot should show:
- Parabands: rapid `1/N_c²` or faster Σ-error decay
- Pseudobands (more slices): slower decay, plateau higher

## Recipe (canonical — locks in the 0.5 meV Si setup)

For each `(WFN_variant, N_c)` combination, write a fresh `runs/Si_pseudobands/00_si_2x2x2_60Ry/D_pb_<variant>_<N_c>c/` with:

```
[cohsex]
wfn_file = WFN.h5                 # symlink to ../<WFN_variant>/WFN_pb.h5
nval = 8
ncond = N_band - 8                # all PB bands as σ window
nband = N_band                    # 116 / 216 / 304 / 4199
sys_dim = 3
bispinor = false
self_consistent = false
x_only = false
do_screened = true
use_ppm_sigma = false             # COHSEX

bare_coulomb_cutoff = 60.0
screening_method = minimax
fermi_reference = midgap
sigma_at_dft_energies = true
sigma_freq_debug_output = true
sigma_debug_split_contrib = true

centroids_file = centroids_frac_<N_c>.txt

# Head + body BGW overlay (matches Si canonical 0.5 meV recipe)
vhead = <from BGW DEBUG>
whead_0freq = <from BGW DEBUG>
use_bgw_vcoul = true
bgw_vcoul_file = bgw_vcoul.dat    # symlink to ../<variant>/vcoul

use_chunked_isdf = true
memory_per_device_gb = 28
minimax_target_error = 1.0e-6
```

The matching BGW for that variant (already done in the variant dir) provides the reference `sigma_hp.log`. Comparison is `compare_bgw_gwjax.py` style.

**Important caveat from `reports/zeta_pruning_sharding_bug_2026-05-04/report.md`:** the pivoted-Cholesky pruning in ζ-fit is sharding-dependent. Si 4×4×4 needs LORRAX_NNODES=2 to land on the 0.05-meV branch. For Si 2×2×2 (8 q-points), 1 node × 4 GPU should be fine because no ny=3 axis problem, but I'll verify with a sanity run before the sweep. If we hit the same trap, document and use whichever sharding we can.

## The two plots

### Plot 1: ISDF-centroid convergence at fixed pseudoband variant

Hold WFN variant constant at `100sl` (216 bands). Sweep `N_c ∈ {400, 800, 1600, 3200, 6400}`. Compute LORRAX COHSEX Σ_total (sigTOT) at the σ window (band_index 1..16). Compare per-(k,n) to BGW `Sig'` from the variant's `sigma_hp.log`.

x-axis: `N_c`. y-axis: MAE |Σ_LORRAX − Σ_BGW| over (k,n).

Repeat the sweep for `parabands_4199` (truth) and `50sl_116` to show how much harder pseudobands are than parabands.

Three curves on one plot. Should show parabands hitting sub-meV early; pseudobands plateau at higher MAE.

### Plot 2: pseudoband convergence at fixed centroid count

At well-converged `N_c` (probably 3200 or 6400), sweep `n_band ∈ {116, 216, 304, 315, 4199}` by switching WFN variant. Compare LORRAX Σ → BGW(parabands_4199) Σ — i.e., the truth target across the variants.

x-axis: `n_band`. y-axis: per-band absolute Σ residual (or HOMO/LUMO QP shift) vs the parabands_4199 truth.

Both BGW and LORRAX get plotted (BGW already does this in `reports/cohsex_highncond_saturation_2026-04-19/`); LORRAX overlay shows whether ISDF tracks BGW's pseudoband convergence or adds its own approximation error.

## Execution order

1. **Sanity**: run LORRAX COHSEX at `100sl, N_c=1600` (1 node), confirm matches BGW `100sl` to ≲5 meV. If not, diagnose before going further.
2. Centroid sweep on `100sl`: 5 N_c values.
3. Add `parabands_4199` and `50sl` curves at the same N_c set.
4. Pseudoband sweep at fixed N_c.
5. Plot. Write report.

## Stretch goals (after the convergence plots land)

- The trR/trG metric from the April assay should also go on the same plot in a second panel — links the ISDF representation quality directly to the Σ residual.
- Add `parabands_4199 / nband=400` (truncated parabands) as a control to separate "ISDF can't represent random-sum states" from "high-energy bands are inherently hard to ISDF".

## Files / dirs to create

```
runs/Si_pseudobands/00_si_2x2x2_60Ry/
  D_pb_pb100sl_400c/    cohsex.in, links, output
  D_pb_pb100sl_800c/
  D_pb_pb100sl_1600c/
  D_pb_pb100sl_3200c/
  D_pb_pb100sl_6400c/
  D_pb_pb4199_400c/     ... (parabands at all N_c)
  D_pb_pb50sl_400c/     ... (50sl at all N_c)

reports/si_pb_centroid_convergence_2026-05-05/
  plan.md       <-- this file
  results.dat   <-- (N_c, variant, MAE_meV) tuples
  centroid_convergence.png
  pb_convergence.png
  report.md
```
