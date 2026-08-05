# MoS2 3×3 SOC — PR3 ψ-side TRS-fix validation baseline

**Purpose.** Load-bearing validation target for **PR3** (`agent/trs-aware-sym-fix`),
which fixes the LORRAX bispinor ψ k-unfold path for non-inversion SOC systems
(Agent 1 audit scope Sites #5/#6/#7 in
`reports/trs_sym_audit_2026-05-14/agent_1_scope_report.md`).

The bug: `U_spinor[ntran:]` rows currently use the spinor of `+S_spatial` instead
of `iσ_y · conj(U_spinor[s])` for the TRS-augmented row, and `_get_umklapp_vector`
skips the τ-phase on TRS rows. The fix only affects k-points reached via
time-reversal in the wfn unfold.

## Symmetry structure (verified)

`sym_analysis.log` — confirmed:

| quantity                    | value     | note |
|-----------------------------|-----------|------|
| `wfn.ntran`                 | 2         | spatial sym ops (E + σ_h, all rotations killed by `no_t_rev=.true.` + SOC) |
| `len(sym.sym_mats_k)`       | 4         | = 2·ntran with TRS-augmentation |
| `#k needing TRS for unfold` | 4         | full-BZ indices {1, 3, 4, 5} |
| `#q needing TRS for fold`   | 4         | full-BZ indices {2, 6, 7, 8} |
| `has_inversion` (−I)        | **False** | MoS2 D3h has no inversion → PR3 fix has a non-trivial signal |

Acceptance test against task spec: `has_inversion=False` ✓ and ≥1 k via TRS ✓.

## Parameters

| field | value | source |
|-------|-------|--------|
| lattice           | a=3.164292 Å, c=12.0 Å (MoS2 1H monolayer) | copied from 00_mos2_3x3_cohsex |
| k-grid            | 3×3×1 unshifted + 3×3×1 q-shifted (BGW WFNq) | unchanged |
| pseudos           | FR-ONCVPSP-PBE standard (Mo.upf, S.upf)      | standard/ |
| `noncolin`, `lspinorb` | `.true., .true.` (spinor)                | required |
| `no_t_rev`        | `.true.` (forces full-BZ k expansion)        | required |
| ecutwfc           | 30 Ry                                         | task spec |
| nelec (spinor)    | 26 (= nval)                                  | from SCF |
| nbnd (NSCF)       | 58 (= nband+2 for BGW clearance)             | task spec |
| BGW `number_bands`| 56                                            | task spec |
| BGW `band_index_min/max` | 19 / 30 (8 val + 4 cond around Fermi) | unchanged |
| BGW `bare_coulomb_cutoff` | 30.0 (= ecutwfc, explicit override)  | task spec |
| LORRAX `nband`    | 56                                            | matches BGW |
| LORRAX centroids  | 399 (orbit-closed, from 206 reps; target ~400)| `centroids_frac_399.txt` |
| LORRAX GPUs       | 2 (divides nval=26)                           | task constraint |

## Directory contents

```
03_mos2_3x3_soc_2026-05-14/
├── README.md                       (this file)
├── manifest.yaml
├── sym_analysis.py                 (rerunnable WfnLoader+SymMaps probe)
├── sym_analysis.log                (probe output, captured 2026-05-14)
├── baseline_sigma_x_summary.py     (extract Σ_X table from sigma_freq_debug.dat)
├── baseline_sigma_x_summary.log    (the printed baseline table, see below)
├── qe/{scf,nscf,nscfq}             (SCF + NSCF + NSCFq + pw2bgw[q] outputs)
├── 00_bgw/                         (BGW epsilon + sigma reference)
└── 00_lorrax/                      (LORRAX cohsex baseline)
    ├── cohsex.in                   (LORRAX input)
    ├── WFN.h5 → ../qe/nscf/WFN.h5
    ├── centroids_frac_399.txt
    ├── dipole.h5  kin_ion.h5
    ├── eqp0.dat                    (QP energies)
    ├── sigma_diag.dat              (per-(k,n) Σ summary)
    ├── sigma_freq_debug.dat        (full per-(k,n) Σ decomposition)
    ├── sigma_mnk.h5                (Σ(ω) matrix output)
    └── gw.out                      (run log)
```

## Pre-PR3 baseline Σ_X — TRS vs non-TRS group means

From `baseline_sigma_x_summary.log` (Σ_X = `x_bare` + `x_head` in eV, mean over
sigma's 12-band (n=19..30) × 9-k window, pre-PR3 src at `796c043`):

```
  TRS k group    : N=48  mean Σ_X = -17.584544 eV   (pre-PR3)
  non-TRS k group: N=60  mean Σ_X = -17.522164 eV   (pre-PR3)
```

Post-PR3 (HEAD `8504994`) at the same window: TRS = -17.585314 eV,
non-TRS = -17.522962 eV.

Σ_X is **finite** at every (k, n), with no NaN/inf cells. The two
`sigma_freq_debug.dat.{pre,post}_pr3` snapshots capture the LORRAX run at
both commits with the same QE+BGW input, so the per-band PR3 diff is exact.

## PR3 diff: pre-PR3 vs post-PR3 (`pr3_diff_summary.log`)

When this task was constructed, PR3 (`8504994`, "symmetry_maps: PR3 —
unfold_psi free function + bispinor U_spinor TRS fix") had just landed.
To produce both reference columns this run captured `gw_jax` outputs at:

- **pre-PR3**: source checked out at `796c043` (post-PR2 + audit fixes), the
  HEAD specified in the original task.  `.pre_pr3` suffix.
- **post-PR3**: HEAD `8504994` (PR3 applied).  `.post_pr3` suffix.

Both runs used the same QE WFN.h5, the same 399 orbit-closed centroids, and the
same `dft_operators.py` API redirect to `WfnLoader.gvecs(k="full_bz")` (the
pre-PR3 source state at `796c043` still has the migration gap; the post-PR3
state inherits the same fix as a separate commit on this branch).

Group statistics for the sigma 12-band window (`pr3_diff_summary.log`):

| group   | field    | N   | mean        | max\|Δ\|  | rms      |
|---------|----------|-----|-------------|----------|----------|
| TRS     | Δx_bare  | 48  | -0.000770   | 0.058950 | 0.015745 |
| TRS     | Δsex_0   | 48  | -0.000116   | 0.002853 | 0.001131 |
| TRS     | Δcoh_0   | 48  | -0.000405   | 0.007846 | 0.002390 |
| TRS     | Δeqp0    | 48  | -0.001291   | 0.064434 | 0.017197 |
| non-TRS | Δx_bare  | 60  | -0.000799   | 0.049017 | 0.012453 |
| non-TRS | Δsex_0   | 60  | -0.000176   | 0.002917 | 0.000953 |
| non-TRS | Δcoh_0   | 60  | -0.000244   | 0.006878 | 0.002074 |
| non-TRS | Δeqp0    | 60  | -0.001219   | 0.053136 | 0.013002 |

Units: eV.  `Δx_head`, `Δsex_head`, `Δcoh_head` are all bit-equal (0.000000)
since the head term is scalar in k and not touched by the spinor-basis fix.

Key observation: **non-TRS k-points show diffs of the same order as TRS
k-points**.  This is physically consistent — Σ at a non-TRS k still convolves
W_q (and hence the χ_0(q)) over the FULL Brillouin zone, so a wrong ψ at
TRS-folded k pollutes Σ at every k through the q = k − k′ offset.  In
isolation the iσ_y rotation only affects ψ on TRS rows of the unfolded grid,
but its downstream effect propagates everywhere.

Max | Δ eqp0 | ~ 64 mΩ.  Per the task spec the expected signal was 10-100 meV
at TRS k.  Observed: 64 mΩ TRS / 53 mΩ non-TRS — within the predicted band,
with TRS rows ~20% larger than non-TRS, consistent with PR3 affecting the TRS
rows of ψ directly + the rest indirectly through W_q.

## Source-code note

`796c043` left a migration gap that blocked the pipeline at `psp.get_dipole_mtxels`
on this branch: `psp/dft_operators.py::generate_gvectors_k` still called the
old `sym.get_gvecs_kfull(wfn, kpoint_idx)` API, which moved into `WfnLoader`
during the P5 refactor (see `file_io/wfn_loader.py:298-318`). The fix in this run
makes it dispatch through the loader's cached `gvecs(k="full_bz")` table — the
same pattern `psp/get_DFT_mtxels.py:_gvecs_full_cache` already uses.

This patch is in scope for the PR3 branch (it's a pre-existing PR2 gap, found by
trying to bring up a non-inversion SOC test bed); see commit on
`agent/trs-aware-sym-fix`.

## Reproducing

```bash
# from a clean lxalloc on Perlmutter, lorrax_B + lorrax_agent loaded:
cd runs/MoS2/03_mos2_3x3_soc_2026-05-14

# QE
cd qe/scf  &&  srun … pw.x  -i scf.in  > scf.out
cd ../nscf  &&  ln -sf ../scf/MoS2.save .
              srun … pw.x  -i nscf.in  > nscf.out
              srun … pw2bgw.x -i pw2bgw.in > pw2bgw.out
              srun … wfn2hdf.x BIN WFN WFN.h5
cd ../nscfq &&  ln -sf ../scf/MoS2.save .
              srun … pw.x  -i nscfq.in > nscfq.out
              srun … pw2bgw.x -i pw2bgwq.in > pw2bgwq.out
              srun … wfn2hdf.x BIN WFNq WFNq.h5

# Symmetry probe (must precede further runs)
cd ..
srun … shifter … python3 -u sym_analysis.py > sym_analysis.log

# BGW reference (optional but cheap)
cd 00_bgw   &&  ln -sf ../qe/nscf/WFN.h5  WFN.h5   &&  ln -sf WFN.h5 WFN_inner.h5
                ln -sf ../qe/nscfq/WFNq.h5 WFNq.h5
                ln -sf ../qe/nscf/vxc.dat .   &&  ln -sf ../qe/nscf/kih.dat .
                srun … epsilon.cplx.x < epsilon.inp > epsilon.out
                srun … sigma.cplx.x   < sigma.inp   > sigma.out

# LORRAX baseline
cd ../00_lorrax &&  ln -sf ../qe/nscf/WFN.h5  .  &&  ln -sf ../qe/nscf/kih.dat .
                  ln -sf ../qe/scf/MoS2.save .  &&  ln -sf ../qe/scf/Mo.upf .  &&  ln -sf ../qe/scf/S.upf .
                  srun … shifter … python3 -u -m centroid.kmeans_cli 400 --seed 42
                  srun … shifter … python3 -u -m psp.get_dipole_mtxels -i cohsex.in
                  srun … shifter … python3 -u -m gw.kin_ion_io      -i cohsex.in
                  LORRAX_NGPU=2 lxrun python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in 2>&1 | tee gw.out

cd ..
python3 baseline_sigma_x_summary.py > baseline_sigma_x_summary.log
```

The Σ_X column in `baseline_sigma_x_summary.log` (TRS rows in particular) is the
**baseline to re-extract post-PR3**.
