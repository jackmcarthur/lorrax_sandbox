# CrI3 6×6×1, 30 Ry, bispinor — Phase 2 symmetry-validation test bed

**Created**: 2026-05-14 by lorrax_B agent. **Status**: pipeline complete through bare Σ_X (2026-05-14, lorrax_B `agent/trs-aware-sym-fix` @ `a00722d`).

## CAVEAT — what this test bed does NOT exercise

CrI3 monolayer has **spatial inversion** (`-I ∈ mtrx`, see "Symmetry analysis" below).
This has two important consequences for the symmetry/TRS work this test bed is
intended to support:

1. **TRS does not fire in q-fold or k-fold** because every q has a partner −q reachable
   via the spatial inversion already in `mtrx`. **TRS-fold count expected = 0** and
   was observed to be 0 in this run.
2. The **bispinor ψ-side TRS fix** (`iσ_y · conj`, sites #5/#6/#7 from the Agent 1
   bispinor-TRS-spinor scope) is therefore **NOT exercised** by this test bed.
   This run validates only:
   - the **cascade machinery layout** for an SOC bispinor system (file shapes,
     ζ writes, V_q tile dispatch), and
   - the **bispinor V_q wiring** (4 ζ channels × CC+TT tiles).
   To validate the iσ_y·conj TRS-spinor patch when PR3 lands, a **non-inversion
   bispinor system** is required (e.g. **monolayer 1H-MoSe₂ with SOC**, **BiI₃**,
   or CrI3 with an applied perpendicular E-field that breaks `-I`).

A separate test bed should be built for that purpose.

## What this is

A small, cheap, fully-relativistic (noncolin + lspinorb) CrI3 monolayer GW
reference designed as a permanent test bed for:

- **Phase 2 unified sym-action refactor** validation (ψ + ζ + V_q paths).
- **Future bispinor symmetry-handling regression tests**.

The intent is that this directory is *re-run frequently* — modify a LORRAX
source file, run `00_lorrax/cohsex.in`, compare to the BGW reference in
`00_bgw/`.

## Specs

| Field                  | Value                                              |
|------------------------|----------------------------------------------------|
| System                 | CrI3 monolayer, P-3 (space group 147), 2D          |
| Lattice                | a = 6.867 Å hex, c = 18 Å vacuum                   |
| k-grid                 | 6×6×1                                              |
| Cutoff                 | 30 Ry                                              |
| Spinor                 | noncolin = .true., lspinorb = .true. (bispinor)    |
| Pseudos                | FR-ONCVPSP PBE (standard) — Cr (14e), I (7e)       |
| nelec / nval / nbnd_GW | 70 / 70 / 84 (= 70v + 14c)                         |
| NSCF nbnd              | 86 (84 + 2 BGW buffer)                             |
| Centroids              | ~300 orbit-closed (see `00_lorrax/kmeans.log`)     |
| BGW                    | static COHSEX (frequency_dependence 0)             |
| LORRAX                 | bispinor, static COHSEX, x_only=true               |

## Symmetry analysis (logged here for future agents)

Read by `python` from `qe/nscf/WFN.h5`:

- **nsym = 6** (P-3 = E, C3, C3², I, S6, S6⁵ — 3 proper + 3 improper).
- **Inversion -I IS present** (mtrx op 4, det=-1, trace=-3).
- IBZ: 8 k-points (note: 36 full BZ folds to 8 because QE's symmetry
  group at this geometry happens to give 6 ops including I, and BGW WFN
  contains 48 k-points due to QE's noinv/no_t_rev unfold behavior).
- **Consequence for TRS work**: because -I ∈ mtrx, the V_q TRS-spinor
  folding path will NOT fire on this system (V_q only needs TRS for q
  points not connected by spatial symmetry, but -I connects all q ↔ -q
  pairs). This system is therefore appropriate for:
  - bispinor ψ-rotation under improper symmetries (det = -1 ops),
  - bispinor ζ-handling under improper symmetries,
  - bispinor V_q charge-channel sym tests (transverse channels not
    wired for IBZ-only on disk in current code — gw_init.py:650),
  - and IS NOT appropriate for testing the V_q TRS-only code path.
  - If TRS-path testing is needed, use a system without inversion (e.g.
    CrI3 + applied E_perp, or a non-centrosymmetric magnetic monolayer).

## Pipeline / what's here

```text
qe/scf/scf.in                      # 3×3 SCF, ecutwfc=30
qe/scf/CrI3.save/                  # charge density
qe/nscf/nscf.in                    # 6×6×1 unshifted, 36 k-pts, nbnd=86
qe/nscf/WFN.h5                     # → BGW / LORRAX
qe/nscf/{vxc,kih}.dat              # → sigma / cohsex
qe/nscfq/WFNq.h5                   # shifted (for BGW eps q→0+)

00_bgw/{epsilon,sigma}.inp         # bare_coulomb_cutoff explicit = 30 Ry
00_bgw/{eps0mat,epsmat}.h5         # eps matrices on disk
00_bgw/sigma_hp.log + eqp0.dat     # Σ(QP) reference
                                   #   NOTE: fermi_level -4.95 eV set in
                                   #   epsilon.inp & sigma.inp because at
                                   #   6×6 / 30 Ry there's a small global
                                   #   valence/conduction overlap (HO_max
                                   #   = -4.86, LU_min = -5.05 eV) that
                                   #   triggers BGW's ifmin/ifmax check.

00_lorrax/cohsex.in                # bispinor=true, x_only=true
00_lorrax/centroids_frac_<N>.txt   # orbit-closed centroids (--seed 42)
00_lorrax/gw.out                   # cohsex log
```

## How to re-run LORRAX side

```bash
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_B lorrax_agent

# IMPORTANT: this run-config OOMs on 40GB A100 even with reduced chunks.
# Use an explicit hbm80g salloc (not `lxalloc` which is hbm-mixed):
salloc --nodes=1 --qos=interactive --time=00:30:00 \
       --constraint="gpu&hbm80g" --gpus=4 --account=m2651 \
       -J "lx-alloc-$USER" bash -c "sleep 100000" &
disown ; sleep 20 ; lxattach

cd 00_lorrax

# Two kmeans runs are required for bispinor (scalar + current density):
LORRAX_NNODES=1 LORRAX_NGPU=2 lxrun python3 -u -m centroid.kmeans_cli 300 \
    --seed 42                                 # produces centroids_frac_300.txt
LORRAX_NNODES=1 LORRAX_NGPU=2 lxrun python3 -u -m centroid.kmeans_cli 300 \
    --seed 42 --density-mode current          # produces centroids_frac_298_current.txt

# Then COHSEX (Σ_X only):
LORRAX_NNODES=1 LORRAX_NGPU=2 lxrun python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in 2>&1 | tee gw.out
```

GPU divisor: nval = 70 → 2, 5, 7, 10, 14 GPUs OK; **4 GPUs does NOT divide
nval** so don't use -G 4 for this system.

Note: the CLI entrypoint is `centroid.kmeans_cli`, NOT `centroid.kmeans_isdf`
(`kmeans_isdf` has no `__main__` block and exits silently). See
`KNOWN_SANDBOX_ERRORS.md` 2026-05-14.

## Build / commit log

Reference commit (lorrax_B at time of test-bed creation):
see `/pscratch/sd/j/jackm/lorrax_sandbox/CHANGELOG.md` for the
`testbed_cri3_6x6_30Ry_bispinor` entry.

## 2026-05-14 first-pass results (lorrax_B `agent/trs-aware-sym-fix` `a00722d`)

The pipeline reached **bare Σ_X** for the bispinor cohsex configuration.
Downstream QP steps are blocked by a separate `kin_ion_io` bug (see
`KNOWN_SANDBOX_ERRORS.md` 2026-05-14). For `x_only = true` this is fine
because Σ_X is printed before the crash.

### Bare Σ_X (eV, k=0, first 8 sigma-window bands, degen-averaged)

```
-50.5679  -50.5679  -50.5733  -50.5733  -38.8713  -38.8713  -38.7781  -38.7781
```

Spin-doubled degeneracy (pairs equal to 4 ULP) is what we expect for a
bispinor system with `-I ∈ mtrx` (Kramers + spatial inversion → 2-fold
band sticking). Magnitudes are characteristic of deep semicore/valence
bands of CrI3 — these are NOT the frontier bands BGW reports.

### Bispinor Σ^B tile traces (eV; trace over sigma-window bands)

```
 (μ_L=1, ν_L=1):  -10.0978       (μ_L=1, ν_L=2):  -0.7102       (μ_L=1, ν_L=3):  -0.7116
 (μ_L=2, ν_L=1):  -0.7102        (μ_L=2, ν_L=2):  -10.0976      (μ_L=2, ν_L=3):  -0.7116
 (μ_L=3, ν_L=1):  -0.7116        (μ_L=3, ν_L=2):  -0.7116       (μ_L=3, ν_L=3):  -10.1946
```

Diagonal tiles (μ=ν) carry ~14× more weight than off-diagonal — consistent
with spin diagonalization dominating in a centrosymmetric SOC system.
Hermiticity (12=21, 13=31, 23=32) holds to 4 ULP.

### Cascade activation

The q-IBZ cascade did **not** fire on this run, by design:
- `gw_init.py:650` sets `_write_ibz_only_charge = not bool(cfg.bispinor)` —
  bispinor mode unconditionally writes the charge ζ full-BZ until
  `compute_V_q_bispinor_to_h5` gains IBZ support.
- All 7 V_q tiles in `gw.out` report `n_q_ibz=36 ... unfold=full-BZ`.
- TRS fold count = 0 (consistent with `-I ∈ mtrx`).

This is the expected state for this run config — the test bed validates
that the bispinor V_q tile pipeline produces finite, hermitian Σ_X with
the correct degeneracy structure even when the cascade is off.
