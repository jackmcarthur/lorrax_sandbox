# VI3 monolayer — noncollinear PBE + SOC + U setup & first SCF

**Date:** 2026-06-16 · **Agent:** D · **Run:** `runs/VI3/00_monolayer_pbe_soc_u_2026-06-16/`
**Status:** ✅ SCF set up and converged on GPUs. Electronic-structure details (gap) need follow-ups.

## Goal

Set up a VI3 monolayer slab QE calculation — noncollinear PBE + spin-orbit + Hubbard U (3.5 eV) —
and run it on Perlmutter GPUs.

## What VI3 is

Vanadium triiodide: isostructural with CrI3 — a honeycomb of V³⁺ (d², S=1) ions in edge-sharing
VI₆ octahedra; layered van-der-Waals **Ising ferromagnet** with an **out-of-plane easy axis** and a
large/unquenched V orbital moment (a separate quantity from the spin moment QE prints below).
Literature (DFT+U+SOC, relaxed, fine k): FM semiconductor,
gap ≈ 0.3–0.5 eV, T_C ≈ 29 K.

## How it was built (isostructural reuse of CrI3)

Adapted the proven CrI3 FM monolayer input (`runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16`), Cr→V.

| Parameter | Value | Source / rationale |
|---|---|---|
| Pseudopotentials | `V.upf`, `I.upf` (standard) | FR-ONCVPSP PBE, `relativistic="full"`; z_val 13 / 7. V.upf has the `3D` PP_PSWFC needed for Hubbard projectors |
| Lattice a, c | 6.84 Å, 18 Å | Experimental bulk R-3 (multiple refs); outlier 7.13 Å discarded |
| Internal coords | CrI3 geometry (**unrelaxed**) | isostructural approximation |
| ecutwfc / ecutrho | 50 / 200 Ry | first-pass balance |
| k-grid | 3×3×1 | coarse first pass |
| Spin | `noncolin=.true.`, `lspinorb=.true.` | SOC, spinor WFs |
| Magnetism | `starting_magnetization(1)=0.5`, angle1=angle2=0 (+z) | required with SOC (else non-magnetic); +z = easy axis |
| DFT+U | `HUBBARD ortho-atomic` / `U V-3d 3.5` | Dudarev (`lda_plus_u_kind=0`); ortho-atomic valid for noncollinear k-point runs |
| Slab | `assume_isolated='2D'` | 2D Coulomb |
| N_val | 68 | 2×13 (V) + 6×7 (I) |

Run: `pw.x -npools 4` on 4× A100 (1 node), attached to the shared pool allocation via `lxattach`.

## Result (converged SCF, 43 iter / 64 s wall)

| Quantity | Value |
|---|---|
| Total energy | −450.2721 Ry |
| Net magnetization (spin) | **m_z = 4.40 μB/cell** (along +z) |
| Absolute magnetization (spin) | 6.91 μB/cell |
| Per-V moment m_z (sphere, spin) | **2.80 μB** |
| V 3d Hubbard occ Tr[ns] | 3.76 (↑3.28 ↓0.48) |
| Highest occ / lowest unocc | −5.79 / −5.87 eV ⇒ **overlapping / near-zero gap** |

All target physics confirmed live: `Hubbard projectors: ortho-atomic`, `U(V-3d)=3.5000`,
`Noncollinear calculation with spin-orbit`, quantization axis (0,0,1).

> **These magnetizations are spin-only.** `pw.x` reports ∫**m**(r) and ∫|**m**(r)| from the spin
> density; it does **not** compute the orbital moment **L**. Net 4.40 μB ≈ 2 V × d² spin-only (~2 μB
> each) + ligand polarization; the per-V 2.80 μB is a sphere-integrated spin value (> 2 μB from I
> covalency / small sphere radius), **not** evidence of orbital moment. For VI3's large orbital
> moment, run `psp/orbital_magnetization.py` (modern theory) on the converged WFN.

## Caveat & next steps

The HOMO sits *above* the LUMO (−5.79 > −5.87 eV): at this **coarse 3×3 + unrelaxed + 50 Ry**
level the system is near-metallic, not the ~0.3–0.5 eV FM semiconductor literature reports. This is
the expected price of the first-pass shortcuts (CrI3 bond lengths strain VI3; coarse k). To get the
physical gap:

1. **Relax internal coordinates** (`calculation='relax'`) — VI3 V–I ≈ 2.80 Å vs the CrI3-derived ≈ 2.73 Å used here.
2. **Converge ecutwfc** (iodine wants ≥ 60–80 Ry) and **k-grid** (6×6+).
3. **Recheck the gap** — VI3+U+SOC has distinct converged orbital solutions (quenched vs large
   orbital moment); may need `starting_ns_eigenvalue` to steer the d-occupation.

## vc-relax attempt (50 Ry) — Pulay collapse

Ran a `vc-relax` (`cell_dofree='2Dxy'`, in-plane only) at 50 Ry. Findings:

- **Mixing/occupation sensitivity:** Gaussian smearing + `plain` mixing **sloshed** and never
  converged (0 BFGS steps); `local-TF` **plateaued** at ~4×10⁻⁶ Ry. Only **fixed occupations +
  `plain` mixing** (the standalone-SCF settings) converge here. Don't "improve" the mixer.
- **Pulay-stress runaway:** with a converging SCF, the cell collapsed by *accelerating* steps —
  a = 6.84 → 6.82 → 6.79 → 6.75 → 6.70 → 6.61 → 6.48 Å (6 steps, stopped manually). Net moment
  held at ~4.35 μB throughout. The accelerating contraction below the experimental 6.84 Å is the
  signature of **Pulay stress** (50 Ry is too low for stress with NC iodine → spurious compression).

**Takeaway:** stress (hence cell) needs ≥80 Ry to be meaningful; forces/ionic relaxation are far
less cutoff-sensitive. Recommended: a **fixed-cell ionic `relax`** at 50 Ry (relax internal coords
at experimental a=6.84 — the standard approach, no Pulay) and/or a **vc-relax at 80 Ry** for the
lattice constant. Input kept at `qe/vcrelax/vcrelax.in`.

## Fixed-cell ionic relax (50 Ry) — V–I bond corrected to the VI3 value

Ran `calculation='relax'` (cell fixed at a=6.84, plain mixing, fixed occupations). **5 BFGS steps,
total force 0.078 → 0.017 Ry/bohr (~78% reduction).** Net moment steady at ~4.4 μB.

**Key structural result — V–I bond:**

| | V–I bond (Å) |
|---|---|
| CrI3-derived start | 2.731 |
| Relaxed (step 5) | **2.810** |
| VI3 literature | ~2.80 |

The I cage expanded from the Cr geometry to V's, landing on the literature V–I bond. So the
isostructural-from-CrI3 guess was off by exactly the bond-length difference, and the ionic relax
fixed it. Input: `qe/relax/relax.in`.

**Not fully converged:** at the 6th geometry the electronic SCF broke down (`eigenvalues not
converged` → `convergence NOT achieved after 200 iter`); QE stopped gracefully. The near-zero-gap
state is delicate with fixed occupations as the cage moves. The relaxed structure (V–I = 2.81 Å) is
nonetheless a solid improvement. Next: re-SCF at the relaxed coords (may need a convergence tweak)
and a production relax at 80 Ry + denser k.

## Infra notes (for the skills)

- The GPU `espresso/7.5-libxc-7.0.0-gpu` module only applies its PrgEnv-gnu→nvidia swap (and puts
  `pw.x` on PATH) under `bash -lc`; in the plain non-interactive agent shell `module load` is a no-op
  beyond a libsci/mpich reload. Run QE as `bash -lc 'module load espresso/...; srun ... pw.x ...'`.
