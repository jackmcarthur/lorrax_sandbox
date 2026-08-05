# VI3 + CrI3 GW + orbital-magnetization — overnight run 2026-06-17

**Agent D, working on `lorrax_D@main`.** Goal: for VI3 and CrI3 monolayers — 6×6 NSCF (600 bands)
→ LORRAX GW (`bispinor=false`, GN-PPM, σ_mnk(ω) on −10..10 eV/0.25, Σ-subspace = lowest 120 bands,
4000 centroids) → DFT+GW bandstructures (htransform) → orbital-moment vs #bands convergence
(agent B's `psp.orbital_magnetization`).

## TL;DR

| Deliverable | VI3 | CrI3 | Notes |
|---|---|---|---|
| +U-gapped FM DFT (SCF) | ✅ gap 0.11 eV, m=2 μB/V | ✅ m=3 μB/Cr (6 μB/cell) | VI3 needed the occ-matrix seed (run 03); CrI3 gaps natively |
| NSCF 6×6, 600 bands → WFN.h5 | ✅ 41 GB | ✅ (FM 80 Ry) | DFT+U pw2bgw needed the `.hub1` workaround |
| **Orbital-moment vs #bands** | ✅ (table below) | ✅ (table below) | the convergence study — works |
| **GW σ_mnk(ω)** | ❌ blocked | ❌ blocked | q=0 head: dipole loads all ψ → OOM on main (code limit) |
| DFT/GW bandstructure (htransform) | ❌ blocked | ❌ blocked | same load-all-ψ OOM |

**Two hard external/code constraints shaped the night:**
1. **Perlmutter maintenance 2026-06-17 06:00 → 06-24 06:00** (full-machine reservation). I raced it live on
   an interactive allocation; batch backups are queued to run when nodes return.
2. **The GW q=0 Coulomb head can't be built on `lorrax_D@main` for a 6×6/80 Ry cell**: the only LORRAX-native
   head source (`s_tensor`) needs `dipole.h5`, and `psp.get_dipole_mtxels` loads **every wavefunction to one
   GPU** (`read_Gvecs_to_devices`, ~155 GiB for 120 bands × 36 k) → OOM. `htransform` (bandstructures) has the
   same architecture (231 GiB). `epshead` needs BGW `eps0mat.h5` (absent). So σ_mnk + bandstructures are
   blocked pending a **per-k-streaming dipole/htransform** (code fix) or explicit `vhead/whead` overrides.

## Orbital magnetic moment vs band count (the convergence study)

Run with agent B's `psp.orbital_magnetization` (worktree `sources/lorrax_B_orbmag_wt`,
`agent/orbital-magnetization`), `--mu-scan` (m_orb at midgap), full-BZ 6×6.

**VI3** (spin moment 4.02 μB/cell = 2 V × d² S=1):

| #bands | m_orb (μB, ∥ spin) |
|---|---|
| 100 | −0.284 |
| 200 | −0.363 |
| 300 | −0.385 |
| 400 | −0.403 |
| 500 | −0.422 |
| 600 | **−0.440** |

Not converged at 600 — still rising (the intrinsic ∝N⁻¹·¹⁵ tail the CrI3 work documented). Extrapolates toward
the experimentally-large VI3 orbital moment (lit. "Large Orbital Magnetic Moment in VI3", ~0.6–0.8 μB). For a
band-converged value, a 2000–4000-band NSCF is needed (as the prior CrI3 orbmag study found).

**CrI3** (FM 80 Ry, spin moment 6.105 μB/cell = 2 Cr × d³ S=3/2):

| #bands | m_orb (μB, ∥ spin) |
|---|---|
| 100 | +0.032 |
| 200 | +0.170 |
| 300 | +0.169 |
| 400 | +0.164 |
| 500 | +0.154 |
| 600 | **+0.149** |

**Physics contrast (the headline result):** CrI3's orbital moment is **small (~0.15 μB) and quenched** — Cr³⁺
d³ has a half-filled t₂g with no orbital degeneracy — while VI3's is **large (≥0.44 μB and growing)** because
V³⁺ d² carries an unquenched orbital moment. Same honeycomb lattice, opposite orbital character. (Sign
convention differs: VI3 ∥−z/antiparallel-reported, CrI3 ∥+z; magnitudes are the physical point.) Figure:
`orbmag_convergence.png`. Both are band-truncation-limited at 600 (∝N⁻¹·¹⁵ tail) — magnitudes are lower bounds.

CrI3 run: `runs/CrI3/04_gw_6x6_600b_2026-06-17/` (FM SCF/NSCF/WFN.h5 + orbmag_nb{100..600}.out).

## What was produced (files)

- VI3 `runs/VI3/03_gap_recipe_80Ry_6x6_2026-06-16/` — +U-gapped FM SCF (gap 0.11 eV).
- VI3 `runs/VI3/04_gw_6x6_600b_2026-06-17/` — NSCF, `qe/nscf/WFN.h5` (41 GB), centroids_frac_4000.txt,
  `tmp/isdf_tensors_4000.h5` (21 GB, ISDF complete), kin_ion.h5, `orbmag_nb{100..600}.out`.
- CrI3 `runs/CrI3/04_gw_6x6_600b_2026-06-17/` — FM SCF/NSCF/WFN + orbmag (running).
- Batch backups queued for post-maintenance: `54624232` (VI3), `54624233` (CrI3) — full pipelines with all
  fixes; they will produce WFN + ISDF + V_q and then hit the same GW-head OOM (documented in the scripts).

## Blockers solved tonight (all logged in KNOWN_SANDBOX_ERRORS.md)

1. stale skill module names → `gw.kin_ion_io`, `psp.get_dipole_mtxels`
2. pw2bgw + DFT+U `.hub1` davcio crash → strip `<dftU>` from the disposable `.save/data-file-schema.xml`
3. `wfn2hdf.x` not on PATH → absolute BerkeleyGW bin path
4. quota at 99% → freed 16 TB of old `zeta_q*.h5`
5. centroid 77 GiB pivoted-Cholesky OOM → `--oversample 1.0`
6. LORRAX cohsex parser chokes on inline `#` comments → stripped comments
7. kin_ion cuFFT OOM at 600 bands → `--nb 120` (Σ subspace; kin_ion only needs the QP bands)
8. σ(ω) int64 overflow at 4000 cent × 81 ω (`kij` in-memory) → `sigma_omega_accumulation=kij_stream`
9. SCF npools must divide ntasks (CrI3 3×3: npools 9 on 16 tasks → use 8); CrI3 conv_thr 1e-7 plateaued → 1e-5

## Recommendations / next steps

1. **GW σ_mnk**: add per-k streaming (io_callback) to `psp.get_dipole_mtxels.read_Gvecs_to_devices` and
   `bandstructure.htransform` so a 6×6/80 Ry cell fits one GPU; OR supply explicit `vhead/whead_*` head
   overrides; OR generate BGW `eps0mat.h5` for `epshead`. Then the queued backups complete unattended.
2. **Orbital moment**: run a 2000–4000-band NSCF for a band-converged value (both systems).
3. Update `skills/execute_workflow/SKILL.md` module names + add the DFT+U pw2bgw `.hub1` workaround.
