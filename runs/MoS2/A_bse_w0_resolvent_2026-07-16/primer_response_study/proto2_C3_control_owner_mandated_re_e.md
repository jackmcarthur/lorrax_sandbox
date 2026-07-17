# proto2 C3 — control + owner-mandated re-exam: bounded-spectrum rank-r solve on interpolated ingredients, judged on the physical metric

**Date** 2026-07-17 · **Fixture** MoS2 3x3 (`00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/{isdf_tensors_640,zeta_q}.h5`, charge channel, n_mu=640, 30 Ry ζ-sphere, full-BZ 9 q) · **Scripts** `proto2_c3.py`, `proto2_c3_diag.py` · **Log** `proto2_out_c3_3x3.log` (SLURM step 56052603.11 — the authoritative record) · **Data** `proto2_c3_3x3.npz`

## 1. What ran

Exactly the measured §3.5 leave-one-out pipeline — interpolate the ingredients
`C_q` (centroid Gram) **and** `Z_q = C_q ζ_q` (r-basis RHS) to each held-out
on-grid `q0` with truncated-R Fourier weights (nR=7, `vq_loo.py` machinery
verbatim; weights `w = f0 · pinv(F)`), then solve at `q0` — with the solve
ladder extended by the **overtly-regularized fixed-rank reduced inverse**

```
eigh(C0_interp) = R diag(λ) R^H   (λ descending; via the same Hermitian SVD as the rankcut rows)
ζ̂ = Σ_{i≤r} λ_i^{-1} R_i R_i^H Z0,     ONE global r per κ target,  κ_eff = λ_1/λ_r ∈ {1e2, 1e4, 1e6}
```

[Task-formula note: "`Rh[:,1:r] Sh_r^-2 Rh[:,1:r]^H`" read as the top-r
eigenpairs, 1-based; `λ = Sh²` so `κ_eff = (Sh_1/Sh_r)²= λ_1/λ_r`. Global `r`
= round(median over q0 of `#{λ ≥ λ_1/κ}` on the interpolated spectra):
κ1e2→r=81, κ1e4→r=286, κ1e6→r=551 of 640; attained κ_eff ranges in the log.
`cond(C_q)` true = 1.7–1.9e7.]

One mathematically-exact deviation from the literal `vq_loo.py` order: solve
and r→sphere projection act on different axes (μ vs r) and commute, so the
solve runs on `Zt = to_sphere(Z_r)` (640×~2000 instead of 640×46080).
Verified against the literal order: relF **2.3e-14** (SELF-CHECK 5).

**Metrics** (per OWNER PUSHBACK, which governs):

- **PRIMARY (a) — gap-window B-block.** `M_cvk(μ) = Σ_s conj(ψ_{c,k−q0,s}(μ)) ψ_{v,k,s}(μ)`,
  v ∈ top-3 valence (23–25), c ∈ bottom-3 conduction (26–28; `ifmax`=26), all
  9 k → 81 rows. `B[t,t'] = Σ_{μν} conj(M_t(μ)) V_q0[μν] M_t'(ν)`. Reported:
  relF, and median per-element error over the top decile of `|B_direct|`.
- **PRIMARY (b) — TDA exciton shift.** 81-dim `H(q0) = diag(D) + K^x − K^d`,
  production convention (`bse_serial.py:23-70`): `K^x = (1/Nk) conj(M) V M^T`,
  `K^d = (1/Nk) A_c(k,k'|q0) W0_{k−k'} B_v(k,k')` from the **stored production**
  `W0_qmunu` + `enk_full`; conduction legs at wrapped `k−q0`; **only the V tile
  inside K^x is swapped** (direct-fit vs interpolated). |Δλ| of the lowest 4, meV.
- **SECONDARY (continuity)** — tile `‖ΔV‖_F/‖V‖` and the §3.5 random-all-band
  pair `d*V d` (verbatim `physical_contract.py` seed-0 pairs), both with the
  §3.5 3D-bare `v = 8π/|q+G|²`.

**Kernels.** Primary metrics use the **production 2D slab-truncated Coulomb**
(`compute_vcoul.py:178-183`: `v = 8π/|K|²·[1−e^{−zc·k_xy}cos(k_z·zc)]/V_cell`,
`zc = π/|b3| = L_z/2 = 11.34` bohr) — validated below to machine precision
against the production tile. Continuity metrics keep the §3.5 3D-bare v
verbatim so the old numbers reproduce.

## 2. Self-checks (all pass; gate for any interpolation number)

| check | result |
|---|---|
| recon↔to_sphere round-trip on sphere(Γ) | 2.3e-16 |
| **null test**: true C, true Z → stored ζ̃ / direct V (q=1) | 1.8e-10 / 1.4e-10 (machine-level given cond 1.8e7 · ε ≈ 4e-9) |
| **Γ tile vs on-disk production `V_qmunu`** (slab kernel) | scale = 1.000000, relF **1.96e-15** — my `make_Vq(slab)` IS the production tile pipeline |
| same with 3D-bare v | sc=2.28e-3, relF 0.17 (expected — kernel mismatch; why the slab kernel is the primary-metric kernel) |
| disk `V_qmunu` at q≠0 vs my slab-full tile | 1.0–1.3e-9 at computed q's — disk includes G=0 at q≠0 (head zeroed at Γ only) |
| solve∘to_sphere commutation (literal vq_loo order) | 2.3e-14 |
| stored `W0` Hermiticity / TRS `conj(W_{−q})=W_q` | 2.8e-14 / 1.7e-11 |
| assembled H(q0) Hermiticity (all 9 q0) | ≤2.5e-12 (validates K^d/K^x index+conjugation conventions) |
| exciton sanity | lowest exciton 0.919 eV, binding 781 meV vs D_min=1.700 eV (Γ/K valley q0's) — physically sane for a coarse 3×3 TDA fixture |
| §3.5 continuity (must reproduce logged study) | tile med raw/rc1e-6/rc1e-4/rc1e-2 = 3.7e6/1.2e4/1.1e1/1.00 ✓ and randpair 7.4e4/1.5e3/2.0e1/0.89 ✓ — **exact** |
| window-edge degeneracy (BGW TOL 1e-6 Ry) | **split Kramers pairs at both edges** (1.8e-15 / 5.3e-16 Ry) — the spec-mandated 3+3 window cuts through spinor doublets; both variants share the window so the swap metric is unaffected; flagged per the closed-shell rule |

### Diagnostic finding (SELF-CHECK 4 anomaly, resolved — `proto2_out_c3_diag.log`)

`V` tiles built from the **stored per-q ζ** violate TRS at O(1)
(`relF(conj(V_{−q}), V_q)` up to 3.8) while the disk `V_qmunu` pairs at 2.6e-15.
Full resolution, per-q: disk == my slab-full tile at q ∈ {0,1,3,4} (1e-9) — the
computed set; disk == `conj`(partner) at {2,6,8} (TRS-unfolded); {5,7} produced
by spatial-sym unfold (match neither directly). So production `V_qmunu` is
IBZ-computed + sym/TRS-unfolded (symmetry-exact by construction), while the
9 independent per-q fits in `zeta_q.h5` carry **independent near-null-space
solver noise** that is not symmetry-covariant. That junk dominates the tile:
rank-truncating the true solve at κ=1e4 leaves `relF(V_trunc, V_full) = 1.00 at
every q` — **the full-rank bare tile is ~100% junk by Frobenius norm** — and
even the truncated tiles don't TRS-pair (0.8–2.0) because `Z = C ζ_stored`
bakes the junk into the RHS. The physical content is the small clean part (next
section). This is the MEMORY note "tile magnitude/covariance are gauge
artifacts" with hard numbers, and it means §3.5's tile bar was largely
comparing noise to noise.

## 3. The re-based bar table (the C3 deliverable)

MoS2 3×3 LOO, nR=7, medians over the 9 held-out q0 (max in parens where shown).
tile+randpair = 3D-bare v (§3.5 continuity); B-block+exciton = production slab v.

| row | tile relF | randpair d*Vd | **B relF** | **B topdec med** | **Δλ_1..4 (meV)** |
|---|---|---|---|---|---|
| TRUE raw (null test) | 1.9e-10 | 4.7e-12 | 6.6e-13 | 4.6e-13 | 0.00 |
| TRUE fixedr κ1e6 (r=551) | 9.0e-1 | 5.0e-2 | **7.6e-4** | 5.4e-4 | **0.01** (0.04) |
| TRUE fixedr κ1e4 (r=286) | 1.0 | 1.3e-1 | **3.3e-3** | 2.7e-3 | **0.05** (0.15) |
| TRUE fixedr κ1e2 (r=81) | 1.0 | 4.0e-1 | 5.1e-2 | 3.7e-2 | 5.5 (9.4) |
| INTERP raw | 3.7e6 | 7.4e4 | 1.4e3 | 8.4e2 | 64.8 (91.3) |
| INTERP tikhonov 1e-6 | 9.9e5 | 2.9e4 | 4.0e2 | 2.6e2 | 64.3 (90.7) |
| INTERP rankcut 1e-8 | 3.7e6 | 7.3e4 | 1.4e3 | 8.4e2 | 64.8 (91.3) |
| INTERP rankcut 1e-6 | 1.2e4 | 1.5e3 | 1.3e1 | 9.0 | 51.6 (81.4) |
| INTERP rankcut 1e-4 | 1.1e1 | 2.1e1 | **1.19** | 0.90 | **17.7** (51.5) |
| INTERP rankcut 1e-2 | 1.00 | 0.89 | **1.14** | 0.95 | **18.2** (39.2) |
| INTERP fixedr κ1e6 | 1.2e4 | 1.5e3 | 1.3e1 | 8.6 | 51.6 (81.4) |
| INTERP fixedr κ1e4 | 1.1e1 | 2.1e1 | 1.22 | 0.96 | 17.7 (51.5) |
| INTERP fixedr κ1e2 | 1.00 | 0.89 | 1.14 | 0.95 | 18.2 (39.2) |
| CLEAN-INTERP fixedr κ1e6/1e4/1e2 | ≈INTERP | ≈INTERP | 13.4 / 1.22 / 1.12 | 8.6 / 0.96 / 0.92 | 51.6 / 17.7 / 18.2 |

Side-by-side with the §3.5 baseline this table re-bases (same LOO points, same
solves; §3.5 columns reproduced exactly by this run):

| solve | §3.5 tile | §3.5 randpair | **re-based: gap-window B relF** | **re-based: exciton Δλ (meV)** |
|---|---|---|---|---|
| raw | 3.7e6 | 7.4e4 | 1.4e3 | 65 |
| rankcut 1e-6 (≈κ1e6) | 1.2e4 | 1.5e3 | 13 | 52 |
| rankcut 1e-4 (≈κ1e4) | 1.1e1 | 2.1e1 | 1.19 | 18 |
| rankcut 1e-2 (≈κ1e2) | 1.00 | 0.89 max 1.70 | 1.14 | 18 |

## 4. Findings

**(1) The owner's pushback is CONFIRMED, quantitatively.** Rank-truncating the
solve on **true** ingredients at κ=1e6 changes the tile by 90% and the
random-all-band contraction by 5% — but the gap-window B-block by only
**7.6e-4** and the exciton eigenvalues by **0.01 meV**. The ill-conditioned
tail (≥ 90% of tile Frobenius weight; §2 diagnostic: ~100%) contributes ~0.1%
to what the BSE actually consumes. Tile-Frobenius and generic-d contractions
over-weight junk directions and are NOT verdict variables; §3.5's "phys 0.89"
(random all-band pairs) was itself junk-weighted. A corollary worth keeping:
the exchange tile is physically rank-compressible 640→286 (κ1e4) at 0.3% B
error / 0.05 meV — potentially useful for storage/solve economy elsewhere.

**(2) But the §3.5 no-window conclusion SURVIVES the re-exam — re-based ~10³
softer, still a failure.** Under the physical metric the interp ladder has a
shallow optimum at κ≈1e4: **B relF ≈ 1.2 (median), top-decile ≈ 0.9, exciton
shifts ≈ 18 meV median / 51 meV max** — vs the few-percent-B / ~meV success
threshold. No row of the interpolated family comes within 20× of the bar.
Globalizing r (fixed-r) vs per-q0 thresholds (rankcut) changes nothing
(identical to 3 digits) — the per-q0 spectra barely vary (r spread ±1).

**(3) The failure is interpolation noise INSIDE the physical subspace, not the
truncation.** At matched rank r=286: TRUE 3.3e-3 vs INTERP 1.22 B relF —
**370× above the truncation floor**. And junk-cleaning the training data
(projecting each training `Z_q` onto its own C's top-r eigenspace before
interpolating — the CLEAN rows) is a **near-no-op** (≤2% changes): expected,
since the reduced inverse only reads `U_r^H Z` and foreign-q junk barely
projects into the top-r subspace. So the non-interpolable content is the
*physical-sector* part of `Z_q` — the genuine q-rotation of the pair-density
span — exactly §3.5(4)/(4b)'s mechanism (`ζ_R` flat, Z inherits it), now shown
to live in the physical subspace rather than being a junk artifact.

**(4) Consequence for the transported-frame program (the ablation rung).** C3 —
same LOO points, same metrics, no frames, no transport — fails at
B ≈ 1.1–1.2 / 18 meV. The controlled-ablation bar for C1/C2 is therefore:
**beat B relF med ~1.1 and Δλ ~18 meV by orders of magnitude; the
metric-floor rung to claim real success is the TRUE-truncation control
(≈3e-3 B / 0.05 meV at κ1e4; reference-junk floor ≈ 0.3% B / ~1–2 meV in Δλ —
the ±q0 asymmetry of the direct references).** If C1 lands few-percent under
this same metric, the half-inverse/frame claim is vindicated by this ablation;
if C1 also fails, arbitrary-Q exchange stays with the per-Q ζ refit
(§4 option 1) and the SR/LR potential-level route.

## 5. Verdict

**NEGATIVE for C3 as an arbitrary-Q scheme** — the bounded-spectrum rank-r
solve on interpolated ingredients does not approach few-percent physical
accuracy at any κ (best ≈ 110–120% B-block error, ≈ 18 meV exciton shifts,
median over LOO q0) — **with the owner-mandated re-basing delivered**: the
§3.5 bar quoted by every other construction is now the physical-metric column
of §3 (B relF / exciton meV), not the tile column, and the tile column is
demonstrated to be ~100% gauge/solver junk. Truncation is exonerated;
physical-sector ingredient interpolation on a 3×3 stencil is the convicted
mechanism.

## 6. Caveats

- 3×3 only. The 4×4 fixture has no `zeta_q.h5` on disk (C-side only), so the
  Z-side LOO cannot run there; 6×6 exists but §3.5(3) already showed error
  *grows* with more R-vectors, and the mechanism is density-independent.
- The metric window (3v×3c) splits Kramers doublets at both edges (flagged in
  §2); shared by both variants, irrelevant to the swap, but a closed-shell 2v×2c
  or 4v×4c window would be cleaner for sub-permille claims.
- Exciton H omits the rank-1 W-head and (at Γ) V-head channels — identical in
  both variants, so the swap metric is unaffected; absolute binding is
  fixture-level, not production-converged.
- Direct-fit references carry the stored-ζ junk at the ≤0.3% (B) / ~1–2 meV
  (Δλ, ±q0 asymmetry) level — the measured signals are 20–400× above that floor.
- Provenance note: the authoritative outputs are the on-disk log/npz from SLURM
  step 56052603.11 (+ .5 for the 13-row subset, identical to 3 digits). A
  transient background-task notification during this session displayed a
  phantom "improved CLEAN" table matching no real execution (no such SLURM step
  exists; the watcher's own output file contains the disk numbers); it was
  discarded. All numbers here are grep-able from the disk log.

## 7. Production mapping (deferred until results warrant)

Every dense op here is a single-device 640-scale call: `eigh`/Hermitian-SVD of
`C0` → the N_mu² `P('x','y')`-sharded cusolvermp/slate FFI form; the reduced
inverse and `U_r^H Z` GEMMs → distributed GEMMs on the same layout; `to_sphere`
→ the existing per-q sphere gather. Nothing in the negative verdict is a
device-count artifact.

## 8. Files

- `proto2_c3.py` — the full pipeline (verbatim-reuse provenance in-file)
- `proto2_out_c3_3x3.log` — authoritative run log (all tables above)
- `proto2_c3_3x3.npz` — per-row per-q0 arrays (tile/rand/brel/btd/dl, spectra, r_glob)
- `proto2_c3_diag.py`, `proto2_out_c3_diag.log` — the TRS/junk/disk-convention diagnostic
- Fixtures (read-only): `runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native/tmp/{isdf_tensors_640,zeta_q}.h5`
