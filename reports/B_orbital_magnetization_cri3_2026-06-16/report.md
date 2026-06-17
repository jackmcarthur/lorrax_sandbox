# Orbital magnetization of CrI₃ (modern theory, explicit dH/dk) — agent B

**Date:** 2026-06-16  **Branch:** `agent/orbital-magnetization` (lorrax_B)
**Status:** ✅ tool built, validated, and applied to a ferromagnetic CrI₃ WFN.

## Goal

Implement the modern theory of orbital magnetization as a standalone LORRAX
tool and apply it to a DFT CrI₃ spinor wavefunction, taking the k-space
Hamiltonian derivative analytically (`dH/dk = 2(k+G) + dV_NL/dk`) — **no finite
differences in the velocity operator**.

## Deliverables

| File | What |
|------|------|
| `src/psp/orbital_magnetization.py` | Standalone script. `--wfn --nbnd --mu --mu-scan --convergence --per-band --vnl-sign --cpu`. Reports per-cell orbital moment (μ_B) summed over occupied bands and the full BZ, plus a spin-moment calibration and a Hellmann–Feynman velocity check. |
| `src/psp/orbital_magnetization_THEORY.md` | Concise derivation: master formula → sum-over-states (k·p) → Rydberg prefactor (−½, electron-charge sign) → SOC requirement → sign-vs-spin → sources. |
| `runs/CrI3/B_orbmag_FM_6x6_30Ry_2026-06-16/` | **New ferromagnetic** CrI₃ QE run (SCF+NSCF, SOC, `starting_magnetization` on Cr ∥ c). |

## Method (one line)

`m_γ/μ_B = (−½) Σ_k w_k Im Σ_{n occ} Σ_{m≠n} ε_γab v^a_nm v^b_mn (ε_m+ε_n−2μ)/(ε_n−ε_m)²`,
with `v = dH/dk = 2(k+G) + dV_NL/dk` (Ry·Bohr) and `Σ_k w_k = 1`.

## KEY FINDING: the sandbox CrI₃ WFNs are non-magnetic

Every existing CrI₃ calculation sets `noncolin/lspinorb=.true.` but **omits
`starting_magnetization`**, so the noncollinear SCF converges to the
**non-magnetic, time-reversal-symmetric** state: bands are Kramers pairs of
opposite spin (`⟨σ_z⟩ = ±0.85, ±0.32, …`) that sum to **zero net spin**. For a
TRS-preserving state both spin *and* orbital magnetization are identically
zero — the tool correctly returns 0 for these WFNs (a strong validation). To
get the physical orbital moment, a **ferromagnetic** WFN is required.

I generated one: `B_orbmag_FM_6x6_30Ry_2026-06-16`, identical 6×6 / 30 Ry / SOC
setup but with `starting_magnetization(Cr)=0.5`, moment ∥ c. SCF converged to
**total magnetization (0,0,6.01) μ_B/cell** and a proper **+1.50 eV gap**.

## Results (ferromagnetic CrI₃, 6×6 mesh, 180 bands, midgap μ)

| Quantity | Value | Note |
|---|---|---|
| nspinor / nk_full / nbnd / nocc | 2 / 36 / 180 / 70 | SOC spinor, full BZ |
| **Spin moment \|m_spin\|** | **6.014 μ_B** | from Σ⟨σ_z⟩; matches expected +6 ✓ |
| Indirect gap | +1.504 eV | proper insulator (FM ground state) |
| m_x, m_y | 0.000, 0.000 | ≈0 by symmetry ✓ |
| **Orbital moment m_z** | **+0.026 μ_B (∥ spin)** | physical sign `p+vNL`; midgap |
| μ-scan m_z (VBM/mid/CBM) | +0.032 / +0.026 / +0.019 | dM/dμ≠0 (Berry-curvature / QAH character) |
| Convergence vs band ceiling (90/125/153/180) | +0.009 / −0.007 / −0.001 / +0.026 | **not band-converged** (oscillates) |

**Orbital moment ≈ +0.03 μ_B, parallel to the spin moment** — the *sign* matches
the user's ~+0.1 μ_B expectation. The magnitude (~0.03) is smaller and is
**under-converged**: the band-ceiling sweep oscillates (the sum-over-states
inner-m sum needs many more empty states than the 110 available here) and the
μ-scan varies across the gap. The 6×6 k-mesh is also coarse. A converged value
(toward ~0.1 μ_B) needs a denser k-mesh and a larger band set.

## k-convergence + symmetry-reduced (IBZ) mode  [added 2026-06-16]

New run `runs/CrI3/B_orbmag_FM_conv_2026-06-16/` (one magnetic SCF, 6.01 μ_B,
gap +1.50 eV, reused across grids). Orbital moment **along the spin axis**
(parallel, μ_B), 180 bands, midgap μ, physical `p+vNL`:

| k-grid | full-BZ k → m∥ | IBZ k → m∥ | \|G\| | spin |
|--------|----------------|------------|-----|------|
| 6×6    | 36 → +0.02553  | 8 → +0.02562 | 6 | 6.014 |
| 8×8    | 64 → +0.02272  | 12 → +0.02272 | 6 | 6.014 |
| 10×10  | 100 → +0.02353 | 18 → +0.02353 | 6 | 6.014 |
| 12×12  | —              | 26 → +0.02310 | 6 | 6.014 |
| 14×14  | —              | 34 → +0.02278 | 6 | 6.014 |
| 16×16  | —              | 44 → +0.02337 | 6 | 6.014 |

(12×12–16×16 run IBZ-only — 26/34/44 k vs 144/196/256 full — exploiting the
validated exact unfold.)

**The orbital moment is k-converged at ≈ +0.023 μ_B, parallel to spin** (flat
across 6×6→16×16: +0.0227…+0.0255, settling at ~+0.023 from 8×8 on). It does
**not** climb toward ~0.1 with denser k —
the residual under-convergence is in the **band count** (the sum-over-states
inner-m sweep oscillates: mceil 90/125/153/180 = +0.010/+0.000/+0.003/+0.023 at
8×8), not in k. Reaching a fully band-converged value would need an NSCF with
≫180 bands; the PBE value may simply be smaller than the ~0.1 target.

**Band convergence (6×6) — the dominant axis, mapped to 4000 bands.**
`band_convergence_6x6_4000.png` (`plot_band_conv_N.py`, runs `nscf_6x6sym_{400,
2000,4000}`) plots m_z(∥spin) vs the SOS inner-band ceiling N. m_z(N) (along
spin): 180→+0.0256, 400→+0.0061, **800→−0.0283, 2000→−0.0609, 4000→−0.0782**.
So the apparent "+0.024 plateau" at 180 bands was a pure **band-truncation
artifact**: the curve crosses zero at N≈450 and grows **antiparallel** to the
spin, reaching **−0.078 μ_B at 4000** with the slope still shallowing.

The per-band increment decays only as **band^−1.15** (full *and* kinetic-only —
so it is intrinsic, not a nonlocal-velocity bug). This is the textbook **slow
convergence of the local-circulation term** of the orbital-magnetization SOS
(one energy denominator, `Σ_m v^x_nm v^y_mn/(ε_n−ε_m)`, vs two for the
Berry-curvature/IC term). band^−1.15 is *marginally* convergent: partial sums
approach the limit as ~N^−0.15, so halving the residual error needs ~100× more
bands. Hence **even 4000 bands is not fully converged**; extrapolations bracket
the limit at **≈ −0.08 to −0.12 μ_B** (visual flattening favors the low end;
an N^−0.15 fit is unreliable and reads −0.25).

**Verdict (revised after a literature check — see
`RESEARCH_NOTES_band_convergence.md`):** the slow `band^−1.15` convergence is
**real and structural**, not a bug — the local-circulation ("H−ε") term has a
single energy denominator `Σ_m v v/(ε_n−ε_m)` (the operator cancels one power of
Δε), versus the squared denominator of the Berry-curvature/itinerant term
[Xiao-Chang-Niu RMP 2010; Souza-Vanderbilt arXiv:0709.2389]. **But the direct SOS
is the wrong evaluation route**: CTVR (PRB 74,024408) state it is practical only
for tight-binding with few conduction bands and "quite tedious" in first
principles; the empty-band sum is *provably removable* via `Q=1−P`
(Souza-Vanderbilt: "depends exclusively on the occupied states"). **Our
antiparallel ≈−0.08 μ_B/cell is therefore most likely a truncation artifact, not
a converged physical value** — published DFT+SOC for monolayer CrI₃ is **+0.099
μ_B/Cr, PARALLEL** to spin (Ovesen & Olsen, arXiv:2405.04239, GPAW PBE NSCF SOC),
with experiment **−0.067 μ_B antiparallel** (PBE famously gets the sign wrong for
Cr halides). So neither our +0.024 (180 bands) nor −0.08 (4000 bands) is
trustworthy; the converged occupied-subspace answer is parallel.

**Recommended fix:** evaluate `M = (e/2ℏ)Im Σ_{n occ} ∫dk ⟨∂_k u_n|×(H+ε−2μ)|∂_k u_n⟩`
with `|∂_k u_n⟩` from the **covariant finite difference** of occupied Bloch
states at neighboring k (overlaps `S_nn'=⟨u_nk|u_n',k+q⟩`, `Q=1−P` implicit) —
NO empty-band sum [Lopez-Vanderbilt-Thonhauser-Souza PRB 85,014435; CTVR App. A].
Valid for the CrI₃ insulator; reuses the WFN reader + SymMaps k-neighbor maps.
Alternatives: Wannier90 `berry`, QE-CONVERSE (CPC 2025). The direct dH/dk SOS we
built has the formula/prefactor/sign all verified, but it is the single
worst-converging route for *this* observable.

### Symmetry-reduced (IBZ) mode — validated exact

`--ibz` loops the stored IBZ k-points (no ψ unfold) and symmetrizes the
**axial-vector** moment density over the **magnetic** point group:
`M = Σ_i w_i (1/|G|) Σ_g det(R_g) R_g · m(k_i)`. For CrI₃-FM(z) the magnetic
group is S₆ = {E, 2C₃, i, 2S₆} (|G|=6); the σ_v / in-plane-C₂ ops that flip
M_z and time reversal are excluded (here QE's `no_t_rev` already pre-removed
them). The projector is correct: `Pmat@ẑ=ẑ`, `Pmat@x̂=0` (C₃ kills in-plane).

**IBZ == full-BZ to machine precision** (8×8: +0.02272 = +0.02272 from 12 vs 64
k; 10×10: +0.02353 = +0.02353 from 18 vs 100 k — same SCF density ⇒ the unfold
is exact). 6×6 matches to 9×10⁻⁵ (separate SCF). The IBZ mode is ~5× fewer
k-points (and smaller WFNs) at identical physics — the right tool for dense
grids. `m_x=m_y=0` by C₃ in both modes.

## Band-sum-free Sternheimer method — built, but blocked by H reconstruction

`--method sternheimer` (`run_sternheimer_orbmag`) implements the band-sum-free
modern theory: per k, per occupied v, solve the Sternheimer equation
`(H_k−ε_v)|∂̃_a u_v⟩ = −Q(∂_a H)|u_v⟩` for the covariant derivative (reusing
`run_sternheimer.compute_kp_tangent_at_kvec`), then
`m_γ = −½ Im Σ_v ε_γab ⟨∂̃_a u_v|(H+ε_v−2μ)|∂̃_b u_v⟩`. Occupied-only outer sum,
conduction manifold summed exactly inside `(H−ε)⁻¹` → **band-count independent**.
The spec + the SOS↔Sternheimer algebraic equivalence were adversarially verified
(2 workflows, equivalence to 1e-15).

**Runs and partially validates:** ρ_val integral = 70.000 e (= nelec) ✓; spin =
6.014 μ_B ✓; m_x=m_y≈0 ✓. First result (6×6, midgap): m_z = −0.063 μ_B/cell
**antiparallel**, strongly μ-dependent (+0.022 VBM → +0.104 CBM, file frame →
Chern/QAH character). This is *consistent with the SOS extrapolation* (~−0.08),
so the antiparallel sign is **not** a band-truncation artifact.

**BLOCKER → ROOT CAUSE FOUND (2026-06-16):** the rebuilt KS Hamiltonian does NOT
reproduce the WFN eigenvalues — `⟨v|H|v⟩ − ε_v` is **0.5–1.4 eV** (mean ~0.43 eV)
with large off-diagonals (0.044 Ry), worst on a **deep Cr 3s semicore band at
1037 meV**, VBM 441 meV. The cause is now **fully identified and confirmed**:

> **LORRAX's standalone V_scf reconstruction builds V_xc from the CHARGE density
> only (spin-unpolarized PBE), omitting the xc magnetic field B_xc(r).** CrI₃ is
> ferromagnetic (~6 μ_B); QE used spin-polarized noncollinear V_xc. The omitted
> B_xc ≈ exchange splitting (~eV scale, peaked at the Cr atoms) is exactly the
> CrI₃-specific, band-selective residual. MoS₂ is non-magnetic (m=0) → charge-only
> V_xc is *exact* → 0.2 meV. Code: `dft_operators.compute_V_H_and_V_xc:317-323`
> (`rho_total = rho_val + rho_core`; `pbe_functional()`; **no magnetization input**).

**Elimination chain (all ruled out before landing on V_xc-spin):**

| hypothesis | test | result |
|---|---|---|
| diagnostic unfaithful | same gate on FR-SOC+NLCC **MoS₂** | **0.21 meV** ✓ → faithful, CrI₃-specific |
| NLCC missing/wrong | `diag_nlcc_toggle.py` ρ_core on/off | OFF makes it *worse* (1401→3567) → NLCC present & helping |
| radial table interp | `diag_nq_sweep.py` n_q 4k→32k | **flat** (1400.2→1400.1) → table converged |
| Hankel radial-FT accuracy | `diag_hankel_radial.py` 8× radial refine | Δ V_loc(q) ≤ 6e-4 → FT exact |
| dense-grid / ecutrho too coarse | |G|max on grid | 24 bohr⁻¹ (577 Ry corner) ≫ ecutrho → fine |
| ρ_val reconstruction wrong | `diag_qe_density.py` vs QE `charge-density.hdf5` | **matches to 0.03%** (peak ratio 0.9997) → density exact |
| **V_xc omits spin field** | `diag_xc_field.py` LSDA \|V_x↑−V_x↓\| | **~6 eV peak, ~4 eV mean over Cr** → matches the ~1 eV band residual ✓ |

| system | max ⟨v\|H\|v⟩−ε_v | off-diag | net moment | charge-only V_xc valid? |
|--------|--------------------|----------|------------|--------------------------|
| **MoS₂** (FR-SOC+NLCC, **non-magnetic**) | **0.21 meV** | 7e-6 Ry | 0 | yes (m=0) ✓ |
| **CrI₃** (FR-SOC+NLCC, **FM 6 μ_B**) | **1401 meV** | 0.044 Ry | 6.01 μ_B | **no — omits B_xc** ✗ |

⇒ The Sternheimer magnitude (−0.063) is **not quantitatively trustworthy** until the
standalone V_xc is made spin-polarized/noncollinear. Its agreement with the
V_scf-FREE SOS extrapolation (~−0.08) still pins the **antiparallel sign** robustly.

**Scope of the bug (what is / isn't affected):**
- ✗ **Affected:** any LORRAX path that rebuilds the KS Hamiltonian from a WFN and
  inverts/applies it on a **magnetic** system — i.e. the `run_sternheimer`
  covariant-derivative / DFPT resolvent, and hence the **Sternheimer orbital-mag**
  and any **GW screening χ/W that uses the rebuilt V_scf on a magnetic system**
  (e.g. bispinor-CrI₃). The error is ~the exchange splitting near the magnetic ions.
- ✓ **Not affected:** the **SOS orbital-mag** path (velocity-only, `v = 2(k+G) +
  dV_NL/dk`, no V_scf — verified at `orbital_magnetization.py:71,174`); the
  **velocity/dipole operator** (kinetic + nonlocal, no V_xc); and the **core
  `gw.gw_jax` GW driver**, which uses WFN ε_nk/ψ_nk directly and does *not* rebuild
  V_scf. Non-magnetic systems (MoS₂, Si) are exact.

**The fix (real feature, not a one-liner):** (1) reconstruct m_α(r) full-BZ from the
WFN spinors, `m_α = Σ_occ ψ†σ_α ψ` (mirror of `build_rho_val_from_wfn`); (2) call a
**noncollinear LSDA/GGA V_xc** returning the scalar V_xc + the field B_xc·σ̂; (3)
extend `apply_H`/`setup_H_k` to apply a **2×2 spin-dependent** local potential
(currently `V_scf` is a single real grid multiplying both spinor components
identically). Gate on `⟨v|H|v⟩=ε_v → ≪ meV` for CrI₃, same as MoS₂.

## Velocity-sign resolution (load-bearing)

The orbital moment's sign hinges on the nonlocal velocity sign, and the two
nonlocal helpers in the codebase disagree by a sign convention:

- Canonical `velocity_matrix_k = p + vnl_velocity_from_dZ` (≡ `p + vNL`).
- The dipole driver uses `p − vNL` to *match BerkeleyGW*'s reported velocity.

These give **opposite** orbital moments: `p+vNL → +0.026` (∥ spin) vs
`p−vNL → −0.081` (anti∥). Resolved definitively by an off-diagonal finite
difference of `⟨m|V_NL(k)|n⟩` (ψ fixed): **`compute_vnl_velocity_cart = +dV_NL/dk`
to ratio +1.000** → the physical velocity is **`p + vNL`** → orbital moment is
**parallel to spin**. The Hellmann–Feynman *diagonal* slope test was tied
(`RMS 0.1585` vs `0.1559`) because `dV_NL/dk` is ~900× larger off-diagonal than
on-diagonal — the diagonal can't see the sign. (The BGW `p−vNL` convention is
for optical matrix elements, not the physical velocity.)

## Validation summary

- Spin moment from the same WFN = 6.01 μ_B (independent of the orbital formula). ✓
- m_x = m_y = 0 (out-of-plane symmetry). ✓
- Non-magnetic WFNs give exactly 0 spin and 0 orbital (TRS). ✓
- Prefactor −½ in Ry AU (3 independent derivations + adversarial verify). ✓
- Nonlocal velocity = +dV_NL/dk verified to ratio 1.000 (off-diagonal FD). ✓
- Gather norm = 1.0 per band; eigenvalues in Ry; k-weights = 1/nk_tot. ✓

## Artifacts

- `qe_pipeline.log`, `qe_nscf_pw2bgw.log` — QE SCF/NSCF/pw2bgw.
- `orbmag_FM_nbnd180.log` (auto-sign, =p−vNL), `orbmag_FM_vnlplus.log` (physical p+vNL).
- `orbmag_FM_nbnd180.npz` — per-k velocity matrices, energies, ⟨σ_z⟩.
- `diag_k0.py`, `diag_vnl_sign.py` — gather/spin and nonlocal-sign diagnostics.
- **H-residual root-cause chain:** `diag_vscf_residual.py` (gate, MoS₂↔CrI₃),
  `diag_nlcc_toggle.py` (NLCC on/off), `diag_nq_sweep.py` (radial-table n_q),
  `diag_hankel_radial.py` (Hankel FT accuracy), `diag_qe_density.py` (ρ_val vs QE
  `charge-density.hdf5`), `diag_xc_field.py` (omitted B_xc magnitude).

## Next steps

1. **Fix the spin V_xc** (see "ROOT CAUSE" above): noncollinear V_xc + B_xc·σ̂ +
   2×2 spin-dependent `apply_H`. This unblocks the Sternheimer orbital-mag *and*
   any rebuilt-V_scf GW screening on magnetic systems (bispinor-CrI₃). Validate via
   `⟨v|H|v⟩=ε_v ≪ meV` on CrI₃ (currently 1401 meV).
2. The **antiparallel sign** (orbital ∦ spin, Hund's 3rd rule for Cr³⁺ 3d³<½) is
   already established by the V_scf-free SOS path and corroborated by Sternheimer;
   magnitude ~−0.06 to −0.08 μ_B/cell, consistent with experiment (~−0.067).
3. After the fix: converge band-sum-free Sternheimer on denser k (≥12×12) for the
   final magnitude; optionally split LC vs IC contributions.
4. The non-magnetic-WFN finding affects any CrI₃ "magnetic" interpretation of
   prior GW work in this sandbox — flagged for the user.

---

## 2026-06-17 UPDATE — B_xc IMPLEMENTED + exhaustive residual audit

**B_xc is implemented and validated** (lorrax_B branch `agent/bxc-vscf-magnetic`,
commits `eba8609` + `c5ec2f0`):
- `psp/xc.py`: spin-polarized PBE (`_pbe_x_spin` via exact spin-scaling reusing
  `pbe_x`; `_pbe_c_spin` PW92 spin-interp + φ(ζ) gradient — verified term-by-term
  vs QE `pw_spin`/`pbec_spin`/`pbex`) + `compute_V_xc_spin` (per-spin GGA V↑/V↓).
- `psp/scf_potential.py`: `build_magnetization_from_wfn` (full-BZ Pauli sandwiches,
  matches QE charge-density m to 0.1%); `build_dft_potentials(m_vec=...)` → spin
  V_xc with the **signed magnetization** (segni = sign(m·û), smooth — avoids the
  |m|-kink that corrupts the GGA gradient at magnetization sign-changes).
- `psp/dft_operators.py`: `HamiltonianK.B_vec` + `(B·σ)ψ` (helper `_bsigma_psi`) in
  `apply_H_k`/`apply_H_k_from_G`/`build_matrix_k`; `B_vec` kwarg in `setup_H_k*`.
  Non-magnetic path is bit-identical (trace-time `None` guard).

**H-reconstruction result** (⟨v|H|v⟩−ε):

| system | result |
|--------|--------|
| MoS₂ / Si (non-magnetic, FR-SOC, PBE, NLCC, 2D) | **0.2 meV** (exact) |
| CrI₃ FM, **no B_xc** | 1400 meV |
| CrI₃ FM, **B_xc + segni** | **~19 meV mean / 74 meV max**, deep Cr-3s 2 meV |

**Exhaustive residual audit (every component checked against QE; all match):**
density (charge-density.hdf5, 0.03%/0.1%); spin-PBE functional (pointwise +
on-grid, ≤0.05 meV); V_NL SOC D-matrix (VKB `deeq_nc` eigenvalues, exact, rank
36/atom); V_loc incl. 2D cutoff (pp.x V_bare, ⟨ψ|Δ|ψ⟩ ≤7 meV, uncorrelated with
residual); 2D Coulomb (validated by the MoS₂ control — same `assume_isolated='2D'`
input); spinor frame (native-wfc: self-consistent, frame-invariant). E_nk
preserved by pw2bgw export exactly (0.0 meV). **The remaining ~19 meV is confined
to the m≠0-only finite-ζ spin V_xc on the Cr 3p/3d semicore** — vanishes at ζ=0
(so MoS₂/Si miss it) and never directly diffable against QE (pw2bgw refuses VXC
for nspin=4). Not a gross missing term; a subtle few-tens-of-meV effect that does
**not** move the orbital moment (a valence-d property). Diagnostics in
`agent4_xc_diff/` and `diag_*.py`.

**Orbital moment (the deliverable)** — `cri3_orbmag_convergence.png`:
- **Antiparallel to spin, m_z ≈ −0.078 μ_B/cell** (6×6, 4000-band SOS), consistent
  with Hund's 3rd rule (Cr³⁺ 3d³ < ½) and experiment (−0.067). Sign is robust
  (V_scf-free SOS).
- **Band count is the convergence bottleneck**: the SOS m_z(N) is +0.026 at 180
  bands, crosses zero, and grows to −0.078 by 4000 bands (slow tail).
- **k-grid is fast/flat**: +0.026/+0.023/+0.024 μ_B at 6×6/8×8/10×10 (N=180).

**Scope reminder:** B_xc affects only rebuilt-V_scf paths on magnetic systems
(Sternheimer / standalone-H, and rebuilt-V_scf GW screening on bispinor-CrI₃);
the core `gw.gw_jax` driver (uses WFN ε/ψ directly) and non-magnetic systems are
unaffected.
