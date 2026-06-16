# Literature check: orbital-magnetization band convergence & method (2026-06-16)

Deep-research synthesis (104 agents, 22 primary sources fetched, 25 claims 3-vote
verified). Bottom line: **our slow band convergence is real and expected, but the
direct sum-over-states (SOS) is the wrong evaluation route, and our antiparallel
result is most likely a truncation artifact — the converged DFT answer is parallel.**

## 1. The slow convergence is structural and known (NOT a bug)
The combined SOS factor splits algebraically into exactly two pieces:
`(ε_m+ε_n−2μ)/(ε_n−ε_m)² = 1/(ε_n−ε_m) [local-circ] + 2(ε_m−μ)/(ε_n−ε_m)² [itinerant]`.
- **Local circulation (self-rotation, "H−ε"): SINGLE denominator** → slow tail.
- **Itinerant (Berry-curvature, "2(ε−μ)"): SQUARED denominator** → fast.
The operator `(H−ε_n)` hitting `|u_m⟩` gives `(ε_m−ε_n)`, cancelling one denominator.
Our measured per-band `~band^−1.15` decay is the structural signature of the
single-denominator self-rotation sum. [Xiao-Chang-Niu RMP 82,1959 (2010) Eq.4.14;
Thonhauser review arXiv:1105.5251 Eq.28; Souza-Vanderbilt arXiv:0709.2389] (3-0)
*Caveat: no source quotes a numerical exponent; −1.15 is our measurement, only the
qualitative single-denominator slowness is literature-backed.*

## 2. Direct SOS is documented as impractical for first-principles
CTVR (PRB 74,024408, 2006) state the direct SOS k-derivative (their Eq.51) is
practical only "in the context of tight-binding calculations, where the sum over
conduction bands runs only over a small number of terms," and warn "in
first-principles calculations the sums over conduction states can be quite tedious."
The empty-band sum is **provably eliminable** via closure `Q=1−P`: Souza-Vanderbilt
(PRB 77,054438) — "the right-hand-side depends exclusively on the occupied states."
So our slow tail reflects a poor route, not a physical requirement. (2-1)

## 3. The correct (band-sum-free) method
`M = (e/2ℏ) Im Σ_{n occ} ∫dk f_nk ⟨∂_k u_nk|×(H_k+ε_nk−2ε_F)|∂_k u_nk⟩` — a single
BZ integral over **occupied states only**. `|∂_k u_nk⟩` is the **covariant
derivative** (projection `Q=1−P`), evaluated by **finite differences of occupied
Bloch states at neighboring k** (overlaps `S_nn'=⟨u_nk|u_n',k+q⟩`) — NO empty-band
sum. Valid for band insulators (CrI₃ qualifies). Both LC and IC come out of the
occupied subspace. [Lopez-Vanderbilt-Thonhauser-Souza PRB 85,014435 (2012) Eq.1;
CTVR PRB 74,024408 Eq.46 + App. A Eqs.64-66] (3-0)
- **Wannier interpolation** (LVTS12): ~20 WFs for occ + partially-occ bands; a small
  FIXED manifold, not the divergent conduction sum. Avoids both the k-mesh cost AND
  the empty-band sum. (Finite-difference covariant derivative can't do metals; CrI₃
  insulator is fine.) (3-0)
- **QE-CONVERSE** (CPC 2025, `mammasmias/QE-CONVERSE`): production code, non-
  perturbative "converse" modern theory in the Wannier representation — the
  recommended tool "where perturbative methods fail." (3-0)

## 4. CrI₃ physics — our sign is likely wrong (artifact)
Ovesen & Olsen, arXiv:2405.04239 (2024), Table II (GPAW PAW, PBE NSCF SOC):
- **Monolayer CrI₃ orbital moment = +0.099 μ_B/Cr, ALIGNED (PARALLEL) to spin** (PBE).
  (PBE+U 0.055; LSDA+U 0.097/0.114.)
- **Experiment = −0.067 μ_B, ANTI-aligned.**
- Convention: positive = spin/orbital aligned. Prose: "the measured orbital moments
  are … anti-aligned with the spin moment. The computed orbital moments, however,
  are all predicted to be aligned." **PBE+SOC predicts the WRONG SIGN for Cr halides.**

⇒ Our SOS antiparallel ≈ −0.08 μ_B/cell does **not** reproduce even the standard
DFT+SOC prediction (which is *parallel*, +0.099/Cr ≈ +0.2/cell). The zero-crossing +
antiparallel tail is most consistent with an **unconverged/truncation artifact** of
the SOS route, not a physical answer. Magnitude is the right ballpark; sign + non-
convergence are the red flags. (single source, 2-1; the broader "PBE wrong sign"
takeaway is the robust part.)

## 5. Open / not resolved here
- Whether the `[r,V_NL]` nonlocal-pseudopotential commutator specifically drives the
  high-band tail (modified f-sum rule) — not isolated by any verified claim. (Check
  Pickard-Mauri; Essin-Turner-Vanderbilt-Souza on orbital mag with nonlocal potentials.)
- Formal Hund's-3rd-rule expectation for Cr³⁺ 3d³ (L antiparallel to S) vs crystal-
  field quenching in the octahedral I₆ cage — not resolved; my earlier "Hund's rule
  ⇒ antiparallel" hand-wave is NOT a safe explanation (DFT predicts parallel).
- Multi-band gauge invariance: only the SUM M^LC+M^IC is gauge-invariant; band-
  ceiling partial sums are not individually meaningful.

## Recommended fix
Re-evaluate CrI₃ with the **covariant finite-difference** method (occupied-state
overlaps between neighboring k; reuses the existing WFN reader + SymMaps k-neighbor
maps; NO band sum), or use **QE-CONVERSE / Wannier90 berry**. Expect it to converge
to the published **+0.099 μ_B/Cr parallel** (confirming the SOS antiparallel tail
was an artifact). If it ALSO comes out antiparallel, that would instead implicate
the SOC/exchange treatment — but the literature says parallel.

Sources (primary): Xiao-Chang-Niu RMP 2010; Thonhauser arXiv:1105.5251; CTVR PRB
74,024408 (2006) + arXiv:0705.3771 (Ceresoli-Resta); Souza-Vanderbilt arXiv:0709.2389,
PRB 77,054438; Lopez et al. PRB 85,014435 (2012)/arXiv:1112.1938; QE-CONVERSE CPC 2025;
Ovesen & Olsen arXiv:2405.04239 (2024).
