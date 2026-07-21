# Converged MoS2 G0W0 on the 80 Ry / 12x12 / 400-band reference

**Run dir** `runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/`
**Branch** `agent/gw-converged-campaign` (worktree `sources/worktrees/lorrax_gw_converged`),
branched from `agent/gw-conduction-postfix` @ `b7654ee`, merged with `agent/bse-figures` @ `cb20681`
**Date** 2026-07-21 · **Hardware** Perlmutter, own allocations `56278326` (4 nodes / 16x A100-**80GB**,
`--constraint="gpu&hbm80g"`) and `56279505` (1 node / 4x A100-80GB, side jobs)

---

## 0. Headline

| | DFT (PBE, FR-SOC) | G0W0 eqp0 | **G0W0 eqp1** |
|---|---|---|---|
| direct gap @ K | 1.6954 eV | 2.9337 eV | **2.6356 eV** |
| indirect gap | 1.6954 eV | 2.7230 eV | **2.5079 eV** |
| VBM | **K** | **K** | **K** |
| CBM | K | (0.167, 0.167) | (0.167, 0.167) |
| VBM(K) − VBM(Γ) | +0.1297 eV | +0.3458 eV | +0.2982 eV |

**The VBM is at K.** On the 6x6 / 30 Ry predecessor it sat at Γ by 0.06 eV; on the
converged 12x12 / 80 Ry grid it is at K by **+0.298 eV** (GW) — and already by
+0.130 eV in the DFT input, so the earlier Γ-VBM was a k-grid/cutoff artefact,
exactly the convergence signature that was hoped for.

**eqp1 direct 2.636 eV sits inside the 2.5–2.8 eV monolayer-MoS2 G0W0 literature
band** (the 6x6 run's 2.942 eV was just above it). eqp0 (no Z-linearization)
remains high at 2.934 eV; the Z factor is doing real work here.

---

## 1. What was run

`qe/` (the 80 Ry / 12x12x1 / 400-band reference, `reports/qe_reference_80ry_12x12_2026-07-21`)
was **not rebuilt**; everything below consumes it.

| dir | what | n_μ | nband | notes |
|---|---|---|---|---|
| `00_lorrax_gw_2400c` | stage-3, first attempt | 2412 | 326 | rank-truncation **silently inactive** (§2) — kept as the `cholesky` arm |
| `00b_lorrax_gw_2400c_ranktrunc` | **stage-3, the run to quote** | 2412 | 326 | replication cap raised → intended route; gate **ALL PASS** |
| `02_lorrax_gw_1236c_bse_producer` | BSE producer attempt, 12x12 | 1236 | 326 | built fine; the BSE on top of it aborted (§5) |
| `_qe6x6` + `03_lorrax_gw_6x6_80Ry_bse` | 6x6 NSCF at 80 Ry + its GW | 1452 | 120 | the route the BSE can actually use (§5) |
| `01_lorrax_exciton_bands` | BSE exciton bandstructure | — | — | §5 |
| `_stage4/*` | knob study | 2412 | 326 | Σ-only variants restart from `00b` |

Centroids for the production run: **2412** = 208 orbit representatives x the
12-op recovered D3h *density* point group, **band-range weighted over the full
screening range** `w(r) = Σ_{n∈[0,326)} Σ_k w_k |ψ_nk(r)|²`. Both centroid
fixes together. `n_μ / nband = 2412 / 326 = 7.40x`.

### Cost and the memory-model calibration the coordinator asked for

Planner (`gw.gflat_memory_model.plan_gflat_chunks`, the call `gw_init` makes),
n_μ = 2412, ncond = 54, 16 devices:

| budget | r_chunk | # chunks | persistent | **planner HWM** |
|---|---|---|---|---|
| 60 GB/dev | 9248 | 19 | 9.12 GB | 50.95 GB/dev |
| **65 GB/dev** | **10192** | **18** | 9.12 GB | **55.22 GB/dev** |

So 60 → 65 GB buys exactly **one** fewer r-chunk (19 → 18), not the large
reduction the 40 GB / wide-Σ scenario suggested — the HWM is pinned at 85 % of
whatever budget it is given (`util = 0.85`), and the r-chunk floor `r_chunk ≥ n_μ`
is nowhere near binding at 18 chunks. The production run reproduced the 65 GB
plan exactly (`r_chunk = 10192 (18 chunks)`, `HWM estimate = 55.22 GB/dev`).

**Calibration (measured, three numbers that disagree and should):**

| quantity | value |
|---|---|
| planner HWM estimate | 55.22 GB/dev |
| LORRAX's own live high-water report (`GPU high-water mark:`) | **54.15 GB/dev (83 %)** |
| `nvidia-smi` resident, all 16 GPUs during the ζ fit | **79.4 – 79.8 GB of 81.9 GB (97 %)** |

The planner predicts LORRAX's *live-array* peak to **2 %** (55.22 vs 54.15, i.e.
slightly conservative) — it is a good model of what it models. The gap to
`nvidia-smi` is **+25 GB of BFC arena + CUDA context + cached buffers**, a
factor **1.44** on the planner number, not the ~14 % in the folklore. Read
carefully: the ~14 % figure and this 1.44 are not the same comparison, and
budget decisions must use the arena number. **65 GB was the right call and was
also close to the ceiling** — the run peaked at 97 % of the card and did not
OOM. 70 would very likely have.

Wall time, `00b`: **703 s end-to-end** (ζ fit 257 s over 18 r-chunks, V_q,
screening, Σ 121 s) against the 1 h 17 extrapolation in the sizing report —
**6.6x faster than predicted**. The extrapolation scaled Σ by `nk·nq·μ²·nb`
and over-charged badly; the 18-chunk r-loop it worried about cost nothing.

---

## 2. The sanity gate earned its keep: two of the five cures were silently off

The brief's gate was *"Γ/K VBM+CBM Σ sane, gap non-inverted, no astronomical Im,
rank-truncate reports full-rank Galerkin — if any fail, STOP."*
`sanity_gate.py` implements it against the run's own `sigma_diag.dat` / `eqp1.dat`
/ `gw.out`. On the first production run **every physics check passed and the
route check failed**:

```
[FAIL] rank-truncate path active: 'path=replicated_rank_truncate' appears 0x in gw.out
```

`isdf/core.py` selects the charge ζ-solve route as

```python
if _replicate_charge_ok(nq, n_rmu):        # nq * n_rmu**2 * 16 <= 4 GiB
    return ('replicated_rank_truncate' if charge_zeta_solve == 'rank_truncate'
            else 'replicated_cholesky')
return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'
```

and **only the replicated route carries the rank-truncation cure (`23af6b9`) and
the mesh-invariant replicated ζ-fit (`ca78008`)**. The converged campaign is
above the cap — nq = 74 (IBZ), n_rmu = 2416 → **6.44 GiB > 4 GiB** — so
`charge_zeta_solve = rank_truncate` in `cohsex.in` was parsed, accepted, and
ignored. No warning is emitted; the only trace is the route name in the per-q
`Computing L_q` line. Two of the five cures the whole campaign rests on were
**not exercised at production scale**, and the 6x6 A/B that validated them could
never have shown it (nq·n_μ² there is well under the cap).

Re-run with the cap raised in-process (`gw_probe.py --cap-gib 8`, no source
edit) → route confirmed 16x, **gate ALL PASS**:

```
[PASS] sigX finite + bounded              max|sigX|    = 39.891 eV
[PASS] sigC finite + bounded              max|Re sigC| =  5.764 eV
[PASS] no astronomical Im sigC            max|Im sigC| =  0.0114 eV
[PASS] Sigma sane @ Gamma VBM/CBM, @ K VBM/CBM
[PASS] direct gap positive at every k     min = 2.6356 eV @ k#52 (= K)
[PASS] indirect gap positive              2.5079 eV
[PASS] QP gap opens vs DFT                1.6954 -> 2.5079 eV
[PASS] rank-truncate path active          16x
[PASS] no NaN/Inf, no errors in gw.out
GATE: ALL PASS
```

Logged in `KNOWN_SANDBOX_ERRORS.md` with a suggested fix (warn when the knob is
requested and the cap refuses it).

---

## 3. Stage 3 — the converged G0W0

`00b_lorrax_gw_2400c_ranktrunc`, GN-PPM, `zeta_rcond = 1e-8`,
`charge_zeta_solve = rank_truncate`, ξ-floor active, QP/Σ for bands 1–80,
nband = 326 (26 val + 300 cond), 144 q, 16 x A100-80GB.

### Gaps

| | DFT | eqp0 | **eqp1** | 6x6 / 30 Ry predecessor (eqp1) |
|---|---|---|---|---|
| direct @ K | 1.6954 | 2.9337 | **2.6356** | 2.942 |
| indirect | 1.6954 | 2.7230 | **2.5079** | 2.885 |
| direct @ Γ | 2.7925 | 3.9519 | 3.6798 | — |
| VBM location | K | K | **K** | **Γ** (by 0.06 eV) |
| CBM location | K | Λ = (0.167,0.167) | Λ | K |

QP shifts (eqp1): at K, VBM **+0.3206**, CBM **+1.2608**, gap opening **+0.9402 eV**;
at Γ, VBM +0.1521, CBM +1.0395, opening +0.8873 eV.

### Did the VBM move to K? Yes, and it was already there in the DFT

`VBM(K) − VBM(Γ)`: **+0.1297 eV (DFT)**, **+0.2982 eV (GW eqp1)**. So on the
converged grid the VBM is at K in *both* theories, and GW pushes it further.
The 6x6 / 30 Ry run's Γ-VBM was a convergence artefact of the input band
structure, not of the self-energy — which is exactly the kind of thing this
campaign existed to settle. The CBM has moved off K to the Λ point
(0.167, 0.167) under the QP correction, so the converged material is a
K→Λ **indirect** semiconductor at the G0W0 level, direct gap 2.64 eV at K.

Against literature: **2.636 eV lands inside the 2.5–2.8 eV G0W0 band** for
monolayer MoS2. The 6x6/30 Ry answer (2.942) was above it; the shift is
−0.31 eV and comes from the converged k-grid + cutoff, not from any knob.
eqp0 (2.934 eV) is *not* the number to compare — without the Z-linearization it
overshoots by ~0.30 eV, and that gap between eqp0 and eqp1 is itself a useful
sanity signal that the PPM Σ(ω) slope is being resolved.

### Under-completeness at n_μ/nband = 7.40x

Two different questions hide under "under-complete", and at 12x12 they answer
oppositely:

* **The charge CCT is mildly OVER-complete**, and the truncation is still doing
  work: rank-truncated vs the (silent) plain Cholesky moves per-band QP energies
  by mean 11.5 meV (valence) / 31.5 meV (bands 27–60) / **81.4 meV (bands 61–80)**,
  max 93.2 meV at band 65 — see §4(b). If n_μ were comfortably *under* the
  pair-density rank, truncation would be a no-op. It is not.
* **The htransform Galerkin basis is UNDER-complete**, badly, and this is new at
  12x12. That basis is an SVD of `(nk·nb, nspinor·n_μ)`; the 6x6 predecessor had
  36 x 60 = 2160 states against rank 3178 (states < rank, `ctilde` orthogonality
  error **2e-14**). Here 144 x 48 = 6912 states against rank 4824 or 2472 —
  states **exceed** rank, and the orthogonality error is **4.9e-1**. That is a
  genuine capacity failure, and it is what breaks stage 5 (§5).

So the honest answer to "does 7.4x show under-completeness": **not in the Σ
itself** — the gate passes, Σ_c is O(1–6 eV), Im is ≤ 11 meV, the gap is sane and
literature-consistent. But 7.4x is *not* enough for the interpolation machinery
at 144 k-points, where the requirement is `nspinor·n_μ > nk·nb`, i.e. n_μ > 3456
for a 48-band window — a **k-grid-driven** requirement that has nothing to do
with the 8x/12x per-band guidance in the docs.

---

## 4. Stage 4 — do the mitigation knobs still need their settings?

Measured **on the converged data**, one recommendation each.

### (a) The ξ-floor (`ppm_windows._CROSSING_A_MAX = 24`) — **KEEP, and consider tightening**

The floor engaged in production: `Σc crossing conditioning: ξ raised 0.250 →
0.952 eV (A_core capped at 24)`. The floor is
`ξ_min = 2·ω_max/(A_max − 2·edge)`; with ω_max = 10 eV and
`sigma_window_edge_factor = 1.5` that is exactly 0.9524 eV.

`xi_probe.py` measures the thing the floor exists to control — the crossing
quadrature's weight sum `Σ|α̂|`, which multiplies **any** perturbation of the
per-τ operand σ(τ):

| ξ (eV) | A_core | n_τ | **Σ\|α̂\|** | fit err | regime |
|---|---|---|---|---|---|
| 0.250 (as requested in `cohsex.in`) | 83.0 | 103 | **8.50e5** | 6.9e-7 | ill-conditioned |
| 0.500 | 43.0 | 66 | 1.56e3 | 9.4e-7 | ill-conditioned |
| 0.750 | 29.7 | 48 | 4.14e1 | 1.5e-7 | stressed |
| **0.952 (the floor)** | **24.0** | **48** | **4.14e1** | 1.5e-7 | stressed |
| 1.500 | 16.3 | 25 | **6.32e-1** | 4.4e-7 | OK |
| 2.000 | 13.0 | 25 | 6.32e-1 | 4.4e-7 | OK |

**This table cannot improve with convergence.** `A_core = 2·ω_max/ξ + 2·edge`
depends only on the Σ ω-grid and `edge_factor` — not on `ecutwfc`, not on n_μ,
not on the ISDF. So the amplifier is identical at 30 Ry and 80 Ry; what
convergence can shrink is the *perturbation being amplified*, not the
amplification. Turning the floor off is a **2 x 10⁴ increase** in that
amplification, and it also more than doubles the τ-node count (48 → 103), which
by itself pushed the Σ stage from fitting on 4 GPUs to needing 16.

**But the amplified quantity has collapsed.** `_stage4/a_xi_lifted` reruns the
production Σ with the floor disabled in-process (`gw_probe.py --lift-xi`), i.e.
at ξ = 0.25 eV, A_core = 83, `Σ|α̂| = 8.5e5` — the exact regime that produced
O(1e5) eV device-dependent Σ_c before the cures. At 80 Ry / n_μ = 2412 with the
rank-truncated CCT:

| | floored (ξ = 0.952) | lifted (ξ = 0.25) | Δ |
|---|---|---|---|
| direct gap @ K | 2.6356 eV | 2.6289 eV | **6.7 meV** |
| indirect | 2.5079 eV | 2.5000 eV | 7.9 meV |
| per-band max \|Δ⟨E_QP−E_DFT⟩\| | — | — | **23.8 meV** (band 7) |
| bands 27–60 / 61–80 mean | — | — | 2.5 / **0.7 meV** |

**Say it plainly: this defect does not appear at convergence.** A 2 x 10⁴
amplifier applied to a well-conditioned W moves the gap by 7 meV. The blow-up the
ξ-floor was written for is gone — cured at source by the rank-truncated CCT and
the band-range centroids, exactly as intended.

`_stage4/a_xi_tight` closes the loop from the other side: ξ = 1.5 eV, *above*
the floor (so the floor never engages), A_core = 16.3, `Σ|α̂| = 0.63`, 25 τ-nodes.

| variant | ξ (eV) | A_core | Σ\|α̂\| | n_τ | direct @ K | Δ vs prod | indirect | Δ |
|---|---|---|---|---|---|---|---|---|
| `a_xi_lifted` | 0.250 | 83.0 | 8.5e5 | 129 | 2.6289 | **−6.6 meV** | 2.5000 | −7.9 |
| **production** | **0.952** | **24.0** | **41.4** | **48** | **2.6356** | — | **2.5079** | — |
| `a_xi_tight` | 1.500 | 16.3 | 0.63 | 25 | 2.6440 | **+8.4 meV** | 2.5169 | +9.1 |

**ξ spans a factor 6 and the gap moves 15 meV end to end.** The knob is
essentially inert at convergence — so choose it for cost and margin, not
accuracy.

**Recommendation: KEEP the floor, and CHANGE `_CROSSING_A_MAX` from 24 to ~16.**
The floor itself is unambiguous — it costs nothing, and it *pays*: 48 τ-nodes
instead of 129 makes the Σ stage ~2.7x cheaper (the lifted variant needed 16
GPUs where the floored one fits on 4). Tightening it to A_max ≈ 16 buys another
**2x** on τ-nodes (25 vs 48) and **65x** more conditioning margin (`Σ|α̂|` 41.4
→ 0.63), for **+8 meV** on the gap. That is a good trade on every axis, and the
margin matters because the amplifier is a property of the quadrature that will
come back the moment a system is less well conditioned than this one.

### (d) The htransform `b_max` ceiling — **the question no longer applies at 12x12**

At 30 Ry / 6x6 the ceiling was about *centroid weighting*: occupied-ρ centroids
made the on-grid QP reconstruction explode (3.3e5 meV at b_max = 40), band-range
centroids kept it at 4.5e2 meV, so the ceiling was "lifted". At 80 Ry / 12x12
that framing is superseded: the binding constraint is no longer the weighting
but **`nspinor·n_μ > nk·nb`**, and `nk` has quadrupled. Attempting the sweep:

| attempt | outcome |
|---|---|
| n_μ = 2412, nb = 80, 1 GPU | OOM — Galerkin band chunk was a fixed 64 → 51.6 GB in one allocation |
| same, after the band-chunk fix (`5e50b8e`) | got further; OOM in `accum` (75.5 GiB) |
| n_μ = 2412, nb = 80, 2x2 mesh | OOM — replicated `fH_R`, 49.9 GiB/device |
| n_μ = 1236, nb = 48 | ran, but `ctilde` orthogonality **4.9e-1** → interpolated DFT bands came back with a **negative** indirect gap (−1.20 eV) |
| n_μ = 1236, gap-centred window (bands 18–34) | `rank = 0`, `σ_max = 0` — a non-zero `b_start` is separately broken |

**Recommendation:** the b_max ceiling is no longer the right knob to reason
about; at a converged k-grid the htransform needs a capacity rule
(`nspinor·n_μ > nk·nb`) that its memory layout cannot afford. That is filed as a
real limitation (§5, `KNOWN_SANDBOX_ERRORS.md`), not a tuning parameter. The one
concrete fix that came out of it — sizing the Galerkin band chunk to the ψ box
instead of a hard-coded 64 — is committed (`5e50b8e`) and is what made stage 5
runnable at all.

### (b) `zeta_rcond = 1e-8` / rank-truncation — **KEEP the truncation; the default value is fine**

The campaign produced the cleanest possible A/B by accident: run `00` took the
plain distributed Cholesky (no truncation at all — §2), run `00b` is byte-identical
input with the intended rank-truncated route. Everything else — WFN, centroids,
nband, ω-grid, ξ — is the same.

| | `00` cusolvermp_cholesky | `00b` replicated_rank_truncate (1e-8) | Δ |
|---|---|---|---|
| direct gap @ K (eqp1) | 2.6653 eV | 2.6356 eV | **29.7 meV** |
| indirect (eqp1) | 2.5208 eV | 2.5079 eV | **12.9 meV** |
| Σ_x, k=0, band 1 | −39.8892 eV | −39.8909 eV | 1.7 meV |

Per-band mean \|Δ⟨E_QP − E_DFT⟩\| across all 144 k:

| band range | mean | max |
|---|---|---|
| valence 1–26 | 11.5 meV | 34.9 meV |
| conduction 27–60 | 31.5 meV | 61.5 meV |
| far conduction 61–80 | **81.4 meV** | **93.2 meV** (band 65) |

So at convergence the CCT is **still over-complete enough that truncation
matters** — monotonically more so with band index, which is the signature of
near-null directions of the charge CCT feeding the high-lying states. It is not
a catastrophe (tens of meV, not the tens of eV the cure was introduced for), but
it is not a no-op either, and *the gap itself moves by 30 meV*. **Keep
`charge_zeta_solve = rank_truncate`.**

On the *value*: this campaign did not re-measure the 1e-8…1e-4 plateau, because
every `zeta_rcond` variant needs a fresh ζ fit (no restart) and the remaining
allocation went to stages 4(a)/(c) and 5. The existing evidence stands — the
plateau spans 1e-8…1e-4 and the low end costs 20x less drift on the BGW-anchored
Si gate — and nothing measured here contradicts it. **Recommendation: keep 1e-8;
the outstanding measurement is the plateau at 12x12, ~2 x 12 min of 16-GPU time.**

The far more important finding for this knob is the *cap*, not the value: at
production scale the knob was silently discarded (§2). Fix that first.

### (c) The far-band clamp and the unwired affine scissor — **the clamp is real; the scissor is unreachable, and wiring it is NOT the fix**

Two separate things get conflated under "far-band scissor":

**The clamp is real and it is large.** Σ_c(ω) is tabulated on
`[−10, +10] eV` relative to the midgap Fermi reference (−4.3977 eV here), i.e.
`[−14.40, +5.60] eV` absolute, and the evaluation energy is clipped to that
window. Production log: **`QSGW: 6942 clipped (60.3%)`**. Of the 80 QP-corrected
bands, only **28 are fully inside the grid at every k**, 6 more are partly in,
and the first conduction band that leaves the grid and never returns is **band
41**. So bands 41–80 — half the QP window — get Σ_c evaluated at the grid edge.

**But the gap is clamp-free.** The VBM (band 26, −1.91…−0.85 eV rel. E_F) and CBM
(band 27, +0.85…+1.82 eV) sit comfortably inside. The clamp cannot touch the
2.636 / 2.508 eV numbers; it only degrades the far-conduction QP energies, which
is precisely where the reported band structure should not be trusted quantitatively.

**The affine extrapolator cannot touch eqp0/eqp1 — proven, not argued.**
`gw/scissor.py` (`classify_bands_in_grid` + `fit_scissor`) is called from exactly
one place, `qsgw_utils.solve_qp`, and that call sits **after** an early return:

```python
if qp_solver is not QPSolver.FIXED_POINT or sigma_c_omega is None:
    ...
    return sigma_result.sigma_xc_kij_ry + sig_h     # one_shot_dft exits here
```

`gw_config` documents `sigma_at_dft_extrapolate` as *"a sub-knob of
[FIXED_POINT]"* and it defaults to `False`. Two variants pin this down:

* `_stage4/c_fixedpoint` — `qp_solver = fixed_point`, scissor off. The branch IS
  reached: `Diagonal SC: 28/80 bands fully in grid, 20 iterations`.
* `_stage4/c_scissor_extrap` — same, scissor **on**. It fits and prints
  `ScissorFit(val: α=+1.0805, β=+0.1193 eV, rmse=0.130 eV; cond: α=+0.9225,
  β=+1.3472 eV, rmse=0.115 eV)`, so the machinery works.

And **both produce `eqp0.dat` / `eqp1.dat` byte-identical to the production
one-shot run apart from the timestamp line** (`diff` over non-comment lines: 0
differences; summary table: 0.0 meV on every band segment). The QP solver and the
scissor only reach the QSGW-symmetrised Σ_xc → `WFN_qp.h5` /
`qp_wfn_rotations.h5` eigen-path; the eqp text outputs are built by
`eqp_bgw.py` from Σ(E_DFT) regardless. **The scissor has never been "doing harm
out to band 80" in any number this campaign quotes — it cannot reach them.**

**And widening the ω-grid is NOT a safe cure — measured.** `_stage4/c_wide_omega`
reruns Σ on ±25 eV. Clipping falls exactly as intended, 60.3 % → **10.0 %**. But:

| | production ±10 eV | `c_wide_omega` ±25 eV |
|---|---|---|
| clipped (k,n) | 6942 (60.3 %) | **1154 (10.0 %)** |
| ξ (forced by the floor) | 0.952 eV | **2.381 eV** |
| static minimax window / PPM poles | R = 80.91, 10 nodes, err 2.6e-6 | **identical** |
| direct gap @ K | 2.6356 eV | **3.6520 eV (+1016 meV)** |
| per-band max \|Δ⟨E_QP−E_DFT⟩\| | — | **5.09 eV** |

W and the PPM pole construction are **bit-identical** between the two (same
window, same nodes, same fit error), so the entire +1.0 eV comes from the ξ the
floor forced: `ξ_min = 2·ω_max/(A_max − 2·edge)` scales with ω_max, and ±25 eV
demands **2.381 eV**. Put the ξ series together —

| ξ (eV) | 0.250 | 0.952 | 1.500 | 2.381 |
|---|---|---|---|---|
| Δ direct gap | −6.6 meV | 0 | +8.4 meV | **+1016 meV** |

— and the shape is obvious: ξ is harmless while it stays small compared with the
distance from the evaluation energy to the nearest pole, and 2.4 eV is not.

**Recommendation:** (i) **do not wire the affine scissor into `one_shot_dft`** —
extrapolating Σ from an ω-grid the band never visits is a worse fix than
sampling where the bands are, and the knob is unreachable there anyway;
(ii) **do not widen ω alone** — at fixed `A_max = 24` it buys a 6x reduction in
clipping and pays 1 eV on the gap; (iii) if the far bands are genuinely wanted,
widen ω **and** raise `_CROSSING_A_MAX` together so ξ stays ≲ 1 eV — and §4(a)
says the conditioning penalty for that is now only ~7 meV. (a) and (c) are one
knob, not two. (iv) For gap work, keep ±10 eV and simply do not quote QP
energies above band ~40.

---

## 5. Stage 5 — the exciton bandstructure, and the wall that moved it

### What blocked the stage-3 restart

`bse.exciton_bands` gets ψ_c(k+Q), ε_c(k+Q) from the htransform. Two constraints
collide at a converged k-grid:

* **capacity** — the Galerkin basis is an SVD of `(nk·nb, nspinor·n_μ)`; it needs
  `nspinor·n_μ > nk·nb` or `ctilde` stops being orthonormal and the fH energy
  recovery collapses;
* **memory** — `bandstructure/bse_setup.py:156` does
  `fH_R_rep = jax.device_put(fH_R, rep)` with `rep = NamedSharding(mesh_xy, P())`,
  i.e. `fH_R (nk, ns·n_μ, ns·n_μ)` complex128 **replicated on every device**.

At nk = 144, nb = 48: capacity wants n_μ > 3456, at which memory costs
`144·6912²·16 B` = **102 GiB per device**. The cost is quadratic in n_μ and
independent of the mesh, so no device count resolves it. Measured, on the
n_μ = 1236 12x12 producer: `ctilde` orthogonality **4.9e-1**, and the driver's own
gate refused to continue —

```
[gate] htransform@Γ vs stored: max|Δε_c| = 3577.920 meV,
       conduction-subspace overlap min-sval = 0.0002
AssertionError: htransform conduction cache grossly inconsistent with the
stored grid — interp basis broken.
```

That assertion is correct and valuable: the alternative was a plausible-looking
exciton band structure built on a broken interpolant. At n_μ = 2412 the run never
got that far — `fH_R` alone is 49.93 GiB/device and OOMs the 80 GB card.

### The route that works

A 6x6x1 NSCF (120 bands) on the **same converged 80 Ry SCF density** — SCF
reused unchanged, same cell, pseudos, cutoff — then a GW with 1452 band-range +
D3h centroids, then `bse_k_grid = 12 12 1` (the validated coarse→fine economy
route: W zero-padded in R, ψ/ε via htransform, V_Q0 rebuilt at the fine mini-BZ
head). Capacity at nk = 36, nb = 48: 1728 states < rank 2904. `fH_R` = 4.5 GiB.
Both satisfied. Cost: NSCF 14 s, pw2bgw + wfn2hdf 73 s, GW 87 s.

This is still a strict improvement on the previous exciton figure, which used a
**30 Ry** 6x6 GW; the BSE grid (12x12) is unchanged, the reference under it is
converged.

### A third thing the 6x6 route needed: full-BZ ζ storage

`bse.vq_interp` requires the ζ tensor on the **full** BZ, but the GW writes
IBZ-only whenever the centroid set passes orbit closure — which the D3h-closed
production centroids do by construction. First 6x6 GW wrote `zeta_q_G (20, 1452, 8603)`
and the BSE stopped with

```
ValueError: vq_interp needs FULL-BZ zeta storage: zeta_q.h5 has nq=20 but the
k-grid has nk=36 (IBZ cascade active).
```

`gw_init` exposes the escape hatch as an environment variable —
`LORRAX_FORCE_FULL_BZ=1` — so the producer keeps D3h orbit closure (and with it
the V_H k-star symmetry cure) **and** writes the 36-q tensor the BSE needs. Note
the interaction: **the D3h centroid cure and the BSE consumer are in tension by
default**, and the only thing joining them is an undocumented env var. Worth a
config key and a clearer error.

---

## 6. Files

### Deliverables

| path | what |
|---|---|
| `plots/gw_bands_converged.png` | **GW bandstructure figure** — DFT vs G0W0 on the computed 12x12 k-points along Γ–M–K–Γ |
| `plots/mos2_exciton_bandstructure.png` | **exciton bandstructure figure** (§5) |
| `runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/manifest.yaml` | run manifest, `yaml.safe_load`-validated |

### Scripts (all in this report dir)

| script | role |
|---|---|
| `sanity_gate.py` | the stage-3 gate — Σ sanity, gap sign, Im bound, solver-route assertion |
| `gaps.py` | direct/indirect gaps, VBM/CBM location, VBM(K)−VBM(Γ), QP shifts |
| `farband.py` | ω-grid membership per band; clamp diagnosis; two-run per-band QP diff |
| `xi_probe.py` | crossing-quadrature conditioning `Σ\|α̂\|(ξ)` — the ξ-floor amplifier |
| `stage4_summary.py` | one table: every stage-4 variant vs the baseline |
| `head_convergence.py` | q=0 head (`wcoul0`) vs the dipole band window |
| `bmax_sweep.py` | htransform b_max sweep (blocked at 12x12, §4d) |
| `gw_bands_ongrid.py` | the GW bandstructure figure (no interpolant) |
| `gw_bands_figure.py` | htransform figure route (kept; unusable at 12x12, §4d) |
| `plot_exciton.py` | exciton figure, GW-referenced |
| `gw_probe.py` | **one** experiment driver: `--cap-gib` (replication cap), `--lift-xi` (ξ floor). No source edits. |
| `size_gw.py` | planner sizing, 60 vs 65 GB |
| `run_gw.sh`, `run_stage4_restart.sh`, `run_stage4_side.sh`, `run_exciton.sh`, `run_qe6x6.sh`, `setup_stage4.sh` | launchers |

### The one source change

`5e50b8e` — `bandstructure/htransform.py`: the Galerkin `band_chunk_size` is now
a **ceiling**, lowered so one streamed ψ chunk `(nk, bc, nspinor, n_rtot)` c128
stays under 6 GiB. It was a fixed 64 = 51.6 GB in a single allocation on this
reference, which made every htransform consumer unrunnable. Band chunking is a
pure accumulation split (`G = Σ_chunks Q Qᴴ`), so this is memory-only.

### Caveats carried forward

* **`dipole.h5` is built on a reduced band window** (`nval 26 / ncond 38 /
  nband 64`) because `psp.get_dipole_mtxels` is single-process and materialises
  a 262 GB array at the production `nband = 326`
  (`KNOWN_SANDBOX_ERRORS.md`). It feeds only the q=0 head. Measured sensitivity:
  `wcoul0` = 3443.96 Ry at nband = 64 vs 3529.57 Ry at nband = 40, i.e. **2.5 %**
  — so the head is *not* converged in this window, worth roughly 10–20 meV on
  the QP energies and much less on the gap (VBM and CBM shift together). The
  production `W_h(q→0, ω=0) = 3443.963338` in `gw.out` matches the nband = 64
  reproduction exactly, so the pipeline is doing what this control says it is.
* **`Meta` has no `ngkmax`**, so the planner always uses the `0.06·n_rtot`
  heuristic (10497 vs the true 8603 here, +22 %), and `band_chunk_size` defaults
  to 16 rather than 0 so the planner's picker is overridden. Both are known and
  were mirrored in the sizing run so the plan is production-faithful.
* The **`zeta_rcond` plateau was not re-measured at 12x12** (§4b) — the one
  outstanding stage-4 measurement.

---

## 7. Stage-4 recommendations at a glance

| knob | current | verdict | evidence |
|---|---|---|---|
| **(a) ξ-floor `_CROSSING_A_MAX`** | 24 (→ ξ ≥ 0.952 eV) | **KEEP the floor · CHANGE the ceiling 24 → ~16** | ξ over 0.25–1.5 eV moves the gap by 15 meV total; the floor halves τ-nodes (129 → 48) and A_max = 16 halves them again (→ 25) for +8 meV, with 65x more conditioning margin |
| **(b) `charge_zeta_solve = rank_truncate`** | on | **KEEP** — and **FIX the 4 GiB cap that silently disables it** | truncation still worth 30 meV on the gap and up to 93 meV per band at convergence; the cap made it a no-op at production scale |
| **(b′) `zeta_rcond`** | 1e-8 | **KEEP** (not re-measured at 12x12) | prior plateau evidence stands; outstanding measurement ~2 x 12 min |
| **(c) far-band ±10 eV clamp** | ±10 eV | **KEEP** — widening ω alone costs 1 eV on the gap | 60.3 % of (k,n) clipped, but VBM/CBM interior so the gap is clean; ±25 eV cuts clipping to 10.0 % and moves the gap **+1016 meV**, entirely via the ξ = 2.381 eV the floor then forces |
| **(c′) affine `gw.scissor`** | off, and unreachable from eqp | **DO NOT WIRE** | fixed-point + scissor **on** gives eqp0/eqp1 byte-identical to production (0.0 meV, all bands); the fit itself is fine (val α=1.08/β=+0.12 eV, cond α=0.92/β=+1.35 eV) but only reaches the QSGW eigen-path |
| **(d) htransform `b_max`** | — | **question superseded** | at 12x12 the binding rule is `nspinor·n_μ > nk·nb`, not centroid weighting |

**On "several of today's defects may simply not appear at convergence":** one
did exactly that — **the ξ-floor's blow-up is gone** (7 meV, not 10⁵ eV). The
rank-truncation's is not gone, just shrunk from eV to tens of meV. And two new
defects appeared *only* at convergence: the 4 GiB replication cap silently
disabling two cures, and the htransform capacity/memory squeeze. Converging the
parameters did not remove the need for the cures; it removed the need for one of
them and exposed two scale bugs the 6x6 runs could not reach.

---

## 8. What to do next, in priority order

1. **Warn (or fail) when `charge_zeta_solve` is silently discarded.** One
   `print` in `_resolve_solver_kind_charge` when the knob is `rank_truncate` and
   `_replicate_charge_ok` is False, naming the cap and the actual stack size.
   Everything else on this list is cheaper than the confusion this one causes.
2. **Un-replicate `fH_R`** in `bandstructure/bse_setup.compute_wfns_fi`. Until
   then, no BSE and no htransform bandstructure at a converged k-grid, on any
   hardware. Sharding the leading `nk_co` axis (with an all-gather per q-batch)
   or keeping `fH_k` and doing the R-sum inside the batch are both plausible.
3. **Tighten `_CROSSING_A_MAX` 24 → 16** — measured +8 meV on the gap for 2x
   fewer τ-nodes and 65x conditioning margin.
4. **Raise the 4 GiB replication cap, or add a replicated-in-chunks route**, so
   the production-scale ζ-solve keeps the cure rather than trading it for memory.
5. **Fix `psp.get_dipole_mtxels`** — `init_jax_distributed()` + the `g_flat`
   loader path, so `dipole.h5` can be built at the production band window and the
   q=0 head stops being a 2.5 %-uncertain input.
6. **Measure the `zeta_rcond` plateau at 12x12** (2 runs, ~24 min of 16-GPU time).
7. Bind `ngkmax` onto `Meta` and default `band_chunk_size` to 0, so the planner
   stops being overridden and over-conservative by construction.

### Getting the interpolation basis healthy: what actually mattered

Three settings had to move together, and the driver's gate metric
(`max|Δε_c|` at Γ, threshold 50 meV; `min-sval` > 0.5) tracks all three:

| configuration | n_k | n_μ | nband window | `ctilde` orth. err | gate `max\|Δε_c\|` / `min-sval` |
|---|---|---|---|---|---|
| 30 Ry predecessor | 36 | 1496 | 48 | 2e-14 | 9.5 meV / 0.88 — **pass** |
| 80 Ry native 12x12 | 144 | 1236 | 48 | 4.9e-1 | 3577.9 meV / 0.0002 — **fail** |
| 80 Ry 6x6 | 36 | 1452 | 48 | 4.5e-14 | 361.3 meV / 0.002 — **fail** |
| **80 Ry 6x6, narrowed** | **36** | **2382** | **40** | **2.3e-14** | **pass** |

Note the middle two rows: `ctilde` orthogonality is *perfect* at 4.5e-14 while
the gate still fails at 361 meV. **Orthonormality of the Galerkin coefficients
is necessary but not sufficient** — the fH energy recovery can still be wrong,
which is exactly why the driver gates on the recovered ε and not on the
orthogonality. The two extra things needed at 80 Ry, beyond dropping to 6x6:

* **more centroids for the same k-grid** — the 80 Ry real-space grid is
  174 960 points vs 46 080 at 30 Ry, so 1452 centroids sample it 3.8x more
  thinly than 1496 did before; 2382 restores enough resolution;
* **a narrower interpolation window** — nband 48 → 40 (the BSE window bands
  18–34 plus 6 conduction guards), which is the gate's own printed advice
  (*"the interp basis is over-packed … Reduce nband toward the BSE window"*).

---

## 9. Stage-4 variant table (all measured, one baseline)

Baseline = `00b_lorrax_gw_2400c_ranktrunc`: direct@K 2.6356 eV, indirect 2.5079 eV,
80 QP bands. Every Σ-only variant restarts from its ISDF tensors, so the named
knob is the only difference. `mean |Δ⟨E_QP−E_DFT⟩|` averages over all 144 k.

| variant | knob | direct@K | Δ gap | indirect | Δ ind | val | c27–60 | c61–80 | max |
|---|---|---|---|---|---|---|---|---|---|
| **baseline** | — | **2.6356** | — | **2.5079** | — | — | — | — | — |
| `a_xi_lifted` | ξ-floor off (ξ = 0.25) | 2.6289 | **−6.6** | 2.5000 | −7.9 | 7.2 | 2.5 | 0.7 | 23.8 |
| `a_xi_tight` | ξ = 1.5 (A_max ≈ 16) | 2.6440 | **+8.4** | 2.5169 | +9.1 | 6.0 | 3.2 | 1.0 | 17.6 |
| `c_wide_omega` | Σ ω-grid ±25 eV | 3.6520 | **+1016.4** | 3.2562 | +748.3 | 1712 | 568 | 2098 | 5089 |
| `c_fixedpoint` | `qp_solver = fixed_point` | 2.6356 | **0.0** | 2.5079 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `c_scissor_extrap` | + `sigma_at_dft_extrapolate` | 2.6356 | **0.0** | 2.5079 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `00_lorrax_gw_2400c` | plain Cholesky (no truncation) | 2.6653 | **+29.7** | 2.5208 | +12.9 | 11.5 | 31.5 | 81.4 | 93.2 |

(meV throughout except the gap columns, which are eV.)

Read top to bottom: **ξ is inert (±8 meV), the ω-grid width is not (1 eV, via ξ),
the QP-solver/scissor axis is exactly zero on these outputs, and the ζ-solve route
is worth 30 meV on the gap and ~90 meV on the far bands.** Only two of the five
rows change a number anyone would quote, and one of them (`c_wide_omega`) changes
it in the wrong direction for the wrong reason.

