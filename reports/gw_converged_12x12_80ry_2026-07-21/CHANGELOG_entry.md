## 2026-07-21: converged MoS2 G0W0 (80 Ry / 12x12 / 400 bands) — VBM moves to K, gap 2.64 eV, and two of the five cures were silently off at production scale

Branch `agent/gw-converged-campaign` (worktree `sources/worktrees/lorrax_gw_converged`),
off `agent/gw-conduction-postfix` @ `b7654ee`, merged with `agent/bse-figures` @ `cb20681`.
Report `reports/gw_converged_12x12_80ry_2026-07-21/`.

**1. The converged G0W0.** 2412 band-range + D3h centroids (208 orbit reps x the
12-op recovered density point group, weighted over the FULL nband range),
nband = 326, QP for bands 1–80, GN-PPM, `zeta_rcond = 1e-8`, 16 x A100-80GB,
`memory_per_device_gb = 65`. **703 s** end-to-end (the sizing report's
extrapolation said 1 h 17 — 6.6x conservative).

| | DFT | eqp0 | **eqp1** | 6x6 / 30 Ry (eqp1) |
|---|---|---|---|---|
| direct @ K | 1.6954 | 2.9337 | **2.6356** | 2.942 |
| indirect | 1.6954 | 2.7230 | **2.5079** | 2.885 |
| VBM | K | K | **K** | **Γ** by 0.06 eV |

**The VBM moved to K** — `VBM(K) − VBM(Γ)` = +0.130 eV in the DFT input and
+0.298 eV after GW, so the earlier Γ-VBM was a k-grid/cutoff artefact, not a
self-energy one. The CBM moves off K to Λ = (⅙,⅙) under the QP correction.
**2.636 eV lands inside the 2.5–2.8 eV monolayer-MoS2 G0W0 literature band**
(the 6x6 answer sat above it).

**2. Two of the five conditioning cures were silently inactive at production
scale**, caught by the stage-3 sanity gate (all physics checks passed; the route
check failed). `isdf/core._replicate_charge_ok` caps the *replicated* ζ-solve
route at `nq·n_rmu²·16 ≤ 4 GiB`, and **only that route carries the
rank-truncation (`23af6b9`) and the mesh-invariant replicated factor
(`ca78008`)**. The converged run is 74·2416²·16 = **6.44 GiB**, so
`charge_zeta_solve = rank_truncate` was parsed, accepted and **ignored** with no
warning. The 6x6 A/B that validated the cures could never have shown this.
Re-run with the cap raised in-process → gate ALL PASS. Measured cost of the
silent fallback: direct gap 29.7 meV, indirect 12.9 meV, per-band QP up to
93.2 meV (band 65). Logged in `KNOWN_SANDBOX_ERRORS.md`.

**3. Planner calibration** (the coordinator's ask). At `memory_per_device_gb = 65`
the plan is `r_chunk = 10192 (18 chunks)`, `HWM 55.22 GB/dev`; at 60 it is
`9248 (19 chunks)`, `50.95` — one chunk, not many, because HWM is pinned at 85 %
of whatever budget it gets. Three numbers that should not be conflated:
planner **55.22**, LORRAX's own live high-water **54.15** (planner accurate to
**2 %**), `nvidia-smi` arena **79.4–79.8 GB of 81.9 GB (97 %)** — a factor
**1.44** over the planner, not 14 %. 65 was right and was near the ceiling.

**4. Stage-4 knob study on converged data.** (a) **ξ-floor: KEEP.** `Σ|α̂|` for the
crossing quadrature is 8.5e5 at the requested ξ = 0.25 eV vs 41.4 at the 0.952 eV
floor — and `A_core` depends only on (ω_max, ξ, edge), never on cutoff or n_μ, so
this cannot improve with convergence. Tightening `_CROSSING_A_MAX` 24 → ~16
(ξ = 1.5 eV, `Σ|α̂| = 0.63`, 25 τ-nodes instead of 48) is the cheap follow-up.
(b) **rank-truncation: KEEP**; still worth 10–90 meV at convergence, growing with
band index. (c) **the ±10 eV clamp is real** (`QSGW: 6942 clipped (60.3%)`; only
28/80 bands fully in grid; first permanently-out band 41) **but the gap is
clamp-free** (VBM/CBM sit at ∓2/+1 eV). The affine `gw.scissor` extrapolator is
**unreachable from `one_shot_dft`** — `solve_qp` early-returns before it — so
`sigma_at_dft_extrapolate` is a no-op on every eqp output; **do not wire it**,
widen the ω-grid instead (which raises the ξ floor: (a) and (c) are coupled).
(d) the **b_max ceiling is superseded**: at 12x12 the binding constraint is a
capacity rule `nspinor·n_μ > nk·nb`, not centroid weighting.

**5. A new scale wall, only visible at a converged k-grid.** The htransform /
BSE machinery needs `nspinor·n_μ > nk·nb` for its Galerkin basis, while
`bse_setup.compute_wfns_fi` **replicates** `fH_R (nk, ns·n_μ, ns·n_μ)` on every
device. At nk = 144, nb = 48 those demand n_μ > 3456 and 102 GiB/device
respectively — incompatible. The 12x12 attempt gave `ctilde` orthogonality
4.9e-1 and the exciton driver's own gate correctly aborted
(`max|Δε_c| = 3577.9 meV, min-sval = 0.0002`). Stage 5 therefore runs
6x6 → 12x12 via `bse_k_grid` on a 6x6 / 80 Ry NSCF of the same SCF density.
Both limits are in `KNOWN_SANDBOX_ERRORS.md`.

**6. Code change** (`5e50b8e`): `bandstructure.htransform.streaming_galerkin_solve`
now treats `band_chunk_size` as a **ceiling**, lowering it so one streamed ψ
chunk `(nk, bc, ns, n_rtot)` stays under 6 GiB. It was a fixed 64, which is
0.81 GB/band here = 51.6 GB in one allocation, and it made every htransform
consumer unrunnable on the converged reference. Memory-only (band chunking is a
pure accumulation split); the converged run picks `band_chunk = 7`.

**Addendum — the (a)↔(c) coupling, measured.** `_stage4/c_wide_omega` (Σ ω-grid
±25 eV instead of ±10) cuts clipping 60.3 % → **10.0 %** as intended, with a
**bit-identical** W and PPM pole construction (same minimax window, 10 nodes,
fit err 2.6e-6) — and moves the K direct gap **+1016 meV** (2.6356 → 3.6520 eV),
per-band QP by up to 5.09 eV. The whole shift is the ξ the floor then forces:
`ξ_min = 2·ω_max/(A_max − 2·edge)` scales with ω_max, so ±25 eV demands
ξ = **2.381 eV**. The ξ series is
−6.6 meV (0.25) · 0 (0.952) · +8.4 meV (1.5) · **+1016 meV (2.381)** —
harmless while ξ stays small next to the pole spacing, catastrophic once it does
not. So **ω-grid and ξ-floor are one knob**: widening ω to cure the far-band
clamp requires raising `_CROSSING_A_MAX` in the same change, and §4(a) shows
that costs only ~7 meV now.

**Addendum — stage 5 (exciton bandstructure) produced NO numbers, and that is
the result.** Eight configurations, each failing for a different documented
reason (full constraint map in the report §5). The last one satisfied every
structural constraint and was refused by the driver's own physics gate:
`max|Δε_c| = 338.2 meV, min-sval = 0.3283` against thresholds 50 meV / 0.5.
Narrowing the interp window fixed the subspace (min-sval 0.002 → 0.328) but not
the energies (361 → 338 meV). The cause is centroid density, not the window or
the k-grid: the 30 Ry run that passed at 9.5 meV had 1496 centroids over 46 080
grid points; 80 Ry has 174 960, so matching it needs **n_μ ≈ 5680**, whose
**replicated** `fH_R` is **69.2 GiB/device** at 6x6 (276.9 at 12x12). Accuracy
and memory are mutually exclusive here. **The BSE/htransform stack has a hard
resolution ceiling set by one replicated array, and an 80 Ry reference is above
it.** Un-sharding `fH_R` (16-way → 4.3 GiB/device at n_μ = 5680) is the unblock.
Nothing was quoted because a 338 meV error in the recovered conduction energies
would have produced plausible-looking exciton bands with meaningless binding
energies — the gate was right to refuse.
