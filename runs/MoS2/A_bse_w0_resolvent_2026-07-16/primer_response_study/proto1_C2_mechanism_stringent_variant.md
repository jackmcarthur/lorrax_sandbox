# proto1 C2 — global BZ-periodic frame, four-tails verdict, spectral-Phi interpolation

**Construction under test** (ARBITRARY_Q_PRIMER_RESPONSE Method B + Strategy A):
build a global smooth BZ-periodic frame for the whitened ISDF sections
`Phi_q = S_q R_q^H zeta_q` by polar-link parallel transport with exact
zone-boundary sewing and log-distributed Wilson obstructions; adjudicate with
the response's own falsification diagnostic (sec 10C): per-shell R-space
tails of four objects; interpolate only if the transported tail decays.

**VERDICT: NEGATIVE — the response's central claim fails its own sec-10C
criterion, with every self-check closed at machine precision and the
mechanism measured.** The whitened pair-feature bundle has no smooth
structure at accessible grid densities: adjacent-q whitened principal
cosines sit at/near the random-subspace floor without transport and reach
only ~0.2-0.3 (median) with proven-covariant transport; plaquette holonomy
is at the random-unitary ceiling; the transported tail `Phi~_R` is *rougher*
than raw `zeta_R` (1.89/1.60 vs 0.39/0.16 vs C_R's 0.023/6.7e-4). Refining
3x3 -> 6x6 does not move the bundle toward smoothness. C2 terminates before
interpolation by the pre-agreed rule; the LOO stage was still run to supply
the physical-metric (owner-pushback) numbers — fit floor, rank ladder,
re-based #3.5 ladder — reported in sec 5.

All scripts/logs/npz in this directory (`proto1_*`). Fixtures: MoS2 3x3
(`runs/MoS2/00_mos2_3x3_cohsex/05_lorrax_cohsex_native`), MoS2 6x6
(`interp_study/mos2_6x6`), both nb=80, ns=2, n_mu=640, zeta cutoff 30 Ry.
Plain numpy/h5py single node (owner prototype ruling). Production mapping:
every eigh/SVD/polar on (n_mu, n_mu) here becomes the N_mu^2 P('x','y')-
sharded cusolvermp/slate FFI form; the per-k H Grams become sharded zgemm
accumulations. Deferred — the physics verdict makes them moot for C2.

---

## 1. Conventions pinned from source (read-only) + corrections found

All derivations in `proto1_prep.py`; every claim numerically closed.

1. **Torus row convention.** `isdf/core.py::c_q_from_psi_sm` builds `C_q` by
   lattice-FFT convolution of same-k pair blocks of stored (wrapped-label)
   u's. Expanding the convolution exactly: `C_q = X_q^H X_q` with SPIN-TRACED
   rows `X[(k,n,m),mu] = sum_s conj(u_{n,wrap(k-q),s}(r_mu)) u_{m,k,s}(r_mu)`,
   no umklapp phases. Verified `relF(X^H X, C_q) = 6.2e-11 / 8.7e-11`
   (accumulation-order floor at cond(C) ~ 1.7e7). The task spec's
   "4-spin-channel rows (m,n,k,a,b)" collapse to these spin-traced rows for
   the charge vertex: the two spin sums factorize per leg.
2. **Seam identities.** In torus labels `q+G0 == q_wrap` (D == I). The
   spec's phys-convention sewing `X(q+G0) == X(q_wrap) D_{G0}`,
   `D_{G0} = diag(e^{-i G0.r_mu})`: verified 4.9e-17. G-sphere relabel
   `zbar_{q+G0}(G) = zbar_q(G+G0)` through the lab recon: 5.4e-16.
3. **q-WRAP TRAP (bug found in MY prep, fixed).** `mf_header/rk` stores the
   unwrapped QE list (0, 1/3, 2/3) but the stored zeta spheres/coefficients
   follow the BGW wrap (2/3 -> -1/3). Using unwrapped q at the 5
   wrap-affected q's puts sphere G's at |q+G|^2 up to 53 Ry on a 30 Ry
   sphere, breaks make_Vq-vs-disk at 0.6-0.8 and fakes a TRS violation of
   med 0.64 exactly on the TRS partners. With the wrap: make_Vq-vs-disk =
   2e-15 (Gamma) / ~1.3e-9 (all finite q); TRS V(-q)==conj(V(q)) = 1.0e-15;
   VcSR +-q eigenvalues 8.4e-15. New self-check `sphere_max|q+G|^2-cutoff = 0`.
   NOTE: interp_study/#3.5 used the unwrapped file q the same way — its
   RELATIVE metrics are unaffected (same v both sides), but its zeta_R
   "flat" table (1.00/0.82/0.65) is the unwrapped-convention object; in the
   correct wrapped lab continuation raw zeta_R = 1.00/0.39/0.16 (weak decay,
   still ~17x above C_R at shell 1, ~230x at shell 2).
4. **blat units.** `mf_header/bvec` is stored in blat units; the physical
   factor blat^2 = 1.104 was measured directly as a 10.4% make_Vq-vs-disk
   residual before the fix. bdot is stored physical (bvec*blat consistency
   verified). Slab kernel: `v = 8pi/K^2 * f2d / V_cell`,
   `f2d = 1 - e^{-zc kxy} cos(kz zc)`, `zc = pi/b_z = Lz/2 = 11.34 Bohr`;
   only the true q=0 G=0 divergence is zeroed (disk includes finite G=0 at
   q != 0; zeroing it at all q breaks the disk match to 0.33).
5. **MATH CORRECTION 1 — K-identity.** With `V[mu,nu] = sum_G conj(zt_mu) v
   zt_nu` (production `v_q_g_flat`: `conj(L)*v @ R.T`) and `Phi = S R^H
   zeta` under the same conj-on-left contraction, the exact identity is
   `K == S R^T V R^* S` — verified 1.2e-14. The spec's `K == S R^H V R S`
   (0.98) and its transpose (0.71) are both wrong in this orientation; the
   spec form holds only in the transposed-V convention (the response's
   implicit `V' = zeta v zeta^H = V^T`). Physical block: `B = conj(a) K a^T`
   with `a = x R S^-1 G(q)` reduces exactly to `conj(x) V x^T` at full rank
   (`conj(R) R^T = I`); the LOO null test enforces the whole chain.
6. **MATH CORRECTION 2 — transport covariance.** The response's pair-space
   rotation `B = t^* (x) I` is NOT gauge-covariant in these conventions.
   The invariant contraction is
   `H[mu,nu] = sum_{k,n,n',m} u^a_n(mu) conj(u_m(mu)) t_k[n,n'] conj(u^b_n'(nu)) u_m(nu)`
   with `t_k = polar(<u_(wrap(k-qa))|u_(wrap(k-qb))>_G)` entering
   UNCONJUGATED (per-element: the band-gauge factors close as V V^H = I only
   in this slot; with conj(t) they close as V V^T != I). Caught by the
   gauge-randomization gate (random within-multiplet unitaries + phases,
   TOL 1e-6 Ry): ||T_rand - T|| = 35 ~ random-unitary distance with
   conj(t); = 5.6e-9 with t (machine roundoff amplified through near-null
   polar directions, min sval(M) ~ 1e-4; the printed FAIL label is a
   too-tight 1e-10 threshold — the gate passes).
7. **Fit-RHS provenance (open, not verdict-relevant).** A non-circular
   rebuild of Z from WFN pair rows (both torus and phys conventions,
   sphere-projected solve) does NOT reproduce the stored zeta~
   (S-weighted relF ~ 0.99), although the same rows' Gram reproduces C_q at
   6e-11 and fftn(recon body) reproduces the stored coefficients at 2.3e-16.
   The production Z ("ortho-FFT ZCT" kernel) evidently differs from the
   naive full-grid pair-row RHS in a way C does not constrain. Not chased
   further: every C2 object is defined from the STORED fit (Phi = S R^H
   zeta_stored), so this affects interpretation of Phi as L^H A only, not
   any measured number.

## 2. Self-check gate (MoS2 3x3, final wrapped-q run)

| check | value | status |
|---|---|---|
| recon/forward sphere round trip | 2.3e-16 | OK |
| sphere max\|q+G\|^2 - cutoff (wrapped q) | 0.0 | OK |
| WFN u(r_mu) vs psi_full_y (k=0) | 2.7e-16 | OK |
| X^H X == C_q (torus) q=0/1 | 6.2e-11 / 8.7e-11 | OK |
| make_Vq(slab) vs disk V_qmunu (all q) | 2e-15 (q=0), <=1.3e-9 | OK |
| seam phys X(q+G0)==X(q_wrap)D_G0 | 4.9e-17 | OK |
| seam G-relabel (lab recon) | 5.4e-16 | OK |
| v_SR + v_LR == v | <=1.2e-16 | OK |
| enk_full vs WFN el | 7.0e-14 | OK |
| window-top closed shell (1e-6 Ry) | min gap 1.1e-3 Ry | OK |
| K-identity (corrected form) | 1.2e-14 | OK |
| gauge-randomization of links | 5.6e-9 (was 35 pre-fix) | pass (see 1.6) |
| TRS V(-q)==conj(V(q)), stored / disk | 1.0e-15 / 2.4e-15 | OK |
| TRS VcSR~ +-q eigenvalues | 8.4e-15 | OK |
| Parseval under gauge | 0.0 | OK |
| self-edge principal cosines | min 1.000000 | OK |
| on-grid alpha-invariance (v-level exact split) | 1.2e-16 | OK |

## 3. The mechanism: no smooth whitened bundle at accessible densities

**C_q spectra.** No spectral gap anywhere (smooth decay; all 640 directions
above 1e-8 lam_max at every q; sqrt-cond 4.2e3). The response's sec 8-9
analyticity premise (protected cluster / uniform gap) has no support: rank
is a fiat choice, r = 640 (all resolved) used.

**Whitened principal cosines between adjacent q** (`proto1_C2_probe_angles`,
covariant t; medians; "floor" = random r-dim subspaces in npair dims,
sqrt(640/npair)):

| variant | 3x3 (dq=1/3) | 6x6 (dq=1/6) |
|---|---|---|
| no transport (v2/v3) | 0.098 (floor 0.105) | 0.054 (floor 0.053) |
| covariant transport, full window (v1) | 0.20 | 0.23-0.33 |
| + PHYS/glued sewing (v6) | 0.18-0.23 | 0.24-0.34 |
| valence-only window nb=26 (v5/v6w) | 0.45-0.57 | 0.29-0.39 |
| # cosines > 0.9 (best variant) | 14-99 of 640 | 0-38 of 640 |
| rank-truncated top-40 (nb80) | 0.23 | 0.24 |

Readings: (i) without transport the adjacent-q pair subspaces overlap at
exactly the RANDOM floor at both densities; (ii) covariant transport
extracts a real but weak alignment (~4x floor at 6x6) that does NOT
approach 1 as dq halves — a smooth bundle requires 1 - O(dq^2); (iii) the
exact glued sewing (v6, the response's sec-6 demand) changes nothing
structurally — the collapse is physics, not convention; (iv) there is no
transportable low-rank core: truncating to the top-40 whitened directions
leaves the median at 0.23; (v) only the valence-window pair family shows
partial alignment, and it WEAKENS on the finer grid (n>0.9: 99 -> 38 on
the best edge/window). This is response-10B case 3 by its own taxonomy:
"the physical pair-feature subspace itself is underresolved by the coarse
grid" — and the 6x6 trend says densification does not repair it at any
accessible mesh. Consistent with (and the subspace-level root cause of)
the #3.2 measured zeta rotation (p ~ sqrt(2)) and #3.3's span-rotation
physics.

**Curvature and Wilson obstructions** (diag D). Plaquette holonomy
||W_box - I||_F med 30.5 (random-unitary ceiling sqrt(2r) = 35.8); Wilson
row/column ||W - I|| ~ 30-34 with eigenphases at +-pi (branch margin
0.001-0.007) — the log-distribution sits at the branch-ambiguity edge
exactly as the KNOWN RISK anticipated; log-branch continuity across q1
||L_i - L_(i-1)|| ~ 61 (none); det-W winding = 0 (no topological
obstruction — moot at this curvature). Gauged-link residuals ~ 31/12
(axis1/axis2): no gauge can make near-random links smooth.

## 4. FOUR TAILS — the response's sec-10C criterion (MoS2 3x3, final)

Per-shell max ||.||_F normalized to shell 0; shells |R| = 0 / 5.98(x6) /
10.36(x2) Bohr; wrapped-q lab continuation; covariant transport; r=640.

| object | R=0 | 5.98 | 10.36 |
|---|---|---|---|
| C_R (reference, this run) | 1.000 | 2.29e-2 | 6.72e-4 |
| raw zeta_R | 1.000 | 0.388 | 0.157 |
| Phi_R (half-whitened, no transport) | 1.000 | 1.366 | 0.989 |
| **Phi~_R (transported, global frame)** | **1.000** | **1.892** | **1.604** |
| Phi~_R cell-periodic | 1.000 | 1.026 | 0.667 |
| Vc~^SR_R (transported SR kernel, alpha in {0.31..2.51} 1/Bohr) | 1.000 | 0.56-0.63 | 0.40-0.44 |
| Vc~^SR_R (taper 20->30 Ry) | 1.000 | 0.581 | 0.409 |
| Vc~_R (total, transported) | 1.000 | 0.752 | 0.586 |
| Vc_R (total, no transport) | 1.000 | 0.472 | 0.381 |

The transported `Phi~_R` does not merely fail to decay — it is ROUGHER than
raw zeta_R (the near-random gauge field injects q-roughness into the
sections; Parseval-conserving). The S-whitening itself (Phi vs zeta) makes
the tail worse, not better: the top-sigma directions carry the span
rotation coherently. The Strategy-B object Vc~^SR is alpha- and
taper-insensitive at 0.56-0.63 — nowhere near C_R's 2.3e-2. By the
response's own criterion ("My prior claim would only be justified if (3)
remains nondecaying after exact seam treatment"), **the claim is dead at
these densities**, with the seam exact (sec 2), transport covariant, and no
topological obstruction to blame.

Historical note: the first (pre-covariance-fix, unwrapped-q) run gave
Phi~_R = 1.01/1.02 — flat rather than super-flat; every correction found
made the response's construction look *worse*, not better.

## 5. Physical-metric LOO (owner pushback: the verdict variables)

MoS2 3x3, all 9 held-out q0. PRIMARY metric: gap-window exchange block
B[p,p'] = sum_G conj(M_p) v(q0+G) M_p' over rows M_cvk = spin-traced pair
densities, top-3 valence x bottom-3 conduction x all k (81 rows), truth =
the stored full-rank fit at q0, slab v, q=0 G=0 divergence excluded. Also
top-decile per-element error, tile relF (secondary), TDA exciton-shift
(81-dim H = D - W_dir + B/nk with stored W0 and enk; max |shift| of the
lowest 4 states when only the exchange block is swapped).

| scheme (median / max over q0) | B relF med | B max | top-decile med | tile med | exc shift |
|---|---|---|---|---|---|
| **null test: C2 full chain, r=640, no interp** | **4.4e-14** | 7.6e-14 | 3.2e-14 | — | — |
| rank truncation r=480 (no interp) | 1.3e-3 | 2.3e-2 | 8.6e-4 | — | — |
| rank truncation r=320 | 2.3e-3 | 2.3e-2 | 1.6e-3 | — | — |
| rank truncation r=160 | 9.5e-3 | 3.0e-2 | 7.7e-3 | — | — |
| rank truncation r=80 | 4.2e-2 | 1.1e-1 | 3.8e-2 | — | — |
| **C2 transported-Phi interp, nR=4** | 0.90-0.95 | 1.2 | 0.91-0.96 | — | — |
| **C2 transported-Phi interp, nR=7** | 0.96-0.98 | 5.3 | 0.88-0.90 | 4.4e2 | 36 meV |
| C2 nR=9 (degenerate pinv fit, 8 train pts) | 1.0 | 1.0 | 1.0 | — | — |
| #3.5 ladder re-scored: raw C^-1 | 2.6e-1 | 3.1e2 | 1.4e-1 | 4.9e2 | — |
| #3.5 ladder: tikhonov 1e-6 | 5.7e-2 | 4.8e-1 | 3.3e-2 | 6.4e1 | — |
| **#3.5 ladder: rankcut 1e-4** | **4.7e-3** | **3.2e-2** | **4.2e-3** | **1.00** | — |
| #3.5 ladder: rankcut 1e-2 | 4.4e-2 | 1.1e-1 | 3.8e-2 | 1.00 | 5.4 meV |
| zeta-direct interp (skip solve), nR=7 | 7.0e-2 | 5.8e-1 | 5.0e-2 | 0.89 | — |

Gauge-protocol spread ||G_align - G_construction|| = 14-32 (vs 35.8
unrelated) — the target-gauge protocols disagree at holonomy scale, as the
curvature data predicts; both give the same failed interpolation.

**Two headline results beyond the C2 kill:**

1. **The owner's pushback is CONFIRMED QUANTITATIVELY — #3.5's "no
   regularization window" verdict inverts under the physical metric.**
   Ingredient interpolation + C^-1 solve at rankcut 1e-4 delivers median
   **0.47% / max 3.2%** on the physical exchange block while the TILE is
   100% wrong (relF 1.00) — the #3.5 tile-metric row for the same solve
   read 11 (1100%) and its d-in-range(C) contraction read 21. The
   ill-conditioned tail is physically inert: the gap-window pair rows
   live in the well-conditioned top of C's spectrum (rank ladder: 160 of
   640 whitened directions give 1%; 480 give 0.13%). There IS a
   regularization window under physical contractions: rankcut between
   1e-4 and 1e-2 is flat-ish (0.5-4%), raw explodes, and the window
   optimum ~1e-4 sits where the kept spectrum matches the rows' support.
2. **C2's transported interpolation is 200x WORSE than the plain #3.5
   ladder it was designed to beat** (0.96 vs 0.005 median B). The global
   frame is not merely useless — the near-random gauge field actively
   scrambles the sections (four-tails sec 4). Strategy A is dominated by
   plain rankcut ingredient interpolation everywhere.

Exciton-shift scale: swapping in the rankcut-1e-2 interpolated exchange
moves the lowest TDA states by <= 5.4 meV median; the C2-interpolated
exchange moves them by ~36 meV (and its B is ~100% wrong — the small shift
merely reflects the exchange kernel's ~few-10s-meV overall weight in this
81-dim window).

## 6. Verdict against the bar, and what survives

**Bar (task):** beat #3.5's rankcut-1e-2 = tile 1.00 / phys 0.89 and raw
3.7e6 by orders of magnitude on the same LOO points.

- C2 (the construction under test): **does not beat anything** — B ~ 0.9-1.0
  at every stencil, equal to the #3.5 rankcut-1e-2 physical row. The
  response's mechanism (smooth whitened bundle + parallel transport +
  sewing) is measured absent at 3x3 AND 6x6: subspace overlaps at/near the
  random floor, holonomy at the random ceiling, transported tails rougher
  than raw. Its own sec-10C falsification criterion fires. TERMINATED
  before production interpolation, per the pre-agreed rule.
- The bar itself moves: under the owner's physical metric the TRUE
  incumbent is the re-based #3.5 ladder at rankcut ~1e-4: **0.5% median /
  3% max on-grid LOO** at 3x3 — i.e. the "aggressive rank-cut +
  interpolation reaches few-percent physical accuracy" scenario the owner
  flagged is REAL (on-grid; the off-grid-with-truth 3x3-subgrid -> 6x6
  test remains to be run — see sec 7).
- What survives of C2's machinery (validated, reusable): the covariant
  link/transport algebra (correction #2) with its gauge-randomization
  gate; the K-identity bookkeeping (correction #1); the null-tested
  physical-metric harness (B-block + exciton swap); the q-wrap and blat
  fixture conventions (sec 1.3-1.4); the whitened rank ladder showing
  r ~ 160-320 suffices physically (feeds any future fixed-rank scheme
  and the owner's centroid-importance intuition).

## 7. Caveats, open items, loose ends

1. **Fixture provenance (open).** psi_full_y is NOT in the band span of
   any WFN on disk at k != 0 (span-projection residual 29-40% vs
   dir-WFN.h5 == qe/nscf/WFN.h5 == WFN_qp.h5, which are mutually
   identical), while enk_full == WFN el at 7e-14 and k=0 matches at
   2.7e-16. psi_full_y is evidently a processed set (not raw eigenvectors
   of the stored WFN). Consequences: (a) the non-circular fit-RHS check
   (sec 1.7) and the "ISDF fit floor vs exact pair rows" number cannot be
   closed against these WFNs (measured floor B ~ 1.2-1.5 with
   best-unitary alignment — NOT interpretable as fit error); (b) the
   band-transport t's used WFN orbitals — however the verdict is carried
   by psi-pure measurements (v2/v3 angle variants at the random floor;
   Phi/zeta tails; all LOO metrics), and t only ever improved alignment,
   so the conclusion is robust to this. Worth an owner look at the
   restart writer's psi_full_y provenance.
2. **The q-wrap trap contaminates the existing #3.5 / C3-rebase ladder
   numbers (A/B-measured, 155x).** The interp_study harness (and C3's
   proto2 re-base, which reuses it) reconstructs the lab zeta fields with
   the UNWRAPPED mf_header rk; at 5 of 9 q's those fields carry a
   spurious e^{i G0.r}. Direct A/B on the same held-out q0=2, same
   rankcut-1e-4 solve, same truth (`proto1_ladder_wrap_ab.py`):
   B relF = 4.5e-3 (wrapped) vs 7.0e-1 (unwrapped). This reconciles the
   250x disagreement between my ladder row (4.7e-3 med) and C3's proto2
   "INTERP rankcut 1e-4" row (1.19): C3's physical-metric re-base
   inherits the unwrapped continuation and must be re-read; BOTH
   corrections (physical metric AND wrapped continuation) are needed to
   expose the true half-percent window. Similarly #3.5's zeta_R "flat"
   table (1.00/0.82/0.65) is the unwrapped object; wrapped it is
   1.00/0.39/0.16 (weak decay). #3.5's tile-relative conclusions at
   rankcut <= 1e-6 (conditioning-dominated) survive; its rankcut
   1e-4/1e-2 and "phys 0.89" rows carry the trap.
3. Off-grid-with-truth (3x3-subgrid -> 6x6 complement) not run for C2 —
   terminated by the tails rule; SHOULD be run for the re-based ladder
   (rankcut 1e-4) by C3/baseline_rebase, since on-grid LOO at 3x3 is the
   easier test (the #3.5 off-grid factor was ~30x worse than on-grid for
   C_q interp at 4x4).
4. The exciton-shift assembly uses a fixed direct term (stored W0) and a
   1/Nk exchange normalization; shifts are V-swap deltas with a common
   assembly, so convention factors cancel in the comparison; absolute
   meV values carry that convention.
5. **6x6 four-tails (landed; shells to 26 Bohr) — the verdict replicates
   at the doubled density.** Per-shell max ||.||_F / shell-0, |R| = 0 /
   5.98 / 10.36 / 11.96 / 15.82 / 17.94 / 20.71 / 21.56 / 26.06 Bohr:

   | object | 5.98 | 10.36 | 15.82 | 26.06 |
   |---|---|---|---|---|
   | C_R (reference) | 2.24e-2 | 5.2e-4 | 4.1e-5 | 4.1e-5 |
   | raw zeta_R | 0.400 | 0.129 | 0.100 | 0.0996 |
   | Phi_R (no transport) | 1.212 | 0.694 | 0.421 | 0.364 |
   | **Phi~_R (transported)** | **1.634** | 0.267 | 0.233 | 0.179 |
   | Phi~_R cell-periodic | 0.524 | 0.241 | 0.119 | 0.045 |
   | Vc~^SR_R (alpha ladder) | 0.41-0.45 | 0.13-0.16 | 0.076-0.085 | 0.013-0.038 |
   | Vc_R (total, no transport) | 0.331 | 0.206 | 0.205 | 0.198 |

   Phi~_R is again WORSE than flat at shell 1 (1.63 — gauge scrambling)
   and plateaus 3-4 orders of magnitude above C_R's profile everywhere;
   raw zeta_R plateaus at ~0.1; the SR kernel at ~0.01-0.16. Plaquette
   holonomy drops to med 10.3 (from 30.5) — per-plaquette curvature
   shrinks with cell area as it must, but the per-edge alignment (sec 3)
   does not approach identity, and no object approaches C_R's decay.
   The "denser grid rescues the bundle" escape is closed empirically.
   (The 6x6 TRS line "max 1.306" is a blind spot of my +-q check at the
   two self-partner boundary points q = 1/2 where -q == q + G0 requires
   a G-shift relabel the simple conj-compare omits; med 9.9e-16 over the
   proper pairs. VcSR +-q eigenvalue max 3.4e-3, boundary-affected,
   diagnostic-only.)

