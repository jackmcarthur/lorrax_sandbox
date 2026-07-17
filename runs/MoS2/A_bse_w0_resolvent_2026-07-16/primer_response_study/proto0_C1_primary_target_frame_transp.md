# proto0 C1 — target-frame transported V^SR interpolation (Method A + Strategy B, exact-target dressing)

Prototype of the ARBITRARY_Q_PRIMER_RESPONSE counterproposal, construction C1,
against real fixture data. Scripts + logs in this directory
(`prep.py`, `proto0_a_mos2_3x3.py`, `proto0_b_mos2_6x6.py`,
`proto0_c_negctrl.py`; logs `proto0_a.log`, `proto0_b.log`, `proto0_c.log`;
machine-readable results `proto0_{a,b,c}_results.json`).

STATUS: RESULTS PENDING — tables below filled by the runs.

## 1. The construction (as implemented)

Pair rows, spin-traced (this realizes the fit's 4-channel Frobenius metric —
see correction C-1 below):

    x_p(mu) = sum_s conj(psi_{m,k-q,s}(r_mu)) psi_{n,k,s}(r_mu),   p=(m,n,k)

over the full 80-band window, k-q wrapped mod kgrid. Gram identity
`X^H X == C_q` asserted against the verbatim falloff/`isdf.core.c_q_from_psi_sm`
rebuild at 1e-12.

Frames: `eigh(C_q) = R_q S_q^2 R_q^H` (descending), ONE global fixed rank r,
ladder r in {640, 480, 320, 160}, never per-q thresholds.

Half-whitened auxiliary from stored data (conjugation resolved numerically,
correction C-2):

    Phi_q(G) = S_q R_q^H conj(zeta~_q(G))     on sphere(q)

Scalar-operator split only (production kernel conventions, correction C-3):

    v_dim(K) = (8 pi / K^2) * T_slab(K) / V_cell,   K = qwrap + G  (true bohr^-1)
    T_slab   = 1 - exp(-|K_par| Lz/2) cos(Kz Lz/2)
    v_SR     = v_dim * (-expm1(-K^2/(4 alpha^2)))   [stable; small-K series wired]
    v_LR     = v_dim * exp(-K^2/(4 alpha^2))
    exact K=0 slot excluded everywhere (production rank-1 head, out of scope);
    G=0 at q != 0 is INCLUDED (production convention, verified against disk).

Interpolated object (the only one):

    Vc^SR_q = Phi_q v^SR_q Phi_q^H     (r x r, Hermitian PSD)

with the identity `Vc^SR_q + Vc^LR_q == S_q R_q^H V_q R_q S_q` (make_Vq
convention `V[mu,nu] = sum_G conj(zt_mu) v zt_nu`) asserted per coarse q at
1e-12 for the whole alpha ladder.

Transport: per-k single-particle links `t_k = polar(<u_{k-q}|u_{k-q'}>)` from
WFN.h5 G-space overlaps (centroid-quadrature fallback computed and compared);
pair cross-Gram without materializing the big rotation,

    H_{qq'} = sum_k X_{q,k}^H [ (conj t_k) X_{q',k} ]_{left-band-rotated}
    M = S_q^-1 R_q^H H R_{q'} S_{q'}^-1 ;  T_{q<-q'} = polar(M)

principal cosines of M recorded (diagnostic B), plaquette holonomies
`||W_box - I||` (diagnostic D).

Target-frame interpolation on the wrapped torus:

    Vc^SR_Q = sum_i w_i(Q) T_{Q<-qi} Vc^SR_{qi} T_{Q<-qi}^H

with (a) nonneg weights (nn-average on-grid / multilinear off-grid,
PSD-preserving) and (b) truncated-R Fourier weights nR=7 (Sec-3.5
apples-to-apples).

Exact-target dressing: `a_p = conj(x_p) R_Q S_Q^-1` (rows of L_Q, norms <= 1,
measured), `B^SR = a Vc^SR_Q a^H`; LR added exactly, low-rank, via exact
pair-density FFTs `F_p(K)` from WFN.h5 over the stored sphere:
`B^LR[p,p'] = sum_K conj(F_p) v^LR F_{p'}`. `V_Q` is never reconstructed
(the tile appears only as the SECONDARY diagnostic).

## 2. Corrections to the task-spec math (documented per mandate)

- **C-1 (pair-row spin structure).** The spec's rows `p=(m,n,k,a,b)` with
  independent spin labels give `X^H X = sum_k [sum_a conj P^{k-q}_{aa}]
  [sum_b P^k_{bb}]` — spin-diagonal products only. The fit's actual metric
  (primer I.2, `C = sum_k sum_{ss'} conj(P^{k-q}_{s's}) P^k_{ss'}`, all four
  channels incl. P_updown) is the Gram of the SPIN-TRACED rows `p=(m,n,k)`,
  `x_p = sum_s conj(psi_m,s) psi_n,s`. Verified numerically: spin-traced rows
  match the falloff C at 1e-12; the 4-label variant does not.
- **C-2 (conjugation).** With the code's `V[mu,nu] = sum_G conj(zt_mu) v zt_nu`,
  the identity `Vc = S R^H V R S` requires `Phi = S R^H conj(zeta~)` (not
  `S R^H zeta~`), and the physical contraction is `B = conj(M) V M^T`
  = `a Vc a^H` with `a = conj(M) R S^-1`. Both signs resolved by 1e-12
  asserts (gates), stored in `proto0_conventions.json`.
- **C-3 (production kernel conventions, found by disk-match gate).** (i) v is
  evaluated at the BGW-WRAPPED q (`v_q_g_flat.py:232`), not the mf_header rk;
  (ii) production `bvec = blat * bvec_hdr` — |K|^2 in true Ry (gw_init.py:292);
  (iii) `psi_full_y = u_WFN / sqrt(N_r)`, zero per-band phase (measured
  4.6585e-3 = 1/sqrt(46080) global scalar, residual 2e-16);
  (iv) `vcoul_cutoff_ry = ecutwfc` zeroing is a no-op on the stored spheres
  (they are exactly the 30-true-Ry WFN spheres). With (i)+(ii) the stored
  `V_qmunu` is reproduced from stored zeta~ at machine precision (gate table).
  Note the Sec-3.5 harness used unwrapped rk + unscaled bvec — self-consistent
  for its interpolation ratios, but the rebase here is done in the production
  labeling (fair to both sides).
- **C-4 (LR G-set).** LR is summed over the stored per-q sphere (not an
  unbounded superset): the stored truth `V_qmunu` is sphere-limited, so
  truth-consistency fixes the LR set = sphere(q). A production off-grid Q
  needs the fixed global superset + w_cut taper exactly as the response
  specifies; noted, not exercised (fixture tests are all at stored-sphere q).

## 3. Self-checks (all closed before any result below was trusted)

MoS2 3x3 (proto0_a.log):

| check | value |
|---|---|
| recon/forward sphere round-trip | 2.3e-16 |
| g0_mu == ZG[:,:,G=0 slot] | 0.0 |
| production-v rebuild vs disk V_qmunu (all 9 q) | med 1.9e-15, max 1.9e-15 |
| el vs enk | 7e-14 Ry; window-top gap min_k(e81-e80) = 1.13e-3 Ry -> CLEAN closed shell |
| WFN u(r_mu) vs psi_full_y | 3.3e-16 (global 1/sqrt(N_r), zero per-band phase) |
| X^H X (spin-traced x-rows) vs falloff/production C | 2.0e-15 (4-label variant: 4.8e-2 — correction C-1 confirmed) |
| cond(C_q) | med 1.8e7 |
| Vc identities (R and C flavors) | 1.2e-14 both |
| B dressing routes | R: 4.2e-15, rowmax(l)=0.189 <= 1 (leverage bound HOLDS); C-ablation: rowmax 2.772 (bound broken — the flavor tell) |
| seam (ii) zbar_(q+G0)(G-G0) == zbar_q(G) | 2.7e-16 |
| seam (i) physical block across seam (X D_G0 + relabel) | 4.0e-15 |
| TRS V(-q) == conj(V(q)) | max 1.0e-15 |
| on-grid alpha-invariance Vc_SR+Vc_LR (all q x alpha) | 1.8e-14 |
| null test / gauge randomization | see run tables below |

## 4. Results

THE BAR (re-based Sec-3.5 ladder under the primary metrics; cross-checked
against the C3 control run proto2_out_c3_3x3.log, which reproduces the logged
Sec-3.5 tile/randpair numbers exactly): best old-scheme entry = interp-C/Z +
rankcut 1e-2 -> gap-window B relF med ~1.1 (max ~8), exciton d-lambda ~18 meV
med / ~39 max; rank-truncating TRUE data alone (no interpolation) costs
7.6e-4 (kappa-cut 1e6) to 5.1e-2 (1e2) in B — the owner's
junk-directions-are-unphysical intuition confirmed on truth, but interpolation
error dominated the old scheme regardless.

### 4.1 Baseline rebase (Sec 3.5 ladder under PRIMARY metrics, MoS2 3x3 LOO)
(table from proto0_a.log)

### 4.2 C1 on-grid LOO MoS2 3x3 (9 q)
(table: weights x rank x alpha -> SR relF, TOT relF, p90 elem, exciton meV,
tile secondary)

### 4.3 C1 off-grid with truth: 3x3 subgrid -> 27 complement q of the 6x6 fixture
(table from proto0_b.log + baseline-off comparison + on-grid 6x6 LOO)

### 4.4 Negative control (Si 4x4x4) + aux (MoS2 4x4)
(table from proto0_c.log)

### 4.5 Diagnostics
principal-cosine spectra (B), holonomies (D), overlap WFN-vs-centroid deltas,
LR fit-vs-exact consistency, ISDF fit-quality context floor.

## 5. Verdict
(filled after runs)

## 6. Production mapping (owner sharding note)

Prototype is single-GPU jax.numpy at fixture size (owner-approved). Production:
every N_mu^2 object here (`C_q`, `Vc^SR`, `H_{qq'}`, `T`) shards `P('x','y')`
per the zeta-fit conventions; `eigh`/SVD/polar route through the
cusolvermp/slate FFI (src/ffi) block-cyclic path; the per-k cross-Gram
accumulation is the same k-slab pattern as `c_q_from_psi_sm` (shard_map,
scan-inside-shard_map per the slot-pile-up rule). `X_Q` at arbitrary Q comes
from htransform psi(k+Q) at centroids (II.1) — the only new production
ingredient; `F_p` LR form factors from the htransform coefficient basis
`M_ab(G)` precompute (response sec 3). Deferred until these results warrant.
