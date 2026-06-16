# Milestone A validation — screened-charge bispinor COHSEX (+ bare Breit) runs and is sensible

**Date:** 2026-06-16 · **Checkout:** `sources/lorrax_C` @ `main` `e85be60` · **Owner:** session-C
**System:** MoS₂ 3×3, 2D, nband=32 / nval=4 (reduced window, mirrors parent `D_60Ry_bispinor`)
**Runs:** `runs/MoS2/C_60Ry_bispinor_cohsex_2026-06-15/` (bispinor) · `…/C_60Ry_scalar_cohsex_2026-06-15/` (scalar ref)

## Goal

Validate **milestone A** of the screened-bispinor-GW roadmap (see
`reports/bispinor_screened_gw_state_2026-06-15/report.md`): a screened **charge-channel**
COHSEX (Σ_SX + Σ_COH on the scalar W⁰⁰) coexisting with the **bare-Breit** Σ^B. Per the
χ_μν = χ⁰⁰-only reduction, this needs no supermatrix — the existing scalar `solve_w` gives
W⁰⁰, bare V^{ij} tiles flow into Σ^B — and it is **already wired** in the one-shot driver
(`gw_jax.py:359-367` → `compute_cohsex_sigma(do_screened=True, …, bispinor args)`). So A is a
*validation* task: flip `x_only→false, do_screened→true` on a bispinor run and confirm it is
correct.

## Result: ✅ runs end-to-end and is physically sensible

The bispinor `do_screened=true` COHSEX completed in **64 s** (1 node / 4 GPU, full-BZ),
producing `eqp0.dat` + `sigma_diag.dat`.

| Check | Result | Evidence |
|---|---|---|
| Screened bispinor run end-to-end | ✅ | χ₀→W (`W.exec` 0.19 s) → Σ_SX/Σ_COH + Σ^B → eqp0; EXIT 0 |
| Screening reduces exchange | ✅ | bare Σ_X(n=0) = **−37.09** → screened Σ_SX = **−22.14 eV** (−15 eV); total \|Σ\| −37.09→−29.81 |
| Σ^B (bare Breit) coexists | ✅ | diag tiles −0.153/−0.153/−0.144 eV; in-plane (1,1)=(2,2) per z-mirror; off-diag ±0.012, Hermitian; Σ_tot^B ≈ −0.52 eV |
| Kramers/spinor degeneracy | ✅ | every Σ and eqp0 level paired (n=0,1 / 2,3 / …) |
| 4-channel ISDF built | ✅ | `zeta_q` (charge, n_rmu=640) + `zeta_q_mu1/2/3` (transverse, 668) |
| COHSEX QP gap | ✅ sensible | eqp0 gap (band-4 VBM → band-5 CBM) = **2.40 eV** (eqp1 2.76 eV); reduced model, not converged |

### Screened Σ decomposition (k=0, eV)

| n (Kramers pair) | bare Σ_X | Σ_SX (screened) | Σ_COH | Σ_TOT = SX+COH |
|---|---|---|---|---|
| 0,1 | −37.09 | −22.14 | −7.67 | −29.81 |
| 2,3 | −30.59 | −17.33 | −6.87 | −24.20 |
| 4,5 | −29.91 | −17.01 | −6.69 | −23.70 |
| 6,7 | −30.02 | −16.85 | −6.85 | −23.69 |

Σ_SX is uniformly smaller in magnitude than the bare Σ_X — the expected static screened-exchange
reduction — and Σ_COH adds the Coulomb hole. The transverse `γ̃^1 indefinite → cusolvermp_lu`
path fired during the zeta fit, confirming the known indefinite-CCT → LU dispatch.

### Charge-block reduction (bispinor Σ_SX/Σ_COH == scalar)

**By construction (code inspection):** `compute_cohsex_sigma` computes `Σ_SX = sigma_sx_k(wfns,
Gij, W_q)` and `Σ_COH = sigma_coh_k(wfns, W_q, V_q)` on the **charge** wfns + charge `W_q`
(`cohsex_sigma.py:211-212`) **identically** whether `bispinor` is true or false; the bispinor
path only *adds* Σ^B to `sig_x` via the separate `wfns_transverse` block (`:240-251`). It never
touches the Σ_SX/Σ_COH path. So the bispinor charge-block Σ_SX/Σ_COH are identical to the scalar
run's by structure, given the same W⁰⁰ (computed identically — `compute_chi0` spin-traces to χ⁰⁰
regardless of bispinor).

**Empirical scalar cross-check: PENDING.** The scalar reference run
(`C_60Ry_scalar_cohsex_2026-06-15`, `bispinor=false`) is blocked by node contention — a
concurrent 4-node (16-GPU) job from another pool agent (sacct step `54544991.16`, RUNNING on all
4 nodes since 00:01) fully occupies the shared allocation. A patient retry is queued; the numeric
match will be appended here when it lands.

## Caveats / scope

- **Reduced model:** nband=32 / nval=4, head zeroed (`vhead=whead_0freq=0`), so absolute QP
  energies and the 2.40 eV gap are not converged MoS₂ values — this validates the *machinery*,
  not a physical gap. A converged + head-corrected run is follow-on.
- **One-shot only.** The SC/QSGW path still drops Σ^B (`sigma_dispatch.py:88` /
  `sc_iteration.py:294` take no bispinor args) and would also need the transverse bundle rotated
  by U_qp each iteration — the remaining code task for milestone A (see roadmap §SC nuance).

## Blocker found and worked around: IBZ-cascade bispinor zeta-fit regression

Current `main`'s **IBZ cascade crashes the bispinor zeta fit**:
`ValueError: B.shape[0]=9 != Nq=5` in `batched_distributed_potrs` (`ffi/cusolvermp/batched.py:200`)
— the cusolverMp Cholesky back-solve gets a **full-BZ Z_q (9 q-points)** against an **IBZ-factored
L_q (5 q-points)**. The IBZ cascade activates on centroid orbit-closure (gate at
`gw_init.py:959-961`); the parent `D_60Ry_bispinor` ran 2026-05-05, *before* the cascade was
activated (2026-05-11), so it was immune. **Worked around with `LORRAX_FORCE_FULL_BZ=1`** (full-BZ
gives identical physics, just disables the symmetry optimization). This is a genuine regression in
the bispinor + IBZ path that should be fixed (full-BZ Z_q must be sliced to IBZ before potrs, or
L_q unfolded) — logged in `KNOWN_SANDBOX_ERRORS.md`.

## Reproduce

```bash
module use /global/u2/j/jackm/modulefiles; module load lorrax_C
module use $PSCRATCH/lorrax_sandbox/modulefiles; module load lorrax_agent
lxattach                       # → JID of lx-alloc-$USER (no pipe: lxattach/module load must not be piped)
export LORRAX_FORCE_FULL_BZ=1  # work around IBZ-cascade bispinor zeta-fit crash
cd runs/MoS2/C_60Ry_bispinor_cohsex_2026-06-15
lxrun python3 -u -m gw.gw_jax -i $(pwd)/cohsex.in > gw.out 2>&1
```

## Next steps

1. Land the scalar empirical charge-block match (pending node availability).
2. Wire Σ^B through `compute_sigma_xc` + `sc_iteration` (with transverse U_qp rotation) → SC-capable A.
3. Fix the IBZ-cascade bispinor zeta-fit shape bug so full-BZ isn't required.
4. Then milestone B: the genuine supermatrix (un-trace χ → channel-blocked δ−Vχ → LU → screened Σ_c^B).
