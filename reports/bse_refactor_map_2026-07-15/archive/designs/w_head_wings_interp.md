# Design: screened-Coulomb interpolation with head AND wings

BSE refactor-map program, 2026-07-15. Checkout `sources/lorrax_D` (audit base
e18d0e5; BSE/gw trees byte-identical at working HEAD per
`kernel_dataflow_trace.md:3-7`). Units Ry throughout, BGW `v(q+G)=8π/|q+G|²`.

Scope: how interpolated `W` on a fine q-grid accounts for the anisotropic q→0
**head** and the directional **wings** — the accuracy gap the user flags. Grounds
every current-state claim in file:line; BGW is the reference implementation.

---

## Current state in LORRAX (file:line grounded)

**What is stored for W.** The GW pass Dyson-solves the full screened interaction in
the ISDF centroid basis and writes one static tile per q:

- `solve_w` (`src/gw/w_isdf.py:384-409`) → `_get_w_solve_fn._solve_w`
  (`w_isdf.py:238-283`): per q, `A = I − pref·V_log·χ_log`, `W = A⁻¹ V_log`
  (`w_isdf.py:264-266`). Output `W` is flat-q `(nq, μ, ν)`, **full** screened W in the
  centroid basis — no head/wing/body split.
- Persisted by `persist_w0_and_head` (`src/gw/gw_output.py:176-241`) as
  `W0_qmunu (nq,μ,ν)` (`gw_output.py:209-211`) plus **two scalars**:
  `vhead = v(q→0,G=G'=0)` and `whead[ω] = wcoul0` (`gw_output.py:223-234`,
  `write_head_scalars_to_h5` `src/file_io/tagged_arrays.py:157-193`).
- `V_qmunu (nq,μ,ν)` carries the bare Coulomb body with **G=G'=0 zeroed** (Henneke
  Eq 2-32, `context_docs.md:324`); `G0_mu_nu[μ] = ζ(q=0,μ,G=0)` is the G=0
  projector of the ζ basis (`tagged_arrays.py:94`).
- PPM head params: `HeadGNParams` (`head_correction.py:44-56`) is a **scalar**
  two-sample pole fit of the q=0 head only (`fit_head_ppm`
  `head_correction.py:280-354`); `whead` grows to length 2 for GN-PPM (static +
  iω_p, `gw_output.py:216-227`). No off-head PPM.

**What the head injection actually does (head-only, isotropic).** BSE reinstates the
stripped q=0 head as a rank-1 update on the **q=0 slice only**:

- `apply_q0_head_rank1_sharded` (`head_correction.py:779-815`):
  `V_q0 += v_scalar·conj(g0_X)⊗g0_Y`, `W_q[:,:,0,0,0] += w_scalar·conj(g0_X)⊗g0_Y`
  with `w_scalar = whead[0]/V_cell` (`head_correction.py:735-740`). Called from
  `bse_io.py:504` (sharded) / `:818` (ring).
- The head scalar is a single isotropic number: `wcoul0` is the mini-BZ average
  `⟨ v(q)/(1 − v(q)·q̂ᵀSq̂) ⟩` over Sobol q-points (`src/gw/coulomb/bulk_3d.py:47-56`),
  collapsing the **anisotropic** generator to one scalar. **No wing term is ever
  added.** No q̂ dependence survives.

**The anisotropic generator already exists but is thrown away.** `compute_S_omega`
(`src/common/chi_from_dipole.py:106-162`) builds the full Cartesian
`S_αβ(ω) (3,3)` head tensor from `dipole.h5` (`v_cvk`, `ΔE`, occupations) — this is
the "GW B2" head machinery. `bulk_3d.q0_average` (`bulk_3d.py:32-73`) consumes it
only to average `v/(1−v·qᵀSq)` into `wcoul0`; the directional
`q̂ᵀ S q̂` information is discarded.

**A sharded head/wing/body kernel already exists, unplumbed.**
`src/gw/experimental/head_wing_schur.py` is a complete, tested (behind
`pytest.mark.extra`, `tests/test_head_wing_schur.py`) sharded Schur decomposition:
`W = W_body0 + W_head·(conj(g0)+A_wing) ⊗ (g0+A_wingp)`
(`head_wing_schur.py:219-235`), with `W_head = v_head/(1−v_head·χ_head_eff)`
(`:199-201`), body solve delegated to `w_isdf._get_w_solve_fn` (`:130-135`), and
zero-comm rank-1 subtract/rebuild (`:88-106`, `:208-237`). It is **not** wired into
production (`head_wing_schur.py:14-18`) and its docstring notes the planned use:
pass centroid `χ_wing` for the literal wing, or `χ_wing=0` to mirror BGW's
`fixwings` q=0 zeroing (`:259-266`). **This module is the reuse seam for wings.**

**BSE is single-grid today.** `apply_W` convolves `W_q(μ,ν,nkx,nky,nkz)` over
q=k−k′ via a 3-D ortho FFT pair (`bse_serial.py:69-78`, `bse_simple.py:141-173`)
on the **coarse** GW k-grid = q-grid. There is **no** coarse→fine interpolation of
any kind (`bgw_fine_grid_reference.md:637-638`), so no fine-q head/wing machinery.
On the coarse grid the single q=0 point makes anisotropy average out and the wing
average to zero (Baldereschi-Tosatti, below) — which is exactly why the isotropic
head-only scalar reproduced Si to ~3 meV (`context_docs.md:135-138`). Wings become
load-bearing **only** once q is refined near 0.

---

## Reference physics + BGW implementation

**Anisotropic q→0 limit (3D semiconductor).** With the head dielectric tensor
`ε_{00}(q→0) = 1 − v(q)·q̂ᵀ S q̂` (S from the dipole/`chi0` head, `chi_from_dipole`):

```
W_00(q)   = v(q) / (1 − v(q)·q̂ᵀ S q̂)          # head:  ~ head(q̂)/q²  (v=8π/q²)
W_0G(q)   ~ v(q)·wing_G(q̂) / (1 − …)            # wing:  ~ 1/q, sign-odd in q̂
W_GG'(q)  = ε⁻¹_GG'(q)·v(q+G')                   # body:  O(1), smooth through q=0
```

**BGW's coarse→fine split** (`bgw_fine_grid_reference.md §2-3`). `kernel.x` stores
the direct kernel as **head / wing / body** with the divergent q-factors stripped so
the smooth remainder can be interpolated and the singular factors re-attached
analytically at fine q:

- head stored as `wptcol(1)=1.0` (SEMICOND) — the whole `v·ε⁻¹₀₀` is stripped
  (`BSE/mtxel_kernel.f90:532-533`);
- wings stored as `ε⁻¹·v·q` = O(1) (SEMICOND no-trunc, `calc_wings`
  `mtxel_kernel.f90:1038`) — the missing `1/q` re-added at fine q; **wing at
  coarse q_co=0 is zeroed** (Baldereschi-Tosatti, `:1043-1046`);
- body stored whole `ε⁻¹_GG'·v(q+G')` (`:555-558`), plus an **unscreened bare tail**
  `v(G)` for G beyond the ε⁻¹ cutoff (`calc_direct_unscreened_contrib` `:844-887`).

`intkernel.x` interpolates the smooth pieces onto each fine `q=k−k′`
(`interpolate`, `intkernel.f90:1266-1574`) and re-multiplies analytic factors
(`bsemat_fac`, `intkernel.f90:1103-1146`):

```
head:  fac_d · w_eff,   w_eff = ⟨v(q_fi)⟩_mBZ · ε⁻¹₀₀(q_fi);  q_fi=0 → wcoul0/8π
wing:  fac_d · oneoverq(q_fi)                          # restores the 1/q stripped coarse
body:  fac_d
```

with `ε⁻¹₀₀(q_fi)` interpolated from the `epsdiag` point cloud tabulated over q+G
beyond the 1st BZ (`epsdiag.f90`, `intkernel.f90:331-347`), and `⟨v⟩_mBZ`/`oneoverq`
from `vcoul_generator` + `minibzaverage_3d_oneoverq2`
(`vcoul_generator.f90:62`, `minibzaverage.f90:35-90`). **`fixwings` is
Sigma-only** — grep finds zero BSE call sites (`bgw_fine_grid_reference.md:57,
660-664`); the BSE wing mechanism is `calc_wings` (coarse) + `oneoverq` re-add
(fine). `average_w` (wcoul0 = ⟨v⟩·ε⁻¹₀₀) is always on in absorption
(`bgw_fine_grid_reference.md:322-324`).

The LORRAX analogue: the ISDF `W(μ,ν;q)` is **not** literally G-indexed, so head/wing/
body live in the ζ-projection. `G0_mu_nu` is the G=0 projector, so the head channel
is the rank-1 `conj(g0)⊗g0` direction, the wings are the `g0⊗(body)` /
`(body)⊗g0` cross-rows, and the body is the complement — precisely the split
`head_wing_schur.py` already computes.

---

## Proposed design

**Core idea.** Stop storing the head as a scalar. Store W as **smooth body**
`W_body(μ,ν;q)` + the **anisotropic head generator** `S_cart(ω)` + the **wing
vectors** `A_wing(μ;q)`, `A_wingp(ν;q)`. On any q (coarse q=0 or fine q≠0),
re-assemble `W(q)` by (a) interpolating the smooth body, (b) evaluating the analytic
head `W_head(q̂)` and the wing rank-1 per fine q. All of (a)-(b) route through the
**promoted** `head_wing_schur` kernel — one code path, no scalar-head branch left.

### Dataflow

```
GW side (producer):
  chi0_q ─┐                         S_cart(ω) = compute_S_omega(dipole.h5)   [reuse]
  V_q  ───┼─► extract_V_body ──► solve_W_body0 ──► W_body(μ,ν;q)   [head_wing_schur]
  g0   ───┘   (rank-1 subtract)   (Dyson, w_isdf)   ├─ chi_wing(μ;q) = χ_body·g0
                                                     └─ A_wing, A_wingp  [schur_reductions]
  persist:  W_body_qmunu (nq,μ,ν),  S_cart (n_ω,3,3),  A_wing/A_wingp (nq,μ),
            g0 (μ,), v_head(q̂-independent bare), χ_head_eff(q)   ── restart h5

BSE side (consumer), per fine q = k_fi − k'_fi:
  W_head(q̂) = v(q_fi) / (1 − v(q_fi)·q̂ᵀ S q̂)                     [analytic, bulk_3d]
  W_body(q_fi) = interp_body(W_body_qmunu, dcc/dvv or FFT)         [smooth]
  W(q_fi) = W_body(q_fi) + W_head(q̂)·(conj(g0)+A_wing)⊗(g0+A_wingp) [assemble_W]
```

### File-level plan

**Promote, do not duplicate** (`sources/lorrax_D`):

1. `src/gw/experimental/head_wing_schur.py` → `src/gw/head_wing.py`. Drop the
   `experimental`/`pytest.mark.extra` gate; it becomes the **single** head/wing/body
   path. Delete `apply_q0_head_rank1` / `apply_q0_head_rank1_sharded`
   (`head_correction.py:743-815`) once every caller is moved — the scalar rank-1
   injection is subsumed by `assemble_W_sharded` with `A_wing=0`,
   `W_head_scalar = whead/V_cell` at q=0 (bit-reproduces today's result, gate G1).
   No parallel old/new path survives.

2. New `src/gw/head_generator.py` — the anisotropic evaluator (procedural, plain
   arrays). One function `w_head_directional(v_q, qhat, S_cart) -> W_head` computing
   `v_q/(1 − v_q·q̂ᵀSq̂)`; and `w_head_minibz(wfn, meta, S_cart) -> wcoul0`
   that is exactly today's `bulk_3d.q0_average` S-branch (`bulk_3d.py:47-56`) —
   **move** that branch here so both the scalar (coarse) and per-q (fine) heads read
   one formula. `bulk_3d.q0_average` calls into it. `S_cart` comes from
   `chi_from_dipole.compute_S_omega` — **reused verbatim**, no new dipole code.

3. Producer wiring `src/gw/gw_output.py:persist_w0_and_head`: replace the
   `W0_qmunu` + scalar `whead` write with `W_body_qmunu` + `A_wing`/`A_wingp` +
   `S_cart` + `χ_head_eff(q)`. `write_head_scalars_to_h5`
   (`tagged_arrays.py:157-193`) grows the wing/S datasets; keep `vhead`/`whead[0]`
   for the coarse-grid back-compat path and gate G1.

4. Consumer wiring `src/bse/bse_io.py`: `load_bse_data_from_restart_sharded`
   (`bse_io.py:358-536`) loads `W_body` + wings + `S_cart` (dual g0 copies already
   built at `:463-484`). New `src/bse/w_interp.py` holds the fine-q assembly
   (below). The head-injection call sites (`bse_io.py:504,818`) call
   `assemble_W_sharded` instead of `apply_q0_head_rank1*`.

**Two-phase interpolation** (`src/bse/w_interp.py`, new, ~procedural):

- *Coarse-grid path (Phase 1, immediately testable).* On the single q=0 point the
  wing averages to zero (Baldereschi-Tosatti) and the head is the mini-BZ scalar —
  so Phase 1 reproduces today's physics while re-plumbing storage through the
  head/wing kernel. Deliverable: identical eigenvalues (G1) with the scalar-head
  code deleted.

- *Fine-grid path (Phase 2).* For fine `q_fi = k_fi − k'_fi`:
  - **body**: interpolate `W_body(q_fi)` from coarse `W_body_qmunu`. Two options,
    chosen to compose with the dcc/dvv design (seam below): (i) BGW-faithful
    Delaunay/`intkernel` interpolation of the smooth body in centroid basis; (ii)
    if the fine grid is a uniform refinement, a larger padded 3-D FFT of `W_body`
    (the body is smooth ⇒ FFT-safe). Start with (ii) for uniform fine grids —
    reuses the existing `make_sharded_ifftn_3d` convolution machinery
    (`bse_ring_comm.py:20-23`) at a larger grid, no new interpolator.
  - **head**: `W_head(q̂_fi) = v(q_fi)/(1 − v(q_fi)·q̂ᵀ_fi S q̂_fi)` analytic per fine
    q (`head_generator.w_head_directional`). At `q_fi=0` fall back to `wcoul0`.
  - **wing**: rank-1 `W_head(q̂_fi)·(conj(g0)+A_wing)⊗(g0+A_wingp)` added per fine q
    (`assemble_W_sharded`). The `1/q_fi`-like growth lives entirely in
    `A_wing∝χ_wing` and `W_head`; unlike the coarse grid it does **not** cancel.
  - **critical FFT caveat**: the head/wing are non-smooth in q (diverge at q→0) and
    must **not** enter the body FFT convolution. Build `W_R` from `W_body` only, and
    fold the analytic head+wing as a **separate additive per-q term** in the k−k′
    accumulation — exactly BGW's structure (smooth interpolated body + analytic
    singular re-add). This is the one place the matvec changes.

### Explicit formulas for the physics-critical pieces

```
q̂ᵀ S q̂ :   qSq(q_fi) = Σ_αβ q̂_α S_αβ(ω=0) q̂_β,   q̂ = q_fi/|q_fi|_bdot
head    :   W_head(q_fi) = v(q_fi) / (1 − v(q_fi)·qSq(q_fi))      # v=8π/|q_fi|²
wing χ  :   chi_wing(μ;q) = Σ_ν χ_body(μ,ν;q) g0(ν)              # G=0 column of χ
A_wing  :   A_wing(μ;q)  = Σ_ν W_body0(μ,ν;q) chi_wing(ν;q)      # schur_reductions
χ_head_eff(q) = χ_head(q) + Σ_μν chi_wingp(μ) W_body0(μ,ν) chi_wing(ν)
assemble:   W(μ,ν;q) = W_body0(μ,ν;q)
                     + W_head(q)·(conj(g0)+A_wing)(μ) · (g0+A_wingp)(ν)
```

All four are already coded in `head_wing_schur.py:178-235`; the only new physics is
`W_head(q_fi)` becoming q̂-directional (Phase 2) instead of the scalar
`v_head/(1−v_head·χ_head_eff)` (Phase 1).

### Sharding + memory plan

Flat-q `(nq,μ,ν)` on `P(None,'x','y')` = `V_FLATQ_SPEC` (`head_wing_schur.py:51`);
dual g0/wing copies `P(None,'x')`/`P(None,'y')` make every rank-1 subtract/rebuild
zero-comm (`head_wing_schur.py:99-106,232-234`). Schur reductions are three small
per-q all-reduces (`:167-169`). `S_cart` is `(n_ω,3,3)` — replicated, negligible.
`W_body` is the same footprint as today's `W0_qmunu`; wings add `2·nq·μ` (tiny vs
`nq·μ²`). Fine-grid body FFT (Phase 2 opt ii) enlarges the k-axes of `W_R` — this is
the memory driver and must go through the existing 1-GPU planner
(`gflat_memory_model` / the BSE `_pad_axis` path), not a new arena. `S_cart` and
`dipole.h5` are host-resident, pulled via the existing loader (no jit args) — the
`chi_from_dipole` reader is host numpy already (`chi_from_dipole.py:99-103`).

---

## Interactions with the other four designs (shared seams)

- **dcc/dvv coarse→fine wavefunction interpolation design.** My `W(q_fi)` is
  contracted by *their* `dcc/dvv` coefficients to form `K^d_fi` (BGW
  `intkernel.interpolate`, `bgw_fine_grid_reference.md §3.5`). Seam: the fine-q
  index convention and whether body interpolation is FFT (uniform) or Delaunay
  (their `intwfn` cloud). We must agree on **one** fine-q table and route it through
  the canonical `SymMaps`, never a parallel "rotate W at q" helper.
- **Exchange-kernel (B1) design.** B1 (`kernel_dataflow_trace.md:368-388`) makes the
  exchange dense in (k,k′). The exchange head is the *bare* `vhead` (no screening);
  my `V_q0` head injection (`apply_q0_head_rank1` today) is shared code being
  deleted — coordinate so the exchange design consumes the same `g0`/`vhead` seam
  from the promoted `head_wing` module.
- **Restart-schema / loader (B3/B4/B5) design.** The loaders are 8-D-only / demand a
  missing `kgrid` attr / inject before the layout shim
  (`kernel_dataflow_trace.md:397-420`). My new `W_body`/wing/`S_cart` datasets must
  land in the **fixed** flat-q schema, not the broken one — sequence my
  `tagged_arrays` writes after their reader fix.
- **Finite-Q design.** Finite-Q shifts the head to `v(G−Q)` and the exchange head to
  `q=−Q` (`bgw_fine_grid_reference.md §4`). My `W_head(q̂)` generator must accept a
  Q-offset q̂; keep the head evaluator Q-aware from the start so finite-Q reuses it.
- **Non-TDA / solver designs.** The coupling block re-attaches head/wing with the
  conjugation swap (`bgw_fine_grid_reference.md:403-410`); `assemble_W_sharded` is
  block-agnostic, so the same kernel serves A and B blocks — no second wing path.

---

## Gates (1-GPU validation plan, BGW anchors)

All gates run on MoS2 3×3 / Si 4×4×4 fixtures, ≤1 GPU (mesh simulated on one
device, as `test_head_wing_schur.py` already does — no 16-GPU gating).

- **G1 — bit-reproduce today.** Promoted `head_wing` with `A_wing=0`,
  `W_head=whead/V_cell` at q=0 must reproduce the current Si 4×4×4 8v×8c eigenvalues
  to machine precision (the scalar-injection result, `context_docs.md:135-138`).
  Proves the storage refactor is behaviour-preserving before wings turn on.
- **G2 — Schur == dense.** `reconstruct_W_via_schur_sharded` vs dense `(I−Vχ)⁻¹V`
  at 1e-12 (this is `tests/test_head_wing_schur.py` check 1, promoted out of
  `extra`) on a Si-scale centroid count.
- **G3 — anisotropic head averages to scalar.** Mini-BZ average of
  `w_head_directional(q̂,S)` over Sobol q reproduces `bulk_3d.q0_average`'s `wcoul0`
  to ~1e-6 (proves the directional generator is consistent with the coarse scalar).
- **G4 — wing re-attach vs BGW fine grid.** On a Si coarse 4×4×4 → fine 8×8×8 run,
  compare LORRAX head+wing eigenvalues against BGW `absorption.x` with
  `kernel.x` head/wing/body (BGW anchor). Head-only vs head+wing delta must match
  BGW's (the wing *is* the accuracy fix). Diff `oneoverq` re-add magnitude vs
  BGW `intkernel.f90:1103-1146`.
- **G5 — Baldereschi-Tosatti coarse cancellation.** On the single coarse grid,
  head+wing must equal head-only (wing→0 at q_co=0) — guards against a spurious
  coarse-grid wing.

---

## Open questions for Jack

1. **Fine-grid priority.** Is the immediate target the fine-grid BSE (full
   coarse→fine, needs the dcc/dvv design too), or just fixing the coarse q=0 head to
   be anisotropic/wing-correct? Phase 1 is cheap and testable now; Phase 2 depends on
   the wfn-interp design landing. Which unblocks your science first?
2. **Body interpolation scheme.** Uniform-refinement FFT (reuses existing
   convolution, only uniform fine grids) vs BGW-faithful Delaunay `intkernel`
   (arbitrary fine points, new interpolator). Do your target fine grids stay uniform?
3. **Unscreened bare tail.** BGW adds `v(G)` beyond the ε⁻¹ cutoff to the body
   (`mtxel_kernel.f90:844-887`). LORRAX's ISDF body has no explicit G cutoff — is the
   ζ basis's implicit cutoff acceptable, or do we need an explicit bare-tail term for
   BGW parity?
4. **S(ω) frequency.** For dynamic (GN-PPM) BSE screening, do we need
   `W_head(q̂,ω)` at ω≠0 (S_cart already supports it, `compute_S_omega`), or is the
   static head+wing sufficient for the TDA BSE you run?

---

## LOC estimate + suggested phasing

- **Phase 1 — storage refactor + coarse anisotropic head, wings plumbed but zero at
  q=0.** Promote `head_wing_schur`→`head_wing` (0 new, delete `extra` gate); new
  `head_generator.py` (~80 LOC, mostly *moved* from `bulk_3d.q0_average`); rewire
  `gw_output.persist_w0_and_head` + `tagged_arrays` writers (~120 LOC); rewire
  `bse_io` loaders + delete `apply_q0_head_rank1*` (~100 LOC net). Gates G1-G3,G5.
  **~300 LOC, mostly move/delete.**
- **Phase 2 — fine-q body interpolation + analytic head/wing re-attach.** New
  `src/bse/w_interp.py` (~250 LOC), matvec hook to keep head/wing out of the body FFT
  and fold as per-q additive term (~150 LOC across `bse_serial`/`bse_simple`/
  `bse_ring_comm` — single-sourced via one `apply_W_headwing` helper, not three
  copies). Gate G4 vs BGW fine grid. **~400-450 LOC.** Depends on the dcc/dvv design
  for non-uniform grids; the uniform-FFT sub-path is independent.

Total ~700-750 LOC, front-loaded on deletion/move. Phase 1 is a self-contained,
behaviour-preserving refactor that de-risks Phase 2.
