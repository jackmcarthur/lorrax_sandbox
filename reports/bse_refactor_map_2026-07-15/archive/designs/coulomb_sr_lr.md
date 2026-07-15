# Design: separable short-range / long-range Coulomb split for fine-grid BSE

Program: BSE refactor-map, 2026-07-15. Checkout `sources/lorrax_D`
(src/bse tree byte-identical between e18d0e5 and working HEAD adc2197 —
`kernel_dataflow_trace.md:3-7`). BGW citations are read-only reference from
`sources/BerkeleyGW`. Ry units throughout; LORRAX stores `v(q+G)=8π/|q+G|²`
already divided by `V_cell` (`coulomb/bulk_3d.py:23-24`,
`coulomb/base.py:44-47`).

This design owns ONE thing: the split `v(q+G) = v_SR + v_LR` that makes the
coarse exchange/direct kernels smooth enough to interpolate onto a fine q-grid,
with the divergent `v_LR` re-added in closed form at fine q. It does **not**
own the wavefunction (dcc/dvv) or ε⁻¹₀₀(q) interpolation — those are separate
designs it hands off to (see Interactions).

---

## Current state in LORRAX (file:line grounded)

**Single-grid BSE, no split, no interpolation.** The kernel is assembled once
on the coarse GW k-grid entirely from the ISDF restart bundle:

- Exchange `V` uses one `V_q0[μ,ν]` slab (bare Coulomb, G=G'=0 zeroed):
  `bse_serial.py:62-64`, `bse_simple.py:89-131`, contracted k-block-diagonally
  (`kernel_dataflow_trace.md:368-388`, suspect B1 — orthogonal to this design).
- Direct `W` FFT-convolves `W_q[μ,ν,kx,ky,kz]` over the coarse k-grid:
  `bse_serial.py:69-78`, `bse_simple.py:141-173`. `W_q` index is exactly
  `q=k−k'` (`kernel_dataflow_trace.md:38-49`).
- `W_q`/`V_q0` are (μ,ν) **centroid** matrices, not G-space
  (`bgw_fine_grid_reference.md:643-645`) — there is no head/wing/body G,G'
  structure to exclude.

**q→0 head, the only "singular re-add" that exists today.** Injected rank-1
after load: `V_q0 += (vhead/V_cell)·conj(g0)⊗g0`, `W_q[:,:,0,0,0] +=
(whead[0]/V_cell)·conj(g0)⊗g0`, with `g0_μ = ζ(q=0,μ,G=0)`
(`bse_io.py:463-513` sharded, `804-836` ring; helper
`head_correction.apply_q0_head_rank1{,_sharded}` at
`head_correction.py:743-816`). `vhead=vc0=⟨v⟩_mBZ`, `whead=wcoul0=⟨v⟩_mBZ·ε⁻¹₀₀`
persisted by `gw_output.persist_w0_and_head` (`gw_output.py:210-241`).

**The live coarse Coulomb seam.** Two builders exist; only one is live:
- `compute_v_q_per_G` (`compute_vcoul.py:81-196`) — **LIVE**. Per-q `v(q+G)` on
  the WFN sphere with an internal `sys_dim` if/elif (3D `165`, 2D `178-183`,
  box `184-192` unwired). Consumed by exchange `v_q_g_flat.py:531` and bispinor
  `v_q_bispinor.py:127`. `vcoul_cutoff_ry` zeroes `v` past `|q+G|²>cutoff`
  (`:193-194`) — this is `bare_coulomb_cutoff`. The 3D G=0 slot is overwritten
  by the mini-BZ head table `build_v_head_miniBZ_avg_3d` (`:32-78, 167-177`).
- `CoulombKernel.v_qG` (`coulomb/base.py:50-52`, `bulk_3d.py:15-26`,
  `slab_2d.py:21-38`, `box_0d.py:24-30`) — **DEAD as a per-G builder**: grep
  finds callers only inside docstrings (`coulomb/__init__.py:11-14`,
  `base.py:6`). Its sibling `q0_average` **is** live via
  `vcoul.compute_q0_averages → get_kernel(sys_dim).q0_average`
  (`vcoul.py:63-68`, `head_correction.py:131-186`). So per-dim `v(q+G)` and
  per-dim `q0` currently live in two places — a split-brain to consolidate here.

`W_q` itself is `W = (1 − Vχ₀)⁻¹ V` in the μν basis, built from the same bare
`V_q` (`w_isdf.py:3`, screening solve). **W is not linear in V** — the SR/LR
split of `W` is therefore *not* the split of its input `V` (see Proposed).

Grep confirms **no coarse→fine interpolation of any kind** in `src/bse`
(`bgw_fine_grid_reference.md:637-638`).

---

## Reference physics + BGW implementation

Two ways to make the Coulomb kernel interpolation-safe. Both remove the same
divergence; they differ in basis-friendliness.

### (A) BGW "exclude head and wings" (the reference implementation)

BGW stores the coarse direct kernel `W_GG'(q)` in three separately-normalized
pieces with the divergent q-factors **stripped**
(`mtxel_kernel.f90:517-559`, `bgw_fine_grid_reference.md:194-218`):

| piece | stored (semicond, untrunc) | re-added at fine q |
|---|---|---|
| head `G=G'=0` | `1.0` (whole `v·ε⁻¹₀₀` stripped) | `w_eff = ⟨v(q_fi)⟩_mBZ·ε⁻¹₀₀(q_fi)`; at q=0 `wcoul0/(8π)` |
| wing `G=0 xor G'=0` | `ε⁻¹_wing·v·q` (O(1)) | `× oneoverq = 1/|q_fi|` |
| body `G,G'≠0` | full `ε⁻¹_GG'·v(q+G')` | `×1` (already smooth) |

Re-add formula (`intkernel.f90:882-890, 1103-1146`): `w_eff = vcoul·eps`, with
`eps=ε⁻¹₀₀(q_fi)` interpolated from an **epsdiag** point cloud
(`epsdiag.f90`, `bgw_fine_grid_reference.md:282-298`) and `vcoul=⟨v⟩_mBZ`
mini-BZ-averaged at **every** fine q for a 3D semiconductor
(`avgcut=∞`, `bgw_fine_grid_reference.md:302-308`). `wcoul0 = ⟨v⟩_mBZ·ε⁻¹₀₀`
(`minibzaverage.f90:79-85`, `vcoul_generator.f90:83-85`). Exchange head zeroed,
`vbar(G=0)=0` (`gx_sum.f90:53`, `bgw_fine_grid_reference.md:228-235`).

**Problem for LORRAX**: this split is indexed by (G,G'). LORRAX's `W_q` is a
compressed (μ,ν) centroid matrix — there is no G=0 row/column to exclude. To
mirror BGW literally, the head/wing separation would have to happen in the
G=0 projection of the ζ basis *before* ISDF compression
(`bgw_fine_grid_reference.md:644-645`), which means re-plumbing the ζ-fit /
W-solve — a large, basis-invasive change.

### (B) erfc/Gaussian range separation (recommended)

Split the *scalar* `v(q+G)` by the Ewald/range-separation identity
(Rydberg, `1/r → erf + erfc`):

```
v_LR(q+G) = (8π/|q+G|²) · exp(−|q+G|²/4α²)          # long-range in r; carries the q→0 divergence
v_SR(q+G) = (8π/|q+G|²) · (1 − exp(−|q+G|²/4α²))    # finite ∀ q,G; smooth ⇒ interpolation-safe
v         = v_SR + v_LR                              # exact, per-G, machine precision
```

Key values: `v_SR(q+G→0) = 2π/α²` (finite — no divergence to interpolate);
`v_LR` inherits the full `1/|q+G|²` singularity but is **closed-form and
smooth in q**, so it is re-added *analytically* at any fine q with zero
interpolation error. At large G, `v_LR ~ exp(−G²/4α²) → 0`, so `v_SR → v`
(the bare-cutoff regime is untouched).

**Why (B) for LORRAX**: the split is a per-G *scalar multiply* on `v(q+G)`,
which threads through the existing centroid contract
`V_q[μν]=Σ_G conj(ζ̃_μ) v(q+G) ζ̃_ν` (`v_q_g_flat.py:24`) **with no G,G'
structure** — basis-agnostic, no ζ-fit surgery. It reuses `compute_v_q_per_G`
(already per-G, already host-side) verbatim. BGW's own head re-add is the α→∞
/ G=0-only limit of this (all singular weight collapses onto G=0); (B) is the
smooth generalization that survives ISDF compression.

Truncation: the Gaussian multiplies the **untruncated** `8π/|q+G|²`; the
dimensional envelope stays as an outer factor, so
`v_{SR/LR} = (8π/|q+G|²)·f_dim(q+G)·{(1−e),e}` and `v_SR+v_LR=v` still holds
exactly for slab (`f2d`, `slab_2d.py:29-37`) and 3D (`f=1`). Box (0D) has a
**finite** `v(q=G=0)` from the WS FFT (`box_0d.py:40-53`) — no divergence, so
`v_LR≡0` and the whole kernel is already interpolation-safe; the box path skips
the split.

---

## Proposed design

### The ONE seam both v and W use

`v(q+G)` is the single upstream object: exchange contracts it into `V_q`, and
the screening solve `W=(1−Vχ)⁻¹V` consumes the same `V_q` (`w_isdf.py:3`). So
the split is defined once, at the per-G Coulomb evaluator, and everything
downstream inherits it.

```
                     ┌───────────────────────────────────────────────┐
   ζ̃(q,μ,G) ──┐      │  get_kernel(sys_dim).v_qG_split(q,G; α, cutoff)│  ← THE SEAM
              ├─────►│      → (v_SR, v_LR)   [v_LR=None for box]      │    (per-dim, single source)
   q+G ───────┘      └───────────────┬───────────────────────────────┘
                                     │ Σ_G conj(ζ̃_μ)·(·)·ζ̃_ν   (v_q_g_flat, unchanged contract)
                     ┌───────────────┴───────────────┐
                     ▼                                ▼
              V^SR_q[μν]  (coarse, smooth)     V^LR_q[μν]  (coarse, analytic-reproducible)
                     │                                │
       exchange ─────┤                                ├──► W^LR = ε⁻¹₀₀(q)·V^LR   (macroscopic-screening rule)
                     │                                │
   W-solve: W=(1−Vχ)⁻¹V ─► W_q[μν] ──► W^SR = W − ε⁻¹₀₀(q)·V^LR   (coarse, smooth)
                                                      │
   ─── restart: store V^SR_q, W^SR_q, + α, + ε⁻¹₀₀(q) table ──────────────────────────►
                                                      │
   ═══ FINE GRID (interpolation design interpolates V^SR/W^SR; THIS design re-adds LR) ═══
        V^fi = interp(V^SR) + Σ_G conj(ζ̃_μ(q_fi)) v_LR(q_fi+G) ζ̃_ν(q_fi)
        W^fi = interp(W^SR) + ε⁻¹₀₀(q_fi) · V^LR_fi[μν]
        q_fi→0 head: rank-1  ε⁻¹₀₀·⟨v_LR⟩_mBZ  via apply_q0_head_rank1 (generalized per fine-q)
```

**Direct-kernel split (the non-trivial piece).** Because `W` is nonlinear in
`V`, split `W` by its *physical* singularity, not its input. Near q→0,
`W(q) → ε⁻¹₀₀(q)·v(q)` (macroscopic screening); local-field/body corrections
are smooth. Isolate that with the same `v_LR`:

```
W^LR_q[μ,ν] = ε⁻¹₀₀(q) · V^LR_q[μ,ν]                 # rank-follows the LR Coulomb tile
W^SR_q[μ,ν] = W_q[μ,ν] − ε⁻¹₀₀(q) · V^LR_q[μ,ν]      # divergence removed ⇒ finite at q→0, smooth
```

`ε⁻¹₀₀(q)` is the scalar dielectric head already available GW-side (it is the
`wcoul0/vc0` ratio, `head_correction.py:177-186`; the fine-grid tabulation over
q is the epsdiag design). On the coarse grid `W^SR + ε⁻¹₀₀·V^LR = W` exactly
→ non-regression is algebraic, not numerical.

**q→0 head, re-partitioned but scalars unchanged.** Today `vhead=⟨v⟩`,
`whead=⟨v⟩·ε⁻¹₀₀` (`gw_output.py:236-238`). With the split, `⟨v⟩=⟨v_SR⟩+⟨v_LR⟩`
and `⟨v_SR⟩=v_SR(0)=2π/α²` is a plain finite tile entry (it rides in `V^SR`/
`W^SR` at q=0 with no special-casing). Only `⟨v_LR⟩_mBZ` stays divergent and is
the rank-1 injection. **The persisted `vhead`/`whead` scalars do not change**;
`apply_q0_head_rank1{,_sharded}` is reused verbatim on the coarse grid, and
generalized to *per fine-q* rank-1 adds `ε⁻¹₀₀(q_fi)·⟨v_LR(q_fi)⟩_mBZ·g0⊗g0`
on the fine grid.

### File-level plan (single-sourced, no parallel paths)

**Consolidate the split-brain first** (no-redundancy rule):
- `coulomb/base.py`: add `_v_lr_gaussian(qG_cart, alpha, envelope)` — the
  shared Gaussian formula (bulk+slab reuse; procedural, plain arrays). Add
  `sample_minibz_qpoints`-based `⟨v_LR⟩_mBZ` to the existing sampler (it
  already draws the points, `base.py:107-162`).
- `coulomb/{bulk_3d,slab_2d}.py`: give `v_qG` a `sr_alpha`/`cutoff_ry` kwarg
  and return `(v, v_lr)` (v_lr `None` when `sr_alpha is None`). Fold the
  truncation envelope in (slab keeps `f2d`, `slab_2d.py:32`). `box_0d.py`:
  return `(v, None)` — box is split-exempt.
- `compute_vcoul.py:compute_v_q_per_G`: **delete the internal `sys_dim`
  if/elif** (`:164-192`) and delegate per-q to `get_kernel(sys_dim).v_qG`.
  This kills the duplicate per-dim Coulomb logic and makes `v_qG` live again —
  one per-dim Coulomb source. Keep the host-side per-q loop, the
  `vcoul_cutoff_ry` gate (`:193-194`), and the `v_head_miniBZ` G=0 overwrite
  (`:167-177`); pass `sr_alpha` through so callers get `(v, v_lr)`.
- `v_q_g_flat.compute_all_V_q_g_flat` (`v_q_g_flat.py:485-556`): when
  `sr_alpha` set, contract `v_lr` through the *same* `_bare_v_per_G` reduction
  (`:530-555`) to emit `V^LR_q` alongside `V_q`. Zero new tiling.
- **New** `coulomb/range_sep.py` (~120 LOC, procedural): `split_alpha_default(
  kgrid, bvec)` (α from k-grid spacing), `v_lr_at_qG(q_frac, comps, wfn, alpha)`
  (the closed-form fine-q evaluator — no interpolation), and
  `readd_lr_direct(W_sr_munu, epsinv00_q, V_lr_munu)` →
  `W_sr + ε⁻¹₀₀·V_lr` (the fine-grid re-add, a local μν add, comm-free like the
  rank-1 head). Exchange re-add is `V_sr + V_lr` (same helper, `epsinv00=1`).
- `gw_output.persist_w0_and_head`: also persist `W^SR_q` (in place of `W_q`
  when `sr_alpha` set), the scalar `α`, and the coarse `ε⁻¹₀₀(q)` table (feeds
  both W^SR construction and the fine re-add). `vhead`/`whead` unchanged.
- `bse_io`: on the fine-grid path, load `V^SR/W^SR/α/ε⁻¹₀₀`; the existing
  coarse-only path loads `V^SR/W^SR` and re-adds LR at coarse q (identity to
  today). Head injection unchanged (`bse_io.py:463-513`).

**Deleted duplicates**: `compute_v_q_per_G`'s `sys_dim` branching (folded into
the per-dim kernels); the dead `CoulombKernel.v_qG` docstring-only status
(resurrected as the real seam).

### Formulas (physics-critical, explicit)

```
α:  default α = c_α · (2π/L̄) / N̄,  L̄ = min cell dim, N̄ = k-grid density   # v_LR ≲ 1 q-shell wide
    (c_α ~ O(1), a Jack knob — see Open questions)

v_LR(q+G) = 8π · exp(−|q+G|²/4α²) / |q+G|² · f_dim(q+G) / V_cell
v_SR(q+G) = 8π · (1 − exp(−|q+G|²/4α²)) / |q+G|² · f_dim(q+G) / V_cell
    f_3D = 1;  f_2D = 1 − e^{−z_c|q‖+G‖|}cos((q_z+G_z)z_c), z_c=π/b_z (slab_2d.py:29,32)
    q+G→0:  v_SR → 2π/(α² V_cell) [·f_2D→0 for slab];  v_LR → 8π/(|q+G|²V_cell)

V^{SR/LR}_q[μ,ν] = Σ_G conj(ζ̃_{q,μ}(G)) · v_{SR/LR}(q+G) · ζ̃_{q,ν}(G)    # v_q_g_flat contract
W^LR_q[μ,ν]      = ε⁻¹₀₀(q) · V^LR_q[μ,ν]
W^SR_q[μ,ν]      = W_q[μ,ν] − W^LR_q[μ,ν]
⟨v_LR⟩_mBZ(q)    = (1/N_s) Σ_s v_LR(q+δq_s),  δq_s ∈ mini-BZ Voronoi (base.py:107-162)
whead unchanged  = ⟨v⟩_mBZ·ε⁻¹₀₀ = (⟨v_SR⟩+⟨v_LR⟩_mBZ)·ε⁻¹₀₀     # gw_output.py:236
```

### Sharding + memory plan

- `v_SR/v_LR` per-G tables: `(n_q, ngkmax)` host float64 built once in
  `compute_v_q_per_G` (host, not jitted — `compute_vcoul.py:125`). Same
  footprint as today's single `v`; store `v_LR` only and derive `v_SR=v−v_LR`
  to add zero storage.
- `V^SR_q`, `W^SR_q`: the existing `V_qmunu`/`W0_qmunu` `(nq,μ,ν)` tiles,
  `P('x','y',…)` (`kernel_dataflow_trace.md:167-172`). Sharding **unchanged**;
  the LR re-add is a rank-following μν add → device-local, comm-free (same
  pattern as the rank-1 head, `head_correction.py:805-815`).
- Coarse `ε⁻¹₀₀(q)`: `(nq,)` scalar-per-q, replicated (tiny).
- Fine `v_LR(q_fi+G)`: `(n_q_fine, ngkmax)`. For MoS2/Si-scale this is tens of
  MB → fine. If it exceeds device budget on larger fine grids, it is a
  read-only host cache pulled per fine-q slice via `io_callback` (never a jit
  arg — per the io_callback rule); the analytic evaluator `v_lr_at_qG` makes
  regeneration cheap so caching is optional.
- No new replicated intermediates; the split adds one `(nq,μ,ν)` LR tile that
  can be freed after `W^SR` is formed (it is reproducible from `α`+ζ).

---

## Interactions with the other four designs (shared seams)

1. **Wavefunction (dcc/dvv) interpolation** — owns `interp(V^SR)`,
   `interp(W^SR)` and the interpolated `ζ̃(q_fi)`. THIS design hands it
   interpolation-safe coarse tiles and, for the fine LR re-add, the closed-form
   `v_lr_at_qG(q_fi)` scalar; it contracts that with *its* interpolated
   `ζ̃(q_fi)`. Seam: `V^SR/W^SR` coarse tiles + `range_sep.v_lr_at_qG`.
2. **ε⁻¹₀₀(q) tabulation (epsdiag analogue)** — owns the fine-q dielectric
   head. THIS design consumes `ε⁻¹₀₀(q)` twice: to build `W^SR = W − ε⁻¹₀₀V^LR`
   coarse, and to scale `W^LR = ε⁻¹₀₀(q_fi)V^LR` fine. Seam: the `ε⁻¹₀₀(q)`
   table (coarse for construction, fine for re-add).
3. **q→0 head / wcoul0 machinery** — already shared: `q0_average`
   (`vcoul.py:63`, `coulomb/*.q0_average`) supplies `⟨v⟩`/`wcoul0`;
   `apply_q0_head_rank1{,_sharded}` (`head_correction.py:743-816`) does the
   rank-1 add. THIS design reuses both unchanged on coarse q and generalizes the
   rank-1 add to per-fine-q. Seam: `⟨v_LR⟩_mBZ` + `apply_q0_head_rank1`.
4. **Finite-Q BSE** — the split is Q-independent: exchange shifts to `v(q+G−Q)`
   (`bgw_fine_grid_reference.md:481-489`); `v_qG_split` takes the shifted
   argument and the same α. No coupling beyond passing the shifted q. Seam:
   the `v_qG_split` argument.

Shared across all: the `bare_coulomb_cutoff` convention (default = ecutwfc,
BGW-parity; MEMORY note) — applied to full `v` after the split, harmless to
`v_LR` (already ~0 past the Gaussian). And `sys_dim` truncation dispatch via
`get_kernel` (`coulomb/base.py:75`), which every design routes through.

---

## Gates (1-GPU validation plan, BGW anchors)

All on MoS2 3×3 or Si 4×4×4 fixtures, 1 GPU (no 16-GPU gating).

1. **Per-G round-trip (unit).** `v_SR + v_LR == compute_v_q_per_G(...)` to
   1e-12 for random q,G, all three `sys_dim`, spanning α — the split is exact.
   Box returns `v_LR is None`. Runs on CPU.
2. **Coarse non-regression (integration, the load-bearing gate).** With the
   split enabled but interpolation OFF, the Si 4×4×4 8v×8c BSE eigenvalues must
   match the current no-split run bit-for-bit (algebraically `V^SR+V^LR=V`,
   `W^SR+ε⁻¹₀₀V^LR=W`). Anchor: the validated ~3 meV vs-BGW ledger
   (`STATUS.md:63-83`, `context_docs.md:135-148`). Any drift = a bug in the
   re-partition, not physics.
3. **Head-scalar invariance.** `vhead`, `whead=wcoul0` unchanged vs current
   `gw_output.persist_w0_and_head` output (`vhead=3303.748` reference,
   `kernel_dataflow_trace.md:141`) — the split re-partitions, does not
   re-value, the head.
4. **α-independence on the coarse grid.** Sweep α over ~decade; coarse
   eigenvalues invariant (α only moves weight between `V^SR` and `V^LR`, whose
   sum is fixed). This isolates split correctness from the interpolation gate.
5. **Fine-grid smoothness (diagnostic).** `‖V^SR(q)−V^SR(q')‖` across adjacent
   coarse q must be ≪ `‖V(q)−V(q')‖` (the divergent jump lives in `V^LR`) —
   quantifies that the split actually buys interpolability. BGW anchor: its
   head is stored `1.0` (maximally smooth, `mtxel_kernel.f90:532`); ours should
   approach that as α→∞.
6. **BGW absorption end-to-end** (once the interpolation design lands): fine
   ε₂(ω) vs `absorption.x` on a Si coarse-4³→fine-8³, reusing the existing
   Haydock compare harness (`context_docs.md:136-141`).

---

## Open questions for Jack (physics/priority only)

1. **α selection policy.** Fixed `c_α·(k-grid spacing)` default, or a
   convergence knob exposed in `cohsex.in` (`coulomb_sr_alpha`)? Tradeoff:
   large α → more of `W` rides the exact analytic `W^LR` channel (safer
   interpolation) but `W^SR` carries less physics (more sensitive ε⁻¹₀₀
   cancellation); small α → `W^SR` ≈ full `W` (interpolation does more work).
   BGW's implicit choice is α→∞ (head-only). Where on that axis do you want the
   default?
2. **Ansatz A vs B commitment.** Confirm we adopt Gaussian range-separation (B)
   and do **not** reproduce BGW's literal G=0/G'=0 head/wing exclusion (A).
   (B) is basis-agnostic and reuses the existing centroid contract; (A) would
   need head/wing separation inside the ζ-fit. Any reason to want bit-identical
   BGW head/wing storage rather than physical equivalence?
3. **Macroscopic-screening `W^LR = ε⁻¹₀₀·V^LR` accuracy.** This uses the scalar
   dielectric head for the whole LR neighborhood (local-field corrections in
   `W^SR`). For strongly anisotropic screening (2D slab, layered VI3/CrI3) do
   you want the anisotropic `S_cart` tensor (`bulk_3d.py:47-56`,
   `head_correction.py:166-186`) folded into the LR channel, or is the scalar
   head sufficient at the fine-grid resolution we target?
4. **Priority vs the interpolation design.** This split is a prerequisite for
   fine-grid W but delivers nothing standalone (coarse-only is identity). Land
   it first as scaffolding (gate 2 proves zero regression), or co-develop with
   the dcc/dvv design so gate 6 exists at merge?

---

## LOC estimate + suggested phasing

~430 LOC net (≈ +560 new, −130 deleted duplication).

- **Phase 1 — the seam + consolidation (~180 LOC, −130).** `_v_lr_gaussian`
  + `v_qG_split` in the three `coulomb/*` kernels; delete
  `compute_v_q_per_G`'s `sys_dim` if/elif and delegate to `get_kernel`.
  `range_sep.py` (α default + `v_lr_at_qG`). Gate 1. **No behavior change**
  (α=None path bit-identical). Ships independently.
- **Phase 2 — coarse V^SR/W^SR + head re-partition (~180 LOC).** LR contract
  in `v_q_g_flat`; `W^SR = W − ε⁻¹₀₀V^LR` in `w_isdf`/`gw_output`; persist
  `α`+`ε⁻¹₀₀`; `readd_lr_direct`. Gates 2-4 (coarse non-regression, the proof
  the re-partition is exact). Still no fine grid.
- **Phase 3 — fine-grid re-add hooks (~200 LOC).** Per-fine-q `v_lr_at_qG`
  contract + `ε⁻¹₀₀(q_fi)` scaling + generalized rank-1 head; io_callback cache
  if needed. Gate 5; gate 6 jointly with the interpolation design.

Phase 1 is safe to land immediately (consolidation + dead-code revival with an
inert α=None default). Phases 2-3 gate on the interpolation and epsdiag designs
existing to consume the fine re-add.
