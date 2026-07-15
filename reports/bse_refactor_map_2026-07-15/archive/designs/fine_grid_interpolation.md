# Design: coarse-to-fine k-grid interpolation of wavefunctions + BSE kernel

Author: fine-grid design agent, BSE refactor-map program, 2026-07-15.
Checkout: `sources/lorrax_D` (BSE tree byte-identical to the requested
`e18d0e5`; verified in `kernel_dataflow_trace.md:3-7`). All "current state"
line refs are HEAD.

Scope: the LORRAX analogue of BGW's `intwfn`/`intkernel` absorption machinery —
add a fine k/q grid on top of the coarse GW ISDF kernel, so exciton fine-structure
converges without recomputing screening on the fine grid. This design decides
**what LORRAX interpolates** (screened body of `W_μν(q)`, in real space) and **what
it samples directly** (fine ψ at the fixed centroids), and defers the divergent
Coulomb pieces to `coulomb_sr_lr` and `w_head_wings`.

---

## Current state in LORRAX (file:line grounded)

LORRAX BSE is **single-grid**: one coarse GW k-grid, no fine/coarse distinction
anywhere (`bgw_fine_grid_reference.md:632-638`; grep for interpolation in
`src/bse/*.py` is empty).

- ψ lives at ISDF centroids only, coarse k only: `psi_full_y = wfns.psi_yr`,
  shape `(nk, nb, nspinor, n_rmu)`, written by `gw_init.py:695`. **The centroids
  `r_μ` are k-independent real-space grid points** (`isdf/core.py:44-68` — the
  pair density is `Σ_n ψ*(r_μ)ψ(r_ν)`; `r_μ` are fixed). This is the pivot of the
  whole design.
- Kernel loaded from the restart bundle (`bse_io.load_bse_data_from_restart_sharded`,
  `bse_io.py:358-536`): `V_q0` `(μ,ν)` `P('x','y')`; `W_q` `(μ,ν,nkx,nky,nkz)`
  `P('x','y',None,None,None)` — **k-axes replicated** (`make_bse_shardings`,
  `bse_ring_comm.py:46-63`).
- Direct term is an FFT convolution over exactly the coarse `(nkx,nky,nkz)`
  (`bse_simple.py:146-162`): `T[b,μ,ν,t,s,k]` → `ifft_k` → `W_R·T_R` → `fft_k`.
  `T` is `P(None,'x','y',None,None,None)` — **k replicated**, and `T` is the
  memory hog (`kernel_dataflow_trace.md:178`).
- q→0 head: rank-1 `(vhead/V_cell)·conj(G0)⊗G0` into `V_q0`, and
  `(whead[0]/V_cell)·conj(G0)⊗G0` into `W_q[:,:,0,0,0]`, with `G0 = ζ(q=0,μ,G=0)`
  (`bse_io.py:487-513`; `apply_q0_head_rank1_sharded`, `head_correction.py:779`).
  This is the **only** q-analytic piece today, and it is applied at q=0 only.
- **Exchange is k-block-diagonal (bug B1, `kernel_dataflow_trace.md:368-388`).**
  `S = einsum('kcvN,bcvk->bNk')` keeps k as a batch index; the physical exchange
  is a global `Σ_k`. This design *depends on B1 being fixed* — see Phase 0.
- Reusable ISDF mini-library (`isdf/core.py`, `isdf/__init__.py`): `pair_density`,
  `c_q_from_psi_sm`, `z_q_from_psi_sm`, `factor_c_q`, `solve_zeta`, `fit_one_rchunk`
  — array-in/out on `mesh_xy`, ψ(G) pulled from a host `PsiGStore` via
  `io_callback` (never a jit arg), local IFFT+r-slab via
  `wfn_transforms.to_rchunk_inner` (`core.py:645`). This is exactly the machinery
  a centroid-sampler and any fine ζ-refit would reuse.
- Symmetry: one canonical `common.symmetry_maps.SymMaps` (`bse_io.py:713` for the
  eqp IBZ→full-BZ unfold); `unfold_v_q`, `kqfull_map`, `kvecs_asints` already
  provide q=k−k′ indexing on a grid.

---

## Reference physics + BGW implementation

Full deep-read in `bgw_fine_grid_reference.md`; the load-bearing pieces:

- **dcc/dvv WFN expansion** (R&L Eq 39, `intwfn.f90` + `mtxel_t.f90:179-188`):
  fine state expanded in the closest coarse states,
  `d(im_fi,im_co) = Σ_{G,σ} c_fi(G)·conj(c_co(G+G_umk))`, `G_umk = nint(k_fi−k_co)`.
  Needs a **fine WFN** (`WFN_fi`); overlaps computed in G-space; renormalized
  `d→d/√Σ|d|²` (`intwfn.f90:1013-1038`).
- **Head/wing/body storage** (`mtxel_kernel.f90:528-559`, `bgw_ref §2`): the
  direct kernel is stored with the divergent q-factors **stripped** — head stored
  as `1`, wing as `ε⁻¹·v·q` (O(1)), body as the full `ε⁻¹_GG'·v` — so only the
  *smooth* part is interpolated.
- **Fine interpolation + singular re-add** (`intkernel.f90:841-1201, 1266-1574`):
  `K_fi = Σ dvv_k·conj(dcc_k)·K_co·conj(dvv_k')·dcc_k'`, then multiply by analytic
  `w_eff = ⟨v(q_fi)⟩_mBZ·ε⁻¹₀₀(q_fi)` (head) and `1/q_fi` (wing) at fine
  `q = k−k'` (`intkernel.f90:882-890, 1103-1146`).
- **Analytic q pieces** (`bgw_ref §3.2-3.3`): `epsdiag` tabulates `ε⁻¹₀₀(q+G)`
  beyond BZ1; `minibzaverage_3d_oneoverq2` gives `⟨8π/q²⟩_mBZ` with the analytic
  small-sphere correction; `wcoul0 = ⟨v⟩_mBZ·ε⁻¹₀₀` at q=0.
- TDA is BGW's default (`inread.f90:168`); full-BSE forces `extended_kernel`.
  Finite-Q shifts valence states, exciton COM = −Q (`bgw_ref §4`). CSI
  subsampling replaces the interpolated kernel below `|q|<cutoff` for 1D/2D
  (`bgw_ref §5`).

---

## Proposed design

### Core insight — LORRAX does not need dcc/dvv

BGW interpolates ψ because its wavefunctions live in G-space and the kernel is a
G-space matrix element; the dcc/dvv overlaps carry the fast Bloch-phase variation
so the residual kernel is smooth on the coarse grid.

LORRAX's kernel factorizes through **fixed real-space centroids** `r_μ`. Two
consequences:

1. **Fine pair amplitudes are exact by direct sampling — no coefficient
   expansion.** `M[k_fi,c,v,μ] = Σ_s ψ*_{c,k_fi,s}(r_μ)·ψ_{v,k_fi,s}(r_μ)` needs
   only `ψ_{n,k_fi}(r_μ)`, which is a fine NSCF wavefunction *sampled at the same
   `r_μ`*. Sampling ψ at a real-space point is one spmv (`to_rchunk_inner`); it is
   cheaper and strictly more accurate than the dcc/dvv overlap approximation
   (which neglects the `k_fi−k_co−G_umk` offset, `bgw_ref §1.1`). We adopt direct
   sampling and **drop dcc/dvv entirely** for ψ.
2. **The screened `W_μν(q)` in the fixed centroid basis is a smooth lattice
   function** (the μν basis is q-independent), so it Fourier-interpolates in one
   step via its real-space image `W_μν(R)` — the LORRAX analogue of the
   head/wing/body split, done in the ζ G=0 projection rather than G-space.

dcc/dvv is retained in exactly one optional place (Phase 4): interpolating **QP
energies** from coarse `eqp` to fine k with `|d|²` weights, if a fine NSCF's DFT
energies + a scissor/eqp interpolation is judged insufficient. That single use is
`gw`-side and out of the hot matvec.

### Dataflow

```
COARSE (existing GW pass)                         FINE (new)
 WFN_co ─ ISDF ─► isdf_tensors_co.h5              WFN_fi.h5  (cheap NSCF, no ε)
   V_qco(μν), W_qco(μν,q), G0, vhead/whead          │  sample at centroids_frac
        │                                            ▼
        │  w_head_wings.split_head_wing_body    psi_full_y_fi (nk_fi,nb,ns,n_rmu)
        ▼                                            │  = ψ_fi(r_μ)   [exact M]
   W_body(μν,q_co) ── ifft_R ──► W_body(μν,R)        │
        │   (coarse BvK support, bounded)            │
        ▼                                            ▼
   Fourier-interp to fine q:  W_μν(q_fi)  +  head/wing_at_q(q_fi)   [analytic]
        │                                            │
        └───────────────► FINE BSE matvec ◄──────────┘
              H = D_fi + V(q=0, global-k reduction) − W_fi(convolution over k_fi)
                       (TDA; solvers unchanged)
```

### File-level plan

- **NEW `src/bse/fine_grid.py`** (procedural, bundle in/out; no classes):
  - `sample_psi_at_centroids(wfn_fi, centroids_frac, meta, mesh_xy) -> psi_full_y_fi`
    — reuses `PsiGStore` + `to_rchunk_inner` (`isdf/core.py:645`) to evaluate
    `ψ_fi(r_μ)` at the coarse centroids; ψ(G) via `io_callback`, never a jit arg.
    Emits `psi_full_y_fi` in the same `(nk,nb,ns,n_rmu)` layout `gw_init.py:695`
    writes, so the loader path is unchanged.
  - `w_body_to_R(W_body_q_co, kgrid_co, mesh_xy) -> W_body_R` — one
    `make_flat_k_ifftn` (`fft_helpers.py:409`) over the coarse q-grid.
  - `w_at_fine_q(W_body_R, q_fi_frac, kgrid_co) -> W_body_μν(q_fi)` — band-limited
    resum `Σ_{R∈WS(coarse)} e^{−i q_fi·R} W_body(R)`; keeps `W` stored in R at
    **coarse** support (bounded), never materializes a fine-q W tensor.
  - `build_fine_kernel_bundle(coarse_bundle, psi_full_y_fi, symmaps_fi, opts)`
    — assembles the fine matvec inputs, calls the two divergence designs for the
    per-q head/wing, returns the same dict shape the solvers already consume.
- **REUSE, do not duplicate:** `common.fft_helpers` (R↔q), `SymMaps`
  (fine k-grid + `kqfull_map` for q=k−k′), `gw.head_correction` (extend to a
  per-q head — see below), `isdf.core`/`PsiGStore`/`to_rchunk_inner` (sampler),
  the existing `bse_simple`/`bse_ring_comm` matvec bodies.
- **SINGLE-SOURCE the head injection.** `apply_q0_head_rank1_sharded`
  (`head_correction.py:779`) hardcodes `omega_index`/q=0. Generalize it to accept
  a per-q `(coef, g0)` from `w_head_wings` and inject into `W_q[:,:,qx,qy,qz]` —
  the **same** routine now serves both the coarse q=0 case and every fine q. No
  parallel "apply head at fine q" helper.
- **DELETE / fix (Phase 0):** the k-block-diagonal exchange (B1). After the fix
  the exchange is a **global-k reduction** `S[b,ν]=Σ_{k,c,v}M[k,c,v,ν]X/√Nk`
  (no k index on S), which is *more* scalable on fine grids, not less. Fix
  single-sourced across `bse_simple.py:101-131`, `bse_serial.py:62-64`,
  `bse_ring_comm.py:276-300`.

### Physics-critical formulas

Fine pair amplitude (exact, direct sample):
```
M[k_fi,c,v,μ] = Σ_s ψ*_{c,k_fi,s}(r_μ) · ψ_{v,k_fi,s}(r_μ)
```
Smooth-body Fourier interpolation of W (μν fixed ⇒ well-defined lattice fn):
```
W_body_μν(R)   = (1/Nk_co) Σ_{q_co} e^{+i q_co·R} [ W_μν(q_co) − W^{hw}_μν(q_co) ]
W_μν(q_fi)     = Σ_{R∈WS(co)} e^{−i q_fi·R} W_body_μν(R)  +  W^{hw}_μν(q_fi)
```
- Exact at `q_fi ∈ coarse grid` (recovers `W_μν(q_co)`); smooth between **iff**
  `W_body_μν(R)` decays inside the coarse Born–von-Kármán cell (gate below).
- `W^{hw}` (head+wing, the non-band-limited 1/q² part) is stripped before the
  transform and re-attached analytically at each fine q — **owned by
  `w_head_wings`**.

Direct term on the fine grid — unchanged convolution, larger FFT:
```
U(k_fi) = (1/√Nk_fi) Σ_{k'} W_μν(k_fi−k') T(k')     [FFT over (nkx_fi,nky_fi,nkz_fi)]
```
Exchange (post-B1, Q=0 optical): global k-reduction through the q=0 kernel:
```
S[b,ν]        = (1/√Nk_fi) Σ_{k,c,v} M[k,c,v,ν] X[b,c,v,k]
U[b,μ]        = Σ_ν V_q0[μ,ν] S[b,ν]
(V X)[b,c,v,k]= (1/√Nk_fi) Σ_μ conj(M[k,c,v,μ]) U[b,μ]
```

### Sharding + memory plan (nk_fine ≫ nk_coarse)

The coarse code replicates k on `W`, `T`, `X` (`make_bse_shardings`). At fine k
that is the failure mode: `T[b,μ_loc,ν_loc,ts,k_fi]` scales with `Nk_fi`. Plan:

1. **Never store W at fine q.** Keep `W_body_μν(R)` at **coarse** R-support
   (`≤ Nk_co` cells, tiny) and embed it into the fine BvK lattice as the
   convolution kernel: the fine-k FFT convolution uses a kernel that is nonzero
   only on the coarse-cell footprint. Memory for W stays `∼ μ·ν·Nk_co`.
2. **Shard the fine-k axis.** The direct-term convolution is
   `U_R = W_R·T_R` after a k-FFT; use the existing custom-partitioned
   `make_sharded_ifftn_3d`/`make_sharded_fftn_3d` (`fft_helpers.py:304-357`) to run
   the FFT with `(kx,ky,kz)` sharded on the `(x,y)` mesh — no new mesh axis (per
   the "no new mesh axes / scan-inside-shard_map" rule). `T_R` sharded on k;
   `W_R` (coarse support) broadcast-replicated; the pointwise product is local.
3. **Scan over centroid-ν blocks** when `μ_loc·ν_loc·Nk_fi` still exceeds budget:
   the (μ,ν) FFT pairs are independent, so `lax.scan` over ν-blocks inside the
   shard_map bounds peak at `μ_loc·ν_block·Nk_fi` (the zeta-fit pattern,
   `isdf/core.py:628-708`).
4. **Exchange costs almost nothing on fine k** (post-B1): `S[b,ν]` reduces k
   away, so the exchange never holds a k×k′ object — a reduce-scatter over the
   sharded k axis.
5. Trial `X[b,c,v,k]`: c,v stay small; re-tile so **k** (the big axis) carries
   the mesh for the fine solve. Exact axis assignment (k on `x`, μ on `y` vs a
   flattened k across the whole mesh) is a perf tune, gated on 1-GPU where it is a
   no-op — see Open Questions.

### Restriction to TDA first — yes

TDA only for Phases 0-3. The full-BSE `build_bse_ring_matvec_full` path is broken
independently (B2, `kernel_dataflow_trace.md:389-396`) and finite-Q is unbuilt;
both are out of scope. The fine-grid layer is orthogonal to the TDA/non-TDA
switch — it changes only how `M`, `V_q0`, `W` are produced, not the matvec
structure — so non-TDA lands for free once its own blocker is cleared.

---

## Interactions with the other four designs (shared seams)

- **`coulomb_sr_lr`** — I consume `v_minibz_average(q_fi_frac, kgrid_fine, cell)`
  (the `⟨8π/q²⟩_mBZ` with analytic small-sphere correction, `bgw_ref §3.3`) for
  the fine-q head normalization and the exchange head, and the SR/LR split so the
  **SR** bare-Coulomb folds into the smooth `W_body` I Fourier-interpolate while
  the **LR** 1/q² head is their analytic per-q term. Seam = the fine-q vector list
  I hand them (`SymMaps.kvecs_asints`-derived) and the scalars they return.
- **`w_head_wings`** — owns `split_head_wing_body(W_q_co)` (gives me the smooth
  body pre-transform) and `head_wing_at_q(q_fi) -> (coef, g0[, wing])` (the per-q
  rank-1/wing update I inject via the single-sourced `apply_q0_head_rank1_sharded`)
  and the `ε⁻¹₀₀(q)` tabulation/interpolation (epsdiag analogue). Seam = the
  head/wing/body contract on the ζ G=0 projection.
- **Kernel/matvec (dense-exchange) design** — owns the B1 fix (global-k
  exchange). My layer assumes exchange is a k-reduction; if that design lands
  first, Phase 0 is theirs. Seam = single-sourced `apply_V` across the three
  matvec files.
- **Restart-IO / bundle design** — owns adding `psi_full_y_fi` + fine `kgrid`/
  `SymMaps` to the loader (`bse_io.py:358-536`, which must also absorb the flat-q
  reader fixes B3-B5). Seam = the fine-grid fields in the restart bundle and the
  `nq == nkx·nky·nkz` grid-resolution chain (`kernel_dataflow_trace.md:344-362`).

---

## Gates (1-GPU validation plan, BGW anchors)

All gates run on MoS2 3×3 or Si 4×4×4 fixtures on **one** GPU (no 16-GPU gating).

1. **Identity gate (must be exact):** set fine grid = coarse grid. Fourier
   interpolation of `W_body(R)` reproduces `W_μν(q_co)` to ULP; the fine matvec
   must equal the current single-grid matvec bit-for-bit. Catches any
   normalization/index error with zero physics ambiguity.
2. **Centroid-sampling gate:** run `sample_psi_at_centroids` on the *coarse* WFN
   and compare to the restart's `psi_full_y` — identical up to per-band gauge
   phase (compare `M[k,c,v,μ]`, which is gauge-invariant).
3. **W-body decay gate (prerequisite check):** verify `|W_body_μν(R)|` is
   negligible at the coarse BvK cell boundary. If it is not, Fourier interpolation
   is invalid → flag to `w_head_wings` (the stripped head/wing was incomplete).
4. **Convergence gate:** Si coarse 4×4×4 W + fine 8×8×8 ψ. Anchor lowest excitons
   against (a) a direct 8×8×8 single-grid LORRAX run (the reference truth) and
   (b) BGW `absorption.x` with `WFN_co` 4×4×4 / `WFN_fi` 8×8×8. Target: lowest-20
   within the ISDF floor (~3 meV, per `context_docs.md:135-140`).
5. **Head-at-q gate:** with `w_head_wings` stubbed to q=0-only, the fine kernel
   must equal "coarse body + coarse head"; turning on per-q head must move the
   lowest bright exciton toward the BGW value monotonically.

---

## Open questions for Jack (physics/priority only)

1. **Target dimensionality.** 3D (Si) first, or is 2D (MoS2) the real target? 2D
   screening varies too fast at small q for plain Fourier interpolation of
   `W_body(R)` (the CSI/`subsample_line` regime, `bgw_ref §5`). If 2D is required
   soon, the R-space body will not be band-limited and we need clustered
   subsampling — a much larger effort. Which do I design the body interpolation
   around?
2. **Fine energy source.** Acceptable to require a fine **NSCF** (fine DFT
   energies + coarse scissor/eqp interpolation), or must QP energies be
   interpolated from coarse `eqp` via the `|d|²` scheme (the one place dcc/dvv
   returns)? The former is simpler and I recommend it.
3. **Priority vs the exchange bug.** The B1 dense-exchange fix is a prerequisite
   and is independently a correctness bug on the coarse grid. Should it land as
   its own PR ahead of any fine-grid work?
4. **Finite-Q excitons** (`Qflag`, exciton COM momentum): in scope for this
   layer, or explicitly Phase-4/deferred? It changes the valence-band sampling
   grid and the exchange `v(G−Q)`.
5. **Extra DFT cost tolerance.** A fine NSCF is cheap relative to GW but non-zero.
   Is one fine NSCF per system acceptable, or is avoiding any new mean-field pass
   a hard constraint (which would force dcc/dvv-from-a-cheaper-source)?

---

## LOC estimate + suggested phasing

- **Phase 0 — dense-exchange (B1) fix** (prereq; may belong to the matvec design):
  ~40-60 LOC net, single-sourced across `bse_simple`/`bse_serial`/`bse_ring_comm`.
- **Phase 1 — fine ψ ingestion:** `fine_grid.sample_psi_at_centroids` + loader
  plumbing for `psi_full_y_fi` and fine `SymMaps`. ~180 LOC new (mostly reuse of
  `PsiGStore`/`to_rchunk_inner`), ~40 LOC loader. Gates 1-2.
- **Phase 2 — W-body Fourier interpolation + per-q head/wing re-attach:**
  `w_body_to_R`, `w_at_fine_q`, generalized `apply_q0_head_rank1_sharded`.
  Depends on `w_head_wings` + `coulomb_sr_lr`. ~180 LOC (−~30 LOC duplication
  removed by single-sourcing the head). Gates 3, 5.
- **Phase 3 — fine-k sharding + convolution:** sharded-FFT wiring + ν-block scan;
  matvec re-tile for k on the mesh. ~150 LOC + edits to the matvec inputs. Gate 4.
- **Phase 4 (deferred):** eqp `|d|²` interpolation, finite-Q, 2D CSI subsampling.

Net new code ~510-570 LOC across Phases 1-3, minus ~30-60 LOC of duplication
removed (single-sourced head + exchange). No new classes, no parallel matvec
path, one new file (`fine_grid.py`), one generalized existing routine.
