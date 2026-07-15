# Solver Program — FEAST maturation + clustered eigenvalues + consolidation

Design agent deliverable, BSE refactor-map, 2026-07-15. Checkout `sources/lorrax_D`
@ 218aeb8 (BSE tree byte-identical to the named base e18d0e5 for every file here:
`git diff e18d0e5..HEAD -- src/bse src/solvers` is empty).

Scope: the BSE eigensolver **program** — the solver zoo, FEAST as the production
interior-eigenvalue engine, the clustered-eigenvalue failure mode, one procedural
solver contract, and what the fine-grid future demands. Not in scope (named seams
to sibling designs): the ISDF matvec kernels/sharding, the restart-bundle loader,
the absorption/spectra stage, coarse→fine interpolation, and the pseudopole-W_c
compression thread (a Σ/χ producer, not a BSE eigensolver).

## Current state in LORRAX (file:line grounded)

**The zoo, as it exists.** Eleven solver-ish modules over ~4 kLOC:

| Module | Role today | Status |
|---|---|---|
| `solvers/lanczos.py` | 6 variants (simple, jit, block, block-jit, block-jit-converged, +`_build_block_tridiag`) | 3 T-assembly impls; `block_lanczos_eig` has a **transposed-β bug** (lanczos.py:96,117-120), all 3 jit variants **overwrite the final Krylov slot** (lanczos.py:281,419,516) |
| `bse/bse_lanczos.py` | `solve_bse`/`solve_bse_sharded` — BSE wiring + Davidson dispatch | `solve_bse` re-jits matvec per call (bse_lanczos.py:54); 1-device path drops CLI knobs (bse_jax.py:294-328) |
| `solvers/davidson.py` | block Davidson, shared with DFT NSCF (`psp/run_nscf.py:46`) | works; Ritz-rotation is fine (Hermitian) |
| `bse/bse_davidson_helpers.py` | BSE init subspace + ΔE precond | works |
| `bse/davidson_absorption.py` | exact-low-eigpair CLI | works; dipole/eigvec pad misalignment (davidson_absorption.py:200) |
| `bse/bse_feast.py` (1320) | **default** BSE solver (bse_jax.py:539-543): bounds→windows→contour filter→GMRES→Rayleigh-Ritz | works on validated Si; caches keyed without matvec identity (bse_feast.py:128,229); mutates caller dict (`data["W_R"]`, :505); W_R triple-built; no residual locking |
| `bse/bse_kpm.py` + `solvers/chebyshev.py` + `solvers/dos.py` | KPM DOS → equal-mass windows | **broken at HEAD**: phantom `v_couples_k` kwarg → `TypeError` on every KPM entry (bse_kpm.py:120,128,137) |
| `bse/absorption_haydock.py` | continued-fraction ε₂ (block-Lanczos, no eigvecs) | works; hard-requires ≥2 devices (absorption_haydock.py:158-161) |
| `solvers/pseudobands*.py` | CJ step-filter + Ritz band compression (DFT only) | 2 parallel versions; Ritz-rotation transpose bug (BUG-1, 3 sites); **no BSE caller** |
| `bse/bse_pseudopoles.py` + eval/sweep | W_c(ω) pseudopole compression | **un-importable** (lost wiring, pseudopoles.md); a Σ/χ producer, not an eigensolver |
| `bse/feast_{sweep,zolo_sweep,ellipse_mixed_sweep}.py`, `bse_feast_dense_debug.py` | FEAST parameter sweeps | dev harnesses; 3 have zero callers, 3 broken at HEAD (feast_experiments.md) |
| `solvers/minres.py`, `projectors.py`, `sternheimer_*` | Paige–Saunders MINRES + Sternheimer primitives | **not on the BSE path** (aux_solvers.md); minres has zero production callers |

**FEAST as built** (feast_core.md, run_feast_ritz bse_feast.py:463-684, read in full):
- Contour: ellipse-trapezoid (CLI default) or Zolotarev step (API default) rational
  quadrature of `P = (1/2πi)∮(zI−H)⁻¹dz`. Zolotarev needs `[λ_min,λ_max]` (:583).
- Subspace: `n_ritz` — a **fixed CLI flag, default 4** (bse_feast.py:1098), with **no
  relation to how many eigenvalues sit in the window**.
- Linear solve: right-preconditioned FGMRES with Jacobi diagonal `1/(z−diag_h)`
  (bse_feast.py:120-200); accuracy notes prove 2-7 iters, tol irrelevant.
- Extraction: overlap-whitened Rayleigh-Ritz, residual
  `Re[c†(G−2λH+λ²S)c]/Re[c†Sc]` (bse_feast.py:349-422).
- Iteration: `feast_iter` subspace sweeps, restart from in-window Ritz vectors padded
  with fresh randoms (bse_feast.py:648-678). **No locking, no deflation, no subspace
  growth.**

**The clustered-eigenvalue failure.** STATUS.md:75: "~100 states/0.5 eV at the band
edge". FEAST subspace iteration converges the window only if `n_ritz > N_window`.
With `N_window≈100` and `n_ritz=4` the filter's `N_window`-dimensional in-window
subspace is under-resolved; overlap-whitening (`s_cutoff`, bse_feast.py:40) silently
collapses the near-degenerate directions, and the Ritz values that survive are
variational-biased-high (feast_accuracy_notes.md Obs 2: +3.6→+51 meV as f(E)→0 at the
boundary). There is **no count-in-window estimate anywhere** to size the subspace,
and KPM (the one tool that could provide it) is broken at HEAD.

**Ghost eigenvalues** (feast_accuracy_notes.md Obs 3): 60-step Lanczos on N=144
without full reorth produces spurious eigenvalues from loss of orthogonality —
Lanczos is trusted for `E_max` bounds only, hence `--n-reorth -1` default.

## Reference physics + BGW implementation

TDA exciton Hamiltonian (Ry; bse_serial.py:80, verified against Henneke 2020 Eq 4-5/4-6):
```
H X = D X + V X − W X
D[b,c,v,k] = (ε_c[k,c] − ε_v[k,v]) X[b,c,v,k]
V[b,c,v,k] = (1/Nk) Σ_μν M*[k,c,v,μ] V_q0[μ,ν] Σ_c'v' M[k,c',v',ν] X[b,c',v',k]
W[b,c,v,k] = (1/Nk) Σ_k' [ψ*_c(k)ψ_c(k')]_μ W_μν(k−k') [ψ_v(k)ψ*_v(k')]_ν X[b,c',v',k']
M[k,c,v,μ] = Σ_s ψ*_c[k,c,s,μ] ψ_v[k,v,s,μ]     (spin-traced)
```
Full (non-TDA) Casida `S = [[A,B],[−B†,−A†]]`, A=D+V−W, B=V−W.

**BGW reference.** BGW does not iterate a contour: `absorption.x` (`BSE/diag.f90`)
builds the dense `hbse_a` (+`hbse_b` non-TDA) and calls ScaLAPACK `p?heevx` (TDA) or
the structure-preserving `BSE_NTDA_SOLVER_SSEIG` (Shao et al. LAA 488, 2016;
`inread.f90:170-172,392-400`); an optional PRIMME iterative TDA path exists
(`inread.f90:565`). `absp_lanczos.f90` supplies the Haydock arm for both TDA and full
BSE. So the interior-eigenvalue problem is BGW's *dense* regime — LORRAX's FEAST/KPM
path is the matrix-free replacement that BGW itself only approximates with PRIMME.
The exact N=144 (4v4c, 3×3×1) reference eigenvalues in feast_accuracy_notes.md
(1.851722…2.067945 eV, three degenerate pairs) are the anchor for any clustered-solver
gate. Validated Si 4×4×4 8v8c: LORRAX FEAST/Lanczos within ~3 meV of BGW (STATUS.md:67),
saturated at the ISDF floor.

## Proposed design

### Solver triage — earn a place vs archive

**Production solvers (keep, single-sourced):**
- **FEAST** — interior windows of the *sparse* low-lying exciton manifold. Matured (below).
- **Lanczos (block, jit, convergence-driven)** — `E_max` spectral bounds (all
  window machinery depends on it) + a Krylov lowest-`n_eig` fallback. **One** routine.
- **Davidson** — exact lowest-`n_eig` bound excitons; shared with DFT NSCF, keep as-is.
- **KPM DOS** — window placement + subspace sizing (fix the phantom kwarg first).
- **Haydock** — spectrum-without-eigenvectors; the *primary* fine-grid path (below).
  Lives in the spectra design but reuses the generic block-Lanczos recurrence.
- **Chebyshev–Jackson polynomial filter** (new BSE consumer of `solvers/pseudobands`
  machinery) — the matrix-free alternative for dense clusters and fine grids.

**Archive to E-tier** (move to `reports/bse_refactor_map_2026-07-15/archive/experiments/`,
out of `src/`): `feast_sweep.py`, `feast_zolo_sweep.py`, `feast_ellipse_mixed_sweep.py`
— their purpose (n_quad/γ/ρ sweep vs the N=144 dense reference) is fully absorbed into
`bse_feast.main`'s `--quadrature`/`--n-quad`-schedule and is done work (context_docs.md).

**Delete outright:** `solvers/lanczos.py:block_lanczos_eig` (transposed-β bug, no
caller) and `simple_lanczos_eig` (fold its M+1-slot bookkeeping into the jit variant);
`bse/bse_preconditioner.py` minus `energy_diff_cv_k`; the dead module-level
`apply_bse_hamiltonian/apply_D/apply_V/apply_W` + `symmetrize_W_q` in bse_jax/bse_serial;
`bse/bse_ring_comm.apply_bse_hamiltonian_ring`; `bse_feast_dense_debug.py`'s inline
quadrature copy (→ import `solvers.quadrature`, then repurpose as a pytest fixture).

**Not a solver** (hand to sibling designs): `bse_pseudopoles.*` → pseudopole-W_c
program; `solvers/minres.py`+`projectors.py`+`sternheimer_*` → Sternheimer program
(delete minres if that program adopts CG per run_sternheimer.py:1174).

### Dataflow

```
                 KPM DOS ρ(E)  ──────────────► equal-mass window partition
 restart bundle      │ (solvers.chebyshev)          │  + per-window count N̂_j = ∫_j ρ·dim
 (I/O design) ──► data ──► Lanczos E_max bounds ──► [λ_min, λ_max]
                     │                                │
                     ▼                                ▼
             matvec closure  ───────────────► SOLVER DISPATCH (bse_solve.solve_bse)
             (matvec design)                   ├─ sparse window  → FEAST (Zolotarev,
                                               │     m0=⌈1.5 N̂⌉, locking+deflation)
                                               ├─ dense  window  → CJ poly-filter + Ritz
                                               ├─ lowest n_eig    → Davidson / block-Lanczos
                                               └─ spectrum only   → Haydock / KPM
                                                         │
                                    evals (Ry, sharded), evecs? (sh.X) or None, info
```

### File-level plan

- **`src/solvers/lanczos.py`** — collapse 6→1 public routine `lanczos_lowest(matvec,
  shape, opts)` (scalar+block via `block_size`, optional `lax.while_loop` convergence,
  reorth window with `-1`=full). Allocate `M+1` Krylov slots (kills the slot-overwrite
  bug at all sites); β=R directly from QR (kills the transpose bug). Keep
  `_build_block_tridiag`. Add `lanczos_bounds(matvec, shape) → (E_min_est, E_max)`
  (the adaptive-`E_max` recurrence currently duplicated in bse_feast.py:695-846) —
  **one** bounds impl, consumed by FEAST, KPM, and `solvers/dos.py`.
- **`src/solvers/feast.py`** (NEW; move the physics-free half of bse_feast.py here) —
  `feast_windows(matvec, sh, diag_precond, windows, bounds, opts) → RitzResult per
  window`. Owns: `quadrature.py` (ellipse+Zolotarev, single-sourced —
  bse_feast_dense_debug's copy deleted), the FGMRES runner, Rayleigh-Ritz, and the
  new subspace-iteration-with-locking loop. Caches keyed by `(id(matvec), id(data),
  n_quad, n_ritz, tol, dtype)` (fixes the silent RPA-reuses-BSE hazard, bse_feast.py:229).
- **`src/solvers/cheb_filter.py`** (NEW; ~150 LOC) — matrix-free window projector:
  reuse `solvers/pseudobands._telescoping_filter` + the CJ step coefficients (currently
  triplicated — pseudobands.py:89, v2:106, v2:294 → single-sourced here). `Y = P_[a,b] Ω`
  as a difference of two Chebyshev-Jackson cumulative step filters, then Rayleigh-Ritz.
  No linear solves. Shared by BSE dense-cluster windows and DFT pseudobands.
- **`src/bse/bse_solve.py`** (NEW; ~300 LOC) — the single BSE eigensolver façade,
  replacing `bse_lanczos.solve_bse{,_sharded}`, `bse_feast.main`'s glue, and
  `bse_jax._preview_lanczos`. `solve_bse(data, mesh, opts)`: builds the matvec (matvec
  design), the sharded diagonal preconditioner (`build_preconditioner_diagonal_sharded`,
  kept), calls the generic solver, applies no BGW-compat conversion (writer boundary
  owns Ry→eV + valence flip). Absorbs the Davidson dispatch. `bse_jax.py` becomes a
  thin argparse→opts adapter.
- **`bse/bse_kpm.py`** — delete the phantom `v_couples_k` kwarg (bse_kpm.py:120,128,137);
  expose `count_per_window(ρ, windows, dim)` for FEAST subspace sizing.
- **Delete duplicates** (no-redundancy rule): `_create_mesh_xy` ×6 → one in
  `bse_ring_comm`; `_to_host` ×4 → `file_io._slab_io_allgather._to_host`; `WindowSpec`
  ×3 → one; `compute_pair_amplitude` ×3 and `energy_diff_cv_k` → matvec module; the
  eqp-override block ×3 → one loader helper (I/O design).

### FEAST maturation — the physics-critical pieces

**(1) Contour = Zolotarev, default.** The sweep evidence is decisive: error is
dominated by filter response at the window boundary, not GMRES (feast_accuracy_notes.md
Obs 1-2). The Zolotarev step filter is the equiripple-optimal rational approximation of
the indicator, so for equal `n_quad` its `f(E)` stays ≈1 far closer to the edge than the
ellipse. Poles (feast_core.md, bse_feast.py:858-913):
```
h_step(λ) = ½ + Σ_j 2 Re[w_j/(z_j−λ)],  z_j = edge + iρ√d_j,  w_j = −β_j ρ/4,
d_j = ε² sn²/cn² at u=(2j−1)K'(1−ε²)/2n,  ε=1/G,  G=max(|λ_min−edge|,λ_max−edge)/ρ
indicator = h_step(a) − h_step(b),  ρ = ρ_scale·(b−a)/2
```
Bounds `[λ_min,λ_max]` already come from `lanczos_bounds`. Unify the CLI default
(currently `ellipse`) with the API default (`zolotarev`) on Zolotarev; keep ellipse as
a flag (needs no bounds → useful when bounds estimation is skipped).

**(2) Subspace sizing from the DOS.** Replace the fixed `n_ritz` with
`m0 = ⌈β·N̂_j⌉ + slack`, β≈1.5, where `N̂_j = ∫_j ρ(E) dE · dim` is the KPM
per-window count (bse_kpm already computes ρ; add `count_per_window`). Validate on the
fly with the FEAST-native **stochastic trace of the filter**: after the first contour
apply, `N_window ≈ (1/R) Σ_r Re⟨x_r|P|x_r⟩` for the random probe batch — `P` is exactly
what the runner already accumulates. If `tr(P) > m0`, the window is under-sized → split
(below). This is the single change that makes clustered windows solvable: the code today
literally cannot size its subspace to the occupancy.

**(3) Linear solver inside the contour = keep FGMRES + Jacobi.** MINRES is *not* adopted:
the shifted systems `(z/s − H)` are complex-non-Hermitian (z off the real axis), so
right-preconditioned GMRES is correct and MINRES (Hermitian-only) does not apply — and
the accuracy notes already show the diagonal preconditioner converges GMRES in 2-7 iters
with tol-insensitivity, so there is nothing for a fancier inner solver to buy. The
per-node shift is the preconditioner `1/(z−diag_h)` itself. Keep `gmres_max_iter=10`,
`gmres_tol=1e-2`, fp32 fast path.

**(4) The clustered fix — what actually works, in priority order:**
1. **DOS-sized subspace** (above) — necessary; without it nothing else matters.
2. **Window sub-splitting** — cap any window at `m0_max` (a memory-derived ceiling,
   ~64 on 1-GPU MoS2/Si). A 100-state/0.5 eV region → ~5 equal-mass sub-windows of ~20,
   each with `m0≈30`. Smaller Gram matrices → the overlap-whitening no longer collapses
   near-degenerate directions, and each contour is cheaper and better separated. Reuse
   `partition_windows` (chebyshev.py:229) recursively on the KPM ρ.
3. **Residual-based locking + deflation** — the missing ingredient in the current
   subspace iteration (bse_feast.py:648-678 keeps in-window vectors but never locks
   them). After each Rayleigh-Ritz, lock pairs with `rel_res_i < tol` (the residual is
   already computed, bse_feast.py:412), remove them from the active subspace, and
   **refill the freed slots with fresh random vectors** so the next contour finds new
   states. This is standard FEAST-with-locking and turns O(iterations) into convergence
   for clusters that a single fixed subspace can't span.
4. **Polynomial-filter alternative for the densest windows** — where FEAST's contour
   cost (`m0 × n_quad × gmres_iters` matvecs) becomes the bottleneck, route the window to
   `cheb_filter`: a Chebyshev-Jackson projector onto `[a,b]` needs only matvecs (no
   linear solves), doesn't care about clustering, and on GPU — where the FFT-heavy W
   matvec dominates — often beats the contour. This is the natural home for the dense
   band-edge manifold and the fine-grid regime. FEAST stays the tool for genuinely
   *interior* sparse windows where the contour's exponential filter roll-off wins.

### Common solver interface contract (procedural, no class hierarchy)

Every solver is a free function of a **matvec closure** + **bundles**, returning plain
arrays. No solver base class, no config dataclass mirror.
```
def <solver>(matvec, shape, sh, *, opts) -> (evals, evecs_or_None, info)
    # matvec:  X(sh.X) -> HX(sh.X)         — the ONLY operator coupling
    # shape:   (n_cond_pad, n_val_pad, nk) — dim-agnostic; fine-grid drops in unchanged
    # sh:      make_bse_shardings(mesh)     — the ONE sharding vocabulary
    # opts:    plain dict (n_eig, tol, block_size, quadrature, m0, …)
    # evecs:   sharded sh.X, or None for spectrum-only paths
    # info:    dict (n_iter, n_matvecs, residuals, in_window_counts)
```
`data` (ψ/ε/W/V bundle) is captured in the `matvec` closure, never passed as separate
jit args (io_callback/host-cache rules live in the matvec+I/O designs). The dispatcher
`bse_solve.solve_bse(data, mesh, opts)` is the only place that unpacks `data`.

### Sharding + memory plan

- **Sharded subspace buffers.** Today the Lanczos `Q_all` and the FEAST Krylov `V/Z`
  and Gram products carry *no* sharding constraint (lanczos_family.md, feast_core.md) →
  XLA replicates them per device, violating the zero-replicated-intermediates principle.
  Pin every `m0`/Krylov buffer to `sh.X = P(None,"x","y",None)`; keep only the small
  `(m,m)` Gram/tridiagonal on host (device_get). A `m0=64`, Si-8×8-64 vector is
  `64·8·8·64·16 B ≈ 4 MB` sharded — trivial; the win is not OOM-ing at fine grid.
- **W_R once.** `run_feast_ritz` builds W_R up to 3× via plain `jnp.fft.ifftn` on the
  sharded W_q (an all-gather, feast_core.md; ~16 GB/device at μ=4000,nk=64). Build it
  once through `common.fft_helpers.make_sharded_ifftn_3d` (the helper the ring matvec
  already uses), pass it in — do not mutate the caller's `data` dict.
- **fp32 GMRES** keeps a second complex64 copy of the 8 heavy arrays; keep it optional,
  but drop the redundant second W_R/fp32 rebuild inside `estimate_spectral_bounds`.

## Interactions with the other four designs (shared seams)

- **Matvec/kernel design** — owns `build_bse_ring_matvec`/`bse_simple`/`make_bse_shardings`
  and `energy_diff_cv_k`/`compute_pair_amplitude`. Seam: the **matvec closure contract**
  (signature, W_R-already-IFFT'd, TDA vs full). The **zero-padded-band spurious-eigenpair
  bug** (B6/pad-mask: padded ψ=0 ⇒ exact eigenvalue ε_c or −ε_v inside the window;
  lanczos_family.md #3, davidson_family.md, feast_core.md #2) must be fixed at the
  matvec/loader (mask ΔE at padded slots or pad ε with a large sentinel) — the solver
  must simply never surface them; agree the mask lives upstream.
- **I/O / restart design** — owns the `data` bundle, eqp unfold, head injection, fp32
  casting. The loader breakage (B3/B4/B5, kernel_dataflow_trace.md) blocks *every* solver
  on fresh restarts; solvers can't be gated until it lands. Seam: the `data` dict keys +
  `n_cond_pad`/`n_val_pad` contract, and the single eqp-override helper.
- **Spectra / absorption design** — Haydock **is** a block-Lanczos solver; it must consume
  the same `solvers/lanczos` three-term recurrence rather than its own copy
  (absorption_haydock.py:54-94). Seam: the generic recurrence + the `matvec`/`sh` contract;
  spectrum-only solvers return `evecs=None`.
- **Fine-grid / coarse→fine design** — the matrix-free `matvec` closure is exactly what
  makes the fine-grid operator drop into these solvers unchanged; the solver interface
  must stay dim-agnostic (no `nk`-baked reshapes outside the matvec). Seam: `shape` +
  matvec closure; the io_callback host residency of the fine-grid ψ/dcc caches is the
  fine-grid design's to own, invisible to the solver.
- **KPM/DOS** is shared infra (`solvers/chebyshev`, `solvers/dos`) across this design's
  windowing and the pseudobands/pseudopole threads — single-source the CJ step
  coefficients and the Chebyshev recurrence.

## Gates (1-GPU validation, BGW anchors)

All gates run on 1 GPU with MoS2/Si-scale fixtures (no 16-GPU gating):
1. **Generic-solver unit tests** (synthetic Hermitian matrix, CPU/1-GPU, no restart):
   `lanczos_lowest` recovers a random dense operator's lowest-`n_eig` to 1e-10; regression
   pins for the (now-fixed) block-β transpose and final-slot overwrite; `cheb_filter`
   projects a known window.
2. **FEAST vs dense diag on N=144** (4v4c, 3×3×1) — the feast_accuracy_notes.md exact
   eigenvalues (1.851722…). Assert Zolotarev in-window error < ellipse at equal `n_quad`.
3. **Clustered-cluster gate** — synthetic operator with 100 eigenvalues packed in 0.5 eV
   plus a sparse tail; assert DOS-sized subspace + locking recovers the *count* (via
   `tr(P)`) and all 100 to window-width tolerance, where fixed `n_ritz=4` fails.
4. **Pad-mask gate** — `--n-cond 7 --n-val 5` on a 2×2 mesh (padding fires); assert no
   spurious ε_c/−ε_v eigenvalues in the returned spectrum (upstream mask honored).
5. **BGW anchor** — Si 4×4×4 8v8c: lowest-20 eigenvalues within 3 meV of BGW's `eqp.dat`
   run (STATUS.md:67), Σ|d|²=2314.177 machine-match preserved through the refactor.
6. **KPM smoke** — the fixed `run_kpm_dos` returns μ_0≈1 and monotone equal-mass windows.

## Open questions for Jack (physics/priority only)

1. **Do the ~100 states/0.5 eV band-edge states need converged eigenvectors, or is the
   Haydock/KPM spectrum enough there?** If eigvecs are only needed for the lowest bound
   excitons, we cap FEAST at the sparse low manifold and route the dense region to
   Haydock — far cheaper than maturing FEAST clustering.
2. **Is full non-TDA (coupling-block) a near-term target?** The B-block matvec is broken
   (B2, kernel_dataflow_trace.md); if non-TDA is deferred, FEAST/KPM lose their non-TDA
   contour paths and we make TDA the only supported solver surface.
3. **Fine-grid timeline** — must the solver run at `nk_fine × nc × nv` in the next
   milestone, or is coarse-grid the target? Sets the urgency of the matrix-free
   Haydock/KPM-primary path vs eigvec-based FEAST/Davidson.
4. **May we change the default contour to Zolotarev?** It's a behavior change vs the
   validated ellipse runs (though the API already defaults to it).
5. **Is sub-3-meV eigenvalue accuracy needed?** The 3 meV wall is the ISDF-compression
   floor, fixable only upstream (symmetry-adapted ISDF / more centroids), not in the solver.

## LOC estimate + suggested phasing

Net **−800 to −1000 LOC**. Deletions: 3 FEAST sweeps (~1287) + `bse_preconditioner`
(~130) + dead Lanczos variants (~250) + dead bse_jax matvec (~100) + dedup
(_create_mesh_xy/_to_host/WindowSpec/eqp-block, ~200) ≈ **−1970**. New: `solvers/feast.py`
is mostly *moved* (net ~+150 for locking/sizing), `bse_solve.py` ~+300 (replaces
~400 of glue), `cheb_filter.py` ~+150, `lanczos_bounds`/`count_per_window` ~+80 ≈ **+680**.

- **P1 — consolidate + unblock.** Single-source `solvers/lanczos` (fix transpose +
  slot-overwrite), delete dead, dedup, fix `bse_kpm` phantom kwarg, agree the pad-mask
  seam with the matvec design. Gate 1 + 6. (Depends on I/O design landing B3/B4/B5.)
- **P2 — FEAST maturation.** Move to `solvers/feast.py`; Zolotarev default; `lanczos_bounds`
  single-source; cache keys with matvec identity; W_R-once. Gate 2 + 5.
- **P3 — clustered eigenvalues.** DOS-sized subspace + stochastic `tr(P)` count, window
  sub-splitting, residual locking+deflation, `cheb_filter` alternative. Gate 3.
- **P4 — fine-grid readiness.** Sharded subspace buffers, Haydock reuse of the generic
  recurrence, spectrum-only `evecs=None` path, matrix-free audit. Gate 4 + fine-grid seam.
