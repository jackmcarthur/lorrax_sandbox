# Bispinor V_qmunu — implementation plan on top of unified V_q tile (2026-05-08)

State after `agent/phdf5_padded` commit `723e1df` (full-kernel AOT chooser
on lorrax_A's main).  Reference for the bispinor pieces is
`origin/agent-B/refactor-compute-vcoul` (commit `9397e35`).

The unified `compute_V_q_tile` already exposes `same_zeta`, externalised
`v_per_G_fn`/`phase_fn`, and applies `v(K)` ONE-SIDED on the L slab
(line 17 of `v_q_tile.py`).  That is exactly the surface the bispinor
orchestrator needs.  Below is what's already in place, what's missing,
and the cleanest top-level shape.

## 1. Lorentz tensor sectorization (recap)

The bispinor pair density carries a 4-vector index from the Pauli
decomposition of ψ†_iσ ψ_jσ′:

    n_iσ,jσ′(r) = Σ_{μ_L=0..3} ζ_{μ_L,a}(r) ⟨σ| τ^{μ_L} |σ′⟩ C^{a}_{ij}
    τ^0 = I,  τ^{1,2,3} = (σ_x, σ_y, σ_z)

The Coulomb kernel in Lorentz gauge couples the four channels by

    V^{μ_L,ν_L}_q(μ,ν) = Σ_K  ζ̄_{μ_L,μ}(K) · t^{μ_L,ν_L}(K) · v(K) · ζ_{ν_L,ν}(K)
    t^{0,0}      = 1
    t^{i,j}      = δ_ij − K̂_i K̂_j        (transverse projector)
    t^{0,i}=t^{i,0} = 0                   (Coulomb-gauge cross term)

So out of the 4×4 = 16 (μ_L, ν_L) blocks:

| count | sector | status                                  |
|------:|:-------|:----------------------------------------|
|     6 | (0,i)/(i,0)                  | identically zero (gauge) — never compute |
|     1 | (0,0) = CC                   | scalar-Coulomb tile                      |
|     3 | (i,i) = TT diagonal          | each is V^{i,i} with weight `1 − K̂_i²`  |
|     6 | (i≠j) = TT off-diagonal      | 3 unique pairs (i<j), 3 by Hermitian tx  |
| **7 unique kernel calls** + **3 Hermitian transposes** + **6 zeros**      |

Hermitian relation (per B's `_HERM_TILES`):
`V^{j,i}(q; μ, ν) = V^{i,j}(q; ν, μ)*`  (centroid axes; q held fixed).

## 2. Cleanest top-level shape

The transverse channels share `n_rmu_T` (single current-density
centroid file → one count for μ_L ∈ {1,2,3}); the charge channel has
`n_rmu_C`.  So the natural container is:

```python
class V_q_bispinor(NamedTuple):
    CC: jnp.ndarray            # (nq, n_rmu_C, n_rmu_C),       P(None,'x','y')
    TT: jnp.ndarray            # (3, 3, nq, n_rmu_T, n_rmu_T), P(None,None,None,'x','y')
    g0_C: jnp.ndarray | None   # (n_rmu_C,) charge q=0 head; transverse head ≡ 0 by gauge
```

Why this beats B's `dict[(μ_L,ν_L)] → Array`:
- Static shape — JIT-friendly downstream consumers (Σ_X^B, Σ_H^B); a
  dict requires per-key `.get` plumbing and breaks `vmap` over the 9
  TT entries.
- Both leading axes of `TT` are size 3, contiguous, and small → easy
  to broadcast against the 3×3 σ_iσ_j weights in the bispinor σ_X
  consumer.
- The 3 Hermitian-transpose tiles are *materialised* (not lazy) so
  consumers don't need branch logic for "did this orchestrator fill
  (i,j) or (j,i)?".  Memory cost: TT is 9× larger than the 6-unique
  packed form, but TT is per-rank ~`9 · μ_T² · nq · 16 / p_prod` —
  for CrI3 16-GPU μ_T=1808 that's 9 · 1808² · 36 · 16 / 16 = 10.6 GB
  unsharded total, ~2.6 GB / rank.  Acceptable.
- The (0,i) / (i,0) zero blocks are not stored at all (the gauge
  guarantee is encoded in the type, not in zero-valued tiles).

The g0 head is only populated for CC because the transverse projector
kills G=0 (`1 - K̂_i² = 0` on the q→0 axial limit when K̂ aligns with i;
more generally the head is gauge-canceled).

## 3. What's already in place on A's main

| piece                                          | location                                    | status |
|------------------------------------------------|---------------------------------------------|--------|
| Small-spinor lift ψ_S = (α/2)(σ·(k+G))ψ_L      | `common/bispinor_init.py`                   | ✅ on main |
| PHDF5 reader bispinor=True path                | `file_io/_phdf5_wfn_reader.py` (e45beea)    | ✅ on main |
| Unified V_q kernel with `same_zeta` static     | `gw/v_q_tile.py::_make_V_q_tile_kernel`     | ✅ on main |
| `v_per_G_fn` / `phase_fn` external callables   | `gw/v_q_tile.py::compute_V_q_tile`          | ✅ on main |
| One-sided v(K) (no √v) — handles non-PSD       | `v_q_tile.py:18, 702`                       | ✅ on main |
| `n_rmu_L`/`n_rmu_R` distinct sizes supported   | `v_q_tile.py:120, 570`                      | ✅ on main |
| AOT chooser handles same/distinct ζ FFT cost   | `_choose_v_q_chunks` (`fft_coef · 2.0` for  `not same_zeta`) | ✅ on main |
| `compute_V_q` ready for `V_q_bispinor` return  | `gw_init.py:625` (comment in code)          | ⏳ comment only — switch + struct still TBD |

So the V_q kernel side is essentially done.  Bispinor is "wire 7 calls
to compute_V_q_tile, fill 3 by Hermitian-transpose, return a struct."

## 4. What still needs to be ported from `origin/agent-B/refactor-compute-vcoul`

### 4.a — Centroid weighting (ζ_T fitting prerequisite)

| file                                      | LOC | what                                                        |
|-------------------------------------------|----:|-------------------------------------------------------------|
| `centroid/current_density.py`             | 178 | Gordon-decomposed Pauli current weight (`Im[ψ†∇ψ] + ½∇×s`)  |
| `centroid/kmeans_cli.py` `--density-mode` |  ~10| switch between charge `\|ψ\|²` and current `Σ_n\|j_n\|²`     |
| `centroid/centroid_io.py` density tag     |   ~5| header tag in `centroids.h5` so callers know which weight   |

Without these, ζ_T (μ_L=1,2,3) can't be fitted.  CrI3 CCT for μ_L=i is
indefinite (per the user's prior memory), so we *also* need:

### 4.b — ISDF fitting changes (4-channel ζ-fit)

| file                  | what                                                                         |
|-----------------------|------------------------------------------------------------------------------|
| `gw/isdf_fitting.py`  | thread `vertex_mu_L` through `fit_zeta_chunked_to_h5`; γ̃^μ ∈ {γ⁰, α^1, α^2, α^3} replaces spin-traced pair density |
| `gw/isdf_fitting.py`  | `compute_L_q_from_CCT`: pick LU instead of Cholesky when `μ_L ≠ 0`           |
| `gw/gw_init.py`       | drive 4-channel loop over μ_L, with ζ-cache reset between channels           |
| `gw/gw_init.py`       | accept second centroids file (`centroids_file_current`) for μ_L=1,2,3        |

The four channels write four separate HDF5 files: `zeta_q.h5` for
μ_L=0 (charge centroids); `zeta_q_mu{1,2,3}.h5` for μ_L=1,2,3 (current
centroids).  Total disk: 4× the scalar ζ size — for CrI3 that's
~1.3 TB instead of 320 GB.  Worth flagging.

### 4.c — V_q orchestrator

New file, ~250 LOC, modelled on B's `gw/v_q_lorentz.py` but emitting
the `V_q_bispinor` struct above instead of a dict:

```python
def compute_V_q_bispinor(
    *,
    zeta_C_io: SlabIO,
    zeta_T_io: tuple[SlabIO, SlabIO, SlabIO],
    coulomb_kernels,                # bundle from make_v_munu_chunked_kernel
    mesh_xy: Mesh,
    kgrid: tuple[int,int,int],
    fft_grid: tuple[int,int,int],
    bvec, cell_volume,
    n_rmu_C: int,
    n_rmu_T: int,
    sys_dim: int = 3,
    bdot=None,
    bare_coulomb_cutoff=None,
    bgw_v_grid_fn=None,             # only meaningful for CC
    budget_bytes=None,
    verbose: bool = True,
) -> V_q_bispinor:
    ...
```

Internals (in order):
1. Build `_projector_weight(K_cart, μ_L, ν_L)` factory (B has it).
2. Build per-tile `v_per_G_fn` closures: scalar v(K) for (0,0); v(K)·t^{i,j}(K)
   for the 6 unique TT (3 diag + 3 off-diag).  Phase fn shared across all
   7.  BGW vcoul overlay applied only on (0,0).
3. Per-tile chooser (`_choose_v_q_chunks`) with the right (n_rmu_L,
   n_rmu_R, same_zeta) — diagonal tiles `same_zeta=True`, off-diagonal
   `same_zeta=False` (two reads).  Cache the chosen (q_chunk, μ_chunk)
   per shape — there are at most 3 distinct shape signatures
   ((C,C,T), (T,T,T), (T,T,F)).
4. Loop:
   - (0,0): `compute_V_q_tile(zeta_C, zeta_C, …, same_zeta=True,  v_per_G_fn=v_CC)` → CC
   - (i,i): `compute_V_q_tile(zeta_T_i, zeta_T_i, …, same_zeta=True,  v_per_G_fn=v_TT_ii)` → TT[i,i]
   - (i<j): `compute_V_q_tile(zeta_T_i, zeta_T_j, …, same_zeta=False, v_per_G_fn=v_TT_ij)` → TT[i,j]
5. Hermitian-fill: TT[j,i] = jnp.conj(jnp.swapaxes(TT[i,j], -1, -2)).
6. Return `V_q_bispinor(CC=CC, TT=TT_full_3x3, g0_C=g0_acc_from_CC)`.

### 4.d — Σ consumer (cohsex)

The bispinor σ_X^B / σ_H^B paths on B consume the dict.  When V_q_bispinor
is the on-disk + in-memory contract, the consumer just calls
`V.CC[q]`, `V.TT[i,j,q]` — no dict lookups, no None-checks.  Will need
to:

- Restructure `gw/cohsex_kernels.py` (or wherever σ_X^B lives on B) to
  take `V_q_bispinor` instead of dict.
- Confirm `flatten_V_qmunu` (current 8-D / 6-D / 3-D shim in
  `w_isdf.py`) is bypassed for bispinor — V_q_bispinor goes straight
  to its consumer; no flatten step.

### 4.e — Tagged-arrays (HDF5 round-trip)

`file_io/tagged_arrays.py` needs to learn `V_q_bispinor` as a known
container.  Two writes per restart: `V_qmunu_CC` (3-D flat-q) and
`V_qmunu_TT` (5-D, axes `(i, j, q, μ, ν)` with sharding
`P(None,None,None,'x','y')`).  Reader symmetric.

## 5. Memory & wall-time budget for CrI3 (back-of-envelope)

CrI3 6×6×1, μ_C ≈ 1500, μ_T ≈ 1808 (per B's notes), 16-GPU 4×4.

Per-tile read+kernel cost ≈ scalar V_q tile ≈ 250 s (read-bound) +
14 s (kernel) ≈ 265 s on the current pipeline.

7 unique tiles × 265 s ≈ 31 min for V_q alone.  Plus the 4-channel
ζ-fit at 33 min/channel × 4 = 132 min.  Total bispinor wall ≈ 165 min
≈ 2.75 h on a 16-GPU job.  B's commit message says "33 min fit_zeta +
~42 min V_q ≈ 75 min total" — that suggests B's V_q implementation
overlaps reads better than 7×scalar suggests, or its 7-tile budget is
single-channel-equivalent because reads are amortized when the same ζ
is consumed by multiple tiles ((1,1)/(1,2)/(1,3) all read ζ_T_1 once
plus ζ_T_2/ζ_T_3 once each).

Read amortization opportunity for the orchestrator: if we land all 7
tiles into a single outer driver that reads each ζ_T_i exactly once
across the (i,i)+(i,j) tiles that need it, V_q wall drops to
~3×265 s ≈ 13 min instead of 7×265 s.  Worth doing — call this the
"tile-fused outer loop" optimization.  Defer to v2; the v1
orchestrator can do the naive 7 calls.

## 6. Phasing

Recommend three commits, each independently testable:

**Commit B1 — port centroid current density and 4-channel ζ-fit.**
Files in §4.a + §4.b.  Test: round-trip a CrI3 ζ_T_1 to disk and
verify ζ_T_1 reconstructs the Pauli current within 1% on |j|² weighting.

**Commit B2 — V_q_bispinor struct + naive 7-tile orchestrator.**
Files in §4.c + the V_q_bispinor NamedTuple in `gw/v_q_tile.py` (or a
new `gw/v_q_bispinor.py`).  Test: run on MoS2 (light → bispinor V_q
should match scalar V_q to α² ~ 10⁻⁴ on the (0,0) tile; TT tiles
should be O(α²) corrections).

**Commit B3 — Σ_X^B / Σ_H^B consumer + cohsex wiring.**
Files in §4.d.  Test: end-to-end CrI3 cohsex with bispinor=True and
bispinor=False; bispinor − scalar should be O(α²·V_TT) ~ tens of meV
on heavy elements.

Tile-fused outer loop (§5) is a v2 follow-on, not in any of B1–B3.

## 7. Open questions

1. **Per-channel `n_rmu`.**  B uses `n_rmu_by_channel[ch]` — possibly
   different sizes for the four channels.  User's earlier remark
   ("the 1-3 sizes are all uniform") suggests we collapse μ_L=1,2,3
   to a single `n_rmu_T`.  Need to confirm: is one current-centroids
   file sufficient (kmeans on `Σ_i |j_i|²` gives one rank-3 tensor),
   or do we want per-component centroids (3 files, possibly
   different sizes)?  Single file is simpler and matches the
   `TT.shape = (3, 3, nq, n_rmu_T, n_rmu_T)` signature above.

2. **g0_T head.**  The transverse projector kills G=0 in the limit
   K̂ → q̂ on the diagonal (i,i) tile, but only when K aligns with the
   i-axis.  At generic q→0 the head is finite for off-diagonal (i,j).
   B's `compute_all_V_q_lorentz_sharded` only writes G0 for (0,0);
   double-check this is correct for the q=0 head correction in the
   transverse channels and add `g0_T: jnp.ndarray | None` if needed.

3. **WFN load_wfns 4-spinor padding.**  e45beea pads bands; check
   that the small-spinor lift in `bispinor_init.py` gets called on
   the padded wavefunctions during the current-density build — i.e.
   that `build_current_density` reads ψ_L, lifts to 4-spinor via
   `get_small_psi_component`, and then computes `j^Gordon`.

4. **Memory model for the orchestrator.**  Each tile call reuses the
   chooser; but if 3 tiles want q_chunk=12 and 4 tiles want q_chunk=8,
   the JIT cache holds 2 compiled kernels per (n_rmu, n_rmu) shape.
   Cache eviction discipline (already in `_drop_unused_v_q_kernel_cache_entries`)
   should clean up between tiles — verify.

## 8. Recommendation

Start with **commit B1** (centroid current density + 4-channel ζ-fit)
once the AOT-vs-slope+intercept distinguishing test on CrI3 lands
(see `reports/v_q_memory_model_open_2026-05-08/report.md`).  B1 is
gated only by data correctness — no V_q infra changes.  B2 and B3 can
be parallelized: B2 is V_q infra; B3 is sigma infra.

The user's expectation that "most of the relevant stuff was already
pushed to main" was right for the V_q kernel surface (same_zeta,
external v_per_G_fn, one-sided v(K), n_rmu_L≠n_rmu_R).  But the four
channels of ζ — and the centroid weighting that drives the
1,2,3-channel fits — are still on B's branch only and are the
critical path for bispinor.
