# G-flat ζ + V_q reference (LORRAX GWJAX)

**Status:** living mathematical/algorithmic reference for the
post-2026-05-11 G-flat ζ pipeline + per-q-sphere V_q kernel.  Covers
*what* is computed, *where the expensive collectives are*, and how
the per-peak memory model picks chunk sizes.  Source code is the
source of truth for exact donations and PartitionSpecs.

**Sharding notation:** axes annotated inline on each array's index
list.  `ψ[k, μ_XY, n, ns]` means ψ with the μ-axis flat-sharded
across both mesh axes; `_X`, `_Y`, `_XY` = single-axis or flat
shards on `Mesh((p_x, p_y), ('x', 'y'))`; no subscript = replicated.
`P = p_x · p_y` is the mesh size.  Reshards are written
`A[…, μ_XY, …] → A[…, μ_X, …]` so the mesh-axis move is visible.

**Sizes (typical → extreme):**
`n_rtot ∈ [5k, 2M+]` (FFT-grid total, dominant memory axis);
`n_band ∈ [50, 10000]`, `n_rmu ≈ 10·n_band`;
`n_G_sph ≈ 0.05·n_rtot`; `n_k, n_q ∈ [1, 10000]`.

---

## 0. What ζ and V_q are

**ζ** is an Interpolative Separable Density Fitting (ISDF) of the
pair-density basis at centroids `{r_μ}`.  For a bispinor pair
density `ρ_nm(r) = Σ_{αβ} ψ̄_n(r,α) γ̃^{μ_L}_{αβ} ψ_m(r,β)`,

```
ρ_nm(r) ≈ Σ_μ ζ_μ(r) · [ψ̄_n(r_μ) γ̃^{μ_L} ψ_m(r_μ)] ,
```

with centroid coefficients fixed by interpolation at `{r_μ}`.  The
fit is a per-q least-squares solve `L_q · ζ = Z_q`, with
`L_q = CCT(q)` the centroid Gram (q-independent in build, q-Fourier-
indexed at use) and `Z_q = ZCT(q)` the centroid-to-grid coupling,
both built band-streamed from pair densities.

**V_q** is the Coulomb interaction in the ISDF basis,

```
V^{μν}_q  =  Σ_G  ζ̄_L(q,μ,G) · v(q+G) · t^{μ_L,ν_L}(q+G) · ζ_R(q,ν,G) ,
```

over the q-sphere `|q+G|² ≤ cutoff`.  The bispinor vertex weight is
`t^{μ_L,ν_L}(q+G) = δ_ij − K̂_iK̂_j` (CC = 1, TT diag = 1−K̂²,
TT off-diag = −K̂_iK̂_j).  Channels `(0, i)` and `(i, 0)` vanish by
Coulomb gauge.

**On-disk handoff.**  ζ is written at G-sphere coordinates per q.
V_q reads one q at a time; the full ζ never sits in memory.

---

## 1. Pipeline overview

```
WFN.h5
   ↓ load_centroids_band_chunked
ψ[k, μ_XY, n, ns]
   ↓
for r_chunk in r_chunks:
    ┌──────────────────────────────────────────────┐
    │ fit_one_rchunk  (one jit, one fused kernel)  │
    │   pair density  (CCT once, ZCT each iter)    │
    │   factor C_q[q, μ_X, ν_Y]  (Chol / piv. LU)  │
    │   solve L_q · ζ_chunk = Z_q                  │
    │     ζ_chunk[q, μ_XY, r]                      │
    └──────────────────────────────────────────────┘
            ↓
    accumulate_rchunk_to_gflat
        gflat_acc[q, μ_XY, G_sph] += FFT_3d(pad(phase(ζ_chunk)))[sphere[q]]

(after loop)
   ↓ SlabIO.write_slab  →  zeta_q_G.h5

(per q, per channel)
V_q kernel  ←  ζ_L[1, μ_XY, G], ζ_R[1, ν_XY, G]  read from zeta_q_G.h5
```

The pipeline is **r-chunked, not q-chunked.**  r is the dominant
memory axis and the one we control via `r_chunk_size`; q, μ, n, G
are either small or sharded.

---

## 2. ζ-fit (`common/isdf_fitting.fit_zeta_to_h5`)

### 2a. Per-channel preamble

1. **Extract ψ at centroids** band-chunked.  Output
   `ψ[k, μ_XY, n, ns]` — the natural load layout from `WfnLoader`;
   the band axis is the only ψ-load axis the rest of the pipeline
   reshards on.
2. **Compute `C_q = CCT(q)`** band-streamed:
   pair density `P_k[k, ns_l, ns_r, μ_X, ν_Y]` →
   IFFT_k(P_l)* · IFFT_k(P_r) → γ̃-contract over spin axes →
   FFT_k→q → `C_q[q, μ_X, ν_Y]`.
   The pair-density input lives at the `XY → (X, Y)` mesh-axis
   split — ψ is resharded into
   `ψ_L[k, μ_X, n, ns]` and `ψ_R[k, n, ns, ν_Y]` — so the einsum's
   M and N axes are each sharded on one mesh axis only.  One
   mesh-axis all-to-all per side; **this is the first big
   collective.**
3. **Factor** `C_q → L_q[q, μ_X, ν_Y]`.  cuSolverMp `potrf`
   (charge — PSD) or `getrf` (transverse — Hermitian indefinite,
   see §4) keeps the factor 2D-block-cyclic in place.  No reshard
   between C_q and L_q.

### 2b. r-chunk loop body — one fused jit

1. **ψ(G) → ψ(r-slab)** band-chunked.  After the FFT to r-space
   the band axis is still the load shard; then the same
   `XY → (X, Y)` split reshards into the L / R einsum operands.
   Optional k-axis chunker `LORRAX_PSIG_KCHUNK` bounds the
   transient k-FFT box at large `n_k`.
2. **Pair density** at centroids × r-slab indices.
   - charge: spin-traced rank-3 `'kmns,knsv → kmv'`.
   - transverse: rank-5 open-spin `'kmna,knbr → karmb'` — output
     spec matches cuBLAS's natural gemm factoring so the rank-5
     → rank-7 reshape for the 3D FFT is a pure bitcast.
3. **IFFT_k → γ̃-contract → FFT_q**.  The FFT axis is k, which is
   replicated, so each rank runs a local cuFFT — no resharding.
   This is the whole point of the
   `P_k[k, ns_l, ns_r, μ_X, r_Y]` layout: mesh-sharded axes are
   inert under k-axis FFTs.  Output `Z_q[q, μ_X, r_Y]`.
4. **Solve** `L_q · ζ_chunk = Z_q`.  Both paths land
   `ζ_chunk[q, μ_XY, r]`:
   - **cuSolverMp branch** (default for distributed runs):
     `potrs`/`getrs` consumes `Z_q[q, μ_X, r_Y]` and outputs
     `ζ[q, μ_X, r_Y]` natively.  One single-axis reshard — 'y'
     moves from r onto μ — gives `ζ[q, μ_XY, r]`.
   - **shard_map fallback:** the triangular solve runs naturally
     at `ζ[q, μ, r_XY]`.  Going there from `Z_q[q, μ_X, r_Y]`
     requires moving both mesh axes on (μ, r) data axes; SPMD
     cannot plan that as one all-to-all, and a direct reshard
     triggers Involuntary Full Rematerialisation.  Stage through
     `Z[q_X, μ, r_Y]` — 'x' parked on the q-axis so each step
     moves one mesh axis:
     ```
     Z_q[q, μ_X, r_Y] → Z_q[q_X, μ, r_Y] → Z_q[q, μ, r_XY]
     ```
     The inverse two-step reshard after the solve lands ζ in
     μ-flat-sharded form.  **Donate `Z_q`** on the first reshard;
     verified at Si 4×4×4 60 Ry to drop HLO peak from 31 → 16 GB
     per device.

`ζ_chunk[q, μ_XY, r]` is exactly what the downstream
`accumulate_rchunk_to_gflat` wants: each rank owns a μ-slab over
the full r-extent so the 3D FFT on r runs as local per-rank cuFFT.

### 2c. ζ_chunk → G-flat (`accumulate_rchunk_to_gflat`)

One `shard_map` over `('x','y')`; **no cross-rank collectives in
the body.**  Inputs and output at `[q, μ_XY, r]` / `[q, μ_XY, G_sph]`.

Per rank, with `n_mu_local = n_rmu_padded / P`: for rows of the
flat `(q · n_mu_local)` axis in chunks of `chunk_size`, zero-pad
into an FFT box of extent `n_rtot`, multiply by the per-q Bloch
phase `exp(-2πi q · r)` (separable `x ⊗ y ⊗ z`), 3D FFT, gather
into `sphere_idx[q_row]`, accumulate in place into the donated
G-flat buffer.

Chunking on the flat row axis (not μ or q separately) drops any
divisibility constraint: pad rows with `q_row ≥ n_q` mask to zero
contribution.  `chunk_size · n_rtot · 16 B ≲ 1 GB/rank` is the
sizing rule; MoS2 3×3 (`n_rtot ≈ 46k`) one-shot fits.

---

## 3. V_q kernel (`gw/v_q_g_flat.py`, `gw/v_q_bispinor.py`)

Per IBZ q, per bispinor channel `(μ_L, ν_L)`:

```
read ζ_L[1, μ_XY, G], ζ_R[1, ν_XY, G] from disk
V[μ, ν] = Σ_G  ζ̄_L · v(q+G) · t^{μ_L,ν_L}(q+G) · ζ_R    (G-chunked GEMM)
```

ζ arrives at `[1, μ_XY, G]` (the layout `gflat_acc` lives in at
§2c).  Inside the kernel the q-axis is dropped and ζ is **recast
to single-axis shardings** to align with the matmul output tile:

```
ζ_L  →  [μ_XY, G] → [μ_X, G]
ζ_R  →  [ν_XY, G] → [ν_Y, G]
V_q   :  [μ_X, ν_Y]     (output tile)
```

Two single-axis all-to-alls per q (`XY → 'x'` for L, `XY → 'y'`
for R) — cheap relative to the GEMM at MoS2 scale, non-trivial at
CrI3 scale.  `same_zeta=True` aliases one buffer for both sides.

The G-chunker `g_chunk` bounds the inner `lax.scan` GEMM's working
set.  MoS2 3×3: one chunk per q (`ngkmax ≈ 1963`).  CrI3 scale:
multiple chunks.

**Async prefetch** (`LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=1`) is
opt-in: the worker-thread phdf5 read can deadlock against an
in-flight NCCL collective.

---

## 4. Bispinor extension (μ_L ∈ {0,1,2,3})

Four ζ files: one charge (μ_L=0, `n_rmu_C ≈ 8·n_band`) and three
transverse (μ_L=1,2,3, `n_rmu_T`).

**Why factorisation differs.** γ̃^0 = I makes `ρ = Σ_s ψ̄ψ` positive,
so charge CCT is PSD and Cholesky is exact.  γ̃^i are Pauli-analog
tensors with mixed signs, so transverse CCT^i is Hermitian
indefinite — Cholesky is invalid.  The code uses pivoted LU with
ridge `1e-12 · |tr(L)|/n_rmu` to lift any TRS-paired near-zero
modes above the LU stability floor without perturbing
well-conditioned modes.

**V_q sectors** (Coulomb-gauge bare, 16 → 7 stored):
six (0, i) / (i, 0) sectors vanish by Coulomb gauge.
Stored: (0, 0) CC; three (i, i) TT-diagonal with `same_zeta=True`,
weight `1 − K̂_i²`; three (i, j), i<j TT-off with
`same_zeta=False`, weight `−K̂_iK̂_j`.  The remaining i>j sectors
are recovered post-hoc as `conj(swap_μν(V[j, i]))`.

**Σ^B trick** (`gw/sigma_x_bispinor.py`).  γ̃ is monomial (one
nonzero per row, `∈ {±1, ±i}`) so folding γ̃ into ψ at the two
self-energy vertices is one `jnp.take` + element-wise phase
multiply — not a 4×4 matmul.  The scalar
`sigma_sx_k(wfns_ij, G, V^{ij})` then runs unmodified.

---

## 5. Memory model — fill the budget with one big r-chunk

### 5.1. The problem in one paragraph

An un-chunked `C_q · ζ = Z_q` fit needs TB-class HBM per rank
(`n_q · n_rmu · n_rtot · 16 B` ≈ 80 GB at MoS2 3×3, ≈ 3 TB at CrI3
6×6 80 Ry).  GPUs have tens of GB.  We chunk over the r-axis so
the working set fits.  The goal is **the smallest possible chunk
count**: each r-chunk pays a fixed FFT overhead (ψ-fetch +
ζ-accumulate) that's identical regardless of how many r-points the
chunk covers, so doubling r_chunk halves that overhead.  Bigger
r-chunks are always better for wall time; memory is the only
constraint.

### 5.2. Three workspaces, two memory pools, one performance goal

**Persistent pool** (alive every r-chunk iter):

```
B_persist = 2 · nk · ns · n_rmu · n_band · 16 / P    ψ at centroids (L+R)
          + n_q · n_rmu² · 16 / P                    L_q factor
          + n_q_disk · n_rmu · ngkmax · 16 / P       gflat_acc (G-flat ζ)
```

Fixed at problem-setup time, can't be reduced.  MoS2 3×3:
~0.5 GB.  CrI3 6×6 80 Ry: ~3 GB (gflat_acc dominates).

**Workspace pool** `W_pool = B − B_persist`.  Three transient
blocks contest this pool **sequentially** inside each r-chunk iter
— they do not co-exist in time, so XLA aliases them to share
physical memory:

```
  Step           Block            Size                                 Knob
  ─────────────  ───────────────  ───────────────────────────────────  ────────────────────
  ψ(G)→ψ(r)      W_wfn            k_chunk · band_chunk · ns · n_rtot   band_chunk_size,
                 (FFT box)        · 16 · fft_factor / P                psig_k_chunk_size
  ─────────────  ───────────────  ───────────────────────────────────  ────────────────────
  C_q / Z_q      W_zeta           3 · n_q · ns² · n_rmu · r_chunk      r_chunk_size
                 (3 pair-density  · 16 / P                              (the dominant lever)
                  slots)
  ─────────────  ───────────────  ───────────────────────────────────  ────────────────────
  ζ → G-flat     W_accum          gflat_chunk_size · n_rtot            gflat_chunk_size
                 (FFT box)        · 16 · fft_factor / P
```

`W_wfn` and `W_accum` are **independent of r_chunk** — they are
per-FFT working sets sized only by their own knobs.  Only `W_zeta`
scales with `r_chunk` (linearly).

Aliasing means the binding peak per iter is
`max(W_wfn, W_zeta, W_accum)` rather than the sum.  In practice
`W_zeta` is the binding peak at any reasonable `r_chunk`; `W_wfn`
and `W_accum` are smaller terms that just need to fit under the
same ceiling.

### 5.3. The performance objective — explicit tradeoffs

Total runtime of the ζ-fit loop (suppressing the once-per-channel
CCT preamble):

```
T_total  ≈  n_rchunks · ( T_zeta(r_chunk) + n_band_chunks · T_wfn_fft
                          + T_accum_fft )

with
  n_rchunks       = ⌈n_rtot / r_chunk⌉
  n_band_chunks   = ⌈n_band / band_chunk⌉
  T_zeta(r_chunk) = r_chunk · τ_zeta_per_r_unit       (linear in r_chunk)
  T_wfn_fft       = const(band_chunk, n_rtot)         (independent of r_chunk)
  T_accum_fft     = const(gflat_chunk_size, n_rtot)   (independent of r_chunk)
```

Substituting and simplifying:

```
T_total  ≈  n_rtot · τ_zeta_per_r_unit                        ← fixed ζ work
          + (n_rtot / r_chunk) · n_band_chunks · T_wfn_fft    ← wfn FFT tax
          + (n_rtot / r_chunk) · T_accum_fft                  ← accum FFT tax
```

The first term is the actual physics work; you pay it no matter
how you chunk.  The second and third are the chunk-count tax:
**every extra r-chunk doubles the FFT count on both the
wavefunction-fetch side and the accumulator side**.

Implications:

- **`r_chunk` is the dominant performance lever** because it
  divides BOTH overhead terms.  Halving `r_chunk` doubles the
  wall time spent in FFTs.
- `band_chunk` is a secondary lever (only divides the wfn term).
  `band_chunk = n_band` (single fetch per r-chunk) gives one wfn
  FFT per r-chunk.
- `gflat_chunk_size` similarly: one-shot eliminates the
  scan-over-rows inside the accumulator.

So the algorithmic rule is:

> Pick each chunk size as **large as memory allows**.
> `r_chunk` first (biggest win), then `band_chunk`, then
> `gflat_chunk_size`.

There is no tradeoff *between* them within memory — `W_wfn`,
`W_zeta`, `W_accum` are independent of each other and all draw
from the same `W_pool`.  The tradeoff is purely against `W_pool`
itself.

### 5.4. The actual algorithm — five logical steps

```
W_pool   ← B − B_persist                                # all transients share this
α_zeta   ← 3 · n_q · ns² · n_rmu · 16 / P               # W_zeta = α_zeta · r_chunk

# Step 1.  Maximise r_chunk — biggest performance lever.
r_chunk  ← min(W_pool / α_zeta,  n_rtot)
r_chunk  ← max(r_chunk,          n_rmu)                  # iter-overhead floor

# Step 2.  Maximise band_chunk subject to W_wfn ≤ W_pool.  XLA
#         aliases the wfn FFT box into a pair-density slot, so the
#         practical ceiling is slightly under one slot:
#           W_wfn ≤ W_pool / pair_density_slots  (≈ W_pool / 3)
#         — gives a fighting chance of clean aliasing.
band_chunk ← largest pow2 ≤ n_band with
             W_wfn(band_chunk) ≤ W_pool / pair_density_slots

# Step 3.  Maximise gflat_chunk_size subject to W_accum ≤ W_pool.
#         Accumulate is a separate XLA module, so it gets the full
#         pool again.  Default to one-shot (full row count); bisect
#         down only if it doesn't fit.
gflat_chunk_size ← N_rows if W_accum(N_rows) ≤ W_pool
                   else largest int with W_accum(...) ≤ W_pool

# Step 4.  If psig_k_chunk_size = 0 and band_chunk · n_rtot would
#         force an unsharded FFT box bigger than W_pool, drop
#         psig_k_chunk_size by halves until W_wfn (unsharded form)
#         fits.  See §5.8.

# Step 5.  Same monotone "largest such that W ≤ ceiling" for
#         vq_g_chunk_size in the V_q pass — but V_q runs after the
#         r-chunk loop, so it's a separate budget problem (no
#         W_zeta competition; just W_pool with V_q replacing
#         W_zeta).
```

Five logical steps; the rest is arithmetic.  The current
`plan_gflat_chunks(...)` follows steps 1–3 with a more
conservative 50/50 W_wfn-vs-W_zeta split; the §5.5 analysis says
`1/pair_density_slots` is the principled choice.

### 5.5. Why the tradeoff is monotone (one knob each)

`W_zeta`, `W_wfn`, `W_accum` each depend on exactly one chunk
knob.  No chunk size appears in two budgets.  Each knob's value is
"largest integer s.t. its W ≤ ceiling," which is a single
inequality, no joint optimisation needed.

The only non-trivial design choice is the band_chunk ceiling
fraction in step 2 — `1/pair_density_slots` is the principled
choice (band_chunk's FFT box has to fit inside one pair-density
slot for XLA's aliasing to actually hold).  Calibrate against an
HLO dump if a different XLA version aliases differently.

### 5.6. The magic constant — `pair_density_slots = 3`

Count of distinct lifetime offsets in XLA's BufferAssignment
holding a pair-density-shaped buffer.  Hand-extracted from
`module_*.jit__kernel.memory-usage-report.txt`.

Was 5 under the legacy decomposed chain (P_l, P_r, P_l_R, P_r_R,
γ̃-contract scratch).  The monolithic-shard_map bake (2026-05-13)
collapsed it to 3 (P_l_R_conj, P_r_R, one XLA scratch).  The
karmb einsum-spec change did not move it.

This is the model's biggest fragility.  Any non-trivial change to
the fused kernel — donation pattern, einsum spec, γ̃-contract
structure — can shift the count.  Re-extract from a fresh dump
after any kernel-structure edit; the planner over-allocates by
the wrong factor otherwise.

### 5.7. Cohsex.in surface

All chunk knobs are cohsex.in fields, all named `*_chunk_size`,
all default `0` → planner decides:

```ini
memory_per_device_gb = 0      # 0 = auto-detect GPU HBM; sets B
band_chunk_size      = 0      # 0 = planner picks band_chunk
r_chunk_size         = 0      # 0 = planner picks r_chunk (the big lever)
psig_k_chunk_size    = 0      # 0 = no inner k-chunking inside WFN fetch
gflat_chunk_size     = 0      # 0 = one-shot, or planner picks
vq_g_chunk_size      = 0      # 0 = V_q kernel picks via _pick_g_chunk
```

A non-zero cohsex value wins over the planner's pick.

### 5.8. Where the model is still off

- **`W_wfn` when XLA refuses to shard the FFT box.**  At CrI3 6×6
  80 Ry the loader's FFT box gets materialised unsharded on every
  rank — `W_wfn` jumps by a factor of `P` and blows the budget.
  `psig_k_chunk_size = 6` is the manual cap; the planner doesn't
  apply it automatically because its `W_wfn` formula assumes
  sharding holds.  **Fix priority #1**: model the unsharded case
  (or locate the unsharded intermediate via HLO grep and shard
  it at the creation site — see §6, the boundary
  `with_sharding_constraint` didn't work).
- **cuSolverMp internal scratch** (~`n_rmu²`-class).  Not
  modelled; small now, flag at CrI3.
- **No `W_vq` term.**  V_q runs after the r-chunk loop with only
  `gflat_acc` persistent — separate budget problem.  The cohsex
  knob `vq_g_chunk_size` exists but the planner doesn't pick it
  from `B`; default falls back to `_pick_g_chunk(ngkmax)` capped
  at 4096.
- **`fft_factor = 4.0` is a single scalar.**  cuFFT scratch +
  pipelining overhead varies by call site.  Empirical within 10%
  at current scales; worth re-verifying at large `n_rtot`.

### 5.9. Suggested implementation plan (for a future agent)

The current `plan_gflat_chunks` already follows §5.4 steps 1–3,
but with a 50/50 split between `W_wfn` and `W_zeta` that's more
conservative than the §5.5 aliasing analysis suggests.  A cleaner
rewrite:

1. Compute `B_persist` from problem geometry; check `B_persist ≤ B`
   or raise an informative error before any kernel compiles.
2. Compute `W_pool = B − B_persist`.
3. Pick `r_chunk` per step 1 of §5.4.  Log
   `"r_chunk = N (W_zeta = X.X GB / W_pool = Y.Y GB, n_rchunks = K)"`.
4. Pick `band_chunk` per step 2 using
   `W_pool / pair_density_slots` (not 50%) as the ceiling.  Log
   `"band_chunk = N (W_wfn = X.X GB / slot-budget Y.Y GB,
   n_band_chunks = M)"`.
5. Pick `gflat_chunk_size` per step 3.  Log
   `"gflat_chunk_size = N (W_accum = X.X GB / W_pool = Y.Y GB,
   one-shot fits / bisected down to ...)"`.
6. If `W_wfn(band_chunk = 1, k_chunk = nk)` already exceeds the
   slot ceiling, raise: the WFN fetch is structurally too big for
   the budget; suggest `psig_k_chunk_size`.

Tests worth having:

- **Calibration test**: after the next non-trivial change to the
  fused kernel, re-extract `pair_density_slots` from an HLO dump;
  fail loudly if the constant in source doesn't match.
- **Budget-edge test**: at a known `(B, geometry)`, verify the
  chosen chunks land at the budget edge (not 2× under).
- **Regression test**: MoS2 3×3 and one CrI3 size; verify the
  picked `(r_chunk, band_chunk, gflat_chunk_size)` triple still
  matches recorded golden values within a tolerance.

### 5.10. Follow-ups (priority order)

1. **Model the unsharded `W_wfn` case** (§5.8 fix priority #1) so
   the planner can pick `psig_k_chunk_size` automatically.
2. **Auto-recalibrate `pair_density_slots`** from an HLO dump
   pass — one r-chunk in dump mode, count pair-density-shaped
   lifetime slots, fail-loud if it differs from source.
3. **Add a `W_vq` term** and matching constraint so
   `vq_g_chunk_size` lands in the same chunker pass.
4. **Per-call-site `fft_factor`** (separate constants for
   Peak A's pre-loop FFT, Peak C's k-FFT inside the fused kernel,
   Peak D's accumulate FFT — they have different fusion
   neighbourhoods).

---

## 6. Sharding traps (lessons not visible in code)

- **Two-mesh-axis reshard in one op = Involuntary Full
  Rematerialisation.**  Always stage through an intermediate that
  moves one mesh axis per step (see §2b.4 shard_map fallback).
  Same pattern appears in `w_isdf._get_w_solve_fn`.
- **Donate any reshard that copies a large array.**  Donating
  `Z_q` on the two-step reshard halved per-device peak on
  Si 4×4×4.
- **PartitionSpec trailing-None trim.**  JAX hands back trimmed
  PartitionSpecs in some contexts; jit-cache keys must normalise
  length or re-compile spuriously (cost +3 s wall at MoS2 3×3
  from five redundant `_kernel` compiles).
- **`dynamic_slice` on a globally sharded axis with a runtime
  start all-gathers that axis.**  The flat-axis chunker in
  `accumulate_rchunk_to_gflat` slices only inside `shard_map`, on
  a per-rank-local flat `(q · μ_local)` axis — never on the
  global μ axis.
- **`with_sharding_constraint` is not free at scale.**  At
  CrI3-scale it can force XLA to keep both the pre- and
  post-constraint layouts live, doubling peak.  Try removing
  constraints before adding them.

---

## 7. Configuration recipes

### MoS2 3×3 (`n_rtot ≈ 46k`), 4 × A100-40GB

Defaults work; no env overrides.  Pipeline fits in one r-chunk
with `r_chunk_size = 0` (chooser picks ~12 k).  Peak HBM ≈ 13 GiB
per rank at the fused kernel, with three concurrent ~4.3 GiB
pair-density slots; transverse-channel per-rank pair density is
`n_q · ns² · (μ/p_x) · (r_chunk/p_y) · 16 B
 = 9 · 16 · 328 · 6064 · 16 B ≈ 4.3 GiB` on a 2×2 mesh.

### CrI3 6×6 80 Ry (`n_rtot ≈ 1.13 M`, `n_q = 36`), 80 GB nodes

The fused kernel's irreducible XLA floor at default chunks is
~28 GiB/rank — only 80 GB A100s fit the 4×4 mesh.  Required:

```ini
# cohsex.in
memory_per_device_gb  = 60.0
band_chunk_size       = 16
r_chunk_size          = 0    # planner picks ~12500
gflat_chunk_size      = 64   # bound accumulate FFT box ≤ ~1 GB/rank
psig_k_chunk_size     = 6    # bound the unsharded band-load FFT box
```

No env vars.  The structural problem: the band-chunk FFT box
inside `psi_G_store.fetch_psi_rchunk` is materialised unsharded on
every rank (~41 GB at default `band_chunk = 16`).  The model has no
cost term for it (§5.1) — `psig_k_chunk_size = 6` is the manual
mitigation.  Open follow-up: locate that intermediate via HLO grep
and shard it at the creation site (a constraint at the call boundary
does not work — see §6).  Landing it would relax the 80 GB hardware
requirement.
