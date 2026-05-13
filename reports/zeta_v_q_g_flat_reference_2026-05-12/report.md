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

## 5. Memory model (`gw/gflat_memory_model.py`)

A static per-peak model decomposes per-rank HBM into **four named
peaks** that span the pipeline:

- **Peak A** — band-chunked centroid load (pre-loop):
  ψ(G) → IFFT → sample at r_μ.  Dominant term: the ψ(r) FFT box
  transient `nk · band_chunk · ns · n_rtot · fft_factor`, sharded
  on `('x','y')`.  Runs once per channel.
- **Peak B** — CCT + Cholesky (pre-loop):
  pair density on (μ, ν) full grid + `C_q` + `L_q`.  Persistent
  centroids dominate.
- **Peak C** — `fit_one_rchunk` fused jit (inside the r-chunk loop):
  this is the runtime bottleneck on every system we've measured.
  Dominant term is `pair_density_slots × n_q · ns² · (μ/p) · (r_chunk/p) · 16`
  — the count of concurrent rank-5 pair-density buffers XLA keeps
  live during the IFFT/contract/FFT sub-pipeline.
- **Peak D** — `accumulate_rchunk_to_gflat`:
  gflat_acc persistent + per-iter FFT box
  `gflat_chunk_size · n_rtot · fft_factor`.

`plan_gflat_chunks(...)` picks `(band_chunk, r_chunk, gflat_chunk_size)`
deterministically: maximise band_chunk subject to Peak A/C
FFT-box headroom, then maximise r_chunk subject to Peak C with
lower bound `r_chunk ≥ μ` (any chunk smaller than that wastes
iteration overhead since the Σ_μν output is itself `μ²` work),
then set gflat_chunk_size to one-shot if Peak D fits, else binary
search down.  The HWM is `max(A, B, C, D)` and the bottleneck name
is surfaced in the run log.

The model is **already wired in as the chooser** for
`band_chunk` / `r_chunk` / `gflat_chunk_size` under
`LORRAX_WRITE_G_FLAT_ZETA=1` (default-on); cohsex.in overrides
short-circuit the corresponding stage.  q-chunk and k-chunk
choices still flow through the legacy `compute_optimal_chunks`.

**The magic constant** is `pair_density_slots`.  XLA's
BufferAssignment keeps 5 concurrent rank-5 pair-density buffers
live during the fused kernel — verified by counting distinct
lifetime offsets in `module_*.jit__kernel.memory-usage-report.txt`
on MoS2 3×3 bispinor (`runs/MoS2/00_mos2_3x3_cohsex/D_perf_after_2026-05-12`).
This is the constant that makes the model match runtime to within
~10% on systems we've measured.

### 5.1. Where the model is still blind

- **`psi_G_store.fetch_psi_rchunk` K-FFT box.**  The unsharded
  transient there is the CrI3-scale OOM root cause; the model
  has no term for it.  Patched out-of-band by
  `LORRAX_PSIG_KCHUNK=6` — but the chooser doesn't know to set it.
- **cuSolverMp internal buffers.**  `potrs` / `getrs` carry their
  own scratch (~`n_rmu²`-class).  Not enormous but unmodelled.
- **V_q kernel.**  No Peak E for the per-q matmul.  `g_chunk`
  comes from cohsex.in or `_pick_g_chunk(ngkmax)`.  At CrI3 scale
  this stays a manual knob.
- **XLA pipelining inflation.**  The model uses a single
  `fft_box_factor = 4.0` to account for cuFFT scratch + pipeline
  overhead.  Empirically XLA can balloon by 2–4× over the
  analytic figure; the 4× is conservative on systems we've
  measured, but for very large `n_rtot` it's worth re-verifying.

### 5.2. Ways to extend it

In rough priority order:

- **Add a Peak A' / Peak C' term for the `psi_G_store` K-FFT box.**
  This is the most-impactful gap: the model currently passes runs
  that actually OOM on CrI3.  The term is
  `_k_chunk · band_chunk · ns · n_rtot · fft_factor`, sharded
  only on `('x','y')` if the with-sharding-constraint path works
  (it does not today — see §6), otherwise unsharded.  Wiring
  this in lets the chooser pick `_k_chunk` (currently `LORRAX_PSIG_KCHUNK`)
  alongside the other knobs.
- **Auto-recalibrate `pair_density_slots` from a profiling pass.**
  Today it is a hard-coded 5 extracted by hand from an XLA dump.
  After any non-trivial change to the fused kernel (donation pattern,
  einsum spec, gamma-contract structure) the slot count can shift —
  the karmb-spec experiment did not change it, but the prior
  `'kabmr'`-spec did.  A short profiling helper that re-runs one
  r-chunk in dump mode and counts P-shaped lifetime slots from
  the `memory-usage-report.txt` would keep the constant honest
  across XLA/JAX upgrades.
- **Add a Peak E for V_q.**  Inputs: ζ slabs at `[1, μ_XY, G]`,
  reshards to `[μ_X, G]` / `[ν_Y, G]`, GEMM output `[μ_X, ν_Y]`,
  scratch for the G-chunked `lax.scan` body.  Lets the chooser
  pick `g_chunk` from the same budget, and surfaces the V_q
  reshard buffers in the HWM breakdown.
- **Split `fft_box_factor` per peak.**  Peak A's pre-loop FFT and
  Peak D's accumulate FFT have different fusion neighbourhoods —
  Peak D's box is followed by a gather + accumulate that XLA may
  fuse, dropping the live multiplier.  Peak C's k-FFT happens
  inside the fused kernel and is the only one that pipelines
  with the pair-density buffers.  A per-peak factor would let
  the chooser stop being globally conservative.
- **Report headroom and bottleneck dependency.**  Today the log
  prints HWM and the bottleneck name.  The natural next step is
  "Peak C at 81%, the slack comes from band_chunk = 16; halving
  it would free 1.4 GB at the cost of 2× pre-loop FFT time" —
  letting the user trade compile/runtime cost against memory
  without re-running.

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
memory_per_device_gb = 60.0
band_chunk_size      = 16
r_chunk_size         = 0          # chooser picks ~12500
```

```bash
LORRAX_GFLAT_CHUNK_SIZE=64   # bound accumulate FFT box ≤ ~1 GB/rank
LORRAX_PSIG_KCHUNK=6         # bound the unsharded band-load FFT box
```

The structural problem: the band-chunk FFT box inside
`psi_G_store.fetch_psi_rchunk` is materialised unsharded on every
rank (~41 GB at default `band_chunk = 16`).  The model has no term
for it (§5.1) and `LORRAX_PSIG_KCHUNK=6` is the manual workaround.
Locating that intermediate via HLO grep and sharding it at the
creation site (a constraint at the call boundary does not work)
is the open follow-up; landing it would relax the 80 GB hardware
requirement.
