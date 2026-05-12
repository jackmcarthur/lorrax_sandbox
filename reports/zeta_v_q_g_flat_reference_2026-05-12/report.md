# G-flat ζ + V_q reference (LORRAX GWJAX)

**Status:** living reference for the post-2026-05-11 G-flat ζ on-disk
pipeline and the per-q-sphere V_q kernel that consumes it.  Source of
truth for shardings, donations, chunkers, and the bispinor extension.
Updated as CrI3 6×6 80 Ry validation lands.

**Mesh:** `Mesh(devices.reshape(p_x, p_y), ('x', 'y'))`,
**P** = `p_x · p_y`.  Sharding notation: `μ_X` = sharded on `'x'`,
`μ_XY` = sharded on flat `('x','y')`, `μ_` = replicated.

**Sizes (typical → extreme):**
`n_rtot ∈ [5k, 2M+]` (largest dim, the bottleneck for anything that
materialises a full FFT box; ALL such ops must be chunkable to ~1
FFT-box working set).
`n_band ∈ [50, 10000]`, `n_rmu ≈ 10·n_band ∈ [500, 100k]` (rounded up
to `mesh.size` as `n_rmu_padded`).  `n_G_sph ≈ 0.05·n_rtot`, comparable
in size to `n_rmu`.  `n_k, n_q ∈ [1, 10000]`.

---

## 0. Pipeline overview

```
WFN.h5 ──┐
         │  load_centroids_band_chunked
         ↓                                                   gflat layout
ψ_rmuT(k, μ_X, nb, ns)   ψ_rmu(k, nb, ns, μ_Y)             (q_, μ_XY, G_)
         │                                                        ↑
         │  for r_chunk in r_chunks:                        accumulate
         │      ┌──────────────────────────────────┐       _rchunk_to_gflat
         │      │ fit_one_rchunk (one big jit):    │             ↑
         │      │   pair_density (CCT, ZCT)        │       ζ_chunk(r_local)
         │      │   factor_c_q (Cholesky / LU)     │       (q_, μ_XY, r_)
         │      │   solve_zeta → reshard → ζ_chunk │ ──────────┘
         │      └──────────────────────────────────┘
         │                                                  zeta_q_G.h5
         ↓                                                  via SlabIO / phdf5 FFI
                                          ┌──── per-q FFT(pad(phase(ζ_chunk)))[sphere[q]]
                                          │
V_q kernel (per-q, G-chunked) ←── zeta_q_G.h5
   V[q, μ_L, ν_L] = Σ_G  ζ̄_L(q,μ,G) · v(q+G) · t^{μ_L ν_L}(q+G) · ζ_R(q,ν,G)
```

---

## 1. ζ-fit pipeline (`common/isdf_fitting.fit_zeta_to_h5`)

### 1a. Per-channel preamble (charge: μ_L=0; transverse: μ_L=1,2,3)

| Step | Shape | Sharding | Cost |
|---|---|---|---|
| centroid extract `to_rmu` | ψ at centroids `(k, n, s, μ)` | `P(None, ('x','y'), None, None)` | band-chunked by `k_chunk_size` (`load_wfns:k_chunk_size_autodetect`); per-rank FFT box `(_k_chunk, nb_padded, ns, nx, ny, nz)` is the transient; current autodetect over-budgets against unsharded `nb_padded`. |
| `compute_CCT_from_left_right` | C_q `(n_q, n_rmu, n_rmu)` | `P(None, 'x', 'y')` | band-streamed (bc-chunked) inside one jit. |
| `factor_c_q` | L_q `(n_q, n_rmu, n_rmu)` | `P(None, 'x', 'y')` (dense view), tile `P(None, 'x', 'y', None, None)` internally | cusolvermp_cholesky default (charge); cusolvermp_lu / `lu` for transverse (γ̃^i CCT is Hermitian indefinite); 2-D blocked fallback. |

### 1b. r-chunk loop (`fit_zeta_to_h5:2186…`)

For `r_start in steps of chunk_r`:

```
zeta_chunk = fit_one_rchunk(psi_l_rmuT, psi_r_rmuT, L_q, …)
           : (n_q, n_rmu_padded, r_chunk)   spec P(None, ('x','y'), None)
```

`fit_one_rchunk` is **one** jit (the big `_kernel`) that composes:

1. **ψ(G) → ψ(r-slab)** band-chunked via `psi_G_store.fetch_psi_rchunk`
   → `to_rchunk` (per-bc).  Optional `LORRAX_PSIG_KCHUNK` k-axis chunker
   bounds the FFT-box transient at large `n_k`.  Spec
   `P(None, ('x','y'), None, None)` → reshard to `(None, None, None, ('x','y'))`
   for pair-density.
2. **ZCT accumulator (`_pair_density / accum_pair_density`)** —
   einsum `'kmna,knbr->kabmr'` for the open-spin path (transverse;
   bispinor four-density work uses the rank-5 form), or
   `'kmns,knsv->kmv'` for the spin-traced charge path.  Output
   `(n_q, [ns,ns,] μ_X, r_Y)`.
3. **K-FFT to R, R-multiply, R→q FFT** → `Z_q (n_q, μ_X, r_Y)` spec
   `P(None, 'x', 'y')`.
4. **`solve_zeta`** → ζ.  Output **`P(None, ('x','y'), None)`**
   (μ_XY, r_) after a one- or two-step single-mesh-axis reshard.
   See §3.

### 1c. ζ-chunk → G-flat accumulator (`accumulate_rchunk_to_gflat`)

```
gflat_acc(n_q, n_rmu_padded, ngkmax)  spec P(None, ('x','y'), None)
   += FFT_3d(pad(phase(ζ_chunk)))[sphere_idx]
```

See §2.

### 1d. Post-loop write

```
gflat_acc  spec P(None, ('x','y'), None)
   ↓ jnp.where mask zeroes per-q pad slots
   ↓ SlabIO.write_slab('zeta_q_G', gflat_acc,
                       valid_shape=(n_q_disk, n_rmu_logical, ngkmax))
on disk    (n_q_disk, n_rmu_logical, ngkmax)  [μ-pad clipped]
```

`valid_shape` clips μ-axis pad on the write so on-disk extent is
logical.  phdf5 FFI when `use_ffi_io=true`.

---

## 2. `accumulate_rchunk_to_gflat` — flat-axis chunker

`src/common/wfn_transforms.py:accumulate_rchunk_to_gflat`.  One
`shard_map` over `('x','y')`.  Per-rank ops; **no cross-rank
collectives in the body**.

```
in  : rchunk    (n_q, n_rmu_padded, r_len)        P(None, ('x','y'), None)
      gflat_acc (n_q, n_rmu_padded, ngkmax)       P(None, ('x','y'), None)   ← donated
out : (n_q, n_rmu_padded, ngkmax)                 P(None, ('x','y'), None)
```

**Body (per-rank, inside shard_map):**

```
N = n_q · n_mu_local  ;  n_mu_local = n_rmu_padded / P
rch_flat ← rchunk.reshape(N, r_len)             # local
acc_flat ← acc.reshape(N, ngkmax)               # local
zero-pad both to ⌈N/cs⌉·cs along axis 0          # cs = chunk_size

for i in range(n_chunks):                       # lax.scan, donated acc carry
    sub        ← dynamic_slice(rch_flat, i·cs, cs, axis=0)
    q_row[cs]  ← clip((i·cs + arange(cs)) // n_mu_local, 0, n_q-1)
    box        ← zeros(cs, n_rtot).update_slice(sub @ r0, axis=-1).reshape(cs, nx, ny, nz)
    if qvec_frac:
        box   *= phx[q_row] · phy[q_row] · phz[q_row]   # separable per-q Bloch phase
    G_box      ← jnp.fft.fftn(box, axes=(-3,-2,-1))     # local cuFFT inside shard_map
    contrib    ← take_along_axis(G_box.reshape(cs, n_rtot), sphere_c[q_row], axis=-1)
    acc_flat   ← dynamic_update_slice(acc_flat, dynamic_slice(acc_flat, i·cs, cs) + contrib, i·cs, axis=0)

return acc_flat[:N].reshape(n_q, n_mu_local, ngkmax)
```

**Donation:** `gflat_acc` is in-place updated under the outer jit
(donated to `_kernel`; carry-donated through `lax.scan`).

**Chunker parameter — `chunk_size` (rows per scan iter):**
- Default `None` ⇒ one-shot (cs = N, scan compiles to 1 iter).
- Env override `LORRAX_GFLAT_CHUNK_SIZE`.
- Memory bound per rank: `chunk_size · n_rtot · 16 B` for the FFT
  box.  **The whole reason we need a chunker** — at CrI3 6×6 80 Ry
  (n_rtot ~ 5·10⁵) we cannot materialise the full N-row box.
- Suggested set: `chunk_size ≈ memory_budget_bytes / (n_rtot · 16) /
  (~6 live-copy slack)`.  Per-rank ~1 GB budget → cs ≈ `8e7 / n_rtot`.
  At n_rtot=243k (CrI3 J_3x3) cs≈300; at n_rtot=46k (MoS2 3×3)
  one-shot fits.
- Padding rows (`pad_N`) zero-padded; pad rows have `q_row≥n_q`
  clipped, contribute zero → no contamination, no divisibility
  constraint on n_q or n_mu_local.

**Per-q tables (closure-baked at trace time, replicated per rank):**
- `sphere_c (n_q, ngkmax)` int32 — per-q sphere into flat FFT box.
- `phx (n_q, nx)`, `phy (n_q, ny)`, `phz (n_q, nz)` — separable
  Bloch-phase exp(-2πi q·r) factors.

---

## 3. `solve_zeta` reshard discipline

Solve `L_q · ζ = Z_q`, output target sharding `P(None, ('x','y'), None)`.

| Solver | Native out | Reshard path |
|---|---|---|
| `cusolvermp_{cholesky,lu}` | `P(None, 'x', 'y')` (q_, μ_X, r_Y) | one step, single mesh axis 'y' moves r→μ: `→ P(None, ('x','y'), None)` |
| `sharded_cholesky` / `lu` (shard_map fallback) | `P(None, None, ('x','y'))` (q_, μ_, r_XY) | two steps via `P(None, 'x', 'y')`: 'x' moves r→μ, then 'y' moves r→μ |

Each step is a clean `(a_X, b) → (a, b_X)` single-mesh-axis all-to-all.
Avoids the q-passthrough trick the existing `_reshard_z` uses (which
worked for r_XY landing but is unnecessary for μ_XY).

The reshard is the only collective in `solve_zeta`'s output path; it
costs ~3 ms/call in the trace (`is_sync:true` per kernel-end placement,
but the actual collective is small).

---

## 4. V_q kernel (`gw/v_q_g_flat.py:compute_v_q_per_q_g_chunked`)

Per IBZ-q, G-chunked GEMM contracting ζ̃ tiles into V^{μ_L,ν_L}:

```
V_q[μ, ν]  +=  Σ_{G ∈ sphere[q]}  conj(ζ_L(q, μ, G)) · v(q+G) · t^{μ_L,ν_L}(q+G) · ζ_R(q, ν, G)
```

```
in  : zeta_q_L  (1, n_rmu_L, ngkmax_q)   P(None, ('x','y'), None)    ← read per-q from disk
      zeta_q_R  (1, n_rmu_R, ngkmax_q)   P(None, ('x','y'), None)
      v(q+G)    (ngkmax_q,)              replicated
      g_chunk   int                      static
out : V_q       (n_rmu_L, n_rmu_R)       P('x', 'y')                  ← μ × ν tile
```

**Per-tile loop** (`gw/v_q_bispinor.py`): 7 unique `(μ_L, ν_L)` tiles
(CC + 3 TT-diag + 3 TT-upper) × n_q_ibz reads, plus 3 Hermitian
fills.  Each tile's `v_per_G_fn` closure bakes in `v(q+G)` and
`t^{μ_L,ν_L}(q+G) = δ_ij − K̂_i K̂_j` (CC = 1, TT diag = 1−K̂²,
TT off-diag = −K̂_iK̂_j).

**Chunker parameter — `g_chunk`** (G-axis chunk size, lines V_q
inner kernel):  one chunk per q for MoS2 3×3 (ngkmax=1963 < budget),
multiple chunks at CrI3 scale.

**Per-q sphere read**: `ZetaReader.get_slab(q)` pulls one q's slab
of shape `(1, n_rmu, ngkmax_q)` from `zeta_q_G.h5` via phdf5 FFI.
Async prefetch (`LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH=1`) is opt-out
because the worker-thread read deadlocks against the NCCL collective
in production (see CHANGELOG 2026-05-11 G-flat shakedown).

---

## 5. Bispinor extension (μ_L ∈ {0,1,2,3})

**Channels:** 4 ζ files (`zeta_q.h5` for charge μ_L=0 with charge
centroids `n_rmu_C ≈ 8·nband`; `zeta_q_mu{1,2,3}.h5` for transverse
μ_L=1,2,3 with current centroids `n_rmu_T`; in production the three
transverse channels share the same current centroids file).

**ζ fit per channel:**
- μ_L=0: spin-traced rank-3 pair density `'kmns,knsv->kmv'`; CCT
  Cholesky-solvable (PSD).
- μ_L=1,2,3: open-spin rank-5 pair density
  `'kmna,knbr->kabmr'`; γ̃-perm/phase contraction at post-IFFT step
  (`_gamma_double_contract`).  CCT indefinite → pivoted LU + ridge
  `1e-12 · |tr(L)|/n_rmu`.

**V_q^{μ_L,ν_L} sectors** (Coulomb-gauge bare):

| Count | Sector | What | Stored as |
|---:|---|---|---|
| 6 | (0,i), (i,0) | identically zero | not stored |
| 1 | (0,0) CC | scalar Coulomb, BGW v(q+G) overlay applies | `V_qmunu_CC` |
| 3 | (i,i) TT-diag | `same_zeta=True`, weight `1 − K̂_i²` | `V_qmunu_TT_ii` |
| 3 | (i<j) TT-off | `same_zeta=False`, weight `−K̂_iK̂_j` | `V_qmunu_TT_ij` |
| (3) | (i>j) TT-Herm | read as `conj(swap(V[i<j], μ,ν))` | not stored |

**Σ^B trick** (`gw/sigma_x_bispinor.py`): fold γ̃ into ψ at the two
vertices, reuse unmodified scalar `sigma_sx_k(wfns_ij, G, V^{ij})`.
γ̃ is monomial (one nonzero per row, ∈ {±1, ±i}) so the fold is a
perm+phase gather, not a matmul.

---

## 6. Chunker inventory (all places n_rtot or big batch is touched)

| Site | Chunker | Param | Default | Env |
|---|---|---|---|---|
| `load_centroids_band_chunked` IFFT (band → r-mu) | k-axis chunk on FFT box | `k_chunk_size` | autodetect against `nb_padded · ns · n_rtot · 16 · peak_copies` | (none) |
| `psi_G_store.fetch_psi_rchunk` | k-axis chunk | `_k_chunk` | nk (off) | `LORRAX_PSIG_KCHUNK` |
| `iter_psi_rchunk_bandwise` | band-chunk on G → r FFT | `band_chunk` | from chunk chooser | (none) |
| outer r-chunk loop | r-axis chunk on Z_q + solve + accumulate | `chunk_r` | AOT chooser (`compute_optimal_chunks`) | (cohsex.in `r_chunk_size`) |
| `accumulate_rchunk_to_gflat` | flat-axis on (q · μ_local) | `chunk_size` | one-shot | `LORRAX_GFLAT_CHUNK_SIZE` |
| `compute_v_q_per_q_g_chunked` | G-axis chunk in V_q GEMM | `g_chunk` | ngkmax (off) | (config) |
| V_q outer loop over tiles | q-axis sync per tile | (none — Python loop) | per-q | (none) |

---

## 7. Key helpers (one-liners)

| Helper | What |
|---|---|
| `common.wfn_transforms.to_box / to_rbox / to_rmu / to_rchunk` | G-flat ψ → FFT box / r-space / r-at-centroids / r-slab.  All preserve psi's band-axis shard.  All four use `_box_kernel` + `_local_box_fft`. |
| `common.wfn_transforms._spec_of(psi)` | PartitionSpec padded to `psi.ndim` length (normalised; `_sharding_key` uses it). |
| `common.wfn_transforms._sharding_key(psi)` | `(mesh_id, normalized_spec)` — jit-cache key for the public transforms. |
| `common.wfn_transforms.apply_bloch_phase_on_slice` | Multiply slab by `exp(±2πi q·r)` on the kept r-cells only (used inside fit_one_rchunk for the cell-periodic seam). |
| `common.coulomb_sphere.compute_per_q_bare_coulomb_components` | Per-q sphere `{G : |q+G|² ≤ cutoff}`, padded to `ngkmax = max_q ngk[q]` with sentinel Miller index. |
| `common.isdf_fitting._reshard_zeta_mu_X_r_Y_to_mu_XY` | cuSolverMp branch's single-step reshard. |
| `common.isdf_fitting._reshard_zeta_r_XY_to_mu_XY` | shard_map branch's two-step reshard. |
| `gw.compute_vcoul.compute_v_q_per_q_g_chunked` | Per-q V_q GEMM kernel (G-chunked, μ_L,ν_L-agnostic via v_per_G_fn closure). |
| `file_io.zeta_reader.ZetaReader` | G-flat zeta_q_G.h5 slab reader with optional async prefetch. |
| `file_io.slab_io.SlabIO` | phdf5 FFI write/read with `valid_shape` clip for padded → logical. |

---

## 8. Donations (so future agents don't break in-place updates)

- `_pair_density` accumulator (`isdf_fitting._accum`): `donate_argnums=(0,)` on P_in.
- `accumulate_rchunk_to_gflat`: `donate_argnums=(1,)` on gflat_acc.
  Inside the shard_map's `lax.scan`, `acc_flat` is the carry → donated
  by scan semantics.
- `_solve_batch_and_update` (per-q-batch ζ solve): `donate_argnums=(2,)`
  on zeta_acc; outer Python loop chains donations to keep one zeta_acc
  live across iterations (alternative `lax.scan(unroll=8)` OOMs with
  preallocated-temp, `fori_loop` SPMD-replicates the sharded carry).
- `_reshard_z`: `donate_argnums=(0,)` on Z_q — caller must `del Z_q`
  immediately after.

---

## 9. Known sharding traps documented in-source

- **Two-mesh-axis reshard at once** = SPMD Involuntary Full Rematerialisation.
  Always stage through an intermediate that moves one mesh axis per step.
  See `solve_zeta:1013` for the `Z_q P(None,'x','y') → P(None,None,('x','y'))`
  two-step via `P('x', None, 'y')` (q-passthrough).
- **PartitionSpec trailing-None trim**: JAX hands back trimmed
  PartitionSpecs in some contexts; cache keys must normalise length
  (`_sharding_key` uses `_spec_of`).  Caught after 5 redundant
  `_kernel` compiles in MoS2 3×3 ⇒ +3 s wall.
- **`dynamic_slice` on a sharded axis with runtime start** all-gathers
  the axis.  `accumulate_rchunk_to_gflat`'s scan body slices only the
  flat `(q · μ_local)` axis **inside** shard_map (per-rank-local),
  never the global μ axis — avoiding the previous μ-chunker's
  cross-rank gather pattern.
- **cuSolverMp output spec** is `P(None, 'x', 'y')` (= input Z_q spec).
  The single-step reshard `(μ_X, r_Y) → (μ_XY, r_)` is one mesh axis
  ('y' moves r → μ); safe.

---

## 10. CrI3 validation log

(Appended chronologically.  Information-dense.)

### 2026-05-12 — AOT memory model is outdated; manual overrides only

`gw/aot_memory_model/` is stale w.r.t. the G-flat refactor; chooser
predictions are off by ~5–12× vs runtime XLA preallocation.
Workaround: **set `r_chunk_size`, `band_chunk_size`,
`memory_per_device_gb` explicitly** in cohsex.in and only consult
the chooser warnings for sanity bounds, not hard fits.

**Manual sizing rule of thumb** (per-rank, 4×4 mesh, c128 = 16 B):

```
ψ at centroids (persistent)  = n_q · (n_rmu/P_x) · nb · ns · 16
Z_q / pair-density / ζ (each, donated through loop)
                              = n_q · (n_rmu/P_x) · (r_chunk/P_y) · 16
band-chunk FFT box (transient, k peripheral)
                              = n_q · band_chunk · ns · r_chunk · 16
L_q replicated (solve)        = n_q · n_rmu² · 16
gflat_acc (persistent)        = n_q_disk · (n_rmu/P) · ngkmax · 16
per-iter accumulate FFT box   = chunk_size · n_rtot · 16
```

Budget target per rank ≤ ~30 GB to leave NCCL + XLA overhead headroom
on an 80 GB A100.  XLA can balloon by 2–4× the analytic estimate
through pipelining/fusion; pad budgets accordingly.

### 2026-05-12 — CrI3 6×6 80 Ry baseline OOMs in fit_one_rchunk

Run dir: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_muchunk_2026-05-12/`.

Geometry: FFT grid `(75, 75, 200)` → **n_rtot = 1.125 M** (24× MoS2 3×3),
n_q_disk = 8 IBZ (n_q_full = 36 for 6×6), nband = 150,
n_rmu = 1504, sys_dim = 2.  4 nodes × 4 GPUs = 16 ranks on
80 GB A100, mesh 4×4.

Symptom: With cohsex.in defaults (`r_chunk_size = 0` AOT-auto,
`memory_per_device_gb = 70.0`), AOT chooser picks r_chunk = 16512;
runtime XLA requests **58.88 GiB** contiguously for `_kernel`'s
preallocated-temp inside `fit_one_rchunk` → OOM at 80 GB GPU.

Forcing `memory_per_device_gb = 20.0` only drops the chooser's
self-reported peak prediction to 19.4 GB; runtime still asks for
60.34 GiB.  AOT predicted-vs-runtime gap is **~3×** with the small
budget, **~7×** with the large budget — the model can't be trusted
for this code path.

The OOM is **upstream** of `accumulate_rchunk_to_gflat`; my chunker's
`chunk_size=64` (per-iter FFT box 1.15 GB/rank) does not help here.
The offender is `fit_one_rchunk`'s composite kernel
(pair-density + CCT + ZCT + Cholesky + solve), where XLA's
BufferAssignment plans a 60 GB temp.

Next: manual `r_chunk_size` override → bisect downwards from 50000.

**Worked sizing math (per rank, 4×4 mesh, c128 = 16 B):**

| Object | Logical shape | Per-rank shape | Bytes |
|---|---|---:|---:|
| ψ_rmuT (persistent) | (n_q=36, n_rmu=1504, nb=150, ns=4) | (36, 376, 150, 4) | 0.36 GB |
| Z_q / pair-density | (n_q, n_rmu, r_chunk) | (36, 376, r_chunk/4) | 87 kB · r_chunk |
| L_q replicated for solve | (n_q, n_rmu, n_rmu) | (36, 1504, 1504) | 1.3 GB |
| gflat_acc | (n_q_disk=8, n_rmu, ngkmax≈55k) | (8, 1504/16, 55000) | 0.66 GB |
| accumulate FFT box / iter | (chunk_size, n_rtot) | (chunk_size, 1.125 M) | 18 kB · chunk_size |

At `r_chunk = 50000`: Z_q-class = 4.4 GB, persistent 1.7 GB, gflat 0.7 GB,
band-chunk FFT box transient (band_chunk·r_chunk·n_k·ns·16/P) = 1.4 GB —
total analytic ≈ 12 GB/rank.  Real XLA peak (per the chooser at the
same budget) = 19 GB; **runtime allocation request = 60 GB** ⇒ ~3×
unexplained pipelining/fusion blow-up.

Until the AOT model is rebuilt for this code path, use `r_chunk_size`
empirically: start at the chooser's pick / 3, halve until it fits.

### 2026-05-12 — XLA tells us the true floor

At `r_chunk = 50000`, XLA's `hlo_rematerialization` pass prints:

```
Can't reduce memory use below 27.43GiB by rematerialization;
only reduced to 41.94GiB, down from 60.35GiB originally.
```

So for `r_chunk = 50000` the kernel HLO has an irreducible ~27 GiB
per-rank peak — some intermediate inside fit_one_rchunk is materialised
unsharded (probably the full-BZ Z_q or pair-density at `(n_q=36,
n_rmu_padded, r_chunk) c128` — analytic would be 43 GiB total / 2.7
GiB sharded, but XLA is replicating somewhere).  This is the AOT
model's blind spot.

**Empirical rule of thumb at this scale:** XLA peak is ~10× the
sharded analytic estimate of `Z_q-class`.  At 4-GPU-per-node (80 GB
A100), a safe per-rank XLA budget is 50 GB ⇒ r_chunk ⪅ `5 ·
(50e9 / 16) / (n_q · n_rmu)` ≈ `1e8 / (36 · 1504)` ≈ 1850 at this
geometry.  Bisecting downward from 50000.

### 2026-05-12 — r_chunk-independent OOM ⇒ band_chunk_size is the lever

Device type: **NVIDIA A100-SXM4-40GB** (not 80 GB; the
``memory_per_device_gb = 60`` in cohsex.in is unenforceable here).

| r_chunk | XLA original | post-remat | XLA floor |
|---:|---:|---:|---:|
| 50000  | 60.35 GiB | 41.94 GiB | 27.43 GiB |
| 14992  | 69.55 GiB | 57.07 GiB | 29.11 GiB |

**XLA's allocation request is r_chunk-independent** (60.24 ≈ 60.35
GiB at both 15 k and 50 k r-chunks).  The OOM is therefore not in
the r-axis intermediates (Z_q-class) but in **a fixed-size
allocation upstream**: the band-chunk FFT box on the
``psi_G_store.fetch_psi_rchunk`` → ``to_rchunk`` path.

Per-rank band-chunk FFT box if fully sharded:
``n_k · band_chunk · ns · nx · ny · nz · 16 / P``
= ``36 · 16 · 4 · 75 · 75 · 200 · 16 / 16 / 16`` ≈ **2.5 GB / rank**
(at band_chunk = 16, 4×4 mesh).

If XLA materialises that FFT box **unsharded on every rank**:
``36 · 16 · 4 · 75 · 75 · 200 · 16`` ≈ **41 GB / rank**.

Plus XLA pipelining / fusion overhead → 60 GB.  On a 40 GB A100, this
is the OOM.  Same FFT box at ``band_chunk = 4``: ~10 GB unsharded
per rank — fits.

**Empirical rule (CrI3-class, 40 GB A100):**
- Always set ``band_chunk_size`` such that ``n_k · band_chunk · ns ·
  n_rtot · 16 < 8 GB`` even when materialised unsharded.  For CrI3
  6×6×1 (n_k=36, ns=4, n_rtot=1.125 M): ``band_chunk ≤ 4``.
- ``r_chunk_size`` is then much less constrained; choose for
  iteration-count efficiency (e.g. ~20–50 chunks total).
- ``LORRAX_GFLAT_CHUNK_SIZE`` (the accumulate FFT box) should still
  be set to bound ``chunk_size · n_rtot · 16 < 1 GB`` (CrI3:
  chunk_size ≈ 64).

The structural root cause — XLA failing to keep the band-chunk FFT
box sharded — is the same regression class the agent's "16× safety
margin" comment in ``load_wfns:657`` is working around.  Real fix
is to add a ``with_sharding_constraint`` inside the per-bc FFT path
to forbid unsharded materialisation.  Not done in this commit;
documented as a follow-up.

### 2026-05-12 — band_chunk_size=4 trips a separate phdf5 read bug

Setting ``band_chunk_size = 4, r_chunk_size = 30000`` to bound the
unsharded FFT box: kernel **doesn't OOM** but now fails at the
``loader.load(...)`` call inside the k-chunked load path
(``common/load_wfns.py:795``) with:

```
HDF5-DIAG: H5S_get_validated_dataspace(): selection + offset not within extent
  major: Dataspace,  minor: Out of range
INTERNAL: phdf5 read_kchunk_union: H5Dread failed
```

phdf5 issues an HDF5 dataspace hyperslab whose `(offset, size)`
extends past the dataset.  With nband = 150 and ``band_chunk = 4``,
the last band-chunk has 150 % 4 = 2 bands and the read-bounds math
in ``read_kchunk_union`` must be off by one for the remainder.
Untested: ``band_chunk_size`` values that divide n_band cleanly
(e.g. 10, 15, 25).

**Status:** CrI3 6×6 80 Ry on 40 GB A100s **requires** code-side
work beyond the μ_XY refactor:

1. Add a `with_sharding_constraint(box, P(None, ('x','y'), None,
   None, None, None))` inside `to_rmu` / band-chunked load to
   prevent XLA from materialising the FFT box unsharded.  This is
   the real fix; would allow `band_chunk_size = 16` to live in
   ~2.5 GB/rank instead of 41 GB/rank.
2. OR fix the `read_kchunk_union` remainder bound to allow
   `band_chunk_size = 4` without overshooting the dataspace.

Either fix unblocks CrI3 6×6 80 Ry.  The μ_XY refactor itself is
correct (MoS2 3×3 bispinor: bit-identical eqp0, wall 51.8 s vs the
pre-fix 53 s; structural eliminations confirmed) — it just doesn't
address either of the two CrI3-scale upstream problems.

### Manual override cookbook (for posterity)

In `cohsex.in`:
```ini
memory_per_device_gb = <≤ device_memory − 8 GB>   # 40 GB A100 → 32
r_chunk_size = <see analytic table §10 above>      # currently 15k–50k untestable
band_chunk_size = <2–8 ideally; check it divides n_band; check phdf5 bug>
```

Env knobs:
```bash
LORRAX_GFLAT_CHUNK_SIZE=<bytes-budget / (n_rtot · 16)>   # accumulate FFT box
LORRAX_PSIG_KCHUNK=<n_k / 4>                             # psi_G_store kchunk
```

Never trust ``memory_per_device_gb`` in cohsex.in as anything more
than a hint for the (stale) AOT chooser; XLA preallocation does its
own thing.

### 2026-05-12 — `with_sharding_constraint` on FFT box made it WORSE

Hypothesis: add ``with_sharding_constraint(box, P(None, ('x','y'),
None, None, None, None))`` immediately after ``_box_kernel`` and
again after ``ifftn`` in ``to_rmu`` / ``to_rchunk`` would force XLA
to keep the (n_k, nb, ns, nx, ny, nz) box band-sharded and prevent
the unsharded 41 GB/rank materialisation.

Result on CrI3 6×6 80 Ry, 80 GB nodes, default cohsex.in:

| Config                     | r_chunk | original | post-remat | floor |
|----------------------------|--------:|---------:|-----------:|------:|
| **no constraints (revert)**| 16512   | 60.35 GiB| 41.94 GiB  |**27.43 GiB**|
| **constraints added**      | 12528   | 87.54 GiB| 73.85 GiB  |**59.44 GiB**|
| **constraints added**      |  4992   | 99.44 GiB| 66.31 GiB  |**59.80 GiB**|

XLA's irreducible **floor doubled** with the constraint added (27 GiB
→ 60 GiB).  The constraint is **counterproductive** here.

MoS2 3×3 with the same constraints did still match (49.3 s vs 51.8 s
pre-constraint, bit-identical eqp0) — likely because at MoS2 scale
the box fits comfortably either way and the constraint just nudges
fusion.  At CrI3 scale it's actively harmful.

**Why?** Speculation (not verified by HLO diff): the constraint
matches the FFT helper's shard_map ``in_spec`` and should be a
no-op, but XLA's spmd-partitioner appears to interpret it as an
explicit "force materialise in this layout" hint, inserting copies
at fusion boundaries that would otherwise be elided.  The unsharded
intermediate is upstream of where the constraint sits and the
constraint forces XLA to keep BOTH the pre- and post-constraint
layouts live.

**Reverted in this session.**  The real fix is deeper than a
boundary constraint — likely needs the unsharded intermediate
**located** (HLO grep for the actual 60-GB tensor) and either
sharded at its creation site or chunked along its dominant axis.

Note: this is the same regression class the ``load_wfns:657`` "16×
safety margin" comment was hedging against.  The unsharded
intermediate is *somewhere* on the band-chunked load path
(``psi_G_store.fetch_psi_rchunk`` → ``to_rchunk`` → reshard to Y),
but pinning down which exact array is materialised unsharded
requires reading the HLO dump.

### 2026-05-12 — 80 GB A100s + `LORRAX_PSIG_KCHUNK=6` unblocks CrI3

What actually worked, after reverting the failed constraint:

1. **Allocated 80 GB A100 nodes** (``--constraint="gpu&hbm80g"``).
   The CrI3 6×6 80 Ry kernel's irreducible floor at default chunks
   is ~28 GiB; on 40 GB GPUs it can't fit, on 80 GB GPUs it does.
2. **Set ``LORRAX_PSIG_KCHUNK=6``** — k-axis chunker inside
   ``psi_G_store.fetch_psi_rchunk``.  Caps the per-call FFT box at
   ``(_k_chunk=6, bc=16, ns=4, n_rtot=1.125 M) ≈ 6.9 GB`` total
   (~0.43 GB/rank sharded; ~6.9 GB unsharded if XLA fails sharding).
   This was the unsharded culprit.

Compile + first r-chunk on 80 GB nodes with kchunk=6:
**no remat warnings, no RESOURCE_EXHAUSTED**, kernel proceeds to
the r-chunk loop.  AOT chooser picked ``r_chunk = 12528 / 90 chunks``.

(Wall-time numbers TBD; compile of fit_one_rchunk's `_kernel` at
CrI3 scale is ~1–2 min, then steady-state chunk rate to follow.)

**Recommended cohsex.in for CrI3 6×6 80 Ry on 80 GB A100s
(1 node × 4 GPUs minimum, 4 nodes × 4 GPUs preferred):**

```ini
memory_per_device_gb = 60.0
band_chunk_size = 16
r_chunk_size = 0          # let AOT chooser pick ~12500
# All other defaults.
```

With env knobs:
```bash
LORRAX_GFLAT_CHUNK_SIZE=64   # bound accumulate FFT box to 1.15 GB/rank
LORRAX_PSIG_KCHUNK=6         # bound band-load FFT box (THE key fix here)
```

On 40 GB nodes: must drop ``band_chunk_size`` to 4 or 8 (and fix the
``read_kchunk_union`` remainder bug at ``load_wfns:795`` first), OR
land the real ``with_sharding_constraint``-at-source-site fix.
