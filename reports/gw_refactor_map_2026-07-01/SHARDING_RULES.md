# LORRAX GW — SHARDING / CHUNKING / RESTAGING rules (the memory spine)

> Companion to `MAP.md` (the dataflow spine). Where MAP.md answers *"where does a new
> stage kernel go,"* this answers *"how may each array be laid out across the mesh, when
> must an operation be chunked instead of materialized, and how are tensors redistributed
> between stages — and WHY."* Synthesized 2026-07-02 from a 5-agent full read of
> `sources/lorrax_D/src/{isdf,gw,common,file_io,runtime}` (ζ-fit · V_q · χ₀/W · Σ · ingest/IO ·
> planner). Every rule cites `file:line`. Checkout: `sources/lorrax_D`.
>
> The one sentence to internalize: **N_rmu (the ISDF centroid count) is the sharding
> currency of the entire code.** Every layout rule below is a consequence of "an N_rmu²
> array cannot live on one device, and the N_rtot FFT box cannot live on one device even
> once."

---

## 0. The budget and the scaling (why any of this exists)

Per-**process** (one GPU/device) HBM budget is **O(5–100 GB)** — 28 GB on a 40 GB A100,
6 GB on the 8 GB local card (`AGENTS.md` memory table; `memory_per_device_gb` config
`gw_config.py:246`, GPU auto-detect when 0 at `gw_config.py:849-856`). Every "must not
materialize" statement below is relative to that per-process number, **after** sharding.

The physical axes and their ranges (this is the whole driver of the taxonomy):

| Axis | Symbol | Range | Role in the memory model |
|---|---|---|---|
| ISDF centroids | `n_rmu` (μ) | 400 – 100,000 | **THE preferred shard dim.** `n_rmu²·16 B` reaches ~160 GB at μ~10⁵ → cannot live on one device → any `(μ,μ)` array is *always* sharded on both mesh axes. |
| Bands | `nb` | 50 – 10,000 (≈ n_rmu/10) | shard/chunk axis; `n_rmu×nb` arrays shard on ONE mesh axis. |
| k / q points | `nk`, `nq` | 1 – few thousand | usually replicated; becomes the shard axis only in the W-solve (q-parallel) and SC eigh. |
| Real-space FFT grid | `n_rtot = nx·ny·nz` | 10⁴ – 10⁷ | **NEVER materialized whole on one device.** Processed in `chunk_r` flat-r slabs. |
| G-vectors / sphere | `ngkmax` | sphere size | the *reduction* axis of V_q; replicated, chunked over `g_chunk`, never sharded. |
| spinor | `ns` | 1 or 2 | replicated; multiplies pair-density memory as `ns²`. |

Mesh: a single 2-D device grid `mesh_xy` with axis names `('x','y')`, built once
(`gw_jax.py:71-77`, `main` `:124`) by **most-square factorization subject to exact
divisibility** — `gx = floor(√P)` decremented until `P % gx == 0`, giving
`p_x = gx` (rows, `'x'`), `p_y = P/gx` (cols, `'y'`). `P = p_x·p_y`. There is exactly ONE
mesh; all stages run under it.

**The two sharding grades — and the rule that picks between them:**
- **single-axis** (`'x'` or `'y'`, ≈ √P ranks): used for `n_rmu × nb` objects (the four ψ
  copies) and for one leg of an `(μ,μ)` tile. μ on `'x'` for the "left/conj" copies, μ on
  `'y'` for the "right" copies.
- **combined-axis** (`('x','y')`, all P ranks): used for the `N_rmu²` objects where μ and ν
  *each* take one mesh axis (`P(None,'x','y')` ⇒ μ on x, ν on y ⇒ together all P), AND for
  the single-index objects too big for √P (ψ(G) band axis, ζ μ-axis on disk, Green's
  function μ-axes, Z_col r-axis, W-solve q-axis).

> **Why ψ(G) is √P-ish while G/W/Σ_μν are all-P** (the question the prompt flags): ψ carries
> `n_rmu × nb` — one factor of μ — so splitting μ over a single axis already gives
> `n_rmu·nb/√P` per rank, which fits. The Green's function, V_q, W_q, χ₀, Σ_μν all carry
> `n_rmu²` — two factors of μ — so **both** mesh axes must be spent (one on each μ leg) to
> reach `n_rmu²/P` per rank. Spending only one axis would leave `n_rmu²/√P` ≈ 20 GB+ at
> production μ and OOM. This is the single most important invariant in the code.

---

## 1. Per-array sharding law

Canonical specs are **centralized** so a mismatch fails at import, not at HLO compile:
the four ψ copies and the five intermediate-tensor specs are module constants in
`gw/wavefunction_bundle.py:91-125` (rationale `WB:96-101`). Quote the actual `P(...)` — do
not paraphrase.

### 1a. Wavefunctions (the boundary bundle out of A1/A2)

| Array | Index signature | PartitionSpec (verbatim) | Mesh mapping | Size regime forcing it |
|---|---|---|---|---|
| `psi_xn` | `(nk, s, μ_X, n)` | `P(None,None,'x',None)` `WB:91` | μ on **x** (√P), bands replicated | `n_rmu×nb` fits on one axis; G/χ₀ LHS-conj |
| `psi_xr` | `(nk, n, s, μ_X)` | `P(None,None,None,'x')` `WB:92` | μ on **x** | Σ-projection LHS-conj |
| `psi_yr` | `(nk, n, s, μ_Y)` | `P(None,None,None,'y')` `WB:93` | μ on **y** | G/χ₀ RHS |
| `psi_yn` | `(nk, s, μ_Y, n)` | `P(None,None,'y',None)` `WB:94` | μ on **y** | Σ-projection RHS |
| `enk`, `occ` | `(nk, nb_full)` | `P(None,None)` `WB:239-241` | **replicated** | tiny; occ built in host numpy (`WB:178-198`, jnp cost 1.79s→0.18s) |

Four copies exist so every contraction finds μ pre-positioned on the correct axis with the
spinor index `s` **adjacent** to μ (contiguous `(s,μ)` sweeps, `WB:1-14`); the transposes
that derive them preserve μ's sharding so **no cross-device reshard is emitted**
(`WB:217-219`).

### 1b. ψ(G) coefficients and their host store

| Array | Index signature | Spec | Residency |
|---|---|---|---|
| ψ(G) load result | `(nk, nb_padded, ns, ngkmax)` | `P(None,('x','y'),None,None)` `wfn_loader.py:499` | DEVICE, band on **all P** (only single-index array that needs all-P) |
| `PsiGStore` host tiles | per-cell `(nk, nb_local, ns, ngkmax)` | consumed as `P(None,('x','y'),None,None)` `psi_G_store.py:58` | **HOST numpy**, one tile per `(x,y)` cell |
| `box_index_dev` (FFT-box gather table) | `(nk,nx,ny,nz)` i32 | `P(None,None,None,None)` `wfn_loader.py:468` | DEVICE **replicated** (cached per `(k,mesh)` to kill a 1.3 GB/rank leak) |

ψ(G) lives on host band-sharded and is pulled one band-chunk at a time via `io_callback`;
it is **never a jit argument** (§4, §5). Rationale: G-flat host tile is ~6–11% of the FFT
box, "~28 GB/process vs g_box ~400 GB, ~14× smaller" (`psi_G_store.py:1-27,405-412`).

### 1c. ζ (ISDF interpolation vectors) — the shard-axis migrates across the pipeline

| Array | Index signature | Spec (verbatim) | Mesh mapping | Stage |
|---|---|---|---|---|
| `psi_*_rmu_Y` (fit input) | `(nk, nb, ns, n_rmu)` | `P(None,None,None,'y')` `isdf_fitting.py:181` | μ on **y** | fit ingest |
| `psi_*_rmuT_X` (fit input) | `(nk, n_rmu, nb, ns)` | `P(None,'x',None,None)` `isdf_fitting.py:184`, `core.py:288` | μ on **x** | fit ingest |
| `C_q` (CCT metric) | `(nq, n_rmu, n_col)` | `P(None,'x','y')` `core.py:290` | μ_x, ν_y (all P) | A2 metric |
| `Z_q` (ZCT) | `(nq, n_rmu, n_zchunk)` | `P(None,'x','y')` `core.py:549` | μ_x, r-chunk on **y** | A2 |
| `L_q` (Cholesky factor) | `(nq, n_rmu, n_rmu)` | `P(None,'x','y')` `core.py:970,1060` | μ_x, ν_y (all P) | A2 solve |
| `zeta_q` (solve out) | `(nq, n_rmu, n_zchunk)` | `P(None,('x','y'),None)` `core.py:1168-1174` | **μ on all P**, r replicated | A2 → G-flat |
| `gflat_acc` (G-flat accum) | `(nq, n_rmu_padded, ngkmax)` | `P(None,('x','y'),None)` `isdf_fitting.py:796` | μ on all P, G replicated | A2 output |
| ζ on disk (`zeta_q_G`) | `(n_q_disk, n_rmu[/pad], ngkmax)` | read as `P(None,('x','y'),None)` `zeta_reader.py:275` | μ on all P | boundary |

The ζ shard axis deliberately walks x → (x,y) → all-P as it moves from the Cholesky solve
(where μ_x/ν_y is natural) to the G-flat FFT accumulate (where the downstream per-rank
cuFFT wants each rank to own a μ-slab over the full r/G extent, `core.py:1168-1174`).

### 1d. V_q (A3) — the μ↔ν split made explicit

| Array | Index signature | Spec (verbatim) | Mesh mapping |
|---|---|---|---|
| ζ̃_L per-q view | `(n_rmu_L, ngkmax)` | `P('x',None)` `v_q_g_flat.py:99` | μ_L on **x** |
| ζ̃_R per-q view | `(n_rmu_R, ngkmax)` | `P('y',None)` `v_q_g_flat.py:100` | ν_R on **y** |
| `v(q+G)` kernel | `(ngkmax,)` | `P(None)` `v_q_g_flat.py:84,101` | **replicated** (reduction axis) |
| `V_q` output block | `(n_rmu_L, n_rmu_R)` | `P('x','y')` `v_q_g_flat.py:81` | μ_x, ν_y |
| `V_acc` (all-q) | `(n_q_ibz, μ_L, μ_R)` | `P(None,'x','y')` `v_q_g_flat.py:80` | q replicated, μ_x, ν_y |
| `g0` head | `(n_q_ibz, μ_L)` | `P(None,'x')` `v_q_g_flat.py:82` | μ on x only |

The contract `V_q[μ,ν]=Σ_G conj(ζ̃_μ)·v(q+G)·ζ̃_ν` reshards ONE ζ buffer onto `'x'` and the
other onto `'y'` (`v_q_g_flat.py:99-100`) so the GEMM naturally lands `V(μ_x, ν_y)`. It
chunks over **G** (a fixed-cost reduction axis), never over μ/ν — "the V[μ,ν] output is the
whole problem at once" (`v_q_g_flat.py:9-12`).

### 1e. Green's function, χ₀, W, Σ (A4/A5/A6) — the five centralized intermediate specs

| Array | Index signature | Spec (verbatim) `WB:106-125` | N_rmu²? → grade |
|---|---|---|---|
| G(k) 7-D FFT box | `(nkx,nky,nkz,s,μ_X,spinor,μ_Y)` | `G_FFT7D_SPEC = P(None,None,None,None,'x',None,'y')` | yes → all P (μ_x, ν_y) |
| G(k) flat-k | `(nk_flat,s,μ_X,spinor,μ_Y)` | `G_FLATK_SPEC = P(None,None,'x',None,'y')` | yes → all P |
| V_q / W_q 5-D | `(nkx,nky,nkz,μ_X,μ_Y)` | `V_FFT5D_SPEC = P(None,None,None,'x','y')` | yes → all P |
| χ₀(q) | `(nq,μ,μ)` | `CHI_Q_SPEC = P(None,None,None,'x','y')` | yes → all P |
| χ₀/W R-space | `(nk_flat,μ_X,μ_Y)` | `CHI_R_SPEC = P(None,'x','y')` | yes → all P |
| χ₀ accumulator `chi_R` | `(nk,μ,μ)` | `P(None,'x','y')` `w_isdf.py:99` | yes → all P |
| PPM poles B_q, Ω_q | `(nq,μ,μ)` | `P(None,'x','y')` `ppm_sigma.py:139-141` | yes → all P |
| σ^τ (re, im) band-space | `(nk, m_X, n_Y)` | `P(None,'x','y')` `ppm_sigma.py:468-469` | nb² → all P |
| Σ_c(ω,k,m,n) accum | `(nω,nk,nb,nb)` | `P(None,None,'x','y')` `ppm_sigma.py:962,1057` | nb² → m_x,n_y |
| **Σ_SX/COH/X, V_H (band)** | `(nk,nb_sigma,nb_sigma)` | `P(None,None,None)` `cohsex_sigma.py:208` | small (≲tens MB) → **replicated** |
| kin_ion, H_qp, QSGW Σ_xc | `(nk,nb,nb)` | `P(None,None,None)` `sc_iteration.py:163`, `qsgw_utils.py:254` | small → **replicated** |

The rule for the band-space (`m,n`) Σ objects: they are `nb²` not `n_rmu²`, ~100× smaller,
so once contracted out of the centroid basis they are pinned **fully replicated** — "the
heavy ω-grid Σ_c tensor stays sharded upstream and is only collapsed after the
energy-domain contraction" (`cohsex_sigma.py:192-198`). The one exception is the streamed
dynamic Σ_c(ω,k,m,n), kept sharded `P(None,None,'x','y')` end-to-end because ω makes it
large again (`ppm_sigma.py:934-942`).

### 1f. W-solve and SC transient q/k sharding

| Array | Spec (verbatim) | Mapping | Why |
|---|---|---|---|
| W-solve `V_q`,`chi_q` | `q_shard = P(('x','y'),None,None)` `w_isdf.py:243` | **nq on all P**, each rank whole `(μ,μ)` | q-parallel LU, "one all-gather + one all-scatter of (μ,μ) blocks" `w_isdf.py:366` |
| SC eigh input | `k_shard_3d = P(('x','y'),None,None)` `sc_iteration.py:183` | nk on all P | one eigh per k, independent |

### 1g. Symmetry tables (B4) — replicated host, baked into HLO

All `SymMaps` tables (`sym_perm`, `L_table`, `U_spinor`, `R_proper`, `irr_idx`/`sym_idx`,
`kq_map`, …) are plain **replicated host numpy**, never sharded at construction
(`symmetry_maps.py:912-1048`). They reach the device only when a consumer device_puts a
derived table **replicated** (`wfn_loader.py:592-609`), or are **baked into the jit closure
as HLO constants** in `unfold_v_q` — "runtime-arg form was ~2× slower per call"
(`symmetry_maps.py:312-318`).

---

## 2. The "never materialize on one process" set

Anything carrying `n_rmu²` or `n_rtot` exceeds a single device at production scale and is
therefore **always sharded and/or chunked**. The thresholds, in budget terms:

| Object | On-one-device size | Verdict | Mechanism |
|---|---|---|---|
| `(μ,μ)` arrays: V_q, W_q, χ₀, χ_R, L_q, C_q, G_FFT7D, Σ_μν | `n_rmu²·16 B` → ~160 GB at μ=10⁵ | **NEVER whole** | always `P(…,'x','y')` — both μ legs sharded (all P) |
| FFT box ψ(r): `(…, nx,ny,nz)` | `n_rtot·16 B` per band, ×nb | **NEVER whole** | `chunk_r` flat-r slabs + local-FFT shard_map (§3) |
| ψ(G) full store | `nb·ns·n_rtot·16 B` → ~400 GB CrI3 | **NEVER on device** | host `PsiGStore`, per-bc `io_callback` (§5) |
| G-flat accumulate FFT box | `cs·n_rtot·16 B` | bounded by cap | `gflat_chunk_size ≤ 100` scan chunk (§3) |
| Σ_c(ω,k,m,n) | `nω·nk·nb²·16 B` | sharded, not replicated | `P(None,None,'x','y')` + ω streaming |

The **all-P rule restated as a gate:** if an array's index signature contains μ *twice*
(or m,n from a μ-double-contraction), it MUST spend both mesh axes. If it contains μ once,
it spends one axis (√P). If it is `nb²` or smaller in band space, it is replicated. A
future contributor adding a stage checks their array against exactly this trichotomy.

**Padding is what makes the all-P rule safe:** `n_rmu_padded = round_up(n_rmu, P)` — rounded
to the **product** of all mesh axes, "the worst-case divisor: any single- or product-axis
PartitionSpec on the μ dim divides cleanly" (`meta.py:130-133`, `runtime/padding.py:128-144`).
Pad μ-rows are exact zeros so every bilinear form (M, C_q, Σ) sees zero contribution
(`core.py:772-776`; Cholesky adds identity only on the pad block, `√1=1` exactly, not a
ridge, `core.py:781-791`). Disk always stores the **logical** (unpadded) extent via
SlabIO `valid_shape` so any process count can re-read (`runtime/padding.py:1-29`).

---

## 3. Chunk-vs-materialize rules (which big axis is chunked, and what sets the size)

| Big axis | Chunk knob | Where size is set | Rule |
|---|---|---|---|
| `n_rtot` (FFT grid) | `chunk_r` (cr) | planner (§ below) | **always chunked**; cr ≥ min(n_rmu, n_rtot), divisible by P |
| `nb` (bands) | `band_chunk` | planner; user cap `gw_config.py:248` (default 16) | chunked in the ζ centroid-load IFFT and the ZCT scan |
| `nq` (q-points) | `q_chunk`/`q_gather` (ζ solve/write), q-parallel (W) | legacy planner `gw_init.py:376` | ζ solve loops q in Python; W-solve shards q on all P |
| `ngkmax` (G) | `g_chunk` (V_q) | `_pick_g_chunk(ngkmax, ≤4096)` `v_q_g_flat.py:224` | V_q GEMM chunks over G; `lax.scan` not unroll |
| G-flat FFT box | `gflat_chunk_size` (cs) | planner, **cap 100** `gflat_memory_model.py:136` | past cs~1000 cuFFT changes plan → non-linear scratch OOM |
| ω (frequency) | `omega_batch_size` (default 4) | `ppm_sigma.py:1402` | stream mode chunks ω for the Σ_c host buffer |

### The r-chunk loop — the crux of ISDF performance

`n_rtot` is never materialized; the ζ-fit sweeps it in `num_chunks = ceil(n_rtot/chunk_r)`
flat-r slabs (`isdf_fitting.py:243,842-933`). Per chunk, the pipeline is:

```
ψ(G) host tile ──io_callback(per-bc)──▶ IFFT(local) ──all_gather(bands,'x','y')──▶ ψ(r-chunk)_Y
   [core.py:633-636]        [core.py:644-647]         [core.py:652-654]
      │  (IFFT-FIRST: gather-first would blow the FFT box to ~80 GB/rank — core.py:400-403)
      ▼
   pair density P_l/P_r  ──γ̃-contract──▶  Z_q(μ_x, r-chunk_y)  ──solve──▶  ζ(r-chunk)
   [core.py:693-699]                       [core.py:710-731]     [solve_zeta]
      ▼
   accumulate_rchunk_to_gflat:  FFT(r-chunk → G-sphere), += into gflat_acc(μ, G)   [isdf_fitting.py:901-909, donated]
```

`chunk_r` **dominates** because it is the multiplier on the five concurrent-live moments of
the fused `fit_one_rchunk` jit — the pair-density accumulators `(nk,ns,ns,μ,cr)`, their
R-space IFFTs, the pre-reshard `Z_q`, all scale linearly in cr (`gw_init.py:80-120`;
`gflat_memory_model.py:317-380`, "usually the binding peak"). The lower bound `cr ≥ n_rmu`
is a *performance* floor, not memory: the Σ_μν output is `n_rmu²·nq·16 B`, so doing fewer
than n_rmu rows of r-work per chunk wastes the fixed per-chunk pipeline overhead
(`gflat_memory_model.py:37-39`). Empirically per-chunk wall is dominated by the spatial FFT
shape not the batch, so small `gflat_chunk_size` costs little (cs=1 within 15% of cs=360,
`gflat_memory_model.py:783-785`) — which is why the cs≤100 cap is free.

**The two planners (C6 — still stacked, MAP.md §4 #1):**
- Legacy `compute_optimal_chunks` (`gw_init.py:154-404`): a **5-moment closed-form** model.
  Each moment is `base + α·cr + c`; invert to `cr ≤ (headroom−c)/α` per stage, take the min
  (`gw_init.py:123-150`). The α's carry their shard divisors explicitly: `α_pair` shard
  `p_x·p_y`, `α_psi_Y_bc` shard `p_y`, `α_zcol`/`α_z_slice` shard `p`, `c_solve` shard
  `p_x·p_x` (`gw_init.py:65-77`). Divisibility: `cr -= cr % (p_x·p_y)` "cr must be divisible
  by p_total for solve sharding" (`gw_init.py:314-316`). Picks `band_chunk`, `q_chunk`,
  `k_chunk`.
- Production `plan_gflat_chunks` (`gflat_memory_model.py:565-922`): **5 HBM peaks A–E**
  (centroid load / CCT+Cholesky / fit_one_rchunk / accumulate / V_q per tile), r-first
  picker, and it **OVERRIDES** the legacy `band_chunk` and `chunk_r`; it alone picks
  `gflat_chunk_size` (`gw_init.py:521-527`). Called with hard-coded
  `target_utilization=0.80` for bispinor 4-channel slack (`gw_init.py:504`); legacy uses
  0.97. Backend-dependent slot count: 3 concurrent pair-density slots on GPU, 4 on CPU
  (`gflat_memory_model.py:75-101`).

The FFT-box scratch is queried **exactly** from XLA (`query_fft_peak_bytes`) rather than
guessed, on the unsharded 6-D shape `(nk, band_chunk, ns, nx, ny, nz)` sharded
`P(None,('x','y'),None,None,None,None)` — dropping the nk factor once under-predicted a Si
10×10×10 peak by ~19 GiB (`gw_init.py:328-360`).

---

## 4. Restaging / redistribution rules (staged reshards, and the intermediate that would blow up)

The recurring failure mode is **"Involuntary Full Rematerialization"**: asking SPMD to move
*both* mesh axes of an `(μ,μ)` array at once makes XLA rebuild the full unsharded array on
every device (≈ P× the shard). Every staged reshard below exists to keep each hop a
**single-axis all-to-all**.

### 4a. Z_q → Z_col (ζ solve), staged via `P('x',None,'y')`
FROM `P(None,'x','y')` TO `P(None,None,('x','y'))`, staged through `P('x',None,'y')`
(`core.py:1364-1391`, `donate_argnums=(0,)`). Direct would hit Involuntary Remat.
**Measured:** Si 4×4×4 60Ry direct-no-donate 31.14 GB/dev vs staged+donate 15.57 GB (HLO
peak 68.94 → 29.94 GB). Staging parks `'x'` on the leading nq axis so each hop moves one
axis (`core.py:1364-1385`).

### 4b. ζ solve-out → G-flat μ-layout, TWO-STEP through `P(None,'x','y')`
`_reshard_zeta_r_XY_to_mu_XY` (`core.py:1095-1112`): the triangular solve lands ζ at
`P(None,None,('x','y'))` (parallel over r-columns); the downstream FFT wants μ-sharded. Move
`'x'` (r→μ) then `'y'` (r→μ) separately, because **"SPMD's all-to-all planner only handles
one mesh axis at a time"** (`core.py:1099-1108`).

### 4c. W-solve staging `P(None,'x','y') → P('x',None,'y') → P(('x','y'),None,None)`
Nested `with_sharding_constraint` (`w_isdf.py:243-246`). Routing through `P('x',None,'y')`
(x parks on nq, y stays on μ₂) makes it two single-axis all-to-alls. **Measured:** via
rep_3d 2.95 GB/dev (temp 2.21) — Involuntary Remat; via `P('x',None,'y')` 1.11 GB/dev (temp
0.37) — 62% reduction (`w_isdf.py:215-223`). χ_flat is donated; V_flat is NOT (reused by
COHSEX Σ_SX/COH/X and the PPM `Wc = W − V`, `w_isdf.py:226-231`).

### 4d. Σ_μν → Σ_mn (centroid basis → band basis) — the staged double reduce-scatter
This is the redistribution the prompt calls out: even the *smaller* band-space `Σ(m,n,k)`
can overflow a rank if built by a naive all-gather, so the projection is done as a
`shard_map` with two staged `psum_scatter` collectives (`ppm_sigma.py:425-496`):

```
in:  ψ_xr  P(None,None,None,'x')      σ  P(None,None,'x',None,'y')      ψ_yn  P(None,None,'y',None)
Stage 1:  einsum('kmsx,ksxty->kmty')  then  psum_scatter('x', scatter_dim=m)   # reduce μ_X AND scatter m onto x
Stage 2:  einsum('kmty,ktyn->kmn')    then  psum_scatter('y', scatter_dim=n)   # reduce μ_Y AND scatter n onto y
out: (re, im) each (nk, m_X, n_Y) at P(None,'x','y')
```

Same NCCL byte volume as the two implicit psums, but the output is *born sharded* `(m_X,
n_Y)` so every downstream `coeff·σ` stays local; the per-rank buffer is
`(nb/p_x)(nb/p_y)·nω·nk` — "~100× smaller than Σ_μν, the whole scaling argument for shipping
this layout end-to-end" (`ppm_sigma.py:934-942,450-457`). Requires `m%p_x==0 && n%p_y==0`.
The band-space Σ_c then stays sharded until the QSGW Hermitisation, where it is forced
replicated `P(None,None,None)` "before Hermitisation, avoids a sharded transpose"
(`qsgw_utils.py:281-285`). COHSEX's static Σ_mn is small, so it skips the reduce-scatter and
is just pinned replicated (`cohsex_sigma.py:192-198`).

### 4e. V_q IBZ → full-BZ unfold — volume-preserving all_to_all, never a full μ/ν on one rank
`unfold_v_q` (`symmetry_maps.py:392-470`) is a `shard_map` (`in/out_specs=P(None,'x','y')`)
whose μ-permute (on `'x'`) and ν-permute (on `'y'`) are each a `take_along_axis` sandwiched
by two `lax.all_to_all` that split the *other* spatial axis. **"Never exceed 1× single-tile
per rank … at no point does any rank hold a full μ or ν axis (which would be Px× or Py× the
single-tile memory)"** (`symmetry_maps.py:373-380`); per-rank volume stays
`n_q·μ·ν/(Px·Py)`. Requires `n_rmu_padded % (Px·Py) == 0`.

### 4f. ζ centroid-load reshard, staged via `P(None,'y',None,None)`
`load_centroids_band_chunked._reshard_all` (`wfn_transforms.py:1834-1850`): pad μ → stage
`P(None,'y',None,None)` → `P(None,None,None,'y')` → conj+transpose → `P(None,'x',None,None)`
— "a single all-to-all on the band axis before the second all-to-all onto the n_rmu axis."

### 4g. Cholesky staging into 2-D block tiles
`factor_c_q` (`core.py:1051-1063`): dense `P(None,'x','y')` → `dense_to_tiles` →
`P(None,'x','y',None,None)` (tile-row on x, tile-col on y, each `b×b` replicated) → 2-D
right-looking blocked Cholesky (`cholesky_2d.py:96-228`, psum-broadcast of the diagonal and
panel blocks) → `tiles_to_dense` → `P(None,'x','y')`. The module header states the point:
this "avoids the involuntary full rematerialization … 2D blocked: 5 MB/device vs reshard
strategy 1.6 GB/device (may OOM)" and needs "√P less bandwidth than 1D"
(`cholesky_2d.py:1-30`). Divisibility: `n%b==0`, `J%Px==0`, `J%Py==0` with `J=lcm(Px,Py)`
(`cholesky_2d.py:57,125`; NRHS padded to a multiple of `Py`, `core.py:1190-1196`).

### 4h. Host round-trips and replicated-broadcast avoidance
- ψ(G) per-bc pull via `io_callback(..., ordered=False)` inside the scan body — the host
  tiles must stay alive for the whole enclosing jit, freed only after `block_until_ready`;
  passing the store as a jit arg would break the aliasing/free lifecycle
  (`core.py:633-636`; `psi_G_store.py:346-351`).
- Replicated tables are `jax.device_put(numpy_array, replicated_sharding)` **directly** — a
  `jnp.asarray` wrap "would force single-device staging that turns device_put into an
  all-reduce" (`isdf_fitting.py:825-829`, `qsgw_utils.py:255-256`, `wfn_loader.py:470-475`).
- Slab I/O (`_slab_io_ffi`): each rank writes its own hyperslab via collective MPI-IO under
  a padded-memory / logical-disk contract (`valid_shape` clips the tail rank); the
  `_slab_io_allgather` fallback (CPU/no-CUDA) instead `process_allgather`s the FULL array to
  rank 0 (materializes the whole thing — acceptable only off the GPU path)
  (`_slab_io_ffi.py:1-11`; `_slab_io_allgather.py:1-15`).

### 4i. Stage-boundary hygiene in the driver
`gw_jax.main` frees each stage's big arrays before the next: `del chi0_q` after the IBZ
slice (`gw_jax.py:255`), `del chi0_q_solve` after the donated W-solve (`:268`), `del
W_q_solve` after unfold (`:286`), `del ppm_outputs` (`:456`), `gc.collect()` before the
sigma section (`:357`). The explicit `block_until_ready` calls are load-bearing — they drop
the last reference so XLA may **donate** the buffer; a `.watch()` would keep the reference
alive and block donation (`gw_jax.py:230-236`).

---

## 5. Invariants a contributor (or model) must preserve to add a stage without OOMing

1. **Trichotomy by index signature.** μ appears twice (`n_rmu²`, or `nb²` from a
   μ-double-contraction) → shard **both** axes `P(…,'x','y')`. μ appears once (`n_rmu×nb`) →
   shard **one** axis (μ_x for left/conj copies, μ_y for right). Band-space `nb²` or smaller
   → **replicate** `P(None,None,None)`. (`WB:91-125`, `cohsex_sigma.py:192-198`.)

2. **Never materialize `n_rtot` or `n_rmu²` whole.** `n_rtot` is always swept in `chunk_r`
   flat-r slabs with a **local-FFT shard_map** (band axis stays sharded, FFT axes
   replicated) — a plain sharded `jnp.fft.ifftn` inserts an all-gather and OOMs (the 121 GB
   `to_rmu` regression, `wfn_transforms.py:275-284`). `n_rmu²` arrays are born sharded on
   both axes.

3. **ψ(G) lives on host, is pulled per band-chunk via `io_callback`, and is never a jit
   arg.** Keep the store alive across the enclosing jit; free only after
   `block_until_ready`. (`psi_G_store.py`, `core.py:633-636`.)

4. **Pad μ (and bands) to `round_up(·, P)` with zero pad rows; store logical extent on
   disk.** Pad zeros must be math-neutral (bilinear forms, `√1=1` Cholesky pad-block, masked
   G-slots). (`meta.py:130-133`, `runtime/padding.py`, `core.py:772-791`.)

5. **Never reshard both mesh axes of an `(μ,μ)` array in one hop.** Stage through a
   one-axis-at-a-time intermediate (`P('x',None,'y')` for the μ,μ case) and **donate** the
   input. Direct = Involuntary Full Rematerialization ≈ P× blow-up. (§4a–4d,
   `core.py:1364-1385`, `w_isdf.py:215-223`.)

6. **Project centroid→band with the staged double `psum_scatter`, not an all-gather.** Even
   the smaller band-space Σ can overflow if gathered; born-sharded `(m_x,n_y)` keeps
   downstream local and ~100× smaller. (`ppm_sigma.py:425-496`.)

7. **Unfold IBZ→full-BZ volume-preservingly** (all_to_all splitting the *other* axis); never
   let a rank hold a full μ or ν axis. Sym tables are replicated host numpy baked into HLO,
   not sharded device arrays. (`symmetry_maps.py:373-380,312-318`.)

8. **Chunk sizes come from the planner, respect divisibility, and stabilize the compile
   cache.** `chunk_r % P == 0`, `band_chunk` a power-of-2 bumped to the mesh floor,
   `gflat_chunk_size` a multiple of 4 ≤ 100. Distinct chunk shapes = distinct XLA compiles,
   so the planners round to stable divisors. (`gw_init.py:314-316`,
   `gflat_memory_model.py:661-668,724-726,790-803`.)

9. **`device_put(numpy, replicated)` directly** for replicated tables/scalars; a `jnp.asarray`
   wrap turns it into an all-reduce. Free stage arrays with `del`+`block_until_ready` to
   enable donation. (`isdf_fitting.py:825-829`, `gw_jax.py:230-236`.)

---

## 6. Flagged / not-fully-explained

- **C6 is still two stacked planners** (MAP.md §4 #1): the legacy 5-moment
  `compute_optimal_chunks` and the production 5-peak `plan_gflat_chunks` both run; the gflat
  plan wins for `band_chunk`/`chunk_r`, legacy still owns `q_chunk`/`k_chunk`. The two use
  *different* target-utilization defaults (0.80 vs 0.97) and *different* memory models for
  the same peaks — a genuine divergence, not just duplication.
- **`c_solve` shard divisor `p_x·p_x`** (`gw_init.py:75`) is unusual — it is the replicated-L
  triangular-solve overhead, modeled as sharded on `p_x²` plus 3 fully-replicated `(μ,μ)`
  slabs. The `p_x·p_x` (not `p_x·p_y`) divisor is asserted by the model but not
  independently re-derived here; flag for verification if the mesh is strongly non-square.
- **`n_bc` unroll cost diverges between the two planners.** Legacy multiplies the FFT moment
  by `n_bc` because it models a Python-unrolled bc-loop (`gw_init.py:88-98`); the gflat model
  dropped the `n_bc` factor after the loop became `lax.scan(unroll=1)` (Round-6,
  `gflat_memory_model.py:323-326`). If the legacy path is ever the binding planner for
  `chunk_r`, it will over-predict; in production gflat overrides `chunk_r` so this is latent.
- **V_q q-loop is strictly sequential** (`v_q_g_flat.py:422-435`, one q with
  `block_until_ready` each) — a deliberate simplicity choice; q-batching is left to a future
  outer vmap. Not a memory rule, but a perf ceiling worth noting for large `nq`.
