# Agent 4 — zeta-fit r-chunk memory model, from-scratch derivation

**Scope:** `common/isdf_fitting.fit_zeta_to_h5` and its callees, post-2026-05-11 G-flat /
monolithic-shard_map state. Read-only on `sources/lorrax_A/`. No HLO dumps consulted —
empirical constants flagged as such.

**Notation.** `P = p_x · p_y` (mesh size). All sizes in c128 bytes (`B = 16`).
`/X`, `/Y`, `/XY`, `/P` denote per-rank shard divisors. Charge / transverse channel
distinguished by `μ_C ≡ n_rmu_C` vs `μ_T ≡ n_rmu_T`; both have the same `n_rmu_padded`
recipe (round up to a multiple of `P`).

---

## 1. Tensor catalog

I split tensors by lifetime relative to the per-channel call to `fit_zeta_to_h5`:

```
LIFETIMES
  L0  caller-resident, alive across all channels   (centroid ψ inputs)
  L1  alive end-to-end inside fit_zeta_to_h5       (one channel)
  L2  alive only during the pre-loop CCT/factor    (Peak B)
  L3  alive only inside one r-chunk fused jit       (Peak C)
  L4  alive only inside accumulate_rchunk_to_gflat  (Peak D)
  L5  alive only inside the pre-loop centroid load  (Peak A) — sibling channel
```

### 1.a Persistent (L0/L1) — `B_persist`

| name | shape | sharding | shard | size / rank | source |
|---|---|---|---|---|---|
| `psi_rmu_Y` | `(nk, nb_full, ns, μ)` | `P(None,None,None,'y')` | `p_y` | `B·nk·nb·ns·μ / p_y` | `fit_zeta_to_h5:1466,2014` (caller's bundle) |
| `psi_rmuT_X` | `(nk, μ, nb_full, ns)` | `P(None,'x',None,None)` | `p_x` | `B·nk·μ·nb·ns / p_x` | `fit_zeta_to_h5:1467` |
| `psi_*_*_fit` views | same as above | views | — | 0 (slices, not copies) | `:1602-1620` |
| `norms_l/r` | `(nb,)` f64 | replicated | 1 | `8·nb` | `:1614` (negligible) |
| `L_q` | `(nq, μ, μ)` | `P(None,'x','y')` | `P` | `B·nq·μ² / P` | `:1703-1706` |
| `cct_trace_per_q` | `(nq,)` | replicated | 1 | `B·nq` (negligible) | `:1718` (transverse only) |
| `gflat_acc` | `(nq_disk, μ_pad, ngkmax)` | `P(None,('x','y'),None)` | `P` | `B·nq_disk·μ_pad·ngkmax / P` | `:2079-2087` |
| `_gflat_sphere_idx_padded` | `(nq_disk, ngkmax)` i32 | replicated | 1 | `4·nq_disk·ngkmax` (tens of MB) | `:1846` |
| `_q_irr_frac_dev` | `(nq_disk, 3)` f64 | replicated | 1 | `24·nq_disk` (negligible) | `:2112` |
| `psi_G_store` host tiles | `(nk, nb/P, ns, ngkmax)` | host (not on device) | — | **not on device** | `psi_G_store.py:107` |

Note 1: `psi_rmu_Y` and `psi_rmuT_X` carry the **full** `nb_full` band range (≥ `nb_left + nb_right`); the L/R band-range views (`:1602-1605`) are just slices into them, no new bytes. The same buffers are reused by `gw.wavefunction_bundle.build_wavefunctions` after the fit (`:1539-1542`) — that's the L0 lifetime.

Note 2: `gflat_acc` accumulates *across r-chunks*, not within one. It's allocated once
before the loop (`:2082`) and donated to `accumulate_rchunk_to_gflat` each iter
(`:2172`). So it's L1 for the duration of this channel, then re-allocated for the next.

### 1.b Pre-loop CCT / factor (L2 — Peak B, transient)

Inside `c_q_from_psi_sm._local` (`:301-351`):

| name | shape | per-rank | notes |
|---|---|---|---|
| `psi_l_X_`, `psi_l_Y_`, `psi_r_X_`, `psi_r_Y_` | per-rank slices of L0 inputs | — | aliases of `B_persist`'s ψ |
| `P_l` (`karmb`) | `(k, ns_l, μ, μ, ns_r)` rank-5 | `B·nk·ns²·μ²/P` | `:318` — full-grid pair density! |
| `P_l_3d` | reshape of `P_l` rank-7 `(kx,ky,kz,…)` | bitcast (no new bytes) | `:324` |
| `P_l_R` | `ifftn(P_l_3d)` | one new rank-7 (cuFFT out) | `:326` |
| `P_l_R_conj` | conj of `P_l_R` | one new rank-7 | `:327` |
| `P_r` rank-5 | same shape as P_l | `B·nk·ns²·μ²/P` | `:320` |
| `P_r_3d` / `P_r_R` | analogous | one new rank-7 | `:329-331` |
| `C_R` (rank-5 reduced) | `(kx,ky,kz, μ, μ)` | `B·nk·μ²/P` | `:335` |
| `C_q_3d` | `fftn(C_R)` | same shape as `C_R` | `:344` |
| `C_q` | `(nq, μ, μ)` `P(None,'x','y')` | `B·nq·μ²/P` | persistent through Cholesky |

Code explicitly `del`s `P_l` after `P_l_3d` (bitcast), `P_l_3d` after `P_l_R`, etc.
After all dels, the **simultaneously live** rank-5 pair-density-shaped tensors at the
moment of `gamma_double_contract` are:

- `P_l_R_conj`
- `P_r_R`
- (XLA's cuFFT scratch + the contract output `C_R` materialising)

This is the source of the magic `pair_density_slots = 3`. Lifetime overlap is encoded
by `del` ordering, not by aliasing intent.

Then `L_q` is built by cuSolverMp `potrf` (charge) or `getrf` (transverse). cuSolverMp's
internal scratch is **not** modelled — it is on the order of `n_rmu²` per rank for
2D-blocked algorithms (one block + workspace), so I assume `≤ 2 · B_Lq_per_rank`. Flag.

Then `del C_q, C_q_flat` (`:1726`) and `jax.clear_caches()` (`:1729`) — XLA reclaims
the pair-density transients before the r-chunk loop.

### 1.c r-chunk fused-jit body (L3 — Peak C)

Inside `_make_fit_one_rchunk_kernel` (factory; not directly read but reconstructed from
`fit_one_rchunk` docstring `aot_memory_model/kernels/fit_one_rchunk.py:24-37` and the
visible structure of `z_q_from_psi_sm._local`):

**Per band-chunk (`bc`, loop unrolled in Python):**

| name | shape | per-rank | notes |
|---|---|---|---|
| `psi_G_bc` (io_callback) | `(nk, bc/P, ns, ngkmax)` on `P(None,('x','y'),None,None)` | `B·nk·bc·ns·ngkmax/P` | `psi_G_store.py:322-345` |
| `psi_r_bc` (FFT box) | `(nk, bc/P, ns, nx, ny, nz)` | `B·nk·bc·ns·n_rtot/P · fft_factor` | `to_rchunk` → IFFT; **the empirically-unsharded one at CrI3** |
| `psi_l_Y_bc`, `psi_r_Y_bc` (resharded) | `(nk, bc, ns, r_chunk)` on `P(None,None,None,'y')` (some shape variant) | `B·nk·bc·ns·r_chunk / p_y` | one each for L, R |

**Once per r-chunk (`z_q_from_psi_sm`, then solve):**

| name | shape | per-rank | notes |
|---|---|---|---|
| `P_l` (`karmb`) | `(k, ns_l, r_chunk, μ, ns_r)` | `B·nk·ns²·μ·r_chunk / P` | `:425` |
| `P_l_3d` | rank-7 bitcast | same bytes | `:429` |
| `P_l_R` | ifftn out (rank-7) | `B·nk·ns²·μ·r_chunk / P` | `:431` |
| `P_l_R_conj` | conj | same | `:432` |
| `P_r`, `P_r_3d`, `P_r_R` | analogous | one extra rank-5/7 live at any time | `:425-436` |
| `Z_R` (γ̃-contracted) | `(kx,ky,kz, r_chunk, μ)` | `B·nk·μ·r_chunk / P` | `:438` |
| `Z_q_3d`, `Z_q` (rank-3) | `(nq, μ, r_chunk)` | `B·nq·μ·r_chunk / P` | `:447,449` |
| reshard intermediate `Z_q[q_X, μ, r_Y]` (shard_map fallback only) | same shape | `B·nq·μ·r_chunk / P` | `report.md §2b.4` |
| `ζ_chunk` (solve output) | `(nq_disk, μ, r_chunk)` `P(None,('x','y'),None)` | `B·nq_disk·μ·r_chunk / P` | `fit_one_rchunk:1414` (output) |

The dominant rank-5 *family* (lifetime-overlapping pair-density-shaped buffers) is
shape `c128[nk, ns², μ, r_chunk]` per rank divided by `P`. The empirical
`pair_density_slots = 3` was extracted from a MoS2 3×3 bispinor HLO dump
(`report.md §5.6`, `gflat_memory_model.py:175-179`).

### 1.d accumulate_rchunk_to_gflat (L4 — Peak D)

Inside `_kernel` (`wfn_transforms.py:617-667`):

| name | shape | per-rank | notes |
|---|---|---|---|
| `rchunk` (input ζ) | `(nq_disk, μ_pad, r_chunk)` on `P(None,('x','y'),None)` | `B·nq_disk·μ_pad·r_chunk / P` | passed in |
| `acc_flat` | flatten of `gflat_acc` | already counted (persistent) | `:620` |
| `sub` | `(cs, r_chunk)` | `B·cs·r_chunk` | `:641` |
| `buf` | `(cs, n_rtot)` | `B·cs·n_rtot` | `:653` — **zero-padded full FFT box** |
| `box = buf.reshape` | `(cs, nx, ny, nz)` | bitcast | `:655` |
| `G = fftn(box)` | same | `B·cs·n_rtot · (1 + cuFFT scratch)` | `:656` |
| `contrib = take_along_axis(G, sphere)` | `(cs, ngkmax)` | `B·cs·ngkmax` | `:657` |
| phase tables `phx, phy, phz` | `(nq_disk, n*)` | `~B·nq_disk·max(nx,ny,nz)` (tens of MB) | `:603` |

The shard_map runs **fully local per rank** (FFT axes are replicated within the shard).
No cross-rank collectives in the body. `buf` and `G` are per-call sized only by `cs`.

### 1.e Pre-loop band-chunked centroid load (L5 — Peak A, transient)

This is **not** in `fit_zeta_to_h5` — the caller (`gw_init.fit_zeta`) already produced
`psi_rmu_Y`, `psi_rmuT_X` before calling. So Peak A is bracketed *before* `fit_zeta_to_h5`
sees its inputs. From `load_centroids_band_chunked` (via `load_wfns.py`, not read in
detail): a band-chunked IFFT box `(nk, band_chunk_load, ns, n_rtot) / P` × `fft_factor`
sized for the entire `nb_full` range, plus the centroid output being filled.

For the *r-chunk planner*, Peak A constrains `band_chunk_size` only if a future redesign
shares state with the chunk loop. Today it doesn't — Peak A is upstream and its
band_chunk knob is independent.

---

## 2. Aliasing analysis — what overlaps in time vs. what XLA can alias

I split this into three lifetime windows.

### 2.a Across L1 (persistent during the whole r-chunk loop)

`B_persist = centroids (X) + centroids (Y) + L_q + gflat_acc + cct_trace`. These cannot be aliased — they are simultaneously live every iter. **Sum**, don't max.

### 2.b Across L3 vs L4 — fit_one_rchunk vs accumulate

`fit_one_rchunk` and `accumulate_rchunk_to_gflat` are **separate jit calls** with a
device sync between them (`zeta_chunk.block_until_ready()` at `:2152`). Their
transients live in *disjoint* XLA modules → XLA cannot alias L3 ↔ L4 transients.
Each gets `W_pool = B − B_persist` separately. **Max over (Peak C, Peak D)**.

### 2.c Inside L3 — the heart of the model

Within the fused fit_one_rchunk jit, the question is which transients overlap. From
the `del` annotations in `z_q_from_psi_sm._local`:

```
P_l live ── del P_l ──> P_l_3d / P_l_R live ── del P_l_R ──> P_l_R_conj live
       ┊                                                                 ┊
       ┊ P_r live ── del P_r ──> P_r_R live ─────────────────────────────┊─> γ̃-contract
                                                                         │   needs both
                                                                         ▼
                                                                   C_R / Z_R rank-5 (small)
                                                                         │
                                                                         ▼
                                                                   FFT → Z_q rank-3 (small)
```

At the moment `gamma_double_contract` is called, both `P_l_R_conj` and `P_r_R` are
simultaneously live — two distinct rank-5 pair-density-shaped buffers. XLA's allocator
typically reserves one more slot for cuFFT scratch / the contract output before the
post-FFT reduction has shrunk the working set. Hence empirical `pair_density_slots = 3`.

**Importantly**: the band-chunk FFT box `psi_r_bc` of shape
`c128[nk, bc/P, ns, n_rtot] · fft_factor` has a lifetime *inside* a single `bc` iter.
Once `psi_l_Y_bc` (and `psi_r_Y_bc`) are sliced/resharded out, `psi_r_bc` is dead. It
**does not** overlap `P_l`/`P_r` accumulation — they are in different unrolled-loop
iterations conceptually, but XLA holds them simultaneously *across* the bc-loop body
because it's a Python-unrolled trace. In practice the aot_memory_model's `psiG_bc`
primitive treats β=1 (one bc's `psi_r_bc` live at a time) — see
`aot_memory_model/kernels/fit_one_rchunk.py:180-188`. I trust this; it relies on XLA
recognising the lifetime constraint, which it does as long as the bc-loop is unrolled
straight-line in trace order.

`Z_q` and the reshard intermediates (`Z_q[q_X, μ, r_Y] → Z_q[q_X, μ, r_XY]` etc.,
report §2b.4) overlap the *tail* of the pair-density work. The §6 trap "donate Z_q on
the first reshard; Si 4×4×4 60 Ry: 31→16 GB/dev" is direct evidence that an
**un-donated** reshard doubles the Z_q footprint. So the cuSolverMp branch (potrs
output `ζ[q,μ_X,r_Y]` → single 'y' move) is materially cheaper than the shard_map
fallback at scale. **In the cuSolverMp branch**, `Z_q` (rank-3, `B·nq·μ·r_chunk/P`) is
small relative to a pair-density slot whenever `nk·ns² ≥ nq` (≥4 in bispinor) — i.e.
always.

### 2.d Aliasing summary

```
B_total_inside_one_rchunk_iter
    = B_persist
    + max( Peak_C_transient, Peak_D_transient )

Peak_C_transient
    = pair_density_slots · (B·nk·ns²·μ·r_chunk / P)     ← W_zeta (binding)
    + max( psi_r_bc_unsharded_or_sharded · fft_factor,  ← W_wfn (may NOT alias, see §5)
           ζ_chunk_alive_at_solve · 1 )

Peak_D_transient
    = cs · n_rtot · 16 · fft_factor + cs · ngkmax · 16   ← W_accum
    + ζ_chunk-in-from-C still live? (no — block_until_ready forced free)
```

The `max(psi_r_bc, ζ_chunk)` inside Peak_C is conjecture: I assume the per-bc FFT box
and the post-solve ζ_chunk don't co-live because they are at opposite ends of the
fused trace, but I have **not verified this from HLO**. Open question.

---

## 3. Budget equations

Let `B = memory_per_device_gb · 1e9` and let `T = fft_factor` (= 4.0 empirical, ±10%).
Let `S_pd = pair_density_slots` (= 3 empirical). Let `s_C` denote the centroid shard
factor: with the X/Y duplication present in `B_persist`, the **sum** of the two copies
divides by `min(p_x, p_y)` not `P` — but each individually divides by its respective
axis size. For a balanced mesh `p_x = p_y = √P` both copies cost `B·nk·μ·nb·ns/√P` each.

### 3.a Persistent footprint

```
B_persist =  B·nk·nb·ns·μ / p_y          # psi_rmu_Y
          +  B·nk·μ·nb·ns / p_x          # psi_rmuT_X
          +  B·nq·μ²       / P           # L_q
          +  B·nq_disk·μ_pad·ngkmax / P  # gflat_acc
          +  small (sphere_idx, q_frac, norms, cct_trace)

W_pool = B − B_persist
```

Hard-fail check: `B_persist > B` ⇒ raise before any kernel compiles. The Cohsex.in
contract requires this to be ≤ ~30% of B in practice; if not, the input is mis-sized.

### 3.b Inside the fused fit_one_rchunk (Peak C)

```
W_zeta(r_chunk; channel)  =  S_pd · B · nk · ns² · μ_chan · r_chunk / P

W_wfn(band_chunk, k_chunk) =
  if sharded-band-FFT holds (small problems):
       B · nk_eff · band_chunk · ns · n_rtot · T / P
  else (large problems, see §5):
       B · nk_eff · band_chunk_per_dev · ns · n_rtot · T

  where  nk_eff = min(nk, psig_k_chunk_size or nk)
         band_chunk_per_dev = max(1, band_chunk / P)
```

Solve intermediate (small relative to W_zeta when `nq·μ < S_pd·nk·ns²·μ`, i.e.
`nq < S_pd·nk·ns²` ≈ 12·nk — always true since nq = nk):

```
W_solve = (1 or 2) · B · nq · μ · r_chunk / P
```

`(1)` is cuSolverMp branch (one Z_q live, donated), `(2)` is shard_map fallback
(input + output co-live unless explicitly donated).

cuSolverMp internal scratch (`potrs`/`getrs`) — assume `≤ B · μ²/P` per rank (one
panel). Not modelled by current code at all.

### 3.c Inside accumulate_rchunk_to_gflat (Peak D)

```
W_accum(cs) = T · B · cs · n_rtot      # the per-iter FFT box (`buf` + cuFFT scratch)
            + B · cs · ngkmax          # the gather contribution
            + B · nq_disk · μ_pad · r_chunk / P    # ζ_chunk still live as input
```

The ζ_chunk term is technically a *carryover from Peak C* — `zeta_chunk` is passed in,
not allocated here — but it is alive during `_kernel` so it counts against the Peak D
budget. (It is freed via `del zeta_chunk` at `:2181`, after the call returns.)

### 3.d Sanity: peaks fit independently

```
Peak_B ≤ B   (pre-loop CCT)
Peak_C ≤ B   (in-loop fit)
Peak_D ≤ B   (in-loop accumulate)
```

Each must hold individually; they happen at different times.

---

## 4. r_chunk picker procedure

The principle (per `report.md §5.3`): **r_chunk first, biggest possible**.
`r_chunk` only appears in `W_zeta` and `W_solve`, and only in `W_zeta` does it scale
with the multiplicative factor `S_pd · nk · ns²`. So:

```python
def pick_chunks(B, meta, mesh, channel='charge', T=4.0, S_pd=3):
    nk, ns, μ, nq, nr = meta.nk_tot, meta.nspinor, meta.n_rmu_chan(channel), \
                        meta.nk_tot, meta.n_rtot
    P, p_x, p_y = mesh.size, mesh.shape['x'], mesh.shape['y']
    nb_full = meta.nb_full
    nq_disk, ngkmax = meta.nq_disk, meta.ngkmax
    μ_pad = round_up_to_mult(μ, P)

    # 1. Persistent footprint, raise on infeasible
    B_persist = (B_c128 * nk*nb_full*ns*μ/p_y
               + B_c128 * nk*μ*nb_full*ns/p_x
               + B_c128 * nq*μ**2 / P
               + B_c128 * nq_disk*μ_pad*ngkmax / P)
    W_pool = B - B_persist
    if W_pool <= 0:
        raise BudgetError(f"persistent footprint {B_persist/1e9:.2f} GB exceeds budget")

    # 2. r_chunk — single-knob inequality, no joint search
    α_zeta = S_pd * B_c128 * nk * ns**2 * μ / P
    r_chunk = min(nr, int(W_pool / α_zeta))
    r_chunk = max(r_chunk, μ)           # don't go below μ-row iter overhead
    r_chunk -= r_chunk % P              # divisibility for the (μ_XY, r_Y) shard
    if r_chunk <= 0:
        raise BudgetError("no positive r_chunk fits — μ too large or P too small")

    # 3. band_chunk — wfn FFT box must fit in one pair-density slot.
    #    Use W_pool/S_pd as ceiling (not 50/50) per §5.5 aliasing argument.
    #    UNSHARDED case (see §5 below) cannot rely on the /P divisor.
    bc_ceiling = W_pool / S_pd
    def W_wfn(bc, k_chunk):
        nk_eff = min(nk, k_chunk if k_chunk > 0 else nk)
        if sharded_fft_box_holds(meta, mesh, bc):    # heuristic, see §5
            return T * B_c128 * nk_eff * bc * ns * nr / P
        else:
            bpd = max(1, bc // P)                    # band axis sharded, rest replicated
            return T * B_c128 * nk_eff * bpd * ns * nr
    band_chunk = nb_full
    k_chunk = 0          # default = no inner k chunking
    while W_wfn(band_chunk, k_chunk) > bc_ceiling and band_chunk > 1:
        band_chunk //= 2
    # If still doesn't fit at bc=1, drop k_chunk (psig_k_chunk_size)
    while W_wfn(band_chunk, k_chunk or nk) > bc_ceiling and (k_chunk or nk) > 1:
        k_chunk = max(1, (k_chunk or nk) // 2)
    if W_wfn(band_chunk, k_chunk) > bc_ceiling:
        raise BudgetError("WFN FFT box cannot be made to fit at bc=1, k_chunk=1")

    # 4. gflat_chunk_size — own jit, full W_pool budget
    N_rows = (nq_disk * μ_pad) // P     # per-rank flat rows
    def W_accum(cs):
        return (T * B_c128 * cs * nr +
                B_c128 * cs * ngkmax +
                B_c128 * nq_disk * μ_pad * r_chunk / P)
    if W_accum(N_rows) <= W_pool:
        gflat_chunk_size = None      # one-shot
    else:
        gflat_chunk_size = max(1, int((W_pool - B_c128*nq_disk*μ_pad*r_chunk/P) /
                                       (T*B_c128*nr + B_c128*ngkmax)))

    return r_chunk, band_chunk, k_chunk, gflat_chunk_size
```

**Bispinor handling.** Charge and transverse channels have different `μ` (`μ_C ≈ 1800`,
`μ_T ≈ 1200` at CrI3 scale). The pipeline calls `fit_zeta_to_h5` once per channel
(four calls total: one charge + three transverse for μ_L=1,2,3). Each call's `B_persist`
uses its own `μ_chan`. The transverse channels share centroids (single current-density
centroid file → one `μ_T`), so the per-channel `B_persist` is identical for μ_L=1,2,3.
The Cholesky-vs-LU difference matters for cuSolverMp scratch (LU has pivoting workspace
≥ Cholesky's), but bytes-per-rank are the same order. **Pick chunks per-channel** —
`μ_C` and `μ_T` give different `α_zeta`, so charge tolerates a smaller `r_chunk` than
transverse.

**Unsharded-FFT-box pathology.** See §5 — I explicitly do *not* try to model "XLA might
or might not shard" as a continuous function; instead I gate by a Boolean
`sharded_fft_box_holds(...)` that the planner must set conservatively. At CrI3 80 Ry
scale on a 4×4 mesh, force `sharded_fft_box_holds = False` and require an explicit
`psig_k_chunk_size`. This is the **fix priority #1** from `report.md §5.8`.

---

## 5. Validation at CrI3 6×6 80 Ry

Plug the §4 numbers into the formulas. Mesh 4×4 (`P=16, p_x=p_y=4`), bispinor (`ns=2`).
B = 60 GB. Charge channel (binding for `μ_C = 1808` padded).

Inputs:
- `nk = nq = 36`, `nb_full ≈ 400`, `μ = μ_pad = 1808`, `n_rtot = 1.125·10⁶`
- `nq_disk ≈ 7` (IBZ, conservative — `report.md §4` says "reduced from up to 400 by symmetry")
- `ngkmax ≈ 30,000` (80 Ry sphere on a 75·75·200 grid — guess; could be 10–50k)
- `T = 4`, `S_pd = 3`

### 5.a Persistent

```
psi_rmu_Y    = 16·36·400·2·1808 / 4   = 208 MB
psi_rmuT_X   = 16·36·1808·400·2 / 4   = 208 MB
L_q          = 16·36·1808² / 16       = 117 MB
gflat_acc    = 16·7·1808·30000 / 16   = 380 MB     (very sensitive to ngkmax)
sphere_idx + small                   ≈  20 MB

B_persist ≈ 0.93 GB → W_pool ≈ 59.1 GB
```

(If `ngkmax = 50k`, gflat_acc ≈ 633 MB — still small. If `nq_disk = 36` instead of 7,
gflat_acc ≈ 2.0 GB; consistent with `report.md §5.2` claim of ~3 GB at "full BZ".)

### 5.b r_chunk

```
α_zeta = 3 · 16 · 36 · 4 · 1808 / 16 = 781,056 B/r-unit
r_chunk_max = 59.1e9 / 781,056      ≈ 75,670
                                    → cap at n_rtot = 1.125M (no cap binds)
                                    → divisibility by P=16: 75,664
```

**My picker says r_chunk ≈ 75,000.** Recorded empirical config in `report.md §7` is
`r_chunk = 0 (auto ~ 12500)` — **6× smaller**.

Why the discrepancy? Possible reasons, in descending probability:

1. **The current `compute_optimal_chunks` has additional cost terms I'm missing.**
   Its 6-stage model (`gw_init.py:154-361`) accounts for `α_zcol`, `α_z_slice`,
   `α_pair`, `α_gather` separately. If the binding stage at CrI3 scale is actually
   the shard_map-fallback reshard (4× Z_col temps per cr) and the cuSolverMp branch
   is *not* the default at this point, that adds `4 · B·nq·μ/P` per r-unit
   = `4·16·36·1808/16 = 260,352 B/r-unit` — comparable to one pair-density slot.
   But that still adds at most ~30% to `α_zeta`, not 6×.

2. **`fft_factor = 4` is too low at CrI3 n_rtot.** The cuFFT scratch on a
   75·75·200 box can plausibly be 2–4× the box itself. If `T = 8` and Peak D's FFT
   binding is stricter than I model, the chosen `gflat_chunk_size = 64` is the real
   constraint, not r_chunk.

3. **A `compute_optimal_chunks` `target_utilization = 0.97`** plus implicit safety
   margins in the stage equations.

4. **The shard_map fallback is in play** and the model in `compute_optimal_chunks`
   correctly counts the donate-or-double penalty (`report.md §6`: 31→16 GB on Si).
   The cuSolverMp branch is preferred at distributed scale (`report.md §2b.4`) so
   it *should* be on at CrI3. Worth confirming from the run log.

5. **`pair_density_slots` is actually higher under bispinor.** The constant 3 was
   extracted on MoS2 3×3 bispinor (`report.md §5.6`); the karmb einsum was supposedly
   the optimisation, but cuFFT on a 75·75·200 rank-7 box may force more scratch slots
   on CrI3 sizes. Plausible: `S_pd = 6–8` at CrI3, giving r_chunk ~ 25–35k — still 2–3×
   larger than 12500.

I'd want an HLO `memory-usage-report` from one CrI3 6×6 80 Ry fit_one_rchunk compile to
resolve this. Without it, **my model is honest about predicting ~6× too generous**.

### 5.c band_chunk and psig_k_chunk

`W_wfn(bc=16, k_chunk=6)` (the empirical config):

```
nk_eff = 6
bpd    = 16 / 16 = 1
W_wfn  = 4 · 16 · 6 · 1 · 2 · 1.125e6   = 0.86 GB  (sharded path)
       OR
       = 4 · 16 · 6 · 1 · 2 · 1.125e6   = 0.86 GB  (band-only sharded path; same here)
```

vs `W_pool / S_pd ≈ 19.7 GB` ceiling — fits comfortably. Without `psig_k_chunk_size`
(`nk_eff = 36`), `W_wfn = 5.2 GB` (sharded). That's also under the ceiling — so why
does the empirical config force `psig_k_chunk_size = 6`?

Per `report.md §5.8`: **XLA refuses to shard the FFT box on every rank** at CrI3
scale. If the box is unsharded, with `nk_eff = 36`, `bc=16`, no `/P`:
`W_wfn_unsharded = 4 · 16 · 36 · 16 · 2 · 1.125e6 ≈ 83 GB` — blows everything.
With `nk_eff = 6`: `4 · 16 · 6 · 16 · 2 · 1.125e6 ≈ 13.8 GB`. Still big, but fits
the 60 GB cohsex.in target.

So `psig_k_chunk_size = 6` is the manual mitigation. My planner's `sharded_fft_box_holds`
gate should default to False at large `n_rtot · nk`, forcing the unsharded formula and
auto-picking `psig_k_chunk_size`.

### 5.d gflat_chunk_size

```
N_rows = (7 · 1808) / 16 = 791 rows / rank
ζ_chunk_alive = 16 · 7 · 1808 · 75000 / 16 ≈ 1.9 GB (at MY r_chunk = 75000)
                                   or 0.32 GB (at empirical r_chunk = 12500)

W_accum(791) = 4·16·791·1.125e6 + 16·791·30000 + 1.9e9
             ≈ 57 GB + 0.38 GB + 1.9 GB = 59 GB    — too big!
                                          (one-shot fails)

W_accum(64)  = 4·16·64·1.125e6 + 16·64·30000 + 0.32e9
             ≈ 4.6 GB + 0.03 GB + 0.32 GB = 5 GB   — fits comfortably
```

So `gflat_chunk_size = 64` at empirical r_chunk = 12500 fits with huge headroom.
At my r_chunk = 75000, ζ_chunk alone is 1.9 GB so cs must be picked smaller — the
formula gives `cs ≤ (W_pool − 1.9e9) / (T·B·n_rtot) ≈ 57.2e9 / 7.2e7 ≈ 794` — still
fine.

This is **strong evidence the empirical `gflat_chunk_size = 64` is conservative** (or
the planner's 50/50 split rule that `gflat_memory_model.py` actually applies is
bisecting from one-shot down to ~64 because of how it counts persistent terms).

### 5.e MoS2 3×3 sanity (one-chunk regime)

`nk=nq=9, μ=328, n_rtot=46k, ns=1, nb≈80`, mesh 2×2 (`P=4`).
```
α_zeta       = 3·16·9·1·328/4 = 35,424 B/r-unit
r_chunk_max  = ~5e10 / 35424 ≈ 1.4M → cap at n_rtot=46k. One r-chunk. ✓
```

Matches `report.md §7`: "one r-chunk with defaults".

---

## 6. Diff against the current code

Three competing models:

### 6.a `compute_optimal_chunks` (`gw_init.py:154-361`)

- **6-stage moments** (`fft`, `zct`, `reshard`, `solve`, `gather`, `pair`) each
  modelled as `base + αᵢ·cr + cᵢ`. Picks `cr = min_i (headroom - cᵢ)/αᵢ`. Conceptually
  closer to the **legacy decomposed CCT/ZCT chain** with separate stages — these
  stages don't all exist as distinct buffer-lifetime windows in the post-2026-05-11
  monolithic shard_map. **Some α coefficients double-count buffers that XLA aliases.**
- Wisely uses `query_fft_peak_bytes` from XLA to *query* the in-loop FFT box, not
  guess. **Keep this.**
- Has an `n_bc` pile-up term (`_fft_moment` with `n_bc` copies — see `_build_chunk_alphas`
  not read here but invoked via `gw_init.py:253`). May overcount if XLA fuses the
  unrolled bc-loop better than assumed. Worth re-baselining.
- **Closest-to-right pieces**: structural `headroom/α` inversion; pre-loop Peak A/B
  budget; query-XLA for FFT. **Wrong pieces**: post-monolithic stage decomposition;
  α inflation across stages that should alias.

### 6.b `gflat_memory_model.plan_gflat_chunks` (the 4-peak model)

- Aliasing right: A/B/C/D as **separate** XLA modules → max, not sum.
- Inside Peak C: identifies pair_density_slots as the binding lever (linear in r_chunk).
  Conceptually matches my §1.c, §3.b.
- **50/50 band_chunk vs r_chunk budget split** in `plan_gflat_chunks` (line 297:
  `bc_cap = 0.5 * target / per_unit_bc`). This is the §5.5 "principled choice
  should be 1/S_pd" objection. Whether 1/3 or 1/2 matters at the budget edge.
- **Misses unsharded FFT box** (§5.c above). FFT box is sized assuming `/p_xy` always
  holds. At CrI3 it doesn't.
- Misses cuSolverMp internal scratch.
- Per-peak breakdown logging is useful.

### 6.c `aot_memory_model/`

- Empirical / data-driven: NNLS-fits coefficients β over a DoE. Most honest about
  "what the HLO actually allocates". Per-primitive `β` should equal the slot count
  if the DoE separates collinear primitives.
- **Coverage gap**: only models `fit_one_rchunk` (and `vq_mu_chunk`, `slab_write`,
  `pair_density`, etc., as separate kernels). Doesn't compose them into a single
  end-to-end channel budget across L1/L3/L4.
- **Tag-versioning**: fits live in `aot_memory_model/artifacts/` per `tag`. A
  fit-vs-source mismatch (kernel-structure edit, new XLA version) silently drifts.
- Has the right shape for *the future answer*. But until the chooser composes peaks
  A+B+C+D and tracks IBZ / bispinor variants, it's incomplete.

### 6.d What needs to go / be added

**Drop:**
- `compute_optimal_chunks`'s 6 stages — collapse to 4 lifetime peaks (A pre-load,
  B CCT/factor, C fit_one_rchunk, D accumulate). Stages 3–6 (`reshard`, `solve`,
  `gather`) are sub-terms of Peak C now, not independent peaks.
- The 50/50 band-vs-zeta split in `plan_gflat_chunks`. Use `W_pool / S_pd` per §5.5.

**Add:**
- An **unsharded W_wfn term**. If `n_rtot · nk · bc · ns · 16 > some_threshold`,
  assume XLA materialises the FFT box unsharded and bound it accordingly. Auto-pick
  `psig_k_chunk_size` to fit.
- **Per-channel chunking** (`μ_C` vs `μ_T`). Today's chunker doesn't redo the budget
  for transverse — it would, if the channels are called in sequence and chunks are
  picked once per `fit_zeta_to_h5`. Verify: does `gw_init.fit_zeta` re-pick per
  channel?
- **cuSolverMp scratch term** at ~`B·μ²/P` per rank (1–2 panels).
- **`W_vq` term** for the V_q kernel pass, picked from the same `B` budget after
  the r-chunk loop (with only `gflat_acc` + ζ files on disk persistent).
- **HLO calibration test** — fail loudly if `pair_density_slots` extracted from a
  fresh dump differs from source.

**Closest-to-right starting point:** `gflat_memory_model.plan_gflat_chunks`, with
the per-channel μ plumbing of `aot_memory_model` and the XLA-query FFT sizing of
`compute_optimal_chunks`. Three half-rights → one whole right, in that order.

---

## 7. Open questions — what I cannot resolve from desk research alone

This is the most important section. Listed in descending priority.

### 7.1 Why does the current planner pick r_chunk ≈ 12500 at CrI3, not ~75000?

My formula says 75k fits comfortably under 60 GB. Empirical is 12500. Six explanations
in §5.b, but I can't disambiguate without:

- The HLO dump from one CrI3 fit_one_rchunk compile (memory-usage-report).
- The actual `compute_optimal_chunks` (or `plan_gflat_chunks`) output for that run —
  which stage binds, and at what HBM. The planner logs this; I'd want to read the
  run log.
- Whether the cuSolverMp branch or the shard_map fallback is in play at this scale.

If the binding stage is the shard_map-fallback reshard with un-donated Z_q, the
working set there is ~4× larger than my model assumes. If `S_pd` at CrI3 scale is
actually 6–8 (cuFFT scratch on a giant rank-7 box), that's another doubling.

**My best guess:** `S_pd_effective ≈ 5–6` at CrI3 (cuFFT scratch on the rank-7 IFFT/FFT
inside `z_q_from_psi_sm._local`), and the planner adds margin → ~12–15k. I'd bet on this
over (1)–(4) in §5.b.

### 7.2 Does XLA shard the band-chunked FFT box at CrI3 scale?

The whole point of `psig_k_chunk_size` is that it doesn't. The fix is to locate the
unsharded intermediate inside `to_rchunk` (`wfn_transforms.py:338-…`) via HLO grep and
insert sharding at the creation site. `report.md §5.8` calls this fix priority #1.
`report.md §6` notes that a `with_sharding_constraint` at the call-site boundary did
**not** work — XLA kept both pre- and post-constraint layouts live. So the constraint
needs to be inside `to_rchunk`, at the IFFT output, before any non-FFT op.

Without HLO confirmation, my planner has to conservatively assume unsharded above some
heuristic threshold. Threshold choice: `n_rtot ≥ 5·10⁵` is my guess.

### 7.3 What is `pair_density_slots` for the bispinor transverse channel at CrI3 scale?

The constant 3 was extracted on MoS2 3×3 (`gflat_memory_model.py:175-179`). Open
questions:
- Does it change for transverse vs. charge? The γ̃-perm/phase contract `gamma_double_contract`
  is monomial — one extra `jnp.take` + phase multiply (`report.md §4`) — but might
  XLA materialise an extra `n_rmu·r_chunk` slot? I doubt it (the take fuses), but
  unconfirmed.
- Does it change with `nq` (more q's → more pre-FFT work → more slots)? Probably not
  — the FFT is over k, replicated.
- Does it change between cuSolverMp and shard_map-fallback solve branches? Probably
  yes — the fallback's 2-step reshard adds Z_q lifetime overlap. I'd expect S_pd_eff
  closer to 4–5 there.

### 7.4 cuSolverMp internal scratch

cuSolverMp `potrf`/`potrs`/`getrf`/`getrs` allocate workspace internally. For 2D-block-cyclic
algorithms, this is at least one panel: `≈ B·μ_block² · 1` per rank where μ_block is the
block size (set by `compute_block_size_for_2d_cholesky`, not read). Default block sizes are
typically 64–256, so workspace ≤ 64 MB per rank — usually negligible. But for transverse
LU with pivoting, the workspace can balloon: cuSolverMp `getrf` keeps pivoting metadata
plus a panel + lookahead. **Not modelled anywhere.**

### 7.5 cuFFT scratch absolute size (`fft_factor`)

The single scalar `T = 4` is empirical within 10% (`report.md §5.8`). For the rank-7
IFFT inside `z_q_from_psi_sm._local` (axes `(0,1,2)` on a shape
`(kx,ky,kz, ns, r_chunk_local, μ_local, ns)`), cuFFT may need 2–8× the box. The
accumulate-side FFT (3D on a (cs, nx, ny, nz) box) may need a different factor. The
band-chunk centroid-load FFT is yet another call site.

`compute_optimal_chunks` partially addresses this by querying XLA directly via
`query_fft_peak_bytes` (`gw_init.py:343`). But that only covers the in-loop band FFT;
the rank-7 IFFT inside the monolithic shard_map kernel and the accumulate FFT are
both modelled with the global `T = 4` constant. Per-call-site `T` would help.

### 7.6 ζ_chunk co-lifetime with the accumulate FFT box

I assumed `ζ_chunk` is alive *during* `accumulate_rchunk_to_gflat`'s `_kernel` because
`del zeta_chunk` happens *after* the call. Confirmed in source (`isdf_fitting.py:2181`).
But: inside `_kernel`, is the rank-3 `rchunk` array (the input) kept resident in
parallel with the per-chunk FFT box `buf`, or does XLA alias them? They have
different shapes (`(n_q, μ_local, r_chunk)` vs `(cs, n_rtot)`); aliasing is unlikely.
So Peak D includes `ζ_chunk` bytes + FFT box bytes — confirmed via line 547 of
`wfn_transforms.py`: "Memory bound: chunk_size · n_rtot · 16 B for the per-iteration
FFT box" — `chunk_size`, not the full N_rows. Good.

### 7.7 IBZ-reduced nq_disk vs full-BZ

For bispinor μ_L > 0, `write_ibz_only=False` is currently the path (`report.md §4`:
"until the bispinor V_q orchestrator gains IBZ support"). So gflat_acc carries
`nq_disk = nq_full` for transverse channels — `36` rather than `~7`. My §5.a numbers
assume nq_disk=7 (charge IBZ). For transverse on bispinor, gflat_acc is ~5× larger.
This doesn't change the binding Peak C analysis but does eat into `W_pool` by ~1.5 GB
extra per transverse channel.

### 7.8 Behaviour of "donation" across separately-compiled jits

The `gflat_acc` donation pattern (`accumulate_rchunk_to_gflat` modifies in place via
the donated arg) is encoded by JAX, but XLA can refuse if the donation pattern
clashes with sharding. Worth dumping HLO to confirm `gflat_acc`'s allocation has
exactly one slot (donation honoured) and not two (input + output co-live → 2×
persistent footprint).

### 7.9 What is `ngkmax` for CrI3 80 Ry?

I guessed 30k. Could be 10k–50k. The persistent `gflat_acc` cost is linear in ngkmax;
3× misestimate → 3× error on a 0.4–2 GB term. Not budget-critical, but affects whether
`gflat_chunk_size` one-shot is feasible.

### 7.10 Cohsex.in interaction: who wins?

Cohsex.in `*_chunk_size = 0 → planner picks; nonzero → wins` (`report.md §5.7`). The
empirical CrI3 config (`report.md §7`) explicitly overrides `band_chunk_size = 16`,
`gflat_chunk_size = 64`, `psig_k_chunk_size = 6`. So the empirical r_chunk ≈ 12500
*is* the planner's auto pick — but with three of four knobs frozen by the user. My
formula picks all four; the comparison in §5.b is *not* apples-to-apples. Worth
re-running the planner with cohsex.in `band_chunk_size = 0` and seeing what it picks
unconstrained.

---

## Summary

The model boils down to:

```
B_persist = ψ_X + ψ_Y + L_q + gflat_acc                    (sum, persistent)
W_pool    = B − B_persist
Peak_C    = S_pd · α_zeta · r_chunk + W_wfn(bc, k_chunk)   (alias rest)
Peak_D    = ζ_chunk_alive + T · cs · n_rtot · 16           (separate jit)
Pick:
  r_chunk          ≤ (W_pool − W_wfn) / (S_pd · α_zeta)
  band_chunk · ...  W_wfn(bc, k_chunk) ≤ W_pool / S_pd
  gflat_chunk_size  W_accum(cs)        ≤ W_pool
```

with three sticky open questions:

1. **`pair_density_slots` at CrI3 scale.** Empirical 3 may be 5–8 there.
2. **Unsharded FFT box.** Needs an HLO grep to fix at source; until then the planner
   has to assume worst-case unsharded above a threshold.
3. **Why the empirical r_chunk ~ 12500 ≪ my 75000.** Either (1) or unmodelled
   reshard/solve overhead — needs HLO + run-log triage.

Items 1 and 3 are the same investigation: one CrI3 HLO dump answers both.

Agent 4 done — see agent_4.md.
