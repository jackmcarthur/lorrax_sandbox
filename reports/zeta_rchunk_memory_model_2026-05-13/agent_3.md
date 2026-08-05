# Agent 3 — zeta-fit r-chunk memory model, from-scratch derivation

Read-only audit of `sources/lorrax_A/` against the prose model in
`reports/zeta_v_q_g_flat_reference_2026-05-12/report.md` §5 and the
three competing implementations (`compute_optimal_chunks`,
`gflat_memory_model.py`, `aot_memory_model/`).  No code edits, no
compute, no peeking at the other agents.

Symbols throughout: `nk = n_q = nk_tot`, `ns ∈ {1, 2}` is `nspinor`,
`n_rmu` is the **logical** centroid count (with `n_rmu_padded`
rounded up to a multiple of `P = p_x · p_y`; assume `n_rmu` below
unless padding matters), `n_rtot = nx·ny·nz`, `ngkmax` is the per-q
ζ-sphere ceiling, `nb_L`, `nb_R` are left/right band-window widths,
`nb_sum = nb_L + nb_R`, `nb_full = max range`, `bc` is `band_chunk`,
`cr` is `r_chunk`, `cs` is `gflat_chunk_size`, `kc` is
`psig_k_chunk_size`.  Bytes-per-c128 = 16.

---

## 1. Tensor catalog

Each row is one physical buffer.  "Per-rank bytes" assumes c128 and
shows the *postsharded* size.  "Lifetime" labels which kernel pass
owns it.  Centroid-side (CCT, pre-loop) vs r-chunk-loop entries are
separated.  Charge/transverse differences are called out where they
exist.

### 1.1 Persistent across the r-chunk loop

| # | Tensor | Shape | Sharding | Bytes/rank | Lifetime / notes |
|---|---|---|---|---|---|
| P1 | `psi_rmuT_X` (left, X-fit) | `(nk, n_rmu, nb_L, ns)` | `P(None,'x',None,None)` | `16·nk·n_rmu·nb_L·ns / p_x` | from `load_centroids_band_chunked`; survives whole fit (also reused for wfn bundle) |
| P2 | `psi_rmuT_X` (right) | `(nk, n_rmu, nb_R, ns)` | `P(None,'x',None,None)` | `16·nk·n_rmu·nb_R·ns / p_x` | same |
| P3 | `psi_rmu_Y` (left, Y-fit) | `(nk, nb_L, ns, n_rmu)` | `P(None,None,None,'y')` | `16·nk·n_rmu·nb_L·ns / p_y` | same |
| P4 | `psi_rmu_Y` (right) | `(nk, nb_R, ns, n_rmu)` | `P(None,None,None,'y')` | `16·nk·n_rmu·nb_R·ns / p_y` | same |
| P5 | `L_q` (charge) or unfactored `C_q` slice (transverse) | `(nq, n_rmu, n_rmu)` | `P(None,'x','y')` | `16·nq·n_rmu²/P` | only one alive at a time — built after CCT, persistent through chunk loop |
| P6 | `gflat_acc` | `(n_q_disk, n_rmu_pad, ngkmax)` | `P(None,('x','y'),None)` | `16·n_q_disk·n_rmu·ngkmax / P` | persistent; donated each iter to `accumulate_rchunk_to_gflat` |
| P7 | `cct_trace_per_q` (transverse only) | `(nq,)` | replicated | `16·nq` | negligible; charge channel passes a tiny placeholder |
| P8 | host ψ(G-flat) tile | `(nk, nb_full/P, ns, ngkmax)` | host-resident per-rank tile | 0 device bytes | host_cache mode — NOT in HBM (the §5 reference is correct that ψ(G) is host-only now; AOT model's old `T_psiG_cache` primitive is dead, see chooser.py `psiG_cache REMOVED` comment) |
| P9 | phase tables for `accumulate` | `(nq, max(nx,ny,nz))` total | replicated in closure | ~tens of MB | precomputed once in `_RCHUNK_TO_GFLAT_CACHE` closure |
| P10 | sphere_idx | `(n_q_disk, ngkmax) int32` | replicated in closure | `4·n_q_disk·ngkmax` | baked into closure (`sphere_c`) |
| P11 | kvecs_frac, g_index | host | trivial | | |

Aggregate persistent device footprint (sum of P1..P7, ignoring trivials):

```
B_persist =  16·nk·n_rmu·(nb_L + nb_R)·ns / p_x       # P1+P2
          +  16·nk·n_rmu·(nb_L + nb_R)·ns / p_y       # P3+P4
          +  16·nq·n_rmu² / P                          # P5
          +  16·n_q_disk·n_rmu·ngkmax / P              # P6
```

For symmetric pseudobands (nb_L = nb_R = nb_full) this is the
dominant fixed cost.  Note **two centroid copies per side** (one
X-fit, one Y-fit) — the §5 reference under-counts at one factor of 2
in `B_persist`.  See diff in §6.

### 1.2 CCT pre-loop transients (Step 2 in `fit_zeta_to_h5`)

Only alive during the one `c_q_from_psi_sm` call (one-shot, no
r-chunk loop).  Inside the shard_map (see `isdf_fitting.py:296–351`):

| # | Tensor | Per-rank bytes | Notes |
|---|---|---|---|
| C1 | `P_l` (rank-5, einsum out, then deleted before `P_l_3d`) | `16·nk·ns²·(n_rmu/p_x)·(n_rmu/p_y)` | `'kmna,knbr->karmb'` |
| C2 | `P_l_3d` reshape (bitcast, alias of C1) | aliased | reshape only |
| C3 | `P_l_R` (`ifftn` output, c128) | same as C1 | alias of C1 after del? No: `ifftn` allocates a new buffer; cuFFT in-place is rare in JAX |
| C4 | `P_l_R_conj` | same as C1 | new buffer |
| C5 | `P_r` then `P_r_R` | same as C1 | mirror of L side |
| C6 | rank-5 reduction output `C_R` `(kx,ky,kz,col,μ)` | `16·nk·(n_rmu/p_x)·(n_rmu/p_y)` | rank-3 after spin reduce |
| C7 | `C_q_3d` (`fftn` output) | same as C6 | new buffer |
| C8 | persistent `C_q` `(nq, n_rmu, n_rmu)` `P(None,'x','y')` | `16·nq·n_rmu² / P` | output of the jit; lives until `factor_c_q` consumes it |

The `del P_l; ... del P_l_3d, P_l_R` lines tell XLA the lifetimes
end, but the buffers are not necessarily aliased — DCE / scheduler
choice.  A reasonable upper bound: 3 concurrent rank-5 P-shape
slots are alive (this matches the `pair_density_slots = 3` reference
constant, see §2).

CCT peak per device, with `α_cct = 16·nk·ns²·n_rmu²/P` (matches the
report's `α_pair · μ` at `cr = μ`):

```
W_CCT  ≈  3 · α_cct        +  2 · 16·nq·n_rmu²/P        # 3 rank-5 + (C_q + L_q workspace)
       =  3·16·nk·ns²·n_rmu² / P  +  32·nq·n_rmu² / P
```

This is a **one-shot** peak — only matters if `B_persist + W_CCT > B`
at problem-setup time.  At CrI3 6×6 80 Ry it's bounded (see §5).

### 1.3 Cholesky / LU factor stage (Step 3)

`factor_c_q` runs `cholesky_2d_batched` (charge) or 2D-block LU
(transverse).  Workspace: cuSolverMp internal scratch + one
`(nq, n_rmu, n_rmu)/P` factor.  Reference §5.8 flags cuSolverMp
scratch as ~`n_rmu²`-class but unmodeled.  In tiles_to_dense /
dense_to_tiles there are also 2-3 transient copies of the tiled
matrix.  Per-rank order of magnitude:

```
W_factor  ≈  k_factor · 16·nq·n_rmu²/P  +  ε_cusolver(n_rmu²)
```

where `k_factor ∈ [2, 4]` is the number of concurrent factor-shape
buffers XLA holds during the call (un-audited).  See open
question Q3.

### 1.4 Per-r-chunk-iter transients (inside `fit_one_rchunk`)

Following the kernel body `_make_fit_one_rchunk_kernel` and
`z_q_from_psi_sm` (isdf_fitting.py:1229–1330, :412–470).  Each
band-chunk iter:

#### a. Wavefunction fetch `psi_G_store.fetch_psi_rchunk`

Per the python-unrolled `for bc_range in band_chunk_ranges:` loop —
`n_bc = ⌈nb_full / bc⌉` iters concatenated into `psi_Y_full`.

| # | Tensor | Per-rank bytes |
|---|---|---|
| F1 | `psi_G_flat` (one bc, after io_callback) | `16·(nk/kc_outer)·bc·ns·ngkmax / P` |
| F2 | FFT box `(k_chunk, bc/P, ns, nx, ny, nz)` inside `to_rchunk` shard_map | `16·kc·(bc/P)·ns·n_rtot · F_fft` — but see §5.8/2 below: empirically materialised **unsharded** at CrI3 scale, so per-rank cost may rise by P |
| F3 | `psi_Y_full` after all bc concatenated | `16·nk·nb_full·ns·cr / p_y` (Y-sharded) |

Two big risks:
- **Unsharded FFT box.** Reference §5.8 fix-priority-1: even though
  `to_rchunk` is inside a `shard_map` over `('x','y')`, the FFT box
  shape `(k_chunk, bc/P, ns, nx, ny, nz)` of size
  `16·kc·bc·ns·n_rtot/P · F_fft` per rank in the sharded picture is
  observed at `~P × that` (i.e. effectively unsharded across ranks)
  at CrI3 scale.  `psig_k_chunk_size = kc` caps `k_chunk = kc` so
  the per-iter box is `16·kc·bc·ns·n_rtot · F_fft` *unsharded*.  The
  python loop over k iterates `⌈nk/kc⌉` times.
- **Concatenation `psi_Y_full`.** All `n_bc` iters of the bc-loop's
  reshard outputs live concurrently after the `jnp.concatenate`
  (because the concat is materialised before `z_q_from_psi_sm`
  consumes it).  Size `16·nk·nb_full·ns·cr / p_y`.  This is **the
  same as a single bc-loop with `bc = nb_full`** — i.e. the bc
  unrolling does not actually save Y-side memory at the
  concatenation boundary.  The bc knob only saves the *FFT box*.

#### b. Z_q from psi (`z_q_from_psi_sm`)

The monolithic shard_map (isdf_fitting.py:412–470) ingests
`psi_l_X` `(nk, n_rmu/p_x, nb_L, ns)`,
`psi_l_Y_sm` `(nk, nb_L, ns, cr/p_y)`,
plus the right-side pair, and outputs `Z_q` `(nq, n_rmu, cr)`
sharded `P(None,'x','y')`.  Inside, per rank:

| # | Tensor | Per-rank bytes |
|---|---|---|
| Z1 | `P_l` rank-5 einsum out `(nk, ns, cr_local, μ_local, ns)` | `16·nk·ns²·(n_rmu/p_x)·(cr/p_y)` |
| Z2 | `P_l_3d` reshape (bitcast — alias) | 0 add'l |
| Z3 | `P_l_R` ifftn output (new buffer) | same as Z1 |
| Z4 | `P_l_R_conj` | same as Z1 |
| Z5 | `P_r` then `P_r_R` (mirror) | same as Z1 |
| Z6 | post-γ̃ rank-3 reduced `Z_R` `(kx,ky,kz,cr_loc,μ_loc)` | `16·nk·(n_rmu/p_x)·(cr/p_y)` |
| Z7 | `Z_q_3d` fft output | same as Z6 |
| Z8 | `Z_q` output `(nq, n_rmu, cr)` `P(None,'x','y')` | `16·nq·n_rmu·cr / P` |

Note: Z1, Z3, Z4, Z5 are all the **same rank-5 shape**.  XLA
BufferAssignment has historically held **3** distinct pair-density-
shape lifetime slots concurrent at peak (reference §5.6 — re-extracted
2026-05-13 after the karmb spec edit).  This is the
`pair_density_slots = 3` constant.

#### c. Reshard + per-q solve (`solve_zeta`)

After `Z_q[q, μ_X, cr_Y]` → reshard → `Z_col[q, μ, cr_XY]`
(replicated-μ, XY-sharded-r) and back.  Each per-q solve writes a
`(n_rmu, cr_XY)` slice into a `zeta` of shape `(nq, n_rmu, cr_XY)`.
Then final reshard `cr_XY → μ_XY` produces the donated
`zeta_chunk[q, μ_XY, cr]`.

Per-rank transients (peak across the reshard+solve sub-graph):

| # | Tensor | Per-rank bytes |
|---|---|---|
| S1 | `Z_col` (input to solve) `(nq, n_rmu, cr/P)` | `16·nq·n_rmu·cr / P` |
| S2 | `Z_q` (input to reshard, freed after) | same |
| S3 | NCCL reshard scratch (×2 in the two-step reshard via `Z[q_X, μ, r_Y]`) | ≤ 2·S1 |
| S4 | per-q `L_q` slice replicated to one rank for triangular solve | `16·n_rmu²` per active q; legacy chunker counts up to `3·n_rmu²` |
| S5 | `zeta` output before final reshard `(nq, n_rmu, cr/P)` | `16·nq·n_rmu·cr / P` |
| S6 | `zeta_chunk` `(n_q_disk, n_rmu, cr)` `P(None,('x','y'),None)` | `16·n_q_disk·n_rmu·cr / P` |

Note S6 is the output of the kernel — alive *across* the
`fit_one_rchunk → accumulate_rchunk_to_gflat` boundary (the
accumulate consumes it, then it's freed).

#### d. Accumulate to gflat (`accumulate_rchunk_to_gflat`)

| # | Tensor | Per-rank bytes |
|---|---|---|
| A1 | `gflat_acc` donated, written in place | (P6, persistent) |
| A2 | `zeta_chunk` (S6) | as S6 |
| A3 | per-scan-iter FFT box `(cs, n_rtot)` | `16·cs·n_rtot · F_fft` |
| A4 | gathered `(cs, ngkmax)` contrib | `16·cs·ngkmax` |
| A5 | phase tables broadcast `(cs, r_len)` × 3 axes | `16·cs·max_axis_extent` |

The accumulate runs in its own jit (separate from `fit_one_rchunk`),
so it does NOT compete with Z1..Z8.  It does compete with A2 and
P5 (L_q is still alive in driver scope).

### 1.5 Charge vs transverse differences

- **n_rmu.** Per CONTEXT §4 and `v_q_bispinor_plan` §2: `n_rmu_C`
  (charge) and `n_rmu_T` (transverse, shared across μ_L=1,2,3).
  Every `n_rmu` in the catalog above is **per-channel**: each
  channel's pass uses its own value.  Charge centroid copies + L_q
  scale by `n_rmu_C`; transverse by `n_rmu_T`.
- **Factor stage.** Charge: Cholesky in `factor_c_q`.  Transverse:
  unfactored `C_q` passes through; per-q LU happens inside the
  per-r-chunk `solve_zeta` instead.  Per-r-chunk LU has higher
  per-q transient (pivots, U-factor scratch ~`2·n_rmu²`) than per-q
  Cholesky back-solve (`n_rmu²`).
- **γ̃ tables.** `(perm, phase)` of length `ns = 2` each — negligible.
- **cct_trace_per_q** is replicated `(nq,)` for the transverse
  channels only; pre-computed once per channel (isdf_fitting.py
  :1716–1721) to avoid the all-reduce inside every solve.

---

## 2. Aliasing analysis

XLA's BufferAssignment merges buffers whose lifetimes don't
overlap.  What aliases and what doesn't, with evidence:

### 2.1 Inside `z_q_from_psi_sm` shard_map

The four rank-5 P-shape buffers (`P_l`, `P_l_R`, `P_l_R_conj`,
`P_r/P_r_R`) appear sequentially.  The `del P_l; ... del P_r_3d`
hints + the rank-5 reduction afterwards constrain how many slots
XLA keeps live.  Per `gflat_memory_model._peak_C_fit_one_rchunk`
docstring + report §5.6 the BufferAssignment-measured count is
**3 concurrent rank-5 slots** at peak (after the karmb-spec edit
that bitcast-merged the rank-7 reshape).  Was 5 under the legacy
decomposed chain.  Evidence: hand-extracted from
`module_*.jit__kernel.memory-usage-report.txt`.

The buffer that the `del` hints free *could* alias with the next
allocation, and the karmb spec was *chosen* so the rank-5 → rank-7
reshape is a bitcast (`P_l_3d` aliases `P_l`).  But XLA's allocator
is not obligated to actually alias `P_l_R_conj` with `P_r_R` even
though the source code's lifetimes don't overlap — cuFFT may not
support in-place; the conj op may force a fresh buffer.  The 3-slot
count is the empirical answer.

### 2.2 FFT box vs P-pair slot

Reference §5.6 claims a band-chunk FFT box can alias into a
P-pair-shape slot.  Compare sizes:

```
FFT_box_per_rank  ≈  16·kc·bc·ns·n_rtot · F_fft / P    (sharded case)
                  or  16·kc·bc·ns·n_rtot · F_fft       (unsharded case)
P_pair_per_rank   ≈  16·nk·ns²·n_rmu·cr / P
```

For FFT-box ≤ P-pair (sharded case):

```
kc · bc · F_fft / nk · n_rtot  ≤  ns · n_rmu · cr
```

CrI3 6×6 80 Ry, kc=6, bc=16, F_fft=4, nk=36, n_rtot=1.13e6:
LHS = 6·16·4/36·1.13e6 ≈ 1.2e7.  RHS at ns=2, n_rmu=1800, cr=12500
= 4.5e10.  → FFT box much smaller than P-pair when sharding holds.

In the **unsharded** case the FFT box is P=16× larger per rank
(≈ 13 GB/rank in §5 numbers), comparable to or larger than a
P-pair slot.  Then aliasing is irrelevant — even if the slot is
unified, the *value* exceeds the slot ceiling and the FFT box
itself becomes the binding peak.  This is the §5.8 fix-priority-1
case.

### 2.3 W_wfn vs W_zeta — what shares with what

Lifetime sequence inside one `fit_one_rchunk` jit:

```
  t1: io_callback returns psi_G_flat_bc[i]                  (small, ngkmax)
  t2: to_rchunk allocs FFT_box[i], does ifftn → psi_Y_bc[i]
      → FFT_box[i] dies after slice
  t3: psi_Y_bc[i] survives to concat
  ... repeat for i = 0..n_bc-1 ...
  t4: jnp.concatenate -> psi_Y_full   (sum of all psi_Y_bc dies here)
  t5: psi_l_Y_sm / psi_r_Y_sm slice + norm
  t6: z_q_from_psi_sm shard_map (the P-pair slots are alive ONLY in
      its body — psi_Y_full is consumed and freed at the boundary)
  t7: solve_zeta (Z_q is consumed; pair-density slots freed; Z_col +
      solve scratch + zeta output alive)
```

Lifetime overlaps (a ≡ overlap, ≠ ≡ disjoint):

- `psi_Y_full` (Y-shape, post-concat) vs `P-pair slot 1`: **disjoint**
  (psi_Y_full dies inside `z_q_from_psi_sm`'s entry boundary; the
  shard_map's outermost buffer is the rank-5 P_l einsum).  XLA can
  alias them only if it sees no cross-edge — which is the case for a
  monolithic shard_map.  Score: aliasable.
- `FFT_box[i]` vs `psi_Y_bc[j]` for j > i: disjoint (the box is
  freed inside `to_rchunk`'s shard_map; only the r-sliced output
  escapes).  Aliasable in principle.
- All `psi_Y_bc[i]` for i = 0..n_bc-1 vs each other: **all alive
  concurrently** after the python unroll — the concat needs them
  all.  Sum, not max.
- Z_q (output of shard_map) vs the 3 P-pair slots: disjoint —
  Z_q is rank-3 `(nq, n_rmu, cr)/P`, much smaller, and is the
  return of the shard_map body.
- `zeta_chunk` (output) vs L_q (input): both alive at function
  return.  Z_col / solve scratch peak occurs WITHIN solve_zeta
  and is then freed before `_reshard_zeta_r_XY_to_mu_XY` runs.

### 2.4 Across `fit_one_rchunk` and `accumulate_rchunk_to_gflat`

Two *separate* jits, so XLA cannot alias across them.  Between
them, driver-scope variables alive:
- `psi_l_rmuT_X_fit` / `psi_r_rmuT_X_fit` (P1, P2) — yes
- `psi_l_rmu_Y_fit` / `psi_r_rmu_Y_fit` (P3, P4) — yes
- `L_q` (P5) — yes
- `cct_trace_per_q` (P7) — yes
- `gflat_acc` (P6) — yes
- `zeta_chunk` returned by `fit_one_rchunk` — yes
- Inside `accumulate_rchunk_to_gflat`: A3, A4, A5

So during the accumulate, the peak is
`B_persist + |zeta_chunk| + A3 + A4 + A5`.  All P-pair slots from
the previous jit are dead.  This is a separate budget point — call
it **Peak D**.

### 2.5 Summary table

| Buffer pair | Same jit? | Lifetimes | Aliasable? |
|---|---|---|---|
| `FFT_box[i]` ↔ `FFT_box[j]`, i≠j | yes (`to_rchunk` cache, called sequentially) | disjoint | yes |
| `FFT_box[i]` ↔ `psi_Y_bc[i]` | yes | sequential | yes |
| `psi_Y_bc[i]` ↔ `psi_Y_bc[j]`, i≠j | yes | overlap (concat) | **no, summed** |
| `psi_Y_full` ↔ P-pair slot 1 | yes | disjoint at shard_map entry | yes |
| 3 P-pair slots inside `z_q_from_psi_sm` | yes | overlap | **no, summed (= the magic 3)** |
| Z_q ↔ P-pair slot N | yes | disjoint | yes |
| `zeta_chunk` ↔ P-pair slots | yes | disjoint | yes |
| `zeta_chunk` ↔ A3 (accumulate FFT box) | no (different jits) | overlap | **no, summed** |
| `gflat_acc` (P6) ↔ everything | always alive | overlap | **no, summed** |

---

## 3. Budget equations

Define:

```
B          = memory_per_device_gb · 1e9                       (HBM ceiling)
η          = target_utilization (default 0.80 in gflat MM)
B_eff      = η · B
F_fft      = 4.0 (the magic `fft_factor`; covers cuFFT scratch ~ 3–8×)
S          = pair_density_slots = 3
```

### 3.1 Persistent floor

```
B_persist  =  16·nk·n_rmu·(nb_L + nb_R)·ns / p_x       # P1+P2
           +  16·nk·n_rmu·(nb_L + nb_R)·ns / p_y       # P3+P4
           +  16·nq·n_rmu² / P                          # P5
           +  16·n_q_disk·n_rmu·ngkmax / P              # P6
```

### 3.2 r-chunk-loop transient pool

```
W_pool = B_eff − B_persist
```

Inside the loop the binding peak per iter is:

```
W_iter(cr, bc, kc)  =
    psi_Y_concat(cr, bc)       # = 16·nk·nb_full·ns·cr / p_y
  + S · P_pair(cr)             # = S·16·nk·ns²·(n_rmu/p_x)·(cr/p_y)
  + W_solve(cr)                # = small Z_col + per-q LU/Cholesky scratch
  − overlaps
```

After applying the aliasing rules:

- The `to_rchunk` FFT box, *if sharding holds*, aliases into a
  P-pair slot.  If **unsharded** (CrI3 regime), it does NOT alias;
  add it explicitly.
- `psi_Y_concat` aliases with the first P-pair slot's lifetime
  (different sub-jits, but XLA's BufferAssignment in the
  monolithic jit may merge them — uncertain; reference §5.2
  optimistically aliases the FFT box but is silent on
  `psi_Y_concat`).  Conservative bound: keep `psi_Y_concat` in
  the sum.

Defining the linear-in-cr coefficient:

```
α_zeta(channel)  =  S · 16·nk·ns² · n_rmu(channel) / P
α_psi_Y          =  16·nk·nb_full·ns / p_y           (per-cr, Y-sharded only)
α_solve          ≈  16·nq·n_rmu / P   (Z_col is per-rank)
```

The dominant linear coefficient is `α_zeta`.  Independent of `cr`,
the FFT-box term (the §5.8 risk) is:

```
W_wfn(bc, kc) =
   sharded:    16 · kc · bc · ns · n_rtot · F_fft / P
   unsharded:  16 · kc · bc · ns · n_rtot · F_fft
```

with kc =`psig_k_chunk_size` (default = nk, i.e. no inner chunk),
bc = `band_chunk`.

The accumulate-side budget (separate jit):

```
W_accum(cs)  =  16 · cs · n_rtot · F_fft
```

with cs = `gflat_chunk_size` (= `n_q_disk · n_mu_local` one-shot).

### 3.3 Peak conditions

Peak A — centroid load (pre-loop, runs once per channel):

```
peak_A  =  16·nk·n_rmu·nb_full·ns / P              # output being filled
        +  16·kc·bc·ns·n_rtot · F_fft / P          # FFT box (or unsharded · P)
        +  small phase tables
```

Peak B — CCT + factor (pre-loop):

```
peak_B  =  B_persist (centroids only — gflat_acc not yet allocated)
        +  3 · 16·nk·ns² · n_rmu² / P              # 3 rank-5 slots at C_q time
        +  16·nq·n_rmu² / P                         # C_q itself
```

Peak C — `fit_one_rchunk` (the dominant peak):

```
peak_C(cr, bc, kc) =
    B_persist
  + W_wfn(bc, kc)                                   # one bc's FFT box (if not aliased)
  + 16·nk·nb_full·ns·cr / p_y                       # psi_Y_concat
  + S · 16·nk·ns²·(n_rmu/p_x)·(cr/p_y)              # pair-density slots
  + 16·nq·n_rmu·cr / P                               # Z_q / Z_col output
```

Peak D — `accumulate_rchunk_to_gflat`:

```
peak_D(cs)  =  B_persist
            +  16·n_q_disk·n_rmu·cr / P              # zeta_chunk
            +  W_accum(cs)                            # FFT box
            +  16·cs·ngkmax                           # gather contrib
```

The binding constraint set:

```
peak_A ≤ B_eff
peak_B ≤ B_eff
peak_C(cr, bc, kc) ≤ B_eff
peak_D(cs) ≤ B_eff
```

---

## 4. r_chunk picker procedure

Inputs: `meta` (system geometry), `mesh_xy`, `B`, `η`, `F_fft`,
`S = 3`, `is_bispinor`, per-channel `n_rmu`.  Knobs to pick:
`(cr, bc, kc, cs)` for each channel (and the V_q pass separately).

### 4.1 Algorithm

1. **Compute `B_persist`** from §3.1.  If `B_persist > B_eff` raise
   informative error pointing at which term dominates (likely
   gflat_acc — `16·n_q_disk·n_rmu·ngkmax / P`).
2. **Pre-loop budget check.**  Check `peak_A` and `peak_B` ≤ `B_eff`
   *before* choosing cr.  These bound `(bc, kc)` regardless of cr:

   ```
   bc · kc · n_rtot · F_fft  ≤  P · (B_eff − B_persist − small)        (sharded)
                              or  (B_eff − B_persist − small)            (unsharded)
   ```

3. **Pick (bc, kc)** via the FFT-box constraint with priority on
   keeping bc large (bc divides total bands; smaller `n_bc` reduces
   the `psi_Y_concat` term in Peak C):

   - Default `kc = nk` (no inner chunk).
   - If unsharded-FFT-box detection is on: lower `kc` until
     `bc · kc · ns · n_rtot · F_fft ≤ B_eff − B_persist` (unsharded
     form).  Halving kc is the §5.9 recipe.  Decide via a *bool
     classifier* — see Q1.
   - With kc fixed, pick `bc` = largest pow-2 ≤ nb_full s.t.
     `peak_A(bc, kc) ≤ B_eff` and the bc-divided FFT box fits the
     "1 slot" budget of Peak C, i.e.
     `W_wfn(bc, kc) ≤ B_eff − B_persist − S · P_pair(cr)`.  This
     couples to cr, so iterate (or split a fraction of W_pool to
     W_wfn — the reference §5.5 says `1/S`, gflat_memory_model
     uses 50%).

4. **Pick cr** from the Peak C linearization, with bc and kc fixed:

   ```
   W_iter(cr)  =  K0  +  K1 · cr
   with
     K0  =  B_persist  +  W_wfn(bc, kc)            (constant)
     K1  =  α_psi_Y  +  S · α_pair  +  α_zcol
         =  16·nk·nb_full·ns/p_y
          + S · 16·nk·ns²·n_rmu / P
          + 16·nq·n_rmu / P

   cr_max  =  (B_eff − K0) / K1
   cr      =  clip(cr_max, p ≤ cr ≤ n_rtot)
   cr      ← cr − (cr mod P)                       (divisibility for r_XY sharding)
   cr      ← max(cr, P)
   ```

   Floor at `n_rmu` is the reference §5 heuristic ("per-iter
   overhead floor"); I don't see strong evidence for it from
   profiling, but it's harmless — see Q4.

5. **Pick cs** from Peak D:

   ```
   cs_one_shot  =  ⌈n_q_disk · n_rmu / P⌉
   cs_max       =  (B_eff − B_persist − |zeta_chunk(cr)|) /
                   (16·n_rtot · F_fft)
   cs           =  min(cs_one_shot, cs_max)
   ```

   If `cs ≥ cs_one_shot`, the scan is one-shot (no python loop);
   otherwise the scan iterates `⌈cs_one_shot/cs⌉` times.

6. **Per-channel re-pick.**  Charge uses `n_rmu_C`; transverse uses
   `n_rmu_T`.  Run steps 1–5 once per channel; the four output
   files use independent `(cr, bc, kc, cs)` triples.  Practical
   note: the four passes are sequential, so each is sized to its
   own budget; charge usually picks **smaller cr** because
   `n_rmu_C > n_rmu_T` and `α_zeta ∝ n_rmu`.  Per CONTEXT §4:
   `n_rmu_C ≈ 1800`, `n_rmu_T ≈ 1200` (1.5× difference).

7. **V_q pass.**  Separate problem.  After the ζ loop, only
   `gflat_acc` (P6) is persistent; the four-channel ζ-files are
   read q-by-q.  V_q peak per-tile:

   ```
   peak_Vq(g_chunk)  ≈  16·n_q_disk·n_rmu·ngkmax / P              # ζ slab L
                      + 16·n_q_disk·n_rmu·ngkmax / P              # ζ slab R (or aliased if same_zeta)
                      + 16·g_chunk·n_rmu²/P · F_GEMM
   ```

   The current code's `_pick_g_chunk(ngkmax, target=4096)` is a
   divisor-of-ngkmax cap, not budget-aware.  A budget-aware pick:
   `g_chunk = min(ngkmax, ⌊(W_pool − ζ-slabs) / (16·n_rmu²/P · F_GEMM)⌋)`,
   then floor to a divisor.

### 4.2 Bispinor handling

Same algorithm, per channel.  Charge channel runs Cholesky path
(smaller per-q solve scratch); transverse channels run pivoted LU
(slightly larger per-q scratch, ~2·n_rmu² vs ~n_rmu²).  No
cross-channel competition in HBM — each channel's ζ-file is
written, then memory freed, before the next channel starts.
Footnote: this needs `is_bispinor=True` to flow through
gflat_memory_model so `n_rmu` is taken per-channel rather than once
at problem-setup.  Currently `plan_gflat_chunks` only takes one
`meta.n_rmu` — that is a **gap** in the current code (Q5).

### 4.3 Unsharded FFT-box mitigation

The unsharded materialization (§5.8) is detected empirically — there
is no HLO-grep heuristic in source today.  Recipe:

```
if shard_holds(bc, kc):
    apply sharded W_wfn
else:
    apply unsharded W_wfn = P · sharded W_wfn
    bisect kc down until unsharded W_wfn fits
```

The proper fix is shading the box at its creation site (the
shard_map body in `to_rchunk` already runs locally — but the
constant n_rtot dimensions are dense and XLA may treat them as
replicated across ranks even inside the shard_map).  Reference
§5.8 notes that an outer `with_sharding_constraint` does NOT
work and the fix has to be at the buffer's creation site.
Until that's fixed, the planner needs to know which mode it's
in — see Q1.

---

## 5. Validation at CrI3 80 Ry

Plug CONTEXT §4 numbers:
`nk = nq = 36`, `n_rmu_C = 1800`, `n_rmu_T = 1200`, ns = 2,
`nb_full ≈ 400` (assume 200 L + 200 R = `nb_sum = 400`; treating
nb_full = nb_L = nb_R = 400 as the GW symmetric case),
`n_rtot ≈ 1.125e6`, `P = 16`, `p_x = p_y = 4`.
ngkmax: report.md uses ~70k for CrI3 at 80 Ry (per psi_G_store
docstring's "ngkmax≈70k").  `n_q_disk = 36` (no symmetry beyond
the 6×6 grid).  budget = 60 GB, η = 0.80, so `B_eff = 48 GB`.

### 5.1 Persistent floor (charge channel, n_rmu = 1800)

```
P1+P2 (X-fit centroids L+R)
   = 16·36·1800·400·2 / 4  =  10.4e9    →  10.4 GB
P3+P4 (Y-fit centroids L+R, same total, p_y = 4)
   =  10.4 GB
P5 (L_q)
   = 16·36·1800² / 16       =  0.117e9   →  0.12 GB     (tiny — surprising)
P6 (gflat_acc, n_q_disk=36, ngkmax=70k)
   = 16·36·1800·70000 / 16  =  4.5e9     →  4.5 GB
─────────────────────────────────────────
B_persist                                  ≈  25.4 GB
```

For transverse channel (n_rmu = 1200) replace n_rmu in P1..P5
(except P5 has n_rmu²):

```
P1+P2 → 6.9 GB,  P3+P4 → 6.9 GB,  P5 → 0.05 GB,
P6 → 3.0 GB    →  B_persist ≈ 16.9 GB
```

Note: the *cohsex.in* working config reports peak HBM ≈ 28 GiB at
CrI3 default chunks — broadly consistent with this 25 GB
persistent floor + a small W_iter on top.

### 5.2 W_pool (charge)

```
W_pool  =  48 − 25.4  ≈  22.6 GB
```

### 5.3 Peak C linearization (charge)

```
α_psi_Y   = 16·36·400·2 / 4                 = 230 400        bytes/cr
S · α_pair= 3·16·36·4·1800 / 16             =   77 760        bytes/cr
α_zcol    = 16·36·1800   / 16               =    64 800        bytes/cr
K1 (charge) ≈ 372 960 bytes/cr   ≈  3.73e5
```

W_wfn (sharded, bc=16, kc=6):

```
W_wfn  =  16·6·16·2·1.125e6 · 4 / 16    =  86.4e6  →  86 MB
```

W_wfn (unsharded, bc=16, kc=6):

```
W_wfn  =  16·6·16·2·1.125e6 · 4         =  13.8 GB
```

The unsharded form bites — uses 13.8/22.6 = 61% of W_pool **before
any cr term**.  This matches the §5 reference's claim that
`psig_k_chunk_size=6` is the manual cap that drops the box to ~14 GB.

K0 (charge, unsharded W_wfn assumed at CrI3 scale, since this is
the empirical regime):

```
K0  =  B_persist + W_wfn_unsharded
    =  25.4 + 13.8                       ≈  39.2 GB
```

Headroom = 48 − 39.2 = 8.8 GB for the cr-scaling terms.

```
cr_max  =  8.8e9 / 3.73e5  ≈  23 600
```

Then floored to a multiple of P=16, capped at n_rtot=1.125e6:
**cr ≈ 23 600**.

Empirical "auto" value per CONTEXT §4 is **12 500**.  My derived
value (23 600) is **~2× larger**.  Possible explanations:

a. cuSolverMp scratch term unmodeled (~`k_factor · 16·nq·n_rmu²/P`
   with k_factor ≈ 3–4 → 0.5 GB; doesn't close the gap).
b. F_fft is more like 6–8× for the unsharded box at this scale
   (cuFFT plan caches multiple plans at extreme grids); F_fft = 6
   → W_wfn = 20.7 GB → K0 = 46.1 → headroom = 1.9 GB → cr_max ≈
   5 000.  Too small.  F_fft = 5 → W_wfn = 17.3 → headroom = 5.3
   → cr ≈ 14 200.  Close to 12 500.  So F_fft ≈ 5 at CrI3 scale.
   The §5.8 "empirical within 10%" claim about F_fft = 4.0 may be
   off at extreme grids — see Q2.
c. The Y-sharded `psi_Y_concat` is **summed across all n_bc**.
   With nb_full=400, bc=16 → n_bc = 25.  If psi_Y_bc[i] for
   i=0..24 are all alive concurrently (the python concat unrolls
   them), `α_psi_Y · 25` per cr — but α_psi_Y is already at
   `nb_full = sum over bc`, so this is accounted for.  Not the
   missing term.
d. Other unmodeled persistent: norms tables, q_irr_idx tables,
   small replicated tables — at most tens of MB; not the gap.
e. The Y-sharded `psi_Y_full` term in K1 is computed at `bc=nb_full`
   in my K1 (since after concat the band axis is full).  The
   actual code holds **bc-shape Y slabs** alive across the python
   unroll — `psi_Y_full` is the result *after* concat.  If XLA
   keeps `n_bc` separate bc-shape slabs alive, that's the same
   total but in many smaller buffers — same bytes.

Most likely explanation: **F_fft ≈ 5 not 4** at CrI3 scale, plus a
small (1–2 GB) cuSolverMp / NCCL scratch cushion that the current
planner does include via the `target_utilization = 0.80` knob.
The 0.80 cushion subtracts 12 GB from B already, leaving only
48 GB; the actual physical HBM is 60 GB.  If I drop η and use the
full 60 GB ceiling: cr_max = (60 − 25.4 − 13.8)·1e9 / 3.73e5 ≈
55 700.  Way too big.  So η is doing the work that an explicit
cuFFT + scratch term should do.

### 5.4 Accumulate (Peak D)

`cs_one_shot = 36·1800/16 = 4050` rows.
W_accum(cs_one_shot) = 16·4050·1.125e6·4 = 292 GB ≫ B_eff.  Way
over.  cs_max = (48 − 25.4 − 5)e9 / (16·1.125e6·4) ≈ 244 rows.

Empirical config says `gflat_chunk_size = 64`.  My formula gives
244 — F_fft = 4 may understate; using F_fft = 6 gives 162; F_fft =
8 gives 122.  Still 2× empirical.  Possibly there's a safety
factor in the user's manual setting, or the W_accum FFT box is
even more cuFFT-scratch-heavy than F_fft = 4 captures (forward
fftn on `(cs, n_rtot)` with `n_rtot ≈ 1.125e6` is a 1-D 1.1M FFT
× cs batch — large 1-D cuFFT plans have particularly chunky
workspaces).

### 5.5 Verdict

Derivation lands in the **right order of magnitude** but is
1.5–2× larger than the empirical working configuration for both
cr and cs.  The gap is most consistent with **F_fft = 4 being
too small at CrI3 grids** (especially for the large 1-D cuFFT in
`accumulate_rchunk_to_gflat`).  Per-call-site F_fft (reference
§5.10 follow-up #4) is the right fix.

---

## 6. Diff against the current code

### 6.1 What each model gets right

| Model | Right | Wrong / missing |
|---|---|---|
| `compute_optimal_chunks` (`gw_init.py:154–404`) | 5-stage moment inversion is mathematically clean.  Honors n_bc unroll on the FFT stage.  Uses `query_fft_peak_bytes` (calls XLA AOT for the in-loop FFT cost) — best **single accurate FFT term** of the three. | Doesn't model the unsharded FFT box case.  α_pair = nk·ns²·n_rmu/P doesn't include the **factor of 3 for pair_density_slots** (S=3) — reference §5.6 puts this at 3 concurrent slots, the moment inversion only counts 2 (`+ 2·α_pair·cr` in `_fft_moment`).  Missing W_accum entirely (the gflat accumulate runs after this chunker's pass with no budget term).  α_psi_Y_bc uses `band_chunk` not `nb_full` (under-counts the post-concat psi_Y_full live set). |
| `gflat_memory_model.plan_gflat_chunks` | Cleanly separates A/B/C/D peaks, knows about gflat_acc, knows about the 3-slot constant.  50/50 bc-vs-cr split is conservative-but-safe.  Per-channel `pair_density_slots_*` parameters are present (charge/transverse). | `B_persist` formula in §3.1 has both X-fit *and* Y-fit centroids — gflat_memory_model only counts L+R once (one of `2 · _bytes_c128(nk, ns, mu, nb, shard=p)` in `_peak_C_fit_one_rchunk` — but with `shard=p` not `shard=p_x`).  The 50/50 split is reference §5.5's "wrong" choice; principled is `1/S = 1/3` for bc.  Doesn't model the unsharded FFT box.  Doesn't model n_rmu_C vs n_rmu_T (single `meta.n_rmu`).  No V_q term.  No cuSolverMp scratch. |
| `aot_memory_model/` | Uses an NNLS fit of β coefficients from a DoE — empirically calibrated.  Bills `T_psiG_bc` (per-bc FFT) and `T_psiY_bc` (per-bc r-slab) as separate primitives — correctly captures the `bc · cr` cross-term.  Dropped dead primitives (`T_psiG_cache`, `T_Lq_rep`) after HLO audit. | Requires a DoE to be run beforehand (`load_fit(kernel_name, tag=tag)`).  Tags ("current") go stale on kernel edits — no validation pass against an HLO dump.  Doesn't model peaks A, B, D — only the fit_one_rchunk peak.  Doesn't know about the unsharded-FFT-box pathology.  Doesn't model bispinor n_rmu split.  No V_q kernel chooser in the same pass (though `vq_mu_chunk.py` exists as a separate kernel). |

### 6.2 Closest-to-right

`gflat_memory_model.py` is the right *shape* — A/B/C/D peaks
separated, per-rank bytes, single-pass deterministic.  The AOT
model's empirical β fit is the right *correction mechanism* for
F_fft and `pair_density_slots` calibration.

What needs to go:
- `compute_optimal_chunks`'s 5-stage moment inversion duplicates
  Peak C's job with stricter (closed-form) per-moment semantics
  that don't correspond to actual XLA buffer assignment.  The
  ZCT/reshard/solve moments are too fine-grained for what XLA's
  scheduler actually does.  Replace its cr/bc output with the
  gflat planner's.  Keep its `query_fft_peak_bytes` integration
  (best FFT-box term in the codebase).
- The AOT `psi_G_cache`/`Lq_rep` removal comments hint that
  whoever wrote that file regularly does HLO audits; productize
  that into an automatic re-calibration step (reference §5.10
  follow-up #2).

What needs to be added:
- **Per-channel n_rmu split.**  `plan_gflat_chunks` should take
  a per-channel `n_rmu_by_channel = {0: n_rmu_C, 1: n_rmu_T,
  2: n_rmu_T, 3: n_rmu_T}` mapping and produce per-channel
  `(cr, bc, kc, cs)` triples.  Currently the planner is called
  once with `meta.n_rmu` ignoring this — see Q5.
- **Unsharded W_wfn term.**  Add a `shard_holds: bool` flag to
  `plan_gflat_chunks`; if False, compute W_wfn unsharded and
  pick kc by halving until it fits.
- **W_vq budget.**  Add a fifth peak (Peak E for V_q) that
  selects `vq_g_chunk_size` from the same budget after gflat_acc
  is persistent.
- **cuSolverMp scratch.**  Add a term `ε_cusolver ≈ k_factor ·
  16·nq·n_rmu²/P` with `k_factor ≈ 3` to peaks B and C.
- **Per-call-site F_fft.**  Separate `F_fft_loader`,
  `F_fft_kshmap`, `F_fft_accum` constants — the §5.10 follow-up
  #4.  My §5 plug-in suggests `F_fft_accum ≥ 6`.

### 6.3 Minimal rewrite

Replace `compute_optimal_chunks` with:

```python
def plan_zeta_fit_chunks(meta, mesh_xy, cfg, channel_nrmu_map):
    plan = {}
    for ch, mu in channel_nrmu_map.items():
        per_ch = plan_gflat_chunks(
            meta_with_nrmu(meta, mu),
            mesh_xy,
            nb_total=cfg.nb_L + cfg.nb_R,
            ngkmax=meta.ngkmax,
            n_q_disk=meta.n_q_ibz,
            budget_gb=cfg.memory_per_device_gb,
            target_utilization=η,
            shard_holds=detect_shard_holds(meta, mesh_xy),
            fft_factor_loader=F_fft_loader,
            fft_factor_accum=F_fft_accum,
            pair_density_slots=S,
            cusolvermp_k_factor=k_factor,
        )
        plan[ch] = per_ch
    return plan
```

with each per-channel call living entirely inside the §4
algorithm.  Tests:
- recompile / re-extract `S` from an HLO dump at session start;
  fail loudly if the source `S=3` constant doesn't match.
- end-to-end on MoS2 3×3 (the easy case, should land at
  cr ≈ n_rtot) and CrI3 6×6 80 Ry (charge cr ≈ 12 500 ±
  tolerance, transverse cr larger by factor `n_rmu_C/n_rmu_T ≈
  1.5`).

---

## 7. Open questions

These are the things I genuinely could not resolve from
code + reports + math.  Honest "I don't know"s.

**Q1.  Unsharded FFT box — detection.**  Reference §5.8 says the
FFT box "gets materialised unsharded on every rank" at CrI3
scale, and the manual mitigation is `psig_k_chunk_size = 6`.  But
I cannot find anywhere in the code that flags WHICH regime XLA
chose — there's no runtime probe, no HLO signature check, no
"shard_holds=True" boolean.  The §5.8 follow-up says "locate that
intermediate via HLO grep and shard it at the creation site (a
constraint at the call boundary does not work)".  The
intermediate must be the `box_l = _box_kernel(psi_l, g_index_l,
ngkmax=ngkmax)` line inside the `to_rchunk` shard_map
(`wfn_transforms.py:390`).  That allocation is inside a shard_map
body so should already be per-rank — yet apparently isn't.
**I do not understand why** the FFT box is materialised unsharded
when wrapped in a shard_map with `out_specs=P(*out_spec)`.
Possibility: the `ngkmax=ngkmax` static kwarg causes
`_box_kernel` to allocate a dense `n_rtot`-element buffer per
rank but that buffer's *contents* are the local-rank's subset, so
the per-rank bytes are still `~n_rtot` — meaning the bytes-per-rank
ARE this big, but it's not actually "unsharded"; it's local-but-
dense.  In that case the §5.8 phrasing is misleading and the
correct mental model is that the box's r-axis is dense (full
n_rtot) on every rank because it is per-rank's *own* FFT box, not
a globally-sharded one.  → The "factor of P" framing may be a
misdiagnosis.  Test: HLO grep at CrI3 scale to confirm whether
the box's per-rank shape is `(k_chunk, bc, ns, nx, ny, nz)` or
`(k_chunk, bc/P, ns, nx, ny, nz)`.  My read of the code says it
should be `bc/P` — so the per-rank size matches the "sharded"
formula and there's no factor of P.  Without an HLO dump I cannot
confirm.  Without that I cannot write the right W_wfn formula.

**Q2.  F_fft scaling.**  Reference §5.8 calls `fft_factor = 4.0`
"empirical within ~10%".  My §5 plug-in suggests F_fft ≈ 5 for the
fit_one_rchunk box and ≥ 6 for the accumulate FFT box.  These are
NOT the same factor: a 6-D forward fftn over 3 trailing axes is
very different cuFFT-wise from a 4-D forward fftn on a
`(cs, nx, ny, nz)` batched box.  A per-call-site F_fft (reference
§5.10 follow-up #4) needs an actual measurement campaign.
**Where:**  I would dispatch a wee `query_fft_peak_bytes` for each
call site at planner time and use that instead of a global F_fft.
**Cost:**  one AOT lowering per call site at startup; cheap.

**Q3.  cuSolverMp internal scratch.**  Per reference §5.8,
unmodeled.  Probably ~`n_rmu²`-class per the docstring.  But:
how many concurrent factor-shape buffers does `cholesky_2d_batched`
hold during the factor?  Could be 1 (in-place tile factorization)
or up to ~4 if XLA SPMD batches q's.  I have no measurement.
Setting `k_factor = 3` in §6.3 is a hopeful guess.

**Q4.  `r_chunk` lower bound — what is the real constraint?**
The reference §5 says "lower-bounded by `n_rmu` (per user spec:
the eventual Σ_μν output occupies `n_rmu²·n_q·16` bytes, so paying
less than `n_rmu` work per chunk is wasted iteration overhead)".
But Σ_μν isn't the output of zeta-fit — it's `(n_q_disk, n_rmu,
n_rtot)` ζ on disk.  The argument doesn't follow.  My read: the
lower bound is just `cr ≥ P` so the (μ_XY, r_) sharding at the
solve output divides cleanly.  The `cr ≥ n_rmu` floor in
`compute_optimal_chunks` and `gflat_memory_model.plan_gflat_chunks`
may be redundant / wrong.  Worth verifying by setting
`r_chunk_override = P` and measuring whether anything else fails.

**Q5.  Per-channel chunker sizing in bispinor.**  Currently
`plan_gflat_chunks` is called once per `fit_zeta` invocation with
`meta.n_rmu`.  In the bispinor pipeline `fit_zeta` is called 4
times (once per μ_L); does the **second/third/fourth** call (the
transverse channels with `n_rmu_T`) re-call the planner with the
updated n_rmu?  Reading `gw_init.fit_zeta` (lines 532–700) I see
only one `plan_gflat_chunks` call per invocation, but the
bispinor loop is *not* inside `fit_zeta` — it's in the outer
caller.  Need to confirm that each `fit_zeta` invocation gets its
own per-channel `n_rmu` from `meta`.  If `meta.n_rmu` is set once
at problem-setup with the **charge** value, all four channels
would size off `n_rmu_C` — over-allocating for transverse and
potentially missing the larger cr opportunity (since
`α_zeta ∝ n_rmu` and transverse has smaller `n_rmu_T`).

**Q6.  `psi_Y_full` aliasing — does XLA actually free the
per-bc slabs after concat?**  The python unroll produces
`psi_Y_parts = [...]` then `psi_Y_full = jnp.concatenate(...)`.
At the concatenation, XLA could either (a) allocate a fresh
buffer and copy each slab in, then free the slabs (peak: 2× the
post-concat size momentarily), or (b) emit a fused gather that
writes directly into the post-concat buffer (peak: 1× the
post-concat size).  Without an HLO dump I assume (b); if (a) the
Peak C formula has a missing 2× factor on the psi_Y term.

**Q7.  Accumulate FFT box vs cuFFT plan workspace.**  The 1-D FFT
of length 1.125e6 inside `accumulate_rchunk_to_gflat` is unusual.
cuFFT's `cufftXtMakePlanMany` workspace for that size can be
~3× the data depending on the algorithm chosen (Bluestein vs
mixed-radix).  Setting `F_fft_accum = 4` may dramatically
under-budget — explaining why `gflat_chunk_size = 64` was the
manual setting and my formula predicts 244.  Need a cuFFT-side
measurement (e.g. `cufftGetSize`) to calibrate.

**Q8.  Bispinor centroid bypass.**  `v_q_bispinor_plan` §4.b says
"four channels of ζ … are still on B's branch only".  Has the
4-channel orchestrator landed on lorrax_A's main?  I see
`v_q_bispinor.py` and `sigma_x_bispinor.py` in the source tree —
suggesting yes — but `fit_zeta` only loops over one channel.  Is
the bispinor 4-channel loop implemented and validated?  If not,
the per-channel chunker design (§4.2) is forward-looking and
its bispinor branch is unexercised.

**Q9.  `n_q_disk` vs `nq`.**  My §5 plug-in used `n_q_disk = 36`
for the full BZ; the IBZ reduction shrinks this.  The reference
§4 CrI3 spec says "n_k = n_q = 36 (reduced from up to 400 by
symmetry)".  Is "up to 400" the L_q-shape full BZ count or just
loose phrasing?  For a 6×6×1 mesh, full BZ has 36 q's, full BZ ≤
6²·m for some monkhorst expansion only if the centroid mesh is
larger than the k-mesh.  Unclear.  Assuming n_q = n_q_disk = 36
is consistent with the code; if the actual full-BZ n_q is bigger
and the IBZ is 36, then `B_persist` is bigger than my §5 number
(by the factor on P5 and pre-IBZ gflat_acc).

**Q10.  HLO dump pipeline.**  Reference §5.10 follow-up #2:
"auto-recalibrate `pair_density_slots` from an HLO dump pass".
The codebase already extracts the constant manually (per the
`gflat_memory_model.py` docstring's instructions).  Building this
into a startup probe would close the largest fragility in the
current model.  Open: what's the API to enable XLA's
`module_*.jit__kernel.memory-usage-report.txt` dump from inside
JAX 0.4.x?  I think it's the `XLA_FLAGS=--xla_dump_to=...` env
var, but the parser to grep "pair-density-shape lifetime slots"
out of the output doesn't exist yet.

---

Agent 3 done — see agent_3.md
