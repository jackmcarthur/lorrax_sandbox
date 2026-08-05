# Agent 1 — zeta-fit r-chunk memory model (from-scratch derivation)

Scope: a defensible per-rank HBM model for `fit_zeta_to_h5`
(`src/common/isdf_fitting.py:1458`) that produces
`(r_chunk, band_chunk, gflat_chunk_size, psig_k_chunk_size)` from
`(B, geometry, mesh)`, accounting for every persistent and transient
tensor and flagging every magic constant.

Conventions:
- `c128 = 16 B`. All sizes below are per-rank c128 bytes.
- `P = p_x · p_y` (mesh size). `p_x = p_y = 4` on the CrI3 4×4 mesh.
- `mu = n_rmu_padded` (always rounded up to a multiple of `P` at
  `Meta` construction; `mu ≈ 1808` on a 16-way mesh when
  `n_rmu_logical = 1800`).
- `n_k = n_q = n_k_tot` in the planner — the C_q / Z_q FFTs go
  `k → q` over the same axis. IBZ reduction only affects `n_q_disk`
  (gflat_acc) and `q_irr_full_idx` (the gather inside
  `fit_one_rchunk`). The C_q / Z_q FFT space is full-BZ.
- `ns = nspinor` (= 2 bispinor, = 1 collinear).
- "Sharded over X" / "over Y" / "over XY" = single mesh axis / both
  flat. Where I use `/P` I mean a tensor whose payload divides
  cleanly across `P` ranks; `/p_x` or `/p_y` means single-axis.
- All "FFT box" lifetimes assume an empirical multiplicative
  `fft_factor` for cuFFT scratch — discussed in §2 and §7.

---

## 1. Tensor catalog

The pipeline has **three temporal phases**: pre-loop (CCT + chol),
inside the r-chunk loop (fit_one_rchunk + accumulate_rchunk_to_gflat),
post-loop (mask + write). I list every device-resident tensor that
appears at any time, with shape, sharding, lifetime, and per-rank
byte formula. Charge vs transverse differences flagged inline.

### 1a. Persistent during the entire chunk loop

| name | shape | sharding | per-rank bytes | notes |
|---|---|---|---|---|
| `psi_l_rmu_Y` | `(nk, nb_L, ns, mu)` | `P(None, None, None, 'y')` | `16·nk·nb_L·ns·mu/p_y` | left ψ at centroids, R-form |
| `psi_l_rmuT_X` | `(nk, mu, nb_L, ns)` | `P(None, 'x', None, None)` | `16·nk·nb_L·ns·mu/p_x` | conjugated, L-form |
| `psi_r_rmu_Y` | `(nk, nb_R, ns, mu)` | `P(None, None, None, 'y')` | `16·nk·nb_R·ns·mu/p_y` | right ψ |
| `psi_r_rmuT_X` | `(nk, mu, nb_R, ns)` | `P(None, 'x', None, None)` | `16·nk·nb_R·ns·mu/p_x` | right ψ, L-form |
| `L_q` (charge) | `(nq, mu, mu)` | `P(None, 'x', 'y')` | `16·nq·mu²/P` | Cholesky factor (`vertex_mu_L=0`) |
| `C_q` passthrough (transverse) | `(nq, mu, mu)` | `P(None, 'x', 'y')` | `16·nq·mu²/P` | identity-padded CCT, fed to per-q LU |
| `cct_trace_per_q` | `(nq,)` complex128 | replicated `()` | `16·nq` | transverse only, one all-reduce hoisted out |
| `gflat_acc` | `(n_q_disk, mu, ngkmax)` | `P(None, ('x','y'), None)` | `16·n_q_disk·mu·ngkmax/P` | donated to accumulate each iter |
| `_q_irr_frac_dev` | `(n_q_disk, 3)` f64 | replicated | `≤ 1 kB` | trivial |
| `psi_G_store` host tiles | `(nk, nb_full, ns, ngkmax)` | per-rank host numpy | **HOST**, 0 device | pulled per-bc via io_callback |
| `q_irr_full_idx` | `(n_q_disk,)` i32 | replicated | small | IBZ gather indices baked into kernel closure |
| `kvecs_frac`, `g_index` tables | small | replicated / per-rank | small | baked into kernel closure |

Note: `psi_l_rmu_Y` and `psi_l_rmuT_X` are **distinct copies on
different mesh-axes** (Y and X respectively). The pair-density kernel
needs an L-form (`P(None, 'x', None, None)`) on one operand and an
R-form (`P(None, None, None, 'y')`) on the other so the gemm's M and
N axes each take one mesh axis only. Total persistent centroid
bytes per rank ≈ `2·16·nk·(nb_L + nb_R)·ns·mu/p_x` if `p_x = p_y`
(the two distinct copies have the same per-rank size because each is
single-axis-sharded).

`gflat_acc` is the only G-flat-shaped buffer; its sharding is
**flat μ over both axes** (`('x','y')`), so each rank holds
`mu/P = n_mu_local` rows × `ngkmax` columns × `n_q_disk` for the
full q.

### 1b. Pre-loop transients (Peak A and Peak B)

These exist only during the centroid-load pass and the CCT/chol
pass, before the r-chunk loop begins. They are released before
fit_one_rchunk fires.

**Peak A — band-chunked centroid extraction (`load_centroids_band_chunked`):**

| name | shape | sharding | per-rank bytes |
|---|---|---|---|
| `psi_G_box_loadchunk` | `(nk, bpd_load, ns, n_rtot)` | `P(None, ('x','y'), …)` | `16·nk·bpd_load·ns·n_rtot/P · fft_factor` |
| `phase_table` | `(nk, n_rtot)` | replicated | `16·nk·n_rtot` |
| centroid output being filled | as in §1a | — | growing |

`bpd_load` (bands-per-load) is the load-time analogue of the
chunk-loop `band_chunk`; today (post-`load_centroids_band_chunked`)
it's separately configurable but is driven by the same `band_chunk`
knob in practice.

**Peak B — CCT + Cholesky:**

The monolithic `c_q_from_psi_sm` shard_map at
`src/common/isdf_fitting.py:296` is fused, so the rank-5 pair-density
intermediates `P_l`, `P_r` (and their k-rank-7 reshapes) are XLA
internal values, not separate user tensors. The three live
preallocated-temp slots inside the shard_map have shape
`c128[nk, ns, n_rmu_loc, n_rmu_loc, ns]` (rank-7 reshape from
`'karmb'` einsum output factoring `(k, ns_l, col, μ, ns_r)`):

| name | shape (per-rank, in the shard_map) | per-rank bytes |
|---|---|---|
| `C_q_flat` | `(nq, mu, mu)` `P(None,'x','y')` | `16·nq·mu²/P` |
| `P_l_R_conj` rank-7 slot | `(nkx,nky,nkz, ns, mu/p_y, mu/p_x, ns)` | `16·nk·ns²·mu²/P` |
| `P_r_R` rank-7 slot | same shape | `16·nk·ns²·mu²/P` |
| XLA scratch | same shape | `16·nk·ns²·mu²/P` |
| `L_q` (after chol) | `(nq, mu, mu)` `P(None,'x','y')` | `16·nq·mu²/P` |
| cuSolverMp internal scratch | ≈ `O(mu²)` class | **unmodeled — see §7** |

The dominant per-rank term is the three pair-density-shaped slots at
`16·nk·ns²·mu²/P` each. At CrI3 80 Ry (nk=36, ns=2, mu≈1808, P=16):
`16·36·4·1808²/16 ≈ 0.47 GB` per slot, ≈ 1.4 GB peak for the three
together. Modest.

`B_pre_total ≈ centroids(persist) + 3·16·nk·ns²·mu²/P + L_q + ψ_G_box_loadchunk`.
At CrI3 80 Ry this is bounded by Peak A's FFT box (a band-load full-
n_rtot tensor), not the CCT slots.

### 1c. Inside `fit_one_rchunk` (Peak C — the binding peak)

The fused kernel cycles through three logical sub-stages per
band-chunk iteration. The bc-loop is **Python-unrolled** at trace
time, so XLA sees `n_bc = ⌈nb_full / band_chunk⌉` copies of the
per-bc sub-trace stacked into one jit. XLA can alias **same-shape,
non-overlapping-lifetime** buffers within and across the unroll, but
cannot reuse a slot whose previous tenant is still live.

Per-bc transients:

| name | shape (logical) | sharding | per-rank bytes | lifetime |
|---|---|---|---|---|
| `psi_G_bc` (post io_callback) | `(nk_slice, bpd, ns, ngkmax)` | `P(None, ('x','y'), None, None)` | `16·nk_slice·bpd·ns·ngkmax/P` | brief |
| `psi_G_FFT_box` (the unsharded pathology) | `(nk_slice, bpd, ns, nx, ny, nz)` | nominally `P(None, ('x','y'),…)`; **see §1f** | `16·nk_slice·bpd·ns·n_rtot/P · fft_factor` | per-bc |
| `psi_r_chunk` (post IFFT, post Bloch, post slab slice) | `(nk_slice, bpd, ns, r_chunk_local)` | `P(None, ('x','y'), None, None)` | `16·nk_slice·bpd·ns·r_chunk/P` | per-bc |
| reshard L-form `psi_l_X_bc` | `(nk, mu_loc, bpd, ns)` | `P(None,'x',None,None)` | `16·nk·bpd·ns·mu/p_x` | per-bc |
| reshard R-form `psi_l_Y_bc` | `(nk, bpd, ns, r_chunk_loc)` | `P(None,None,None,'y')` | `16·nk·bpd·ns·r_chunk/p_y` | per-bc |

`n_bc · α_psi_Y_bc · r_chunk` (the cumulative reshard slab across
the unrolled bc-loop) is a **real, observed cost** that the current
`compute_optimal_chunks` model accounts for (`_fft_moment` at
`gw_init.py:80`) but `gflat_memory_model.py` does **not**. This is
the source of a 17-GB underestimate on CrI3 16-GPU at chunk_r=112016,
band_chunk=16, n_bc=5 quoted in the gw_init `_fft_moment` docstring.

Per-r-chunk transients (alive once per fit_one_rchunk call, scaled
by `r_chunk`):

| name | shape | sharding | per-rank bytes | comment |
|---|---|---|---|---|
| `P_l_R_conj` rank-7 slot | `(nkx,nky,nkz, ns, r_chunk_loc, mu_loc, ns)` | shard_map-internal | `16·nk·ns²·mu·r_chunk/P` | hot |
| `P_r_R` rank-7 slot | same shape | shard_map-internal | `16·nk·ns²·mu·r_chunk/P` | hot |
| XLA scratch slot | same shape | shard_map-internal | `16·nk·ns²·mu·r_chunk/P` | hot, **§6 magic** |
| `Z_q[q, μ_X, r_Y]` | `(nq, mu, r_chunk)` | `P(None,'x','y')` | `16·nq·mu·r_chunk/P` | brief, post-fftn |
| ζ_chunk (post-solve) | `(n_q_disk, mu, r_chunk)` | `P(None, ('x','y'), None)` | `16·n_q_disk·mu·r_chunk/P` | output, lives into accumulate |
| solver scratch (cuSolverMp `potrs`/`getrs`) | `O(mu·r_chunk)` class | distributed | **unmodeled** | §7 |

The pair-density slot count is `pair_density_slots ≈ 3` per the
HLO dump that the current planner references. This is the §5.6
"magic constant" — it controls the entire r-chunk picker.

### 1d. Inside `accumulate_rchunk_to_gflat` (Peak D)

A separate XLA module called immediately after fit_one_rchunk
returns; ζ_chunk is the only fit_one_rchunk transient still live
when accumulate starts. Inputs are donated.

Per-scan-iter transients (alive once per scan iteration of size
`cs = gflat_chunk_size`):

| name | shape | sharding | per-rank bytes | comment |
|---|---|---|---|---|
| `sub` (slice of rchunk_flat) | `(cs, r_chunk)` | `P('x','y')`-internal | `16·cs·r_chunk` | per-rank slab |
| `buf` (zero-padded FFT box pre-) | `(cs, n_rtot)` | per-rank | `16·cs·n_rtot` | hot |
| `box` reshape `(cs, nx, ny, nz)` | same payload | per-rank | bitcast | — |
| `G = fftn(box)` | `(cs, n_rtot)` | per-rank | `16·cs·n_rtot · fft_factor` | cuFFT scratch |
| phase tables `phx/phy/phz` | `(nq, nx)` etc. | replicated | `16·nq·(nx+ny+nz)` | trivial |

The FFT-box transient `cs · n_rtot · 16` dominates Peak D. Note
that this is per-rank, **not** divided by P — every rank does its
own local FFT on its own per-rank μ-slab, so there is no sharding
to apply to the box axis.

### 1e. Bispinor differences (charge vs transverse)

| quantity | charge (μ_L=0) | transverse (μ_L=1,2,3) | source |
|---|---|---|---|
| `n_rmu` | `n_rmu_C ≈ 1800` | `n_rmu_T ≈ 1200` | report.md §4 |
| L_q factor | `potrf` (Cholesky), lower-tri | `getrf` (LU), full matrix | `factor_c_q`, `_resolve_solver_kind_*` |
| solve | `potrs` 2× tri-solve | `getrs` LU back-substitute (cuSolverMp); per-q `jnp.linalg.solve` (legacy) | `solve_zeta` |
| extra trace `cct_trace_per_q` | not stored (None) | stored: `(nq,)` complex128 | `isdf_fitting.py:1716` |
| γ̃ folding | identity short-circuit; perm/phase ignored | gather + element-wise phase multiply at γ̃-contract step | `gamma_double_contract` |
| identity-pad on factor | both | both | same code path |

Memory consequence: transverse has **smaller** `mu`, so its
persistent and per-r-chunk pair-density terms are smaller by
roughly `(n_rmu_T / n_rmu_C)²` on the L_q / centroids axes and
`(n_rmu_T / n_rmu_C)` on the pair-density per-cr term. **Therefore
the charge channel always sets the binding r_chunk budget** — the
planner must size to fit charge; transverse fits trivially at the
same r_chunk.

The current planner (`plan_gflat_chunks`) **does not branch on
`vertex_mu_L`** at all; it always uses `meta.n_rmu_padded`, which
is set to the *charge* centroid count. That's the right thing to
do if charge is always the larger channel, but it's an implicit
assumption — see §7.

### 1f. The unsharded ψ(G) FFT box

`psi_G_store.fetch_psi_rchunk` (`src/common/psi_G_store.py:268`)
builds an `(nk_slice, bpd, ns, nx, ny, nz)` FFT box inside its inner
`to_rchunk` call. The intended sharding is
`P(None, ('x','y'), None, None, None, None)` (band axis flat-sharded),
giving per-rank size `16·nk·bpd·ns·n_rtot/P · fft_factor`. **XLA
empirically materialises this tensor unsharded at CrI3 6×6 80 Ry
scale** (report.md §5.8, §7) — i.e. the per-rank cost becomes
`16·nk_slice·bpd·ns·n_rtot · fft_factor` (no `/P`). At CrI3 80 Ry
with `nk_slice = nk = 36`, `bpd = band_chunk/P = 16/16 = 1` (or
`bpd = 16` if XLA decides band isn't really sharded either),
`ns = 2`, `n_rtot ≈ 1.13M`:

- sharded ideal: `16·36·1·2·1.13M / 16 · 4 = 0.32 GB · 4 = 1.3 GB`
- unsharded ψ_G_box: `16·36·16·2·1.13M · 4 ≈ 83 GB` per rank (!) at
  `band_chunk = 16`
- with `psig_k_chunk_size = 6` capping nk_slice: same formula with
  `nk_slice = 6` → `16·6·16·2·1.13M · 4 ≈ 14 GB` per rank.

This is what the manual `psig_k_chunk_size = 6` knob mitigates. The
current planner does **not** model the unsharded case — it always
computes W_wfn assuming the sharding holds. This is a known bug
(report.md §5.8 fix priority #1).

---

## 2. Aliasing analysis

Aliasing rules (XLA BufferAssignment behavior):

1. Two buffers can occupy the same physical slot iff their
   **lifetimes do not overlap**, i.e. there is no instruction at
   which both are live.
2. Sum vs max: if A is freed strictly before B is allocated, peak
   sees `max(A, B)`; if they overlap, peak sees their sum.
3. Donation (`donate_argnums`) lets an output reuse an input's
   buffer — the input is freed at the moment the output is born.
4. Same-shape constraint: aliasing requires matching c128 byte
   counts (XLA's allocator is shape-aware). Different shapes only
   alias if they share a preallocated-temp slot of the larger size,
   wasting the difference.

### 2a. Inside `fit_one_rchunk` (one jit)

The fused kernel structure is (per bc, then once after the bc-loop):

```
for bc in bc_ranges:
    psi_G_bc = io_callback(...)              # transient
    psi_G_box = scatter_to_box(psi_G_bc)      # transient (THE unsharded slot)
    psi_r_chunk = ifft(psi_G_box)             # consumes psi_G_box
    psi_r_chunk = bloch_phase(psi_r_chunk)    # in-place mul
    psi_l_X_bc, psi_l_Y_bc = reshard(psi_r_chunk)  # 2× output, input freed
    P_l_slot, P_r_slot, scratch_slot = c/z_q_from_psi_sm(  # pair pipeline
        psi_*_X_bc, psi_*_Y_bc, …)            # rank-7 internally
    Z_q += local_contribution                 # accumulator over bc
Z_q reshard for solve
ζ_chunk = potrs(L_q, Z_q)                     # or getrs / SVD pseudoinverse
```

**Overlaps and aliases inside one bc:**
- `psi_G_box` and `psi_r_chunk` cannot alias: cuFFT needs both
  input and output live during the FFT call, and `psi_G_box` is
  bigger by `fft_factor`. Once IFFT completes, `psi_G_box` can be
  freed (it isn't read again).
- `psi_l_X_bc` and `psi_l_Y_bc` are simultaneously live during the
  pair-density einsum (`'kmna,knbr → karmb'` reads both).
- Inside `c_q_from_psi_sm`/`z_q_from_psi_sm`: `P_l_R_conj`, `P_r_R`,
  and an XLA scratch are all rank-7 buffers of the same shape that
  are **simultaneously live during `gamma_double_contract`**. This
  is the `pair_density_slots = 3` count (HLO-derived).
- Z_q is born when the pair pipeline returns; it can alias one of
  the freed pair-density slots if shapes match (they don't in
  general — Z_q is rank-3, slots are rank-7 with same byte count
  for charge/transverse since `nk · ns² · mu_loc · r_chunk_loc` =
  `nq · mu · r_chunk / P` ⇔ `ns² = P/(p_x · p_y) · …` — equal only
  by coincidence). Conservatively assume no alias.

**Across the Python-unrolled bc-loop:**
- The 3 pair-density slots from bc_i are freed by the end of the
  pair pipeline's contribution to Z_q; bc_{i+1}'s pair-density
  slots can reuse the same physical memory. So slot count is **not
  multiplied by n_bc** — it stays at 3 across the unroll.
- The reshard outputs `psi_l_Y_bc[r_chunk]` are read again only
  when the pair-density einsum of the *same* bc runs. But XLA
  empirically keeps them live across the unroll to overlap io_callback
  fetch with prior bc's accumulate — measured cumulative-live cost
  `n_bc · α_psi_Y_bc · r_chunk` in `_fft_moment` docstring. This is
  a **schedule-dependent** cost; in principle one cleaner unroll
  could free each `psi_l_Y_bc` before fetching the next, but XLA's
  current schedule doesn't.

**Across the r-chunk loop** (outer Python loop):
- fit_one_rchunk is a fresh jit call per r-chunk. Between calls
  XLA frees all transients except the donated/returned tensors
  (`gflat_acc`, `L_q`, centroids).
- ζ_chunk is the output of fit_one_rchunk and the input to
  accumulate_rchunk_to_gflat. They are **not** in the same jit, so
  XLA must materialise ζ_chunk fully before calling accumulate.
  But ζ_chunk is then donated to accumulate's scan, so its memory
  is released as the scan walks the (q·μ_local) axis.

### 2b. The binding aliasing rule (where the report.md §5.2 model
comes from)

`fit_one_rchunk` has three "phases" each owning a workspace block:

| phase | workspace block | size |
|---|---|---|
| W_wfn (ψ FFT + reshard) | psi_G_box + psi_r_chunk (or 1 alias) | `~16·nk·bpd·ns·n_rtot/P · fft_factor` |
| W_zeta (pair pipeline + solve) | `pair_density_slots` rank-7 slots + Z_q + ζ_chunk | `pair_density_slots · 16·nk·ns²·mu·r_chunk/P` |
| W_accum (only after fit_one_rchunk returns) | `cs · n_rtot · 16 · fft_factor` | — |

`W_wfn` and `W_zeta` are **not in the same XLA module hot phase** —
W_wfn peaks during the bc-loop ψ-fetch+reshard; W_zeta peaks during
the post-bc-loop pair pipeline and solve. Their lifetimes inside
the *same* jit overlap during the pair-density einsum (the reshard
outputs are *inputs* to the einsum, so still live when slot 1 of
the pair pipeline allocates). But once the pair pipeline starts
streaming its rank-7 IFFT, the reshard outputs are mostly past their
last use — XLA can alias one rank-7 slot into the freed reshard
slab.

The **principled aliasing model** (report.md §5.2): treat W_wfn
and W_zeta as separate workspaces sharing a `W_pool` such that
**peak ≤ B_persist + max(W_wfn, W_zeta)** *if* XLA aliases the
shared region — but **peak = B_persist + W_wfn + W_zeta** *if* it
doesn't. The 50/50 split in current `plan_gflat_chunks` is a
hedge against the pessimistic case; the §5.5 1/`pair_density_slots`
analysis is the principled-but-optimistic case. Honest answer:
**both are guesses** until HLO confirms.

`W_accum` is in a **separate jit** (called after fit_one_rchunk
returns), so its full budget is `B − B_persist − (live outputs of
fit_one_rchunk)`. The only live output is ζ_chunk itself which is
donated to accumulate. So `W_accum ≤ B − B_persist`.

---

## 3. Budget equations

Let `B` = budget (e.g. `60 GB` on Perlmutter, `0.97·memory_per_device_gb·1e9` in code).

### 3a. B_persist (always alive)

```
B_persist = B_psi_l_Y + B_psi_l_X + B_psi_r_Y + B_psi_r_X
          + B_L_q
          + B_gflat_acc

with
  B_psi_l_Y = 16 · nk · nb_L · ns · mu / p_y
  B_psi_l_X = 16 · nk · nb_L · ns · mu / p_x
  B_psi_r_Y = 16 · nk · nb_R · ns · mu / p_y
  B_psi_r_X = 16 · nk · nb_R · ns · mu / p_x
  B_L_q     = 16 · nq · mu² / P
  B_gflat_acc = 16 · n_q_disk · mu · ngkmax / P
```

(Transverse channel also adds `16·nq` for `cct_trace_per_q` —
negligible.)

### 3b. W_pool

```
W_pool = B − B_persist
```

### 3c. W_wfn(band_chunk, k_chunk)

Assuming the FFT box **is sharded as advertised**:

```
W_wfn_sharded = 16 · k_chunk · band_chunk · ns · n_rtot / P · fft_factor
              + 16 · nk · band_chunk · ns · r_chunk / P    (reshard slab term)
              · n_bc                                       (per Python-unroll concurrent-live)
```

Assuming the FFT box is **unsharded** (CrI3 pathology):

```
W_wfn_unsharded = 16 · k_chunk · band_chunk · ns · n_rtot · fft_factor
                + 16 · nk · band_chunk · ns · r_chunk / P · n_bc
```

`k_chunk = psig_k_chunk_size` when non-zero, else `nk`. `band_chunk
= cohsex.band_chunk_size` or planner pick.

### 3d. W_zeta(r_chunk)

```
W_zeta = pair_density_slots · 16 · nk · ns² · mu · r_chunk / P
       + 16 · nq · mu · r_chunk / P                 (Z_q transient)
       + 16 · n_q_disk · mu · r_chunk / P           (ζ_chunk output)
```

In practice the first term dominates by an `ns²` factor (= 4 on
bispinor) and the `nq` term is `nk` so they're commensurate.

A defensible reduction: `W_zeta ≈ (pair_density_slots + 1) · 16 ·
nk · ns² · mu · r_chunk / P`. Keeping pair_density_slots=3
as the magic constant.

### 3e. W_accum(gflat_chunk_size)

```
W_accum = 16 · cs · n_rtot · fft_factor          (per-iter FFT box)
        + 16 · cs · r_chunk                       (slab slice)
        + 16 · cs · ngkmax                        (acc slice + contrib)
```

The first term dominates because `n_rtot >> r_chunk, ngkmax`.

### 3f. W_vq (post-r-chunk, V_q pass — separate budget)

Not part of the zeta-fit planner, but worth recording:

```
W_vq_persist = B_gflat_acc' (one IBZ q at a time read from disk)
             + n_rmu² · 16 / P
W_vq_transient = 16 · n_rmu · vq_g_chunk_size / P   (per-G-chunk GEMM)
```

The current planner ignores this and `_pick_g_chunk` caps at 4096.
**Not a zeta-fit concern**, but the unified planner should pick
both passes consistently — flagged as §7 open question.

### 3g. Pre-loop peaks (Peak A, Peak B)

These must also fit under `B`, but they're transient and don't
constrain the chunk-loop knobs except via shared persistent
allocations.

```
Peak_A = B_psi_*_*_partial    (centroid output being filled)
       + W_wfn(bpd_load, nk)
       + 16 · nk · n_rtot         (phase table)

Peak_B = B_centroids_full      (all 4 copies)
       + 3 · 16 · nk · ns² · mu² / P      (3 pair-density slots, mu², not mu·r_chunk)
       + B_L_q                  (= B_C_q)
       + cusolvermp_scratch     (unmodeled, ~O(mu²))
```

Peak_B is the **only place** the `mu²` term appears in a transient.
This is the term that's at most ~`(mu/r_chunk)` times bigger than
the per-r-chunk Peak C term — at CrI3 80 Ry, mu ≈ 1800 vs r_chunk
≈ 12500, so Peak B per-cr-equivalent term is `1800/12500 ≈ 0.14`,
i.e. Peak B is smaller than Peak C in the rank-7 slots. Modest.

---

## 4. r_chunk picker procedure

Goal: maximize r_chunk subject to **all peaks fit under B**, with
secondary maxima on band_chunk and gflat_chunk_size.

### 4a. Inputs

`(B_bytes, mesh: p_x, p_y, meta: nk, ns, mu, nq, n_rmu_logical,
n_rtot, ngkmax, n_q_disk, nb_L, nb_R, vertex_mu_L, is_bispinor)`,
plus overrides `(band_chunk_override, r_chunk_override,
psig_k_chunk_override, gflat_chunk_override)`.

### 4b. Step-by-step

```
1. Compute B_persist (§3a).
   If B_persist > 0.95 · B: raise — no room for any transient.

2. W_pool = B − B_persist.

3. Pick band_chunk:
   - if band_chunk_override: use it.
   - else: largest pow-2 ≤ nb_full s.t.
       W_wfn(band_chunk, k_chunk = nk) ≤ W_pool / pair_density_slots
     (the slot-budget ceiling for clean aliasing into a pair-density slot)
   - if even band_chunk = P fails: drop to band_chunk = P and
     reduce k_chunk in step 4. Otherwise round up to a multiple of P.

4. Pick psig_k_chunk_size (handle the unsharded pathology):
   - if W_wfn_sharded(band_chunk, nk) ≤ W_pool/pair_density_slots
     AND XLA-sharding-trustworthy at this scale: k_chunk = nk.
   - else (use unsharded formula to be safe):
       k_chunk = largest int s.t. W_wfn_unsharded(band_chunk, k_chunk)
                 ≤ W_pool/pair_density_slots
   - "XLA-sharding-trustworthy" gate: I'd default to "assume
     unsharded" at large n_rtot (≥ ~500k) and shard-trustworthy
     below — see §7 for why we can't currently decide this from
     the planner.

5. Pick r_chunk:
   r_chunk ≤ W_pool / α_zeta
     where α_zeta = (pair_density_slots + 1) · 16 · nk · ns² · mu / P
   r_chunk = min(W_pool/α_zeta, n_rtot)
   r_chunk = max(r_chunk, mu)     (don't pay r-chunk overhead for less
                                   than mu r-points of work — per gflat_memory_model)
   r_chunk = floor(r_chunk / P) · P    (sharding divisibility on the
                                        solve output)
   For BISPINOR runs: use mu = n_rmu_C (charge), not n_rmu_T.
   Charge is always the larger channel → fitting charge fits transverse.

6. Pick gflat_chunk_size (one-shot if feasible):
   N_rows = n_q_disk · mu / P    (per-rank flat (q·μ_loc) row count)
   if W_accum(N_rows) ≤ W_pool: cs = None (one-shot)
   else: cs = floor(W_pool / (16 · n_rtot · fft_factor))
         (then n_chunks = ceil(N_rows / cs))

7. Validate by replaying all peaks (A, B, C, D) at the chosen knobs
   and verify max_peak ≤ B. If not, halve band_chunk and retry from
   step 3.
```

### 4c. Bispinor handling

The fit is run four times (one per `vertex_mu_L ∈ {0,1,2,3}`). The
**planner runs once** with `mu = n_rmu_C` and the resulting
`(r_chunk, band_chunk, gflat_chunk_size, psig_k_chunk_size)` is
re-used for all four channels:

- charge sets the binding budget (larger mu).
- transverse fits the same knobs with slack.
- The trace `cct_trace_per_q` is hoisted out of the solve in the
  transverse case — adds `16·nq` bytes, negligible.

A separate consideration: the LU path (`getrs`) has different
scratch behaviour than the Cholesky path (`potrs`). Both are
distributed in cuSolverMp 0.7.2; relative sizes I don't know
exactly (§7).

### 4d. The unsharded-FFT-box decision tree

Today's code uses `psig_k_chunk_size = 6` as a manual override at
CrI3 scale. A planner that auto-picks it would need to know
whether XLA will or won't shard the box. Options:

1. **Always use the unsharded formula in step 4 above.** Slightly
   conservative on small problems where the box IS sharded, but
   correct everywhere. Cost: at MoS2 3×3 we'd pick smaller k_chunk
   than necessary; at MoS2 the box fits trivially anyway, so
   harmless.
2. **HLO-grep at config time:** compile fit_one_rchunk in AOT mode
   on a tiny problem with the same `(nk, band_chunk, ns)` and check
   whether the FFT-box thunk in the resulting HLO carries a
   sharding annotation. Reliable but expensive.
3. **Carry the fix to the source.** Locate the unsharded
   materialisation site in `to_rchunk` / `psi_G_store` and add a
   `with_sharding_constraint` at the *creation* point (the
   call-boundary constraint apparently didn't work — report.md
   §6). Best long-term answer; out of scope for the planner.

For the planner I'd take option 1 (always use unsharded). Open
question: at what `n_rtot` does this become noticeably more
conservative than option 2?

---

## 5. Validation at CrI3 80 Ry

Plug in: `n_k = n_q = 36`, `ns = 2`, `n_rmu_C = 1800`,
`n_rmu_padded ≈ 1808` (next multiple of 16), `n_rtot = 75·75·200 =
1,125,000`, `nb_L ≈ 400`, `nb_R ≈ 400`, `nb_full ≈ 400`,
`ngkmax ≈ 0.06 · n_rtot ≈ 67,500` (per gw_init's default estimate;
could be smaller with explicit zeta_cutoff), `n_q_disk ≈ 9–11`
(IBZ-reduced from 36, factor ~3.5×), `P = 16`, `p_x = p_y = 4`,
`B = 60 GB · 0.97 = 58.2 GB`, `pair_density_slots = 3`,
`fft_factor = 4`.

### 5a. B_persist

```
B_psi_l_Y = B_psi_r_Y = 16·36·400·2·1808 / 4 = 0.208 GB
B_psi_l_X = B_psi_r_X = 16·36·400·2·1808 / 4 = 0.208 GB
B_centroids_total = 4 · 0.208 = 0.83 GB

B_L_q = 16·36·1808² / 16 = 0.117 GB

B_gflat_acc (with n_q_disk = 10, ngkmax = 67500):
            = 16·10·1808·67500 / 16 = 1.22 GB

B_persist ≈ 0.83 + 0.12 + 1.22 ≈ 2.17 GB
W_pool   ≈ 58.2 − 2.17 ≈ 56 GB
```

### 5b. α_zeta and r_chunk

```
α_zeta = (3 + 1) · 16 · 36 · 4 · 1808 / 16
       = 4 · 16 · 36 · 4 · 1808 / 16
       = 1.04 MB/r-unit per rank

r_chunk_budget = W_pool / α_zeta = 56e9 / 1.04e6 ≈ 53,800
r_chunk = min(53,800, n_rtot=1.125M, after clamp by Peak_C aliasing) ≈ 53,800
```

But this **assumes W_wfn fits inside the same pool without
contention**. If we instead apply the §4b step-3 ceiling
(W_wfn ≤ W_pool / pair_density_slots = 18.7 GB) and the §4b step-5
ceiling (W_zeta ≤ W_pool = 56 GB), my formula gives r_chunk ≈
53,800.

The empirical working number is **~12,500**, ~4× smaller than my
formula's prediction. **Why the discrepancy?**

Candidate explanations:
1. The "magic" `pair_density_slots = 3` is wrong for the
   `karmb`-ordered kernel at CrI3 scale. The legacy `'kabmr'`
   chain had 5 slots; if the bispinor case actually runs with 4–5
   slots at CrI3 (e.g. because a `with_sharding_constraint` between
   the einsum and the IFFT prevents one alias), α_zeta is 1.5–2×
   bigger and r_chunk ≈ 27–35k. Still 2× larger than 12,500.
2. **The unsharded ψ_G FFT box dominates.** At `band_chunk = 16,
   psig_k_chunk = 6`: `W_wfn_unsharded = 16·6·16·2·1.125M·4 ≈ 13.8
   GB`. That leaves `W_pool − W_wfn ≈ 42 GB` for W_zeta, and
   `r_chunk ≤ 42e9 / 1.04e6 ≈ 40,400`. Still bigger than 12,500.
3. The `n_bc · α_psi_Y_bc · r_chunk` Python-unroll term (counted
   in gw_init's `_fft_moment` but not gflat_memory_model). With
   `n_bc = ⌈800/16⌉ = 50, α_psi_Y_bc = 16·36·16·2/4 ≈ 4.6 KB/cr,
   contribution at r_chunk = 12500 ≈ 2.9 GB`. Not a 4× factor.
4. cuSolverMp scratch. Unknown — could easily be `O(n_rmu² · P_q)`
   per panel, gigabytes at mu = 1808.
5. **`max_chunks` floor in plan_gflat_chunks:** `r_chunk ≥
   ceil(n_rtot / 64) = 17,578`. So 12,500 *cannot* be coming from
   the current gflat planner — that planner's output should be
   ≥ 17,578. Either the empirical 12,500 comes from
   `compute_optimal_chunks`'s closed-form moments (which set
   r_chunk by the binding ZCT/FFT moment, not by a max_chunks
   floor), or from manual cohsex override, or it's a stale
   number in the recipe.

**My honest call**: I don't have a clean derivation that lands on
12,500. The closest defensible answer from my from-scratch model
is r_chunk in the range **20k–50k**, depending on how I credit the
unsharded FFT box and pair_density_slots. To match 12,500 I'd
need either (a) `pair_density_slots ≈ 14`, (b) a missing
multi-GB cuSolverMp scratch term, (c) a missing live tensor I
haven't catalogued, or (d) the value isn't actually planner-derived.

Open question for the discussion phase: **what actually picks
r_chunk = 12,500?** Is it the legacy compute_optimal_chunks model
(via its `_zct_moment` 5-slot count), the gflat_memory_model
(but how, given the max_chunks floor), the AOT chooser, or a
human override I haven't traced?

### 5c. Peak A / Peak B sanity at CrI3 80 Ry

```
Peak_A FFT box (sharded): 16·36·bpd_load·2·1.125M/16 · 4
  = bpd_load · 1.6 GB ⇒ at bpd_load = 16: 26 GB.
  + centroid output partial (small).
  → 26 GB ≤ 58 GB ✓ (assumes sharding holds; if unsharded:
    bpd_load = 16: 415 GB, OOM. Same pathology as W_wfn — needs
    psig_k_chunk_size mitigation here too, or the load passes
    multiple bc-passes through.)

Peak_B: B_centroids (0.83) + 3 pair-density rank-7 slots at mu²
  = 3 · 16 · 36 · 4 · 1808² / 16 = 0.94 GB + cusolvermp scratch.
  ≪ B. ✓
```

### 5d. gflat_chunk_size sanity

```
N_rows = n_q_disk · mu / P = 10 · 1808 / 16 = 1130
W_accum(one-shot, cs = 1130) = 16 · 1130 · 1.125M · 4 ≈ 81 GB
  → too big.
cs_budget = W_pool / (16 · n_rtot · fft_factor)
          = 56e9 / (16 · 1.125M · 4) ≈ 775 rows
But we also want some headroom — at cs = 64 (the recipe):
  W_accum = 16 · 64 · 1.125M · 4 ≈ 4.6 GB ≪ 58 GB ✓
```

Recipe `gflat_chunk_size = 64` is comfortable and matches my
formula at ~14× under the cs_budget — leaves room for ~14 chunks
× cs=64 if XLA can't alias, gives a single-rank peak under 5 GB.
Good.

### 5e. psig_k_chunk_size sanity

Recipe uses `psig_k_chunk_size = 6`. My §5b candidate-2 says
`W_wfn_unsharded ≈ 13.8 GB` at k_chunk=6, band_chunk=16. That's
22% of B = 60 GB. Tight but feasible. Halving k_chunk to 3 would
halve W_wfn → 7 GB; at k_chunk = nk = 36 → 83 GB OOM. So 6 is in
the right neighborhood. The auto-planner should land at
`k_chunk ∈ {4, 6, 9}` depending on how conservative the unsharded
budget is.

---

## 6. Diff against the current code

| current piece | role | verdict |
|---|---|---|
| `compute_optimal_chunks` (gw_init.py:154) | "6-stage" model; closed-form min over `(headroom − c_i)/α_i` per stage | Has the right *structure* (linear-in-cr per stage) but the stage list mixes lifetimes inappropriately. `_zct_moment` uses **5 slots** (2 persistent P_l/P_r + 3 transient), inconsistent with the gflat planner's 3. The `_fft_moment` adds the `n_bc · α_psi_Y_bc · cr` term that the gflat planner is missing. Result: this model gives smaller `r_chunk` than the gflat planner. Keep its `n_bc` term; rationalise the slot count against HLO. |
| `gflat_memory_model.plan_gflat_chunks` | A/B/C/D per-peak model; 50/50 W_wfn vs W_zeta split | Right shape but underspecified. Misses: (a) `n_bc · α_psi_Y_bc · r_chunk` term entirely, (b) cuSolverMp scratch, (c) unsharded-FFT-box pathology, (d) does not differentiate charge vs transverse — uses `meta.n_rmu_padded` (charge). The 50/50 split is hand-tuned, not principled. `max_chunks=64` floor is opaque. **Closest to right; this is the one I'd evolve.** |
| `aot_memory_model/` (chooser + DoE-fit kernels) | Regressed `α·primitive` model; FLOP-cost tiebreak | Architecturally interesting but pays high complexity cost for what's a fundamentally simple closed-form problem. The regressed primitives (`load_psi_rchunk_fft β=3`, `zct_lr β=4`, etc.) encode the same HLO-derived slot counts; the analytic chooser inverts them in closed form (`choose_chunks_analytic`). The 20/80 heuristic chooser inside this file is the **simplest defensible model** — fixed `wfn_workspace_frac = 0.2`, derived from physical slot counts (3-copy FFT, 4-slot ZCT). It hard-codes `pair_temp_count = 4` (not 3 or 5) — different from both other models. Worth keeping the analytic chooser if anyone has time to maintain DoE fits; otherwise the 20/80 heuristic alone is fine. |

### 6a. What to keep, what to remove

Keep:
- `gflat_memory_model.plan_gflat_chunks` overall A/B/C/D structure.
- The peak A/B closed-form formulas (correct as written).
- `aot_memory_model/chooser.choose_chunks_heuristic` as a fallback /
  sanity-check second-opinion path. Its 80/20 split is the most
  honest single-knob hedge against unmodelled terms.

Remove or merge:
- `compute_optimal_chunks` — superseded by gflat_memory_model.
  Keep only the `_fft_moment(n_bc=…)` formula as a math reference
  for the cumulative-bc-unroll cost that needs to fold into
  gflat_memory_model's Peak C.
- The DoE fitting infrastructure in `aot_memory_model/kernels/`,
  unless someone is actively re-running sweeps. The analytic
  closed-form chooser is what's actually doing the work; the
  primitive fits are a footprint maintenance burden.

Add:
- Peak C must include `n_bc · 16·nk·band_chunk·ns·r_chunk/p_y`
  (cumulative reshard slab across the unrolled bc-loop). Currently
  missing entirely from the gflat planner.
- cuSolverMp scratch as a flagged-but-bounded line item.
  Conservative bound: `O(n_rmu² / panel_count)` per active panel
  ≈ tens of MB at mu=1800 distributed across P=16. Probably small;
  needs a one-shot HLO measurement.
- Unsharded FFT-box auto-detection or default-to-unsharded
  formula in W_wfn. See §4d.
- Differentiated mu for charge vs transverse: at least an assertion
  that the planner is being run with `mu = max(n_rmu_C, n_rmu_T)`,
  with a log line documenting which channel is binding.
- A `W_vq` term so `vq_g_chunk_size` falls out of the same model
  rather than `_pick_g_chunk(ngkmax)` capped at 4096.

### 6b. Where the 50/50 split sits vs report.md §5.5

Report §5.5 advocates `band_chunk` ceiling at `W_pool /
pair_density_slots = W_pool/3` (≈ 33%), based on the assumption
that XLA can alias W_wfn into one freed pair-density slot once the
pair pipeline starts. This is **more aggressive than 50/50**.

My honest take: I don't know which is right without an HLO dump.
The aliasing requires that (a) the freed slot has matching shape
(unlikely — psi_r_chunk vs P_pair rank-7 are different shapes),
and (b) the lifetime ordering puts the free strictly before the
allocate, which depends on XLA's schedule. The 50/50 split is a
conservative hedge that I'd keep as the **default** with a 1/3
ceiling as an opt-in `--aggressive` mode. The cost of being wrong
is exactly what we're seeing on CrI3: an extra 2× r-chunk count
and the FFT tax that goes with it.

---

## 7. Open questions

This is the section that matters. Things I genuinely could not
resolve from code + reports.

### 7.1. What actually picks r_chunk = 12,500 at CrI3 80 Ry?

The empirical "auto" value of ~12,500 doesn't fall out of any
formula I can defend from the code. The `gflat_memory_model`
should pick ≥ 17,578 (the `max_chunks=64` floor) or much larger
(my 53,800 derivation). The legacy `compute_optimal_chunks` has
different slot counts but I haven't traced through how its
moments land at 12,500 either. **Need a one-line dump from a
live CrI3 run: which planner produced this number, and what was
the binding stage?** Until I see that, my model can't be
validated.

### 7.2. `pair_density_slots`: 3, 4, or 5?

Three competing values in the repo:
- `gflat_memory_model.pair_density_slots_charge/transverse = 3`
  (HLO-derived, monolithic-shard_map era, 2026-05-13).
- `compute_optimal_chunks._ZCT_ADDITIONAL_COEF = 3` → 5 total
  (2 persistent + 3 transient).
- `aot_memory_model heuristic.pair_temp_count = 4`.

Only one is right at any given XLA version. The 3 is documented
as the "monolithic shard_map (2026-05-13)" count; the 5 is
documented as the "legacy decomposed chain" count; the 4 is
undocumented. **Need a fresh HLO dump from a current build to
confirm 3 is still right.** This is the §5.6 magic constant and
its drift is the model's single biggest fragility.

### 7.3. Does XLA shard the ψ_G FFT box?

The known pathology: at CrI3 6×6 80 Ry, the FFT box inside
`psi_G_store.fetch_psi_rchunk` is materialised unsharded on every
rank. The W_wfn formula in current code assumes it's sharded; the
manual `psig_k_chunk_size = 6` mitigates. **Need either:**
1. The HLO dump from CrI3 80 Ry confirming the unsharded
   intermediate; or
2. The threshold n_rtot above which XLA bails on sharding (is it
   really a per-`n_rtot` thing, or per-`bpd · n_rtot` byte count,
   or a different heuristic?).

Without (1) or (2) the planner can only choose between "always
assume unsharded" (over-conservative on small problems) and
"always assume sharded" (current behaviour, OOMs on CrI3).

### 7.4. cuSolverMp internal scratch

`potrf` / `potrs` / `getrf` / `getrs` in cuSolverMp 0.7.2 allocate
some panel-scratch internally during the distributed factor /
solve. Not modeled. **How big?** Order of magnitude: at mu=1808
and P=16, a single panel might be ~`(n_rmu / panel_count)²`. I
don't know cuSolverMp's internal block size. A test: at fixed
problem size, sweep mu (or grow it artificially with the pad-block)
and watch the peak — if there's a `O(mu²)` term the planner is
missing, it shows up here.

### 7.5. Bispinor transverse n_rmu_T independence

The planner assumes `n_rmu_T < n_rmu_C` so fitting charge fits
transverse. **Is this always true?** The bispinor design report
(`v_q_bispinor_plan_2026-05-08`) doesn't pin a strict ordering. If
a future system flips it, the planner would silently under-budget
transverse. Defensive fix: assert `mu = max(n_rmu_C, n_rmu_T)` in
the planner explicitly, even if today they're set identically.

### 7.6. `fft_factor = 4.0` — site-dependent?

The single scalar covers three call sites:
- Peak A: pre-loop band-load FFT (large box per call, no scan).
- Peak C: per-bc FFT inside fit_one_rchunk (k-axis chunked, can
  re-use cuFFT plan across bc-iterations).
- Peak D: per-cs FFT inside accumulate (small box, many scan
  iterations).

These three FFT call sites have different fusion neighbourhoods
and different cuFFT plan-cache hit rates. Using one `fft_factor`
is a 10%-class approximation today but could drift at larger
n_rtot or if XLA inlines differently. **Need three separate
calibrations**, ideally as a sweep at one fixed problem size:
measure W_wfn (Peak A and Peak C) and W_accum (Peak D)
separately from HLO and back out per-site factors.

### 7.7. ngkmax estimate for `gflat_acc`

`gw_init.py:596` falls back to `0.06 · n_rtot` when meta lacks an
explicit ngkmax. With explicit `zeta_cutoff_ry`, the actual ngkmax
can be much smaller (`compute_per_q_bare_coulomb_components`
returns the true value). The planner is called *before* the
sphere construction, so it has to estimate. At CrI3 80 Ry with
zeta_cutoff = 80 Ry, what's the actual ngkmax? The 0.06 estimate
could be 2× off in either direction, which moves `B_gflat_acc`
by a few GB — not binding at the chunk-loop level but eats into
W_pool. Need to either thread the cutoff into the planner or
accept the over-estimate.

### 7.8. `n_bc` Python-unroll cumulative cost — model or measure?

`_fft_moment` claims XLA keeps `n_bc · α_psi_Y_bc · r_chunk` live
concurrently across the unrolled bc-loop. This is **schedule-
dependent** — a future XLA version might serialise the bc-loop
differently and free per-bc state aggressively. Is this term
robust or transient? If it stops applying, the planner would
over-budget by `(n_bc − 1) · α_psi_Y_bc · r_chunk` and pick
chunks that are too small. Need to: (a) re-confirm this term in
the current XLA, and (b) consider whether `lax.scan` over the
bc-loop (instead of Python unroll) would let XLA reuse one slot
and eliminate the term.

### 7.9. Why does plan_gflat_chunks have a `max_chunks=64` floor?

The comment says "so we don't blow up chunk count" but it's a
lower bound on `r_chunk`, i.e. an *upper* bound on the chunk
count. With B_persist + W_zeta + n_bc·… all linear in r_chunk,
the chunk count cap is already implicit. Why 64? Is it tied to
the IBZ q-count, the number of band-chunks, or just an
arbitrary cap to limit per-r-chunk Python-iteration overhead? If
arbitrary, it's the wrong knob in a model that's trying to
maximise r_chunk anyway.

### 7.10. Joint W_pool vs separate W_pool for W_wfn/W_zeta/W_accum

Report §5.5 claims W_wfn, W_zeta, W_accum each have one knob
that goes into one budget term, and no two budget terms
interact. **But W_wfn and W_zeta are in the same XLA module** —
they DO share a physical pool, and the interaction is governed by
whether XLA aliases their slots. This is the precise question §2b
left open. The 50/50 split vs 1/`pair_density_slots` debate is a
direct consequence of this unresolved question.

Cleanest resolution: pick `band_chunk` such that
`W_wfn(band_chunk, k_chunk = nk) + W_zeta(r_chunk) ≤ W_pool` for
the chosen `r_chunk` (i.e. assume no aliasing). This is pessimistic
but unambiguous. The planner then leaves r_chunk smaller than
strictly necessary, but never OOMs.

### 7.11. Sharding behaviour of the `karmb` einsum output

The CCT/ZCT shard_map outputs `(k, ns_l, col, μ, ns_r)` at
`P(None, None, None, 'x', 'y', None)` (per the docstring) but the
final `transpose` lands at `P(None, 'x', 'y')` on rank-3. The
intermediate rank-7 buffers are described as "shard_map internal"
in the docstring but their actual sharding (per axis) controls
the per-rank slot size. **Need HLO confirmation** that the rank-7
slot is `c128[nk, ns, mu_loc, r_chunk_loc, ns]` and not
`c128[nk, ns, mu, r_chunk_loc, ns]` (where the μ axis stayed
replicated). The current formulas assume `/P` sharding on
`mu·r_chunk`, but actually one axis is `/p_x` and the other is
`/p_y`, so total `mu·r_chunk/P` is correct only if `p_x · p_y =
P`. That's a tautology, so the formula is right — but the per-
axis sharding matters if one mesh axis is degenerate (e.g.
`p_x = 1, p_y = P`, a 1-D mesh). The shard_map probably refuses
to run on a 1-D mesh entirely (cuSolverMp falls back). **Need to
confirm:** does the model also work for 1-D meshes, and if not,
what's the failure mode?

### 7.12. `bare_coulomb_cutoff_ry` and `zeta_cutoff_ry` interplay

Both default to `ecutwfc` per gw_init. `bare_coulomb_cutoff =
4·ecutwfc` (LORRAX default per my memory) vs BGW default
`ecutwfc` — see `project_bare_coulomb_cutoff_default` memory.
The size of `ngkmax` (and thus `B_gflat_acc`) depends on
`zeta_cutoff`. **Does the planner ever see `zeta_cutoff_ry`
when sizing `B_gflat_acc`?** I traced the code: at `gw_init.py:596`
`_ngkmax_est = … or int(0.06 · meta.n_rtot)` — a static fallback.
The real cutoff is only known after `compute_per_q_bare_coulomb_components`
runs, which is inside `fit_zeta_to_h5`. So the planner uses a
stale estimate. At default `zeta_cutoff = 80 Ry` vs `n_rtot
density grid = 320 Ry` say, the 0.06 factor might be 2× off.
Concrete fix: hoist the sphere construction out of
`fit_zeta_to_h5` and pass `ngkmax_actual` into
`plan_gflat_chunks`. Low priority but tidy.

### 7.13. Validating my `r_chunk ≈ 53,800` answer

If a maintainer can confirm that the *actual* binding constraint at
CrI3 80 Ry is the unsharded ψ_G box (W_wfn ≈ 13.8 GB at k_chunk=6,
bpd=16), and that pair_density_slots is really 3, then I'd expect
r_chunk closer to 35–50k, not 12,500. **What am I missing?** This
is the gap I most want closed in the discussion phase.

---

Agent 1 done — see agent_1.md
