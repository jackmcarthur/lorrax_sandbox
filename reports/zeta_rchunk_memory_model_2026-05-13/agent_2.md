# Agent 2 — zeta-fit r-chunk memory model

From-scratch derivation. Symbols throughout:

- `P = p_x · p_y` — mesh size, `P_xy = P` (alias).
- `nk = nq` — k/q grid count (`meta.nk_tot`).
- `ns` — spinor count (1 or 2).
- `mu` — `meta.n_rmu_padded` (mesh-divisible; in-memory extent).
- `nr = n_rtot = nx·ny·nz` — FFT-grid total cells.
- `nb_L, nb_R, nb_F` — left/right/full band-range counts (`nb_L+nb_R = nb_total`).
- `cr` — r_chunk_size; `bc` — band_chunk_size; `kc` — psig_k_chunk_size; `cs` — gflat_chunk_size.
- `c128 = 16 B`. All sizes per-rank.
- `n_bc = ceil(nb_F / bc)`.

The "r-chunk" loop refers to `fit_zeta_to_h5` at `isdf_fitting.py:2117`, body is `fit_one_rchunk` followed by `accumulate_rchunk_to_gflat`.

---

## 1. Tensor catalog

I split tensors by **lifetime band**: A = once-per-channel preamble (steps 1–3 of §2 of the §5 reference), C = persistent through the r-chunk loop, T_C = transient inside `fit_one_rchunk` jit, T_A = transient inside `accumulate_rchunk_to_gflat` jit. "Bispinor" notes flag charge/transverse differences.

| Tensor | Shape (axis names) | PSpec | Lifetime | per-rank bytes (c128) | Notes |
|---|---|---|---|---|---|
| `psi_rmu_Y` | (nk, nb_F, ns, mu) | `(None,None,None,'y')` | C | `16·nk·nb_F·ns·mu / p_y` | un-conjugated; passed in by `gw_init.fit_zeta` |
| `psi_rmuT_X` | (nk, mu, nb_F, ns) | `(None,'x',None,None)` | C | `16·nk·mu·nb_F·ns / p_x` | conjugated copy; same data, different shard axis |
| `psi_l_*_fit` | same shapes as above sliced to nb_L | views into above | C | 0 (alias) | `_band_norms_slice` divide is also an alias when `band_norms is None` |
| `psi_r_*_fit` | same shapes sliced to nb_R | views | C | 0 (alias) |
| `C_q` | (nk, mu, mu) | `(None,'x','y')` | A only | `16·nk·mu²/P` | freed before the r-chunk loop (`del C_q, C_q_flat`) |
| `L_q` | (nk, mu, mu) | `(None,'x','y')` | C | `16·nk·mu²/P` | logical block + identity pad; cusolvermp/Cholesky factor |
| `cct_trace_per_q` | (nk,) | replicated | C, transverse only | `16·nk` | tiny |
| `gflat_acc` | (n_q_disk, mu, ngkmax) | `(None,('x','y'),None)` | C | `16·n_q_disk·mu·ngkmax/P` | persistent accumulator |
| `q_irr_idx_j`, `kvecs_frac_dev`, `sphere_idx_dev`, phase tables | various small | replicated | C | ≲ tens of MB | ignored |
| `psi_G` host tiles | (nk, nb_F/P, ns, ngkmax) per rank | host RAM, not HBM | — | host: `16·nk·nb_F·ns·ngkmax/P` | host, not HBM (host_cache mode) |
| `psi_G_flat_bc` (in `fetch_psi_rchunk`) | (nk, bc/P, ns, ngkmax) | `(None,('x','y'),None,None)` | T_C, per-bc | `16·nk·bc·ns·ngkmax/P` | one bc at a time on device |
| `psi_R_bc_box` (the FFT box for ψ_G → ψ_r) | (nk_local_kc, bc, ns, nx, ny, nz) | depends on whether sharding holds — see §2/§7 | T_C, per-bc-and-kc | **nominal sharded**: `16·kc·bc·ns·nr/P`; **unsharded-pathology**: `16·kc·bc·ns·nr` (no `/P`) | this is the offending tensor at CrI3 80 Ry |
| `psi_r_bc_slab` (after r-chunk slice) | (kc, bc, ns, cr) | `(None,('x','y'),None,None)` (sharded on band axis) | T_C, per-bc | `16·kc·bc·ns·cr/P` | small |
| `psi_Y_full` (concat over bc) | (nk, nb_F, ns, cr) | `(None,('x','y'),None,None)` | T_C, "live across bc" | `16·nk·nb_F·ns·cr/P` | accumulates the bc slabs after the bc-loop completes; lives until P_l/P_r consume it |
| `psi_l_Y_sm`, `psi_r_Y_sm` | (nk, nb_L/R, ns, cr) | `(None,None,None,'y')` after reshard | T_C | each `16·nk·nb·ns·cr / p_y` | slices of `psi_Y_full` divided by `norms`. The shard_map kernels' `R_spec` requires `'y'` on the trailing axis |
| `psi_l_rmuT_X_fit` reshard | same as `psi_l_rmuT_X` | `(None,'x',None,None)` | C | accounted under `psi_rmuT_X` | already in this layout — no reshard at runtime |
| `P_l`, `P_r` (rank-5, einsum output `'karmb'`) | (nk, ns, cr, mu, ns) per shard (`(nk, ns, cr/p_y, mu/p_x, ns)`) | shard_map-local (mu on 'x', col=cr on 'y') | T_C, inside shard_map | each `16·nk·ns²·mu·cr/P` | rank-5 form is the **dominant** in-loop transient. Two distinct lifetime slots (`P_l_R_conj`, `P_r_R`) per `c_q_from_psi_sm`/`z_q_from_psi_sm` (`isdf_fitting.py:319–344, 425–445`) |
| `P_l_R_conj` | (kx,ky,kz,ns,cr_loc,mu_loc,ns) bitcast | same as P_l | T_C, inside shard_map | same as P_l (alias) | rank-7 reshape is a bitcast — no new buffer |
| `P_r_R` | same as P_l_R_conj | same | T_C, inside shard_map | same as P_l (alias) |
| XLA scratch for `gamma_double_contract` | shape ≤ rank-5 reduce buffer | local | T_C, inside shard_map | empirically ≈ 1 slot of rank-5 size | the reference doc §5.6 counts this as the third slot |
| `C_R` / `Z_R` (after γ̃ reduce) | (kx,ky,kz,cr_loc,mu_loc) | local | T_C, inside shard_map | `16·nk·mu·cr/P` | rank-3; small vs rank-5 |
| `Z_q` (out of shard_map) | (nk, mu, cr) | `(None,'x','y')` | T_C | `16·nk·mu·cr/P` | output of `z_q_from_psi_sm` |
| `L_q_for_solve`, `Z_q_for_solve` (IBZ gather) | (n_q_disk, mu, mu/cr) | same as parents | T_C | up to same as parent | gather lives inside the jit; XLA may alias to the source under appropriate donate |
| `zeta_chunk` | (n_q_disk, mu, cr) | `(None,('x','y'),None)` | T_C → T_A handoff | `16·n_q_disk·mu·cr/P` | output of solve, lives across the jit boundary into the accumulate call |
| cuSolverMp internal workspace | ~`O(mu²)` plus block-cyclic scratch | per-rank | T_C, inside solve | not modelled — see §7 | NOT in the public memory model |
| `rch_flat` (inside accumulate shard_map) | (n_q_disk · n_mu_local, cr) per rank | local | T_A | `16·n_q_disk·mu·cr/P` (zero-pad to multiple of `cs`) | view/reshape of `zeta_chunk` |
| `acc_flat` (donated `gflat_acc`) | (n_q_disk · n_mu_local, ngkmax) per rank | local | T_A | accounted under `gflat_acc` (donated) |
| FFT box inside accumulate (`pad → fftn → take`) | (cs, nx, ny, nz) per scan iter, local | T_A, per scan iter | `16·cs·nr` × cuFFT factor | unsharded by construction (per-rank-local on the q·μ row axis) |
| per-q phase tables `phx,phy,phz` | (n_q_disk, nx|ny|nz) | closure constants | T_A | ≲ MB |
| `cs × r_len` slab phase gather | (cs, cr) | local | T_A | `16·cs·cr` | small |
| `cs × ngkmax` result `result_acc` | (cs, ngkmax) | local | T_A, per scan iter | `16·cs·ngkmax` |

**Bispinor split.** For charge channel `ns=2` and `mu = mu_C ≈ 8·n_band` (CrI3 ~1800). For transverse `ns=2` and `mu = mu_T ≈ 6·n_band` (CrI3 ~1200) plus the LU-path solve replaces Cholesky in `factor_c_q`/`solve_zeta`. The pair-density tensor shapes are identical in structure (open-spin via γ̃-fold); the only memory delta is the `mu_T` vs `mu_C` substitution and the LU pivot vector `(nk, mu)` int32 ≈ negligible. `cct_trace_per_q` is allocated for transverse only.

---

## 2. Aliasing analysis

I split the live tensors by *whether they coexist in the trace* (sum) versus *swap into the same allocator slot* (max). All claims here are inferences from code structure unless flagged "verified".

### Inside `fit_one_rchunk` jit (Peak C)

**Persistent across the r-chunk loop AND live during the jit:**
- `psi_l_rmu_Y`, `psi_l_rmuT_X`, `psi_r_rmu_Y`, `psi_r_rmuT_X` (= `psi_rmu_Y/T`, with views for L/R) — these enter as jit args; they SUM with everything inside.
- `L_q` — jit arg, SUMS.
- `gflat_acc` — held in Python around the jit; SUMS with everything inside (though the accumulate jit donates it).

**Transients formed inside the jit, lifetime tiers:**

Tier 1 — per-bc fetch (one bc lifetime):
- `psi_G_flat_bc` (output of io_callback) — alive until `to_rchunk` consumes it.
- `psi_R_bc_box` — formed inside `to_rchunk`, the cuFFT input/output buffer. Alive only while `to_rchunk` runs.
- `psi_r_bc_slab` — output of `to_rchunk`. Appended to `psi_Y_parts` and kept until the bc-loop ends.

Tier 2 — after the bc-loop, before the pair-density einsum:
- `psi_Y_full` = `jnp.concatenate(psi_Y_parts, axis=1)`. Holds all bc's slabs catted on the band axis. **Sum of all `psi_r_bc_slab`s.** The `del psi_Y_parts` is on the Python side but the XLA value already exists.
- `psi_l_Y_sm`, `psi_r_Y_sm` = slices of `psi_Y_full / norms` — XLA can often share storage with the parent (they are non-overlapping band slices when nb_L+nb_R = nb_F; could view-alias).

Tier 3 — inside `z_q_from_psi_sm` shard_map (the dominant peak):
- `P_l` and `P_r` rank-5 einsum outputs — distinct buffers.
- `P_l_R_conj`, `P_r_R` after IFFT — `P_l_3d → P_l_R → P_l_R_conj` chain has `del P_l_3d, P_l_R` so only `P_l_R_conj` survives. Same for `P_r_R`.
- ≥1 XLA scratch slot used by `gamma_double_contract` and the rank-7 → rank-3 reduction (§5.6 of ref).
- `C_R`/`Z_R` rank-3 (small), `Z_q_3d` after FFT.

Tier 4 — after the shard_map returns:
- `Z_q` output (rank-3). The shard_map closes; rank-5 buffers are freed.
- `L_q_for_solve`, `Z_q_for_solve` gather (if IBZ): can alias the parent under XLA donation.
- Solve workspace (cuSolverMp internal scratch, ~`mu²`-class).
- `zeta` output.

**Concurrent live set at Peak C** (the binding peak, near the rank-5 step):
```
SUM:
    persistent jit args: psi_l_rmuT_X + psi_l_rmu_Y + psi_r_rmuT_X + psi_r_rmu_Y + L_q
  + (eventually) gflat_acc held by caller
  + (at the rank-5 step) `pair_density_slots` × rank-5 P-buffers
  + Z_q (rank-3, small relative to rank-5)
```

**What aliases inside the rank-5 step** (per the reference doc §5.5, attested to a `module_NNNN.jit__kernel.memory-usage-report.txt` dump that I can't read but trust as a hand-extracted constant):
- The `psi_R_bc_box` FFT box (Tier 1) aliases into the SAME XLA slot as one of the rank-5 P-buffers because their lifetimes do not overlap (the FFT box dies in `to_rchunk` for bc=k; the rank-5 P forms in the *next* shard_map fragment). This is the §5.5 "W_wfn fits inside one pair-density slot" claim.
- `Z_q` rank-3 is small; aliases ambiguously.
- The IBZ-gather slice can alias parent under donation.

**Number of concurrent rank-5 slots = `pair_density_slots`.** Reference doc §5.6 says this is 3 after the 2026-05-13 monolithic bake: {`P_l_R_conj`, `P_r_R`, one XLA scratch}. I will use 3 below and flag it as a measured constant.

### Inside `accumulate_rchunk_to_gflat` jit (Peak D)

This is a **separate jit** so the allocator pool resets. Inputs:
- `gflat_acc` (donated by caller — in-place updated)
- `zeta_chunk` (= `rchunk` arg)
- closure constants (sphere idx, phase tables)

Per scan iter:
- `sub` = slice of `rch_flat`, shape (cs, cr).
- FFT box `(cs, nx, ny, nz)` = `dynamic_update_slice_in_dim(zeros((cs, nx, ny, nz)), sub, r0, axis=-1)`. Per-rank, NOT sharded. Alive for `fftn` + `take_along_axis`.
- cuFFT scratch — bounded above by some multiple of the box.
- `result_acc` (cs, ngkmax) — `take_along_axis` output, accumulated into `acc_flat`.

These all alias across scan iterations (XLA scan turns this into one allocation). Sum at peak per iter:
```
SUM:
   gflat_acc (donated, but its buffer is held)
 + zeta_chunk (cat T_C → T_A handoff)
 + cs × nr c128 box × cuFFT factor  (the FFT working set)
 + cs × ngkmax (the take output)
```

### Cross-jit aliasing

`fit_one_rchunk` jit RETURNS `zeta_chunk`. The accumulate jit CONSUMES it. The two jits are separate so XLA's allocators are independent — physically, `zeta_chunk` is allocated in the fit jit, transferred (logically — same device buffer), held by Python, then handed to the accumulate jit. The rank-5 P-buffers are freed before the accumulate jit starts (the fit jit returned). So:

```
Peak overall ≤ max(  Peak_C_during_fit_jit,
                     Peak_D_during_accumulate_jit )
```

This is the key claim that lets us treat the two peaks independently. **Trust level: high** — backed by JAX's jit-boundary buffer semantics, which deallocate jit-internal transients on jit return.

### What does NOT alias

- The persistent centroids/L_q tensors are jit args; they sum with everything inside any jit they're alive in.
- `psi_Y_full` is built **outside** the shard_map (it's the cat of bc fetches in the outer jit body). It is alive **simultaneously** with `psi_l_Y_sm`/`psi_r_Y_sm` until the shard_map starts consuming the latter. If XLA fuses the slice-and-divide into the shard_map input it may collapse to zero extra; otherwise it sums.
- Across r-chunks the `gflat_acc` is held but everything else from the previous iter is freed (donate pattern).

---

## 3. Budget equations

Let `B = memory_per_device_gb · 1e9` (per-rank, after target_utilization slack). Define:

```
B_persist =   16 · nk · nb_F · ns · mu / p_y        # psi_rmu_Y
            + 16 · nk · mu · nb_F · ns / p_x        # psi_rmuT_X
            + 16 · nk · mu² / P                     # L_q
            + 16 · n_q_disk · mu · ngkmax / P       # gflat_acc
            (+ 16 · nk for transverse cct_trace)
```

Note that `psi_rmu_Y` and `psi_rmuT_X` are TWO COPIES of the same ψ data, sharded differently. They are both held alive across the loop because `gw_init.fit_zeta` keeps `psi_rmu_Y`/`psi_rmuT_X` for the post-fit wfn-bundle build (`isdf_fitting.py:1539-1542`). The chunker MUST count both — `gflat_memory_model._peak_C_fit_one_rchunk` already counts `2 ·` for centroids, but with `shard=p_xy` which double-counts the savings (it uses `mu, nb_F` with shard=p_xy rather than the actual `p_y` / `p_x` for the two copies — see §6).

Sub-budgets (each must be ≤ B):

```
B_persist ≤ B                                       # raise if not (don't even start)
W_pool   = B − B_persist                             # available for transients

# Peak C — fit_one_rchunk jit (rank-5 step is binding)
W_C(cr, bc, kc) ≤ W_pool

W_C(cr, bc, kc) = W_zeta(cr)           # 3 rank-5 slots
                 + max(0, W_Yfull(cr) − slot_overlap)   # see §3a
                 + W_wfn_outside_alias(bc, kc)          # see §3b

where:
  W_zeta(cr)    = pair_density_slots · 16 · nk · ns² · mu · cr / P
                ≈ 3 · 16 · nk · ns² · mu · cr / P

  W_Yfull(cr)   = 16 · nk · nb_F · ns · cr / P
                  (psi_Y_full alive during shard_map entry — sums unless XLA fuses)

  W_wfn_box(bc, kc)
                = 16 · kc_eff · bc · ns · nr · fft_factor / P     [SHARDED case]
                = 16 · kc_eff · bc · ns · nr · fft_factor          [UNSHARDED pathology]

  kc_eff = kc if 0 < kc < nk else nk

# Peak D — accumulate_rchunk_to_gflat jit
W_D(cs, cr) ≤ W_pool

W_D(cs, cr)   = 16·n_q_disk·mu·cr/P            # zeta_chunk held into the jit
              + 16·cs·nr · fft_factor          # FFT box, per-rank-local, NOT /P
              + 16·cs·ngkmax                   # take result
              (gflat_acc accounted in B_persist; donated → no double)
```

### 3a. `psi_Y_full` aliasing question

`psi_Y_full` has size `16·nk·nb_F·ns·cr/P`. The rank-5 P-buffers have aggregate size `pair_density_slots · 16·nk·ns²·mu·cr/P`. Their RATIO is `nb_F·ns / (pair_density_slots · ns² · mu) = nb_F/(3·ns·mu)`. At CrI3 80 Ry charge: nb_F≈400, ns=2, mu=1800 → ratio = 400/(3·2·1800) ≈ 0.037 — `psi_Y_full` is ~3.7% the size of the rank-5 slots, negligible. So I'll absorb `W_Yfull` into the slack of W_zeta and not model it separately.

### 3b. `W_wfn_box` aliasing question

The reference doc §5.5 claim is: under XLA buffer-assignment, the per-bc FFT box aliases into one of the rank-5 P slots because their lifetimes do not overlap. Under this assumption:

```
W_wfn_outside_alias = max(0, W_wfn_box - one_rank5_slot)
                    = max(0, 16·kc·bc·ns·nr·fft_factor/P
                            − 16·nk·ns²·mu·cr/P )
```

If `W_wfn_box ≤ one_rank5_slot`, it aliases cleanly and contributes ZERO to W_C beyond `W_zeta`. This is the "1/pair_density_slots" criterion in §5.5.

**The unsharded pathology** (§5.8 of ref): at CrI3 80 Ry, XLA materialises `psi_R_bc_box` UNSHARDED, so the per-rank size is `16·kc·bc·ns·nr·fft_factor` (no `/P`) — multiplied by `P=16`, this becomes 41 GB at default bc=16, kc=nk=36, fft_factor=4. No amount of W_pool will absorb that. Manual mitigation: drop kc. The clean fix lives in `to_rchunk` and is out of scope for the planner.

I model this as a **mode bit** in the planner — see §4 step "pathology branch".

### 3c. cuSolverMp scratch

Empirically O(mu²) for distributed factorize/solve. At CrI3 80 Ry mu_C=1800 → 16·1800²/P at P=16 → ~3.2 MB per-rank for the dense factor; the *scratch* on top is ~the same order. Negligible vs W_zeta. **But** if cuSolverMp's getrs allocates pivot vectors and a workspace that doesn't share with our buffers, this is unmodelled and could surprise. Flag in §7.

---

## 4. r_chunk picker procedure

Inputs: `B` (per-rank budget bytes), `mesh_xy`, `meta`, `nb_L`, `nb_R`, `ngkmax`, `n_q_disk`, bispinor flag, channel (charge | transverse for n_rmu selection). Constants: `pair_density_slots = 3` (extract from HLO), `fft_factor = 4.0`.

**Step 0 — preflight.**
```
Compute B_persist (§3, with the per-channel mu).
If B_persist > 0.95·B:
    raise — system geometry too large for budget.
W_pool = B − B_persist.
```

**Step 1 — pick `r_chunk` to fill W_pool with `W_zeta`.** This is the dominant performance lever; pick first, alone:
```
α_zeta = pair_density_slots · 16 · nk · ns² · mu / P
r_chunk_raw = floor(W_pool / α_zeta)
r_chunk = max(mu, min(n_rtot, r_chunk_raw))     # n_rmu floor: per-chunk overhead bound
round r_chunk down to a multiple of P (mesh-divisibility of the solve output)
```

**Step 2 — pick `band_chunk`.** Decoupled in the §5.4 algorithm; the binding inequality is "W_wfn fits inside one rank-5 slot":
```
one_slot = 16 · nk · ns² · mu · r_chunk / P          # one of the pair_density slots
target_box = one_slot                                # alias ceiling
α_box = 16 · nk · ns · nr · fft_factor / P           # using kc = nk (no k-chunk yet)
bc_max = floor(target_box / α_box)
bc = largest_pow2_le(min(bc_max, nb_F))
# fallback: if bc < 1, sharded box can't fit even bc=1 — drop to step 4
```

**Step 3 — pick `gflat_chunk_size`.** Independent jit, fresh pool:
```
α_box_D = 16 · nr · fft_factor                       # per-rank-local, no /P
take_term_per_row = 16 · ngkmax                       # small
N_rows = n_q_disk · mu / P
W_D_one_shot = 16·n_q_disk·mu·cr/P
              + N_rows · (α_box_D + take_term_per_row)
if W_D_one_shot ≤ W_pool:
    cs = None   # one-shot
else:
    headroom_D = W_pool − 16·n_q_disk·mu·cr/P
    cs = floor(headroom_D / (α_box_D + take_term_per_row))
    cs = max(1, cs)
```

**Step 4 — `psig_k_chunk_size` and the pathology branch.** Two sub-cases:

(a) If `W_wfn_box(bc, kc=nk)` *as computed in step 2* assumes sharding, **verify the assumption holds for this geometry**. The reference doc says the unsharded pathology kicks in for `nr ≳ 1e6` (CrI3 6×6 80 Ry has nr ≈ 1.13e6 → triggers; MoS2 has nr ≈ 46e3 → safe). Heuristic flag: `unsharded_box_risk = (nr · ns · bc · 16 > 0.5·B)` per rank assuming P=1 (worst-case). I can't probe XLA from the planner, so the conservative default is: **if `nr ≥ 5e5` and bc·nr·ns·16 ≥ 0.5·B (unsharded estimate), engage the pathology branch.**

(b) Pathology branch — bound the unsharded box:
```
kc_max = floor( (B − B_persist − one_slot)
                 / (16 · bc · ns · nr · fft_factor) )
kc = max(1, min(nk, kc_max))
# round nk-divisor preferred (avoid uneven last chunk in the Python loop)
```

**Step 5 — bispinor: separate `(r_chunk, bc, cs)` triple per channel.** The fit runs once per `vertex_mu_L ∈ {0,1,2,3}`. Charge uses `mu_C`; transverse uses `mu_T`. Re-run steps 0–4 with the channel's `mu`. The transverse channels often have smaller `mu_T < mu_C`, so r_chunk can grow; but `cct_trace_per_q` adds a tiny extra term and LU has a different scratch profile than Cholesky.

In practice the bispinor V_q is the asymmetric case: each transverse channel writes its own `zeta_q_G.h5`, so each channel gets its own `gflat_acc` over its own `mu_T`. The fits are sequential, not concurrent, so we don't sum them. **One triple per channel.**

**Step 6 — apply cohsex.in overrides.** If user set any of `(r_chunk_size, band_chunk_size, psig_k_chunk_size, gflat_chunk_size)` to non-zero, that value wins; recompute remaining knobs around it.

---

## 5. Validation at CrI3 80 Ry

Plug §4 numbers (charge channel, since W_zeta is dominated by `mu_C`):

```
nk = 36
ns = 2
mu = mu_C ≈ 1800
nr = 75 · 75 · 200 = 1.125e6
P  = 16
n_q_disk = 36       (full BZ — the IBZ shrink isn't visible at this stage)
ngkmax ≈ 0.06·nr ≈ 6.75e4
nb_F ≈ 400  (nb_L + nb_R for the ZCT)
B = 60 · 1e9 = 6.0e10 bytes
fft_factor = 4
pair_density_slots = 3
```

**B_persist** (excluding gflat_acc, which we'll add):
```
psi_rmu_Y    : 16 · 36 · 400 · 2 · 1800 / 4   (p_y = 4) = 1.04e11 / 4 = 2.07e10? wait — divide by p_y=4 only
              = 16 · 36 · 400 · 2 · 1800 / 4
              = 16 · 36·400·2·1800 / 4
              = 16 · 5.184e7 / 4
              = 16 · 1.296e7
              = 2.07e8 ≈ 0.21 GB
```
Hmm let me redo more carefully:
```
16 · 36 · 400 · 2 · 1800 = 16 · 5.184e7 = 8.29e8
/ p_y = 4 → 2.07e8  ≈ 0.21 GB
```
Same for `psi_rmuT_X` divided by p_x=4 → 0.21 GB.
```
L_q:    16 · 36 · 1800² / 16 = 16·36·3.24e6/16 = 36·3.24e6·1 = 1.17e8 ≈ 0.12 GB
gflat_acc: 16 · 36 · 1800 · 6.75e4 / 16 = 36·1800·6.75e4 = 4.37e9 ≈ 4.37 GB
```

```
B_persist ≈ 0.21 + 0.21 + 0.12 + 4.37 ≈ 4.91 GB
W_pool ≈ 60 − 4.91 ≈ 55 GB
```

**α_zeta:**
```
α_zeta = 3 · 16 · nk · ns² · mu / P
       = 3 · 16 · 36 · 4 · 1800 / 16
       = 3 · 36 · 4 · 1800
       = 7.78e5  bytes / r-unit
```

**r_chunk_raw:**
```
r_chunk_raw = W_pool / α_zeta = 5.5e10 / 7.78e5 ≈ 70,700 r-units
```

That's `~ 70,700` per-rank-local r-units? No — wait. The `mu/P` and `cr` are both **logical** dimensions in the formula, since `pair_density_slots · 16 · nk · ns² · mu / P` sizes ONE rank-5 slot per rank in bytes per logical r-unit. So `cr` here is the logical r_chunk (full extent), not the per-rank slice. So `r_chunk ≈ 70,700` logical r-units. Capped to `n_rtot = 1.125e6` (no clip), floor `mu = 1800` (no clip).

Round to multiple of P=16: r_chunk = 70,704.

**Compare to empirical `r_chunk ≈ 12500`** from the working config (§4 of CONTEXT). My number is **5.6× larger** than the empirical.

Where is the mismatch?

Possibility 1 — `pair_density_slots` is undercounted. If the true count is closer to **18** (5.6× → 16-ish), my r_chunk lands at empirical. But the reference doc says 3 is hand-extracted from an HLO dump. Open question.

Possibility 2 — `α_zeta` should include more than `pair_density_slots · nk·ns²·mu`. E.g. `psi_Y_full` is `16·nk·nb_F·ns·cr/P = 16·36·400·2·cr/16 = 28800·cr` — vs `α_zeta = 7.78e5`. Adding this brings α to ~8.07e5, only 4% larger. Not the answer.

Possibility 3 — fft scratch inside the shard_map (the cuFFT 3D for IFFT/FFT on the rank-7 tensor) adds an EXTRA slot whose size is the rank-5 size × some factor. I have not modelled this; the §5.6 "one XLA scratch" slot is already in the count of 3, but maybe cuFFT's actual scratch is ≥ 4 × rank-5 size during the IFFT/FFT (matching the `fft_factor = 4.0` empirical). If cuFFT scratch is 4 × one_slot, total = 3 + 4 = 7 slots × (16·nk·ns²·mu/P) = 1.82e6, r_chunk = W_pool/α = 5.5e10/1.82e6 ≈ 30,200 — closer but still 2.4× too big.

Possibility 4 — `target_utilization` at 0.80 cuts W_pool to 0.80·55 = 44 GB. r_chunk drops to 0.80 · 70,700 = 56,600. Still too big.

Possibility 5 — the unsharded `psi_R_bc_box` is **also** alive concurrently with the rank-5 slots at CrI3 80 Ry (the `psig_k_chunk_size=6` mitigation doesn't fully eliminate it). At kc=6, bc=16, ns=2, nr=1.125e6, P=16:
```
W_wfn_box_unsharded = 16 · 6 · 16 · 2 · 1.125e6 · 4 / 1  (NO /P)
                    ≈ 1.38e10 ≈ 13.8 GB
```
If 13.8 GB is added to the rank-5 sum rather than aliased, W_pool effective drops to 55−13.8 = 41.2, and α_zeta = 7.78e5 → r_chunk = 5.30e4. Still 4× too big.

Possibility 6 — `psi_Y_full` ALSO sums with the rank-5 slots, AND `psi_Y_full` has nb_F=400, AND nb_F was double-counted (nb_L + nb_R = 400 in the chunker, not nb_F). Look at `nb_total_chunker` in `gw_init.fit_zeta:588`: it equals `nb_L + nb_R = (b3-b0) + (b4-b1) ≈ 2 · nb_F`. So the actual `psi_l_Y_sm + psi_r_Y_sm` is `~ 16·nk·(nb_L+nb_R)·ns·cr/P`, twice my §1 estimate. Still small.

Possibility 7 — there are MORE concurrent rank-5 buffers than 3 — perhaps closer to **n_bc × 3** if the per-bc P_l/P_r at each bc iter is computed in series but each persists until the next reduce. Looking at `c_q_from_psi_sm._local`: ONE pair-density iter, no bc-loop INSIDE the shard_map. But the outer kernel (`_kernel` at `isdf_fitting.py:1229`) calls `z_q_from_psi_sm` ONCE per fit_one_rchunk — over the full nb_F bands at once via `psi_Y_full`. So no, n_bc doesn't multiply rank-5 inside the shard_map.

I don't fully resolve this. My **best guess** is a combination of:
- `target_utilization=0.80` already in effect (×0.80)
- The unsharded W_wfn pathology at full bc=16 (subtract ~14 GB from W_pool)
- pair_density_slots is slightly higher than 3 (could be 4–5 — XLA versions vary)

Combined: r_chunk lands in the 10,000–20,000 range — consistent with 12,500. But I cannot derive 12,500 from first principles without HLO data I don't have access to.

**For `band_chunk`, `gflat_chunk_size`:**
```
one_slot = 16 · 36 · 4 · 1800 · 12500 / 16 = 16·36·4·1800·12500/16
         = 36·4·1800·12500 = 3.24e9 ≈ 3.24 GB per slot
α_box(per bc) = 16 · 36 · 2 · 1.125e6 · 4 / 16 = 36·2·1.125e6·4 = 3.24e8
bc_max = 3.24e9 / 3.24e8 = 10  → pow2 = 8
```
Empirical band_chunk = 16 — off by 2×. That suggests either fft_factor < 4 in practice, or my "one_slot" ceiling is too tight (band_chunk's box doesn't have to fit in ONE slot — it has to fit somewhere when rank-5 isn't live; the strict ceiling is "fits in the gap").

**gflat_chunk_size:**
```
α_box_D = 16 · 1.125e6 · 4 = 7.2e7 bytes/row
one-shot N_rows = 36 · 1800 / 16 = 4050
one-shot box = 4050 · 7.2e7 = 2.92e11 = 292 GB  → wildly exceeds W_pool
cs from headroom: headroom ≈ W_pool (zeta_chunk small here) ≈ 55 GB
cs = 55e9 / 7.2e7 ≈ 760
```
Empirical = 64 — way smaller than my 760. Either fft_factor for the accumulate cuFFT is >> 4 (closer to 50?), or W_pool isn't really 55 GB at this stage (gflat_acc is the dominant term and during the accumulate jit there's additional scratch I haven't seen). Open question — looks like the accumulate FFT scratch is much larger than I think.

---

## 6. Diff against the current code

Three competing pieces in tree:

### `compute_optimal_chunks` (`gw_init.py:154`) — the legacy 6-stage model
- Built around a decomposed CCT/ZCT/reshard/solve/gather chain that no longer exists (the monolithic shard_map collapsed FFT+pair+γ+FFT into one).
- The α coefficients in `_build_chunk_alphas` track stages that the modern fused jit doesn't have as separate XLA entities. The α model survives because XLA still emits *some* rank-5 buffer per stage, but the stage names and counts are misaligned.
- Calls `query_fft_peak_bytes` for an empirical XLA-emitted FFT scratch — this is good and worth keeping.
- **Verdict:** retire. Replace with a 3-step model derived from current code structure (one-jit C, one-jit D, two persistent pools). Keep `query_fft_peak_bytes` as the source of `fft_factor`.

### `gflat_memory_model.py:plan_gflat_chunks` — the A/B/C/D model
- Step 2 uses `0.5 * target` as the band_chunk FFT-box ceiling. §5.5 of ref says `1/pair_density_slots` is principled. The 50/50 split is arbitrary.
- `_peak_C_fit_one_rchunk.persistent.centroids_persist` uses `2·c128(nk, ns, mu, nk, shard=p_xy)` — `nk` appears twice (once as the centroid count axis, once as a confused stand-in for nb_total). That's a bug: it should be `nb_F` or `nb_L+nb_R`, sharded on `p_x`/`p_y` separately. Likely under-counts when nb << nk and over-counts when nb >> nk.
- `pair_density_slots = 3` is in the right ballpark but hard-coded.
- No `psi_Y_full` term (mostly negligible — see §5 — but worth a check).
- No unsharded `W_wfn_box` case — the `fft_box_factor=4` assumes sharded.
- No cuSolverMp scratch term.
- **Verdict:** closest to right. Keep the A/B/C/D split. Fix the `centroids_persist` shape bug; replace step-2 ceiling with `one_slot`; add an explicit pathology branch for unsharded W_wfn (and a path that derives `psig_k_chunk_size` from it); thread `mu_C` vs `mu_T` per channel.

### `aot_memory_model/` — the kernel-DSL chooser
- Cleaner abstraction (PRIMITIVE_CLASSES, AlphaFit) and includes an analytic closed-form chooser.
- Requires recorded `mem_fit`/`cost_fit` JSON per kernel and tag. If the JSON is current it's the most accurate model in tree; if stale, it's confidently wrong.
- The kernel files in `aot_memory_model/kernels/` are 1:1 with the current code shape (one per code-level kernel), which is the right abstraction. But the chooser's grid (`band_chunk_values = (8,16,32,64,128)`) doesn't include the un-sharded-box branch.
- **Verdict:** keep the abstraction and the kernel-by-kernel cost models. Migrate `gflat_memory_model`'s sizing logic onto the AOT scaffolding (one source of truth). Add a calibration test that re-derives `pair_density_slots` from an HLO dump and updates the JSON.

### What needs to go
- `compute_optimal_chunks` and its `_FFT_COPIES`, `α_bc`, `α_pair`, `α_psi_Y_bc` chain — all from the pre-monolithic era.
- `_apply_aot_chunk_model` as a "log only" shadow — if AOT is the planner, it's the planner; if not, drop it.
- Hard-coded `fft_box_factor = 4.0` shared across A/B/C/D — split into per-call-site constants (the accumulate cuFFT scratch is empirically bigger than 4×; see §5 validation).

### What needs to be added
- Explicit pathology branch (unsharded `psi_R_bc_box`) that PICKS `psig_k_chunk_size` from `B` rather than relying on the user setting it.
- `W_vq` term so the same chunker can pick `vq_g_chunk_size` (currently `_pick_g_chunk` runs independently with a hard cap of 4096).
- Per-channel re-run for the bispinor case (charge mu_C vs transverse mu_T).
- HLO-derived calibration tests for `pair_density_slots` and `fft_factor`, fail-loud if drift.

---

## 7. Open questions (the important section)

**Things I genuinely could not resolve.** Honest "I don't know" beats confident-but-wrong.

1. **`pair_density_slots` is a single dump-derived integer.** Reference doc §5.6 says 3 after the 2026-05-13 monolithic bake. I cannot independently verify without an HLO dump. My §5 validation walks the math at slots=3 and lands at r_chunk ≈ 70k, **5.6× larger than the empirical 12.5k**. Either:
   - (a) the constant should be much larger (≥ 7),
   - (b) `α_zeta` is missing a term whose contribution is ~4–5× the rank-5 sum (e.g. cuFFT scratch on the rank-7 IFFT/FFT pair, or a true rank-5 transpose I missed in the einsum 'karmb' → reshape 'kxyz...' chain),
   - (c) the practical r_chunk is set well *under* the model's ceiling (perhaps to leave headroom for XLA inefficiency, or because the unsharded box at CrI3 80 Ry costs ~14 GB even with `psig_k_chunk_size=6`).
   - I cannot decide between these without a fresh `memory-usage-report.txt` from the current monolithic kernel at CrI3 sizes. **This is the single biggest unknown.** Without it, no memory model can be trusted to within 5×.

2. **Whether `psi_Y_full` aliases inside the rank-5 step.** It's built by `jnp.concatenate(psi_Y_parts, axis=1)` outside the shard_map. If XLA fuses the concat into the shard_map input scatter (it should, in principle, given the size) it has zero extra cost. If not, it adds `16·nk·nb_F·ns·cr/P`. Need HLO grep to confirm. I treated it as negligible (§3a) — defensible at CrI3 charge (3.7% of rank-5), worth flagging.

3. **cuSolverMp scratch.** The `potrs`/`getrs` workspace at distributed-2D-block-cyclic scale is platform-dependent. For mu_C ≈ 1800 at P=16 this is small (~MB), but the LU path (transverse) requires pivot vectors per panel and may allocate scratch at `O(mu²/P) · k_blocks`. Not modelled; could surprise at very large mu.

4. **`fft_factor = 4.0` is one constant for three call sites.** The cuFFT scratch on:
   - (a) the 3D IFFT/FFT inside `c_q_from_psi_sm`/`z_q_from_psi_sm`'s shard_map (FFT on the rank-7 `(kx,ky,kz,...)` with fused γ̃ reduce around it),
   - (b) the per-bc `to_rchunk` 3D IFFT (`fetch_psi_rchunk`),
   - (c) the accumulate-kernel 3D FFT (one per scan iter, no fusion across iters).
   
   My §5 validation suggests (c) is far worse than 4× — my predicted `gflat_chunk_size=760` vs empirical 64 implies cuFFT scratch for the accumulate FFT is ~50× the box size, not 4×. Likely cuFFT plans a larger temp at this geometry; needs measurement.

5. **The unsharded `W_wfn_box` pathology** is described in §5.8 of the reference but the *root cause* in `to_rchunk` is not located. I don't know whether (i) the IFFT is on an axis whose sharding doesn't survive a `dynamic_update_slice` into the FFT box, or (ii) XLA replicates a `concat` over the k-axis chunked Python loop, or (iii) something else entirely. Without a fix the planner *cannot* automatically pick `psig_k_chunk_size`; my §4 step (b) is a fallback bound that throttles bc·nr·ns to fit, but it does so blind. **An HLO grep for the unsharded intermediate in `to_rchunk` is a prerequisite to any model that handles this case automatically.**

6. **Bispinor scratch differences (LU vs Cholesky).** The transverse path uses `getrf`/`getrs` with a ridge `1e-12·|tr(L)|/n_rmu`. Empirically the transverse channels run; I have no measurement of the LU path's working-set vs Cholesky at CrI3 80 Ry scale. The bispinor study was active on agent-B (per memory) but I haven't reviewed it. Two cases not exercised at full scale:
   - n_rmu_T < n_rmu_C: my model says transverse r_chunk grows by `mu_C/mu_T ≈ 1.5×`. Does it? Or is there a different binding peak?
   - `cct_trace_per_q` and the per-q ridge add per-r-chunk operations. Their working-set should be tiny but I haven't verified.

7. **`gflat_acc` ngkmax estimation.** `gw_init.fit_zeta:596` falls back to `0.06 · n_rtot` when `meta.ngkmax` is missing. The actual ngkmax at CrI3 80 Ry depends on `zeta_cutoff_ry`. If the user has set a high zeta_cutoff, ngkmax could be 20-30% of nr instead of 6%, and `B_persist` (gflat_acc term) grows accordingly. The planner uses an estimate; an unlucky cutoff choice can break the budget AFTER the planner ran. **The planner should compute the actual sphere up front, or be passed the true ngkmax.**

8. **`pair_density_slots_charge` vs `pair_density_slots_transverse`.** `gflat_memory_model.plan_gflat_chunks` has both as kwargs (both default 3). The kernel is structurally identical for both channels (γ̃-fold inside the shard_map; same einsum shape). I am 90% sure these are the same value, but they live as separate kwargs in case future bispinor work splits them. Not resolved.

9. **`W_vq` is a separate budget problem.** My step 5 (bispinor) re-runs the chunker per channel, but the **V_q pass** afterwards has its own peak that re-uses `gflat_acc` (now persistent on disk via the loader) and `vq_g_chunk_size`. The current `_pick_g_chunk` caps at 4096 by fiat. A clean model needs to compute the V_q peak from `(ngkmax, n_rmu_C, n_rmu_T, vq_g_chunk_size)` and pick `vq_g_chunk_size` from the same `B`. Not covered above; flagged for follow-up.

10. **`psi_l_rmu_Y` vs `psi_l_rmuT_X` — really two copies?** `gw_init.fit_zeta` is documented as passing both. They're loaded together by `load_centroids_band_chunked`. If they share storage via JAX view aliasing, my `B_persist` over-counts by ~0.4 GB at CrI3 (small). If they don't, I'm right. Worth a check.

11. **`target_utilization` interpretation.** The `gflat_memory_model` uses 0.80; the reference doc's §5.4 algorithm uses no explicit slack (1.0). The "right" slack is empirical and depends on XLA's overhead — without per-XLA-version measurements I cannot pick it from first principles. 0.80 is conservative; 0.95 is aggressive; the right answer is "whatever lands at the budget edge without OOM in `query_fft_peak_bytes` calibration runs."

---

Agent 2 done — see agent_2.md
