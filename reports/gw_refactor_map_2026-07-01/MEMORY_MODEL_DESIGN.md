# ISDF memory model — what it should be

_2026-07-03. Synthesized from `SHARDING_RULES.md` (the substrate) + a quantitative
gather of the real `isdf/core.py` allocations, the gflat 5-peak model, and the legacy
5-moment model. This is the **design**: what the one model should compute, why, and how
the two current planners collapse into it. Companion to `MAINTAINABILITY.md`._

Budget: **O(5–100 GB) per process.** Dims: `N_rmu` (μ) 400–1e5, the shard currency;
`nb` ≈ N_rmu/10; `nk`,`nq` 1–few-thousand; `N_rtot` 1e4–1e7, never whole; `ngkmax`;
`ns∈{1,2}`; `P = p_x·p_y`. Complex128 = 16 B.

---

## 0. The model's job (it has two outputs, not one)

Given a system (μ, nb, nk, nq, ngkmax, ns, n_rtot) and a per-process budget, the model
must answer **two** questions. The current planners answer only the second:

1. **The rank floor — "can this run on P processes at all?"** Some arrays are *un-chunkable*
   (the ÷P μ² family: `L_q`, `gflat_acc`, `Z_q`, `zeta`). Their per-rank size shrinks only
   with more ranks, never with chunking. So there is a hard minimum:
   ```
   P_min = ceil( nq · N_rmu² · 16 / (util · budget) )     # the L_q / μ² floor
   ```
   At μ=5e4, nq=100, 28 GB budget, util 0.8 → **P_min ≈ 150**. Neither model computes this;
   both silently assume P is given and then fail to find a `chunk_r` that fits. The model
   should say *"this system needs ≥150 ranks"* first, then size chunks under it.

2. **The chunk sizes — "given P ≥ P_min, how big can the chunks be?"** Pick the largest
   `chunk_r` (and `band_chunk`, `q_chunk`, `k_chunk`, `gflat_chunk_size`) such that the
   **binding peak ≤ util·budget**. Only one large array is tunable (the pair-density carry,
   via `chunk_r`); everything else is either fixed (the ÷P floor) or a small knob.

The whole model is: **compute the floor, then dial `chunk_r` down from n_rtot until the
binding peak fits.**

---

## 1. The consequential array inventory (the only things the model should carry)

Each array's per-rank bytes = (logical size)·16 / (its trichotomy divisor). **Model these;
drop everything else** (§4).

| Array | Per-rank bytes | Divisor (trichotomy) | Tunable? | Lifetime |
|---|---|---|---|---|
| `L_q` (Cholesky factor) | `nq·N_rmu²·16 / P` | **P** (μ²) | no → **floor** | persistent |
| `gflat_acc` | `nq·(N_rmu/P)·ngkmax·16` | **P** (μ¹-allP) | no → floor | persistent |
| ψ fit copies (×4) | `nk·N_rmu·nb·ns·16 / √P` | **√P** (μ×nb) | no | persistent |
| **pair-density carry** (×slots, ×ns²) | `slots·nk·ns²·N_rmu·cr·16 / P` | **P** (μ_x, r_y) | **yes (cr)** | Peak C transient |
| `Z_q` / `zeta` (r-chunk) | `nq·N_rmu·cr·16 / P` | **P** | yes (cr) | Peak C/solve |
| fit FFT box | `nk·bpd·ns·n_rtot·16` (+ cuFFT scratch) | band; FFT-axes replicated | via band_chunk | per-iter (aliased) |
| accumulate FFT box | `gflat_chunk_size·n_rtot·16` (+ scratch) | replicated | cs≤100 | Peak D transient |
| gathered `psi_Y_bc` | `nk·nb·ns·cr·16` | **1** (band gathered) | via band_chunk | one aliased slot |

`slots` = **3 on GPU, 4 on CPU** (HLO-calibrated, not derivable from shapes). ns²=4 for
bispinor — this is the whole reason to run util≈0.8, not 0.97.

---

## 1a. The clean framing: PERSISTENT set + one binding STAGE transient

The whole model is two things summed. This is what the code should look like — nothing more.

**PERSISTENT set** — allocated once, resident across the entire r-chunk loop (the *floor*):
```
persistent(P) = L_q            nq·N_rmu²·16 / P            (÷P, μ²)
              + gflat_acc      nq·N_rmu·ngkmax·16 / P      (÷P)
              + 4·ψ_copies     4·nk·N_rmu·nb·ns·16 / √P    (÷√P — the corrected centroid term)
```
This is un-chunkable → it *is* the rank floor (§0).

**STAGE transients** — each stage adds ONE transient on top of the persistent set; they don't
co-exist, so the HWM is a `max`, not a sum:

| Stage | Transient (per-rank, on top of persistent) | Tunable by |
|---|---|---|
| A  centroid load | fit FFT box `nk·bpd·ns·n_rtot·16` (+ cuFFT scratch, XLA-queried) | band_chunk |
| B  CCT + Cholesky | `C_q` + full-(μ,μ) pair density `nk·ns²·N_rmu²·16 / P` | — |
| **C  fit_one_rchunk** | **pair-density carry `slots·nk·ns²·N_rmu·cr·16 / P` + `Z_q` `nq·N_rmu·cr·16 / P`** | **chunk_r** ← the binder |
| D  accumulate | accumulate FFT box `cs·n_rtot·16` (+ scratch), cs≤100 | gflat_chunk_size |
| E  V_q per tile | resharded ζ slabs (÷√P) + `V_acc` (÷P) | — (post-fit) |

```
HWM(cr, bc, P) = persistent(P) + max( A(bc), B, C(cr), D(cs), E )
```
Pick `chunk_r`, `band_chunk` so `HWM ≤ util·budget`; the rank floor requires `persistent(P) ≤
util·budget` on its own. Stage C is the binder for molecules/small crystals; for large-nq the
floor binds and C is dialed under it; for coarse grids A/D (the FFT box) can bind. That is the
*entire* model — a persistent floor plus a max over five stage-transients.

## 1b. Bispinor: model the charge channel only (the transverse is strictly smaller)

Bispinor fits 4 ζ channels: charge (μ_C centroids) + 3 transverse/current (μ_T). **μ_T ≤ μ_C
always.** Every fit-loop operation on a transverse channel is *exactly parallel* to the charge
operation with μ_T instead of μ_C → strictly ≤ the charge peak → **never the binder.** So the
model uses `N_rmu = μ_C` and does **not** separately size the transverse channels — it only
carries the spinor factor `ns² = 4` in the pair density (Stage C). This collapses the 4-channel
bookkeeping to a single-μ model. (The one place this doesn't hold is Stage E V_q's TT-off-diagonal
tile — two *distinct* transverse ζ, not parallel-to-charge — so Peak E keeps its own μ_T term;
the fit loop, A–D, is charge-only.)

---

## 2. The two-phase model

**Phase 1 — the floor (un-chunkable ÷P family).** Sum the persistent per-rank bytes that
don't depend on `chunk_r`: `L_q` + `gflat_acc` + 4·ψ-copies. Require that under budget:
```
floor_bytes(P) = nq·N_rmu²·16/P  +  nq·(N_rmu/P)·ngkmax·16  +  4·nk·N_rmu·nb·ns·16/√P
P_min = smallest P (a valid mesh factorization) with floor_bytes(P) ≤ util·budget
```
If the requested P < P_min → the model reports *infeasible, need P_min ranks*, not an OOM.

**Phase 2 — dial `chunk_r`.** With the floor reserved, the remaining headroom goes to the
tunable transient (pair-density carry + Z_q/zeta), plus the FFT box:
```
headroom = util·budget − floor_bytes(P) − fft_box_bytes(band_chunk)   # fft_box from XLA query
cr_max   = headroom / ( slots·nk·ns²·N_rmu·16/P  +  nq·N_rmu·16/P )    # the per-cr slope
chunk_r  = clamp( cr_max, low = min(N_rmu, n_rtot), high = n_rtot ), rounded down to a multiple of P
```
The lower bound `cr ≥ N_rmu` is a **performance** floor (amortize per-chunk pipeline
overhead), not memory. `band_chunk`, `q_chunk`, `k_chunk`, `gflat_chunk_size` are sized the
same way against their own (small) terms — see §5 for q/k.

This is one closed-form pass + one XLA FFT query. Microseconds.

---

## 3. Binding regimes — the model must know which array binds, not assume Peak C

Three index-signatures, three regimes (the model should report which one it's in):

| Regime | Binder | Why | `chunk_r` role |
|---|---|---|---|
| **Molecule** (nq=nk=1) | pair-density carry (Peak C) | μ² family collapses (nq=1) | binds directly |
| **Small crystal** | pair-density carry | μ² family comfortable | binds directly |
| **Large-nq crystal** | μ² family (÷P floor) | `nq·μ²` un-chunkable | sets P; cr dialed under it |
| **500-atom / huge-μ, big P** | ψ copies (÷√P) rival μ² | ÷√P shrinks slower than ÷P; crossover `√P > 10·nq/nk` | cr still the knob |
| **Coarse-grid / big box** (n_rtot~1e7) | FFT box + cuFFT scratch | independent of μ | band_chunk/cs bind |

The current gflat model always treats Peak C as the binder; that's right for molecules/small
crystals but wrong at the large-nq (floor-bound) and coarse-grid (FFT-bound) ends. The
redesigned model takes `max` over the regimes and *names* the binder.

---

## 4. Consequential vs negligible (what to drop)

**Drop (≪1% of budget):** `cct_trace_per_q` (nq·16), band norms (nb·16), γ̃ perm/phase,
accumulate phase tables (nq·(nx+ny+nz)·16), `kvecs_frac`, replicated `sphere_idx`/`g_index`
int32 (~MB). These clutter the current models without moving the answer.

**One exception to check:** `box_index_dev` in the ψ(G) loader is `nk·n_rtot·4` replicated
— GBs at large nk·n_rtot. It's outside the r-chunk loop but shares the budget; include it if
the loader is co-resident.

---

## 5. Collapse the two planners into one

The two current models **overlap on the band/r arrays and diverge everywhere it matters.**
The reconciliation:

**Redundant (both model the same array):** pair density (legacy `α_pair` ≡ gflat Peak-C
slots, bit-identical per-slot), `Z_q`/zeta (legacy `α_zcol` ≡ gflat zeta_out + Peak-D
zeta_chunk), centroids, `L_q`, the in-loop FFT box. Legacy runs its full 5-moment inversion
for `band_chunk`/`chunk_r` **whose results gflat then discards** (`gw_init.py:521`) — pure
dead compute.

**Exclusive:** legacy owns `q_chunk`/`q_gather`/`k_chunk` (gflat computes no q/k); gflat owns
`gflat_chunk_size`, Peak E (V_q, all 7 tiles + unfold + Lorentz-mix), and the sphere-idx
term.

### The design: gflat's peak skeleton is the model; fold q/k into it; delete legacy's band/r
- **Keep** gflat's A–E peak structure (it's the newer, HLO-calibrated one) + Peak E (V_q).
- **Add** Phase-1 rank-floor reporting (§0/§2) — new, neither model does it.
- **Move** the q/k sizing into the same model, evaluated at **gflat's `chunk_r`** (fixes the
  bug below). q/k need only `zeta_out`/`gather` per-q bytes, which Peak C already has.
- **Delete** legacy `compute_optimal_chunks`' 5-moment band/r model (redundant + dead).

### Six divergences to resolve while merging (each a real decision, three are bugs)
1. **`target_utilization`: 0.97 (legacy) vs 0.80 (gflat).** Pick one; make it ns²-aware
   (bispinor's 4× pair density is exactly why 0.80 exists) rather than hard-coded.
2. **`n_bc` unroll — STALE (bug, latent).** Legacy multiplies the FFT slope by
   `n_bc=⌈nb/band_chunk⌉` for a Python-unrolled loop that is now `lax.scan(unroll=1)`. Drop
   it (gflat already did).
3. **Pair-density slots: legacy 5 (2+3), gflat 3/4 (HLO-calibrated).** Use 3/4. Legacy's 5 is
   1.67× over-conservative → different `chunk_r`.
4. **Centroid term — WRONG IN BOTH (bug).** Legacy = 2 copies at `/p_x,/p_y` (right divisor,
   wrong count); gflat = 4 copies at `/p_xy` (right count, wrong divisor). The physical truth
   is **4 copies, single-axis: 2·(/p_x) + 2·(/p_y) ≈ 4/√P**. At square P=16 the two models
   disagree 4× on the same persistent array. Fix to 4-copies-single-axis.
5. **FFT box: legacy queries XLA exactly, gflat guesses (`factor_A=4.0`/`factor_D=2.0`).**
   Keep the XLA query (`query_fft_peak_bytes`) — it caught a 19 GiB Si-10³ under-prediction
   the static factor missed, because cuFFT plan scratch is not shape·16.
6. **Solve peak: legacy charges 3 replicated `(μ,μ)` L-slabs (`c_solve`, ÷p_x²), gflat none.**
   That replicated-L is the **1-D `sharded_cholesky` path** (40 GB at μ=5e4 → why 2-D meshes
   force cuSolverMp). Production (cuSolverMp 2-D) has no replicated L, so the solve peak is
   just `zeta_out`+`L_q` (both ÷P). Model the *production* path; flag the 1-D path as
   never-in-production.

### The bug the merge fixes (concrete failure mode)
Legacy sizes `q_chunk`/`q_gather`/`k_chunk` from per-q costs evaluated at **legacy's**
`chunk_r`, but the run uses **gflat's** `chunk_r`. If `cr_gflat > cr_legacy`, the true per-q
solve/write buffers exceed what q/k were sized for → **over-packing → OOM in the solve/write
stage.** Folding q/k into the one model (sized at the actual `chunk_r`) removes this.

---

## 6. What the model must QUERY, never compute (the honesty boundary)

A pure shape·16 model is wrong on three things; the redesign must delegate them:
- **cuFFT plan scratch** → call `query_fft_peak_bytes` on the unsharded 6-D FFT shape (the
  `nk` factor matters — dropping it under-predicted by 19 GiB).
- **Backend slot count + scan aliasing + donation** → the "3 vs 4 concurrent slots", the
  collapse of `n_bc` per-iter copies to one aliased slot, and the in-place reuse of
  `gflat_acc`/`Z_q` are XLA BufferAssignment facts, not shape algebra. Encode them as the
  calibrated `slots` constant + the aliasing assumption (one slot per per-iter transient).
- **Involuntary Full Rematerialization** → a reshard the model scores ÷P silently becomes ÷1
  (P× blow-up) if mis-staged. The model *assumes correct single-axis staging* (per
  SHARDING_RULES §4); it does not try to predict a mis-staged peak. If staging is wrong, the
  fix is the code, not the model.

---

## 7. Net shape of the redesign

```
plan_isdf_chunks(system, budget, mesh):
    P_min = rank_floor(system, budget)                 # NEW — the un-chunkable ÷P family
    if P < P_min: report infeasible(P_min); return
    peaks = { A: centroid_load, B: cct_chol, C: fit_one_rchunk,   # gflat skeleton, corrected
              D: accumulate, E: v_q_tile }              #   (centroid 4×-single-axis, slots 3/4,
    fft_box = query_fft_peak_bytes(...)                 #   XLA-queried FFT, ns²-aware util)
    chunk_r = dial_cr(peaks, floor, headroom)           # Phase 2
    q_chunk, q_gather, k_chunk = size_qk(chunk_r, ...)  # MOVED IN, at the real chunk_r
    band_chunk, gflat_chunk_size = ...                  # unchanged
    binder = argmax_peak(...)                           # report the binding regime
    return plan(chunk_r, band_chunk, q/k, cs, P_min, binder, hwm)
```
Deletes: `compute_optimal_chunks`' 5-moment band/r inversion (~150 L, dead). Fixes: the
centroid 4× bug, the stale `n_bc`, the slot count, the q/k-chunk_r inconsistency. Adds: the
rank floor + the named binding regime. One model, one target-utilization, one `chunk_r`.

**Bottom line:** the model isn't "predict the HWM"; it's *"report the minimum ranks, then
give the largest chunks that fit, and name what's binding."* Everything else — the per-array
sizes — is the trichotomy applied to the consequential inventory, with cuFFT scratch and the
backend slot count queried, not guessed.

---

## AS-BUILT (2026-07-03, commit 4c833e4) — verified by the lead

Implemented + HLO-validated on a live 16-GPU session. **Simplicity (the primary goal): met.**
- `gflat_memory_model.py` 922 → **374 L**; `gw_init.py` 1169 → **741 L**; planner total −976 L.
- `compute_optimal_chunks` (384 L legacy 5-moment band/r model) **deleted** — one planner now.
- Form is exactly §1a: `_persistent_bytes(P)` + max of 5 stage transients + two-phase picker.
  The centroid bug is fixed (`psi_copies = 2·psi_one/p_x + 2·psi_one/p_y`, ÷√P). q/k folded in
  at the real chunk_r; dead `k_chunk`/`q_gather` removed; ns²-aware util (0.90/0.85/0.78).

**Faithfulness (BFC `peak_bytes_in_use`):** predicted HWM tracks real to **≤0.1% wherever the
algorithmic ÷P memory binds** — 4-GPU charge 8.50/8.49 & 13.94/13.93; bispinor 15.30/15.31 &
(28 GB) 21.84/21.85 at both 4 and 16 GPU. The ns=4 util 0.78 fixes a real bispinor OOM (Stage-C
pair density was a 23 GB single arena the allocator couldn't place; 0.78 → 21.2 GB fits).

**The one honest gap (reported, not fudged):** a **~8 GB P-independent runtime floor** on 16 GPU /
4 nodes (NCCL cross-node buffers + cuFFT plan scratch + CUDA context). `real ≈ max(algorithmic,
~8 GB)`. The model predicts the *algorithmic* part faithfully and under-predicts only when
algorithmic < ~8 GB (16-GPU charge: 3.50 vs 8.00) — the low-occupancy, huge-headroom regime that
never OOMs. Modeling it needs topology-aware NCCL accounting (design §6 puts NCCL out of scope for
shape algebra). **Practical caveat:** on ≥16 GPUs the effective usable budget is ~8 GB less than
`memory_per_device_gb`; only matters at absurdly tight budgets on many GPUs (never in practice on
40/80 GB A100s). A `max(hwm, floor_estimate)` guard could be added if that regime ever bites.

**Gates:** 5 e2e green. gnppm reference re-frozen — the new (valid) chunk_r shifts GN-PPM's
chunk-order-sensitive Σc by 5.9e-5 while **sigX and VH are bit-identical** (verified: r-chunk
accumulation roundoff, not a regression). Planner unit tests rewritten to the new contract, 21 passed.
