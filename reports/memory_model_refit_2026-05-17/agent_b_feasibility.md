# Agent B — feasibility audit for symmetric ψ/ζ chunk knobs (2026-05-17)

Read-only audit of `accumulate_rchunk_to_gflat`, `gflat_to_rmu`,
`to_rchunk_inner`, `_make_fit_one_rchunk_kernel` / `fit_one_rchunk`,
`z_q_from_psi_sm`, and the planner. Files cited: `…/lorrax_B/src/common/wfn_transforms.py`,
`…/lorrax_B/src/common/isdf_fitting.py`, `…/lorrax_B/src/common/fft_helpers.py`,
`…/lorrax_B/src/common/psi_G_store.py`, `…/lorrax_B/src/gw/gflat_memory_model.py`.

---

## Q1 — Can `gflat_chunk_size = 1` work in `accumulate_rchunk_to_gflat`?

**Yes, structurally.** No divisibility constraint along the chunked axis.
Tracing `wfn_transforms.py:887-1113`:

- The flat axis `N = n_q · n_mu_local` is zero-padded to a multiple of `cs`
  (line 1011-1013): `n_chunks = (N + cs - 1) // cs; pad_N = n_chunks * cs - N`.
  So `cs = 1` is legal: `n_chunks = N`, `pad_N = 0`.
- Padding rows would be created when `cs > 1` and `N % cs != 0`; with `cs = 1`
  every row is its own chunk and the docstring explicitly says
  "the chunk size is a free integer — no divisibility constraint on either
  n_q or n_mu_local" (line 946-948).
- The body uses `lax.scan(body, init, jnp.arange(n_chunks))` (line 1103-1104).
  At cs=1 each iter does one row's pre-FFT phase multiply + a singleton
  `(1, n_rtot)` FFT box + sphere gather. Per-iter FFT box transient is
  `1 · n_rtot · 16 B ≈ 18 MB` at n_rtot=1.125M (plus cuFFT scratch).
- The sphere gather uses `take_along_axis(..., mode='promise_in_bounds')`
  (line 1097-1098); singleton row dim is fine.
- `donate_argnums=(1,)` on `acc_` lets XLA alias the accumulator in place
  (line 1109) — orthogonal to cs.

Identical analysis for `gflat_to_rmu` (line 611-851): same pad+scan+
free-integer cs contract (line 670-671), per-iter box transient is
`cs · ns · n_rtot · 16 B`.

**Verdict:** `gflat_chunk_size = 1` compiles and runs. The cost is
`n_chunks = N` scan iters — that's `n_q · n_mu_local`. For the CrI3 6×6 SOC run:
N = 36 · 1520/16 = 3420 iters per r-chunk. cuFFT plan cost per iter
(see Q5) is the practical issue, not correctness.

## Q2 — Can `band_chunk = world_size = 16` work in `fit_one_rchunk`?

**Yes, and the chunker already enforces this floor.** From
`gflat_memory_model.py:310-322`:

```
if p_xy > 1 and band_chunk_pre % p_xy != 0:
    band_chunk = ((band_chunk_pre + p_xy - 1) // p_xy) * p_xy
if band_chunk < p_xy:
    band_chunk = p_xy
```

Going below `world_size` would set `bpd_per_bc = (b_hi - b_lo) // p = 0`
in `psi_G_store.py:146`, then `_bpd_max = 0` (line 157), and the
downstream `lax.all_gather(axis_name=('x','y'), axis=1, tiled=True)` in
`z_q_from_psi_sm._local` would emit "all_gather_dim cannot be zero"
(documented at line 304-309 of the planner).

At `band_chunk = 16` (= p_xy on 4×4):
- `bpd_per_bc = 16 / 16 = 1` band per rank per bc.
- The io_callback returns `(nk, _bpd_max=1, ns, ngkmax)` per bc
  (`psi_G_store.py:340-341`).
- `to_rchunk_inner` is called on shape `(nk, 1, ns, ngkmax)` →
  produces `(nk, 1, ns, r_chunk)` per rank (wfn_transforms.py:413-428).
  The leading-3-axis contract is preserved; no op breaks at nb_local=1.
- `all_gather(axis=1, tiled=True)` over a size-1 band axis is valid;
  output band axis = P · 1 = 16 (`isdf_fitting.py:655-657`).
- The einsums `'kmna,knbr->karmb'` contract over the band axis; nb=1 is
  algebraically fine (line 696-701).

**Verdict:** `band_chunk = 16` is the actual floor, already auto-bumped by the
planner. No further blockers in the kernel code.

## Q3 — Is the bc-loop Python-unrolled or `lax.scan`?

**`lax.scan` with `unroll=1`.** The planner docstring (`gflat_memory_model.py:19-21`,
"Python-unrolled bc-loop ... n_bc copies of the FFT-box workspace stack")
is **stale** — Round 6 (cited inline at `isdf_fitting.py:704-710`) replaced
the Python-unroll with a scan-inside-shard_map. The body now closes:

```python
(P_l, P_r), _ = jax.lax.scan(
    body, (P_l_init, P_r_init),
    jnp.arange(n_bc, dtype=jnp.int32))
```

The comment at `isdf_fitting.py:704-707` is explicit:

> DO NOT unroll — the FFT-box and psi_G_bc aliasing depends on per-iter
> sequential lifetime. unroll=1 keeps the WhileOp atomic and lets XLA's
> scan-internal allocator reuse the slot.

**Impact on memory model:** `n_bc` no longer multiplies the FFT-box term in
Peak C. The actual model code at `_peak_C_fit_one_rchunk` lines 191-199 also
no longer multiplies by `n_bc` (the slot count is `pair_density_slots`
verified from XLA's BufferAssignment, with `psi_bc_Y` / FFT box reusing
slots). So the planner *implementation* is already scan-aware; only the
module-header docstring lags.

The Peak A pre-loop centroid load (legacy `to_rmu` path) used to be
the Python-unrolled site. The new code at `wfn_transforms.gflat_to_rmu`
also scans (line 842-843).

So both ψ-side `band_chunk` and ζ-side `gflat_chunk_size` are scan iter
counts now; smaller chunks ⇒ more iters at constant per-iter memory ⇒
no change in compile-time trace size.

## Q4 — Structural ψ-side vs ζ-side differences

Symmetric in spirit, asymmetric in detail. Below is the diff:

| Aspect | ψ-side (band_chunk, `z_q_from_psi_sm` scan + `gflat_to_rmu`) | ζ-side (gflat_chunk_size, `accumulate_rchunk_to_gflat`) |
|---|---|---|
| Scanned flat axis | `(k, n_local)` rows (centroid load) / `bc_idx` (in z_q) | `(q, μ_local)` rows |
| Per-iter FFT box shape | `(cs, ns, nx, ny, nz)` c128 — spinor in batch | `(cs, nx, ny, nz)` c128 — spin-traced ζ |
| Per-iter box bytes | `cs · ns · n_rtot · 16` | `cs · n_rtot · 16` (no `ns`) |
| FFT direction | IFFT (G → r) | FFT (r → G) |
| Phase site | post-gather, per-centroid (saves `(n_rtot − n_rmu)/n_rtot` of phase work) — `gflat_to_rmu:826-838`. In z_q the phase is on the full r-chunk (`to_rchunk_inner` applies Bloch phase on the slab). | pre-FFT, on slab only (`r_len` cells, not full box) — `accumulate_rchunk_to_gflat:1084-1092` |
| Gather kernel | `_box_kernel` scatter from G-sphere into FFT box, then sample centroids | `dynamic_update_slice` slab → box, `take_along_axis` from G-flat | 
| Sharding on chunked axis | band-flat over `('x','y')` — `bpd_per_bc = band_chunk // p_xy` divisibility (Q2) | μ-flat over `('x','y')` — `n_rmu_padded % p_xy == 0` divisibility, but cs itself is a free integer (Q1) |
| Bloch-phase kvec/qvec | `kvecs_frac[k]` per row | `qvec_frac[q]` per row |
| k-axis unfolding | irrep→full via `q_irr_full_idx` slicing (`_make_fit_one_rchunk_kernel:1564-1567`); ψ enters the kernel at full k | n/a on ζ-side; ζ is built directly at full q |
| Carry buffer | Rank-5 `P_l_acc / P_r_acc (nk, ns, r_loc, mu_loc, ns)` (`isdf_fitting.py:625-628`) — donates substantial HBM | `acc_flat (N+pad, ngkmax)` — donated via `donate_argnums=(1,)`, persistent gflat_acc |
| Effective scan length | `n_bc = ceil(nb_total / band_chunk)` | `n_chunks = ceil(N / cs)` |
| r-chunk axis interaction | ψ-side scan body owns the **full** `r_chunk` per rank (then sliced post-gather); rank-5 carry grows with r_chunk | ζ-side scan body is **inside** the r-chunk loop; reads one `r_chunk` slab as input, writes G-flat acc once per r-chunk |
| Compile-time r-divisibility | `r_chunk_size % p_y == 0` required (`isdf_fitting.py:451-454`) | `r_len = r_chunk` is just the input slab width; no further constraint inside |

Two additional architectural asymmetries worth flagging:

- **ψ-side scan body lives inside the production fused jit** (`_make_fit_one_rchunk_kernel`,
  `isdf_fitting.py:1573-1593`). Its memory budget composes with everything else
  in the same trace — centroids, L_q, P_pair slots. A bigger band_chunk
  doesn't pay for itself if the lifetime overlaps another already-binding peak.
- **ζ-side scan body is a standalone jit** (`wfn_transforms.py:1109`,
  `jax.jit(_kernel, donate_argnums=(1,))`). It runs after `fit_one_rchunk`
  returns and P_l/P_r are freed (`gflat_memory_model.py:209-227`). Its
  Peak D budget is essentially independent of Peak C and can be tuned
  in isolation.

This is the source of the bottleneck-coupling that motivates the refit:
ζ-side picks gflat_chunk_size against a Peak-D-only budget that doesn't
see Peak C's headroom, even though the user's HBM is one shared pool.

## Q5 — Realistic minimum chunk size + cuFFT considerations

The hard-floor per-rank FFT-box transient at n_rtot=1.125M, c128:

- ψ-side at band_chunk=16 (bpd_per_bc=1, ns=2):
  `1 · 2 · 1.125M · 16 B = 36 MB` raw + ~3-4× cuFFT scratch ⇒ ~110-150 MB
  per rank for one (k,nb) batched IFFT.
- ζ-side at cs=1 (no spin axis):
  `1 · 1.125M · 16 B = 18 MB` raw + cuFFT scratch ⇒ ~55-75 MB per rank.

`fft_helpers.py:14-23` is explicit about cuFFT batch-size sensitivity:

> Nominal `N_copies × data_size` fudge factors under-predict badly for
> mixed-radix boxes (24 = 2³·3, 10 = 2·5) at small batch sizes — cuFFT's
> planner picks different algorithms there with non-linear workspace growth.

Lorrax handles this by AOT-compiling each FFT shape and reading
`compiled.memory_analysis()` (`fft_helpers.py:82-84`, `query_fft_peak_bytes`).
That means the planner can **measure**, not guess, the workspace at any
candidate (cs, ns, n_rtot) — the realistic floor is whatever the cuFFT
plan actually picks at that batch, not a static `fft_box_factor`.

**Per-rank min footprint of one batched 3D FFT** (CrI3 fft_grid ≈ 60×60×200,
mixed radix 2·3·5):
- ψ-side single iter: input + output + cuFFT scratch ≈
  `2 × 36 MB + scratch`. At cs=ns=1 cuFFT may pick a non-strided plan that
  uses 2-4× scratch — say 100-200 MB. At cs=16 (e.g. band_chunk=256 on
  16 ranks), scratch amortises down to ~1.5×; per-iter cost ≈ 800 MB.
- ζ-side single iter: ~50-100 MB at cs=1; ~250 MB at cs=4-8.

Below `cs ≈ 4`, the cuFFT plan picks a small-batch algorithm with
relatively worse FLOPS/byte (no docstring quantifies this for our boxes,
but the warning at fft_helpers.py:14-23 reflects observed nonlinearity).
Above `cs ≈ 32-64` the scratch term flattens. A reasonable **practical
floor** is `cs = 4-8` for ζ-side and `bpd_per_bc = 1` (i.e. `band_chunk = 16`)
for ψ-side — the latter is the structural floor already.

The user's proposed floor — "4× single FFTbox + slab being processed,
per rank" — agrees with this analysis: 4× covers the cuFFT in/out + scratch
plus the in-flight slab. The planner could express this as
`floor_cs = ceil(min_box_bytes / (n_rtot · 16 · fft_box_factor))`.

## Joint refit feasibility — verdict

**Implementable, no structural blockers.** Concretely:

1. Setting `gflat_chunk_size = 1` or any small int is correctness-safe;
   the kernel pads and scans (Q1). Below ~4 you pay non-linearly worse
   cuFFT FLOPS/byte (Q5).
2. Setting `band_chunk = world_size` is the floor already auto-enforced;
   going lower is a hard correctness break in `all_gather` (Q2). No
   further floor to discover.
3. The bc-loop and gflat-chunk scan are both `lax.scan` — XLA's scan-
   internal allocator aliases per-iter FFT boxes across iters, so smaller
   chunks ⇒ more iters at constant per-iter memory and identical HLO
   trace size (Q3). The planner's "n_bc copies in trace" header comment
   is stale; the actual model code is correctly scan-aware.
4. The structural asymmetries (Q4) are quantitative, not qualitative:
   ζ-side has no ns axis, no kvec→qvec, no k-unfolding, is a standalone
   jit. None of these prevents treating both knobs in one budget.
5. The two knobs share the same HBM pool but the current planner
   (`gflat_memory_model.py:287-390`) picks them sequentially with no
   feedback. A joint refit should:
   - pick `r_chunk` first against Peak C with `band_chunk` at floor and
     `gflat_chunk_size` at a measured-min floor;
   - then grow each chunk up symmetrically against remaining headroom,
     using `query_fft_peak_bytes(input_shape=(cs, [ns,] *fft_grid), …)`
     for honest workspace numbers;
   - return per-peak component breakdown (already supported by the
     `breakdown` dict at line 425-426) so the user sees which term ate
     the budget.

No code change is needed in `accumulate_rchunk_to_gflat`, `gflat_to_rmu`,
`fit_one_rchunk`, or `z_q_from_psi_sm` to support a joint planner. The
floors are already in the kernels; the planner is the only locus that
needs to change.
