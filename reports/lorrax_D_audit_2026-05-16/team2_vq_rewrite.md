# Team-2 audit: V_q algorithm rewrite on `agent/env-var-cleanup`

Branch tip: `488e870`. Checkout: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.
Scope: algorithm correctness of the new G-flat V_q orchestrator
(`gw/v_q_g_flat.py`) and its bispinor wrapper
(`gw/v_q_bispinor.py:compute_V_q_bispinor_g_flat_to_h5`), given an IBZ
ζ̃ already on disk. IBZ→full-BZ unfold and centroid-orbit closure are
Team 1's territory and are NOT audited here.

## 1. Executive verdict

The new G-flat V_q orchestrator is **algorithmically correct** for both
the scalar charge channel and the 7-tile bispinor decomposition, given
an IBZ ζ̃ in the writer's per-q WFN.h5-style sphere layout. All
synthetic per-q einsum-reference unit tests pass under CPU (3/3 for
scalar `test_compute_all_V_q_g_flat`; 12/12 for bispinor
`test_compute_V_q_bispinor_g_flat` + helpers; 9/9 for
`test_rchunk_gflat_pair`). The `lax.scan` over G is HLO-equivalent to
the deleted Python-unrolled loop. The "batched IBZ ζ̃ pre-read" is a
correctness-preserving I/O re-shape. **Async I/O is dead code** —
section 5 details that the orchestrator that the audit prompt asked
about no longer has any async path; the `queue` / `threading` imports
remain, the `async_prefetch` arg is accepted but ignored, and the
docstring documents this.

Specific defects found, by severity:

* **High** — `compute_all_V_q`'s G-flat dispatcher drops
  `mc_average_vcoul_body` on the floor when forwarding to
  `compute_all_V_q_g_flat` (`src/gw/compute_vcoul.py:854-870`). For 3D
  bulk runs without a BGW vcoul overlay, this causes a silent
  numeric divergence from BGW at small q≠0 G=0. 2D and 0D systems
  are unaffected. See §6 Issue 1.
* **Low** — `tests/test_v_q_per_q_g_chunked.py:182` has been broken
  since commit `8043ec6` ("Make mesh required everywhere"). The test
  that compares the new `compute_v_q_per_G` against the legacy
  `get_sqrt_v_and_phase` formula now raises `TypeError: missing
  positional arg mesh_xy` before reaching the numeric comparison.
  Two pytest parametrizations fail; the rest of the suite is clean.
* **Cosmetic** — `gw/v_q_g_flat.py:33-34` imports `queue` and
  `threading` that are no longer referenced (legacy async machinery
  removed when `_compute_V_q_g_flat_one_tile` switched to the
  single-batch IBZ pre-read in `0880066`).

The bispinor 7-tile partition is exhaustive and non-overlapping over
the 16 Lorentz blocks. Hermitian-redundant reconstruction is
exact-by-symmetry for the bare Coulomb v(K) (real K, real v).

## 2. Per-element V^{ij}(q+G) formula and code instantiation

For the scalar charge channel:

$$V_q[\mu,\nu] = \sum_G \overline{\tilde\zeta_{q,\mu}(G)}\; v(q+G)\;
                  \tilde\zeta_{q,\nu}(G)$$

For bispinor with Lorentz indices `(μ_L, ν_L) ∈ {0,1,2,3}²`:

$$V^{\mu_L,\nu_L}_q[\mu,\nu] = \sum_G \overline{\tilde\zeta^{\mu_L}_{q,\mu}(G)}\;
   v(q+G)\; t^{\mu_L,\nu_L}(q+G)\; \tilde\zeta^{\nu_L}_{q,\nu}(G)$$

with `t^{0,0}=1`, `t^{0,i}=t^{i,0}=0` (Coulomb-gauge zeros), and
`t^{i,j} = δ_ij − K̂_i K̂_j` for `i,j∈{1,2,3}` where
`K = q+G` in Cartesian Bohr⁻¹.

The kernel (`gw/v_q_g_flat.py:89-142`) instantiates this per q via a
`lax.scan` over G-chunks:

```python
def _g_chunk_body(V_carry, i):
    start = i * g_chunk
    L_chunk = dynamic_slice_in_dim(zeta_L, start, g_chunk, axis=-1)   # (n_μL/p_x, g_chunk)
    R_chunk = dynamic_slice_in_dim(zeta_R, start, g_chunk, axis=-1)   # (n_μR/p_y, g_chunk)
    v_chunk = dynamic_slice_in_dim(v_q, start, g_chunk, axis=0)       # (g_chunk,)
    L_w = jnp.conj(L_chunk) * v_chunk[None, :]
    return V_carry + L_w @ R_chunk.T, None
V_q, _ = lax.scan(_g_chunk_body, V_q, jnp.arange(n_chunks))
```

Per-element math at chunk `i`, intra-chunk index `g`:

* `L_chunk[μ, g] = ζ̃_L[μ, i·gc + g]`, `R_chunk[ν, g] = ζ̃_R[ν, i·gc + g]`.
* `L_w[μ, g] = conj(L_chunk[μ, g]) · v_chunk[g] = conj(ζ̃_L[μ, i·gc + g]) · v(q+G_{i·gc+g})`.
* `(L_w @ R_chunk.T)[μ, ν] = Σ_g L_w[μ, g] · R_chunk[ν, g]` (note `R_chunk.T[g, ν]`).
* Carry update: `V_q[μ, ν] := V_q[μ, ν] + Σ_g conj(ζ̃_L[μ, i·gc+g]) · v(q+G) · ζ̃_R[ν, i·gc+g]`.

Summed over `i ∈ [0, n_chunks)` this is exactly `Σ_G conj(L) · v · R`
provided `ngkmax = n_chunks · g_chunk` (enforced at line 352-355: the
function raises if `ngkmax % g_chunk != 0`). The chunking is a pure
partition of the G-axis with disjoint windows — no double-counting.

The legacy r-space kernel (`gw/v_q_tile.py:_zeta_disk_to_G` at line
664-694 in `_make_V_q_tile_kernel`) does the algebraically identical
computation but with two extra stages: an in-kernel 3D FFT
(`r_box → G_box`) and a per-q sphere gather. The new code skips both
because the writer already deposited ζ̃ on the per-q (q+G) sphere
(`accumulate_rchunk_to_gflat` does the FFT once at writer-side).
Equivalence is established by:

* The 5/5 numeric unit tests in `test_compute_all_V_q_g_flat.py` and
  `test_compute_V_q_bispinor_g_flat.py` that compare the new
  orchestrator's output to the explicit per-q
  `np.einsum('mG,G,nG->mn', conj(ζ), v, ζ)` reference on the logical
  sphere length `ngk[q]` (pad slots zeroed out by writer construction).
* Bit-identity on the MoS2 3×3 bispinor end-to-end run (see
  `reports/gflat_perf_before_after_mos2_2026-05-12/report.md` §
  Correctness — "eqp0.dat bit-identical between before/after").

## 3. Bispinor 7-tile partition

The bispinor V_q is a `(4, 4)` Lorentz-block tensor of `(n_μ, n_ν)`
matrices. The Coulomb kernel in Lorentz gauge couples channels via
the projector `t^{μ_L,ν_L}(K)` defined above. Out of the 16 blocks:

* **6 vanish identically** by Coulomb gauge (cross terms between
  μ_L=0 and μ_L∈{1,2,3}).
* **1 charge-charge (CC)** block at (0,0).
* **9 transverse-transverse (TT)** blocks at (i,j) with `i,j∈{1,2,3}` —
  3 diagonal, 6 off-diagonal split as 3 unique upper-triangular + 3
  Hermitian-redundant.

The orchestrator computes 7 unique tiles. Hermitian-redundant tiles
are reconstructed at read time via `V[j,i] = conj(swapaxes(V[i,j],
-1, -2))` (`BispinorVqReader.get_tile` at `v_q_bispinor.py:730-736`).
Zero tiles are returned as zeros without disk access
(`_zero_tile`, lines 708-715).

| Tile idx | (μ_L, ν_L) | Class | Coverage | Code location | Formula |
|---:|:---|:---|:---|:---|:---|
| 0 | (0, 0) | CC | scalar charge-charge | `v_q_bispinor.py:57` UNIQUE_TILES; `_make_per_q_v_builder_for_tile` line 208-225 short-circuits `is_CC=True` ⇒ returns `v(q+G)` unmodified | `Σ_G conj(ζ^0_μ) · v · ζ^0_ν` |
| 1 | (1, 1) | TT-diag i=1 | transverse along x-axis | `_make_per_q_v_builder_for_tile` line 226-234 with `i=j=0` (after `i,j = μ_L-1,ν_L-1`) | `Σ_G conj(ζ^1_μ) · v · (1 − K̂_x²) · ζ^1_ν` |
| 2 | (2, 2) | TT-diag i=2 | transverse along y-axis | same path, `i=j=1` | `Σ_G conj(ζ^2_μ) · v · (1 − K̂_y²) · ζ^2_ν` |
| 3 | (3, 3) | TT-diag i=3 | transverse along z-axis | same path, `i=j=2` | `Σ_G conj(ζ^3_μ) · v · (1 − K̂_z²) · ζ^3_ν` |
| 4 | (1, 2) | TT-offdiag xy | mixed x-y transverse | `is_CC=False`, `i=0, j=1`, `t = −K̂_x K̂_y` | `Σ_G conj(ζ^1_μ) · v · (−K̂_x K̂_y) · ζ^2_ν` |
| 5 | (1, 3) | TT-offdiag xz | mixed x-z transverse | `i=0, j=2`, `t = −K̂_x K̂_z` | `Σ_G conj(ζ^1_μ) · v · (−K̂_x K̂_z) · ζ^3_ν` |
| 6 | (2, 3) | TT-offdiag yz | mixed y-z transverse | `i=1, j=2`, `t = −K̂_y K̂_z` | `Σ_G conj(ζ^2_μ) · v · (−K̂_y K̂_z) · ζ^3_ν` |

Gauge-zero tiles (returned as zeros, not computed):
`{(0,1), (0,2), (0,3), (1,0), (2,0), (3,0)}`
(`v_q_bispinor.py:63-66`, `ZERO_TILES`).

Hermitian-redundant tiles (reconstructed by transpose-conj of
companion):
`{(2,1)→(1,2), (3,1)→(1,3), (3,2)→(2,3)}`
(`v_q_bispinor.py:70-74`, `HERMITIAN_PAIRS`).

**Completeness check.** 7 unique + 6 zero + 3 hermitian = 16 = 4×4. ✓
**Non-overlap.** `test_v_q_bispinor_helpers.py:54-67` asserts the three
sets are pairwise disjoint and their union covers all 16 blocks. ✓
**Hermitian reconstruction correctness.** For real Cartesian K and real
v(K), t^{i,j} is real and symmetric in (i,j), so

  `conj(V^{i,j}[q,μ,ν]) = Σ_G ζ^i[μ,G] · v · t^{j,i} · conj(ζ^j[ν,G])
                       = V^{j,i}[q,ν,μ]`

giving `V^{j,i} = conj(swapaxes(V^{i,j}, -1, -2))` ✓. This holds
provided `ζ^i` is the genuine fit for tile i (not a transpose of
something else); the bispinor ζ pipeline writes four separate files
(`zeta_q_mu1/2/3.h5`) via the dedicated channel-aware fit and the
reader loads them as L/R operands per tile.

**Subtle point.** The same ζ file is used on both L and R sides for
TT-diagonal tiles (`same_zeta=True` in `_compute_V_q_g_flat_one_tile`
at `v_q_bispinor.py:546`), so the diagonal tiles use one read.
Off-diagonal tiles read two distinct files (`same_zeta=False`, line 547),
so two reads. The kernel branch at `v_q_g_flat.py:96-99` handles both
modes — for `same_zeta` it aliases `zeta_R = zeta_L_3d` from a single
slab read, for distinct it re-shards a separate slab.

## 4. Scan / chunking correctness proofs

### 4.1 `lax.scan` over G-chunks (commit `0880066`)

The replaced Python loop was:
```python
for i in range(n_chunks):
    V_q = V_q + jnp.conj(L_chunk_i) * v_chunk_i[None, :] @ R_chunk_i.T
```
The new scan body returns `(V_carry + L_w @ R_chunk.T, None)`, with
`V_carry` threaded as the scan carry. After `lax.scan` returns,
`V_q := scan_carry_final`. The two are HLO-isomorphic as long as the
loop iterations are dependency-chained on the carry, which they are
(`V_q = V_q + ...` is a serial update). XLA folds a one-iteration
scan into the body; for `n_chunks > 1` the scan compiles once and
executes `n_chunks` times, vs the Python loop's `n_chunks×`
unrolled HLO.

Per-chunk update: `V[μ,ν] += Σ_{g=0}^{gc−1} conj(L[μ, i·gc+g]) · v[i·gc+g] · R[ν, i·gc+g]`.
Final accumulation over `i ∈ [0, n_chunks)`:
`V[μ,ν] = Σ_{i, g} ... = Σ_G ...` (G ranges over all `n_chunks · gc =
ngkmax` slots). Bit-identical to the unchunked
`'mG,G,nG->mn'` einsum — same FP order under XLA's typical sum
reduction (cumulative left-to-right within each chunk, then cumulative
left-to-right across chunks). FP roundoff differences vs the einsum
reference are at `1e-10` rtol in the unit tests (well within float64
noise on `O(ngkmax)` sums).

### 4.2 Single batched IBZ ζ̃ pre-read (commit `0880066`)

The pre-read constructs `zeta_L_all = read_all_ibz(n_q_ibz)` of shape
`(n_q_ibz, n_rmu_padded, ngkmax)` once before the q-loop. Inside the
loop, each iteration does:
```python
zeta_L_q = lax.dynamic_slice_in_dim(zeta_L_all, q, 1, axis=0)
```
producing a `(1, n_rmu_padded, ngkmax)` view at the q-th slab.
`zeta_L_all` is **not mutated** anywhere in the loop (Python `del
zeta_L_all` after the loop body is the only write). The kernel's
`donate_argnums=(0, 1)` only donates `V_acc` and `g0_acc`, not the
ζ̃ slabs — `zeta_L_q` is an aliased view, not a fresh buffer. So
the "constant across the scan" claim holds: `zeta_L_all` is
read-only across all q's.

The same is true for `same_zeta=False`: `zeta_R_all` is also read-only,
sliced per q with `dynamic_slice_in_dim`.

### 4.3 `accumulate_rchunk_to_gflat` chunking (commits
`e73fd10`, `7b6b2f1`, `4fa5598`, `20c5fac`, `3171308`)

`wfn_transforms.py:468-673`. This is the bridge from r-space ζ to the
on-disk G-flat layout, NOT the V_q forward direction — but the V_q-side
correctness depends on the writer producing the correct ζ̃[q, μ, G]
slabs.

The writer flattens the `(n_q, n_mu_local)` axes into a single
`N = n_q · n_mu_local` row axis (line 566). Each scan iteration
processes `cs` rows. For row `r` in iteration `i`:

* `q_row[r] = clip((i·cs + r) // n_mu_local, 0, n_q-1)`.
* `i0 = i·cs`; sub-slab `sub = rch_flat[i0:i0+cs]` of shape `(cs,
  r_len_i)`.
* Pad rows where `i·cs + r ≥ N` land on `q = n_q-1` with zero data
  (the post-pad `jnp.pad` at line 622 zero-extends both
  rch_flat and acc_flat), so their FFT contribution is exactly zero
  — no contamination of the legitimate q rows.

The "n_q axis vs μ axis vs flat axis" chunking decisions all preserve
the math because the flat (q · μ_local) axis is a direct concatenation
of per-(q, μ_local) rows — no spatial mixing across q within a row,
and per-row FFTs are independent. The `dynamic_slice` + `pad_N` clamp
in `[:N]` after the scan (line 666) is the canonical "drop pad rows
before reshape" pattern. The flat-axis shard_map chunker version
(`4fa5598`) is correct for the same reason — it just changes how the
scan dispatches Python iterations.

### 4.4 Per-q kernel donation (charge + bispinor)

`donate_argnums=(0, 1)` on `_make_per_q_kernel.fn` (`v_q_g_flat.py:89`)
donates `V_acc` and `g0_acc`. The outer Python `for q in
range(n_q_ibz)` calls `V_acc, g0_acc = kernel(V_acc, ...)`, so each
Python iteration receives the donated buffer back through the JAX
ABI. The `block_until_ready(V_acc)` at line 440 forces the previous
write to complete before the next iteration's `dynamic_update_slice`,
preventing any host-side aliasing race (though JAX's data dependence
already orders the ops; the block is defensive). Correct.

## 5. Async I/O race-condition analysis

The audit prompt asks about async slab I/O race conditions and
prefetch-vs-consume barriers. **There is no async I/O in the current
code path.** Evidence:

* `v_q_g_flat.py:430-444` is a synchronous Python `for q` loop on
  device-resident, fully-read ζ̃ slabs (`zeta_L_all`,
  `zeta_R_all`).
* The pre-read (`read_all_ibz`) is one batched PHDF5 call, with
  `jax.block_until_ready(zeta_L_all)` at line 422 to force completion
  before any kernel dispatches. No worker thread, no queue, no event.
* `queue` and `threading` imports at lines 33-34 are dead. The
  `async_prefetch` argument to `compute_all_V_q_g_flat`
  (`v_q_g_flat.py:488`) is **silently ignored** by
  `_compute_V_q_g_flat_one_tile`; the docstring at lines 494-498
  documents this as intentional ("currently has no effect — the sync
  per-q loop is already ~6× faster than the legacy μ × ν tile driver
  on MoS2 3×3").

The historical async-prefetch path that existed in the initial commit
`93a316c` (worker thread reading ζ̃_{q+1} while compute thread
contracted ζ̃_q) was removed by `6ebfc3e` and replaced by the
unconditional sync loop. Commit `0880066` then replaced the per-q
synchronous read with the single batched IBZ pre-read.

Race-condition surface: **none in the V_q orchestrator path**. The
remaining concurrency surface is internal to the PHDF5 FFI backend's
read implementation, which is Team 1's territory.

**Recommended cleanup** (low priority): delete `import queue`,
`import threading`, and the `async_prefetch` parameter from
`compute_all_V_q_g_flat` since they have no functional effect.

## 6. v(q+G) builder review

The new per-q builder is `compute_v_q_per_G` (`compute_vcoul.py:706`),
called by both the charge `compute_all_V_q_g_flat` and the bispinor
`_make_per_q_v_builder_for_tile`. Math:

```
v(q+G) = (1/V_cell) · regularized_kernel(|q+G|², sys_dim)
```

with

* `sys_dim=3`: `8π/|q+G|²` for `|q+G|² > 1e-12`, else 0.
* `sys_dim=2`: `(8π/|q+G|²) · (1 − exp(−z_c · k_xy) · cos(k_z · z_c))`
  where `z_c = π/b_zz` (Bohr).
* `sys_dim=0`: not wired (raises). The 0-D box truncation requires
  building v on the full FFT grid via `compute_sqrt_vcoul_0d`; the
  per-q per-G gather isn't implemented. Charge dispatcher
  `compute_all_V_q_g_flat` already raises `NotImplementedError` for
  `sys_dim=0` at `v_q_g_flat.py:502-505`.

Optional `vcoul_cutoff_ry` zeros `v` at G's with `|q+G|² > cutoff`.

### Issue 1 (High): `mc_average_vcoul_body` dropped on the G-flat path

`compute_v_q_per_G` does **not** implement the mini-BZ-averaged head
that the legacy `make_v_munu_chunked_kernel.get_sqrt_v_and_phase`
applies under `mc_average_vcoul_body=True` (the default). For 3D bulk
at small q≠0, the legacy code replaces the point value
`8π/|q|²` at G=0 with the Voronoi-cell average
`⟨8π/|q+δq|²⟩` over 50³ mini-BZ samples (`compute_vcoul.py:306-337`).
The new builder uses the point value.

The dispatcher (`compute_vcoul.py:854-870`) accepts
`mc_average_vcoul_body` but does **not** forward it to
`compute_all_V_q_g_flat`. The flag is silently dropped on the G-flat
path.

**Impact.** For 3D bulk with `cfg.head.mc_average_vcoul_body=True`
(default) and no `bgw_v_grid_fn` overlay, the G=0 entry of v(q+G=0)
at small q differs from BGW by a few percent. The G=G'=0 head at q=0
is separately injected via `apply_q0_head_rank1` and is unaffected.
For 2D systems the slab truncation factor `f2d → 0` as `k_xy → 0`,
regularizing v at G=0 already, so the issue is essentially moot for 2D
(MoS2 / CrI3 production runs). For 3D Si / Si-like bulk runs the
discrepancy is real.

**Severity.** High because it's a silent flag drop with no warning;
production runs against BGW will show small body-channel disagreements
that look like physics noise. Related to and exacerbating the documented
`bare_coulomb_cutoff` default mismatch (`project_bare_coulomb_cutoff_default.md`).

**Fix.** Plumb `mc_average_vcoul_body` and the corresponding miniBZ-
averaged table through to `compute_v_q_per_G`. Either:
* (preferred) compute the miniBZ-averaged G=0 row inside
  `compute_v_q_per_G` when `mc_average=True`, host-side, vectorized
  over q. The function is already host-side and not jitted — adding
  the 50³ sample loop is a ~10 ms one-shot cost per V_q run.
* (alternative) wire a `head_v_per_q_kgrid` precomputed table into the
  G-flat dispatcher, mirroring the legacy `_v_head_avg_j` array.

**Test.** Re-enable `test_v_q_per_G_matches_legacy_kernel` (Issue 2)
and extend it with a `mc_average=True` parametrization that compares
the head row.

### Issue 2 (Low): broken legacy-equivalence test

`tests/test_v_q_per_q_g_chunked.py::test_v_q_per_G_matches_legacy_kernel[2]`
and `[3]` fail with `TypeError: make_v_munu_chunked_kernel() missing
1 required positional argument: 'mesh_xy'` at
`tests/test_v_q_per_q_g_chunked.py:182,194`. The signature change
that broke this is commit `8043ec6` ("Make mesh required everywhere").
The test is the only one that compares the new per-q v(q+G) builder
output to the legacy formula bit-for-bit. **It has been silently
broken since the mesh-required commit landed; the equivalence claim
is currently untested.**

**Fix.** Two-liner: pass a `single_device_mesh` fixture to both
`make_v_munu_chunked_kernel` calls in the test. Re-runs in ~1 s on CPU.

### Issue 3 (Low): bispinor v_per_G builder `eps_K2` divergence at q=0

`_make_per_q_v_builder_for_tile` at `v_q_bispinor.py:217-234` uses
`K2_safe = max(K2, eps_K2=1e-30)` to keep `K̂_i K̂_j = K_i K_j /
K2_safe` finite at K=0. At K=0 the bare `compute_v_q_per_G` already
returns `v=0` (denom_zero branch), so the product `v · t` is 0
regardless of the chosen K̂. **Net effect: K=0 contributes 0 to the
TT tile sums — correct by construction.** No bug, just worth
noting that the `eps_K2` guard is for AD/jit cleanliness, not numerics.

### Issue 4 (Cosmetic): q=0 head for TT tiles

Docstring at `v_q_bispinor.py:147-148`: "the K=0 limit is gauge-
singular; we guard the denominator with `eps_K2`". The CC tile gets
the explicit rank-1 head injection
(`head_correction.apply_q0_head_rank1`); the TT tiles do not. The
docstring at `v_q_bispinor.py:204-205` says "Head correction at q=Γ
flows through the CC tile's `g0_acc`; transverse tiles intentionally
omit it." This is consistent with the bare-Breit Phase-1 design
(`docs/BISPINOR_DHFB_DESIGN.md`); the transverse-projector kills the
G=0 head naturally for K aligned with the projection axis but leaves a
finite contribution for generic q→0 directions. **No action; verified
intentional.**

## 7. Specific issues (severity, location, fix)

| # | Severity | File:line | Problem | Fix |
|---:|:---|:---|:---|:---|
| 1 | High | `src/gw/compute_vcoul.py:854-870` | G-flat dispatcher drops `mc_average_vcoul_body` arg. 3D bulk runs against BGW silently use point v(q+G=0) at small q≠0 instead of miniBZ-averaged value. | Wire flag through to `compute_v_q_per_G`; implement miniBZ-average host loop in the per-q builder. |
| 2 | Low | `tests/test_v_q_per_q_g_chunked.py:182,194` | Legacy-equivalence test broken since `8043ec6`. Equivalence to legacy v(q+G) formula is currently UNTESTED. | Add `mesh_xy=single_device_mesh` to both `make_v_munu_chunked_kernel` calls. |
| 3 | Cosmetic | `src/gw/v_q_g_flat.py:33-34, 488-498` | Dead imports (`queue`, `threading`) and dead parameter `async_prefetch`. | Delete the imports and the parameter. |
| 4 | Cosmetic | `src/gw/v_q_g_flat.py:225-228` | TYPE_CHECKING import block declares `ZetaLoader` / `ZetaReader` but they're never used as annotations anywhere in this module — the actual dispatch on `has_load` / `has_read_slab` happens via duck-typing. | Delete the import block. |
| 5 | Cosmetic | `src/gw/v_q_g_flat.py:323-330, 332-339` | The two `gvec_components.shape[0] != n_q_ibz` checks raise `ValueError` with "was the file written with the same write_ibz_only setting?" but this is exactly the use_ibz cascade Team 1 owns. If they ever get out of sync the error is clear, just noting the helpful diagnostic depends on the file metadata Team 1 wrote. | None — defensive validation is good. |

## 8. Open questions

1. **3D run regression test.** Is there a Si or Si-like 3D run on a
   sym kgrid that exercises the G-flat path AND would surface the
   mc_average flag drop? Memory note `project_trs_blind_sym_bug` says
   "the 286 meV CrI3 gap is the TRS bug, NOT basis convergence", but
   that's 2D. I don't see a recent Si run that goes through the G-flat
   bispinor=False path with sym; if there isn't one, Issue 1 has not
   been exercised end-to-end since the rewrite. Worth a small Si
   smoke-test before declaring this stack complete.

2. **Hermitian on-disk redundancy.** The Hermitian-redundant tiles
   are reconstructed on read, saving 3× tiles of disk storage. But
   the bispinor V_q output buffer in memory (`compute_sigma_x_bispinor`)
   materializes all 9 transverse pairs. The audit didn't surface a
   consistency check that the reader's `conj(swapaxes)`
   reconstruction matches a "would have computed directly" tile —
   e.g., a unit test that computes V^{2,1} directly via the same
   kernel and compares to `reader.get_tile(2, 1)`. The
   Hermitian-symmetry argument in §3 holds analytically, but a numeric
   smoke test would be cheap insurance. Test
   `test_v_q_bispinor_orchestrator.py:185
   ::test_reader_hermitian_pair_matches_companion_transpose` covers
   the **algebraic** check on a returned-by-reader tile vs the same
   reader's companion — it doesn't compute V^{2,1} from scratch.

3. **Async I/O.** Should the dead `queue`/`threading` machinery be
   removed in this branch's cleanup or preserved as a future opt-in
   seam? The docstring at lines 14-20 still mentions "async I/O —
   kept from the legacy driver — is the only orchestration trick we
   keep" but that's misleading given the current single-batch read
   architecture. The doc and the dead imports should reconcile.

4. **`mc_average_vcoul_body=False` users.** If any production run
   has `cfg.head.mc_average_vcoul_body=False` (e.g., to match a
   no-average BGW reference), the G-flat path's silent drop is a
   no-op and there's no observable issue. Worth checking the standard
   cohsex.in templates to see if any explicitly disable the flag.
