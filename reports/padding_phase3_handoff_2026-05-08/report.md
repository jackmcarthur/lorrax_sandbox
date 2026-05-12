# Phase 3 handoff — n_rmu padding for indivisible-by-4 (2026-05-08)

State on `agent/padding-helpers` (lorrax_A) at session end:

```
NEW    common/isdf_fitting: padded-input Cholesky path for prime/indivisible n_rmu
3a837a0  common/meta,load_wfns: Meta.n_rmu_padded + load_centroids pad-on-output
7474eb7  runtime/padding: PadAxis + array-level pad/unpad helpers + valid_shape bridge
4b5968b  gw/v_q_tile,v_q_bispinor: pad n_rmu to mesh-product for indivisible-by-16 μ
05d5670  gw: plumb wfns_transverse + bispinor V_q path; fix W_q rank for flat-q W0
0607073  gw: bispinor bare exchange Σ^B = …
```

`pytest` 133 passed, 7 skipped at HEAD.  Smoke beds verified on
`52728775` SLURM allocation (4 nodes × 4 GPUs):

| Test | n_rmu_T | Result |
|---|---|---|
| 656 (16-divisible) | regression | byte-identical V_qmunu on disk |
| 668 (4-divisible only) | Phase 2 working | V_qmunu at logical 668 on disk |
| 670 (2-divisible only) | not yet run | needs Phase 3b |
| 661 (prime) | not yet run | needs Phase 3b |

## Microbench result (load-bearing for Phase 3b design)

`scripts/profile_uneven_sharding.py` — 6-test sweep on a 4×4 mesh, μ ∈
{672, 668, 661}:

* **Top-level uneven sharding errors hard** (`device_put` ValueError),
  not silently inflate.  No silent OOM risk.
* **In-jit WSC to uneven product spec compiles via collective-permutes,
  zero new all-gathers.**  V_q-tile-shaped body at μ=668 (uneven WSC
  throughout) had 6 all-gathers identical in count and size to the
  μ=672 (clean) baseline — XLA SPMD pads the product axis internally
  to mesh-divisible (per-rank shard size unchanged).
* **The pad rows hold undefined values when XLA pads internally.**
  Explicit `jnp.pad` is required for *correctness* (consumers contracting
  along the padded axis expect zeros), not for memory.

## Phase 3b — what's left

`fit_zeta_chunked_to_h5` interior runs ψ through 4 separate JIT'd
helpers:

1. `compute_pair_density_spin_traced` — `out_shardings=P(None,'x','y')`
2. `compute_CCT_from_left_right` — same
3. `compute_L_q_from_CCT` — same (Cholesky)
4. `fit_one_rchunk` — same

At logical n_rmu=661 each of those committed `out_shardings` fails the
top-level divisibility check.  At padded n_rmu=672 the math is wrong —
the C_q matrix has zero rows/cols at the pad block, Cholesky/LU
encounters a singular block.  The user's spec rejects ridge
regularisation (`feedback_no_redundancy.md` + the original brief: "the
ISDF zeta fitting (triangular solve needs unpadded matrices)"), so the
fix must keep the math at logical n_rmu while satisfying boundary
divisibility at padded n_rmu.

### Cholesky helper (RESOLVED 2026-05-08)

`compute_L_q_from_CCT` now takes an optional ``n_rmu_logical`` kwarg.
When set and strictly less than the input dim, the function:

1. ``jax.lax.slice``-es C_q to its leading ``(nq, n_rmu_logical, n_rmu_logical)``
   block — the only physically meaningful part of a padded C_q (Phase 3a's
   ``load_centroids_band_chunked`` zeros the trailing μ pad).
2. Pins the slice to ``P(None, None, None)`` (replicated) via WSC.
3. Runs ``jnp.linalg.cholesky`` on the dense replicated matrix at
   logical extent.
4. Returns L at logical extent, replicated.

When ``n_rmu_logical`` is None or equals the input dim, the historic
2D-blocked / dense path runs unchanged.  See ``test_padding.py``:

* ``test_compute_L_q_indivisible_logical_n_rmu`` — n_log=7 padded to
  n_pad=8, P(None,'x','y') boundary on a 2×2 mesh, L L^H matches G_log
  to fp64 noise.
* ``test_compute_L_q_legacy_divisible_path_unchanged`` — public
  contract guard: ``n_rmu_logical=n`` ≡ ``n_rmu_logical=None``.
* ``test_compute_L_q_indivisible_indefinite_path`` — vertex_mu_L=1
  passthrough also slices to logical when padded.
* ``test_compute_L_q_logical_exceeds_input_raises`` — programmer
  guard.

Why not pivoted Cholesky (the original brief's leading suggestion)?
The existing ``src/centroid/pivoted_cholesky.py`` produces a
*rectangular* (M, k_keep) factor with pivot ordering — a different
contract from the square unpermuted L the back-solve consumes
(``solve_triangular(L)`` then ``solve_triangular(L^H)``).  Wiring it
in would require either tracking a permutation through the back-solve
or reconstructing an unpermuted L from pivoted columns; both
gold-plate compared to the slice-then-dense path.  At production
n_rmu ≲ 1000, replication is ≤ 16 MB per q (cheap), the per-q dense
Cholesky is negligible, and the algorithm sees zero zero-rows
(strictly the logical block, never the padded one).

Why not 2D-blocked at logical extent?  ``compute_block_size_for_2d_cholesky``
needs ``n_rmu`` divisible by ``lcm(p_x, p_y)`` (or a multiple).  For
n_rmu=661 (prime) on a 4×4 mesh that's ``lcm(4,4)=4``, and 661 has no
divisor ≥ 1 that lands in a valid (b, J) pair — there's no 2D-blocked
decomposition of a prime n_rmu that satisfies the per-axis tile-count
divisibility.  Padding inside the kernel and running the blocked
algorithm on the padded matrix would Cholesky-NaN on the singular
trailing block — back to the ridge-regularisation hole the user
explicitly rejected.

### Recommended approach for the rest of Phase 3b: outer-jit wrap

Replace the Python flow at `isdf_fitting.py:1268-1306` with a single
`@jax.jit` that takes ψ at PADDED n_rmu and returns L_q at PADDED
n_rmu:

```python
@jax.jit
def _build_L_q_padded(
    psi_l_rmuT_X, psi_l_rmu_Y, psi_r_rmuT_X, psi_r_rmu_Y,
):
    # Inputs at padded n_rmu (top-level boundary, divisible).
    sl = jax.lax.slice                              # static start/limit
    psi_l_T_log = sl(psi_l_rmuT_X, [0, 0, 0, 0],
                     [psi_l_rmuT_X.shape[0], n_rmu_logical,
                      psi_l_rmuT_X.shape[2], psi_l_rmuT_X.shape[3]])
    psi_l_Y_log = sl(psi_l_rmu_Y, [0, 0, 0, 0],
                     [psi_l_rmu_Y.shape[0], psi_l_rmu_Y.shape[1],
                      psi_l_rmu_Y.shape[2], n_rmu_logical])
    # … similarly for r
    # Inner jit calls at LOGICAL n_rmu — their out_shardings become
    # intermediates of the outer jit, where uneven WSC is accepted.
    P_l = compute_pair_density_spin_traced(psi_l_T_log, psi_l_Y_log, mesh_xy)
    P_r = compute_pair_density_spin_traced(psi_r_T_log, psi_r_Y_log, mesh_xy)
    C_q = compute_CCT_from_left_right(P_l, P_r, kgrid, mesh_xy)
    L_q_logical = compute_L_q_from_CCT(C_q.reshape(nq, n_rmu_logical, n_rmu_logical),
                                        mesh_xy, vertex_mu_L=vertex_mu_L)
    # Pad L_q back to padded n_rmu for the boundary
    pad_after = n_rmu_padded - n_rmu_logical
    return jnp.pad(L_q_logical, ((0, 0), (0, pad_after), (0, pad_after)))
```

Risks:

* JAX's nested-jit semantics: when an inner `@jax.jit` is called inside
  an outer jit, its `in_shardings` / `out_shardings` translate to WSCs
  in the outer trace.  At logical n_rmu these WSCs are uneven.
  Microbench Test 6 (V_q-tile-shaped body with full uneven WSC chain)
  confirmed XLA accepts this — but the microbench used a single jit,
  not nested ones.  **Verify by running the smoke bed at 661 with HLO
  dump and grepping for `all-gather`** to confirm zero unexpected
  collectives.
* ~~`compute_L_q_from_CCT` calls into `compute_L_q_2d_blocked` (the
  Cholesky), which uses `cholesky_2d.py` — that path may have its own
  divisibility assumptions.  Verify the chol cache key handles
  logical n_rmu at e.g. 661 cleanly.~~  RESOLVED — see "Cholesky
  helper" subsection above.  The outer-jit wrap should pass
  ``n_rmu_logical=meta.n_rmu`` to the Cholesky call; no further
  cholesky_2d.py work needed.

### `fit_one_rchunk` — same pattern, smaller scope

Modify `_make_fit_one_rchunk_kernel` to:

1. Accept ψ at padded n_rmu (top-level input).
2. Slice ψ to logical at the kernel body entry.
3. Slice L_q to logical at entry.
4. Run interior at logical n_rmu (Z_q computation, ZCT solve all use
   single-axis sharding; uneven WSCs are OK as intermediates).
5. Pad zeta_chunk back to padded n_rmu before return.

### Final seam — SlabIO write at logical extent

`fit_zeta_chunked_to_h5` currently calls
`zeta_io.create_dataset('zeta_q', shape=(nq, n_rtot, n_rmu), ...)`.
After Phase 3a `meta.n_rmu` is the logical centroid count, so the
existing call already creates the dataset at logical extent.  The
`write_slab` calls at `isdf_fitting.py:1554-1577` need
`valid_shape=(actual, n_rtot, meta.n_rmu)` so the padded zeta_chunk's
trailing μ slots are clipped on write.

## Test plan for Phase 3b

1. Generate a centroid file at n_rmu_T = 670 (4-divisible-by-2 stress
   test) via `centroid.kmeans_isdf -i cohsex.in 670` on the smoke bed.
2. Run smoke bed with `centroids_file_current=centroids_frac_670.txt`.
   ζ-fit + 7 V_q tiles must complete; on-disk zeta_q_mu*.h5 at logical
   670; on-disk V_qmunu_TT_ij at logical 670.
3. Run `XLA_FLAGS=--xla_dump_to=…` and `analyze_hlo_dump.py` — assert
   zero new `all-gather` collectives in `compute_L_q_from_CCT` and
   `fit_one_rchunk` HLO compared to the 668 baseline.
4. Repeat at n_rmu_T = 661 (prime) — every padding boundary stressed.

## Assumptions that came from the original brief and shaped the design

* "ISDF zeta fitting (triangular solve needs unpadded matrices)" — the
  Cholesky/LU **must** see logical n_rmu, no ridge regularisation.
  ([first user message in `padding_phase0_2026-05-08`'s referent])
* "zero tolerance" for all-gathers at function boundaries — every
  Phase 3b PR should ship with an HLO grep verifying `all-gather`
  count unchanged from a divisible baseline.
* "I don't want to change the plumbing that actually obtains the
  wavefunctions" — load_centroids' output is the only place the
  helper kicks in for ψ; don't restructure the FFT / scatter / unfold
  internals.
* "we also need all of the file writes to not be padded, since we want
  to be able to read those back into different processor configurations"
  — every SlabIO write site uses `valid_shape=logical`.
