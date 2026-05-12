# Ring (ppermute) algorithm for the V_q tile contract — design study

**Status:** design only, no code written, no GPU runs.
**Reference source:** `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src/gw/v_q_tile.py::_make_V_q_tile_kernel` (lines 353–555), gather-stage formula at `_gather_stage_per_q_bytes` (lines 87–108).

## 1. Math — ring contract pseudocode

Notation: mesh `(p_x, p_y)` with `p_prod = p_x · p_y`. Currently after FFT + sphere-pick, `zeta_L_G : (Q, μ, n_G)` lives in `P(None, ('x','y'), None)` — μ is **flat-sharded** across `p_prod` ranks, G is replicated. Call the per-rank slice `ζ_loc[μ_loc, G]` with shape `(Q, μ/p_prod, n_G)`.

Ring contract (single q for clarity; broadcast over Q is trivial):

```python
# Static: ring of size p_prod over axis ('x','y').
# Inputs: ζ_loc : (Q, μ/p_prod, n_G_sph)  on P(None, ('x','y'), None)
#         v(K)  : (Q, n_G_sph)             replicated
# Output: V_blk : (Q, μ/p_x, μ/p_y)        on P(None, 'x', 'y')

@shard_map(mesh=mesh_xy, in_specs=(P(None,('x','y'),None), P(None,None)),
           out_specs=P(None,'x','y'))
def ring_contract(zeta_local, v_per_G):
    rank = lax.axis_index(('x','y'))
    P    = p_prod
    incoming = zeta_local                                # owns ν_block_0
    # Pre-multiply the L-side ONCE on the resident μ slab (cheap, n_G):
    zeta_L = jnp.conj(zeta_local) * v_per_G[:, None, :]  # μ side gets v(K)
    V_local = jnp.zeros((Q, mu_per_rank, mu_per_rank * P), c128)
    perm = [(i, (i + 1) % P) for i in range(P)]          # right-shift ring
    for s in range(P):
        # einsum: my μ_local rows × current incoming ν rows
        V_partial = jnp.einsum('qmG,qnG->qmn', zeta_L, incoming)
        # which global ν columns did 'incoming' come from?
        src_rank = (rank - s) % P
        col_lo   = src_rank * mu_per_rank
        V_local  = lax.dynamic_update_slice(
                       V_local, V_partial, (0, 0, col_lo))
        if s < P - 1:
            incoming = lax.ppermute(incoming, ('x','y'), perm)
    return V_local
```

After `shard_map`, `V_local` has shape `(Q, μ/p_prod, μ)` per rank. To match the existing `V_acc` sharding `P(None,'x','y')` with shape `(Q, μ/p_x, μ/p_y)` we either (a) reshape inside `shard_map` since rows/cols on `('x','y')` map deterministically, or (b) emit `P(None,('x','y'),None)` and add one `with_sharding_constraint` — XLA inserts a single all-to-all to redistribute columns, which is one collective vs the current two all-gathers. Option (a) is preferred (no extra collective); the row-major flat→2D mapping is exact when `mu_per_rank` is the same on every rank (the chooser already enforces `μ_chunk % p_prod == 0`).

## 2. Memory analysis

Per-rank live bytes at the contract, complex128 (16 B):

- **Current** (from `_gather_stage_per_q_bytes`, lines 105–108):
  `16 · μ · n_G · (1/p_x + 1/p_y + 1/p_prod)`
- **Ring**: resident `ζ_loc` + rotating `incoming` (same shape):
  `16 · μ · n_G · (2 / p_prod)`

Crossover (ratio current/ring):

```
ratio = (1/p_x + 1/p_y + 1/p_prod) / (2/p_prod)
      = (p_x + p_y + 1) / 2          (square mesh: ratio = (2·p_x + 1)/2)
```

| mesh | p_prod | current factor | ring factor | savings |
|---|---|---|---|---|
| 2×2 |  4 | 1.25  | 0.500 | 2.5× |
| 4×4 | 16 | 0.5625| 0.125 | 4.5× |
| 4×8 | 32 | 0.406 | 0.063 | 6.5× |
| 8×8 | 64 | 0.266 | 0.031 | 8.5× |

For CrI3 16-GPU 4×4 with μ=1500, n_G≈30k: current gather peak ≈ `16 · 1500 · 30000 · 0.5625` = **0.40 GB/rank**; ring ≈ **0.090 GB/rank**. The chooser's `_Q_COMPUTE_COEF_FFT` path predicts FFT stage at ~5–10 GB/rank for the same problem, so 0.4 → 0.09 is invisible at this size. Savings only open up budget for **bigger μ at the same mesh**, or smaller n_rtot:n_G ratios where FFT stops dominating (see §6).

## 3. Communication analysis

Bytes moved per rank, single q, complex128:

- **Current**: 2 one-axis gathers, each gathers `μ · n_G / p_prod` from peers. Total `≈ 16 · μ · n_G · (1/p_x + 1/p_y) · (p_min − 1)/p_min` per rank along NCCL all-gather. Order: `O(μ · n_G)`.
- **Ring**: `p_prod − 1` ppermute hops, each `16 · μ · n_G / p_prod` bytes per rank. Total `16 · μ · n_G · (p_prod − 1) / p_prod` — **same order** as current, but spread across `p_prod − 1` small ops vs 2 fat all-gathers.

Topology effects on Perlmutter (4 GPUs/node intra-NVLink ≈ 250 GB/s, inter-node Slingshot ≈ 25 GB/s/NIC):

- NCCL all-gather is bandwidth-optimal: ring all-gather already moves `(P−1)/P · message` bytes, and hierarchical NCCL collapses the inter-node phase, so all-gather effectively pays `~max(intra, inter)` once.
- NCCL ppermute is point-to-point send/recv, no tree fusion. On 4×4 (one node) cheap. On 4×8 across 2 nodes, half the hops cross Slingshot one-by-one — same total bytes, but **latency-dominated** for small messages.

Order of magnitude for CrI3 16-GPU, μ=1500, n_G=30k → per-rank message `μ · n_G · 16 / p_prod ≈ 45 MB`. At NVLink 250 GB/s a 45 MB hop is ~0.18 ms; 15 hops → ~2.7 ms. Current 2 all-gathers move ~720 MB/rank over the all-gather tree at NVLink rate → ~6 ms. **Ring is comparable or faster on a single NVLink island.** Cross-node hops degrade ring more than all-gather since NCCL doesn't fuse.

## 4. Compute

Identical FLOPs: `Q · μ² · n_G` complex multiply-adds, just partitioned `p_prod` ways across `p_prod` rotations instead of one big local einsum after gathers. **No extra all-reduce** — each rank's `V_local` covers a distinct contiguous μ-row band, so accumulation is local. One `dynamic_update_slice` per rotation, no collective on V.

## 5. Implementation cost

Prototype in `gw/v_q_tile.py`:

- New `_kernel_body_ring(...)`: ~40 lines (pseudocode in §1 inlined under `shard_map`).
- New cache-key tag (`'unified-ring'`) in `_v_q_tile_kernel_cache`: 2 lines.
- Chooser flag: add `algorithm: 'gather'|'ring'` to `_choose_v_q_chunks` return dict; gate on env `LORRAX_V_Q_ALGORITHM=ring`. ~10 lines.
- Memory model: a `_ring_stage_per_q_bytes` parallel to `_gather_stage_per_q_bytes`, returning `16 · μ · n_G · 2/p_prod`. ~5 lines.
- Static flag plumbing: `same_zeta` and `write_g0` work unchanged. Distinct-ζ adds a second resident slab and a second rotating buffer → memory model becomes `4/p_prod`. `write_g0` is trivial: g0 is computed in `_zeta_disk_to_G` *before* the ring.

**Total: ~60–80 LOC** for the prototype, plus ~20 LOC of chooser/test scaffolding. Cleanest injection point: a new `algorithm` key in `chooser_choice`, switched on inside `_make_V_q_tile_kernel` before building the body. Env flag `LORRAX_V_Q_ALGORITHM` for A/B testing without touching call sites.

## 6. When to bother

The ring is a net win only when **gather-stage memory is the binding constraint**, i.e. when `(1/p_x + 1/p_y) · μ · n_G > FFT-stage peak`. From the chooser comments (lines 67–71), the FFT stage dominates whenever `n_rtot ≫ n_G`. So ring unlocks problems where:

- **Smaller `n_rtot/n_G`**: cutoffs that don't blow up the FFT box (denser k, sharper sphere), or 2D systems with thin z.
- **Larger μ** (>4–5k) where `μ · n_G / p_x` dominates even with moderate n_rtot.
- **Wider meshes** (4×8, 8×8): savings grow as `(p_x + p_y)/2`. At 64 GPUs, ring is **~8.5× memory savings** at gather stage.

CrI3 16-GPU at μ=1500 is **not** that regime — FFT controls; ring would be invisible. Triggers that move into the regime: μ≥4000 on 4×8, 2D systems on 8×8, or after the planned NUFFT/flat-k chi0 refactor (which deflates the FFT-stage peak and exposes gather).

## 7. Recommendation

**File as future work.** Concretely:

- Do not prototype now: chooser shows FFT-stage peak dominates for active CrI3/MoS2/Si workloads (`max(gather, fft)` in `_choose_v_q_chunks`), so ring would save memory in a stage that's not binding. Time-to-value zero.
- Revisit when one of: (a) μ exceeds ~4000 and gather-stage shows up in `per_rank_peak`; (b) a study uses a >4×8 mesh where `(p_x + p_y)/2` pays off; (c) the FFT stage is replaced (NUFFT / flat-k chi0 work referenced in `project_flat_k_chi0_pipeline.md`), exposing gather as the new bottleneck.
- Reuse path: the same rotating-`ζ` primitive could feed CCT/ZCT contractions downstream (those also all-gather a μ-sharded slab). If the flat-k chi0 refactor adopts ring for chi0/W, folding V_q onto the same primitive is nearly free.

**Bottom line:** prototype is small (~80 LOC) and well-scoped, but immediate workloads don't benefit. Park it as "ring V_q contract — prototype when gather stage appears in `per_rank_peak`."
