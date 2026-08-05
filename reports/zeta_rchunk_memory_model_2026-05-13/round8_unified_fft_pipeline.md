# Round 8 — Unified FFT-box pipeline (draft §§ 1–3)

**Status**: Agent 2 first draft, sections 1–3. Agent 1 expected to weight in on the SPMD/sharding correctness of §1's universal-body shape; Agent 4 + Agent 3 join after Round 7 validation lands. §§ 4–8 are stubs awaiting Agent 1's response.

Source of truth for "what we have today": `lorrax_B` `agent/zeta-bc-scan-shardmap` @ `c796420` (post Round-7 back-pad fix).

## 1. The unified primitive

### 1.1. Two functions, one body shape

The seven FFT-box-touching sites split cleanly along the FFT direction:

- **Forward** (G-sphere → r-sampling): five sites — `to_rmu`, `to_rchunk`, `gflat_to_rmu`, `gflat_to_rchunk` (DEAD), the per-iter front-half of `z_q_from_psi_sm._local`, the per-iter front-half of `c_q_from_psi_sm._local` (mirror of z_q).
- **Reverse** (r-sampling → G-sphere): one site — `accumulate_rchunk_to_gflat`.

Forward and reverse share enough body structure that a single `box_pipeline_scan` parametric function is workable, but the conditional branching (ifftn vs fftn, sample-then-phase vs phase-then-pad-then-scatter) makes the conditional read worse than the deduplication is worth. **Recommend two cousins**, both with the same scan-inside-shard_map skeleton:

```python
def box_scan_forward(...)  # G-sphere → r-sampling, per-iter scan
def box_scan_reverse(...)  # r-sampling → G-sphere, per-iter scan
```

`_box_kernel` (the G-sphere ↔ FFT-box gather primitive) remains a shared building block for both — it's already the one shared point in today's code.

### 1.2. Universal body shape — forward

```python
@partial(shard_map, mesh=mesh, in_specs=..., out_specs=..., check_rep=False)
def _local(*args_in):
    # ── PRE-SCAN SETUP ─────────────────────────────────────────────
    x_idx = jax.lax.axis_index('x')
    y_idx = jax.lax.axis_index('y')
    carry_init = carry_init_fn(...)              # rank-local zeros, or None for "no carry"

    # ── PER-ITER BODY ──────────────────────────────────────────────
    def body(carry, iter_idx):
        # (1) PULL — per-rank input slab.
        psi_G_slab = pull_fn(x_idx, y_idx, iter_idx, *args_in)
            # → (nk, bands_per_iter, ns, ngkmax)   c128 per-rank-local
            # Source variants:
            #   io_callback (host-tile slicer; one bc per iter)
            #   dynamic_slice on a jit-arg (per-rank device psi_G slab)
            #   identity (single-shot variant; n_iters=1, pulls all of psi_G)

        # (2) PRE-NORMALIZE — pseudobands or identity.
        psi_G_slab = pre_norm_fn(psi_G_slab, iter_idx)
            # Identity for regular bands.  Per-band divide for pseudobands;
            # composable as `psi *= 1/norm[band]`.

        # (3) FFT-BOX GATHER (G-sphere → FFT-box; the one shared primitive).
        box = _box_kernel(psi_G_slab, g_index_local, ngkmax=ngkmax)
            # → (nk, bands_per_iter, ns, nx, ny, nz)

        # (4) IFFT — per-rank local cuFFT on the trailing 3 axes.
        rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm=norm)

        # (5) APPLY BLOCH PHASE — pre-sample (full-box) or post-sample (on-slice).
        rb = apply_phase_fn(rb, kvecs_frac, sample_spec)
            # 'full_box' kind: phase on box pre-sample (to_rmu).
            # 'flat_r_slab' / 'centroids' kind: phase on the sampled cells
            #     only (apply_bloch_phase_on_slice; cheaper).

        # (6) SAMPLE at the point set.
        psi_r_local = sample_fn(rb, sample_spec)
            # 'full_box':       identity (return rb)
            # 'flat_r_slab':    dynamic_slice on flat-r axis, given (r0, r_len)
            # 'centroids':      gather at r_mu[:,0], r_mu[:,1], r_mu[:,2]

        # (7) BAND GATHER — collect bands across mesh axes, if input is band-sharded.
        if band_gather_axes is not None:
            psi_r = jax.lax.all_gather(
                psi_r_local, axis_name=band_gather_axes,
                axis=BAND_AXIS, tiled=True)
        else:
            psi_r = psi_r_local

        # (8) POST-GATHER REFINE — e.g. ZCT's per-rank r-slice.
        psi_r = post_gather_fn(psi_r, x_idx, y_idx, iter_idx)
            # Identity for sites where step (6) already produced the
            # per-rank sample.  For ZCT: dynamic_slice on the r-axis to
            # the y-rank's r_loc slab (Round 6 lesson — gather BEFORE
            # this slice or the r-axis becomes incoherent).

        # (9) POST-OP — consume the per-iter ψ(sample).
        new_carry = post_op_fn(carry, psi_r, iter_idx, *args_in)
            # Variants (composable callables):
            #   'return-as-output':    dus_into_output(out, psi_r, iter_idx)
            #   'einsum-into-carry':   carry + einsum(psi_X_bc, mask·psi_r)
            #   'accumulate-donated':  add into a donated jit-arg buffer

        return new_carry, None

    # ── SCAN ─────────────────────────────────────────────────────
    final_carry, _ = jax.lax.scan(
        body, carry_init, jnp.arange(n_iters, dtype=jnp.int32))
        # unroll=1 (default; do NOT override — relies on per-iter
        # sequential lifetime for FFT-box aliasing).

    # ── POST-SCAN ────────────────────────────────────────────────
    # Optional site-specific tail (e.g. ZCT's IFFT-k → γ̃ → FFT-k → transpose).
    return post_scan_fn(final_carry, *args_in)
```

The universal body is **9 named hooks** (`pull_fn`, `pre_norm_fn`, `apply_phase_fn`, `sample_fn`, `band_gather_axes`, `post_gather_fn`, `post_op_fn`, `carry_init_fn`, `post_scan_fn`) plus the static `_box_kernel + ifftn` core. Each hook is either:
- A pure Python lambda / callable closure (composable transform), OR
- A static config value (parametric knob).

### 1.3. Universal body shape — reverse

```python
def _local(rchunk, donated_acc, *args_in):
    x_idx, y_idx = jax.lax.axis_index('x'), jax.lax.axis_index('y')

    def body(acc, iter_idx):
        # (1) PULL — per-rank slab row from the input rchunk (jit-arg slice).
        sub = pull_fn(rchunk, iter_idx)                       # (cs, ns, r_len)

        # (2) PRE-PHASE — exp(σ · 2πi q·r) BEFORE the FFT.  Cheap on-slice.
        sub = apply_phase_fn(sub, qvec_frac, sample_spec)     # (cs, ns, r_len)

        # (3) SCATTER to padded full box.
        buf = sample_inverse_fn(sub, sample_spec)             # (cs, ns, n_rtot)

        # (4) FFT (forward direction) — local cuFFT on the trailing 3 axes.
        box = buf.reshape(cs, ns, nx, ny, nz)
        G_box = jnp.fft.fftn(box, axes=(-3, -2, -1), norm=norm).reshape(cs, ns, n_rtot)

        # (5) GATHER at sphere indices.
        G_sph = jnp.take_along_axis(
            G_box, sphere_idx_per_iter, axis=-1, mode='promise_in_bounds')

        # (6) POST-OP — accumulate into donated acc at iter offset.
        acc = post_op_fn(acc, G_sph, iter_idx)                 # dus(acc, acc[i0]+G_sph, i0)
        return acc, None

    out_acc, _ = jax.lax.scan(body, donated_acc, jnp.arange(n_iters, dtype=jnp.int32))
    return out_acc
```

Same scan-inside-shard_map skeleton; the per-iter pipeline runs in **opposite order** (PRE-phase → scatter → FFT → gather, vs forward's gather → IFFT → POST-phase → sample). The hook surface is similar (`pull_fn`, `apply_phase_fn`, `sample_inverse_fn`, `post_op_fn`).

The reverse case has only ONE in-tree caller today (`accumulate_rchunk_to_gflat`); the unification value is mostly aesthetic. **Recommend keeping `box_scan_reverse` thinner** — match the skeleton but don't force every hook to be parametric.

### 1.4. Signature sketch (forward)

```python
def box_scan_forward(
    # Inputs the body consumes:
    *args_in,                              # passed through shard_map.in_specs
    # Source / sampling configuration:
    pull_fn,                                # per-iter (x, y, iter, *args) → ψ_G_slab
    pre_norm_fn=lambda psi, _: psi,        # default: identity
    apply_phase_kind='post_slice',         # 'pre_box' (full-box phase) or
                                            #   'post_slice' (only sample cells)
    sample_kind,                            # 'full_box' / 'flat_r_slab' / 'centroids'
    sample_args,                            # (r0, r_len) / r_mu / None
    band_gather_axes=None,                  # mesh axis tuple, e.g. ('x','y'); None for no gather
    post_gather_fn=lambda x, *_: x,        # default: identity
    post_op_fn,                             # (carry, psi_r, iter, *args) → new_carry
    carry_init_fn,                          # callable that returns rank-local zeros
    post_scan_fn=lambda carry, *_: carry,  # default: identity
    # Scan configuration:
    n_iters,
    # Mesh + sharding:
    mesh,
    in_specs,
    out_specs,
    # Closure constants (replicated):
    g_index,                                # threaded through in_specs as P() replicated
    kvecs_frac=None,                        # threaded similarly
    fft_grid,
    norm='ortho',                           # FFT norm
) -> jax.Array:
```

The `*_fn` hooks are pure-jax callables (NOT shard_map-aware — they run inside the body's rank-local context). The `*_kind` strings select among a small set of pre-built helpers (`_sample_full_box`, `_sample_flat_r_slab`, `_sample_centroids`, etc.) — keeps the body's branching at trace time.

## 2. Each of the 7 current sites mapped onto the primitive

### 2.1. `gflat_to_rchunk` (DEAD after `c796420`)

**Verdict**: delete the standalone function in the unification cleanup commit. The kernel bypasses it; the only in-tree caller was `_make_fit_one_rchunk_kernel._kernel` (now using `z_q_from_psi_sm` directly).

If a future caller needs G-sphere → r-slab without the pair-density einsum, the unified primitive recovers the helper as a 3-line wrapper:

```python
def gflat_to_rchunk_wrapper(psi_G, g_index, *, r0, r_len, mesh, ...):
    return box_scan_forward(
        psi_G, args_in=(psi_G,), n_iters=1,
        pull_fn=lambda *_: psi_G,                                # one-shot
        sample_kind='flat_r_slab', sample_args=(r0, r_len),
        post_op_fn=lambda _, psi_r, *__: psi_r,                  # return as-is
        carry_init_fn=lambda: jnp.zeros((nk, nb, ns, r_len), c128),
        ...)
```

### 2.2. `gflat_to_rmu` (currently used; centroid load)

Maps cleanly onto `box_scan_forward`:

| Hook | Value |
|---|---|
| `pull_fn` | `dynamic_slice_in_dim` on jit-arg `psi_G` — pull one bc-equivalent flat slab per iter |
| `pre_norm_fn` | identity (norms divide is upstream in `load_centroids_band_chunked`) |
| `apply_phase_kind` | `'post_slice'` (post-IFFT, on the gathered centroid cells) |
| `sample_kind` | `'centroids'` |
| `sample_args` | `r_mu` (replicated `(n_rmu, 3)` int32 table) |
| `band_gather_axes` | `None` (output keeps band axis sharded; no inter-rank gather) |
| `post_op_fn` | `dynamic_update_slice` into the flat output slab at row offset `i·cs` |
| `carry_init_fn` | `jnp.zeros((N + pad_N, ns, n_rmu), c128)` |
| `post_scan_fn` | `out_flat[:N].reshape(nk, nb_local, ns, n_rmu)` |

Today's `gflat_to_rmu` is already structurally this; the unification mostly renames variables and moves the body into the `box_scan_forward` shell. No semantic change.

### 2.3. `accumulate_rchunk_to_gflat` (reverse direction)

Maps onto `box_scan_reverse`:

| Hook | Value |
|---|---|
| `pull_fn` | `dynamic_slice_in_dim` on jit-arg `rchunk` at row offset `i·cs` |
| `apply_phase_fn` | `apply_bloch_phase_on_slice(sign=-1)` on the r-slab (BEFORE FFT) |
| `sample_inverse_fn` | `dynamic_update_slice` into a zeroed `(cs, n_rtot)` buffer at offset `r0` |
| `post_op_fn` | `dynamic_update_slice(acc, acc[i0] + G_sph, i0)` (sphere index gathered per row via `q_row` lookup) |
| Donated input | `gflat_acc` |

Today's `accumulate_rchunk_to_gflat` is structurally this; same renaming-only port.

### 2.4. `to_rmu` / `to_rchunk` (single-shot fallbacks)

`n_iters=1`, no scan needed at runtime (XLA folds the single-iter scan). Maps onto `box_scan_forward` with `pull_fn=lambda *_: psi`:

| Hook | `to_rmu` | `to_rchunk` |
|---|---|---|
| `pull_fn` | one-shot pass-through of jit-arg `psi` | same |
| `apply_phase_kind` | `'pre_box'` (the classic to_rmu does the full-box phase before centroid gather) | `'post_slice'` |
| `sample_kind` | `'centroids'` | `'flat_r_slab'` |
| `sample_args` | `r_mu` | `(r0, r_len)` |
| `band_gather_axes` | `None` | `None` |
| `post_op_fn` | return as-is (carry-less variant; output = single-iter slab) | same |

These could keep their current public API as thin wrappers around `box_scan_forward` so existing callers don't break.

### 2.5. `_box_kernel` (primitive)

**Stays as-is.** It's the single shared building block under step (3) of the forward body. No unification work needed — it already serves all five forward callers and the equivalent of `box_kernel.inverse` lives inline in the reverse body's scatter step.

Recommended cleanup: extract the scatter symmetrically as `_box_scatter(rb_slab, sphere_idx, fft_grid)` so reverse has its named primitive too. Small (~20 LOC), aesthetic.

### 2.6. `z_q_from_psi_sm._local` (NEW: c796420)

This is the load-bearing case. Maps onto `box_scan_forward` with **the most hooks engaged**:

| Hook | Value |
|---|---|
| `pull_fn` | `io_callback(psi_G_store._slice_local_tile_bc, ...)` — host-tile slice for one bc per iter, padded to `bpd_max` |
| `pre_norm_fn` | identity (norms pre-divided into `psi_l_X` / `psi_r_X` outside the kernel) |
| `apply_phase_kind` | `'post_slice'` (Bloch phase on the per-rank r-slab cells only) |
| `sample_kind` | `'flat_r_slab'` |
| `sample_args` | `(r_start_dyn, n_zchunk)` — **FULL r-chunk per rank**, NOT the per-rank r_loc (Round 6 lesson) |
| `band_gather_axes` | `('x','y')` — all_gather along the band axis to collect across joint-sharded host |
| `post_gather_fn` | `dynamic_slice_in_dim(psi_Y_bc, y_idx * r_loc, r_loc, axis=R_AXIS)` — slice the y-rank's r_loc slab AFTER the gather |
| `post_op_fn` | L/R mask + 2 einsums into 2 rank-5 carries (`P_l_acc`, `P_r_acc`) |
| `carry_init_fn` | `(jnp.zeros((nk, ns, r_loc, mu_loc, ns), c128),) * 2` |
| `post_scan_fn` | IFFT-k → γ̃·γ̃ contract → FFT-k → transpose tail (the unchanged post-pair pipeline) |

Two non-trivial wrinkles c796420 carries that the unified primitive must surface:

- **Front+back pad of `psi_l_X` / `psi_r_X`** — needed for the L/R mask to coexist with `dynamic_slice_in_dim`'s silent OOB clamp. The pad happens OUTSIDE the scan body (one-shot pre-scan) and is closure-static. Belongs in the `post_op_fn`'s closure setup, NOT in the universal body.
- **Carry-and-output-both** — the post-scan tail returns the actual output (`Z_q`), so `post_scan_fn` does substantial work. This is the strongest argument for keeping `post_scan_fn` as a first-class hook in the universal signature (not just an identity-default add-on).

### 2.7. `c_q_from_psi_sm._local` (TO MIRROR ZCT — currently OLD design)

Mirror of §2.6 with **only the `sample_kind` changing**:

| Hook | `c_q_from_psi_sm._local` |
|---|---|
| `pull_fn` | `io_callback(psi_G_store._slice_local_tile_bc, ...)` — same as ZCT |
| `sample_kind` | `'centroids'` (instead of `'flat_r_slab'`) |
| `sample_args` | `r_mu` |
| `band_gather_axes` | `('x','y')` — same as ZCT |
| `post_gather_fn` | identity (no r-axis to slice — the `r_mu` axis is replicated across all ranks naturally; n_col = n_rmu instead of r_chunk) |
| `post_op_fn` | L/R mask + 2 einsums into 2 rank-5 carries — identical math to ZCT |
| `carry_init_fn` | `(jnp.zeros((nk, ns, n_rmu_local, mu_loc, ns), c128),) * 2` (n_col = n_rmu instead of r_chunk) |
| `post_scan_fn` | IFFT-k → γ̃·γ̃ contract → FFT-k → transpose tail (identical structure to ZCT) |

**The only differences between CCT and ZCT in this design are**:
1. `sample_kind`: `'centroids'` vs `'flat_r_slab'`.
2. `sample_args`: `r_mu` vs `(r_start, r_chunk_size)`.
3. The post-gather r-slice (CCT skips it; ZCT slices to the y-rank's r_loc).
4. The carry's column axis: `n_rmu_local` vs `r_loc`.
5. CCT output is `(nq, n_rmu, n_rmu)` (square); ZCT output is `(nq, n_rmu, n_zchunk)`.

Everything else is bitwise identical, including the L/R mask + einsum + post-pair pipeline. This is the unification payoff for the CCT side: it eliminates the OLD `c_q_from_psi_sm._local`'s upstream `psi_l_rmu_Y_fit` / `psi_r_rmu_Y_fit` materialization (currently ~2 GB each, lives across the kernel call) by pulling centroids via io_callback per bc inside the scan body — same memory win as ZCT got in Rounds 3–6.

Round 9 implementation order should be: **port CCT first** (smaller surface, well-understood numerics from today's tests), then **delete the rough edges** in `wfn_transforms.py` once `gflat_to_rmu` is also folded into the unified primitive.

## 3. Parametric vs composable vs site-specific

Three categories for the variation points, by who controls them and at what trace level.

### 3.1. Parametric (static, closure-time string / config)

These are dispatched at trace time via small `if` ladders in the universal body. Adding a new value here requires editing the universal primitive.

| Knob | Values | Affects |
|---|---|---|
| `direction` | `'forward'` / `'reverse'` | Whether to call `box_scan_forward` or `box_scan_reverse` (two different primitives, not a runtime branch) |
| `sample_kind` | `'full_box'` / `'flat_r_slab'` / `'centroids'` / `'sphere_scatter'` (reverse only) | Selects which `_sample_*` / `_sample_inverse_*` helper runs at step (6) / (3) |
| `apply_phase_kind` | `'pre_box'` / `'post_slice'` / `'pre_fft'` (reverse) / `'none'` | Where in the pipeline the Bloch phase goes |
| `band_gather_axes` | `None` / `('x',)` / `('y',)` / `('x','y')` | Which mesh axes to gather along after sampling |
| `norm` | `'ortho'` / `'forward'` / `'backward'` | FFT normalization passed to `jnp.fft.(i)fftn` |

The `sample_kind` × `apply_phase_kind` matrix is small (~12 valid combinations); a flat dispatch in the body is acceptable. No site needs a sample-kind not listed here as far as I can see.

### 3.2. Composable (jax callables; closure-captured)

These are user-supplied Python functions threaded into the body. They run inside the rank-local scan body and must be pure-jax (no Python control flow on traced values that can't be JIT'd).

| Hook | Type | Purpose |
|---|---|---|
| `pull_fn` | `(x_idx, y_idx, iter_idx, *args_in) → ψ_G_slab` | Source of the per-iter input. Composes io_callback-host-tile, dynamic-slice-jit-arg, and identity (single-shot) cases without primitive-level branching |
| `pre_norm_fn` | `(ψ_G_slab, iter_idx) → ψ_G_slab` | Identity by default. Pseudobands: `ψ * 1/norm[band_idx]` per row. ONE clean place for this; no need for a separate "pseudobands path" anywhere |
| `post_gather_fn` | `(ψ_r, x_idx, y_idx, iter_idx) → ψ_r` | Identity by default. ZCT uses it for the per-rank r-axis slice; future sites could use it for masking, transposition, etc. |
| `post_op_fn` | `(carry, ψ_r, iter_idx, *args_in) → new_carry` | The single point where each site differs in how the per-iter ψ(sample) is consumed: write-to-slab, einsum-into-carry, accumulate-into-donated. Composable closure (over band-mask tables, psi_l_X, etc.) |
| `carry_init_fn` | `() → carry_init` | Rank-local zeros of the right shape. Doesn't take traced args |
| `post_scan_fn` | `(final_carry, *args_in) → output` | Identity by default. For ZCT/CCT: the IFFT-k + γ̃ + FFT-k + transpose tail. For everything else: identity or a `final_carry[:N].reshape(...)` |

Pseudobands as a `pre_norm_fn` is the user-directive-honoring detail: "pseudobands are not a separate code path; they're a normalization composable". No special branching anywhere except the `pre_norm_fn` callable's closure.

### 3.3. Site-specific (not unified — different per site)

These ought NOT be lifted into the universal primitive. The wrapper functions (e.g. `z_q_from_psi_sm`, `gflat_to_rmu`, `accumulate_rchunk_to_gflat` callsites) build the closures.

| Site detail | Why not unified |
|---|---|
| Front+back-pad of `psi_l_X` / `psi_r_X` (ZCT/CCT) | Only ZCT/CCT need it; pad sizes depend on the (L_lo_g, L_hi_g) × `band_chunk_ranges` cross product. Lives in the `post_op_fn` closure |
| Per-bc L/R mask tables | Same — only ZCT/CCT, only when L ≠ R or L ≠ full |
| Static front-pad shape for `_slice_local_tile_bc` callback's `out_sds` | Only the host-tile-io_callback `pull_fn` needs it; `_bpd_max` is a `PsiGStore` field |
| Donated accumulator for `accumulate_rchunk_to_gflat` | Reverse-direction-specific; the forward primitive doesn't donate (output is freshly allocated) |
| The post-pair IFFT-k + γ̃ + FFT-k tail (ZCT/CCT) | Only ZCT/CCT; the tail is the same kgrid-FFT-and-γ̃ sequence so the two sites share the SAME `post_scan_fn` closure (parameterized by `n_col`, `mu_loc`, `gamma_*`). Mid-level helper, not body-level |
| `bpd_max` static padding contract (band-flat-sharded host) | Only sites that pull from `PsiGStore`'s band-flat-sharded host tiles — currently ZCT (and CCT after Round 9). The padding logic stays in `PsiGStore._slice_local_tile_bc` |

The site-specific pieces are kept LOCAL to each wrapper — the universal primitive is unaware of them and treats them as opaque closures. This is the "as much unification as possible without polluting the primitive" line.

### 3.4. Summary table — what's where

| Variation point | 2.1 (DEAD) | 2.2 gflat_to_rmu | 2.3 acc_to_gflat | 2.4 to_rmu | 2.4 to_rchunk | 2.6 z_q | 2.7 c_q (after) |
|---|---|---|---|---|---|---|---|
| direction | fwd | fwd | **rev** | fwd | fwd | fwd | fwd |
| sample_kind | flat_r_slab | centroids | flat_r_slab→box | centroids | flat_r_slab | flat_r_slab | centroids |
| apply_phase_kind | post_slice | post_slice | pre_fft | pre_box | post_slice | post_slice | post_slice |
| band_gather_axes | — | None | None | None | None | ('x','y') | ('x','y') |
| pre_norm_fn | id | id | id | id | id | id (norms pre-divided) | id (norms pre-divided) |
| post_op_fn | dus | dus | accumulate-donated | return | return | mask+einsum→2 carries | mask+einsum→2 carries |
| post_scan_fn | reshape | reshape | id | id | id | IFFT-k+γ̃+FFT-k+T | IFFT-k+γ̃+FFT-k+T |
| n_iters | 1 (cs=N) | ceil(N/cs) | ceil(N/cs) | 1 | 1 | n_bc | n_bc |
| pull_fn source | jit-arg | jit-arg | jit-arg | jit-arg | jit-arg | io_callback | io_callback |
| Front+back-pad? | no | no | no | no | no | **yes** | **yes** |

The seven sites collapse to **at most 2 functions** (`box_scan_forward`, `box_scan_reverse`) plus the shared `_box_kernel` / `_box_scatter` primitives. CCT and ZCT differ in **exactly 2 cells of the row above** (sample_kind, sample_args + post-gather details).

---

## §§ 4–8 (stubs awaiting Agent 1)

### 4. Double-chunking (io batch × fft batch)  ⟨Agent 1⟩

**The user's "double-chunked wfn in file → wfn rchunk or centroids" is already covered by `n_iters` + `pull_fn`** — it doesn't need a second nesting level inside the universal body. Below: why, where the two knobs would actually pull apart, and what the primitive needs to accept for future-proofing.

#### 4.1. Today: `n_iters` IS the io-chunking knob

In the current `z_q_from_psi_sm._local` (commit `c796420`):
- `n_iters = len(band_chunk_ranges)` (the bc count; 10 for CrI3 charge).
- Per iter: one `io_callback` (one bc's bands → device) + one `to_rchunk_inner` IFFT (full FFT box, `bpd_max=1` band per rank) + one `all_gather` + L/R einsums.
- Single FFT batch per iter: trace-time-fixed at `bpd_max · ns` (= 2 for CrI3 charge).

So there's ONE chunking knob (`n_iters` = `n_bc`). The user's mention of "double-chunked" is read most charitably as: "**the io-side chunking** (bc grouping for the host fetch) and **the fft-side chunking** (batch size for cuFFT amortization) are *conceptually distinct*; the universal primitive should expose both even if they collapse to 1 today."

#### 4.2. The two knobs, decoupled in concept

| Knob | Symbol | Semantics | Affects |
|---|---|---|---|
| **io_batch** | `cs_io` | rows pulled per `pull_fn` call (per scan iter) | host→device bandwidth, Python overhead per call, transient `psi_G_slab` size |
| **fft_batch** | `cs_fft` | rows processed per `_box_kernel + ifftn` call (per inner mini-scan) | cuFFT batch dimension, per-batch FFT-box size |

Three concrete decoupling regimes:

- **`cs_io == cs_fft == bpd_max`** (today's z_q/c_q): pull one bc, FFT it, repeat. The two knobs collapse. **Optimal at CrI3 production scale.** Per-iter transients: io tile ≈ 80 MB, FFT box ≈ 5 GB (cuFFT scratch dominant), gathered band slab ≈ 340 MB. All scan-aliased to single slots.
- **`cs_io > cs_fft`** (io-coarse, fft-fine): one io_callback pulls several bcs at once; inside, an inner scan does FFT in smaller batches. Useful when **host call overhead dominates** (e.g., NSF-class systems with very small bc, ~10 µs per io_callback × hundreds of bcs). Not the current bottleneck.
- **`cs_io < cs_fft`** (io-fine, fft-coarse): would require buffering multiple io pulls into one FFT call. **Doesn't fit the scan model** without an outer staging buffer — equivalent to enlarging `bpd_max` upstream in `PsiGStore`'s tile layout. Out-of-scope for this primitive; a `PsiGStore` redesign question instead.

For the universal primitive, regime 1 is the only one shipped today. Regimes 2/3 are speculative.

#### 4.3. Recommendation: ship with `n_iters` only; reserve `fft_batch_size` as a planner-time knob

**Do NOT add an inner `lax.scan` inside each `body(carry, iter_idx)` call** for fft-batching. Two reasons:

1. **No production demand.** At CrI3 80 Ry the FFT box per iter is ~5 GB (cuFFT scratch dominant); it's already well below the 28 GB / GPU budget. Adding a second nested scan layer would cost compile time + complexity for zero memory win.
2. **The cuFFT-efficiency case (`cs_fft > bpd_max`) is achievable by a different mechanism**: enlarge `bpd_max` upstream by changing `PsiGStore`'s host-tile bc-stacking to group multiple bcs together. That's a `PsiGStore` policy decision, not a primitive shape change.

Instead, the primitive signature should make the io/fft equivalence explicit:

```python
def box_scan_forward(
    ...,
    n_iters,                          # outer scan length; = ceil(N_rows / cs_io)
    rows_per_iter,                    # static int; cs_io = cs_fft today (Agent 2 §1.4)
    # Future-proofing: a second mini-scan if any caller demands it.
    # NOT IMPLEMENTED in the first version; the parameter slot stays
    # absent until a site needs decoupled cs_io != cs_fft.
):
```

The primitive guarantees `rows_per_iter` is the FFT batch dimension (`cs_fft = rows_per_iter`). If a caller wants decoupled io batching, they layer it via the `pull_fn`'s closure (buffer up multiple io_callback responses, dispatch one FFT call). The primitive's body sees `rows_per_iter` rows per `body()` invocation — that's the only thing it needs to know.

#### 4.4. Concrete sites, mapped onto `n_iters` × `rows_per_iter`

| Site | `n_iters` | `rows_per_iter` (= cs_fft) | Notes |
|---|---|---|---|
| `gflat_to_rchunk` (DEAD wrapper) | `ceil(N/cs)` | `cs = 8` | flat-axis (k·n_local) chunking; the existing default. |
| `gflat_to_rmu` | `ceil(N/cs)` | `cs = 8` | same flat-axis pattern. |
| `accumulate_rchunk_to_gflat` | `ceil(N/cs)` | `cs = 8` | reverse direction; same. |
| `to_rmu` / `to_rchunk` | `1` | `N` (one-shot) | single-iter scan; XLA folds. |
| `z_q_from_psi_sm._local` | `n_bc` (e.g. 10) | `bpd_max = 1` (band-flat sharded) | bc-aligned; matches Round 6 design. |
| `c_q_from_psi_sm._local` (after Round 9) | `n_bc` | `bpd_max = 1` | mirror of z_q. |

**No site needs a nested inner scan.** The flat-axis sites use `n_iters > 1, rows_per_iter > 1` with a uniform `cs`; the bc-aligned sites use `n_iters = n_bc, rows_per_iter = bpd_max`. Both are single-level scans.

#### 4.5. The HLO consequence

Single-level scan inside shard_map → single `WhileOp` inside the manual-mode body, single FFT-box slot aliased across iters. **The Round 6 HLO findings (5cadd4b → c796420 trajectory) confirm this works**: the FFT-box-class slot count collapsed from 58 (Round 6 pre-Path-D) to 1 (after the scan-aliased single body). Adding a second nested scan would risk reintroducing slot multiplication (XLA's WhileOp-inside-WhileOp aliasing is less aggressive than single-WhileOp aliasing). **Avoid until forced.**

#### 4.6. When to revisit

Trigger conditions for re-opening the double-chunking decision:
- A site materializes a FFT box larger than `memory_per_device_gb / 2` per rank — implies need to fft-batch smaller than the io batch.
- Profiling shows host-side Python overhead dominates a hot path — implies need to io-batch more iters together.
- A new sampling kind (e.g. multi-r-slab gather) needs different cs_io vs cs_fft semantics.

Until then, **`cs_io == cs_fft == rows_per_iter`** is a single knob and the primitive ships with one.

### 5. CCT path: how to mirror ZCT path  ⟨Agent 1⟩

§2.7 has the structural answer in 5 cells of a table. This section spells out the Round-9 migration: API change at the callsite, body delta from ZCT, the centroid-specific quirks, and the validation gate set.

#### 5.1. The current shape of `c_q_from_psi_sm` (pre-migration)

Today (`f567aa0` / `c796420` — unchanged by Round 6/7):

```python
def c_q_from_psi_sm(
    psi_l_X, psi_l_Y, psi_r_X, psi_r_Y,    # 4 pre-materialized tensors
    gamma_L=None, gamma_R=None,
    *, kgrid, mesh_xy,
) -> jax.Array:    # C_q at (nq, n_rmu, n_rmu) sharded P(None, 'x', 'y')
```

The `psi_l_Y` / `psi_r_Y` inputs are `(nk, nb, ns, n_rmu)` sharded `P(None, None, None, 'y')` — built upstream by the centroid loader. They live across the kernel call as **two ~2 GB rank-replicated tensors per rank** (n_rmu=376, nb=160, full bands per rank because R_spec replicates bands).

This is the same defect class ZCT had pre-Round-6: pre-materialized full-bands Y tensors with a reshard from band-sharded host → band-replicated consumer. ZCT got the streaming-scan rewrite in Round 6. **CCT still has the old design.**

#### 5.2. Target post-migration shape (mirror of z_q)

```python
def c_q_from_psi_sm(
    psi_l_X, psi_r_X,                          # only X-side (μ-on-'x', bands replicated)
    psi_G_store,                                # closure: per-bc io_callback source
    *, band_chunk_ranges, band_range_left, band_range_right,
    fft_grid, gamma_L=None, gamma_R=None,
    kgrid, mesh_xy,
) -> jax.Array:    # C_q at (nq, n_rmu, n_rmu)
```

The signature change drops `psi_l_Y` / `psi_r_Y` from the input list (mirror of Round 6 z_q rewrite at `isdf_fitting.py:378-393`). The `r_start_dyn` / `r_chunk_size` are absent because CCT samples at fixed centroid positions, not r-chunks.

Body structure mirrors `z_q_from_psi_sm._local` (commit `c796420`) with three swaps:

| Step | ZCT (z_q) | CCT (c_q) — proposed |
|---|---|---|
| (5) sampling | `to_rchunk_inner(psi_G_bc, ..., r_start_, n_zchunk, ...)` — full r-chunk, sliced after gather | `to_rmu_inner(psi_G_bc, ..., r_mu, ...)` — centroid gather; no post-gather r-slice |
| `sample_args` | `(r_start_dyn, n_zchunk)` | `r_mu` (closure-static `(n_rmu, 3)` int32 table) |
| `r_loc` (carry's column) | `n_zchunk // p_y` (= 18412 at CrI3) | `n_rmu // p_y` (= 94 at CrI3, since `n_rmu = 376` per the Round 5 retraction) |
| post-pair pipeline | unchanged (IFFT-k → γ̃ → FFT-k → transpose) | unchanged (same structure; only `n_col` changes) |

Everything else — io_callback slicer API, IFFT, all_gather across `('x','y')` on the band axis, L/R mask construction, front+back-pad of `psi_l_X`/`psi_r_X` (Round 7 `c796420` fix), interleaved L+R einsums into rank-5 carries — copies byte-for-byte from `c796420`'s `z_q_from_psi_sm`.

#### 5.3. The CCT-specific differences (3 cells in §2.7's table)

##### 5.3.1. `sample_kind = 'centroids'`

`to_rmu_inner` exists at `wfn_transforms.py:527` (parallel to `to_rchunk_inner`, kept per Round 7 audit). It does the full-box IFFT + Bloch phase + centroid gather. Per-rank FFT-box transient is identical to the rchunk variant (the IFFT runs on the full grid regardless of sampling). The CENTROID gather samples at `r_mu[:, 0], r_mu[:, 1], r_mu[:, 2]` instead of slicing flat-r — pure-jax fancy index.

The centroid r-axis is **NOT sharded the same way the r-chunk axis is in ZCT**. In ZCT, the kernel emits output sharded `P(None, 'x', 'y')` with the third axis being `n_zchunk` (r-axis on `'y'`). For CCT, the output is `P(None, 'x', 'y')` with the third axis being `n_rmu` (centroid axis on `'y'`).

Critically: **`n_rmu` is `n_rmu_padded` upstream of this kernel** (centroid loader's padding contract: `n_rmu_padded ≡ ∏ p_a`). So `n_rmu // p_y` divides cleanly. The pre-flight `n_rmu_padded % p_y == 0` check from ZCT (Round 5 §5.3.6) carries over.

##### 5.3.2. `post_gather_fn = identity`

ZCT's `post_gather_fn` slices the y-rank's r_loc slab AFTER the all_gather (the Round 6 §2.10 r-coherence fix). CCT doesn't need this slice. Reason:

- ZCT's sampling produces a **full r-chunk per rank** before the gather (each y-rank has the SAME full r-chunk for ITS bc-bands). After gather, each rank has all bands × full r-chunk, then slices to its r_loc.
- CCT's sampling produces **the full r_mu centroid array per rank** before the gather (each rank has its 1/P bc-bands at ALL r_mu positions). After gather across `('x','y')`, each rank has all bands × all r_mu positions. The y-rank's slice of the r_mu axis happens in the einsum's output shape (the carry has `n_rmu / p_y` in the col dim).

So CCT's post-gather is identity: gather completes, einsum directly into the carry whose col dim is already `n_rmu / p_y`. No explicit slice.

⚠️ **Subtle point worth re-confirming during impl**: the einsum `'kmna, knbr → karmb'` reduces over `n` (bands) and produces an output of shape `(k, ns, n_col, μ, ns)` per rank. For ZCT, `n_col` ends up as `r_loc = n_zchunk / p_y` per-rank-y-sliced. For CCT, `n_col` is `n_rmu / p_y` per-rank — but psi_l_Y_bc and psi_r_Y_bc after gather have FULL `n_rmu` in the col dim (since they're full-`r_mu` gather outputs). **The slice has to happen somewhere.**

There are two clean places:
- (A) Slice `psi_l_Y_bc` / `psi_r_Y_bc` to the y-rank's `n_rmu / p_y` slab **inside** `post_gather_fn` (mirrors ZCT exactly; only the axis differs — col axis instead of r axis).
- (B) Let the einsum reduce on the full `n_rmu`, then slice the carry's col dim at `post_op_fn` time.

(A) is faster (the einsum operates on `n_rmu / p_y` cells per rank, not the full `n_rmu`). (B) is simpler. **Recommend (A)** for symmetry with ZCT.

Updated table for §2.7 should add this:

| Hook | `c_q_from_psi_sm._local` |
|---|---|
| `post_gather_fn` | `dynamic_slice_in_dim(psi_Y_bc, y_idx * (n_rmu // p_y), n_rmu // p_y, axis=COL_AXIS)` — slice col axis to y-rank's slab |

##### 5.3.3. `n_iters` same, FFT box same, mask + pad same

`band_chunk_ranges`, `band_range_left`, `band_range_right` semantics identical to ZCT. L/R mask construction identical. Front+back-pad of `psi_l_X`/`psi_r_X` identical (same Round 7 `c796420` fix — needed because L and R band windows can be asymmetric in CCT just like ZCT). The io_callback slicer (`PsiGStore._slice_local_tile_bc`) is the same restored helper.

The ONE pre-flight check that changes: replace `n_zchunk % p_y == 0` with `n_rmu_padded % p_y == 0` (already enforced by centroid loader; trust but assert).

#### 5.4. Upstream callsite impact

The two production callers of `c_q_from_psi_sm` are in `_make_fit_centroids_kernel` (around `isdf_fitting.py:1661, 1667`; the CCT-build path that runs once per channel pre-rchunk-loop). After migration:

```python
# Before (current):
C_q = c_q_from_psi_sm(
    psi_l_rmu_X_fit, psi_l_rmu_Y_fit, psi_r_rmu_X_fit, psi_r_rmu_Y_fit,
    kgrid=kgrid, mesh_xy=mesh_xy)

# After:
psi_l_X_scaled = psi_l_rmu_X_fit / norms_l[None, None, :, None]    # pseudobands fold
psi_r_X_scaled = psi_r_rmu_X_fit / norms_r[None, None, :, None]
gamma_mu = None if is_charge else (gamma_perm, gamma_phase)
C_q = c_q_from_psi_sm(
    psi_l_X_scaled, psi_r_X_scaled, psi_G_store,
    band_chunk_ranges=band_chunk_ranges,
    band_range_left=band_range_left,
    band_range_right=band_range_right,
    fft_grid=meta.fft_grid,
    gamma_L=gamma_mu, gamma_R=gamma_mu,
    kgrid=kgrid, mesh_xy=mesh_xy)
```

(Direct mirror of Round 6's z_q callsite update at `isdf_fitting.py:1505-1520`.)

The `psi_l_rmu_Y_fit` / `psi_r_rmu_Y_fit` tensors become unused; the centroid-load path that built them (`load_centroids_band_chunked`) can be simplified or deleted. **Open question** for Round 9: does any other code path consume those Y tensors? If yes, the cleanup is staged; if no, it's a single commit.

#### 5.5. Memory profile (CCT, predicted)

For CrI3 80 Ry 4×4 mesh, charge channel:
- Carry per side: `c128[nk=36, ns=2, n_rmu/p_y=94, μ_loc=94, ns=2]` = `36·2·94·94·2·16 ≈ 20 MB`. **~200× smaller than ZCT's 3.71 GiB carry** (because `n_rmu_padded / p_y = 94` vs `n_zchunk / p_y = 18412`).
- FFT box transient: same as ZCT (~5 GB, scan-aliased).
- `psi_Y_bc` post-gather: `c128[36, 16, 2, 94]` ≈ 1.7 MB per rank. Tiny.
- Pre-migration memory: 2 × ~2 GB of `psi_l_Y_fit` / `psi_r_Y_fit` materialized upstream = 4 GB persistent residency.

**Net memory win for CCT migration**: ~4 GB persistent reduction (replaces the pre-materialized Y tensors with a 20 MB carry that only lives during the kernel call).

CCT runs once per channel (~4 calls per `fit_zeta`), so this isn't a per-rchunk hot path. But the **4 GB of persistent residency between calls** matters — it competes with ψ(G) host caches and the chol factor for the same budget.

#### 5.6. Round-9 migration plan (per commit)

Refines §7's bullet 4 ("port `c_q_from_psi_sm` to mirror ZCT"):

1. **Commit 1** — Add the new `c_q_from_psi_sm` signature alongside the old one (alias the old as `c_q_from_psi_sm_legacy` temporarily). New impl is byte-for-byte port of `z_q_from_psi_sm` (`c796420` version) with the 3 swaps from §5.2 + the post-gather slice from §5.3.2.

2. **Commit 2** — Add bit-identity test `tests/test_cq_from_psi_sm_bit_identity.py` mirroring `tests/test_zq_from_psi_sm_bit_identity.py` with all 6 sub-gates (G1.1 / .1a / .1b / .1c / .1d / .1e) — including the back-pad-required cases (short final bc, asymmetric L/R) that caught the Round 7 BLOCKER. Run on synth WFN at MoS2 3×3 scale. Pass criterion: `rtol=1e-10, atol=1e-12` vs the legacy implementation.

3. **Commit 3** — Switch the production callsite (`_make_fit_centroids_kernel`) to the new signature; remove the legacy alias. Delete `psi_l_rmu_Y_fit` / `psi_r_rmu_Y_fit` materialization upstream if no other consumers remain.

4. **Commit 4** — HLO dump on CrI3 6×6 80 Ry (analog of Round 6/7 G2 for the CCT path). Verify ~4 GB persistent-residency reduction at the planner's HWM tracker.

Total estimated diff: ~250 LOC added (mirror of `z_q_from_psi_sm`), ~150 LOC deleted (legacy `c_q_from_psi_sm._local` body + upstream Y-build path). Single-branch sequence on `agent/zeta-bc-scan-shardmap` after Round 7 G3 e2e passes.

#### 5.7. Risks specific to CCT

1. **`r_mu` indexing inside `to_rmu_inner`**: the centroid gather uses fancy indexing (`rb[:, :, :, r_mu[:,0], r_mu[:,1], r_mu[:,2]]`). Need to confirm this composes cleanly inside the scan-inside-shard_map body — same composition novelty class as Round 6's io_callback × all_gather × scan × shard_map, but now with fancy indexing instead of dynamic_slice. **Smoke-test gate** (analog of Round 6 G0): a CPU 1×1 mesh repro that does centroid gather inside the scan body. ~30 LOC.

2. **`n_rmu_padded % p_y == 0` enforcement**: confirmed by centroid loader (`load_centroids_band_chunked` enforces `n_rmu_padded ≡ ∏ p_a`). Still add a pre-flight assert in `c_q_from_psi_sm` so the failure mode is loud, not silent.

3. **CCT solver downstream sees the new C_q output shape**: `(nq, n_rmu, n_rmu)` sharded `P(None, 'x', 'y')` — identical to today's shape and sharding. **No downstream impact** if the new shape/sharding match. Verify via Agent 4's bit-identity test on the production sigma calculation.

The migration is a STRAIGHTFORWARD MIRROR — the structural risk surface is much smaller than ZCT was in Round 6, because the design has now been proven on ZCT through 2 rounds of validation (G0–G3) + a real production bug fix (back-pad). CCT inherits all of that.

---

**Agent 1 §§4-5 done. §6 (pseudobands) / §7 (migration plan) / §8 (risks) still TBD — likely Agent 2 or joint after open-questions ping below.**

### 6. Pseudobands: where the normalization slots in

_Single point: `pre_norm_fn` hook inside the body, applied to `psi_G_slab` before `_box_kernel`. Pseudobands stop being a separate code path; the existing `band_norms` pre-divide of `psi_l_X` / `psi_r_X` either stays (current behavior) or moves into the `pre_norm_fn` closure on the Y-side too (cleaner; deferred decision)._

### 7. Migration plan (per-commit sequence)

_Suggested ordering once Agent 1 reviews §§ 1–3:_
1. _Extract `_box_scatter` symmetrically with `_box_kernel`._
2. _Add `box_scan_forward` skeleton (no callers; tests only)._
3. _Port `gflat_to_rmu` onto `box_scan_forward` (smallest existing scan-inside-shard_map; verify no regression)._
4. _Port `c_q_from_psi_sm` to mirror ZCT (the big payoff; mirror of `z_q_from_psi_sm._local` with sample_kind change)._
5. _Port `to_rmu` / `to_rchunk` as thin wrappers (single-shot variants)._
6. _Add `box_scan_reverse` skeleton; port `accumulate_rchunk_to_gflat`._
7. _Delete DEAD code (`gflat_to_rchunk` standalone, `to_rmu_inner` if unused after port)._

### 8. Risks + fallbacks

_Open. Will fill after Agent 1's response surfaces SPMD-side risks (mostly: do the parametric `*_kind` strings + composable `*_fn` callables actually fold into a clean shard_map body, or does JAX trip on the conditional ladder? Need to check on a small reproducer)._

---

## Open questions for Agent 1

1. **Composable callable hooks inside shard_map manual mode**: are there JAX-level restrictions on closure-captured pure-jax callables called inside a `shard_map` body? My read is "no, as long as the callable doesn't reference jax.Arrays with mismatched mesh sharding" — but worth confirming. (We hit a related issue in Round 6 with closure-captured `jnp.asarray` tables; the fix was to lift them inside the body. Does that pattern apply here too?)

2. **`band_gather_axes` parametric**: a single `lax.all_gather(axis_name=...)` call where the `axis_name` is a parametric tuple. JAX's all_gather takes a constant axis_name; closure-static is fine. But what about the case where some sites want `None` (no gather) — does an `if band_gather_axes is not None:` branch fold cleanly in shard_map manual mode, or do we need a separate body fn for each variant?

3. **`post_scan_fn` as first-class hook vs separate per-site wrapper**: ZCT/CCT's IFFT-k + γ̃ + FFT-k + transpose tail is substantial (~30 LOC of body, multiple FFT ops, gamma_double_contract). It's tempting to keep it as a free-floating helper called by each wrapper AFTER `box_scan_forward` returns the final carry. But that splits the shard_map (two shard_map invocations instead of one). Your call on SPMD correctness — is splitting acceptable, or must the tail stay inside the same shard_map body?

4. **Cost of the dispatch ladder in the universal body**: 4 `sample_kind` × 4 `apply_phase_kind` × 2 `band_gather` options = small constant fan-out, all closed at trace time. Should be invisible to XLA. But if you spot a pathological lowering, flag it.

Will iterate on this thread; will publish §§ 4–8 once we converge on §§ 1–3.

— Agent 2, 2026-05-14
