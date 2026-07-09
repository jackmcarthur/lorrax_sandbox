# src/common/wfn_transforms.py — deep-read notes (2026-07-01)

LOC: 1357. No file I/O, no LorraxConfig/cohsex.in consumption. Pure JAX transform
library: G-flat ψ (WfnLoader layout `(n_k, nb_padded, nspinor, ngkmax)` c128) →
FFT-box / r-space / centroid samples / flat-r slabs, plus the inverse r-chunk →
G-flat accumulator used by the ζ writer. All transforms preserve band-axis
sharding `P(None, ('x','y'), None, None)`; no cross-rank communication.

Category: **transform kernel library (ψ↔r↔G layout transforms + Bloch phase), memory-layout/sharding machinery**.

## Module-level infrastructure

| Symbol | Lines | Role |
|---|---|---|
| `_KERNEL_CACHE` / `_cached_jit(name, key, build)` | 69–79 | Single signature-keyed cache for every jit factory in the module. Key = `(kernel_name, *signature_tuple)`. Unbounded dict (bounded in practice by shape/channel variety). |
| `_GINDEX_DEV_CACHE` / `_cached_gindex_dev(g_arr)` | 91–128 | Dedups the device-resident `(nk, nx, ny, nz) int32` g_index buffer by **content hash** (`hash(g_arr.tobytes())` + shape). Fixes a +1-replicated-buffer-per-channel leak (agent_h §3 Finding 3). jax.Array int32 input passes through unchanged (Round-6 canonical-accessor path via `WfnLoader.box_index_dev`). |
| `_resolve_gindex_dev(g_index)` | 131–179 | Returns `(device buffer, cache_id)`. jax.Array → cache_id `('jax_id', id(g_index), shape)`; numpy → `('np_hash', hash(bytes), shape)`. cache_id feeds `_cached_jit` keys for closure-bake sites. |
| `_box_kernel(psi, g_index, *, ngkmax)` | 194–222 | THE shared gather kernel: G-flat → FFT box. Appends a zero slot on the G-axis (sentinel index `ngkmax` gathers 0), builds flat index `k*(ngkmax+1) + g_index`, one `jnp.take(..., mode='clip')` fills the whole box. `mode='clip'` deliberately skips jnp's OOB `_where` mask (killed 8 retrace cache misses in MoS2 3×3 profile). Pure jax, no shard_map; sharding propagates. Output `(n_k, nb, ns, nx, ny, nz)`. Algorithm comment says it "matches `common/gvec_fft_box.make_fft_box_kernel` but on WfnLoader's c128 layout" — see redundancy notes. |
| `_spec_of(psi)` | 231–246 | PartitionSpec of psi normalized to `psi.ndim` length (JAX trims trailing None). |
| `_output_sharding(psi, mesh, n_extra_axes)` | 249–256 | Output NamedSharding: preserve band spec (axis 1), replicate everything else. |
| `_maybe_constrain` | 259–260 | thin `with_sharding_constraint` wrapper (name is a lie — it always constrains). |
| `_local_box_fft(psi, mesh, *, kind, norm)` | 274–288 | The ONE FFT primitive for the public `to_*` transforms: wraps `common.fft_helpers.make_sharded_{i}fftn_3d` (shard_map local cuFFT). Rationale: plain `jnp.fft.ifftn` on sharded tensors let XLA insert an all-gather + global FFT → the original 121 GB OOM in `to_rmu` at CrI3 6×6×1 80 Ry. |
| `_sharding_key(psi)` | 295–311 | `(id(mesh), normalized spec)` jit-cache key component. Normalization fix saved 11 spurious `_kernel` recompiles (~7 s) at MoS2 3×3 bispinor. |

## Public transforms (forward direction, ψ)

### `to_box(psi, g_index, fft_grid, *, mesh)` — lines 325–353
G-flat → FFT box `(n_k, nb, ns, nx, ny, nz)`. jit-cached closure = `_box_kernel` + sharding constraint.
Callers (grep src/tests/tools/scripts):
- `src/centroid/charge_density.py:99` (charge density from |ψ|², IBZ)
- `src/common/load_wfns.py:40, 166` (legacy single-k / boxed loads)
- `tests/test_wfn_transforms.py` (multiple)

### `to_rbox(psi, g_index, fft_grid, *, mesh, norm, kvecs_frac)` — lines 356–401
to_box → local IFFT → optional `apply_bloch_phase` (`exp(+2πi k·r)`); materializes the full r-space box.
Callers: **tests only** (`tests/test_wfn_transforms.py:144,167,186`) plus a docstring mention in `file_io/wfn_loader.py:44`. No production caller found (grepped `\bto_rbox\b` across src/tests/tools/scripts). Dead-in-production suspect (kept as reference/API completeness — the module docstring itself says "prefer to_rmu/to_rchunk").

### `to_rmu(psi, g_index, fft_grid, r_mu, *, mesh, norm, kvecs_frac)` — lines 404–452
ψ at centroid FFT-grid indices: IFFT box then gather `rb[:, :, :, r_mu[:,0], r_mu[:,1], r_mu[:,2]]` → `(n_k, nb, ns, n_rmu)`. Physics: ψ_nk(r_μ) = Σ_G c_nk(G) e^{iG·r_μ} (·e^{+2πik·r_μ} when kvecs given).
Callers:
- `src/centroid/pivoted_cholesky.py:118` (centroid selection, norm="ortho" pivoted-Cholesky path)
- tests. NOTE: still materializes the full unsharded FFT box per band-chunk (Peak A); the production centroid load moved to `gflat_to_rmu` (see `load_wfns.py:372–485` commentary; `gw_init.py:1212` caps band_chunk_size to avoid to_rmu OOM).

### `to_rchunk_inner(psi, g_index, fft_grid, r0, r_len, *, norm, kvecs_frac)` — lines 455–528
Per-rank-local body of to_rchunk: `_box_kernel` → `jnp.fft.ifftn` → flat reshape (`r_flat = rx·ny·nz + ry·nz + rz`) → `dynamic_slice_in_dim(r0, r_len)` → optional `apply_bloch_phase_on_slice`. No shard_map — callable inside another shard_map/scan body. Assumes exactly 3 leading axes before ngkmax.
Callers:
- `src/common/isdf_fitting.py:743` (**production** ζ-fit pair pipeline, inside its scan-inside-shard_map body — imported at isdf_fitting.py:140)
- `src/gw/aot_memory_model/kernels/fit_one_rchunk.py:33,37` (memory-model commentary references)
- tests (`test_wfn_transforms.py`, `test_zq_from_psi_sm_bit_identity.py`).
STALE DOCSTRING: lines 473–479 say "Not yet wired into the production fit kernel — the consumer refactor ... is deferred to a follow-up session"; isdf_fitting.py:743 shows it IS wired in.

### `to_rchunk(psi, g_index, fft_grid, r0, r_len, *, mesh, norm, kvecs_frac)` — lines 531–620
shard_map wrapper around `to_rchunk_inner`; whole gather→IFFT→slice(→phase) pipeline in one shard_map region (verified to remove ~506 MiB all-gathers from HLO at MoS2 3×3 / 4×A100). `r0` may be Python int (bounds-checked) or traced scalar. in_specs g_index = `P(None,None,None,None)`, `check_rep=False`.
Callers:
- `src/common/load_wfns.py:256, 274` (band-chunked r-slab generator feeding ζ-fit legacy path; norm="ortho")
- tests (`test_wfn_transforms.py`, `test_rchunk_gflat_pair.py:124,167,242`, `test_zq_from_psi_sm_bit_identity.py`).

### `to_rmu_inner(psi, g_index, fft_grid, r_mu, *, norm, kvecs_frac)` — lines 640–698
Per-rank-local body of to_rmu (mirror of to_rchunk_inner for the centroid-sample direction). Full-box `apply_bloch_phase` before gather.
Callers: **tests only** (`tests/test_wfn_transforms.py:291,319`). Grepped `\bto_rmu_inner\b` across src/tests/tools/scripts — zero production callers. Notably `gflat_to_rmu` does NOT call it; it re-implements box+IFFT+gather inline in its scan body (with post-gather phase instead of full-box phase). Dead-in-production / scaffolding-that-never-got-consumed suspect.

## Fused scan-inside-shard_map pair (the ψ↔ζ "round-trip primitive")

### `gflat_to_rmu(psi_G, g_index, r_mu, *, mesh, fft_grid, kvecs_frac, norm, chunk_size)` — lines 711–970
Production centroid extraction: ψ(G-flat) → ψ(r_μ) fused over all (k, n), one shard_map over `('x','y')`, `lax.scan` over chunks of the flat `N = nk·nb_local` row axis. Per iter: slice cs rows → `k_row = clip((i0+arange(cs))//nb_local, 0, nk-1)` → `_box_kernel(sub4=(cs,1,ns,ngkmax), g_index_[k_row])` → `jnp.fft.ifftn` (per-rank-local) → centroid gather `(cs, ns, n_rmu)` → optional per-row Bloch phase applied **post-gather on sampled cells only** (`samples * (phx_q * phy_q * phz_q)[:, None, :]`, scratch drops from cs·n_rtot to cs·n_rmu; algebraically identical since phase is pointwise) → `dynamic_update_slice_in_dim` into out_flat. Zero-pad N→⌈N/cs⌉·cs; padding rows clip to k=nk−1 but carry zero data. Requires `nb_total % mesh.size == 0`. XLA aliases the per-iter FFT box across scan iters → slot count 1; box is per-rank-local (closes the unsharded-FFT-box violation of legacy to_rmu, "Defect 3").
Phase tables `phx/phy/phz = exp(+2πi k_a · (arange(n)/n))` per axis, `(nk, n_*)`, pre-gathered at r_μ coords, **baked into the closure as constants**; cache key includes `r_mu_id = hash(r_mu.tobytes())`, `kvecs_id = hash(kvecs.tobytes())`, `g_index_id` from `_resolve_gindex_dev`. g_index threaded through shard_map in_specs (NOT closure capture) so the WfnLoader canonical device buffer is shared (Round-6; pre-fix the closure captured a divorced SingleDeviceSharding copy).
Callers:
- `src/common/load_wfns.py:485` (**production** centroid load `load_wfns.load_centroids` path; timing section "load_centroids.gflat_to_rmu")
- `src/gw/gflat_memory_model.py:304,312,343` (planner models its per-iter box)
- tests (`test_wfn_transforms.py:374,404,431,436` — matches bc-loop+to_rmu reference, chunked vs one-shot).
`chunk_size` bound: `cs · ns · n_rtot · 16 B` per-iter FFT box.

### `accumulate_rchunk_to_gflat(rchunk, gflat_acc, *, mesh, fft_grid, r0, sphere_idx, qvec_frac, norm, chunk_size)` — lines 1006–1236
Inverse-direction mirror (the ζ writer, Phase C of PLAN_zeta_g_flat_migration.md). Math (lines 997–1004, verbatim):
```
ζ_G[q, μ, G_sph] = Σ_r  exp(-2πi q·r) ζ_r[q, μ, r]  e^{-2πi G·r}
                = FFT_{r→G}( exp(-2πi q·r) ζ_r[q, μ, r] )[G_sph]
ζ_G += FFT_{r→G}( phase · pad_to_full(rchunk_slab) )[G_sph]     (linearity over r-chunks)
```
Shapes: `rchunk (n_q, n_rmu_padded, r_len)` / `gflat_acc (n_q, n_rmu_padded, ngkmax)`, both `P(None, ('x','y'), None)`; `n_rmu_padded % mesh.size == 0`. Scan over flat `N = n_q·n_mu_local` rows: per iter slab → **phase-on-slice** (`exp(-2πi q·r)` applied to the r_len slab cells only, per-q separable tables `phx/phy/phz` with sign −, gathered by `q_row` and slab-decoded `rx_slab/ry_slab/rz_slab`; full box never phase-multiplied) → zero box + `dynamic_update_slice_in_dim` at r0 → `jnp.fft.fftn` → `jnp.take_along_axis(G, sphere_c[q_row], mode='promise_in_bounds')` → add into acc slice. `gflat_acc` **donated** (`donate_argnums=(1,)`) → in-place accumulation. sphere_idx `(n_q, ngkmax)` int32 flat-FFT indices, per-q, pad slots use a sentinel index whose coeffs the caller zeroes post-loop; device buffer deduped via `_cached_gindex_dev(sphere_arr)`. Cache key includes `sphere_id`/`qvec_id` content hashes.
Callers:
- `src/common/isdf_fitting.py:2613` (**production** ζ G-flat writer loop; donated acc built at :2497)
- `src/gw/gflat_memory_model.py:40,68,435` (Peak D model); `src/gw/gw_config.py:233` (`gflat_chunk_size` flag bounds this kernel's cs — the only config flag reaching this module, passed by the caller)
- `src/file_io/isdf_header.py:127` (writer-path sentinel doc)
- tests (`test_rchunk_gflat_pair.py:170,204,209,245`, `test_per_q_sphere.py:154`).

## Bloch-phase helpers (declared single source of truth)

### `apply_bloch_phase(box, kvecs_frac, fft_grid, *, sign=1)` — lines 1259–1296
`box × exp(sign·2πi k·r)` as three separable 1D broadcast multiplies (`px (n_k,nx)`, `py`, `pz`); scratch `n_k·(nx+ny+nz)` c128 instead of the 4D product. sign=+1 ψ post-IFFT (`ψ_nk(r) = e^{+2πik·r} u_nk(r)`); sign=−1 ζ pre-FFT (`z_q,μ(r) = e^{-2πiq·r} ζ_q,μ(r)`). Header comment (1250–1257): "this is the ONLY place in LORRAX where the Bloch-phase formula lives."
Callers: `src/gw/v_q_tile.py:682` (ζ disk→G, sign=−1), `src/file_io/zeta_reader.py:409` (`_do_disk_to_G`, sign=−1), internal (to_rbox/to_rmu/to_rmu_inner bodies), tests; convention cross-refs in `gw/compute_vcoul.py:1012`, `gw/v_q_bispinor.py:344`, `common/isdf_fitting.py:2209`.

### `apply_bloch_phase_on_slice(slab, kvecs_frac, fft_grid, r0, r_len, *, sign=1)` — lines 1299–1357
Same phase restricted to a flat-r slab `[r0, r0+r_len)`: decode `flat = r0 + arange(r_len)` → `(rx, ry, rz)` → gather per-axis 1D factors → `(n_k, r_len)` phase. Cost n_k·r_len instead of n_k·n_rtot.
Callers: internal (`to_rchunk_inner:526`), tests (`test_rchunk_gflat_pair.py:91`, `test_wfn_transforms.py:226`).

## I/O
None. Pure in-memory transform module. Composes with `file_io.wfn_loader.WfnLoader` (psi + `box_index`/`box_index_dev` producer) and `common.psi_G_store` (host g_flat cache) upstream; `common.isdf_fitting` (ζ fit), `common.load_wfns` (chunk generators), `centroid/*`, `gw/v_q_tile`, `file_io/zeta_reader` downstream.

## Flags
No direct LorraxConfig/cohsex.in reads. Indirect: `gflat_chunk_size` (gw_config.py:233) → `accumulate_rchunk_to_gflat(chunk_size=...)`; `band_chunk_size` (gw_init.py:1212) exists partly to cap legacy `to_rmu` box memory.

## Dead suspects
1. **`to_rbox`** — grep `\bto_rbox\b` over src/tests/tools/scripts: only `tests/test_wfn_transforms.py` + docstring mention in `file_io/wfn_loader.py:44`. Zero production callers. (Test-scaffolding role: other transforms are validated against it.)
2. **`to_rmu_inner`** — grep `\bto_rmu_inner\b`: only `tests/test_wfn_transforms.py:291,319`. Its intended consumer (`gflat_to_rmu`) inlines the equivalent logic in its own scan body instead of calling it. Scaffolding never consumed.

## Redundancy suspects
1. **`_box_kernel` vs `common/gvec_fft_box.make_fft_box_kernel`** — acknowledged near-duplicate (line 186 comment: "matches ... but on WfnLoader's c128 layout"). Bonus: grep shows `make_fft_box_kernel` itself has no callers outside gvec_fft_box.py + its MD doc (only `build_g_index_for_fft_box` is used, by `wfn_loader.py:428`) — the older kernel looks dead, classic parallel old/new path.
2. **`to_rmu_inner` vs `gflat_to_rmu` scan body** — the scan body (lines 921–965) re-implements `_box_kernel + ifftn + centroid gather` rather than calling `to_rmu_inner`, differing only in phase placement (post-gather vs full-box). Two parallel implementations of the same math. Same relationship holds between `to_rmu` (legacy full-box path, still live via pivoted_cholesky) and `gflat_to_rmu` (fused path) — legacy path retained.
3. **Bloch-phase 1D-factor formula lives in 4 places despite the "single source of truth" claim (line 1251)**: `apply_bloch_phase` (1284–1286), `apply_bloch_phase_on_slice` (1333–1335), `gflat_to_rmu.build._ph` lambda (890–892, sign +), `accumulate_rchunk_to_gflat.build._ph` lambda (1164–1166, sign −). All four are `exp(±2πi k_a · arange(n)/n)` copies.
4. **`load_wfns.py` chunk generators vs the fused pair** — load_wfns.py:256/274 still drives `to_rchunk` per band-chunk (Defect-1-style outer bc loop) while isdf_fitting's production path uses `to_rchunk_inner` inside its own scan; two live generations of the r-slab pipeline.

## Weird code
1. **Nonexistent `gflat_to_rchunk` referenced 5×** (lines 627, 638, 731, 868, 952: "mirroring :func:`gflat_to_rchunk`", "same lesson as gflat_to_rchunk", "cf. gflat_to_rchunk's phase-on-slice pattern"). `grep -rn "def gflat_to_rchunk" src tests tools scripts` → nothing. Hypothesis: the planned r-slab twin was renamed/absorbed into `isdf_fitting`'s scan (`to_rchunk_inner` consumer) and docstrings were never updated. Also `gflat_memory_model.py:304` cites "wfn_transforms.py:611-851" for `gflat_to_rmu._kernel`, which is now at ~917–967 — line-number rot.
2. **Python `hash(bytes)` as content identity** for g_index/r_mu/kvecs/qvec/sphere cache keys (lines 121, 869, 873, 1144, 1145). Collision → silently reusing a compiled closure with stale baked-in phase tables / centroid coords. The code itself flags this for qvec (lines 1136–1142: "a latent correctness hazard for any caller that varies qvec_frac at fixed shape"). Also `hash()` is per-process salted for bytes (PYTHONHASHSEED) — fine intra-process, but not a stable content hash.
3. **`id(g_index)` as cache_id for jax.Array inputs** (line 175, `_resolve_gindex_dev`). If the canonical buffer is freed and a different g_index later reuses the same id() with the same shape, `_cached_jit` returns the wrong closure. Comment assumes "stable across the process lifetime for the canonical buffer" — true only for the WfnLoader-cached buffer, unenforced for arbitrary callers.
4. **`mode='clip'` gather** (line 221) and **`mode='promise_in_bounds'`** (line 1221): both intentionally disable OOB masking, relying on construction invariants (zero-slot pad; caller-zeroed sentinel sphere slots). Correct today, fragile under refactor — a g_index built without the sentinel convention silently reads wrong cells instead of erroring.
5. **`check_rep=False` on every shard_map** (587, 601, 920, 1179) — standard for this codebase but means replication errors surface as silent wrong numbers.
6. **Sign conventions**: ψ phase +2πi (line 890 "Forward Bloch phase: sign = +1"), ζ/q phase −2πi (line 1165); `apply_bloch_phase(sign=±1)` parameterizes it. Padding rows in both scan kernels clip to the LAST k/q index (`jnp.clip(..., 0, nk-1)`, lines 934–935, 1205–1206) and rely on zero-padded data for correctness — index gymnastics that only works because pad rows are exactly zero.
7. **`to_rchunk_inner` "not yet wired into production" docstring** (473–479) is stale — `isdf_fitting.py:743` calls it in the production fit kernel.
8. **norm default asymmetry**: `to_rmu`/`gflat_to_rmu` default `norm="backward"` "to match the legacy to_rmu default" while every production centroid caller passes `"ortho"` (load_wfns.py:256/274 ortho, pivoted_cholesky ortho) — a default nobody in production uses.
9. **`_maybe_constrain`** (259) unconditionally constrains — misnamed vestige.
