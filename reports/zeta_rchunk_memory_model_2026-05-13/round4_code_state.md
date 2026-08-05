# Round 4 — Agent 2: state of the code, post-Path-D

Read-only audit of `sources/lorrax_A` `agent/zeta-r-chunk-fixes-2026-05-13` @ `ff5873c` vs `sources/lorrax_B` `agent/zeta-bc-scan-shardmap` @ `5cadd4b`. Three deliverables.

---

## 1. lorrax_B commits — per-commit summary

`main` (`488e870`) → 5 commits → `5cadd4b`. All on the `agent/zeta-bc-scan-shardmap` branch.

| # | SHA | Subject | Files | Substantive change | New API | Removed API | Cohsex.in field changes |
|---|-----|---------|-------|--------------------|---------|-------------|--------------------------|
| 1 | `cdd0fba` | Path D scaffolding: `to_rchunk_inner` + `_slice_local_tile_bc` | `wfn_transforms.py` (+76), `psi_G_store.py` (+60), `tests/test_wfn_transforms.py` (+72) | Two read-only helpers landed ahead of the structural rewrite. `to_rchunk_inner` = pure-jax body of `to_rchunk` (no shard_map wrapper, callable inside another shard_map / scan body). `_slice_local_tile_bc` = host-tile slicer for a *traced* `bc_idx`, returns `(nk, _bpd_max, ns, ngkmax)` padded c128 — designed for the eventually-not-taken "scan-inside-shard_map over bcs" approach. Three tests for `to_rchunk_inner`. | `wfn_transforms.to_rchunk_inner`; `PsiGStore._slice_local_tile_bc`, `PsiGStore._bpd_max`, `PsiGStore._bpd_per_bc` | — | none |
| 2 | `d7eaf1c` | `wfn_transforms`: `gflat_to_rchunk` forward helper (Path D §4 fix) | `wfn_transforms.py` (+259), `tests/test_wfn_transforms.py` (+102) | New shard_map+scan helper that pulls ψ(G-flat)→ψ(rchunk) for the full (nk, nb) extent in one jit. Structural twin of `accumulate_rchunk_to_gflat`. Per-iter FFT box `c128[cs, ns, nx, ny, nz]` aliased across scan iters. Cache key includes `kvecs_id` content hash (lesson learned the hard way — see commit 3d0636c). Three CPU bit-identity tests vs the bc-loop+concat reference. | `wfn_transforms.gflat_to_rchunk` (`(nk, nb, ns, ngkmax)` → `(nk, nb, ns, r_len)`) | — | none |
| 3 | `3d0636c` | `wfn_transforms`: content-hash `qvec_frac` in `accumulate_rchunk_to_gflat` cache | `wfn_transforms.py` (+10) | Mirror of the kvecs content-hash fix from `gflat_to_rchunk`. Two callers with same `qvec_frac.shape` but different content silently hit the same compiled fn whose closure holds stale `phx/phy/phz`. Latent — masked in production because the q-grid is constant per run. Pattern matches existing `sphere_id` content-hash on `sphere_idx`. | — | — | none |
| 4 | `3606138` | `wfn_transforms` + `load_wfns`: `gflat_to_rmu` + `to_rmu_inner` (Defect 3 mirror, Agent 4) | `wfn_transforms.py` (+326), `load_wfns.py` (+99/-153 net), `tests/test_wfn_transforms.py` (+178) | Mirror of `d7eaf1c` for the centroid-sample direction. Added `to_rmu_inner` (pure-jax body of `to_rmu`) and `gflat_to_rmu` (shard_map+scan twin of `gflat_to_rchunk`, centroid-gather instead of r-slab). Phase applied post-gather (algebraically identical, `cs · n_rmu` scratch instead of `cs · n_rtot`). `load_centroids_band_chunked` rewritten to use `gflat_to_rmu` (single shard_map+scan, no driver bc / k loops). | `wfn_transforms.to_rmu_inner`, `wfn_transforms.gflat_to_rmu` | — (additive on `wfn_transforms`; `load_centroids_band_chunked` rewritten in place) | none |
| 5 | `5cadd4b` | `isdf_fitting` + `psi_G_store`: integrate `gflat_to_rchunk`; drop bc-loop + `psig_k_chunk` plumbing | `isdf_fitting.py` (±52), `psi_G_store.py` (±333), `aot_memory_model/kernels/fit_one_rchunk.py` (±116), `gw_config.py` (±17), `gw_init.py` (±27), `tests/test_psi_g_store.py` (+107) | Three coupled changes: (i) `PsiGStore.psi_G_device_full` lazy property + `g_index` / `kvecs_frac` properties (per-bc io_callback + `jnp.concatenate(axis=1)` — concat is load-bearing for canonical band sharding); (ii) `_make_fit_one_rchunk_kernel._kernel` body rewrite (drop bc-loop + concat, single `gflat_to_rchunk` call, `gflat_to_rchunk_chunk_size` knob auto-picked from `cfg.memory.per_device_gb`); (iii) dead-plumbing removal (deleted `fetch_psi_rchunk`, `_slice_local_tile_bc`, `_bc_index`, `_k_chunk_size` field, `psig_k_chunk_size` cohsex knob across config + init + isdf_fitting + AOT stub). AOT stub `_AotStubPsiGStore` rewritten to expose `psi_G_device_full` / `g_index` / `kvecs_frac` (was `fetch_psi_G`). | `PsiGStore.psi_G_device_full`, `PsiGStore.g_index`, `PsiGStore.kvecs_frac` (all properties); `MemoryConfig.gflat_to_rchunk_chunk_size`; cohsex.in `gflat_to_rchunk_chunk_size` knob; `_make_fit_one_rchunk_kernel`/`fit_one_rchunk`/`fit_zeta_to_h5` `gflat_to_rchunk_chunk_size` kwarg | `PsiGStore.fetch_psi_rchunk`, `PsiGStore._slice_local_tile_bc` (added in `cdd0fba`!), `PsiGStore._bc_index`, `PsiGStore._k_chunk_size`, `PsiGStore._bpd_max`, `PsiGStore._bpd_per_bc`; `k_chunk_size` arg from `PsiGStore` / `HostPsiGStore` / `RereadPsiGStore` / `build_psi_G_store`; `MemoryConfig.psig_k_chunk_size`; `fit_zeta_to_h5(psig_k_chunk_size=...)` arg; `_AotStubPsiGStore.fetch_psi_G` | **drop** `psig_k_chunk_size`; **add** `gflat_to_rchunk_chunk_size` |

Net surface delta over 5 commits:
- New public helpers in `common.wfn_transforms`: `to_rchunk_inner`, `gflat_to_rchunk`, `to_rmu_inner`, `gflat_to_rmu`.
- New `PsiGStore` properties: `psi_G_device_full`, `g_index`, `kvecs_frac`.
- Net deletion in `psi_G_store`: `fetch_psi_rchunk`, `_slice_local_tile_bc`, `_bc_index`, `_k_chunk_size`/`_bpd_max`/`_bpd_per_bc` fields.
- Cohsex.in: net `−psig_k_chunk_size` / `+gflat_to_rchunk_chunk_size`. Identical knob count; replacement, not growth.
- Note `cdd0fba` added `_slice_local_tile_bc` + `_bpd_max` + `_bpd_per_bc` and `5cadd4b` removed them — that scaffolding ended up unused because the integration took the "single helper called from `_kernel`" path rather than the "scan-inside-shard_map over bcs" path. Surface net: zero new dead helpers.

---

## 2. lorrax_A vs lorrax_B reconciliation

`lorrax_A` head `ff5873c` is one commit ahead of `488e870`. Its single commit (`gflat_memory_model: account for unsharded band-FFT pool in Peak C + centroid persist bug`) introduces four pieces. None survive to `lorrax_B`'s `5cadd4b` — `lorrax_B`'s `gflat_memory_model.py` is bit-identical to `488e870` (`diff` confirmed empty).

| Piece (`lorrax_A` `ff5873c`) | Mechanism it models | Still relevant on lorrax_B? | Action |
|------------------------------|---------------------|------------------------------|--------|
| `_bytes_centroids_LR(nk, ns, mu, nb_total, p_x, p_y)` helper + use in `_peak_C_fit_one_rchunk` and `_peak_D_accumulate` | The two persistent centroid copies (`psi_rmu_Y` on `'y'`, `psi_rmuT_X` on `'x'`) live on **disjoint mesh axes**; the old `2 * _bytes_c128(... shard=p_xy)` over-credited sharding by ~√P. Pure planner accounting bug, **independent of the bc-loop structural fix**. | YES — independent bug. | **Cherry-pick to lorrax_B.** Trivial conflict-free port (`gflat_memory_model.py` only). |
| `band_fft_unsharded` term in `_peak_C_fit_one_rchunk` (the `n_bc · band_fft_slots` Python-unrolled FFT-box pool) | Models the 58-slot pile-up from the Python-unrolled bc-loop in `fetch_psi_rchunk`. | NO. After `5cadd4b` the bc-loop is gone, replaced by `gflat_to_rchunk`'s scan-internal aliased FFT box. The slot pile-up no longer exists. The morning's CrI3 HLO dump (post-Path-D) confirmed this on production scale: 200 GiB → ~48 GiB total. | **Drop.** Carrying it codifies a mechanism that no longer exists; over-estimating the budget shrinks the planner's chunk picks unnecessarily. |
| `band_fft_pool` feasibility `raise ValueError` in `plan_gflat_chunks` | Refuses configs where the unsharded FFT pool alone exceeds budget. Was the user-facing safety net for the now-eliminated mechanism. | NO. Same reason as above — no pool to refuse. | **Drop.** |
| `psig_k_chunk_size` threading: cohsex.in knob → `MemoryConfig.psig_k_chunk_size` → `plan_gflat_chunks(psig_k_chunk=…)` → planner term + `psi_G_store.PsiGStore(k_chunk_size=…)` | Linear knob to reduce the unsharded FFT pool by capping the per-fetch k batch. Was the only knob that touched the offending term. | NO. `5cadd4b` already deleted the cohsex knob, the `MemoryConfig` field, the `fit_zeta_to_h5` kwarg, the `build_psi_G_store(k_chunk_size=…)` parameter, and the `_k_chunk_size` field plus the inner k-chunk Python loop in `fetch_psi_rchunk` (the method itself was deleted). | **Already done on lorrax_B.** No-op vs `lorrax_A` (the lorrax_A planner *consumes* the knob, the lorrax_B kernel doesn't *have* one). |

**Net for lorrax_B:** cherry-pick `_bytes_centroids_LR`. Skip everything else.

The `_bytes_centroids_LR` cherry-pick is genuinely correctness-preserving — Peak C and Peak D will both get the more accurate centroid estimate, which lets the planner pick larger chunk sizes (= fewer r-chunks, faster runs). At CrI3 6×6 80 Ry on 4×4 mesh the morning's planner showed `centroids_persist` shrank by ~60% vs the over-counted formula.

`lorrax_A` direction: the morning's `band_fft_*` accommodations should be **removed** there once Path D has been validated end-to-end at CrI3 (i.e., once the orchestrator merges the work back). Until then they're a planner-only safety net that errs on the side of refusing infeasible configs — fine for `lorrax_A` to keep until Path D lands on `main`.

---

## 3. Cohsex.in surface diff (488e870 → 5cadd4b)

Memory-related cohsex.in fields: identical count, one rename in spirit.

| Knob | 488e870 | 5cadd4b | Notes |
|------|---------|---------|-------|
| `psig_k_chunk_size` | present (default `0`); fed `PsiGStore._k_chunk_size`; capped per-fetch FFT box | **removed** | Mechanism it controlled (Python-unrolled k-chunk loop in the deleted `fetch_psi_rchunk`) no longer exists. |
| `gflat_chunk_size` | present (default `0` ⇒ one-shot); flat-axis chunker for `accumulate_rchunk_to_gflat` (reverse: ψ(rchunk)→ψ(G)) | unchanged | Reverse helper unchanged. |
| `gflat_to_rchunk_chunk_size` | not present | **new** (default `0` ⇒ auto-pick from `cfg.memory.per_device_gb`) | Forward `ψ(G)→ψ(rchunk)` flat-axis chunker. Structural twin of `gflat_chunk_size`. Cohsex.in `> 0` overrides the auto-pick. |
| `vq_g_chunk_size` | unchanged | unchanged | V_q kernel knob. |
| `memory_per_device_gb`, `band_chunk_size`, `r_chunk_size`, `chunk_size`, etc. | unchanged | unchanged | High-level chunker / budget knobs. |

**User-facing change:** users with `psig_k_chunk_size = N` in their cohsex.in will get a "no such field" warning from the parser (or be silently dropped, depending on `_g`'s strictness — should verify). Conservative path: leave the parser tolerant of unknown keys *and* publish the rename in `CHANGELOG.md`. Aggressive path: emit a `DeprecationWarning` mentioning `gflat_to_rchunk_chunk_size` as the (different but semantically related) replacement.

**Documentation gap:** `docs/docs_gwjax/COHSEX_INPUT.md` (the cohsex.in reference per `CLAUDE.md`'s pointer table) — not updated this round. Both the removal of `psig_k_chunk_size` and the addition of `gflat_to_rchunk_chunk_size` should land there. Same applies to any cohsex.in templates under `templates/`.

The auto-pick logic in `gw_init.py`:

```python
_p_prod = int(jax.device_count())
_nb_local = int(meta.b_id_4) // _p_prod
_N_rows = int(meta.nk_tot) * _nb_local
_box_bytes_per_row = int(meta.nspinor) * int(meta.n_rtot) * 16
_budget_bytes = int(0.5 * float(cfg.memory.per_device_gb) * (1 << 30))
_cs_auto = max(1, _budget_bytes // max(1, _box_bytes_per_row))
chunks['gflat_to_rchunk_chunk_size'] = (
    0 if _cs_auto >= _N_rows else int(_cs_auto))
```

This is reported in the chunk plan header (`G→r cs:` line in the log) when nonzero. Worth documenting in `COHSEX_INPUT.md` so users know the heuristic.

---

## 4. Recommended branch-management action

**Recommendation: merge `lorrax_B` `agent/zeta-bc-scan-shardmap` to `main` as a fast-forward (or non-fast-forward merge commit), then cherry-pick `_bytes_centroids_LR` from `lorrax_A` `ff5873c` into `main` as a follow-up commit.**

Rationale:
- The 5 lorrax_B commits are **structured for review**: scaffolding (`cdd0fba`), helper (`d7eaf1c`), latent-bug fix (`3d0636c`), Defect-3 mirror (`3606138`), integration (`5cadd4b`). Each is a self-contained, testable unit with the right tests in the right commit. Squashing would erase that legibility for the next agent debugging the boundary remat warnings or the CrI3 HLO regressions.
- Force-push is wrong here: `main` hasn't moved since `488e870` (the branch already includes everything on `main`), and there's no rewriting upstream history needed.
- `lorrax_A`'s `ff5873c` includes 1 keep + 3 drops; cherry-picking the whole commit pulls in the `band_fft_pool` accommodation we don't want. Either:
  - Cherry-pick `ff5873c` and immediately revert the `band_fft_*` and `psig_k_chunk` parts in a follow-up commit (leaves the keep as the documented intent), or
  - Hand-port `_bytes_centroids_LR` + the call-site updates in `_peak_C_fit_one_rchunk` / `_peak_D_accumulate` / `plan_gflat_chunks` directly. Cleaner; shorter diff. **Preferred.**
- Once `main` has the integrated state, retire `lorrax_A`'s `agent/zeta-r-chunk-fixes-2026-05-13` branch and `lorrax_B`'s `agent/zeta-bc-scan-shardmap` (after cherry-pick verification).

Open prerequisites before the merge:
1. CrI3 6×6 80 Ry end-to-end must complete cleanly on `5cadd4b` (the run is "still progressing" per the round 4 status snapshot — wait for the chunk loop to finish + ζ to be written + Σ to converge).
2. Address the new "involuntary full rematerialization" warning at the `gflat_to_rchunk` → `z_q_from_psi_sm` reshard boundary (per round 4 status snapshot §"The new defect"). Whether to land that fix as a 6th commit on `agent/zeta-bc-scan-shardmap` before merging or as a follow-up on `main` depends on whether it changes any wire formats — if it's pure axis-spec tweaks in `z_q_from_psi_sm._local`, fine to follow up on `main`.
3. Decide on the `psig_k_chunk_size` deprecation strategy (warn vs. silent-ignore) so existing user cohsex.in files don't break silently.

---

Agent 2 round 4 done.
