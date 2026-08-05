# Agent I — V_q HBM characterization for the memory-model refit

**Branch:** `agent/bispinor-ibz` (`sources/lorrax_B`, head `5c884ac`)
**Target:** CrI3 6×6×1 80 Ry / bispinor / 16 GPUs (p_x=p_y=4, p_xy=16) / hbm80g
**Production knobs:** `r_chunk=24576`, `band_chunk=32`, `gflat_chunk_size=707` (cs)
**Scope:** Read-only audit of V_q's per-rank HBM footprint + planner formulas.
Empirical confirmation deferred — formulas below are derived from the source.
**Source files studied** (all `sources/lorrax_B/src/`):
* `gw/v_q_g_flat.py` — `_compute_V_q_g_flat_one_tile`, `_make_per_q_kernel`, `_make_read_all_ibz`, `_resolve_ibz_q_list`
* `gw/v_q_bispinor.py` — `compute_V_q_bispinor_g_flat_to_h5` (7-tile orchestrator)
* `gw/gw_init.py:855-1129` — `compute_V_q` entry point
* `file_io/zeta_reader.py:233-330` — `read_zeta_G_slab` (G-flat slab read path)
* `common/symmetry_maps.py:161-470` — `unfold_v_q` (IBZ→full-BZ unfold)

---

## 1. Scale parameters at the target config

| Symbol            | Value     | Source |
|-------------------|-----------|--------|
| n_q_ibz           | 36 (full-BZ) or 6 (IBZ when cascade active) | `_resolve_ibz_q_list` |
| n_q_full          | 36        | `nkx · nky · nkz` |
| ngkmax            | 59 990    | `gw.out:221` (5.33% of n_rtot) |
| n_rmu_C (logical) | 1508      | `gw.out:284` |
| n_rmu_T (logical) | 1504      | `gw.out:560` |
| n_rmu_C_padded    | 1520      | `_pad(1508) = 1508 + (16-4)%16 = 1520` |
| n_rmu_T_padded    | 1504      | `_pad(1504) = 1504` (already mesh-divisible) |
| p_x, p_y, p_xy    | 4, 4, 16  | mesh shape |

> **IBZ cascade status.** The 0X production run (`gw.out:205-220`) reports
> "q axis on disk: full BZ (36 q-points)" — the bispinor ζ files were
> written full-BZ because the orbit-aware centroid set was either absent
> or `LORRAX_FORCE_FULL_BZ=1` was set on the cs=707 mem probe. With the
> cascade active (kmeans_cli --orbit-aware), `n_q_ibz = 6` for CrI3 6×6×1
> and per-tile V_acc / ζ_all slabs shrink ~6×. Both regimes modelled below.

---

## 2. Per-tile HBM at peak — alive-simultaneously breakdown

Reading `_compute_V_q_g_flat_one_tile` line-by-line (v_q_g_flat.py:271-465):

**Pre-loop allocations** (lines 372-384, live throughout the per-q loop):
* `V_acc`: `c128[n_q_ibz, μ_L_pad, μ_R_pad]` at `P(None, 'x', 'y')` → `/p_xy`
* `g0_acc`: `c128[n_q_ibz, μ_L_pad]` at `P(None, 'x')` → `/p_x`
* `v_q_dev`: `c128[n_q_ibz, ngkmax]` at `P(None, None)` → **replicated** (every rank holds full table)
* `zeta_L_all`: `c128[n_q_ibz, μ_L_pad, ngkmax]` at `P(None, ('x','y'), None)` → `/p_xy`
* `zeta_R_all`: same shape, **distinct buffer iff same_zeta=False** (off-diag TT tiles); aliased to `zeta_L_all` for same_zeta=True (CC + 3 TT diag).

**Per-q kernel transients** (lines 88-141, inside `_make_per_q_kernel.fn`):
* `zeta_L_q` slice: `c128[1, μ_L_pad, ngkmax]/p_xy` (dynamic_slice of `zeta_L_all`)
* `zeta_L` reshard to `P('x', None)`: `c128[μ_L_pad, ngkmax]/p_x` — **replicated on the y axis**, so each rank holds a `p_y×` expansion of its slab share.
* `zeta_R` reshard to `P('y', None)`: `c128[μ_R_pad, ngkmax]/p_y` — replicated on x analogously.
* `V_q` block accumulator: `c128[μ_L_pad, μ_R_pad]/p_xy` (small).
* G-chunk scan body's `L_chunk, R_chunk, L_w`: ~`μ/p · g_chunk · 16` each, ~tens of MB at g_chunk≤4096; negligible vs ζ.

The per-rank peak inside a single tile is the **sum** of pre-loop allocations + per-q transients (V_acc is donated in-place; both ζ resharded copies coexist with the slab).

### Production numbers (n_q_ibz=36 full-BZ regime, p_xy=16)

| Term                       | Shape (c128)              | Shard | GB/rank |
|----------------------------|---------------------------|-------|---------|
| V_acc (CC)                 | (36, 1520, 1520)          | /p_xy | **0.083** |
| g0_acc (CC only)           | (36, 1520)                | /p_x  | 0.0002 |
| v_q_dev table              | (36, 59990)               | repl. | 0.035 |
| zeta_L_all (CC)            | (36, 1520, 59990)         | /p_xy | **3.283** |
| **zeta_L reshard P(x,⋅)**  | (1520, 59990)             | /p_x  | **0.365** |
| **zeta_R reshard P(y,⋅)**  | (1520, 59990)             | /p_y  | **0.365** |
| V_q block                  | (1520, 1520)              | /p_xy | 0.0023 |
| **CC tile peak (alive concurrently)** |                |       | **≈ 4.13 GB/rank** |

| Tile class            | same_zeta | Peak HBM (GB/rank) |
|-----------------------|-----------|--------------------|
| CC                    | True      | **4.13** |
| TT diagonal (1,1)/(2,2)/(3,3) | True | **4.09** |
| TT off-diagonal (1,2)/(1,3)/(2,3) | **False** | **7.34** ← **dominant** |

> **TT off-diagonal tiles are the binding peak.** `same_zeta=False` doubles the ζ-slab term: both `zeta_L_all` AND `zeta_R_all` are distinct `(36, 1504, 59990)/p_xy` ≈ 3.25 GB/rank buffers, giving 2 × 3.25 + 2 × 0.36 (resharded copies) + V_acc + v_q ≈ **7.34 GB/rank** during off-diag tile kernel.

### IBZ-cascade regime (n_q_ibz=6, same scale otherwise)

| Tile class            | Peak HBM (GB/rank) |
|-----------------------|--------------------|
| CC                    | 0.97 |
| TT diagonal           | 0.94 |
| TT off-diagonal       | **1.79** ← unfold post-loop expands V_acc to full BZ |

Cascade-active drops per-tile peak ≈ 4× via n_q_ibz axis. Note the
post-`unfold_v_q` V_acc IS at `n_q_full=36`, so it lives at the full-BZ
shape from line 451 onward (still ~0.083 GB/rank — fits trivially).

---

## 3. Post-unfold V_q shape — does it ever exceed the per-tile peak?

`unfold_v_q` (symmetry_maps.py:392-470) replaces V_acc in-place under a
`shard_map` whose `out_specs=P(None,'x','y')` and uses volume-preserving
`all_to_all` collectives. The output shape is `c128[n_q_full, μ_pad, μ_pad]/p_xy`
= **0.083 GB/rank** (CC) or **0.081 GB/rank** (TT) — **strictly smaller** than
the ζ_all slab (3.28 GB/rank) which has just been freed (`del zeta_L_all`
at v_q_g_flat.py:437). The unfold's internal scratch (`perm_q`, `qL`,
`phase` tables) is `n_q_full · n_rmu` ints/c128 — tens of MB, negligible.

**Verdict on full-BZ vs per-tile lifetime:** the unfolded V_acc never
coexists with the per-tile peak. No new memory bracket required.

---

## 4. Bispinor orchestrator: Lorentz-mix buffer (use_ibz_T branch only)

`compute_V_q_bispinor_g_flat_to_h5:587-728`: when the transverse IBZ
cascade is **active**, the 6 unique TT tiles are buffered post-unfold in
`tt_buffer`, then a 3×3 Lorentz mixing is applied via
`unfold_v_q_bispinor_lorentz` before write. During mixing:

* `tt_full_in` expands to 9 c128 tiles (3 unique upper + 3 Hermitian-mirror + 3 diag) at full-BZ shape — **9 × 0.081 = 0.73 GB/rank**.
* `tt_mixed` holds 6 unique outputs concurrently — **6 × 0.081 = 0.49 GB/rank**.
* Both dicts live during the mix call: **≈ 1.22 GB/rank** transient.

This is small relative to the per-tile kernel peak. When the cascade is
**off** (current production), tiles stream straight to disk and `tt_buffer`
is never populated — zero overhead.

---

## 5. V_q caller-scope persistent state during compute_V_q

`compute_V_q` is called from `_orchestrate_isdf_compute:1230`. At that
point the following are **live in caller scope** and will be added to
every V_q tile's peak:

* `psi_rmu_Y`, `psi_rmuT_X`: c128[nk, ns, μ_pad, nb_total]/p_xy
  - CrI3 80Ry: 2 × c128[36, 2, 1520, 150]/16 ≈ **2.05 GB/rank**
* `centroid_indices`: int32, negligible
* The fit_zeta tmp arrays were freed before return.

Total V_q persistent baseline ≈ **2 GB/rank** on top of every per-tile peak.

---

## 6. Verdict on the planner: should V_q get its own Peak E?

**Yes — V_q needs an independent Peak E**, for two reasons:

1. **Different shape regime.** ζ-fit peaks A–D are dominated by either
   FFT-box transients (A, D) or rank-5 pair-density buffers (C). V_q's
   binding term is the `ζ̃_all` 3-tensor `(n_q, μ, ngkmax)/p_xy` plus
   its `P(x,⋅)`/`P(y,⋅)` resharded copies. None of the existing peaks
   model this shape — Peak D's `gflat_acc` term `c128[nq_disk, μ, ngkmax]/p_xy`
   is the closest analog and **is** the same persistent ζ accumulator
   on disk, but V_q reads it back fresh and adds the two resharded
   `μ × ngkmax / p_axis` copies (Py× and Px× expansions) which exist
   only inside the per-q kernel.

2. **Distinct lifetime.** Peak D's `gflat_acc` is freed before
   `compute_V_q` runs (the ζ file is on disk; the JAX array goes out of
   scope between fit_zeta and V_q). Peak E starts from the V_q
   persistent baseline (ψ centroids only, ~2 GB/rank) and adds the
   per-tile kernel peak (≈ 7.34 GB/rank for TT off-diagonal at full BZ).

**HWM with current planner additions:** the old HWM = max(A,B,C,D) was
calibrated assuming V_q fit inside one of those. At 16 GPU CrI3 80Ry,
budget=70 GB/rank, Peak E ≈ 2 (persistent) + 7.34 (TT off-diag) ≈
**9.4 GB/rank** — far below budget. **V_q is NOT the binding peak at
this scale**; it's headroom. But the planner should still know about it
so it doesn't silently break at smaller systems / fewer GPUs / different
cascade states. For example at p_xy=1 (single GPU), the off-diag tile
balloons to 7.34 × 16 = 117 GB — which obviously can't run, and the
planner needs to refuse it cleanly.

---

## 7. Proposed `_peak_*` formulas (meta-variable style)

Match the existing `gflat_memory_model._peak_*` style. Add:

```python
def _peak_E_v_q_per_tile(*, n_q_ibz, mu_L, mu_R, ngkmax,
                         p_x, p_y, p_xy,
                         same_zeta: bool, write_g0: bool) -> dict:
    """Per-tile peak inside _compute_V_q_g_flat_one_tile's per-q kernel.

    The binding term is the IBZ ζ̃ slab plus its two P(x,⋅)/P(y,⋅)
    resharded copies inside the kernel (each replicated on the *other*
    mesh axis).  CC + TT-diagonal share zeta_L_all (same_zeta=True);
    TT-off-diagonal allocates a distinct zeta_R_all.
    """
    zeta_slab = _bytes_c128(n_q_ibz, mu_L, ngkmax, shard=p_xy)
    out = {
        "V_acc":          _bytes_c128(n_q_ibz, mu_L, mu_R, shard=p_xy),
        "v_q_table_replicated":
                          _bytes_c128(n_q_ibz, ngkmax),  # NOT sharded
        "zeta_L_all":     zeta_slab,
        "zeta_R_all":     (0.0 if same_zeta
                           else _bytes_c128(n_q_ibz, mu_R, ngkmax, shard=p_xy)),
        # Resharded ζ copies inside the per-q kernel — each replicated on
        # the OTHER mesh axis, so divides by p_x (resp. p_y), not p_xy.
        "zeta_L_on_x_axis":  _bytes_c128(mu_L, ngkmax, shard=p_x),
        "zeta_R_on_y_axis":  _bytes_c128(mu_R, ngkmax, shard=p_y),
        "V_q_block":      _bytes_c128(mu_L, mu_R, shard=p_xy),
        "g0_acc":         (_bytes_c128(n_q_ibz, mu_L, shard=p_x)
                           if write_g0 else 0.0),
    }
    return {f"E.{k}": v for k, v in out.items() if v > 0}


def _peak_E_v_q_unfold(*, n_q_full, mu_L, mu_R, p_xy) -> dict:
    """Post-loop unfold_v_q output (centroid double-permute + L-phase).

    Runs after zeta_L_all is `del`'d — never coexists with the per-tile
    slab.  Output is the full-BZ V_acc only; the unfold's internal
    perm_q / phase tables are ~tens of MB, ignored.
    """
    return {
        "E.V_acc_full_BZ":
            _bytes_c128(n_q_full, mu_L, mu_R, shard=p_xy),
    }


def _peak_E_v_q_bispinor_buffer(*, n_q_full, mu_T, p_xy,
                                use_ibz_T: bool) -> dict:
    """Bispinor Lorentz-mix transient (only when the transverse IBZ
    cascade is active — otherwise tiles stream straight to disk and this
    is zero).  At peak: tt_full_in (9 tiles) + tt_mixed (6 tiles) coexist.
    """
    if not use_ibz_T:
        return {"E.lorentz_mix_buffer": 0.0}
    tile = _bytes_c128(n_q_full, mu_T, mu_T, shard=p_xy)
    return {
        "E.tt_full_in_9_tiles": 9 * tile,
        "E.tt_mixed_6_tiles":   6 * tile,
    }
```

**Peak E total** for a single tile = `sum(_peak_E_v_q_per_tile)` (worst over
all 7 tiles — TT off-diagonal binds, `same_zeta=False`). The `_unfold`
output piggybacks (same buffer slot as V_acc, smaller for IBZ→full
because V_acc was at n_q_ibz and unfold lands at n_q_full; the all_to_all
trick keeps it 1× per rank).

**Lifetime in the orchestrator:** the 7 tiles run sequentially with
`del V_acc, g0_acc` between iterations — no per-tile accumulation. Only
the bispinor TT-buffer + caller-scope ψ centroids span tiles.

**HWM definition:** `Peak E = max(per_tile) + lorentz_mix_buffer + ψ_centroids_persistent`.

---

## 8. Comparison: post-unfold V_q vs per-tile peak

| Quantity                              | Shape                       | GB/rank (CrI3 80Ry, full-BZ) |
|---------------------------------------|-----------------------------|------------------------------|
| Per-tile peak (TT off-diag, binding)  | sum of 7-term breakdown     | **7.34** |
| Post-unfold V_acc (single tile)       | c128(36, 1520, 1520)/p_xy   | 0.083 |
| Ratio (per-tile / post-unfold)        |                             | **88×** |

The post-unfold artifact is always negligible compared to the per-tile
binding term. The `(n_q_full, μ, μ)` shape never threatens the HWM at
production scale; the **ζ̃ slab + resharded P(x,⋅)/P(y,⋅) copies are the
V_q-stage dominators**.

---

## 9. Empirical check status

**Not run.** Existing HLO dumps under
`runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_D_bispinor_hlo_2026-05-17/xla_dump/`
contain only ζ-fit `jit_fn` modules — that run had `LORRAX_EXIT_AFTER_ZETA=1`
and never compiled the V_q per-q kernel.  The earlier cs=707 mem probe
in `0X_lorrax_bispinor_fullbz_16gpu_2026-05-16` also exited after
fit_zeta.  Formulas above are derived analytically from the v_q_g_flat /
v_q_bispinor / zeta_reader source and the live `_PER_Q_KERNEL_CACHE`
shape signature.

A targeted V_q-only HLO run can be done on JID 53075115 by loading the
existing zeta_q*.h5 files in
`0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/tmp/` and removing the
`LORRAX_EXIT_AFTER_ZETA` env from the launcher.  Estimated wall: ~3 min
for V_q compile + 7-tile loop at p_xy=16 (each tile does 36 sync per-q
kernel calls; per-q is small at p_xy=16).  Recommend doing this as a
follow-up calibration before merging Peak E into the planner.

---

## Top-3 V_q HBM consumers at production scale (summary)

| Rank | Term                            | GB/rank | Meta-variable formula |
|------|---------------------------------|---------|-----------------------|
| 1    | `zeta_L_all + zeta_R_all` (TT off-diag) | **6.50** | `2 · _bytes_c128(n_q_ibz, mu_T, ngkmax) / p_xy` |
| 2    | `zeta_L on P(x,·) + zeta_R on P(y,·)` (kernel resharded) | **0.73** | `_bytes_c128(mu_L, ngkmax)/p_x + _bytes_c128(mu_R, ngkmax)/p_y` |
| 3    | `V_acc` (per tile)              | **0.083** | `_bytes_c128(n_q_ibz, mu_L, mu_R) / p_xy` |

Persistent caller-scope (ψ centroids) adds ~2 GB/rank; bispinor
Lorentz-mix buffer adds ~1.2 GB/rank when the IBZ cascade is active.
**V_q is sub-binding at 70 GB budget but needs its own Peak E in the
planner so the formula degrades gracefully at smaller systems / fewer
GPUs.**
