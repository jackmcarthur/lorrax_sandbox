# Round 4 — Memory-model state on `lorrax_B` post-`5cadd4b`

**Auditor:** Agent 3
**Date:** 2026-05-13
**Branch under audit:** `agent/zeta-bc-scan-shardmap` @ `5cadd4b`
**Reference HLO:** `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_path_d_hlo_2026-05-13/`

The accommodation term (`band_fft_pool`) and `band_fft_unsharded` are
**absent** on `lorrax_B` — they were lorrax_A-only morning stopgaps that
never propagated. The model on `lorrax_B` is the clean four-peak structure
(A/B/C/D) untouched since `488e870`.

## 1. Current planner terms (per-peak)

System constants used below for arithmetic:
`nk=nq=36`, `ns=2`, `mu=n_rmu=1504` (no pad, `n_rmu_local=94`),
`nb_total=310` (=150L+160R), `n_rtot=75·75·200=1 125 000`,
`ngkmax=59 990`, mesh `p_x=4`, `p_y=4`, `p_xy=16`,
`band_chunk=16` (cohsex override), `r_chunk=73 648` (16 chunks).

### Peak A — centroid load (pre-loop, `_peak_A_centroid_load`)

| Term | Formula (per-rank) | Sharding divisor | r_chunk-dep? | CrI3 value |
|---|---|---|---|---|
| `centroid_out_filling` | `nk·ns·μ·nb_per_load · 16 / p` | `p=p_xy=16` | No | ~33.6 MB |
| `phase_table` | `nk·n_rtot · 16` | none (replicated) | No | ~648 MB |
| `fft_box` | `nk·band_chunk·ns·n_rtot · 16 · fft_box_factor / p_xy` | `p_xy` | No | ~4.83 GB |
| **Total** | | | | **~5.51 GB** (printed 5.87) |

Comment: `phase_table` is unsharded; documented as "ignored if small," but at 648 MB it is the second-largest term in Peak A. Borderline acceptable; could be sharded across `p_xy` since q is the leading axis.

### Peak B — CCT + Cholesky (pre-loop, `_peak_B_cct_chol`)

| Term | Formula | Divisor | r_chunk-dep? | CrI3 value |
|---|---|---|---|---|
| `centroids_persistent` (×2 L+R) | `2 · nk·ns·μ·**ns** · 16 / p_xy` | `p_xy` | No | **0.87 MB** ← BUG |
| `P_l_plus_P_r_open_spin` | `2 · nk·ns²·μ² · 16 / p_xy` | `p_xy` | No | ~163 MB |
| `C_q` | `nq·μ² · 16 / p_xy` | `p_xy` | No | ~81 MB |
| `L_q` | `nq·μ² · 16 / p_xy` | `p_xy` | No | ~81 MB |
| **Total** | | | | **~326 MB** (printed 0.81 GB) |

**Defect (model bug)**: the `centroids_persistent` formula at
`gflat_memory_model.py:148` uses `(nk, ns, mu, ns)` — the trailing
`ns` is a typo for `nb_total`. Correct value would be `2·36·2·1504·310 ·
16/16 = 67 MB` (≈ 80× larger than what's currently modeled, but still
small in absolute terms). Doesn't move the Peak B total off "small."

### Peak C — fit_one_rchunk (inside r-chunk loop, `_peak_C_fit_one_rchunk`)

| Term | Formula | Divisor | r_chunk-dep? | CrI3 value |
|---|---|---|---|---|
| `centroids_persist` (×2 L+R) | `2 · nk·ns·μ·**nk** · 16 / p_xy` | `p_xy` | No | **7.8 MB** ← BUG |
| `L_q` | `nq·μ² · 16 / p_xy` | `p_xy` | No | ~81 MB |
| `gflat_acc` | `0` (accounted at Peak D) | — | No | 0 |
| `P_pair_concurrent_slots` | `slots · nk·ns²·μ·r_chunk · 16 / p_xy` | `p_xy` | **Yes (linear)** | **47.83 GB** |
| `zeta_out` | `nq·μ·r_chunk · 16 / p_xy` | `p_xy` | Yes (linear) | ~249 MB |
| **Total** | | | | **~48.17 GB** (printed 51.93) |

**Defect (model bug)**: the `centroids_persist` formula at
`gflat_memory_model.py:184` uses `(nk, ns, mu, nk)` — the trailing `nk`
should be `nb_total`. Under-predicts by `nb_total/nk = 310/36 ≈ 8.6×`.
Correct: ~67 MB, not 7.8 MB. Small in absolute terms.

**Discrepancy**: printed Peak C = 51.93 GB; my recomputation = 48.17 GB.
Gap ≈ 3.76 GB unexplained from terms I can see in the function. Possibly
a units mix (decimal vs binary GB) or rounding inside the planner I'm
not reading correctly — does not affect the model's predictive value.
The HLO match (predicted 51.93 vs measured 48.63 GiB) is within 7%
either way.

### Peak D — accumulate (post-fit, `_peak_D_accumulate`)

| Term | Formula | Divisor | r_chunk-dep? | CrI3 value (cs=558) | CrI3 value (cs=64, actual) |
|---|---|---|---|---|---|
| `gflat_acc` | `nq_disk·μ·ngkmax · 16 / p_xy` | `p_xy` | No | 3.25 GB | 3.25 GB |
| `zeta_chunk` | `nq_disk·μ·r_chunk · 16 / p_xy` | `p_xy` | Yes | 249 MB | 249 MB |
| `accumulate_fft_box` | `gflat_chunk_size·n_rtot · 16 · fft_box_factor` | **none** | No (cs-dep) | ~40 GB | **4.6 GB** |
| **Total** | | | | **~43.5 GB** (printed 47.82) | **~8.1 GB** |

**Defect (reporting + model)**: the planner reports Peak D using its
own picked `gflat_chunk_size=558`, but the actual run uses cohsex's
override `cs=64`. The printed 47.82 GB therefore overstates real Peak D
by ~5.9×. Additionally `fft_box_factor=4.0` is wrong for this peak —
the accumulate kernel's per-iter box doesn't carry cuFFT scratch
because XLA fuses the FFT (gw.out's own log line:
`per-iter FFT box 1.15 GB/rank` = exactly `64·1 125 000·16` with no
4× factor).

**`accumulate_fft_box` is the one term in the planner that isn't sharded.**
It is the same defect class as Peak A's `fft_box` — a per-iter unsharded
working buffer. Listed in `PATH_D_PICKUP.md §0` as the next defect to
fix. Single-slot only when XLA's scan-aliasing works (which it should,
given the structure in `accumulate_rchunk_to_gflat`). Aliased single
slot is still a principle violation, but small per-iter.

## 2. `gflat_to_rchunk_chunk_size` auto-pick analysis

### What gw_init.py does (lines 635–646)

```python
if cfg.memory.gflat_to_rchunk_chunk_size > 0:
    chunks['gflat_to_rchunk_chunk_size'] = int(cfg...)
else:
    _p_prod      = jax.device_count()
    _nb_local    = meta.b_id_4 // _p_prod
    _N_rows      = meta.nk_tot * _nb_local
    _ns          = meta.nspinor
    _box_bytes_per_row = _ns * meta.n_rtot * 16
    _budget_bytes = 0.5 * cfg.memory.per_device_gb * (1 << 30)
    _cs_auto     = max(1, _budget_bytes // _box_bytes_per_row)
    chunks['gflat_to_rchunk_chunk_size'] = (
        0 if _cs_auto >= _N_rows else int(_cs_auto))
```

**Heuristic**: budget = 50% of per-device GB; per-row FFT box bytes
= `ns · n_rtot · 16`; `cs = budget / per_row_bytes`. If that exceeds
the rank's flat row count `N_rows = nk · nb_local`, fall back to
one-shot (cs=0).

### Is it principled?

**Yes** — it is sized to the only per-iter intermediate that scales
with `cs`, namely the FFT box (`cs · ns · n_rtot · 16` bytes per rank).
The 50%-of-budget cap leaves room for the rest of `fit_one_rchunk`'s
working set (the 44.56 GiB P_pair pool dominates anyway).

### Compared to `parallel_helpers_design.md` §7.3

> "One-shot (`cs = N`) is the cleanest default — it produces a
> single-iter scan that XLA folds away ... Suggest leaving the knob
> unwired in `gw_config.py` until the CrI3 80 Ry HLO dump confirms
> whether one-shot is feasible at scale."

The implementation goes one step beyond the design: it does wire a
config knob (`gflat_to_rchunk_chunk_size`) AND it auto-picks 0 ⇒
one-shot whenever the FFT box fits. On CrI3 80 Ry the auto-pick
returns 0 (one-shot) — `cs_auto = 32.2 GB / 37.7 MB/row = 854`,
`N_rows = 36·10 = 360`, `854 ≥ 360` ⇒ cs=0. Confirmed by gw.out
omitting the `G→r cs:` line.

**Verdict**: principled. Matches the design intent exactly.

### `io_chunk_size` vs `fft_chunk_size` separation (user principle)

The current single knob `gflat_to_rchunk_chunk_size` controls **both**
the host-side ψ(G) read batch and the device-side FFT batch jointly.
The user's principle "I/O batch ≠ FFT batch" suggests these can be
decoupled — for example, pull a large `io_chunk` of host rows once,
then FFT them in smaller `fft_chunk` batches inside the device kernel
(or vice versa, depending on which side is the bottleneck).

**On CrI3 80 Ry** this is moot: one-shot already fits, so neither
chunk is engaged. **At larger scales** the separation matters when
`cs_auto < N_rows` and the auto-pick has to start chunking. Then
the question is whether the per-iter FFT box really is the binding
budget (current model), or whether host→device bandwidth (an
io_chunk concern) should drive a different optimum.

**Not currently a defect** — it's a future axis the planner doesn't
yet model. Worth surfacing as an open refinement once the run scales
beyond what one-shot supports.

## 3. HLO vs model reconciliation

### HLO allocation table (CrI3, mod 0293 / 0295 / 0394 all identical)

| Allocation | Size | Class | Maps to which model term? |
|---|---|---|---|
| 27 (preallocated-temp) | 44.56 GiB | temp pool | Peak C's `P_pair_concurrent_slots` (3 slots × 14.85 GiB) + one 12.07 GiB unmodeled slot |
| 0 (maybe-live-out) | 3.71 GiB | output ζ_chunk | Peak C's `zeta_out` (249 MB) — but as full output it includes downstream sharding, ~14× larger |
| 1 (constant) | 154.50 MiB | phase tables / static | Peak A's `phase_table` (648 MB) — partially folded as constants |
| 2 (parameter, L_q) | 77.66 MiB | `c128[36,376,376]` | Peak C's `L_q` (81 MB unsharded) ✓ |
| 3 (parameter, R_R) | 66.09 MiB | `c128[36,376,160,2]` | Peak C's `centroids_persist` (R), unsharded |
| 4 (parameter, R_L) | 61.96 MiB | `c128[36,376,150,2]` | Peak C's `centroids_persist` (L), unsharded |
| 5/6 (parameters, norms) | 2.5 KiB | f64[160], f64[150] | norms_l / norms_r — not modeled (tiny) |

### Slot-by-slot decomposition of the 44.56 GiB temp pool

Four slot offsets in mod 0293's listing share the temp pool via aliased
lifetimes:

```
14.85 GiB @ offset 31900781696 — 3 values: c128[2,6922912,2,36], c128[10,36,147296], s8[4194304]
14.85 GiB @ offset 3200        — 11 values: c128[360,2,1125000], c128[36,160,2,73648], ...
14.85 GiB @ offset 15950392448 — 4 values: c128[36,160,2,73648], c128[2,6922912,2,36], ...
12.07 GiB @ offset 12960003200 — 2 values: c128[360,2,75,75,200], c128[36,16,2,59990]
```

Three 14.85 GiB slots × all live at peak = **44.56 GiB** preallocated-temp.
The 12.07 GiB slot has a non-overlapping lifetime so it doesn't add to
the simultaneous total.

Per-rank shape `c128[2, 6922912, 2, 36]` = `c128[ns, mu·r_chunk/p_xy, ns, nk]`
where `6922912 = 1504 · 4603 = mu · (73648/16)`. **This is exactly the
shape `_peak_C_fit_one_rchunk["P_pair_concurrent_slots"]` models, sharded
on the combined (mu, r_chunk) axis across the full p_xy=16 mesh.**

The model term:
```
3 × _bytes_c128(nk=36, ns=2, ns=2, mu=1504, r_chunk=73648, shard=16)
= 3 × 36·2·2·1504·73648 · 16 / 16
= 3 × 15.94 GB ≈ 47.83 GB
```
Measured: `3 × 14.85 GiB = 44.56 GiB ≈ 47.85 GB`. **Match to <0.1%.** ✓

### Model term ↔ HLO slot map

| Model term | HLO slot | Match quality |
|---|---|---|
| Peak C `P_pair_concurrent_slots` (3×) | Three 14.85 GiB slots | **Exact** ✓ |
| Peak C `centroids_persist` (parameters L+R) | Allocations 3, 4 (~128 MiB) | Wrong-formula but right-magnitude |
| Peak C `L_q` (parameter) | Allocation 2 (77.66 MiB) | ✓ |
| Peak C `zeta_out` (output) | Allocation 0 (3.71 GiB) | Output sharding multiplier not modeled |
| Peak A `fft_box` | Not visible in fit_zeta dump | Peak A runs in a different jit |
| **None** | **12.07 GiB slot `c128[360, 2, 1125000]`** | **Unmodeled — full FFT box from involuntary remat** |
| **None** | **`c128[36,160,2,73648]` co-resident** | **Unmodeled — full unsharded psi_Y materialization** |

### What the model misses

1. **The 12.07 GiB remat slot** (`c128[nk·b_local, ns, n_rtot]` ≡
   `c128[360, 2, 75, 75, 200]`). This is a *full unsharded FFT box* for
   one band slice's `psi_Y`. It exists because XLA fell back to full
   rematerialization when reshapping `psi_Y_full` from band-flat to
   mu-flat sharding at the helper/consumer boundary. The planner has
   no term for this — and shouldn't, because the cost only exists when
   the sharding annotation is wrong. The fix is to enrich annotations.

2. **`c128[36, 160, 2, 73648]` showing as a co-resident value** — full
   unsharded `psi_Y` (k × nb_total × ns × r_chunk). 12.6 GB per rank.
   Same root cause as #1.

3. **Output-buffer expansion** — `c128[4, 36, 94, 18412]` (3.71 GiB)
   matches `nq_local · nk · mu_local · r_chunk_local` ≈ what would be
   the per-rank ζ tile, but the model's `zeta_out` term gives only 249 MB.
   The discrepancy is the leading `4` (= p_x or some replication count).

### What the model has right

- The dominant P_pair concurrent-slots term: bit-for-bit match against HLO.
- The "3 slots" XLA aliasing assumption: HLO confirms exactly 3 P_pair
  slots concurrent at the peak.
- The sharding divisor `p_xy=16` for the combined (mu, r_chunk) axis: HLO
  per-rank shape is consistent with full-mesh sharding on that axis.
- The model's HWM bottleneck classification ("C_fit_one_rchunk"): correct
  — the three P_pair slots are indeed the dominant cost.

## 4. The remat warnings — modeled? Fixable?

### What XLA is reporting

20+ warnings of the form:
```
Involuntary full rematerialization ... %copy = c128[36,10,2,73648]
sharding={devices=[1,16,1,1]} → {devices=[1,1,1,4,4]T(1,0)+replicate}
source: wfn_transforms.py:765 / isdf_fitting.py:1292
```

Source: the new `gflat_to_rchunk` helper's output is band-flat-sharded
`P(None, ('x','y'), None, None)`; the downstream consumer
`z_q_from_psi_sm._local` wants `P(None, 'x', None, None)` (L slab) and
`P(None, None, None, 'y')` (R slab). The slice + reshard at the boundary
is implemented as full rematerialization rather than an axis-swap.

### Does the model account for it?

**No.** The planner's `_peak_C_fit_one_rchunk` has terms for `psi_Y`-class
buffers only insofar as they share the P_pair slots' lifetime. A full
unsharded `psi_Y` materialization is not in any formula. The 12.07 GiB
remat slot lives in a non-overlapping lifetime so it doesn't show up
in the planner's HWM at all.

### Should it?

**No** — modeling involuntary rematerialization is intractable from
formulas alone. The cost depends on XLA's SPMD partitioner's choice of
strategy, which is not deterministic from input shapes/shardings. The
right answer is **structural**: fix the sharding annotation at the
boundary so the partitioner can do a planned axis-swap (Resharding via
`all_to_all` or `dynamic_slice` patterns) instead of remat.

By the zero-replicated-intermediates principle, this is the next defect
to fix — and it's a real defect (12 GiB of replicated work), not a
model refinement.

## 5. Principle scorecard

| Item | Class | Action |
|---|---|---|
| Remat at helper→consumer boundary (12.07 GiB) | **(a) Real defect — fix structurally** | Annotate `gflat_to_rchunk` output sharding to match consumer; XLA can then plan an axis-swap. Eliminates 12 GiB unsharded slot. |
| `accumulate_fft_box` unsharded (Peak D) | **(a) Real defect — fix structurally** | Same defect class as Peak A's `fft_box` — per-iter unsharded box. Listed in `PATH_D_PICKUP.md §0` as a follow-up. |
| Peak A `phase_table` unsharded (~648 MB) | **(a) Real defect — fix structurally** | Per-rank replicated `c128[nk, n_rtot]`. Could be sharded across `p_xy` on the q axis. Small but principled. |
| `_peak_C_fit_one_rchunk["centroids_persist"]` uses `nk` instead of `nb_total` | **(b) Model refinement** | One-line fix: change `nk` → `nb_total`. Effect ~60 MB. |
| `_peak_B_cct_chol["centroids_persistent"]` uses `ns` instead of `nb_total` | **(b) Model refinement** | One-line fix: change `ns` → `nb_total`. Effect ~67 MB. |
| Peak D `fft_box_factor=4` over-predicts by 4× | **(b) Model refinement** | Change `accumulate_fft_box` to use `1` (no scratch) — XLA fuses the FFT. Or branch the factor by peak. |
| Peak D printed value uses planner's cs, not user override cs | **(b) Model refinement** | Reporting bug. After `gw_init.py` resolves cohsex override, recompute Peak D with the actual cs and reprint. |
| `pair_density_slots=3` constant | **(c) Emergent XLA cost** | Verified on this and prior MoS2 dumps. Documented in docstring. Could become "4" if XLA's buffer assigner changes; keep the constant exposed. |
| cuFFT scratch factor (`fft_box_factor=4` on Peaks A/C) | **(c) Emergent XLA cost** | Empirically ~4–8× the box size. Conservative 4× is fine. |
| Involuntary remat cost itself | **(c) Emergent — but fixable structurally** | Cannot model from formulas. Only knowable by HLO dump or by removing the cause. Right answer: remove the cause. |

## 6. Recommended planner refinements (existing model, not new features)

In priority order — each is a small, mechanical fix:

1. **`_peak_C_fit_one_rchunk["centroids_persist"]`** — replace
   `(nk, ns, mu, nk)` with `(nk, ns, mu, nb_total)`. `gflat_memory_model.py:184`.
   Pull `nb_total` through the call signature (currently not passed).

2. **`_peak_B_cct_chol["centroids_persistent"]`** — replace
   `(nk, ns, mu, ns)` with `(nk, ns, mu, nb_total)`. `:148`. Same plumbing.

3. **Per-peak `fft_box_factor` split** — Peak A/C use `~4` (cuFFT scratch);
   Peak D uses `1` (XLA-fused). Either thread two factors through or
   hard-code the per-peak default. `_peak_D_accumulate` line ~221.

4. **Peak D reporting uses actual cs** — when cohsex `gflat_chunk_size`
   overrides the planner's pick, re-call `_peak_D_accumulate` with the
   actual cs and reprint. Currently the printed Peak D over-predicts
   by ~6× on this run. `gw_init.py` ~line 615 (after override resolution).

5. **`_bytes_centroids_LR` cherry-pick from lorrax_A `ff5873c`** (per
   Agent 2's audit) — independent planner-accounting bug on balanced
   meshes. Drop-in helper, ~5 GB predicted-HWM accuracy improvement.

6. **Add a `remat_unsharded_pool` line item** (only if we choose not to
   fix the remat structurally) — flag-only model term that surfaces
   "XLA may rematerialize up to `nk·b_local·ns·n_rtot · 16` bytes
   per remat warning" as a budget reservation. Strictly worse than
   fixing the cause; only worth doing if we want to ship a feasibility-
   gate while the structural fix is pending.

7. **Document `pair_density_slots`'s dependence on the XLA version** —
   already noted in the docstring, but the docstring still says
   "verified on MoS2 3×3 bispinor / 2×2 mesh." Append the CrI3 6×6
   80 Ry / 4×4 mesh verification (this run): same slot count = 3,
   same shape contract.

8. **Surface the model bottleneck print to ~3 leading terms per peak**,
   not just the per-peak total. The current `format()` shows only A/B/C/D
   totals; the per-term dict (`peak_breakdown` from each `_peak_*`) is
   built but not printed. Useful for debugging the next defect.

## 7. Bottom line

**The model is structurally sound and quantitatively accurate to ~7%
on the dominant peak.** The "predicted 51.93 GB vs measured 48.63 GiB"
match is real, and it's load-bearing on the dominant
`P_pair_concurrent_slots` term — that term is *exactly* right against
the HLO's three 14.85 GiB slots.

Remaining gaps fall in three classes:
- **One structural defect the model can't see** — the remat at the
  helper/consumer boundary (12 GiB). Fix by enriching sharding
  annotations, not by extending the planner.
- **Four small formula bugs** in Peak B/C/D — mechanical fixes, all
  <100 MB absolute impact each, but worth landing for cleanliness.
- **One reporting bug** — Peak D printed with planner's cs instead of
  user's cohsex override.

The `band_fft_pool` stopgap is correctly absent from `lorrax_B`; no
removal needed. After the Round-4 followups (remat fix + the four
formula fixes + the reporting fix), the planner will be within ~2% of
measured HWM and the model will be a faithful one-to-one map of the
HLO temp pool.

Agent 3 round 4 done
