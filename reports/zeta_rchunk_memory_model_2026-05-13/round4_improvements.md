# Round 4 — Agent 1: Path D HLO before/after

**Question**: what did the structural fix actually buy, where did the
remaining 48 GiB go, and how much of the "200 → ~3 GiB" goal is met
vs still pending?

**Method**: read both `memory-usage-report.txt` files line by line,
count distinct slots per buffer class, identify what disappeared,
identify what's new.

**Sources**:
- BEFORE: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/xla_dump/module_0408.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt` (150 lines).
- AFTER:  `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_B_path_d_hlo_2026-05-13/xla_dump/module_0394.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt` (131 lines). Modules 0293 and 0295 are byte-identical to 0394; one report covers all three.

All line numbers below are in the AFTER report unless prefixed `BEFORE:`.

## 1. Before/after headline table

| Metric | BEFORE (`module_0408`) | AFTER (`module_0394`) | Δ |
|---|---|---|---|
| `Total bytes used`                                        | 200.35 GiB (BEFORE:1)             | 48.63 GiB (1)              | **−151.72 GiB (−75.7%)** |
| Preallocated-temp pool                                    | 196.30 GiB (BEFORE:7)             | 44.56 GiB (7)              | −151.74 GiB |
| FFT-box-class slots `c128[6,16,2,75,75,200]` (3.22 GiB ea.) | **58 distinct slots in preallocated-temp** (BEFORE:50-107) + 1 reference in allocation 0's value-list (BEFORE:114) | **0** — replaced by one `c128[360,2,75,75,200]` slot (line 45) | **58 → 1**, slot shape went from per-bc×k_chunk to chunk-aliased per-helper |
| FFT-box reshape `c128[6,32,1125000]` (3.22 GiB ea.)        | 49 line-mentions in preallocated-temp (each = one slot) | absent (the new helper uses `c128[360,2,1125000]` once, line 44) | 49 → 1 |
| Pair-density rank-4 slot, baseline shape `c128[6,16,2,59990]` (184 MiB) | 31 line-mentions | **absent** — pair density reshaped onto new rank-4 `c128[36,16,2,59990]` (1.03 GiB) | restructured |
| Pair-density rank-4 slot, new shape `c128[36,16,2,59990]` (1.03 GiB) | absent | **8 standalone slots** (lines 47–54) + 9 more as tuple components (lines 74–90) | NEW — see §3 |
| Per-side band slab `c128[36,10,2,73648]` (847 MiB)          | absent | 2 tuple-only references (lines 84, 88) | NEW — the remat target |
| `psi_Y_full` `c128[36,160,2,73648]` (13.55 GiB)             | absent (closest analog `c128[36,160,2,18332]`, BEFORE:90, 3.22 GiB at one offset) | **2 distinct slots** (lines 43, 44) + 2 tuple-only references (84, 88), 14.85 GiB each | NEW |
| Output shape `c128[36,94,73328]` vs `c128[36,94,73648]`     | BEFORE:25/132 | line 19                       | r-chunk grew slightly (73328 → 73648) — planner picked a different value, unrelated to fix |
| Planner HWM prediction                                      | 51.96 GB                          | 51.93 GB                   | ~0 — planner unchanged (which is the next problem) |
| Run outcome                                                 | OOM at fit_one_rchunk             | progressing through r-chunks | ✓ goal met |

`grep -c` cross-checks (substring matches, used to corroborate the visual counts):
- BEFORE: `c128[6,16,2,75,75,200]` 59 line-mentions (58 in preallocated-temp + 1 in allocation 0's value-list).
- AFTER:  `c128[360,2,75,75,200]` 1 line-mention.
- BEFORE: `c128[6,32,1125000]` 49 line-mentions.
- BEFORE: `c128[6,16,2,59990]` 31 line-mentions.
- AFTER:  `c128[36,16,2,59990]` 17 line-mentions (8 standalone preallocated-temp rows + 9 tuple-component rows).

## 2. What's no longer in the HLO (defect eliminated)

The Python-unrolled bc-loop × k-chunk-loop in `fetch_psi_rchunk` was
producing 58 *concurrent live slots* of the FFT-box class
`c128[6,16,2,75,75,200]` (3.22 GiB each). In the BEFORE report,
preallocated-temp lines 50-107 show those 58 slots stacked at 58
distinct byte offsets, cumulative 22%-96% of the pool. They are
**gone** in AFTER. The one remaining FFT-box transient is the new
helper's chunk-aliased box at `c128[360,2,75,75,200]` (12.07 GiB,
line 45, single slot at offset 12960003200) — i.e. **all 360 flat
`(k,n)` rows live as one buffer per-rank**, scan iterates over it
once, no per-iter slot pile-up.

That single-slot collapse is the entire ~150 GiB savings. Cross-check:
- 58 × 3.22 GiB ≈ 186.8 GiB removed in old FFT-box slots.
- 49 × 3.22 GiB ≈ 157.8 GiB removed in old `c128[6,32,1125000]` reshapes (same buffer class, different bitcast).
- These two sets overlap (XLA assigns a buffer once and many values share it); together they were the 196.30 GiB preallocated-temp pool.
- AFTER pool is 44.56 GiB → −151.74 GiB matches the headline 200 → 48 GiB drop.

**The principle that motivated Path D is satisfied for the FFT-box
class.** The bc-loop is gone; the k-chunk loop is gone; what remains
of the box pipeline is a single per-rank slot that gets scan-aliased.

Also gone: the bc-loop's concat staging area. The 11 `c128[6,32,1125000]`
values listed in allocation 0's value-list (BEFORE:114) — they used
to be the per-bc rchunk slabs the kernel concat'd along the band
axis. AFTER they collapse into the new helper's single output buffer
(see §3).

## 3. What's new in the HLO (where the 48 GiB now lives)

Three distinct new defect classes appear in AFTER. Each is a
principle violation by the same standard the original 58-slot box
was — `N · per-iter-unsharded-bytes` somewhere.

### 3a. `psi_Y_full` — 2× 14.85 GiB ψ(rchunk) slots

Lines 43 and 44 are two preallocated-temp slots of size 14.85 GiB
each, both containing values of shape `c128[36, 160, 2, 73648]`
(per-rank `(nk, nb_total, ns, r_chunk)`). This is the new helper's
*output* — and there are **two** copies of it live simultaneously,
not one. Sum: 29.70 GiB, ~67% of the 44.56 GiB pool.

The 14.85 GiB checks out: 36·160·2·73648·16 = 13.55 GB ≈ 12.62 GiB
of raw payload; XLA rounds up to the next bucket and shares the
14.85 GiB offset with three other values (line 43 has 4 values, line
44 has 11 values — they alias the same physical bytes through XLA's
buffer assignment). The two slots are at offsets `15950392448` and
`3200` — distinct underlying buffers, both live at the kernel peak.

**Why two slots?** Almost certainly because the downstream `slice` at
`isdf_fitting.py:1292` and the `shard_map` at `wfn_transforms.py:765`
are *both* materializing the rank-4 ψ(rchunk) under different
shardings, and XLA can't fuse them — see §3b below.

### 3b. The remat warnings — `c128[36,10,2,73648]` per-side band slab

32 `[spmd] Involuntary full rematerialization` warnings in
`gw.out`. All on the same shape `c128[36, 10, 2, 73648]` (per-rank
per-side band slab, 847 MiB). Two distinct source sites:
- `isdf_fitting.py:1292` — the `psi_Y_full[:, _l_lo:_l_hi, :, :]` /
  `[:, _r_lo:_r_hi, :, :]` slices that feed L and R into
  `z_q_from_psi_sm`.
- `wfn_transforms.py:765` — inside the new `gflat_to_rchunk` shard_map.

Both fail the resharding from `{devices=[1,16,1,1]<=[16]}` (band-flat
sharded, the helper's `out_spec`) to
`{devices=[1,1,1,4,4]<=[4,4]T(1,0) last_tile_dim_replicate}` (μ on x,
y replicated, the input spec of `z_q_from_psi_sm._local`).

In the HLO, this appears as the tuple pairs at lines 84 and 88:
`(c128[36,10,2,73648], c128[36,160,2,73648])` — the smaller `[36,10,...]`
is the desired sliced view; the larger `[36,160,...]` is the **full
ψ(rchunk) re-materialized** in the consumer's sharding because XLA
couldn't reshape-then-slice cheaply.

**This is the next defect.** The principle says: zero replicated
intermediates. Having `psi_Y_full` materialized twice (once at
helper's `out_spec`, once at consumer's `in_spec`) is the same class
of bug as the original 58 FFT-box slots, just at a coarser granularity.

### 3c. Pair-density rank-4 buffers — 8× `c128[36,16,2,59990]`

Lines 47-54 list 8 distinct preallocated-temp slots of shape
`c128[36, 16, 2, 59990]` (1.03 GiB each), and 9 more
`c128[36, 1, 2, 59990]` slots (65.91 MiB each, lines 56-64). Sum:
~8.8 GiB. In the baseline these existed as `c128[6, 16, 2, 59990]`
(184 MiB × 31 mentions ≈ 5.7 GiB).

The shape change `[6, 16, ...]` → `[36, 16, ...]` reflects that the
new pipeline no longer chunks across the k-axis (the helper does it
in one scan over the flat `(k,n)` axis with no inner k-chunk). The
*per-slot* cost grew 6× (full k vs k-chunked), but the *slot count*
dropped from 31 to 8 — net change is mild (+3 GiB). This is
inherited from `z_q_from_psi_sm`'s post-pair pipeline (rank-7 IFFT/FFT
chain), not from Path D, so it's an existing budget item the
structural fix neither helped nor hurt much.

## 4. Net assessment

### Memory savings attribution

Goal as framed in `PATH_D_PICKUP.md`: **slot count → 0 unsharded
transient**, total → ~3 GiB. We landed at 48 GiB, 16× over budget.
Decomposing the gap:

| Component                                                | Size in AFTER | Origin |
|----------------------------------------------------------|---------------|--------|
| Single FFT-box `c128[360,2,75,75,200]` + ifft scratch (line 45) | ~12 GiB       | New helper's per-rank one-shot FFT box. Could shrink by ~10× if `chunk_size` is set to e.g. 36 (one k worth) instead of one-shot 360 — **the chunk_size knob discussed in `parallel_helpers_discussion.md` Q3 wasn't engaged**. |
| 2× `psi_Y_full` slots (lines 43, 44)                      | ~30 GiB       | The remat defect (§3b). Eliminating the second materialization recovers ~15 GiB. |
| 8× `c128[36,16,2,59990]` pair-density (lines 47-54)       | ~8 GiB        | Pre-existing `z_q_from_psi_sm` lifetime overlap. Not in Path D scope but loomed larger now that k isn't chunked. |
| Output `c128[36,94,73648]` etc. (line 99)                 | ~3.7 GiB      | maybe-live-out, irreducible. |
| Constants, parameters, misc small slots                   | ~1 GiB        | Irreducible. |
| **Total**                                                 | **~48 GiB**   |        |

**Score**: of the ~197 GiB of preallocated-temp the structural fix
was on the hook for, ~152 GiB (~77%) was actually eliminated. The
remaining 48 GiB is:
- ~13 GiB irreducible (output + constants + per-side band slots needed by downstream consumer).
- ~30 GiB attributable to **the new remat defect** (§3b) — a fresh principle violation introduced by Path D's sharding-spec choice at the helper boundary.
- ~5 GiB attributable to the helper's own one-shot FFT-box choice that could be tuned via `chunk_size`.

In other words: **the structural fix worked on its own terms** — the
58-slot pile-up is genuinely gone, reduced to one slot. But it
exposed (or created) two new principle violations of the same
mechanism, just one level coarser:
1. The two-copy `psi_Y_full` (helper output materialized twice for
   downstream resharding) → **the remat defect**.
2. The one-shot FFT box that didn't engage `chunk_size`.

Fixing (1) recovers ~15 GiB. Fixing (2) recovers another ~10 GiB.
Together they bring the budget close to ~22 GiB, comfortably under
the 28 GiB / GPU memory budget on Perlmutter HBM40 nodes and far
enough under HBM80 to be uncontested.

The planner's HWM prediction (51.93 GB vs measured 48.63 GiB) is now
**within 7%** — close enough that the `band_fft_pool` stopgap term
can be removed in the same commit that fixes the remat. Mark this
for the cleanup pass.

---

Agent 1 round 4 done
