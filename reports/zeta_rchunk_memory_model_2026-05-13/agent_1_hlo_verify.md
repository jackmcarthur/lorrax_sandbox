# Agent 1 — independent HLO verification

Source dump: `module_0408.jit__kernel.sm_8.0_gpu_after_optimizations-memory-usage-report.txt`
Run log: `runs/CrI3/M_6x6_80Ry_2026-05-07/lorrax_A_hlo_dump_2026-05-13/gw.out`

Key run parameters from `gw.out`:
- "Zeta fitting: 16 r-chunks x 73328 r-points, **160 bands (150 left + 160 right)**"
- band_chunk = 16, psig_k_chunk_size = 6, n_rmu_padded = 1504 (per `c128[36,376,376]` = L_q with μ_X = 376 = 1504/4), n_rtot = 75·75·200 = 1,125,000, ns = 2, nk = 36, mesh 2×2 (per offset patterns and μ_local = 94 = 1504/16).

**Note:** mesh appears to be **2×2 (P=4, p_x=p_y=2)**, not 4×4 — the parameter shape `c128[36,376,376]` shows μ sharded once (376 = 1504/4) on each axis, and the output shape `c128[36,94,73328]` shows μ_XY = 94 = 1504/16. So 4 ranks total with each rank holding μ/4 = 376 on either of the two single-axis-sharded forms. Wait — that's inconsistent. Re-examining: output `c128[36,94,73328]` has 94 in the μ slot; 1504/16 = 94 ✓ → P = 16 → mesh **could** still be 4×4 with `gflat_acc` flat-sharded over both axes. Single-axis copies show 376 = 1504/4, consistent with p_x = 4 or p_y = 4. So **P = 16, mesh = 4×4** — consistent with the orchestrator's stated config (4 A100-40GB nodes). Setting that aside.

---

## 1. Verification of Claim 1 — `pair_density_slots = 3`

**VERIFIED.** Exactly three 14.79 GiB slots in `allocation 31`'s "Allocations sorted by size with their values" section (lines 46, 47, 48 of the HLO report):

| Line | Size | Offset | Shapes (representative) |
|---|---|---|---|
| 46 | 14.79 GiB | 15881095040 | `c128[2,6892832,2,36]`, `c128[36,36664,752]`, `c128[2,18332,376,2,6,6,1]` |
| 47 | 14.79 GiB | 31762179968 | `c128[36,150,376,2]`, `c128[2,6892832,2,36]` |
| 48 | 14.79 GiB | 10112 | `c128[36,1,4,94,18332]`, `c128[36,36664,752]`, `c128[18332,376,6,6,1]`, … |

Per-rank size check: `c128[2, 6892832, 2, 36]` with 6892832 = 376·18332 = μ_X·r_Y → bytes = 2·6892832·2·36·16 = 15,876,184,576 B = 14.79 GiB ✓. Likewise `c128[36,36664,752]` = `nk·(2·r_Y)·(2·μ_X)` = 14.79 GiB ✓. The rank-7 bitcast variant `c128[2,18332,376,2,6,6,1]` is the same payload at the kgrid-3D layout `(ns_l, r_Y, μ_X, ns_r, nkx, nky, nkz)`.

These are the rank-5/7 pair-density slots `P_l_R_conj`, `P_r_R`, and a third that holds the γ̃-contract scratch / XLA's internal slot, exactly per the existing `pair_density_slots = 3` constant in `gflat_memory_model.py:175`. No 4th slot, no slot smaller than 14.79 GiB carrying a pair-density shape. **Confidence: high.**

---

## 2. Verification of Claim 2 — `S_fft ≈ 3` and 58 slots

**58 slots confirmed; `S_fft = 3` works algebraically under the orchestrator's "`nb_total = nb_L + nb_R`" parameterization, but the slot/bc-iter mapping has a notational subtlety worth flagging.**

### Slot count

Counted lines 50–107 inclusive (3.22 GiB slots) → **58 slots**. ✓

Per-slot bytes: `c128[6,16,2,75,75,200]` = 6·16·2·1,125,000·16 = 3.456 GB = 3.22 GiB ✓.
**No `/P` factor.** Sharded on (k_chunk=6, bc=16, ns=2) with the full n_rtot per rank → confirms the unsharded-FFT-box pathology. The expected sharded size would be 3.22 GiB / 16 = 0.20 GiB per rank; observed is 3.22 GiB.

### Recurring shape variants in each slot

Surveying the 58 slot rows, three shapes recur:

1. `c128[6, 16, 2, 75, 75, 200]` — band-FFT box at 3D layout (present in ~all 58 slots)
2. `c128[6, 16, 2, 59990]` — G-sphere variant (G-flat ψ at ngkmax=59990; appears in ~36 slots)
3. `c128[6, 32, 1125000]` — k × (ns·bc) × n_rtot flat layout, same payload as (1) at a different reshape (appears in ~50 slots)

A few slots also hold smaller buffers (e.g. `c128[36,1,2,59990]`, `c128[36,160,2,18332]` — the latter is the L/R reshard slab; see Claim 3). These all share lifetime slots with the dominant three.

### S_fft derivation

From `gw.out`: **nb_full = 160** (the bc-loop range = b4 − b0), `band_chunk = 16` → **N_BC (Python-unrolled iterations) = 10**, not 20.

`58 / 10 = 5.8` — so if you parameterize by N_BC (the actual Python loop count), `S_fft ≈ 6`.

The orchestrator's framing in hlo_findings.md asserts "N_BC = 20 from nb_total = 310/16, S_fft = 3". This works algebraically because `(nb_L + nb_R) / band_chunk · S_fft = 310/16 · 3 ≈ 58.1`. The formula `(nb_L+nb_R) · S_fft · k_chunk · ns · n_rtot · 16` gives 200.88 GB = 187 GiB, matching the observed 186.76 GiB to within 0.1%. So the **formula is correct**.

### Caveat — the slot-per-bc factor isn't really 3

The 20-iter framing requires interpreting each Python-bc-iter as TWO XLA sub-iters (one for L, one for R) — which isn't literally what the Python unroll produces (it produces one io_callback + FFT + slice per bc, with internal L/R extraction from the resulting r-slab). The 5.8 slots-per-Python-bc is the honest accounting.

The reason the algebra still works: the total bytes of band-FFT scratch are linear in `(nb_L + nb_R)` because the per-bc reshard outputs `psi_l_Y_bc` (size ∝ nb_L_in_bc) and `psi_r_Y_bc` (size ∝ nb_R_in_bc) **both** live concurrently, and `nb_L_in_bc + nb_R_in_bc` summed over bc iters equals `nb_L + nb_R`. So the total slot-bytes are `(nb_L + nb_R) · 16·k_chunk·ns·n_rtot` regardless of how you partition into "iters × slots-per-iter".

**Verdict on Claim 2:** The numerical claim (58 slots, 186.76 GiB) is exact. The S_fft = 3 framing is consistent with the orchestrator's formula but conceptually folds the L/R duplication into the slot count rather than into the band axis. As long as the planner formula uses `(nb_L + nb_R)`, not `nb_full`, the prediction is faithful. **If a future reader uses `nb_full` with S_fft = 3, they will under-predict by a factor `(nb_L + nb_R)/nb_full` ≈ 2.** Flag this in the source comment.

**Confidence: high on bytes, medium on the precise S_fft interpretation.**

---

## 3. Verification of Claim 3 — `psi_Y_full` aliases cleanly

**Supported, but not strictly proven from this dump alone.**

### What I looked for

Search for any rank-4 buffer carrying both a `nb_full`-class band dim (≈ 160, or padded 160–176) and a `r_chunk_local = 18332`-class r dim. The post-concat `psi_Y_full = jnp.concatenate(psi_Y_parts, axis=1)` would have per-rank shape `(nk, nb_full, ns, r_chunk_local) = (36, 160, 2, 18332)`, bytes = 36·160·2·18332·16 = **3.37 GB ≈ 3.14 GiB**.

### What I found

- `c128[36, 160, 2, 18332]` appears at line 90 — but **co-located in a 3.22 GiB band-FFT slot** (offset 82944010112, sharing lifetime with `c128[6, 32, 1125000]`). This is consistent with the orchestrator's claim that the post-concat (or one-side reshard slab) shares a band-FFT slot.
- `c128[36, 150, 2, 18332]` — would be the nb_R = 150 side (not nb_L = 150, my reading reverses since "Left wfns: (36, 150, 2, 1504)" in gw.out). Did not find this exact shape in the top-64 slots; could be in the "rest 137 values are less than 5% of the total size and not shown" (< 9.8 GiB cut-off).
- **No dedicated rank-4 slot** with size noticeably > 3.22 GiB for a separate psi_Y_full buffer.

### Caveat

If XLA's concat allocates a fresh buffer separate from psi_Y_bc slabs, it would be 3.14 GiB — close enough to the band-FFT slot size (3.22 GiB) that it might be aliased into one of those 58 slots OR it might sit in the < 5% rest-pool. The dump can't disambiguate without the `buffer-assignment.txt` companion file.

The practical implication for the model: there is **no separate psi_Y_full term** to add to W_wfn. The post-concat bytes are either fused into the band-FFT pool's lifetime or are folded into the per-bc reshard slabs that already share band-FFT slots. **Confidence: medium.** The claim is right for modeling purposes (no separate term needed), but "aliases cleanly" overstates what the dump alone proves. A `grep psi_Y_full` against `buffer-assignment.txt` would tighten this.

---

## 4. Issues found in `hlo_findings.md`

### 4.1. The `N_BC = 20` framing is misleading

`hlo_findings.md` line 12 says "at CrI3 with `band_chunk=16, nb_total≈310, psig_k_chunk_size=6`, the dump shows 58 slots of 3.22 GiB each". The math `58 ≈ 20 · 3` requires `N_BC = (nb_L + nb_R) / band_chunk = 20`, not the Python-loop count `N_BC = nb_full / band_chunk = 10`. The formula is correct under either framing if you keep `S_fft` consistent, but reading the report as written, a future engineer would expect to see 20 Python iterations and would be confused that the run only has 10.

**Fix suggestion:** rename `nb_total` to `nb_L + nb_R` (or `nb_sum`) and add a note that `N_BC_effective = (nb_L + nb_R) / band_chunk` because each Python bc-iter contributes both L and R reshard slabs. Use `nb_full` only for the actual loop count.

### 4.2. "band_chunk ↑ reduces N_BC linearly (the dominant lever)" is wrong

`hlo_findings.md` line 64–66 lists `band_chunk ↑` as a lever against W_wfn_actual. **This is incorrect under the formula given:** `W_wfn = (nb_L+nb_R) · S_fft · k_chunk · ns · n_rtot · 16` — band_chunk **does not appear**. The N_BC decrease (from larger band_chunk) is exactly cancelled by the per-iter FFT-box growth, leaving W_wfn independent of band_chunk.

The discussion below line 66 ("Net W_wfn_actual is monotone-decreasing in band_chunk only up to the point where band_chunk = nb_total… there's an interior minimum") implies a tradeoff that doesn't exist under the formula. Under the formula, **band_chunk is neutral against W_wfn**; only `psig_k_chunk_size` and `(nb_L + nb_R)` move the needle.

This matters for the picker algorithm: contrary to the orchestrator's framing, **the planner cannot reduce W_wfn by raising band_chunk**. The only knobs are:
- `psig_k_chunk_size` ↓ (linear reduction)
- the system-level `(nb_L + nb_R)` (fixed by physics)

Reducing N_BC via larger band_chunk does help against other peaks (the persistent reshards, the rank-5 lifetime overlaps inside `z_q_from_psi_sm`'s pair pipeline) but **not against the 58-slot FFT pool**.

**Fix suggestion:** in `hlo_findings.md §2`, replace the "band_chunk ↑ dominant lever" line with "band_chunk does not appear in `W_wfn_actual`; the only memory levers are `psig_k_chunk_size` and (immutable) `nb_L + nb_R`." Update Path A/B/C in §3 accordingly — Path A (`lax.fori_loop`) is the only one that actually reduces `W_wfn_actual` (collapses N_BC to 1).

### 4.3. Mesh size assertion implied but not stated

The dump's `c128[36,376,376]` L_q parameter implies μ_X = 376 = 1504/4, consistent with `p_x = 4`. The output `c128[36,94,73328]` shows μ_XY = 94 = 1504/16 → P = 16. Together: mesh = 4×4, 4 ranks per node times 4 nodes = 16. `hlo_findings.md` doesn't explicitly state the mesh used; readers comparing the slot sizes against expected sharding need to know it. **Fix suggestion:** add "mesh: 4×4 (P=16)" to the run-details section §5.

### 4.4. The "200 GiB per rank" framing vs the planner's per-rank budget

`hlo_findings.md` line 22 says "Total: 200.35 GiB per rank. Compare planner's HWM estimate: 51.96 GiB. Miss: 3.85×." This is the right number, but worth noting that on a 4×4 mesh **the planner expected 51.96 GiB total per device, and XLA tried to allocate 200.35 GiB on one device** — i.e. the OOM is per-device, not a global pool overflow. This is implicit in the run log but worth saying explicitly so readers don't think the model is off by P (16×) when it's actually off by 3.85×.

---

## 5. Confidence assessment

**Formula `W_wfn = (nb_L + nb_R) · S_fft · psig_k_chunk_eff · ns · n_rtot · 16` (with S_fft = 3):**

- **Supported by the dump for predicting total band-FFT pool bytes.** 187 GiB predicted vs 186.76 GiB observed — within 0.1%.
- **Independent of band_chunk** — this is mechanically correct given XLA's per-bc-iter slot-non-aliasing behavior in the Python-unrolled bc-loop. Contradicts the framing in `hlo_findings.md §2` text but matches the formula.
- **Caveats for future use:**
  1. The `(nb_L + nb_R)` factor is the right one (not `nb_full`). Document this in the source comment for the planner term.
  2. S_fft = 3 is empirical from one dump at one geometry. At different `psig_k_chunk_size`, the slot lifetimes might re-align differently — but the per-slot bytes scale by k_chunk, so the total should still be linear in k_chunk. **Test:** dump at a second geometry (e.g. MoS2 3×3, or CrI3 at psig_k_chunk_size = 3) and confirm 58 → roughly half (29 slots) at half k_chunk, OR 58 slots of half size. Either confirms the formula.
  3. The formula assumes the unsharded pathology persists across XLA versions. The fix-path-A (`lax.fori_loop`) would invalidate the formula entirely (would collapse to S_fft = 3, N_BC = 1, no `(nb_L + nb_R)` factor at all). When/if that fix lands, the formula needs to be replaced.

**Net:** I support the commit's formula as a model term, with two amendments to the surrounding `hlo_findings.md` prose:
- Make `nb_L + nb_R` (or `nb_sum`) the explicit variable, not `nb_total`.
- Remove the "band_chunk ↑ dominant lever" claim — the formula shows band_chunk is neutral against W_wfn.

The commit `ff5873c` on `agent/zeta-r-chunk-fixes-2026-05-13` is **directionally correct**. If the source comment for the new W_wfn term names the variable `nb_total` ambiguously (could be read as nb_full = 160 in this run), readers will under-predict by 2×. Verify the commit's variable naming uses the L+R sum, not nb_full.

Agent 1 HLO verification done
