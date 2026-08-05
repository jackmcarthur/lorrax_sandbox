# Memory model audit — per-peak component accounting

**Config (production):** CrI3 6×6×1 80 Ry SOC bispinor, 16 A100 hbm80g, 4×4 mesh.
`nk = nq = 36`, `ns = 2`, `nb = 150`, `n_rmu_padded = 1520`, `ngkmax = 59990`,
`n_rtot = 1,125,000`, `fft_grid ≈ (60, 60, 200)`, `p_xy = 16`, fft_box_factor = 4.

**Definitions for the byte tables:** all values per-rank, c128 = 16 B. "shard=P"
means divided by 16 across the 4×4 mesh; "shard=1" means replicated.

---

## `gflat_chunk_size` units

**Per-rank rows** of the flat `(n_q_disk · n_mu_local)` axis. Source:
`accumulate_rchunk_to_gflat._kernel` flattens per-rank arrays
`(n_q, n_mu_local, ...) → N = n_q · n_mu_local` rows and scans in chunks
of `cs`. Planner one-shot at line 383 is `ceil(nq_disk·mu/p_xy) = N` per-rank.
Runtime print (line 2422) confirms: `per-iter FFT box = cs · n_rtot · 16` —
no per-rank division. Consistent.

Production: `N = 36 · 95 = 3420` per rank. Planner picks `cs=707` → 5 iters.

---

## Peak A — band-chunked centroid load

**The actual path is `gflat_to_rmu`** (`load_wfns.py:474-479`), not the legacy
`to_rmu`. FFT inside `gflat_to_rmu._kernel.body` is over `(cs, ns, nx, ny, nz)`
— **no nk axis** (k is folded into the flat `(nk · nb_local)` row axis).

Per-rank live tensors at FFT peak:

| Term | Formula | CrI3 80Ry |
|---|---|---|
| `psi_G_flat` persistent input | `nk · nb / p_xy · ns · ngkmax · 16` | `0.65 GB` |
| `psi_rmu_band` output filling | `nk · nb / p_xy · ns · n_rmu · 16` | `16 MB` |
| FFT-box transient | `cs · ns · n_rtot · 16 · fft_factor` | `144 MB · cs` |
| 1D phase tables phx/phy/phz | `nk · (nx+ny+nz) · 16` | negligible |

**Planner formula vs reality — bug 1:**
```python
"fft_box": _bytes_c128(nk, band_chunk, ns, n_rtot, shard=p_xy) * fft_box_factor
```
Planner models batch as `nk · band_chunk` and divides by `p_xy`. The real
`gflat_to_rmu` body has `cs` batch rows (no separate nk·bc) and operates
inside a shard_map so the FFT box is already per-rank — no further `p_xy`
division applies.

**Numerical check:** at `band_chunk=32` planner gives
`36·32·2·1.125M·16·4/16 = 41.5 GB`. User report says Peak A = 21.45 GB,
consistent with `band_chunk=16`. At the **actual** kernel's `cs`, the FFT
box is `cs·2·1.125M·16·4`. Picking cs against budget would give `cs ~ 256`
→ FFT box ~37 GB, much less than the planner's nk-multiplied formula.

**Verdict A:** Planner over-estimates by factor `nk = 36`. Correct formula:
`cs · ns · n_rtot · 16 · fft_factor` (no nk multiplier; already per-rank).

---

## Peak B — CCT + Cholesky pre-loop

`c_q_from_psi_sm._local` (`isdf_fitting.py:300-350`):

| Term | Per-rank bytes | CrI3 |
|---|---|---|
| `psi_l_X` + `psi_r_X` (X-shard) | `2 · nk · μ/p_x · nb · ns · 16` | `0.66 GB` |
| `psi_l_Y` + `psi_r_Y` (Y-shard) | `2 · nk · nb · ns · μ/p_y · 16` | `0.66 GB` |
| `P_l`/`P_r` rank-7 (each) | `nk · ns² · μ²/p_xy · 16` | `333 MB` |
| `P_l_R_conj`/`P_r_R` (each) | same | `333 MB` |
| `C_q` final | `nq · μ²/p_xy · 16` | `83 MB` |
| `L_q` output | `nq · μ²/p_xy · 16` | `83 MB` |

**Planner formula** `2·M_cent + 2·M_P_open_spin + C_q + L_q` ≈ 1.4 GB,
matches code shape. Structurally far below A/C/D — never binding.
**Verdict B: correct.**

---

## Peak C — fit_one_rchunk (production binder)

`z_q_from_psi_sm._local` body (`isdf_fitting.py:565-734`). At natural pick
`r_chunk=21232`, `r_loc=21232/4=5308`, `mu_loc=1520/4=380`:

| Term | Formula | CrI3 |
|---|---|---|
| `psi_l_X`/`psi_r_X` persistent | `2 · nk · μ/p_x · nb_LR · ns · 16` | `~0.7 GB` |
| `P_l_acc`/`P_r_acc` rank-5 carry | `2 · nk · ns² · r_loc · μ_loc · 16` | `9.30 GB` |
| `psi_Y_bc_local` (pre-gather) | `nk · bpd_max · ns · n_zchunk · 16` | `49 MB` |
| FFT box (`to_rchunk_inner`) | `nk · bpd_max · ns · n_rtot · 16 · 4` | `~20.7 GB` |
| `psi_Y_bc_full_r` post-gather | `nk · P · bpd_max · ns · n_zchunk · 16` | `0.78 GB` |
| Tail P_l_3d/P_r_R rank-7 | `nk · ns² · r_loc · μ_loc · ns · 16` | each `~9.3 GB` |

**HLO evidence (Si 4×4×4 80Ry, module_0357, single-device, r_chunk=8366):**
total 21.08 GiB, **3 distinct preallocated-temp slot windows of 6.89 GiB each**.
Each slot contains `4× c128[64, 2, 4183, 432, 2]` (= 4× P-pair-shaped buffers
at the **same offset** — aliased uses of the same bytes, not concurrent).
Slots also house FFT boxes (`c128[64,8,2,24,24,24]`) and rank-7
`c128[2,4183,432,2,4,4,4]` (P_l_3d after the kx·ky·kz reshape) at the same
offset — confirming the docstring's claim that the FFT box aliases into
P-pair slots.

**Planner formula:**
`α_C = pair_density_slots · _bytes_c128(nk, ns², μ, shard=p_xy)`
with `slots=3`. At CrI3 r_chunk=21232: `3 · 36 · 4 · 1520 · 21232 · 16 / 16 ≈
14 GB`. The HLO slot count (3) matches; the per-slot 4× aliasing is internal
to one slot, so the planner is right to count one P-pair per slot.

**Verdict C:** α_C structurally correct. Two minor over-counts in the
`centroids_persist` constant: planner divides by `p_xy` but psi_l_X/psi_r_X
shard only `μ` on `'x'` (not `('x','y')` product), and planner uses `nb_total`
instead of `nb_left`/`nb_right`. Net effect: low-single-GB over-count, not
binding. CrI3 4×4 mesh slot count needs **explicit re-verification** (see
§"pair_density_slots" below).

---

## Peak D — accumulate_rchunk_to_gflat

`accumulate_rchunk_to_gflat._kernel.body` (`wfn_transforms.py:1057-1107`).
FFT shape is **`(cs, nx, ny, nz)` — NO ns axis** (ζ is spin-traced upstream;
line 1095 confirms: `box = buf.reshape(cs, nx, ny, nz)`).

| Term | Formula | At cs=707 |
|---|---|---|
| `gflat_acc` persistent | `n_q_disk · μ/p_xy · ngkmax · 16` | `3.28 GB` |
| `zeta_chunk` transient | `n_q_disk · μ/p_xy · r_chunk · 16` | `1.16 GB` (r=21232) |
| `buf` / `box` FFT input | `cs · n_rtot · 16` | `12.7 GB` |
| `G` post-FFT (may alias buf) | `cs · n_rtot · 16` | `12.7 GB` |
| cuFFT scratch (out-of-place 3D) | `~1× buf` | `~12.7 GB` |

**Planner formula:** `cs · n_rtot · 16 · 4 = 50.9 GB` at cs=707 — the
"phantom" 51 GB the user calls out.

**Verdict D — the big bug:** `fft_box_factor=4` was calibrated for the
*centroid-load* 3D IFFT (4 staging buffers + output validated in
`MEMORY_MODEL.md` against MoS2/Si shards `(nk, B_b/P, ns, n_r)`). The
accumulate kernel is a different FFT — `(cs, nx, ny, nz)` with `cs` as a
small batch and the spatial axes huge — and cuFFT's out-of-place 3D fftn
needs ~2× the box, not 4×.

**Empirical anchor:** runtime debug log
"chunk_size=360 → per-iter FFT box 6.48 GB/rank" = `360·1.125M·16` = bare
1× term. Planner predicts `4× = 25.9 GB` for the same cs. So 4× is
**conservative by ~3×**. A defensible factor is **2.0–2.5**.

At cs=707, factor=2: total Peak D ≈ `3.28 + 1.16 + 2·12.7 = ~30 GB`
(vs planner's 55 GB and a likely runtime ~17 GB). At factor=2 the planner
would also size cs much larger — possibly one-shot.

---

## `fft_box_factor` analysis

- **Peak A:** factor applied (`fft_box` term). Validated in `MEMORY_MODEL.md`
  against MoS2/Si — 4× exact on `(nk, B_b/P, ns, n_r)` shards >0.3 GB.
  **Right.**
- **Peak C:** factor NOT included in α_C — XLA aliases the FFT box into
  P-pair slots, verified in HLO. **Right.**
- **Peak D:** factor applied to `(cs, n_rtot)` FFT. **Wrong** — runtime
  evidence shows ~1×, not 4×.

**Recommendation:** per-peak factor: `factor_A = 4.0`, `factor_D = 2.0`.

---

## `query_fft_peak_bytes` integration

Currently unused by the planner. Replaces hand-tuned `fft_box_factor` with
per-shape `compiled.memory_analysis()` measurement.

- **Pros:** exact within XLA BufferAssignment; solves the A-vs-D factor
  mismatch automatically; per-shape so different FFT contexts get
  different scratch estimates.
- **Cons:** ~100ms–1s per unique FFT shape compile (one-shot at planner
  startup, amortized; ≤5 s total). The AOT path lowers a **standalone**
  FFT jit — real FFTs run inside `shard_map(scan(body))` and XLA may
  schedule scratch differently in that nested context. Could under-predict
  by 10–20% for the accumulate kernel.

**Recommendation:** wire into Peak A and Peak D, fallback to factor when AOT
compile fails (fft_helpers.py:85-100 already has 3× fallback).

---

## `pair_density_slots = 3` — re-verification

Docstring (`gflat_memory_model.py:177`) says "Verified on MoS2 3×3 bispinor
/ 2×2 mesh." **Not re-verified on CrI3 80Ry / 4×4 mesh.**

Closest dump: Si 4×4×4 80Ry single-device
(`runs/Si/08_4x4x4_sym_vs_nosym_2026-05-14/run_sym_hlo_dump_2026-05-15/xla_dump/module_0357...memory-usage-report.txt`).
Shows **3 distinct preallocated-temp slot windows × 4× P-pair-shaped buffers
each at the same offset** (aliased). The slot count is 3 (matches planner);
per-slot 4× is internal aliasing.

**To re-verify on CrI3 4×4 mesh:** rerun with
`XLA_FLAGS="--xla_dump_to=$PWD/hlo --xla_dump_hlo_pass_re=memory-usage-report"`,
find the largest `module_NNNN.jit__kernel...memory-usage-report.txt` whose
preallocated-temp contains `c128[36, 2, 2, μ_local, r_loc]` shapes, count
distinct offset windows. Si has nk=64 with mu replicated; CrI3 has nk=36
with mu sharded on `'x'` — buffer-assignment may pack differently.

---

## Top 3 specific bugs / over-conservative estimates

1. **`fft_box_factor=4` on Peak D's accumulate FFT
   (`gflat_memory_model.py:222`).** Runtime shows the (cs, n_rtot) 3D FFT
   needs ~1× (the bare box bytes), planner predicts 4×. Empirical: at cs=360
   the runtime box is 6.48 GB/rank vs planner 25.9 GB. Halving the factor for
   Peak D frees ~25 GB phantom budget — would let the planner take cs much
   larger (likely one-shot) and stop limiting r_chunk via Peak D.

2. **Peak A FFT-box uses `nk · band_chunk` batch dim
   (`gflat_memory_model.py:139`).** Actual `gflat_to_rmu` kernel batches on
   the flat `(nk · nb_local)` axis with chunk size `cs`. Planner over-counts
   by factor ~nk = 36 for the FFT-box term. The user observed Peak A
   ramping with `band_chunk` because the *planner* models it that way, not
   because the kernel does. Correct formula: `cs · ns · n_rtot · 16 · 4`
   (cs replaces nk·bc; already per-rank inside shard_map so no further
   division).

3. **`pair_density_slots=3` unverified on CrI3 4×4 mesh
   (`gflat_memory_model.py:244-245`).** Production is CrI3 6×6 80Ry bispinor
   (charge + 3 transverse channels). Existing verification only at MoS2 3×3
   / 2×2 mesh; Si 4×4×4 single-device dump consistent (3 slots × 4× aliased
   uses) but mu-sharding differs. Action: grab a memory-usage-report from
   the next 4×4-mesh production run and confirm slot count.

**Bonus over-conservative:** `target_utilization = 0.80` compounds with
bugs 1+2. Si runtime peak landed within 5% of planner HWM (γ ≈ 0.95) so the
planner is slightly **under** at Si but clearly **over** at CrI3 due to
bugs 1+2. Don't change `target_utilization` until those are fixed.
