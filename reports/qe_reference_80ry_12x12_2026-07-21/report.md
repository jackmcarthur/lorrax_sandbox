# Converged MoS2 QE reference — 80 Ry / 12×12×1 / 400 bands, and the GW sized to sit on it

**Run dir**: `runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/`
**Date**: 2026-07-21 · **Platform**: Perlmutter, own allocation `56276468` (4 nodes / 16× A100-40GB, `qe80ry-jackm`)
**Motivation (owner)**: *"i feel like i'm fighting a lot of uphill battles because i started with too many convergence parameters which were too small."*

Every prior MoS2 run in this sandbox is 30 Ry. This is the redo: a from-scratch QE
reference at production cutoff, built per `skills/build_inputs/SKILL.md` with **no reuse
and no symlinking** of any stale output (SCF, NSCF, pw2bgw all rebuilt), plus a
memory/cost sizing of the downstream GW — done with LORRAX's own production planner —
*before* any 16-GPU-hours are committed.

**Stage 1 = §1–3 (the reference). Stage 1b = §4–7 (sizing only; no GW was run).**

---

## 1. What was built

| | |
|---|---|
| Structure | unchanged from every prior MoS2 run (`templates/scf.in` cell + positions) — the 80 Ry reference is directly comparable to the 30 Ry history |
| Pseudopotentials | `assets/pseudopotentials/standard/{Mo,S}.upf` (FR-ONCVPSP PBE, PseudoDojo). md5 `4e1c3579…` / `a7319d53…` — **byte-identical** to those `A_bse_figures_2026-07-20` used |
| N_val | 26 (Mo `z_valence=14` + 2 × S `z_valence=6`) |
| SCF | `ecutwfc = 80.0` Ry, `ecutrho = 320.0` Ry (QE default 4×; `templates/scf.in` sets no `ecutrho` and 4× is the norm-conserving convention), `K_POINTS automatic 12 12 1`, `nbnd = 40`, `conv_thr = 1e-10` |
| NSCF | same cutoffs, `K_POINTS crystal` **144 points** = full unshifted 12×12×1 BZ, `nbnd = 400` |
| Spinor | `noncolin`, `lspinorb`, `no_t_rev`, `assume_isolated = '2D'` — all `.true.`/set |
| pw2bgw | **two** calls: (a) `wfng + vxc + kih` → `WFN`/`vxc.dat`/`kih.dat`; (b) `rhog` only → `RHO`. Split deliberately so a `rhog` failure could not cost the 15.6 GB WFN write. |
| Products | `qe/nscf/{WFN.h5 (15.65 GB), vxc.dat, kih.dat, RHO}` |

`manifest.yaml` validated to parse with `yaml.safe_load` (a recent run shipped invalid
YAML; this one is checked).

Byte-for-byte the k-list is the uniform i/12, j/12 grid, regenerated from scratch (not
copied from `04_mos2_12x12_bands_2026-07-18`), 144 unique rows, K = (⅓,⅓,0) present.

---

## 2. Verification of `WFN.h5` — **ALL PASS**

Read with LORRAX's own `file_io.mf_header` reader (no ad-hoc HDF5 parsing);
script `verify_wfn.py`, log `verify_wfn.log`.

| quantity | value | required | |
|---|---|---|---|
| `mnband` | **400** | ≥ 400 | PASS |
| `ecutwfc` | **80.0000 Ry** | 80 | PASS |
| `ecutrho` | 320.0000 Ry | 4× | PASS |
| `kgrid` | **(12, 12, 1)** | 12×12×1 | PASS |
| `nrk` | **144** | full BZ | PASS |
| `shift` | (0, 0, 0) | unshifted | PASS |
| K = (⅓,⅓,0) | file k-point **#52**, \|Δk\| = 4.7e-10 | sampled | PASS |
| `nspin`/`nspinor` | 1 / 2 | spinor | — |
| **`FFTgrid`** | **(36, 36, 135) → n_rtot = 174 960** | — | drives GW memory |
| `ngkmax` | **8603** (ngk 8436 … 8603, mean 8483) | — | |
| `ng` (ρ sphere) | 67 737 | — | |
| cell volume | 702.2012 bohr³ | — | |
| sym ops `ntran` | 2 | `no_t_rev` | — |

**n_rtot grew 3.80×** vs the 30 Ry grid (24×24×80 = 46 080). That single number is what
makes the downstream GW expensive — see §5.

Physics sanity: at K, `ifmax = 26` (= N_val, correct), direct DFT gap **1.6954 eV**
(80 Ry) vs **1.7322 eV** measured on the 30 Ry 6×6 WFN — i.e. the cutoff alone moves the
K-point gap by **−36.8 meV**. SCF total energy −184.87288559 Ry, converged in 10
iterations; HOMO/LUMO −5.2454 / −3.5501 eV.

---

## 3. Wall-times (measured, from the real logs)

| step | geometry | wall | note |
|---|---|---|---|
| SCF | 1 node, 4 GPU, `-npools 4` | **16.85 s** | 10 iterations to `conv_thr 1e-10` |
| NSCF | 4 nodes, 16 GPU, `-npools 16` | **29.06 s** | 144 k / 400 bands, ethr 3.85e-13, avg 20.9 Davidson its, max 1.09 GB/proc |
| pw2bgw (WFN+vxc+kih) | 1 rank, CPU (`MPICH_GPU_SUPPORT_ENABLED=0`) | **606 s** | `write_wfng` 68 s, `write_vxc_g` 239 s, `write_kih` ≈ 290 s |
| pw2bgw (RHO) | 1 rank | 35 s | |
| wfn2hdf | 1 rank | 121 s | 15.65 GB BIN → HDF5, 118 s conversion |
| **total pipeline** | | **833 s = 13 min 53 s** | |

**The DFT is 46 s of that 833 s.** k-pool parallelism is doing exactly what it should
(`-npools 16` → 9 k-points/GPU, no inter-GPU communication). The other 94 % is
single-rank serialization: `pw2bgw` computing 144 × 400 = 57 600 `vxc`/`kih` diagonal
matrix elements on one MPI rank, and `wfn2hdf` streaming 15.6 GB. Both are inherently
serial in the current pipeline and neither benefits from more nodes.

Disk: `qe/nscf` = 58 GB (WFN binary 15.65 + WFN.h5 15.65 + `.wfc*` scratch + save dir).
Quota is 4.3 T of 20 T — no pressure.

---

## 4. GW sizing — method, and the control that validates it

Sizing uses **LORRAX's own production planner**, `gw.gflat_memory_model.plan_gflat_chunks`
— the identical call `gw.gw_init` makes at the top of the ISDF pipeline, with the same
argument pattern (`nb_total = (b3−b0)+(b4−b1)`, `n_q_disk = nk_tot`). It is fed a real
`common.meta.Meta` built from the measured WFN.h5 header, and run on a **real 4×4 GPU
mesh (16 GPUs / 4 nodes)** so the Stage-A/D FFT box is XLA-queried (true cuFFT plan
scratch) rather than falling back to the analytic factor. Script `size_gw.py`.

### Control: reproduce a known production plan

Before trusting any new number, the driver was pointed at the **6×6 WFN of the already
completed** `A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu` run and asked to reproduce
that run's printed plan. Result (`control_6x6_prodfaithful.log`):

| field | production `gw.out` | this driver |
|---|---|---|
| `band_chunk` | 16 | **16** |
| `r_chunk` | 32720 (2 chunks) | **32720 (2 chunks)** |
| `q_chunk` | 36 | **36** |
| `gflat_cs` | 100 | **100** |
| `P_min` | 1 | **1** |
| persistent | 0.76 GB/dev | **0.76 GB/dev** |
| **HWM** | **23.80 GB/dev** | **23.80 GB/dev** |
| binder | `C_fit_one_rchunk` | **`C_fit_one_rchunk`** |
| stage peaks C/D/B/A/E | 23.80 / 2.68 / 1.50 / 0.98 / 0.53 | **23.80 / 2.68 / 1.50 / 0.98 / 0.53** |

Every field reproduces. Two production conventions had to be matched to get there, and
both are worth recording:

1. **`gw_init` never sees the real `ngkmax`.** It does
   `_ngkmax = int(getattr(meta, 'ngkmax', 0)) or int(0.06 * meta.n_rtot)`, but
   `common.meta.Meta` has **no `ngkmax` field** and nothing in the tree ever assigns
   `meta.ngkmax` — so the `getattr` branch is dead and the `0.06·n_rtot` heuristic is
   what is *always* used. Here that is **10497 vs the true 8603** (+22 % conservative);
   on the 6×6 anchor 2764 vs 1964 (+41 %). The sizing below uses the heuristic, i.e. it
   predicts what `gw_jax` will actually do. *Follow-up: bind `ngkmax` onto `Meta` (the
   value is right there in the mf_header) and delete the heuristic.*
2. **`band_chunk_size` defaults to `16`, not `0`**, in `gw_config` — so the planner's
   `band_chunk` picker is overridden in every production run unless the user opts out.

Naive first pass (true `ngkmax`, no band-chunk override) gave `band_chunk = 256` and
`r_chunk = 32784` instead — a 0.2 % HWM difference, but the discrepancy was traced to
these two causes exactly (arithmetic reproduces both branches) rather than waved away.

### GW being sized

`nval = 26`, `nband = 326` (screening sum-over-states), `nq = nk = 144`, `nspinor = 2`,
`is_bispinor = False`, `n_rtot = 174 960`. Two Σ-window scenarios, because
`nb_total = (b3−b0)+(b4−b1)` — the thing that sizes the four ψ centroid copies — depends
on the Σ QP window (`b3 = nelec + ncond`) as well as the screening sum:

* **wide Σ** `ncond = 300` → QP for bands 1..326, `nb_total = 662` (the owner's literal spec)
* **narrow Σ** `ncond = 74` → QP for bands 1..100, `nb_total = 436` (what the 6×6 runs use)

---

## 5. GW sizing — results (16 devices, 4×4 mesh)

`n_μ` padded to a multiple of 16. "fits?" = HWM ≤ `util·budget` (util 0.85 for nspinor=2)
**and** `P_min ≤ 16`. **The binder is `C_fit_one_rchunk` in every single case.**

### 16 × A100-**40 GB** (`memory_per_device_gb = 28`, target 23.8 GB/dev)

| Σ window | n_μ | HWM/dev | persistent | P_min | r_chunk (#) | CCT stack | ζ(G) disk | fits? |
|---|---|---|---|---|---|---|---|---|
| wide (662) | 1600 | **23.77 GB** | 7.67 | 4 | 5376 (33) | 5.90 GB | 38.7 GB | **YES** |
| wide | 2400 | 24.00 GB | 11.78 | 8 | 2720 (65) | 13.27 GB | 58.0 GB | no (0.8 % over) |
| wide | 3000 | 31.96 GB | 15.03 | 9 | 3008 (59) | 20.85 GB | 72.8 GB | **NO** (114 % of budget) |
| wide | 4000 | 50.50 GB | 20.55 | 15 | 4000 (44) | 36.86 GB | 96.7 GB | **NO** (180 %) |
| narrow (436) | 1600 | 23.78 GB | 6.00 | 3 | 5936 (30) | 5.90 GB | 38.7 GB | **YES** |
| narrow | 2400 | **23.80 GB** | 9.28 | 6 | 3232 (55) | 13.27 GB | 58.0 GB | **YES** (exactly at target) |
| narrow | 3000 | 28.83 GB | 11.89 | 8 | 3008 (59) | 20.85 GB | 72.8 GB | **NO** |
| narrow | 4000 | 46.34 GB | 16.39 | 12 | 4000 (44) | 36.86 GB | 96.7 GB | **NO** |

### 16 × A100-**80 GB** (`memory_per_device_gb = 60`, target 51 GB/dev)

| Σ window | n_μ | HWM/dev | persistent | P_min | r_chunk (#) | fits? |
|---|---|---|---|---|---|---|
| wide | 1600 | 50.99 | 7.67 | 2 | 14464 (13) | **YES** |
| wide | 2400 | 50.96 | 11.78 | 3 | 8720 (21) | **YES** |
| wide | 3000 | 50.97 | 15.03 | 4 | 6384 (28) | **YES** |
| wide | 4000 | **50.98** | 20.55 | 6 | 4064 (44) | **YES** (at the edge) |
| narrow | 4000 | 50.89 | 16.39 | 4 | 4608 (38) | **YES** |

ζ(r) is never resident in full — it is 645 GB (n_μ=1600) to 1612 GB (n_μ=4000) across all
q; the r-chunk loop holds 15–37 GB aggregate (0.9–2.3 GB/dev) at a time.

### The memory wall, in closed form

Stage C carries a **performance floor `r_chunk ≥ n_μ`** (`r_lo = min(mu, n_rtot)`) that
overrides the budget. So below a certain `r_chunk` the planner simply *cannot* chunk
further, and the HWM floor is quadratic in n_μ:

```
HWM_floor(μ)  =  [3·nk·ns² + nq]·16·μ²/P     (Stage C at r_chunk = μ)
               + nq·16·μ²/P                   (L_q, the CCT stack)
               + nq·ngkmax·16·μ/P             (gflat_acc)
               + nk·ns·nb·16·μ·(1/p_x + 1/p_y)  (ψ centroid copies)
             =  2016·μ²  +  4.562e6·μ   bytes      (wide Σ, P = 16, this system)
```

Setting that equal to the 23.8 GB target gives the hard ceiling on 16 × 40 GB:

| Σ window | 40 GB wall | 80 GB wall |
|---|---|---|
| wide (nb_total 662) | **n_μ ≈ 2490** | n_μ ≈ 4030 |
| narrow (nb_total 436) | **n_μ ≈ 2670** | n_μ ≈ 4300 |

which reproduces the table (2400 ≈ at the wall, 3000 and 4000 well past it; 4000 lands at
50.98 vs the 51 GB target on 80 GB). **This is a wall, not a knob** — no chunk size,
`gflat_chunk_size`, or utilization tweak moves it; only more devices (`P`), a narrower Σ
window, or bigger cards do.

### Where the docs' centroid guidance lands

`docs/docs_gwjax/COHSEX_INPUT.md` and `skills/build_inputs/SKILL.md` both say
**8 × nband for standard accuracy, 12 × for production** (not 10×):

| guidance | n_μ | 16 × 40 GB | 16 × 80 GB |
|---|---|---|---|
| 8 × 326 | **2608** | **past the wide-Σ wall (2490)**; marginal narrow-Σ (2670) | fits comfortably |
| 10 × 326 (as phrased in the brief) | 3260 | no | fits |
| 12 × 326 | **3912** | no | fits, at the edge |

So: **the documented production centroid count for nband = 326 does not fit on 16 ×
A100-40 GB.** It fits on 16 × A100-80 GB.

---

## 6. Wall-time extrapolation

Anchor: the measured `02_lorrax_gw_d3h_16gpu` 6×6 run — 16 GPU, 30 Ry, nq = nk = 36,
n_rtot = 46 080, n_μ = 1496 (pad 1504), nb_total = 308, ngkmax = 1964, **87.414 s total**,
with its per-section `--- Timing ---` breakdown. Each section scaled by its leading
algorithmic term (device count held at 16, so P cancels):

`cholesky, χ0/W: nq·μ³` · `ζ solve: nq·μ²·n_rtot` · `z_q_build: nk·μ·n_rtot·nb` ·
`V_q: nq·μ²·ngkmax` · `Σ: nk·nq·μ²·nb` · `load/misc: nk·μ·nb`

Script `extrapolate_walltime.py`, log `walltime_extrapolation.log`.

**Wide Σ (nb_total = 662), 16 GPU / 4 nodes:**

| section | n_μ=1600 | 2400 | 3008 | 4000 |
|---|---|---|---|---|
| ζ solve | 561 s | 1261 | 1981 | 3503 |
| **Σ** | **1167 s** | **2625** | **4123** | **7291** |
| z_q_build | 207 | 311 | 389 | 518 |
| V_q | 97 | 217 | 341 | 603 |
| cholesky | 29 | 97 | 191 | 450 |
| χ0/W + load + misc | 65 | 115 | 168 | 289 |
| **TOTAL** | **35 min** | **1 h 17** | **1 h 59** | **3 h 30** |
| node-hours (4 nodes) | 2.4 | 5.1 | 8.0 | 14.1 |

**Narrow Σ (nb_total = 436):** 27 min / 59 min / 1 h 33 / 2 h 45 (1.8 / 4.0 / 6.2 / 11.0
node-hours).

Σ dominates because the k×q double sum grows 16× (36² → 144²) while μ² and nb also grow.
Everything at n_μ ≤ 3008 fits inside the 4 h `interactive` QOS wall; n_μ = 4000 wide-Σ at
3 h 30 does not leave enough margin and should go to a `regular` sbatch. Treat these as
±2× — they ignore the growth from 2 to 33–65 r-chunks (per-chunk launch + HDF5 write
overhead) and the 39–97 GB of ζ(G) now being written to Lustre.

---

## 7. Recommendation

**Recommended centroid count for a 16-GPU campaign: `n_μ = 2400`, on 16 × A100-80 GB.**

* **HWM 50.96 GB/dev** = 85 % of the 60 GB budget, comfortably inside the 80 GB card.
* **≈ 1 h 17 (wide Σ) / 59 min (narrow Σ)** → 4–5 node-hours per GW. Cheap enough to run
  the ISDF-convergence ladder rather than argue about it.
* 2400 = **7.4 × nband**, just under the docs' 8× "standard" and a genuine step up from
  the 1496 (7.5 × nband=200) the 6×6 runs used — but at 4× the q-points and 3.8× the real-space grid.
* Then **confirm convergence at n_μ = 3008 (9.2×, ≈ 2 h) and 4000 (12.3×, ≈ 3 h 30)** on
  the same 80 GB nodes. That brackets the docs' 12× production guidance and gives a real
  n_μ convergence curve — which is exactly the "stop fighting under-converged parameters"
  outcome the owner asked for.

**If only 40 GB nodes are available** (the default `lxalloc`): `n_μ = 1600` is the only
safe choice with the wide Σ window (23.77 GB, 85 %, ≈ 35 min). `n_μ = 2400` is reachable
*only* by narrowing the Σ window to `ncond = 74` (23.80 GB — exactly at target, zero
margin). Do not attempt 3000 or 4000 on 40 GB: 114 % and 180 % of budget.

Get 80 GB nodes with `--constraint="gpu&hbm80g"` and set `memory_per_device_gb = 60`.
Per `feedback_lxalloc_gpu_constraint_mixes_hbm`, plain `lxalloc` mixes 40/80 GB at
random, so this must be an explicit raw `salloc` with `-J lx-alloc-$USER`.

### Caveats to carry forward

* **Conditioning is no longer the constraint, memory is.** With `zeta_rcond = 1e-6`
  rank-truncation (default since 2026-07-21) over-completeness is safe, so raising n_μ is
  purely a memory/time decision. Confirmed by design; not re-measured here.
* **The planner under-predicts the true BFC-arena peak by ~14 %** (`project_planner_conservative_8x`).
  A predicted 23.8 GB is really ~27 GB — which is why `memory_per_device_gb = 28` on a
  40 GB card is the right convention and why the 2400/40 GB "0.8 % over" row should be
  read as *no*, not *almost*.
* `n_μ = 2400` at 40 GB misses by 0.8 % partly because `gw_init` hard-codes
  `max_chunks = 64`, which floors `r_chunk` at 2734 when the budget wanted 2675. Raising
  that cap would make the row fit on paper — but see the previous bullet; it should not
  be used to squeeze past the wall.
* **`Meta` has no `ngkmax` field**, so the planner always uses `0.06·n_rtot` (+22 % here).
  Worth fixing; it makes every plan slightly conservative in the persistent term.
* **Disk**: ζ(G) is 38.7 GB (n_μ=1600) → 96.7 GB (n_μ=4000) per GW run, on top of the
  15.65 GB WFN.h5. Quota headroom is fine (4.3 T of 20 T) but variants add up fast.

---

## 8. Artifacts

| file | what |
|---|---|
| `runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/manifest.yaml` | run manifest (YAML-validated) |
| `.../qe/{scf,nscf}/*.in`, `qe/run_qe.sh`, `qe/run_qe.log` | inputs + runner + timing log |
| `.../qe/nscf/{WFN.h5, vxc.dat, kih.dat, RHO}` | **the reference** |
| `verify_wfn.py` / `verify_wfn.log` | WFN.h5 acceptance checks (ALL PASS) |
| `size_gw.py` / `size_gw.log` / `gw_sizing.json` | planner-driven sizing, 2 windows × 2 budgets × 4 n_μ |
| `control_6x6_prodfaithful.{log,json}` | the control reproducing the 6×6 production plan |
| `control_6x6.{log,json}` | naive first pass (true ngkmax, planner-picked band_chunk) |
| `extrapolate_walltime.py` / `walltime_extrapolation.log` | 6×6 → 12×12 cost extrapolation |
| `run_sizing.sh` | shifter launcher for the two sizing steps |

Code used for the estimate: `sources/lorrax_D` @ `a506a71` (`agent/suite-speedup`);
`gw/gflat_memory_model.py` there is byte-identical to the worktree the 6×6 production run
used, so the control is a like-for-like comparison. **No LORRAX source was modified and
no GW was run.**
