# Phase 0 — n_rmu padding HLO baseline (2026-05-08)

Goal: dump compiled HLO for ζ-fit, V_q tile, χ₀, and W-solve at the
**known-good mesh-divisible** centroid count, enumerate every collective,
and decide which jit boundaries the upcoming PadMeta helper must wrap.

Branch: `agent/v_q_bispinor` on lorrax_A (HEAD `5a3b0af`).

## TL;DR

* **Phase 0a (bispinor smoke, n_rmu_C=640, n_rmu_T=656, 4×4 mesh, 16 GPUs):
  120 collectives total, 0 unexpected.**  Every all-gather-start /
  all-to-all sits at a known reshard between intentional shardings —
  load-bearing, not waste.  The kernel input shardings (μ-axis on the
  product mesh ('x','y')) divide cleanly because 656=16·41 and 640=16·40.
* **Phase 0b (4×4 GN-PPM, do_screened=true, n_rmu=640): 66 collectives,
  also 0 unexpected.**  Critically, the W-solve V/χ reshard
  (`w_isdf.py:235/243/244/245`) compiles to **pure all-to-alls** —
  no all-gathers, no rematerialization.  The two-step reshard
  `P(None,'x','y')` → `P('x',None,'y')` → `P(('x','y'),None,None)`
  works exactly as the comment block at `w_isdf.py:215-223` advertises:
  "two single-axis all_to_alls instead of Involuntary full
  rematerialization."
* Padding-helper scope is therefore **narrow and well-defined**: the only
  jit-input layout that demands product-axis divisibility on the μ axis is
  the V_q tile kernel's ζ-disk read at `P(None, None, ('x','y'))`.  The
  ζ-fit interior, χ₀ interior, and W-solve interior all run on
  single-axis μ shardings (`P(None, 'x', ...)` or `P(..., 'y')`), which
  divide on n_rmu / max(p_x, p_y) — that's a much weaker constraint and
  most centroid counts already meet it.

## Phase 0a — bispinor smoke

Run: `runs/MoS2/00_mos2_3x3_cohsex/A_phase0_hlo_2026-05-08/`.
Re-run of `A_bispinor_smoke_2026-05-08/cohsex.in` with
`run_profiled.py` and `XLA_FLAGS=--xla_dump_to=./profile/xla_dump`.
Expected post-V_q failure: `head_correction.py` `RuntimeError:
Failed to resolve q=0 Coulomb head` (pre-existing for `x_only=true` per
the handoff).  ζ-fit + 7 V_q tiles ran cleanly before the failure;
HLO is captured.

* **Modules dumped:** 805
* **Sum of per-module peak HBM:** 48.96 GiB (upper bound across all
  modules; per-module peak listed below).
* **Top per-rank module peak:** 1.77 GiB on the V_q tile kernel
  (`module_05XX.jit__kernel`).

### All collectives by source

| Count | Op | Source | Note |
|------:|----|--------|------|
| 24    | all-gather-start | `isdf_fitting.py:738` | C_q reshard (160→640 along μ; 164→656); single→full for the Cholesky |
| 12    | all-gather-start | `isdf_fitting.py:790` | `_reshard_z` Z_q P(None,'x','y') → P('x',None,'y') stage 1 |
| 12    | all-to-all       | `isdf_fitting.py:791` | `_reshard_z` stage 2 → P(None,None,('x','y')) for ZCT solve |
| 12    | all-gather-start | `load_wfns.py:429`    | `_reshard_rchunk` ψ-rchunk band-axis stage 1 |
| 12    | all-to-all       | `load_wfns.py:433`    | `_reshard_rchunk` stage 2 → P(None,None,None,'y') for pair-density |
| 12    | all-gather-start | `v_q_tile.py:703`     | ζ_L_G P(None,('x','y'),None) → P(None,'x',None) (μ-axis on L) |
| 12    | all-gather-start | `v_q_tile.py:704`     | ζ_R_G P(None,('x','y'),None) → P(None,'y',None) (μ-axis on R) |
|  8    | all-gather-start | `v_q_tile.py:676`     | tiny (40→160) — `g0_blk = zeta_box[...,0]` slice cleanup |
|  4    | all-gather-start | `load_wfns.py:745`    | ψ load reshard (band 5×4 → 20) |
|  4    | all-to-all       | `load_wfns.py:752`    | ψ load second-stage all-to-all |
|  4    | all-gather-start | `tagged_arrays.py:294`| restart-h5 V_q reshard before write |
|  2    | all-gather-start | `tagged_arrays.py:309`| restart-h5 g0 reshard before write |

Volume per call: peaks at ~73 MiB at `isdf_fitting.py:738` (CCT solve
input) and ~71 MiB at `v_q_tile.py:703-704` (V_q kernel input
reshard).  All are **expected** product↔single-axis transitions.

### V_q tile kernel boundary (the bispinor-relevant one)

`_kernel` in `gw/v_q_tile.py:733` declares::

    @partial(jax.jit,
             in_shardings=(V_sh, g0_sh, zeta_disk_sh,
                           v_per_G_sh, phase_sh, rep, rep, rep),
             ...)

with `zeta_disk_sh = P(None, None, ('x','y'))`.  The ζ slab arrives
with **product-axis sharding on the μ-axis** — top-level uneven
sharding is not supported (JAX dev `2026-01-30`), so the call is only
valid when `mu_size % (p_x · p_y) == 0`.  At n_rmu_T=656 that's
656/16=41 ✓.  The 668 case fails here.

Inside the kernel, the all-gather-start at `:703-:704` resharrs to
`P(None,'x',None)` / `P(None,'y',None)` — a single-mesh-axis layout.
This intermediate IS allowed to be uneven inside a jit, but the **input**
must be product-divisible.

### V_q tile output

`V_acc` returns sharded `P(None, 'x', 'y')` — single-axis on each
μ-dim.  Per-axis divisibility: 656/4=164 ✓, 640/4=160 ✓.  Even at
n_rmu=668 that satisfies single-axis (668/4=167); the divisibility
problem is purely the input boundary.

## Phase 0b — χ₀ / W-solve

Run: `runs/MoS2/01_mos2_4x4_cohsex_gnppm/A_phase0_hlo_chi0w_2026-05-08/`.
Mesh 4×4 (16 GPUs).  Re-runs `01_lorrax_gnppm/cohsex.in`
(do_screened=true) under the same `run_profiled.py` harness.  Expected
post-W failure: `phdf5 async write: dataset rank mismatch ds=/W0_qmunu
file_rank=3 write_rank=8` — the rank-3 placeholder vs rank-8 W bug
listed in the handoff as "pre-existing post-V_q failure unrelated to
padding."  ζ-fit + V_q + χ₀ + W-solve all ran cleanly before the
write failure; HLO captured.

* **Modules dumped:** 325
* **Sum of per-module peak HBM:** 23.25 GiB
* **Total collectives:** 66 (42 all-gather-start + 24 all-to-all)

### W-solve collectives (all clean)

| Count | Op | Source | Note |
|------:|----|--------|------|
| 3 | all-to-all | `w_isdf.py:235` | `chi_scaled = pref * chi_flat` reshard (intermediate) |
| 3 | all-to-all | `w_isdf.py:243` | V_q P(None,'x','y') → P('x',None,'y') stage 1 |
| 3 | all-to-all | `w_isdf.py:244` | V_q stage 2 → P(('x','y'),None,None) for q-LU |
| 3 | all-to-all | `w_isdf.py:245` | χ_q stage 2 (mirror of line 244) |

Each ~6.6 MiB (40-row slabs on 16 ranks).  **Zero all-gathers — the
two-step reshard via `P('x',None,'y')` intermediate works exactly as
the comment at `w_isdf.py:215-223` claims.**  This validates the user's
existing memory of the design.

### χ₀ + V_q tile + ζ-fit collectives — same pattern as 0a, scaled

Phase 0b's V_q tile (`v_q_tile.py:703/704/676`) and ζ-fit
(`isdf_fitting.py:738`) show the same intentional reshard sites as
Phase 0a, just at scale matching the 4×4 k-grid (nq=16 vs 9, fewer
unique kernel cache entries since this is the scalar non-bispinor
path with one V_q tile shape).  See `profile/hlo_summary.md` for
the full table.

### Implication for the W-solve unpad question

I argued at the start of this initiative that `(I − V·χ)` has identity
in the pad block when V/χ are zero in pad rows/cols, so LU can run on
the padded matrix and W_pad = 0.  Phase 0b's HLO confirms there are no
all-gathers in the W-solve jit at the divisible n_rmu=640 case.  The
remaining open question — **do FFT-roundoff non-zero values appear in
χ pad rows that would couple the pad block to the real block? — is
not answered by Phase 0** and needs an actual run with
non-mesh-divisible n_rmu (e.g. 668) plus a numpy comparison of
`χ[:, n_rmu_logical:, :]` to zero.  Defer to Phase 2 once SlabIO+
helper wiring lets us run that case.

## Implications for the helper module

Phase 0a confirms the helper's surface area is small:

1. **Mandatory pad target:** `compute_V_q_tile` ζ-disk read.  SlabIO
   already supports `shape=padded, valid_shape=logical`; the caller
   (`gw/v_q_tile.py:1058-1063`) needs to pass them.  This is the only
   site where the padding contract is hard.
2. **Optional pad targets:** any future product-axis-sharded jit input
   on the μ-axis.  Today there are none beyond V_q.
3. **Single-axis sharding sites** (ζ-fit interior, ψ centroids,
   χ₀ output, V_q output, W I/O, COHSEX inputs) need padding only if
   `n_rmu < max(p_x, p_y) · ceil(n_rmu / max(p_x, p_y))` — the typical
   centroid file already satisfies this at p=4.
4. **Pad-row noise check (deferred from Phase 0):** scan W-solve χ
   pad-rows post-FFT to confirm exactness; only insert defensive
   unpad-before-LU if non-zero values appear.

## Next steps

1. Phase 0b completes → fold collectives into the table above.
2. **Decide on n_rmu_jax retirement:** confirm with `agent-A` user that
   the per-callsite `PadMeta` (logical, padded, partition_spec) is the
   single source of truth; rip out `Meta.n_rmu_jax`.
3. Phase 1: implement `runtime/padding.py::pad_to_mesh` /
   `unpad_from_mesh` + tests.
4. Phase 2: wire SlabIO `valid_shape=` calls at `v_q_tile.py:1058-1063`
   and remove the upstream "pre-pad on disk" hack the smoke-bed
   inherited from the reverted commit `579690d`.
5. Phase 3: 668 regression run + HLO grep — assert zero new collectives
   appear at top-level boundaries.
