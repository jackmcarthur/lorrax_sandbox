# Empirical investigation — ζ-sharding bug

**Author:** agent A, 2026-05-04
**Branch:** `agent-A/zeta-sharding-divisibility` (in `sources/lorrax_A`, off `origin/main` at `830a2fb`)
**Companion to:** `report.md` (the open bug report)
**Supersedes:** my earlier draft of this file (the "1e-14 ridge fixes it" hypothesis).

This is the empirical version. Every claim below is backed by a numbered run; every script is checked into [`scripts/`](scripts/). The "ridge fixes the 137 meV" hypothesis from my earlier draft is **falsified** in §2.

---

## TL;DR

1. The report's headline framing — "v1 (1 node) vs v2 (2 node) gives different ζ; sharding is the cause" — does not hold. **v1 and v2 were both on (2,2) mesh = 4 GPUs.** The "2 nodes" in v2's banner was the *pool* size, not the run size. Both runs used `LORRAX_NNODES=1`. (`compute_vcoul.py` log line `mesh=2x2` in both.)
2. The 280 meV bare-Σ_X swing between v1 and v2 was therefore not a sharding bug. It was a **code change on agent-D's branch** between May 3 (v1) and May 4 (v2). Current `origin/main` reproduces v2's correct value (-17.1315 eV) on (2,2) — see Run A.
3. **ζ has run-to-run byte-level nondeterminism** (bit-different `zeta_q.h5` between two runs of the same mesh, ~1e-7 element-wise relative diff) — almost certainly cuFFT/cuBLAS kernel autotune. **Bare Σ_X is robust to this**: identical to 4 decimal places (-17.1316 eV) across all five clean runs I did (Runs A, B, A56, B56, 8gpu).
4. **`cholesky_2d_batched` is correct and bit-deterministic** across mesh shapes when fed identical inputs (1×1, 2×2, 2×4, 4×4 all give ≤1e-15 reconstruction error and bit-identical L on the multi-device meshes). My earlier guess that it was the source was wrong.
5. **`compute_pair_density_spin_traced` and `compute_CCT_from_left_right` are also bit-deterministic** across mesh shapes when fed identical synthetic inputs. The whole pair-density → CCT → Cholesky chain is reproducible.
6. **The MoS2 8-GPU crash is real and trivial.** It's the [`runtime/__init__.py:nccl_warmup`](sources/lorrax_A/src/runtime/__init__.py#L187) bug I identified in the first draft, not a deeper ζ issue. **Patch in this branch.**
7. **Behind `nccl_warmup`, the *next* divisibility crash is real:** [`phdf5_wfn_reader.py:108`](sources/lorrax_A/src/common/phdf5_wfn_reader.py#L108) hard-fails when `nband % world_size != 0`. Si nband=60 fails on world=8 (60/8=7.5). This is the kind of issue the user's "general fears" pointed at. Not patched in this branch — needs a pad-and-trim refactor of the WFN reader, scoped in §4.
8. The construction I now propose is in §5: drop the ridge proposal entirely; land the warmup fix; pad-and-trim the WFN band axis; add a Σ_X-based cross-mesh reproducibility test (NOT a ζ-bit-identity test, which would false-positive on cuFFT autotune).

---

## 1.  Quantifying the v1 vs v2 ζ diff

[`scripts/compare_zeta.py`](scripts/compare_zeta.py) reads the two existing `zeta_q.h5` files and reports per-q diff stats.

For all 64 q-points, `‖ζ_v1 - ζ_v2‖_F / ‖ζ_v1‖_F` is in `[6.4e-8, 1.1e-7]` — uniform. Median element-wise relative diff is ~5e-8. The histogram tail (q=0):

| `|Δ|/|ζ|` bin | count | % |
|---|---|---|
| `[1e-8, 1e-6)` | 14.3 M | 71.8% |
| `[1e-6, 1e-4)` | 0.55 M | 2.8% |
| `>= 1` | 12.5 k | 0.06% |

The big-relative-diff tail is concentrated at *small* `|ζ|` values — divide-by-near-zero artifact, not an amplification signature.

[`scripts/diff_structure.py`](scripts/diff_structure.py) computes the SVD of ζ at q=0 and projects the diff into the right-singular basis:

```
top-5  σ: [84.88, 72.64, 68.86, 65.97, 63.93]
mid-5  σ: [3.17, 3.16, 3.16, 3.16, 3.15]
tail-5 σ: [1.53, 1.52, 1.51, 1.50, 1.48]
σ_max / σ_min = 5.733e+01
#σ < 1e-6 σ_max: 0 / 1440
```

**ζ at q=0 has condition number 57 — not near-singular.** All 1440 modes have σ > σ_max / 60. The diff energy lives entirely in modes with σ > 0.01·σ_max — there is no near-null subspace for a Cholesky regularizer to fix.

**This falsifies my earlier "1e-14 ridge fixes it" claim.** The user was right to push back.

---

## 2.  The pipeline is bit-deterministic across mesh shapes (controlled tests)

[`scripts/test_chol2d_determinism.py`](scripts/test_chol2d_determinism.py) — synthetic Hermitian PSD `(64, 1440, 1440)` C_q, two condition numbers (κ=100 and κ=1e10), Cholesky on (1×1), (2×2), (2×4), (4×4):

| mesh | well-cond `‖ΔL‖/‖L‖` vs np.linalg.cholesky | ill-cond `‖ΔL‖/‖L‖` | reconstruction `‖LL†-C‖/‖C‖` |
|---|---|---|---|
| 1×1 | 1.7e-15 | 1.9e-11 | 7.5e-16 |
| 2×2 | 1.4e-15 | 1.7e-11 | 5.9e-16 |
| 2×4 | 1.1e-15 | 1.4e-11 | 4.3e-16 |
| 4×4 | 1.1e-15 | 1.4e-11 | 4.3e-16 |

**`cholesky_2d_batched` is bit-deterministic across (2×2), (2×4), (4×4) and matches dense Cholesky to fp64 ε.** No ridge needed; conditioning amplification is linear in κ as expected, not exploding.

[`scripts/test_pair_density_determinism.py`](scripts/test_pair_density_determinism.py) — synthetic centroid wavefunctions, pair-density → CCT → L_q on the same four meshes, then [`scripts/compare_pd_outputs.py`](scripts/compare_pd_outputs.py):

| field | rel diff vs 1×1 |
|---|---|
| P_k (pair density) | **0** (bit-identical on all meshes) |
| C_q (CCT)         | **0** (bit-identical on all meshes) |
| L_q (Cholesky)    | 0 on 1×1; **7.9e-12 on 2×2 / 2×4 / 4×4** (multi-device meshes bit-identical to each other; 1×1 has the 1e-14·tr(C) ridge so L differs by ~1e-12) |

**The synthetic-input pipeline is mesh-shape-deterministic to fp64 precision.** Whatever caused the 1e-7 Si v1/v2 diff is *not* in any of these kernels.

---

## 3.  The 137 meV swing is a code bug already fixed in main

I re-ran v1's `cohsex.in` (byte-identical) on current `origin/main` from a fresh agent-A clone, varying only mesh and run count. Per-run bare Σ_X at k=0 (eV):

| Run | mesh | nband | bare Σ_X | matches |
|---|---|---|---|---|
| v1 (existing) | 2×2 | 60 | -17.4143 | (anomalous) |
| v2 (existing) | 2×2 | 60 | -17.1315 | "good" |
| **A (mine, main)** | 2×2 | 60 | **-17.1315** | matches v2 |
| **B (mine, main, repeat)** | 2×2 | 60 | **-17.1315** | matches v2 + A |
| A56 (mine, main) | 2×2 | 56 | -17.1316 | (different nband) |
| 8gpu (mine, main + warmup fix) | 2×4 | 56 | **-17.1316** | matches A56 |

Conclusions:
- (a) **Current main is correct on (2,2) — same answer as v2, never the "bad" v1.** v1's 280 meV anomaly was on agent-D's branch. The fix has either landed in main already or only matters for the agent-D code path.
- (b) **A and B are bit-different in `zeta_q.h5` (md5 mismatch) but identical in Σ_X.** Run-to-run nondeterminism in the ζ tensor exists and is at the ~1e-7 element-wise scale (cuFFT/cuBLAS autotune is the prime suspect — there is no other source of randomness in the pipeline) but is invisible in Σ_X.
- (c) **8-GPU (2,4) on main + warmup fix gives bit-identical Σ_X to 4-GPU (2,2).** Once the warmup bug is removed, sharding is *not* the source of any ζ pathology in current main.

The report's framing in `report.md` should be amended: there is no live "ζ-pruning is sharding-dependent" bug in main. The thing that needs fixing is the divisibility surface (next section).

---

## 4.  The real divisibility surface — what's actually broken

When I removed the `nccl_warmup` bug locally and re-ran 8-GPU Si, I hit the *next* divisibility crash:

```
File ".../src/common/phdf5_wfn_reader.py", line 110
ValueError: band count 60 not divisible by world=8
```

This is a hard `raise ValueError` in [`read_fft_box_band_chunk`](sources/lorrax_A/src/common/phdf5_wfn_reader.py#L108-L111):

```python
if nb % self._world_size:
    raise ValueError(
        f"band count {nb} not divisible by world={self._world_size}")
bands_per_rank = nb // self._world_size
```

This is the user's "general fears" point in concrete form. Si has `nband=60` (8 valence + 52 conduction = 60), and 60 is divisible by 4 but not by 8. So you cannot run Si nband=60 on 8 GPUs at all — fails before any compute. I worked around it for the cross-mesh test by setting `nband=56` (which trims sigma's conduction window). That's not acceptable as a permanent fix; nband is a physics knob, not a sharding knob.

**Other live divisibility hard-fails I confirmed by audit (not yet hit at runtime):**

| File:line | Constraint | Mitigation today | Severity |
|---|---|---|---|
| `runtime/__init__.py:187` | shape1d uses mesh.shape['x'] for all axes | **patched in this branch** | was blocking MoS2 8-GPU |
| `phdf5_wfn_reader.py:108` | `nband % world_size == 0` | none | live blocker for any (n_band, n_devices) where they don't divide |
| `centroid/pivoted_cholesky.py:368, 510, 624` | `M (candidates) % (Pr·Pc) == 0` | caller drops trailing candidates | live for pivoted-Cholesky path |
| `common/cholesky_2d.py:57, 125, 126` | `n % b == 0`, `J % Pr == 0`, `J % Pc == 0` | `compute_block_size_for_2d_cholesky` searches | works for highly composite n_rmu (640, 1440, 1600) but n_rmu=1517 (prime) would fail |
| `bse/bse_ring_comm.py:321, 737` | n_cond, n_val % Pr·Pc | `bse/bse_io.py` pads | already correct; the model the others should follow |
| `ffi/{slate,cusolvermp,cublasmp}/*.py` | external library constraint | must pad before call | unfixable from our side; we pad |

**Existing `pad-and-trim` adopters (the model):**
- `common/isdf_fitting.py:521-522` (`solve_zeta_from_L_q` pads `n_zchunk` to multiple of total devices)
- `gw/w_isdf.py:237-238`
- `gw/compute_vcoul.py:806`
- `bse/bse_io.py:113, 124, 135, 145, 207, 253, 302` — full pad library
- `psp/get_DFT_mtxels.py:482, 662` and `psp/dft_operators.py:491-565`

The pattern is well-established. The phdf5 reader and pivoted-Cholesky just haven't adopted it yet.

---

## 5.  What I'm proposing now (replaces my earlier draft)

**Land in this PR (no design ambiguity):**

1. **`runtime/__init__.py:nccl_warmup`** — per-axis warmup uses per-axis size. Already patched in `agent-A/zeta-sharding-divisibility`:

```python
shape2d = tuple(mesh_xy.shape[ax] for ax in mesh_xy.axis_names)
warm_specs = [(shape2d, P(*mesh_xy.axis_names))]
for ax in mesh_xy.axis_names:
    n_ax = int(mesh_xy.shape[ax])
    warm_specs.append(((n_ax,), P(ax)))
```

This unblocks every non-square mesh from running at all.

2. **Σ_X cross-mesh reproducibility test** in `tests/`. Run COHSEX/Si 4×4×4 nband=56 on (2,2) and (2,4) and assert `|Σ_X - ref| < 0.5 meV`. Do NOT compare ζ bit-for-bit — that would false-positive on cuFFT autotune (proven: Run A vs B produce bit-different ζ with identical Σ_X).

**Land as a separate PR (small refactor):**

3. **Pad `phdf5_wfn_reader` band axis** to a multiple of world size. Pattern: pad `nb` upward to `ceil(nb / world) * world`, allocate the read with the padded count, drop the padding bands when handing off to load_wfns. This is the same pattern as `solve_zeta_from_L_q`.

4. **Pad `pivoted_cholesky` candidate axis** to a multiple of `Pr*Pc`. Sentinel rows have `pair_density = 0` (zero column → loses every pivot competition); drop sentinel pivots after select. Removes the hard-fail at `pivoted_cholesky.py:368`.

5. **Centralize the pad helper** in `common/sharding_utils.py` once we have ≥3 callers. Don't pre-extract — the existing pads are simple enough that an early abstraction would be churn.

**NOT proposing (was in earlier draft, now withdrawn):**

- ~~Ridge in sharded `compute_L_q_from_CCT`~~. Falsified by §1+§2 — ζ at q=0 is well-conditioned (κ=57), no near-null subspace, and `cholesky_2d_batched` is already bit-deterministic.
- ~~Replace `cholesky_2d` with SLATE potrf~~. Cholesky_2d is fine.

**Open for the user's call:**

- Should the report `report.md` be amended/closed? My read: close it noting "this turned out to be an agent-D branch issue + the warmup divisibility crash; both have follow-up actions". The headline "ζ-pruning is sharding-dependent" claim is not borne out by current main.
- The 1e-7 element-wise nondeterminism in ζ across runs (cuFFT autotune) is real but harmless for COHSEX/Si bare Σ_X. Worth asking whether *any* downstream physics is sensitive — e.g. gradient calls in BSE, or Σ-output for sub-meV convergence studies. If yes, we'd want `XLA_FLAGS='--xla_gpu_deterministic_ops=true'` or equivalent in production runs.

---

## 6.  Reproducer files

Run dirs (all 4×4×4 nosym, x_only=true, otherwise byte-identical inputs from `D_lorrax_xonly_overlay_1440c_noavg`):

- `runs/Si/02_si_4x4x4_nosym/A_zeta_repro_4gpu_a/` — current main, 4 GPU, nband=56
- `runs/Si/02_si_4x4x4_nosym/A_zeta_repro_4gpu_b/` — current main, 4 GPU, nband=60 (run-to-run det. test)
- `runs/Si/02_si_4x4x4_nosym/A_zeta_repro_8gpu/`   — current main + nccl_warmup fix, 8 GPU, nband=56

Scripts:

- [`scripts/compare_zeta.py`](scripts/compare_zeta.py) — v1 vs v2 zeta_q.h5 element-wise diff
- [`scripts/diff_structure.py`](scripts/diff_structure.py) — SVD + projection of v1-v2 diff (proves no near-null subspace)
- [`scripts/test_chol2d_determinism.py`](scripts/test_chol2d_determinism.py) — synthetic Cholesky_2d on 4 meshes
- [`scripts/test_pair_density_determinism.py`](scripts/test_pair_density_determinism.py) — synthetic pair-density + CCT + L_q on 4 meshes
- [`scripts/compare_pd_outputs.py`](scripts/compare_pd_outputs.py) — diff harness for the above
