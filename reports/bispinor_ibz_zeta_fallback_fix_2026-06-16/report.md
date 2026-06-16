# Fix: bispinor IBZ-cascade zeta-fit crash — and sym==nosym validation

**Date:** 2026-06-16 · **Branch:** `lorrax_C agent/bispinor-ibz-zeta-fallback-fix` (`fc9984e`, off `main e85be60`)
**Owner:** session-C · **System:** MoS₂ 3×3, nband=32, bispinor screened COHSEX

## Why

Milestone-A validation (see `reports/bispinor_screened_a_validation_2026-06-16/`) required
`LORRAX_FORCE_FULL_BZ=1` to dodge a crash in the bispinor zeta fit. Forcing full-BZ to mask a
symmetry bug is exactly the wrong move — so this fixes the real bug and validates the IBZ
symmetry path against a same-basis full-BZ reference.

## The bug (ordering)

`fit_zeta_to_h5` sliced the per-q CCT (`C_q`/`L_q`) to IBZ rows
(`isdf_fitting.py:~2079`, *before* `factor_c_q`) using the **initial** `write_ibz_only`. But the
orbit-closure **auto-fallback** — the only place `write_ibz_only` is flipped to `False` — ran
**after** `factor_c_q` (`isdf_fitting.py:~2149`). So on a centroid set that isn't orbit-closed:

- **Charge channel** (`vertex_mu_L=0`): closure fails → silent fallback `write_ibz_only=False`
  → `q_irr_full_idx=None` → `Z_q` stays full-BZ (9), but `L_q` was **already IBZ-sliced (5)** →
  `ValueError: B.shape[0]=9 != Nq=5` in `batched_distributed_potrs`.
- **Transverse channels** (`vertex_mu_L≠0`): the fallback loud-fails (raises) — but the charge
  channel crashed first, so the user only saw the cryptic shape error.

It was **not** a wrong-unfold bug; the IBZ math is correct (proven below). It was purely that the
fallback path produced inconsistent q-axis shapes.

## The fix

Move the orbit-closure check (which finalizes `write_ibz_only`) to **before** the `C_q`/`L_q`
slice, so a closure failure falls back to full-BZ **consistently** for `L_q`, `Z_q`, and the
on-disk q-axis. Remove the now-redundant late check. `+48 / −49` lines, one file
(`src/common/isdf_fitting.py`).

## Validation

| Test | Result |
|---|---|
| `pytest tests/test_q_ibz_and_centroid_perm + test_sigma_x_bispinor + test_mf_isdf_header_roundtrip` | **21 passed (4.07s)** |
| Fixed code, **no** `FORCE_FULL_BZ`, original 640+668 centroids | **Completes** (was: `B.shape 9!=5` crash). `sigma_diag` **bit-identical** to the forced-full-BZ milestone-A run. |
| Transverse IBZ (5q) vs transverse full-BZ, same 668 centroids | bit-identical (the auto-fallback run ran transverse IBZ-active while charge fell back) |
| **Full IBZ (charge 641 + transverse 668, all 4 channels IBZ) vs full-BZ, same 641+668 centroids** | **BIT-IDENTICAL** ✅ — `sym == nosym` |

Runs:
- `runs/MoS2/C_60Ry_bispinor_ibztest_2026-06-16/` — fixed code, no FORCE, orig centroids (auto-fallback; transverse IBZ).
- `runs/MoS2/C_60Ry_bispinor_fullibz_2026-06-16/` — full IBZ, 641+668 closed (`sigSX(n=0)=-22.146`).
- `runs/MoS2/C_60Ry_bispinor_fullbz641_2026-06-16/` — full-BZ ref, same 641+668 (`FORCE_FULL_BZ=1`).

**Conclusion:** the bispinor IBZ→full-BZ unfold (ζ̃ and V_q, charge **and** transverse channels)
is numerically exact. `LORRAX_FORCE_FULL_BZ` is no longer needed — a non-closed centroid set now
falls back gracefully, and a closed set runs the full IBZ cascade correctly.

## Data finding: the charge 640 centroids weren't orbit-closed

The crash's trigger was a **data** issue, not just the code path: the original
`centroids_frac_640.txt` is not closed under the WFN sym group — the z-mirror maps centroid
fft_idx `[0,6,21] → [0,6,99]` (=120−21) which is absent (588/1280 such failures). The transverse
`centroids_frac_668_current.txt` *was* already closed. Regenerated the charge set orbit-aware:
`kmeans_cli 640 --seed 42` → `centroids_frac_641.txt` ("325 orbits → 641 unfolded centroids,
orbit-closed"). With both closed, the full IBZ cascade activates (disk shrink 1.8× on this 3×3).

## Notes / risks

- **Concurrent JAX gotcha (cost one run):** launching two multi-GPU `gw.gw_jax` jobs on the *same*
  node simultaneously crashed one with `ABORTED: ... different incarnation ... likely restarted`
  (JAX distributed coordination-service collision, EXIT 134). Run multi-GPU JAX jobs sequentially
  per node, or on distinct nodes. Logged in KNOWN_SANDBOX_ERRORS.
- Absolute Σ differs slightly between the 640 (non-closed) and 641 (closed) centroid sets — expected
  (different ISDF basis); the sym==nosym comparison is always same-basis.
- The fix is committed on the feature branch only (not merged/pushed) per git discipline.
