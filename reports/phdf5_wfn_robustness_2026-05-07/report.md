# PHDF5 WFN Robustness Pass

**Date:** 2026-05-07  
**Source tree:** `sources/lorrax_D`  
**Branch:** `agent/cri3-bispinor-repair`  
**GPU allocation:** `52596047` (`1` Perlmutter GPU node, `4` ranks/GPUs)

## Summary

I stress-tested the PHDF5 WFN reader's `read_kchunk_union` pathway on both
nosym and symmetry-unfolded WFN files.  The important discontiguous-k cases are
now covered by a reusable driver, and the PHDF5 path now supports bispinor
small-component expansion instead of returning only the two spinor components
stored in `WFN.h5`.

The remaining full-suite failures are unrelated k-means label mismatches that
were already present before this PHDF5 work.

## Code Changes

| File | Change |
|---|---|
| `sources/lorrax_D/src/common/phdf5_wfn_reader.py` | Added shared k-read ordering validation, fixed nosym arbitrary/duplicated `k_ids`, and added `bispinor=True` FFT-box small-component expansion. |
| `sources/lorrax_D/src/common/load_wfns.py` | Threaded `bispinor` into PHDF5 reads for r-chunk and centroid consumers, including k-chunked loops. |
| `sources/lorrax_D/src/common/psi_G_store.py` | Added a PHDF5 adapter that carries the driver's `bispinor` flag into cached PHDF5 reader calls. |
| `sources/lorrax_D/src/common/phdf5_plumbing_test.py` | Refreshed the Meta stub with `b_id_4_user` and added `--bispinor` coverage. |
| `sources/lorrax_D/src/common/phdf5_wfn_robustness.py` | New Shifter/MPI robustness driver comparing arbitrary PHDF5 `k_ids` requests to one-k-at-a-time legacy reads. |

## WFN Coverage

| WFN | Type | Size | Notes |
|---|---:|---:|---|
| `/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5` | nosym, `ntran=1`, `nk=9`, `nb=82` | `0.047 GB` | Exercises ntran=1 file-order reordering. |
| `sources/lorrax_D/tests/regression/cohsex_debug/WFNsmall.h5` | sym, `ntran=12`, `nk_full=9`, `nb=150` | `0.015 GB` | Exercises symmetry dedupe and TR-unfold warning path. |

## Results

| Command / scenario | Result |
|---|---|
| `common.phdf5_wfn_robustness` on nosym MoS2, head/mid/pad-past-file ranges | PASS, worst diff `0.000e+00`. |
| `common.phdf5_wfn_robustness` on sym WFNsmall, head/mid/pad-past-file ranges | PASS, worst diff `6.206e-17`. |
| Nosym bispinor arbitrary-k check, `--band-range 0 8` | PASS, worst diff `1.158e-19`. |
| Sym bispinor arbitrary-k check, `--band-range 0 8` | PASS, worst diff `2.861e-17`. |
| `common.phdf5_plumbing_test` on nosym MoS2, all-k + kchunk | PASS; rchunk all-k PHDF5 `27.4 ms` vs legacy `367.8 ms`; kchunk rchunk `81.6 ms` vs `650.8 ms`. |
| `common.phdf5_plumbing_test` on sym WFNsmall, all-k + kchunk | PASS; worst consumer diff `1.111e-17`. |
| Sym bispinor `common.phdf5_plumbing_test`, all-k + kchunk | PASS; worst consumer diff `1.552e-17`. |
| `common.phdf5_profile` MoS2, `band_chunk_size=4`, `kchunk=3` | PASS; mean wall `62.7 ms` over two timed iterations. |
| `common.phdf5_profile` MoS2, `band_chunk_size=20`, all-k | PASS; mean wall `31.3 ms` over two timed iterations. |
| `uv run python -m pytest -q tests/test_phdf5_wfn_reader_order.py -vv` | PASS, `3 passed`. |
| `ISDF_COHSEX_TEST_PLATFORM=cpu uv run python -m pytest -q tests/test_gw_jax_regression.py -m regression -vv` | PASS, `1 passed in 11.87s`. |
| `uv run python -m pytest -q` | Fails only pre-existing k-means tests: `3 failed, 75 passed, 5 skipped`. |

## Notes

- The native PHDF5 union handler already does the right core safety work:
  sorted disjoint hyperslabs, `ngkmax` zero-fill, and selected-point-count
  checks to catch overlaps.  The bug was Python-side ordering for nosym and
  missing bispinor propagation above the native reader.
- First PHDF5 calls per new shape include XLA compile time.  The per-read
  native timing on these small files is typically sub-ms to a few ms; the
  higher wall times in the first row of each case are compile dominated.
- Symmetric WFNsmall still warns that `3/9` full-BZ k-points require
  time-reversal unfolding and that non-symmorphic phases are not applied in
  that TR branch.  PHDF5 now matches the legacy `SymMaps` behavior for that
  pathway; the physics warning itself remains pre-existing.

## Status

- Done: arbitrary/discontiguous/duplicated k IDs are covered by a reusable
  PHDF5 robustness driver.
- Done: nosym, sym, k-chunked, pad-past-file, and bispinor WFN loading paths
  match legacy within zero or roundoff-scale differences.
- Not done: no commit was made because the broad suite still has unrelated
  k-means failures and `src/gw/ppm_sigma.py` has pre-existing dirty edits.
