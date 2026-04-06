# compute_vcoul dead-code cleanup

## Summary

Audited `sources/lorrax/src/gw/compute_vcoul.py` for genuinely dead code versus dormant but still meaningful capability. Removed only code that was both unreferenced and redundant with live implementations; preserved the single-`q` APIs and the sharded HDF5 read stub because they still represent useful debugging / future-distributed-I/O entry points not otherwise exposed cleanly.

## Code changes

| File | Change |
|------|--------|
| `sources/lorrax/src/gw/compute_vcoul.py` | Removed two unused duplicated helpers (`compute_sqrt_vcoul_2d`, `compute_phase_q`), removed the unused compatibility wrapper `make_v_munu_kernel_chunked`, and deleted a few stale locals / comments. |

## Findings

### Removed as dead and redundant

- `compute_sqrt_vcoul_2d`
  - Not referenced anywhere in `src/` or active tests.
  - Its logic is duplicated in the live `sys_dim == 2` path inside `make_v_munu_chunked_kernel()`.
- `compute_phase_q`
  - Not referenced anywhere in `src/` or active tests.
  - Its logic is duplicated in the live `get_sqrt_v_and_phase()` kernels.
- `make_v_munu_kernel_chunked`
  - Not referenced anywhere in `src/` or active tests.
  - It was only a thin wrapper around `compute_V_q_from_zeta_array()` and did not provide a distinct implementation path.

### Intentionally preserved

- `compute_V_q_from_zeta_h5`
- `compute_V_q_from_zeta_array`
- `read_zeta_q_sharded`

These are currently dormant in production, but they still provide distinct single-`q` and future distributed-I/O entry points that are not otherwise represented as clearly elsewhere in the file.

## Validation

- `python3 -m py_compile src/gw/compute_vcoul.py`
- `uv run python -m pytest -q`

## Status

- [x] Dead duplicated helpers removed
- [x] Dormant-but-meaningful entry points preserved
- [x] Repo tests passed
