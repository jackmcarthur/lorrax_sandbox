# Non-Symmorphic Symmetry Phase Audit and Fix

## Summary

LORRAX's full-zone wavefunction reconstruction was missing the fractional-translation phase used by BerkeleyGW for non-symmorphic symmetries. I added the BGW-style phase in `SymMaps`, kept the pre-existing time-reversal path unchanged, and verified the change with a focused unit test plus the existing COHSEX regression.

The implemented phase is the BGW `gmap` convention
`exp[-i (G_target + kg0) · tau]`, evaluated in LORRAX as the equivalent source-G expression `exp[-i (S G_source) · tau]`. On the existing Si 4x4x4 symmetric WFN, `44/64` full-zone k-points now pick up a nontrivial non-symmorphic phase.

## Code Changes

| File | Change |
|------|--------|
| `sources/lorrax/src/common/symmetry_maps.py` | Added explicit symmetry-context and BGW-style `kg0` helpers, computed fractional-translation phases from `tnp`, and applied those phases in `get_cnk_fullzone()` and `get_cnk_fullzone_batch()`. |
| `sources/lorrax/tests/test_symmetry_maps_nonsymmorphic.py` | Added a unit test covering BGW umklapp convention, non-symmorphic coefficient phases, and scalar/batch consistency. |

## Key Implementation Notes

- `tnp` in the WFN files is already stored in radians (`2π * fractional_translation`), matching BGW's internal convention.
- The new phase is applied only for spatial symmetries (`sym_idx < ntran`).
- The pre-existing time-reversal G-vector mapping and conjugation path were left unchanged, so this patch stays isolated to the spatial non-symmorphic correction.
- The selected symmetry operation is still the first match in file order, matching BGW's search behavior in `genwf_mpi.f90`.

## Results

### Formula match to BerkeleyGW

| Quantity | BerkeleyGW | LORRAX after patch |
|----------|------------|--------------------|
| Umklapp vector | `kg0` s.t. `k_full = S k_irred + kg0` | `_get_umklapp_vector(...)` |
| Fractional-translation phase | `exp[-i (G_target + kg0) · tau]` in `Common/gmap.f90` | `_get_fractional_translation_phase(...)` |
| Coefficient reconstruction | `zin(ind) * ph` in `Sigma/genwf_mpi.f90` | `cnk * phase` before spinor rotation |

### Existing-run sanity check

Using `runs/Si/00_si_4x4x4_60band/qe/nscf/WFN.h5`:

| Check | Result |
|------|--------|
| Full-zone k-points with nontrivial non-symmorphic phase | `44 / 64` |
| Spatial symmetry indices with nonzero `tnp` used in `irk_sym_map` | `1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 17, 18, 19, 21, 22, 23` |

### Test results

```text
uv run python -m pytest -q
10 passed, 1 warning in 11.21s
```

The warning is the pre-existing `jax.experimental.shard_map` deprecation in `src/common/load_wfns.py`.

## Status

- [x] BGW non-symmorphic phase convention identified from source
- [x] LORRAX coefficient reconstruction updated to include the phase
- [x] Existing time-reversal behavior preserved
- [x] Focused unit test added
- [x] Existing COHSEX regression kept passing
- [ ] Full BGW-vs-LORRAX production rerun with this patch
- [ ] Explicit runtime guardrails for unsupported TR/non-symmorphic combinations beyond the current scope

## Open Questions

1. This patch fixes the phase for whichever symmetry operation `irk_sym_map` selects. If future QE/BGW cases rely on multiple symmetry operations with identical rotational action but different translations, we may eventually want a more explicit audit of symmetry selection order.
2. Time-reversal remains only partially formalized in `SymMaps`; this patch intentionally did not change that path.
