# Phase 2 — sym-map consolidation (revised, procedural)

**Date:** 2026-05-14
**Status:** scope locked with user; ready for dispatch when desired
**Replaces:** the prior abstraction-heavy version of this file (`BzIbzTable` + `SymAction` classes — rejected per [[no-new-api-layers]] in memory).
**Trigger:** user 2026-05-14 — "symmetry operations need to truly be unified between wavefunctions and zetas (since they are fundamentally the same) and in all i/o paths"; and "Sounds good if those are the minimal signatures (e.g not clear to me why we would need the 'orbits' ever, i think we just need a list of length k_full that identifies for each k its IBZ kpt id, symmetry id, and maybe a separate trs thing or something if it helps). g0 we should also not support unfolding for now because the umklapp vectors make it nontrival (will completely redo later)."

## Style ground-rules (load-bearing)

- **Procedural, not OO**: free functions on numpy/jax arrays, in `symmetry_maps.py` (or a sibling sym-ops module if that file gets too long). No new classes, no new dataclasses, no wrapper objects.
- **`SymMaps` stays as the canonical sym-table holder**. Augment it. Don't replace it.
- **Consolidate by deletion, not addition** — every parallel routine that this refactor unifies must be DELETED, not wrapped.
- **Minimal signatures**: see [[feedback_minimal_signatures]]. Pass the `SymMaps` instance (a bag of arrays) plus the array(s) being acted on; don't list 9 individual array kwargs.

## Current fragmentation (recap from Agent 1's scope)

Parallel index tables on the same kgrid:
- `find_symmetry_ops_simple` → `irk_to_k_map`, `irk_sym_map` (k-side, length N_k_full)
- `find_irreducible_qpoints` → `q_full_to_irr_idx`, `q_full_to_irr_sym`, `q_irr_kgrid_int`, `q_irr_full_idx` (q-side, same kgrid)

Parallel unfold helpers:
- ψ k-unfold in `wfn_loader.py:825-851, 446-494, 967-985`
- V_q q-unfold in `_unfold_v_q_ibz_to_full` (`v_q_tile.py:1452`)
- g0 unfold in `_unfold_g0_ibz_to_full` (also `v_q_tile.py`) — **OUT OF SCOPE for this refactor**
- Current-channel unfold `_unfold_v_q_ij_ibz_to_full` — dead, fold in opportunistically if needed

## The three PRs

Each PR lands sequentially onto `agent/trs-aware-sym-fix`. Each is bit-equality-verifiable against the prior state.

### PR1 — consolidate index tables

In `symmetry_maps.py`, add one function:

```python
def find_irreducible_bz_points(kgrid_int, sym_mats_k, ntran):
    """For each row of kgrid_int (full-BZ point in integer kgrid coords),
    find (a) its canonical IBZ representative's row index, and (b) the
    sym_mats_k row index that maps the IBZ representative to this point.

    Args:
        kgrid_int: (N, 3) int32, the full-BZ point set in kgrid-int coords
        sym_mats_k: (2*ntran, 3, 3) int32, TRS-augmented sym ops on k
        ntran: int, number of spatial sym ops (so TRS rows are indices ntran..2*ntran-1)

    Returns:
        irr_idx: (N,) int32 — for each row of kgrid_int, the row index of its IBZ partner
        sym_idx: (N,) int32 — for each row of kgrid_int, the sym_mats_k row that maps IBZ → full

    Notes:
        - is_trs is just `sym_idx >= ntran`; not materialized separately.
        - The IBZ point list is `kgrid_int[unique(irr_idx)]`; not returned.
        - Callers that need "for each IBZ q, which full-BZ row(s) belong to it" can
          derive that on-the-fly: `(irr_idx == i_irr).nonzero()`. No separate
          orbit table.
    """
```

In `SymMaps.__init__`, call this twice — once with `wfn.kpoints` (or its kgrid-int version) and once with the q-difference grid — and store `(irr_idx_k, sym_idx_k)` and `(irr_idx_q, sym_idx_q)` directly on `self`.

Delete:
- `SymMaps.find_symmetry_ops_simple`
- `SymMaps.find_irreducible_qpoints`
- The cached fields `_q_irr_table_cache`, `irk_to_k_map`, `q_irr_full_idx`, `q_irr_kgrid_int`.
- The `_get_umklapp_vector` TRS branch (will be computed inside `unfold_psi` in PR3).

Bit-equal regression: existing callers (V_q IBZ cascade, wfn_loader) read different field names; verify the actual q-fold and k-fold tables are numerically identical to the prior code on MoS2 3×3, the new CrI3 30 Ry bispinor test bed once it lands, and Si 4×4×4 nosym.

### PR2 — move `unfold_v_q` into the sym module

Lift Phase 1's `_unfold_v_q_ibz_to_full` body from `v_q_tile.py:1452` into `symmetry_maps.py` (or `sym_ops.py`):

```python
def unfold_v_q(V_q_ibz, sym, sym_perm_ext):
    """V_q_full[q, μ, ν] = (conj on TRS rows)(V_q_ibz[irr_idx[q], π_s(μ), π_s(ν)])
    where π_s = sym_perm_ext[sym_idx[q]] (sym_perm_ext has 2*ntran rows;
    TRS rows duplicate spatial rows since centroids are TRS-invariant).
    """
    # body unchanged from Phase 1; reads sym.irr_idx_q, sym.sym_idx_q, sym.ntran
```

`v_q_tile.py:_unfold_v_q_ibz_to_full` becomes a one-line forwarder or is deleted entirely (and `v_q_g_flat.py:459` + `compute_vcoul.py:1025` call `unfold_v_q` directly).

The hard-fail guard ("sym_perm length too small to cover sym_idx range") moves with the function.

Bit-equal against Phase 1's R1 (MoS2 same-basis cascade-vs-full-BZ).

### PR3 — `unfold_psi` (the ψ-side TRS fix)

In `symmetry_maps.py` (or `sym_ops.py`):

```python
def unfold_psi(psi_irr, sym, sym_idx, k_full_frac, g_full_list):
    """ψ at a full-BZ k from ψ at its IBZ representative.

    For sym_idx < ntran (spatial):
        ψ_full(G) = U_spinor[s] · ψ_irr(S^{-1}(G + kg0)) · e^{-i(S·k_irr + G)·τ_s}
    For sym_idx >= ntran (TRS-augmented):
        ψ_full(G) = (iσ_y) · conj(U_spinor[s % ntran] · ψ_irr(-S^{-1}(G + kg0))) · e^{+i...}

    All four sub-rules (G rotation, τ-phase, spinor rotation, umklapp wrap)
    live here. No companion `_get_umklapp_vector` — the umklapp `kg0` is
    computed inline.
    """
```

`wfn_loader.py`'s eager (`825-851`) and phdf5 (`446-494`, `967-985`) ψ-unfold paths become one-line calls into `unfold_psi`. Both `SymMaps.U_spinor` and the related cartesian-frame stuff stay in `SymMaps` as data, BUT the value of `U_spinor` for TRS rows is computed correctly (the existing `if det<0: R=-R` flip in `get_spinor_rotations` was wrong for the TRS half — fix it at construction time, not at unfold time).

Sites resolved by this PR:
- Site #5 (wfn_loader ψ-unfold)
- Site #6 (U_spinor TRS rows)
- Site #7 (`_get_umklapp_vector` TRS branch — deleted, logic absorbed)

Bit-equal regression on inversion-symmetric systems (Si 4×4×4 nosym + MoS2 3×3 charge — TRS doesn't fire so ψ-unfold output should be byte-identical to pre-PR3). NEW non-equal regression on the CrI3 30 Ry bispinor test bed if it turns out to have non-inversion-symmetric setup — pre-PR3 ψ at TRS-fold k-points is WRONG; post-PR3 it's correct. Compare Σ_X at affected k-points pre/post; quantify expected shift.

## Out of scope (explicit)

- **g0 unfold** (`_unfold_g0_ibz_to_full` in `v_q_tile.py`): stays as Phase 1 left it. Per user: "we should also not support unfolding for now because the umklapp vectors make it nontrivial (will completely redo later)." Don't move it to `symmetry_maps.py`, don't try to unify it with `unfold_v_q`.
- **Current-channel `_unfold_v_q_ij_ibz_to_full`**: dead code, leave alone. Will get its own treatment when bispinor V_q transverse channels are wired up.
- **eqp0/eqp1 I/O convention**: per user decision, stays IBZ-only. Add a one-line docstring at the writer noting "canonical IBZ-only output file; readers must call `unfold_v_q`/`unfold_psi` to obtain full-BZ values."
- **BGW vcoul overlay**: already correct (mathematically TRS-safe by Coulomb's symmetry). Leave alone.

## Validation gates (per PR)

| PR | Bit-equal gate | New behavior gate |
|---|---|---|
| 1 | SymMaps fields after `__init__` numerically identical to prior on MoS2/CrI3/Si | none |
| 2 | MoS2 3×3 same-basis cascade-vs-full-BZ bit-equal (R1 from Phase 1 verification) | none |
| 3 | Si 4×4 + MoS2 3×3 charge ψ bit-equal | CrI3 30 Ry bispinor (non-inversion variant if applicable): Σ_X shifts by 10s-100s of meV on TRS-fold k-points — quantify, document |

## Open questions for dispatch

1. Should PR1+PR2 land before the CrI3 30 Ry test bed (already dispatching, ~45 GPU-min)? They're independent — PR1+PR2 only need MoS2/Si validation. PR3 needs the bispinor system.
2. Are the three PRs a single ~1-2 day session for one orchestrator+light-subagents, or do they deserve a 4-agent team again? Suggested: orchestrator-led, with subagent only for the per-PR regression test runs (which is the high-tool-call part).

## Memory references

- [[no-new-api-layers]] — why this version is procedural, not class-based
- [[unified-sym-action]] — the architectural principle
- [[feedback_minimal_signatures]] — minimal arg lists
- [[trs-blind-sym-bug]] — Phase 1 bug + fix that PR2 lifts
- [[lorrax-zeta-fit-architecture]] — the file/function map this refactor touches
