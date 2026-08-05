# PR3 — ψ k-unfold consolidation + bispinor TRS-spinor fix

**Date**: 2026-05-14
**Status**: design draft; implementation gated on (a) user sanity-check of the math below, (b) MoS₂ 3×3 SOC test bed landing as validation target.
**Branch**: `agent/trs-aware-sym-fix` (off PR2 audit fixes commit `796c043`).

## Three interacting bugs (Agent 1 scope Sites #5, #6, #7)

### Site #6: `U_spinor[ntran:]` is wrong

`SymMaps.__init__` builds `sym_mats_k` of length `2·ntran` (line 220 of pre-PR1, line 220 still in PR1) where the second half is `-sym_mats_k[:ntran]`. Then it calls `syms_crystal_to_cartesian(wfn)` to get `R_cart` of length `2·ntran`, then `get_spinor_rotations(R_cart)`.

`get_spinor_rotations` (line 548-632 of pre-PR1, body unchanged) contains:

```python
for isym, R in enumerate(sym_matrices_cart):
    if np.linalg.det(R) < 0:
        R = -R                                 # improper → proper SU(2) prep
    # ... compute SU(2) of R via quaternion ...
```

For the second half of `R_cart` (the `-S_spatial` rows), this flip sends `det = -det(S) < 0` back to `+S_spatial`, so:

  U_spinor[ntran + i] = SU(2)(S_spatial[i])  =  U_spinor[i]      (NOT the physical TRS spinor)

The **physically correct** value for a TRS-augmented row is:

  U_spinor[ntran + i]_correct  =  iσ_y · conj(U_spinor[i])

(per the time-reversal operator T = iσ_y K in the standard SOC convention.)

### Site #5: `wfn_loader.py:839-850` skips τ-phase on TRS rows

```python
if sym_idx >= ntran:
    cnk = np.conj(cnk)                       # K applied to ψ_kbar
else:
    tau = self.translations[sym_idx]
    if np.any(np.abs(tau) > 1e-12):
        phase = np.exp(-1j * rotated @ tau)
        cnk *= phase[None, None, :]

cnk = np.einsum("jk,nkl->njl", U_per[sym_idx], cnk)   # spinor rotate
```

The TRS branch has TWO bugs:
- No τ-phase. For non-symmorphic ops (`τ ≠ 0`) the TRS row drops the phase entirely.
- Calls `U_per[sym_idx]` with `sym_idx >= ntran`, which reads the broken Site #6 row.

The same buggy pattern repeats in the phdf5 ψ-unfold paths at lines 446-494 and 967-985.

### Site #7: `_get_umklapp_vector` TRS branch

When `sym_idx >= ntran`, `_get_umklapp_vector` (line 743 of pre-PR1) returns the umklapp for `sym_krep @ kbar` directly (no TRS-aware wrap). The function's own docstring notes the τ-phase isn't applied. This bakes into the wfn_loader call at line 313.

## The correct rule

Let `s = sym_idx`, `s_spatial = s mod ntran`, `is_trs = s >= ntran`. Then for each full-BZ k:

```
G_rot = sym_mats_k[s] @ G_kbar - kg0          # umklapp into BZ
phase = exp(-i · (S_k k_irr + G_rot) · τ_{s_spatial})
inner = U_spinor[s_spatial] @ ψ_kbar(G_kbar) · phase

if is_trs:
    ψ_full(G_rot) = (iσ_y) · conj(inner)      # apply T = iσ_y K to the spatial form
else:
    ψ_full(G_rot) = inner
```

Equivalent re-grouping (saves one matmul):

```
if is_trs:
    spinor_op  = iσ_y · conj(U_spinor[s_spatial])
    phase_op   = exp(+i · (S_k k_irr + G_rot) · τ_{s_spatial})
    cnk_ψ_op   = conj(ψ_kbar(G_kbar))
else:
    spinor_op  = U_spinor[s_spatial]
    phase_op   = exp(-i · (S_k k_irr + G_rot) · τ_{s_spatial})
    cnk_ψ_op   = ψ_kbar(G_kbar)

ψ_full(G_rot) = spinor_op @ (cnk_ψ_op · phase_op)
```

The TRS branch's phase sign flips (conj of `exp(-i...)` = `exp(+i...)`); the ψ_kbar gets conjugated; the spinor matrix becomes `iσ_y · conj(U_spinor[s_spatial])`.

For systems with inversion (`-I ∈ spatial group`), TRS adds no new orbits — the equivalent spatial op covers the fold. So the TRS branch is **never exercised** in production for inversion-symmetric materials. This is why the bug has been latent.

## `unfold_psi` free function signature

In `src/common/symmetry_maps.py`, add:

```python
def unfold_psi(psi_irr_k, *,
               sym_idx, n_sym_spatial,
               sym_mats_k, translations, U_spinor_spatial,
               g_irr, k_full_frac):
    """ψ at a full-BZ k from ψ at its IBZ representative.

    Parameters
    ----------
    psi_irr_k : (nb, ns, ngk_irr) complex — IBZ ψ coefficients on the IBZ G-list.
    sym_idx : int — row in sym_mats_k (length 2·n_sym_spatial).
    n_sym_spatial : int — ntran.
    sym_mats_k : (2·ntran, 3, 3) int — TRS-augmented sym table.
    translations : (ntran, 3) float — non-symmorphic τ_s (NOT TRS-extended).
    U_spinor_spatial : (ntran, 2, 2) complex — SPATIAL spinor rotations only.
        Callers must NOT pass the buggy (2·ntran, 2, 2) table; pass
        sym.U_spinor[:ntran] explicitly. The TRS-row spinor is computed
        inside unfold_psi from this spatial half + iσ_y · conj.
    g_irr : (ngk_irr, 3) int — IBZ G-list.
    k_full_frac : (3,) float — target k in fractional coords (for τ-phase).

    Returns
    -------
    psi_full : (nb, ns, ngk_irr) complex — ψ at the target full-BZ k.
        The G-axis is rotated via sym_mats_k[sym_idx]; downstream code
        is responsible for any further unfold of the G-axis into a
        common full-BZ G-list (existing wfn_loader.gvecs handles this).
    """
```

Sites #5 (wfn_loader.py ψ-unfold paths) become one-line calls into this. The `U_spinor` instance attr on SymMaps stays length `2·ntran` for backward compatibility, BUT we ALSO add `sym.U_spinor_spatial = sym.U_spinor[:ntran]` (a deliberately-cleaner accessor) and mark the TRS-half as "do not use directly, see unfold_psi" in the SymMaps docstring. PR3 doesn't *delete* the buggy TRS-half values from `U_spinor` — only one consumer (wfn_loader) reads them, and that consumer migrates to `unfold_psi`. The buggy values get nullified by being unreachable.

Alternative: delete the TRS-half of `U_spinor` entirely (shape becomes `(ntran, 2, 2)`). This would be cleaner but risks breaking any callers I missed; safer to keep the array and just stop trusting its second half.

## Migration plan

### Step 1: write `unfold_psi` in `symmetry_maps.py`

Pure-numpy body (no jax for now — the wfn_loader paths are host-side; sharded GPU lift happens later in `_load_cnk_h5`). Body is the equivalent of the existing `wfn_loader.py:823-850` block with the TRS fix wired in.

### Step 2: migrate `wfn_loader.py`

Three callsites all become `psi_full = unfold_psi(psi_irr, sym_idx=..., n_sym_spatial=ntran, ...)`. The local τ-phase + spinor-rotate code blocks delete.

`_get_umklapp_vector`'s TRS branch is also handled inside `unfold_psi` (the umklapp computation is part of the unfold). The function stays for the non-TRS callsite at `wfn_loader.py:313` (`gvecs(k='full_bz')` G-list rebuild), which doesn't actually need the τ-phase fix.

### Step 3: docstring + sym.U_spinor_spatial accessor

Add the accessor on SymMaps. Mark `U_spinor[ntran:]` as deprecated-and-ignored.

### Step 4: tests

A new pytest at `tests/test_unfold_psi_trs.py`:
- Synthetic non-inversion bispinor system (D3-like, 3 spatial + 3 TRS rows in sym_mats_k).
- Hand-construct ψ_kbar with non-trivial spinor structure.
- Apply `unfold_psi` for each full-BZ k.
- Reference: hand-rolled `(iσ_y) · conj(U_spinor[s_spatial] · ψ_kbar · phase)` formula.
- Gate: max rel err < 1e-12 for both spatial and TRS k's.

The pre-existing tests in `test_wfn_loader_eager.py` and `test_wfn_loader_phdf5.py` (if any) should still pass at literal bit-equality on inversion-symmetric systems (Si 4×4 nosym) since TRS never fires.

## Validation gates (e2e)

After PR3 commits:
- Pytest green for all new + existing tests.
- **Inversion-symmetric regression**: rerun MoS₂ 3×3 (charge, non-SOC) and CrI3 6×6 80 Ry. Σ_X must be bit-equal to PR2.
- **Non-inversion bispinor**: rerun MoS₂ 3×3 SOC (test bed in flight at `runs/MoS2/03_mos2_3x3_soc_2026-05-14/`). Σ_X at TRS-fold k's should shift by ~10-100 meV (the iσ_y fix activates); document the magnitude.
- **Nosym vs sym validation** (task #30): once PR3 lands, set up `runs/MoS2/03_mos2_3x3_soc_nosym/` (same as SOC variant but `noinv=.true., nosym=.true.` for full k-grid in WFN), run LORRAX on both, compare Σ_X. Sub-meV agreement is the gate.

## Open questions for user before implementation

1. **Delete buggy U_spinor TRS half, or just stop reading it?** Cleaner to delete (shape `(ntran, 2, 2)`); safer to leave (in case there's a consumer I haven't found yet).

2. **Pure numpy or JAX for `unfold_psi`?** Current wfn_loader does pure numpy on host (the array is host-resident; sharding happens later in `_load_cnk_h5`). I'll match — but if there's a future path where ψ-unfold runs inside a jit, the signature should be jax-friendly. Keeping it pure numpy is fine for now.

3. **Migration scope: just `wfn_loader.py`, or also the `bse/bse_io.py` and `psp/*.py` callers?** Those use `sym.irr_idx_k[ik]` directly to fetch IBZ ψ — they don't apply the spinor rotation themselves AFAICT. Skip them in PR3; revisit if a bug shows up.
