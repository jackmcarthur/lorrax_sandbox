"""Independent per-element audit of ``common.symmetry_maps.unfold_psi`` (PR3).

This is the PR3 audit's hand-rolled reference — NOT copied from
``tests/test_unfold_psi_trs.py``. Built from first principles per
``reports/trs_sym_audit_2026-05-14/pr3_design.md § "The correct rule"``.

Geometry differences from the in-tree test (deliberate, to avoid
hand-copying the existing reference):

* Spatial group is {I, σ_x} (not {I, σ_y}); σ_x is a Z2 in-plane mirror
  represented integer-wise as diag(-1, 1, 1).
* Non-symmorphic τ on the IDENTITY row (not on σ_x). This exercises
  whether the τ-phase is applied even when ``s == 0``. The pre-PR3 code
  would have skipped the phase only on TRS rows (or, in the early bug,
  on s >= ntran); putting τ on identity guarantees we hit the spatial
  τ-phase code path on EVERY full-BZ k, and the TRS τ-phase on the
  TRS-rows row 2 (which corresponds to spatial s=0, i.e. pure TRS with
  a non-symmorphic τ).
* Spinor rotation U_spinor[1] is the σ_x SU(2) form  -i·σ_x  (not σ_y).
* A different G-list (random integer triples) and a different RNG seed.

The reference formula, re-derived independently:

Given ψ_kbar(G) at the IBZ representative k̄, the full-BZ k is reached
via a sym op (R, τ). For pure spatial ops (R = S spatial),
    ψ_{full,σ}(R·G) = Σ_{σ'} U_R[σ,σ'] · exp(-i (R·G) · τ) · ψ_{kbar,σ'}(G)
For TRS-augmented ops (effective R = -S, with T = iσ_y K acting after the
spatial part):
    ψ_{full,σ}(-S·G) = Σ_{σ'} (iσ_y)_{σσ''} · conj( Σ_{σ'} U_S[σ'',σ'] · exp(-i (S·G)·τ) · ψ_{kbar,σ'}(G) )
                    = Σ_{σ''σ'} (iσ_y · conj(U_S))_{σ,σ''_via_K} · exp(+i (S·G)·τ) · conj(ψ_{kbar,σ''}(G))
where the K (complex conjugation) propagates through both the matrix U_S
(→ conj(U_S)) and the phase (→ +i sign) and the wavefunction
coefficients (→ conj).

Mathematically equivalent: ψ_full = (iσ_y) · conj( spatial-result )
where "spatial-result" is what U_S · ψ_kbar · phase_spatial would be
for the same S and τ but no TRS.

We compute both forms in the reference and verify ``unfold_psi`` matches
each to relative error < 1e-12.
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

# Locate src/ on sys.path — sandbox-standard import shape.
import sys
SANDBOX = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B"
if SANDBOX not in sys.path:
    sys.path.insert(0, os.path.join(SANDBOX, "src"))

from common.symmetry_maps import unfold_psi  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry: {I, σ_x} spatial, non-symmorphic τ on identity.
# ---------------------------------------------------------------------------
NTRAN = 2

I3 = np.eye(3, dtype=np.int32)
sigma_x_int = np.diag([-1, 1, 1]).astype(np.int32)

SYM_MATS_K_SPATIAL = np.stack([I3, sigma_x_int], axis=0)              # (2, 3, 3)
SYM_MATS_K = np.concatenate(
    [SYM_MATS_K_SPATIAL, -SYM_MATS_K_SPATIAL], axis=0)                # (4, 3, 3)

# Non-symmorphic τ on the IDENTITY row (s=0). This puts τ-phase on
# the very first spatial op, exercising the spatial+TRS code paths
# on every full-BZ k.
TRANSLATIONS = np.array([
    [1.0 / 3.0, 0.0, 0.0],     # identity gets τ = (1/3, 0, 0)
    [0.0, 0.0, 1.0 / 4.0],     # σ_x gets τ = (0, 0, 1/4)
], dtype=np.float64)

# Spinor rotations.  U_0 = identity, U_1 = -i σ_x.
# σ_x as a real 2×2 is [[0, 1], [1, 0]], so -i σ_x = [[0, -i], [-i, 0]].
U_SPINOR_SPATIAL = np.zeros((NTRAN, 2, 2), dtype=np.complex128)
U_SPINOR_SPATIAL[0] = np.eye(2)
U_SPINOR_SPATIAL[1] = -1j * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

# i σ_y (complex 2×2):  [[0, 1], [-1, 0]].
I_SIGMA_Y = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.complex128)


# ---------------------------------------------------------------------------
# Independent reference (two equivalent forms — we check BOTH).
# ---------------------------------------------------------------------------

def reference_form_A(cnk_kbar, *, sym_idx, g_kbar):
    """Direct application of the design-doc formula."""
    cnk_kbar = np.asarray(cnk_kbar)
    g_kbar = np.asarray(g_kbar)
    is_trs = sym_idx >= NTRAN
    s = sym_idx - NTRAN if is_trs else sym_idx

    S_spatial = SYM_MATS_K_SPATIAL[s]
    tau = TRANSLATIONS[s]
    U_S = U_SPINOR_SPATIAL[s]

    # Spatial phase: exp(-i (S·G) · τ).
    rotated = (S_spatial.astype(np.int64) @ g_kbar.astype(np.int64).T).T
    phase_spatial = np.exp(-1j * rotated.astype(np.float64) @ tau)

    # Apply U_S · ψ_kbar · phase_spatial (spinor first then phase — order
    # doesn't matter on the spatial branch, the phase is a scalar per-G).
    spatial_form = (np.einsum("ij,nja->nia", U_S, cnk_kbar)
                    * phase_spatial[None, None, :])

    if is_trs:
        # ψ_full = (iσ_y) · conj(spatial_form)
        return np.einsum("ij,nja->nia", I_SIGMA_Y, np.conj(spatial_form))
    return spatial_form


def reference_form_B(cnk_kbar, *, sym_idx, g_kbar):
    """Equivalent form: pull the conj through and use the (iσ_y · conj(U_S))
    explicit spinor matrix.  This is the form unfold_psi is built from,
    BUT we re-derive it independently here from form A:

        ψ_full = iσ_y · conj(U_S · ψ_kbar · phase_spatial)
               = iσ_y · conj(U_S) · conj(ψ_kbar) · conj(phase_spatial)
               = (iσ_y · conj(U_S)) · conj(ψ_kbar) · exp(+i (S·G)·τ)
    """
    cnk_kbar = np.asarray(cnk_kbar)
    g_kbar = np.asarray(g_kbar)
    is_trs = sym_idx >= NTRAN
    s = sym_idx - NTRAN if is_trs else sym_idx

    S_spatial = SYM_MATS_K_SPATIAL[s]
    tau = TRANSLATIONS[s]
    U_S = U_SPINOR_SPATIAL[s]

    rotated = (S_spatial.astype(np.int64) @ g_kbar.astype(np.int64).T).T
    phase_spatial = np.exp(-1j * rotated.astype(np.float64) @ tau)

    if is_trs:
        U_eff = I_SIGMA_Y @ np.conj(U_S)
        return (np.einsum("ij,nja->nia", U_eff, np.conj(cnk_kbar))
                * np.conj(phase_spatial)[None, None, :])
    return (np.einsum("ij,nja->nia", U_S, cnk_kbar)
            * phase_spatial[None, None, :])


def run_audit():
    rng = np.random.default_rng(7919)
    nb, ns, ngk = 4, 2, 7
    cnk_kbar = (rng.standard_normal((nb, ns, ngk))
                + 1j * rng.standard_normal((nb, ns, ngk)))
    # Random G-list (no duplicates needed for the unfold_psi math).
    g_kbar = rng.integers(-3, 4, size=(ngk, 3)).astype(np.int32)

    # Sanity: form A == form B as identities (this also exercises the
    # algebraic re-derivation independently of unfold_psi).
    for sym_idx in range(2 * NTRAN):
        A = reference_form_A(cnk_kbar, sym_idx=sym_idx, g_kbar=g_kbar)
        B = reference_form_B(cnk_kbar, sym_idx=sym_idx, g_kbar=g_kbar)
        d = np.max(np.abs(A - B)) / max(np.max(np.abs(A)), 1e-30)
        assert d < 1e-13, f"form A/B disagree at sym_idx={sym_idx}: rel={d:.3e}"

    # unfold_psi agreement.
    max_rel = 0.0
    for sym_idx in range(2 * NTRAN):
        out = unfold_psi(
            cnk_kbar,
            sym_idx=sym_idx,
            n_sym_spatial=NTRAN,
            g_kbar=g_kbar,
            sym_mats_k=SYM_MATS_K,
            translations=TRANSLATIONS,
            U_spinor_spatial=U_SPINOR_SPATIAL,
        )
        ref = reference_form_A(cnk_kbar, sym_idx=sym_idx, g_kbar=g_kbar)
        d = np.max(np.abs(out - ref))
        rel = d / max(np.max(np.abs(ref)), 1e-30)
        is_trs = sym_idx >= NTRAN
        s = sym_idx - NTRAN if is_trs else sym_idx
        has_tau = bool(np.any(np.abs(TRANSLATIONS[s]) > 1e-12))
        print(f"  sym_idx={sym_idx}  is_trs={is_trs}  s_spat={s}  "
              f"has_tau={has_tau}  rel={rel:.3e}")
        assert rel < 1e-12, (
            f"unfold_psi disagrees with reference at sym_idx={sym_idx} "
            f"(is_trs={is_trs}): rel={rel:.3e}")
        max_rel = max(max_rel, rel)

    print(f"\nMAX rel err across all 4 sym rows: {max_rel:.3e}  (gate: < 1e-12)")
    print("PASS" if max_rel < 1e-12 else "FAIL")


if __name__ == "__main__":
    run_audit()
