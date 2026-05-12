"""Phase 0: confirm we can read sym data from `wfn` + `sym` and reproduce
real-space orbits correctly. No kmeans changes yet — this is just the
foundational sanity check.

What we verify:

1. ``sym.validate_atomic_symmetries(wfn)`` returns no failures.
   This is LORRAX's own check that the (R, Rinv, tau) data is internally
   consistent: applying every sym op to the atomic basis must reproduce
   the basis up to permutation (mod lattice).

2. ``orbit_images(rep)`` for each atom produces a set of n_sym images
   that, after lattice-wrapping and dedup, gives an orbit whose elements
   match the actual atom positions of the same species. This is the
   end-to-end test that our orbit-construction formula is right.

3. For each atom, count the stabilizer order k = n_sym / |orbit|.
   Verify k is consistent with the Wyckoff multiplicity.

4. Tie-multiplicity check: for an atom on a special Wyckoff site, sym ops
   that map the atom to itself (the stabilizer) must produce coincident
   images mod 1. Verify they're equal to fp64 noise (~1e-15) so a
   tie_tol of 1e-10 is clearly safe.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "/global/u2/j/jackm/software/lorrax_A/src")

from runtime import set_default_env
set_default_env()
import jax  # noqa
import numpy as np

from file_io import WFNReader
from common import symmetry_maps

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5"


def build_real_space_syms(wfn, sym):
    """Spatial-only sym data for r-space orbit construction.

    Returns
    -------
    R     : (n_sym, 3, 3) int32 — reciprocal-space rotation; folds disp back
            via δ_rep_row = δ_image_row @ R[s].T
    Rinv  : (n_sym, 3, 3) int32 — real-space rotation; image of rep is
            image_row = rep_row @ Rinv[s].T + tau[s]   (mod 1)
    tau   : (n_sym, 3) float64 — fractional translations, BGW sign convention
            (already negated relative to QE; see qe_save_reader.py:162).
    """
    n_sym = int(wfn.ntran)
    R = np.asarray(sym.R_grid[:n_sym], dtype=np.int32)
    Rinv = np.asarray(sym.Rinv_grid[:n_sym], dtype=np.int32)
    tau = np.asarray(wfn.translations[:n_sym], dtype=np.float64) / (2.0 * np.pi)
    return R, Rinv, tau


def orbit_images_np(rep, Rinv, tau):
    """Apply every sym op to a single rep. Returns (n_sym, 3) mod 1."""
    return (rep @ Rinv.transpose(0, 2, 1) + tau) % 1.0


def lattice_close(a, b, tol=1e-6):
    """True iff a and b are lattice-equivalent fractional coords (mod 1)."""
    d = (a - b) - np.round(a - b)
    return np.max(np.abs(d), axis=-1) < tol


def main():
    wfn = WFNReader(WFN)
    sym = symmetry_maps.SymMaps(wfn)

    print(f"system: ntran = {wfn.ntran}, nat = {wfn.nat}")
    print(f"FFT grid = {tuple(wfn.fft_grid)}")
    print()

    # 1. validate sym data via LORRAX's own check
    failures = sym.validate_atomic_symmetries(wfn)
    if failures:
        print("FAIL — validate_atomic_symmetries returned:")
        for f in failures[:5]:
            print(f"  {f}")
        return 1
    print("✓ sym.validate_atomic_symmetries: clean")
    print()

    # 2. build sym table our way + manually check orbit closure on atoms
    R, Rinv, tau = build_real_space_syms(wfn, sym)
    print(f"R    shape {R.shape}, dtype {R.dtype}")
    print(f"Rinv shape {Rinv.shape}, dtype {Rinv.dtype}")
    print(f"tau  shape {tau.shape}, range {tau.min():.4f} .. {tau.max():.4f}")
    print()

    # apos is in alat-cartesian; convert to fractional
    apos_cart = np.asarray(wfn.atom_positions, dtype=np.float64)
    avec = np.asarray(wfn.avec, dtype=np.float64)
    atom_frac = apos_cart @ np.linalg.inv(avec)             # (nat, 3) frac
    atom_types = np.asarray(wfn.atom_types)
    print("atom positions (fractional):")
    for i, (z, p) in enumerate(zip(atom_types, atom_frac)):
        print(f"  atom {i}: Z={z}  pos = {p % 1.0}")
    print()

    # 3. for each atom, build its orbit and count stabilizer
    print("orbit closure under (R, Rinv, tau):")
    for i, (z, p) in enumerate(zip(atom_types, atom_frac)):
        p_wrapped = p % 1.0
        images = orbit_images_np(p_wrapped, Rinv, tau)       # (n_sym, 3)

        # dedup mod lattice (round to fp ~6 digits to catch numerical drift)
        unique, counts = [], []
        for img in images:
            for ku, u in enumerate(unique):
                if lattice_close(img, u):
                    counts[ku] += 1
                    break
            else:
                unique.append(img)
                counts.append(1)

        orbit_size = len(unique)
        stab_order = wfn.ntran // orbit_size
        # Find which atoms in the basis (of same species) are in this orbit
        same_z = np.where(atom_types == z)[0]
        matched = []
        for u in unique:
            for j in same_z:
                if lattice_close(u, atom_frac[j] % 1.0):
                    matched.append(j)
                    break
            else:
                matched.append(-1)
        print(f"  atom {i}: |orbit| = {orbit_size}, stabilizer order = {stab_order}, "
              f"counts = {counts}, matched basis atoms = {matched}")
        if -1 in matched:
            print(f"    ⚠ orbit member not matched to any basis atom — "
                  f"sym data may be wrong")
        if any(c != stab_order for c in counts):
            print(f"    ⚠ uneven stabilizer counts {counts} (expected all = {stab_order})")
    print()

    # 4. tie-multiplicity check on a special position (any of the atoms above)
    #    For each atom, the stabilizer ops should produce IMAGES coincident
    #    with the atom mod 1. Check the max drift.
    print("tie-multiplicity check on atom 0 (Mo):")
    p = atom_frac[0] % 1.0
    images = orbit_images_np(p, Rinv, tau)
    # drift of each image from the original atom's orbit
    drift = np.array([
        np.min([np.max(np.abs((img - other) - np.round(img - other)))
                for other in [p % 1.0]])
        for img in images
    ])
    # but the right "tie drift" check: among ops that map p to p, the residual.
    self_mapping = np.array([lattice_close(img, p % 1.0) for img in images])
    print(f"  ops that map atom 0 to itself: {int(self_mapping.sum())} / {wfn.ntran}")
    if self_mapping.any():
        # Compare these images to the atom — drift should be ε·|tau| ~ 1e-15
        residuals = np.array([
            np.max(np.abs((images[s] - p) - np.round(images[s] - p)))
            for s in range(wfn.ntran) if self_mapping[s]
        ])
        print(f"  residual drift (max): {residuals.max():.3e} "
              f"(tie_tol=1e-10 is {'SAFE' if residuals.max() < 1e-10 else 'TIGHT'})")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
