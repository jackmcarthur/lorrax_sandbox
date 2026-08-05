"""Verify orbit closure of a centroid file against the WFN's sym group.

Loads centroids_frac_<N>.txt, builds sym arrays from WFN.h5, and calls
compute_centroid_sym_perm with validate=True. Exits 0 on success, 1 on
failure (closure violation) — printing the failure index breakdown.

Usage (from a run dir containing WFN.h5):
    python3 verify_orbit_closure.py <centroids_file.txt>
"""
from __future__ import annotations

import sys
import numpy as np

# Bring up the runtime so file_io is importable in the lorrax_B layout.
from runtime import set_default_env
set_default_env()

from file_io import WfnLoader
from centroid.orbit_syms import compute_centroid_sym_perm


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <centroids_frac_N.txt>")
        return 2
    centroids_file = sys.argv[1]

    wfn = WfnLoader("WFN.h5")
    fft_grid = tuple(int(x) for x in wfn.fft_grid)
    ntran = int(wfn.ntran)
    sym_matrices = np.asarray(wfn.sym_matrices[:ntran], dtype=np.int64)
    translations = np.asarray(wfn.translations[:ntran], dtype=np.float64)

    print(f"WFN: fft_grid={fft_grid}, ntran={ntran}")

    centroids_frac = np.loadtxt(centroids_file, comments="#")
    n_cent = centroids_frac.shape[0]
    print(f"Loaded {n_cent} centroids from {centroids_file}")

    # Snap to FFT grid -> integer indices.
    idx = (np.round(centroids_frac * np.asarray(fft_grid)[None, :])
            .astype(np.int64) % np.asarray(fft_grid)[None, :])

    try:
        sym_perm = compute_centroid_sym_perm(
            idx, sym_matrices, translations, fft_grid, validate=True,
        )
    except RuntimeError as e:
        print(f"FAILED orbit closure: {e}")
        return 1

    print(f"OK: sym_perm shape={sym_perm.shape} (n_sym, n_cent)")
    # Sanity: every row should be a permutation of [0, n_cent).
    for s in range(sym_perm.shape[0]):
        row = sym_perm[s]
        if (sorted(np.unique(row).tolist()) != list(range(n_cent))):
            print(f"sym {s}: not a permutation!")
            return 1
    print("Every sym op produces a valid permutation of [0, n_cent).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
