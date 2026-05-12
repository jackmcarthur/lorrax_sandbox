"""Check whether a saved centroid file is closed under the WFN's spatial
symmetry group: applying every sym op s to every centroid c must produce
another centroid in the file (modulo lattice).
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
from centroid.orbit_syms import build_real_space_syms

WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/WFN.h5"
CENTROIDS = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/01_lorrax_gnppm_newcode/centroids_frac_400.txt"


def check_orbit_closure(cent_path, wfn_path, fft_grid_for_int_keys=None,
                        tol_grid_units=0.5):
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)
    R, Rinv, tau = build_real_space_syms(wfn, sym)
    R_np, Rinv_np, tau_np = np.asarray(R), np.asarray(Rinv), np.asarray(tau)
    n_sym = R_np.shape[0]
    fft_grid = np.array([int(x) for x in wfn.fft_grid]) if fft_grid_for_int_keys is None \
        else np.array(fft_grid_for_int_keys)

    cents = np.loadtxt(cent_path, comments="#")
    print(f"Loaded {len(cents)} centroids from {cent_path}")
    print(f"FFT grid: {tuple(fft_grid)}, n_sym = {n_sym}")

    # Encode each centroid as an integer FFT-grid triple for set lookup.
    def to_int_key(frac):
        return tuple(np.round(frac * fft_grid).astype(int) % fft_grid)
    cent_set = {to_int_key(c) for c in cents}
    if len(cent_set) != len(cents):
        print(f"  ⚠ {len(cents) - len(cent_set)} duplicate grid keys in input")

    # For each centroid, test that every sym image is also in the set.
    n_open = 0
    open_examples = []
    for ic, c in enumerate(cents):
        for s in range(n_sym):
            img = (c @ Rinv_np[s].T + tau_np[s]) % 1.0
            img_key = to_int_key(img)
            if img_key not in cent_set:
                n_open += 1
                if len(open_examples) < 5:
                    open_examples.append((ic, s, c, img))
                break       # one missing image is enough; move to next centroid
    print(f"\nCentroids whose orbit is NOT entirely contained in the set: "
          f"{n_open} / {len(cents)}")
    for ic, s, c, img in open_examples:
        print(f"  centroid {ic} = {c}, sym op {s} → {img}  (not in set)")

    # Conversely: count distinct orbits represented (every orbit fully in the
    # set is counted once; partial orbits also contribute their members).
    visited = set()
    n_orbits = 0
    n_orbit_sizes = []
    for c in cents:
        key = to_int_key(c)
        if key in visited:
            continue
        orbit_keys = {to_int_key((c @ Rinv_np[s].T + tau_np[s]) % 1.0)
                      for s in range(n_sym)}
        present = orbit_keys & cent_set
        if not orbit_keys.issubset(cent_set):
            pass    # already counted in n_open above
        visited |= orbit_keys
        n_orbits += 1
        n_orbit_sizes.append(len(orbit_keys))
    print(f"\norbit count seen in the set: {n_orbits}")
    sizes_hist = sorted(set(n_orbit_sizes))
    print(f"orbit-size histogram: " +
          ", ".join(f"size={k}:×{n_orbit_sizes.count(k)}" for k in sizes_hist))


if __name__ == "__main__":
    check_orbit_closure(CENTROIDS, WFN)
