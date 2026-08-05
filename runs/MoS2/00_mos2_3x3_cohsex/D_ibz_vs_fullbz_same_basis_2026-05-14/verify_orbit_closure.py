"""Verify the regenerated centroid file is orbit-closed under the WFN sym group."""
import sys
import os
os.environ.setdefault("JAX_ENABLE_X64", "true")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import h5py

from file_io.centroids import load_centroids
from centroid.orbit_syms import compute_centroid_sym_perm


WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/WFN.h5"
CENT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/centroids_frac_642.txt"

with h5py.File(WFN, "r") as f:
    ntran = int(f["/mf_header/symmetry/ntran"][()])
    mtrx = f["/mf_header/symmetry/mtrx"][:]              # (n_sym_max, 3, 3) — int
    tnp = f["/mf_header/symmetry/tnp"][:]                # (n_sym_max, 3) — frac/(2π)
    fft = f["/mf_header/gspace/FFTgrid"][:]              # (3,)

print(f"WFN: ntran={ntran}, fft_grid={tuple(int(x) for x in fft)}")
print(f"  sym_matrices first {ntran}:\n{mtrx[:ntran]}")
print(f"  translations[:{ntran}] = {tnp[:ntran]}")

_, cent_idx, n_cent = load_centroids(CENT, tuple(int(x) for x in fft))
print(f"Loaded {n_cent} centroids from {CENT}")

# `compute_centroid_sym_perm` expects:
#   - sym_matrices = full set incl identity, shape (n_sym, 3, 3)
#   - translations / (2π), shape (n_sym, 3)
sym_perm = compute_centroid_sym_perm(
    np.asarray(cent_idx, dtype=np.int32),
    sym_matrices=np.asarray(mtrx[:ntran]),
    translations=np.asarray(tnp[:ntran]),
    fft_grid=np.asarray(fft, dtype=np.int32),
)

print(f"OK: sym_perm shape={sym_perm.shape} (n_sym, n_cent)")

# Sanity: each row should be a permutation of [0, n_cent)
ok = True
for s in range(sym_perm.shape[0]):
    row = sym_perm[s]
    if set(row.tolist()) != set(range(n_cent)):
        ok = False
        print(f"  FAIL: sym {s} is not a permutation")
print(f"All {sym_perm.shape[0]} sym ops produce valid permutations of [0, {n_cent})." if ok else "PERMUTATION CHECK FAILED")
