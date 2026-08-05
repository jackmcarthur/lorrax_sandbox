"""Pre-flight check: do CC/TT centroids close under WFN sym?"""
import sys
import numpy as np
import h5py

wfn_path = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/qe/nscf/WFN.h5"
ts_path = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/00_lorrax/centroids_frac_300.txt"
tt_path = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_30Ry_bispinor_2026-05-14/00_lorrax/centroids_frac_298_current.txt"

with h5py.File(wfn_path, 'r') as f:
    mtrx = f['mf_header/symmetry/mtrx'][...]
    tnp = f['mf_header/symmetry/tnp'][...]
    ntran = int(f['mf_header/symmetry/ntran'][()])
    fft_grid = f['mf_header/gspace/FFTgrid'][...]
print(f"ntran={ntran}, fft_grid={fft_grid.tolist()}")
mtrx = mtrx[:ntran]
tnp = tnp[:ntran]

from centroid.orbit_syms import compute_centroid_sym_perm

for label, path in [("CC (scalar)", ts_path), ("TT (current)", tt_path)]:
    print(f"\n--- {label}: {path} ---")
    frac = np.loadtxt(path)
    print(f"  n_centroids: {frac.shape}")
    r_mu_fft_idx = np.rint(frac * fft_grid[None, :]).astype(np.int32) % fft_grid[None, :]
    try:
        alpha, L = compute_centroid_sym_perm(
            r_mu_fft_idx,
            sym_matrices=mtrx,
            translations=tnp,
            fft_grid=fft_grid.astype(np.int32),
        )
        print(f"  PASS  alpha.shape={alpha.shape}  L.shape={L.shape}")
    except RuntimeError as e:
        first = e.args[0].splitlines()[0] if e.args else str(e)
        print(f"  FAIL  {first}")
