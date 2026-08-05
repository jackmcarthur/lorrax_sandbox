"""Pre-flight check: do CC/TT centroids close under WFN sym (80 Ry CrI3 6x6)?"""
import sys
import numpy as np
import h5py
from runtime import set_default_env  # noqa: F401 — must precede 'import jax'
set_default_env()

PARENT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07"
wfn_path = f"{PARENT}/qe/nscf/WFN.h5"
cc_path = f"{PARENT}/lorrax_B_ibz_cascade_postfix_2026-05-14/centroids_frac_1508.txt"
# Transverse path is auto-discovered (centroids_frac_<N>_current.txt under run A dir).
import glob
tt_candidates = glob.glob(
    f"{PARENT}/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/centroids_frac_*_current.txt"
)
if not tt_candidates:
    print("FAIL: no transverse centroids file found yet "
          "(centroids_frac_*_current.txt under run-A dir).")
    sys.exit(1)
tt_path = sorted(tt_candidates)[-1]

with h5py.File(wfn_path, 'r') as f:
    mtrx = f['mf_header/symmetry/mtrx'][...]
    tnp = f['mf_header/symmetry/tnp'][...]
    ntran = int(f['mf_header/symmetry/ntran'][()])
    fft_grid = f['mf_header/gspace/FFTgrid'][...]
print(f"ntran={ntran}, fft_grid={fft_grid.tolist()}")
mtrx = mtrx[:ntran]
tnp = tnp[:ntran]

from centroid.orbit_syms import compute_centroid_sym_perm

any_fail = False
for label, path in [("CC (scalar 1508)", cc_path), ("TT (current)", tt_path)]:
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
        any_fail = True
        first = e.args[0].splitlines()[0] if e.args else str(e)
        print(f"  FAIL  {first}")

sys.exit(1 if any_fail else 0)
