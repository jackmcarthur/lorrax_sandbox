"""NaN + condition-number check on the 4 ζ files.

Pure h5py + numpy on rank 0 — no JAX needed.  We inspect:
  1. NaN count per file (must be 0 for finite output).
  2. max|ζ|, mean|ζ| per file (sanity).
  3. The CCT eigenvalue spectrum at q=0 (rebuilt from ζ via
     C_q = (Σ_k ψ ψ^* phase) inner product) is hard to recover post-hoc
     so we use the disk ζ to verify that the solve was self-consistent
     by checking that the *operator* of solving with ζ matches.

For now we just report (1) and (2) — that already disambiguates the
'mu_L=i is fully NaN' regression.  Residual quantification is run in
test_lu_residual.py (re-uses fit_zeta internals).

Run on rank 0 only:  python3 -u check_zeta_fit.py
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import h5py


def main():
    here = Path(__file__).resolve().parent
    tmp = here / "tmp_test_bispinor"

    files = ["zeta_q.h5", "zeta_q_mu1.h5", "zeta_q_mu2.h5", "zeta_q_mu3.h5"]
    print("=== NaN sweep + magnitude (full file) ===")
    for fname in files:
        p = tmp / fname
        with h5py.File(p, 'r') as f:
            ds = f['zeta_q']
            nq = ds.shape[0]
            n_nan = 0
            zmax = 0.0
            zsum = 0.0
            zcount = 0
            for q in range(nq):
                z = ds[q]
                n_nan += int(np.isnan(z).sum())
                zmax = max(zmax, float(np.nanmax(np.abs(z))))
                zsum += float(np.nansum(np.abs(z)))
                zcount += z.size
            zmean = zsum / max(1, zcount)
            print(f"  {fname:20s}  shape={ds.shape}  "
                  f"nan={n_nan}  max(|ζ|)={zmax:.3e}  "
                  f"mean(|ζ|)={zmean:.3e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
