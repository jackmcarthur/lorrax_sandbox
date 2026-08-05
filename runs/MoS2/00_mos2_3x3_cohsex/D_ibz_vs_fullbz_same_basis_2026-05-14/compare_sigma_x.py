"""Compare bare Σ_X eigenvalues between Run A (IBZ cascade) and Run B (full-BZ same basis).

Reads sigma_freq_debug.dat from each run, extracts the `x_bare` column per (k, n),
and prints side-by-side + max |diff|.
"""
from __future__ import annotations

import os
import sys
import numpy as np


def parse_sigma_freq_debug(path: str) -> dict[int, np.ndarray]:
    """Return {k: array of x_bare per n}.

    Format: tab-separated columns; column 0=k, 1=n, 5=x_bare.
    Blocks delimited by `k-point N:` headers.
    """
    by_k: dict[int, list[tuple[int, float]]] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#") or line.startswith("k-point"):
                continue
            parts = [p.strip() for p in line.split("\t") if p.strip()]
            if len(parts) < 6:
                continue
            try:
                k = int(parts[0])
                n = int(parts[1])
                # Columns: k n E_dft Edft-Ef kin_ion V_H x_bare ...
                x_bare = float(parts[6])
            except (ValueError, IndexError):
                continue
            by_k.setdefault(k, []).append((n, x_bare))
    out = {}
    for k, lst in by_k.items():
        lst.sort()
        out[k] = np.array([x for _, x in lst])
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    pA = os.path.join(here, "run_A_ibz", "sigma_freq_debug.dat")
    pB = os.path.join(here, "run_B_fullbz", "sigma_freq_debug.dat")
    A = parse_sigma_freq_debug(pA)
    B = parse_sigma_freq_debug(pB)
    ks = sorted(set(A) & set(B))
    print(f"k-points: A={sorted(A)}  B={sorted(B)}  intersection={ks}")
    max_abs = 0.0
    max_loc = (None, None)
    rows = []
    for k in ks:
        a, b = A[k], B[k]
        n = min(a.size, b.size)
        d = a[:n] - b[:n]
        for i, di in enumerate(d):
            if abs(di) > max_abs:
                max_abs = abs(di)
                max_loc = (k, i)
        rows.append((k, n, np.max(np.abs(d)), np.argmax(np.abs(d))))
    print()
    print("Per-k summary:")
    print(f"  {'k':>3} {'n_bands':>8} {'max|ΔΣ_X|/eV':>14} {'argmax n':>10}")
    for k, n, mx, ami in rows:
        print(f"  {k:3d} {n:8d} {mx:14.6e} {int(ami):10d}")
    print()
    print(f"GLOBAL  max |ΔΣ_X| = {max_abs:.6e} eV  at (k, n) = {max_loc}")

    # Detail at the worst location:
    if max_loc != (None, None):
        k, n = max_loc
        print(f"\nDetail at worst location k={k}, n={n}:")
        print(f"  A (IBZ)     : {A[k][n]:.8f} eV")
        print(f"  B (full-BZ) : {B[k][n]:.8f} eV")
        print(f"  diff (A-B)  : {A[k][n] - B[k][n]:.8e} eV")

    # Print first 8 bands at k=0 side-by-side (matches "Bare Σ_X" line in gw.out)
    print(f"\nFirst 8 bands at k=0 (matches gw.out 'Bare Σ_X' print):")
    print(f"  {'n':>3} {'A (eV)':>14} {'B (eV)':>14} {'A-B (eV)':>14}")
    for i in range(min(8, A[0].size, B[0].size)):
        print(f"  {i:3d} {A[0][i]:14.6f} {B[0][i]:14.6f} {A[0][i]-B[0][i]:14.6e}")

    # Verdict
    print()
    if max_abs < 1e-9:
        print(f"VERDICT: BIT-EQUAL (max |ΔΣ_X| = {max_abs:.3e} eV < 1e-9)")
    elif max_abs < 1e-6:
        print(f"VERDICT: ACCEPTABLE-SMALL-DRIFT (max |ΔΣ_X| = {max_abs:.3e} eV, sub-µeV)")
    elif max_abs < 1e-3:
        print(f"VERDICT: SMALL-DRIFT (max |ΔΣ_X| = {max_abs:.3e} eV, sub-meV — review)")
    else:
        print(f"VERDICT: REAL-BUG (max |ΔΣ_X| = {max_abs:.3e} eV, ≥ 1 meV)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
