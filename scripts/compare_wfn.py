#!/usr/bin/env python3
"""Compare two BGW WFN.h5 files: headers, G-vectors, eigenvalues, coefficients.

Usage:
    python compare_wfn.py ref.h5 test.h5
    python compare_wfn.py ref.h5 test.h5 --coeffs   # also compare wavefunctions
"""
import sys
import argparse
import numpy as np
import h5py


def compare_datasets(f1, f2, path, rtol=1e-6, atol=1e-8):
    """Compare a single HDF5 dataset between two files."""
    if path not in f1:
        return "MISSING_IN_REF"
    if path not in f2:
        return "MISSING_IN_TEST"

    d1, d2 = f1[path], f2[path]

    if d1.shape != d2.shape:
        return f"SHAPE_MISMATCH: {d1.shape} vs {d2.shape}"

    a1, a2 = d1[()], d2[()]

    if a1.dtype.kind in ('i', 'u'):  # integer
        if np.array_equal(a1, a2):
            return "EXACT"
        n_diff = int(np.sum(a1 != a2))
        return f"INT_DIFF: {n_diff}/{a1.size} elements differ"

    if a1.dtype.kind == 'f':  # float
        if a1.size == 0:
            return "EMPTY"
        diff = np.abs(a1 - a2)
        mae = float(np.mean(diff))
        max_err = float(np.max(diff))
        if max_err == 0.0:
            return "EXACT"
        rel = max_err / max(float(np.max(np.abs(a1))), 1e-30)
        return f"MAE={mae:.2e}  max={max_err:.2e}  rel={rel:.2e}"

    if a1.dtype.kind in ('S', 'U', 'O'):  # string
        if np.array_equal(a1, a2):
            return "EXACT"
        return "STRING_DIFF"

    return f"UNKNOWN_DTYPE={a1.dtype}"


def collect_paths(group, prefix=""):
    """Recursively collect all dataset paths in an HDF5 group."""
    paths = []
    for key in group:
        full = f"{prefix}/{key}" if prefix else key
        item = group[key]
        if isinstance(item, h5py.Group):
            paths.extend(collect_paths(item, full))
        else:
            paths.append(full)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Compare two BGW WFN.h5 files")
    parser.add_argument("ref", help="Reference WFN.h5 (e.g. from QE/pw2bgw)")
    parser.add_argument("test", help="Test WFN.h5 (e.g. from LORRAX)")
    parser.add_argument("--coeffs", action="store_true",
                        help="Also compare wavefunction coefficients (slow for large files)")
    parser.add_argument("--atol", type=float, default=1e-8,
                        help="Absolute tolerance for 'PASS' (default 1e-8)")
    args = parser.parse_args()

    f1 = h5py.File(args.ref, "r")
    f2 = h5py.File(args.test, "r")

    paths_ref = set(collect_paths(f1))
    paths_test = set(collect_paths(f2))

    all_paths = sorted(paths_ref | paths_test)

    # Skip coefficients unless requested (they're huge)
    if not args.coeffs:
        all_paths = [p for p in all_paths if "coeffs" not in p]

    print(f"REF:  {args.ref}")
    print(f"TEST: {args.test}")
    print(f"{'Dataset':<50s} {'Shape':>20s}  Result")
    print("=" * 100)

    n_exact, n_close, n_diff, n_missing = 0, 0, 0, 0

    for path in all_paths:
        if path in paths_ref and path in paths_test:
            shape_str = str(f1[path].shape)
        elif path in paths_ref:
            shape_str = str(f1[path].shape)
        else:
            shape_str = str(f2[path].shape)

        result = compare_datasets(f1, f2, path)

        if result == "EXACT":
            n_exact += 1
            tag = "  "
        elif result.startswith("MISSING"):
            n_missing += 1
            tag = "!!"
        elif result.startswith("MAE"):
            # Check if it's "close enough"
            mae_val = float(result.split("max=")[1].split()[0])
            if mae_val < args.atol:
                n_close += 1
                tag = "~ "
            else:
                n_diff += 1
                tag = "* "
        else:
            n_diff += 1
            tag = "**"

        print(f"{tag} {path:<48s} {shape_str:>20s}  {result}")

    print("=" * 100)
    print(f"EXACT: {n_exact}  CLOSE: {n_close}  DIFF: {n_diff}  MISSING: {n_missing}")

    f1.close()
    f2.close()


if __name__ == "__main__":
    main()
