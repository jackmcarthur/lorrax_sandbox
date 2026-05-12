#!/usr/bin/env python3
"""MAE: LORRAX sigTOT (eqp0.dat) vs BGW Sig' (sigma_hp.log col 11).

Si 4x4x4 nosym → both files use the same k-point ordering (64 full BZ
points, no symmetry folding), so we can match row-by-row by (k_index, n).
"""
import re
import sys
import math
from pathlib import Path

def parse_bgw(path):
    """Returns {(k_idx, n): sig_prime} (eV).

    k_idx is 1-based as BGW reports it. n is 1-based.
    """
    out = {}
    k_idx = None
    in_table = False
    for line in Path(path).read_text().splitlines():
        m = re.search(r"k\s*=\s*(\S+)\s+(\S+)\s+(\S+)\s+ik\s*=\s*(\d+)", line)
        if m:
            k_idx = int(m.group(4))
            in_table = False
            continue
        if line.strip().startswith("n"):
            in_table = True
            continue
        if in_table and k_idx is not None and line.strip():
            parts = line.split()
            if len(parts) >= 12 and parts[0].isdigit():
                n = int(parts[0])
                sig_prime = float(parts[11])  # col 11 = Sig`
                out[(k_idx, n)] = sig_prime
    return out

def parse_lorrax(path):
    """Returns {(k_idx, n): sigTOT} (eV). k_idx 1-based, n 1-based."""
    out = {}
    k_idx = None
    for line in Path(path).read_text().splitlines():
        m = re.match(r"k-point\s+(\d+)\s*:", line)
        if m:
            k_idx = int(m.group(1)) + 1  # LORRAX is 0-based
            continue
        m = re.match(r"\s*n=(\d+)\s+sigSX=\s*(\S+)\s+sigCOH=\s*(\S+)\s+sigTOT=\s*(\S+)", line)
        if m and k_idx is not None:
            n = int(m.group(1)) + 1
            sigTOT = float(m.group(4))
            out[(k_idx, n)] = sigTOT
    return out

if __name__ == "__main__":
    bgw_path = sys.argv[1]
    lorrax_path = sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else "LORRAX"
    bgw = parse_bgw(bgw_path)
    lor = parse_lorrax(lorrax_path)
    common = sorted(set(bgw) & set(lor))
    if not common:
        print(f"No matching (k,n) pairs found! BGW={len(bgw)} LORRAX={len(lor)}", file=sys.stderr)
        sys.exit(1)
    diffs = [(lor[kn] - bgw[kn]) for kn in common]
    abs_diffs = [abs(d) for d in diffs]
    mae = sum(abs_diffs) / len(abs_diffs)
    rmse = math.sqrt(sum(d * d for d in diffs) / len(diffs))
    max_d = max(abs_diffs)
    print(f"# {label}: N = {len(common)} (k,n) pairs ({len(bgw)} BGW, {len(lor)} LORRAX)")
    print(f"MAE   = {mae*1000:.3f} meV")
    print(f"RMSE  = {rmse*1000:.3f} meV")
    print(f"max|Δ| = {max_d*1000:.3f} meV")
    print(f"mean Δ = {sum(diffs)/len(diffs)*1000:.3f} meV (signed)")
    # Top 5 largest absolute deviations
    ranked = sorted(zip(abs_diffs, common, diffs), reverse=True)[:5]
    print("\nTop 5 |Δ|:")
    print(f"{'k':>4} {'n':>4} {'BGW Sig`':>14} {'LORRAX sigTOT':>16} {'Δ (meV)':>10}")
    for ad, (k, n), d in ranked:
        print(f"{k:>4} {n:>4} {bgw[(k,n)]:>14.6f} {lor[(k,n)]:>16.6f} {d*1000:>10.3f}")
