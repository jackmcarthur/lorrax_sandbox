#!/usr/bin/env python3
"""R3 verification: compare MoS2 run_A_ibz_postfix x_bare at Γ (k=0) vs BGW.

BGW sigma_hp.log has band column = X (col 3 in 1-based; col index 3 from p[3] in the
SKILL.md parser). The LORRAX 'x_bare' is the bare exchange Σ_X = -Σ_{occ} M²·v.
The BGW 'X' column is the SAME quantity (bare exchange).

Band mapping: LORRAX n is 0-indexed; BGW is 1-indexed. Physical band = LORRAX n + 1.
"""
from __future__ import annotations
import re
import numpy as np


def parse_sigma_hp(path):
    """SKILL.md §2a parser, lightly trimmed."""
    blocks = []
    ik = None
    kcrys = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k\s*=\s*([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+([\d.Ee+-]+)\s+ik\s*=\s*(\d+)', s)
        if m:
            kcrys = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            ik = int(m.group(4))
            continue
        if ik is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            if not any(b.get('ik') == ik for b in blocks):
                blocks.append({'kcrys': kcrys, 'ik': ik, 'bands': {}})
            blocks[-1]['bands'][n] = {
                'X': float(p[3]), 'SXmX': float(p[4]), 'CH': float(p[5]),
            }
    return blocks


def load_lorrax_x_bare_at_k(path, k_idx=0):
    """Load LORRAX sigma_freq_debug.dat, return {physical_band: x_bare} at given k."""
    bands = {}
    cols = None
    for line in open(path):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") and "x_bare" in s:
            cols = [c.strip() for c in s.lstrip("#").strip().split("\t") if c.strip()]
            continue
        if s.startswith("#") or s.startswith("k-point"):
            continue
        parts = s.split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        k = int(round(vals[0]))
        n = int(round(vals[1]))
        if k != k_idx:
            continue
        idx_x = cols.index("x_bare")
        n_phys = n + 1
        bands[n_phys] = vals[idx_x]
    return bands


def main():
    bgw = parse_sigma_hp("/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/00_bgw_cohsex/sigma_hp.log")
    # Find the Γ block (ik=1, kcrys≈(0,0,0))
    gblk = next(b for b in bgw if b['ik'] == 1)
    print(f"BGW ik=1 kcrys={gblk['kcrys']}, bands present: {sorted(gblk['bands'].keys())}")

    lor = load_lorrax_x_bare_at_k(
        "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/D_ibz_vs_fullbz_same_basis_2026-05-14/run_A_ibz_postfix/sigma_freq_debug.dat",
        k_idx=0,
    )
    print(f"LORRAX k=0 bands: {min(lor)}..{max(lor)} ({len(lor)} entries)")

    print(f"\n{'band':>4} {'BGW_X (eV)':>14} {'LORRAX_x_bare (eV)':>22} {'Δ (LORRAX-BGW)':>18}")
    bands = sorted(set(gblk['bands'].keys()) & set(lor.keys()))
    diffs = []
    for n in bands:
        bx = gblk['bands'][n]['X']
        lx = lor[n]
        d = lx - bx
        diffs.append((n, bx, lx, d))
        print(f"{n:>4} {bx:>14.6f} {lx:>22.6f} {d:>+18.6f}")

    arr = np.array([d[3] for d in diffs])
    nabs = np.abs(arr)
    imax = int(np.argmax(nabs))
    nmax = diffs[imax][0]
    print(f"\nGlobal max |Δ(x_bare)| = {nabs.max()*1000:.3f} meV  at band {nmax}")
    print(f"MAE |Δ| = {nabs.mean()*1000:.3f} meV")

    GATE = 0.070  # 70 meV
    pass_ = nabs.max() <= GATE
    print(f"\nR3 GATE (≤ 70 meV): {'PASS' if pass_ else 'FAIL'}")


if __name__ == "__main__":
    main()
