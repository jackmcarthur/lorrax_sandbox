#!/usr/bin/env python3
"""Extract the PR3 baseline Σ_X table from the post-PR2 LORRAX cohsex run.

Output: prints the per-(k, n) bare-exchange Σ_X (col `x_bare` in
``sigma_freq_debug.dat``) for the band-index range covered by sigma
(BGW band_index_min=19, max=30 → physical bands 19..30), grouped by
whether the full-BZ k was reached via TRS in the wfn unfold.

TRS k indices for MoS2 3x3 SOC (no_t_rev=True, D3h):
    full-BZ k=1, 3, 4, 5  (sym_idx_k == 2, i.e. >= ntran=2)

Reads ``00_lorrax/sigma_freq_debug.dat`` and writes the summary table
to stdout.  Save the output for the PR3 verification diff.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RUNDIR = Path(__file__).parent
# Pre-PR3 (796c043) is the task-spec baseline; *.post_pr3 file also
# present for comparison.  See pr3_diff_summary.py for the diff.
DBG = RUNDIR / "00_lorrax" / "sigma_freq_debug.dat.pre_pr3"

# From sym_analysis.log: TRS-unfolded full-BZ k indices
TRS_K = {1, 3, 4, 5}
# BGW band_index_min/max in sigma.inp (1-indexed); convert to LORRAX
# 0-indexed (col n).
BAND_MIN, BAND_MAX = 19, 30  # inclusive, 1-indexed (physical bands)


def main() -> None:
    rows = []
    with DBG.open() as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("k-point"):
                continue
            p = s.split()
            if len(p) < 14:
                continue
            try:
                k = int(p[0])
                n = int(p[1])
            except ValueError:
                continue
            n_phys = n + 1  # LORRAX 0-indexed → physical 1-indexed
            if not (BAND_MIN <= n_phys <= BAND_MAX):
                continue
            rec = {
                "k": k,
                "n": n,
                "n_phys": n_phys,
                "E_dft": float(p[2]),
                "x_bare": float(p[6]),
                "x_head": float(p[7]),
                "sex_0": float(p[8]),
                "coh_0": float(p[9]),
                "trs": k in TRS_K,
            }
            rows.append(rec)

    print(
        "# PR3 baseline Σ_X (= x_bare + x_head) for MoS2 3x3 SOC\n"
        "# Source: 00_lorrax/sigma_freq_debug.dat\n"
        "# TRS-unfolded full-BZ k: {1, 3, 4, 5} (sym_idx_k>=ntran=2)\n"
        "#\n"
        "# k    n_phys  E_dft         x_bare         x_head         Σ_X_total      TRS\n"
        "# (n_phys = LORRAX n+1 = physical band index, 1-indexed)"
    )
    for r in rows:
        sigx = r["x_bare"] + r["x_head"]
        trs_str = "TRS" if r["trs"] else "   "
        print(
            f"  {r['k']:2d}     {r['n_phys']:3d}    "
            f"{r['E_dft']:+10.6f}   "
            f"{r['x_bare']:+14.8f}   "
            f"{r['x_head']:+14.8f}   "
            f"{sigx:+14.8f}    {trs_str}"
        )

    # Summary by TRS / non-TRS group
    x_trs = np.array([r["x_bare"] + r["x_head"] for r in rows if r["trs"]])
    x_nontrs = np.array([r["x_bare"] + r["x_head"] for r in rows if not r["trs"]])
    print()
    print("# Group statistics (mean over included (k, n) entries):")
    print(f"#   TRS k group    : N={len(x_trs):3d}  mean Σ_X = {x_trs.mean():+.6f} eV")
    print(f"#   non-TRS k group: N={len(x_nontrs):3d}  mean Σ_X = {x_nontrs.mean():+.6f} eV")
    print()
    print("# PR3 prediction: ψ-side iσ_y fix only re-rotates the TRS-side")
    print("#   U_spinor rows AND adds the τ-phase to TRS-row Gkk; so only")
    print("#   TRS-k Σ_X is affected.  10-100 meV shifts expected on those.")


if __name__ == "__main__":
    main()
