#!/usr/bin/env python3
"""PR3 verification: per-band Σ_X / Σ_c diff (post-PR3 − pre-PR3).

Inputs:
  00_lorrax/sigma_freq_debug.dat.pre_pr3   (LORRAX at 796c043 + dft_op fix)
  00_lorrax/sigma_freq_debug.dat.post_pr3  (LORRAX at 8504994 + dft_op fix)

Both files have the same schema:
  k  n  E_dft  Edft-Ef  kin_ion  V_H  x_bare  x_head  sex_0  coh_0  sex_head  coh_head  eqp0  eqp1

PR3 only changes the bispinor ψ k-unfold path (Agent 1 Sites #5/#6/#7).
The mathematically-correct rotation iσ_y·conj(U_spinor) on TRS rows + the
τ-phase on Gkk for TRS rows affect Σ_X, Σ_c, and Σ_head at the k-points
reached via TRS.

Expectation:
  - Non-TRS k (full-BZ 0, 2, 6, 7, 8): diff should be at floating-point noise
    (the ψ-rotation matrix is unchanged on those rows).
  - TRS k (full-BZ 1, 3, 4, 5): diff is the PR3 signal; ~mΩ-meV scale is
    expected for a 3×3 grid with no inversion.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RUNDIR = Path(__file__).parent
PRE = RUNDIR / "00_lorrax" / "sigma_freq_debug.dat.pre_pr3"
POST = RUNDIR / "00_lorrax" / "sigma_freq_debug.dat.post_pr3"

TRS_K = {1, 3, 4, 5}
BAND_MIN, BAND_MAX = 19, 30  # sigma's band_index_min/max (1-indexed)


def parse(path):
    rows = []
    for line in path.open():
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
        rows.append({
            "k": k, "n": n, "n_phys": n + 1,
            "E_dft": float(p[2]),
            "x_bare": float(p[6]),
            "x_head": float(p[7]),
            "sex_0": float(p[8]),
            "coh_0": float(p[9]),
            "sex_head": float(p[10]),
            "coh_head": float(p[11]),
            "eqp0": float(p[12]),
        })
    return rows


def main():
    pre = parse(PRE)
    post = parse(POST)
    assert len(pre) == len(post)

    print(
        "# PR3 ψ-side TRS-fix diff (post-PR3 − pre-PR3)\n"
        "# sigma's band window: physical bands 19..30 (n_phys)\n"
        "# TRS k full-BZ indices: {1, 3, 4, 5}\n"
        "#"
    )

    fields = ["x_bare", "x_head", "sex_0", "coh_0", "sex_head", "coh_head", "eqp0"]

    # Per-band table
    print(
        "# k  n_phys  Δx_bare  Δx_head  Δsex_0  Δcoh_0  Δsex_h  Δcoh_h  Δeqp0   TRS"
    )
    for a, b in zip(pre, post):
        assert a["k"] == b["k"] and a["n"] == b["n"]
        if not (BAND_MIN <= a["n_phys"] <= BAND_MAX):
            continue
        diffs = {f: b[f] - a[f] for f in fields}
        trs = "TRS" if a["k"] in TRS_K else "   "
        print(
            f"  {a['k']:2d}    {a['n_phys']:3d}  "
            + "  ".join(f"{diffs[f]:+9.6f}" for f in fields)
            + f"  {trs}"
        )

    print()
    # Group stats
    for label, kgroup in [("TRS", TRS_K), ("non-TRS", set(range(9)) - TRS_K)]:
        for f in fields:
            d = np.array([b[f] - a[f] for a, b in zip(pre, post)
                          if a["k"] in kgroup and BAND_MIN <= a["n_phys"] <= BAND_MAX])
            if d.size == 0:
                continue
            print(
                f"# {label:8s}  Δ{f:9s}  N={d.size:3d}  "
                f"mean={d.mean():+.6f}  max|Δ|={np.abs(d).max():.6f}  rms={np.sqrt(np.mean(d**2)):.6f}"
            )
        print()


if __name__ == "__main__":
    main()
