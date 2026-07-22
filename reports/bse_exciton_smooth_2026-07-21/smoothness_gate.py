"""ABSOLUTE smoothness gate on an ``bse.exciton_bands`` .dat.

The exciton bandstructure is sampled on two equally-spaced legs radiating
from Gamma, so the discrete 2nd difference

    d2(i) = |E(i-1) - 2 E(i) + E(i+1)|

is the natural roughness measure: for a smooth band it is the curvature
times the (uniform) step squared, and for interpolation ringing it is
unbounded.  Reported per leg, for E_1 and for the whole multiplet.

CALIBRATION — what "smooth" costs.  d2 does NOT go to zero for a correct
calculation: real dispersion has curvature, and sorted eigenvalue branches
have kinks wherever two branches approach.  The scale is set by the
2026-07-20 known-good run, whose figure
``reports/bse_multinode_2026-07-20/exciton_bands_16gpu_on.png`` is a
genuinely smooth 40-Q exciton bandstructure; run this script on its .dat to
get the reference number for the same material and the same branch count.
A run is GOOD when it is at or below that reference, not when it reaches
some absolute few-meV target the reference itself does not meet.

--extra-q rows (mode ``extra``) are pulled out of the path and used for the
REFERENCE-FREE symmetry test: a Q and its point-group image must give
identical E_S, so |E_S(Q) - E_S(gQ)| is interpolation error measured
without any external reference at all.

usage: python3 smoothness_gate.py <bands.dat> [--json out.json]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def read_dat(path):
    """(Q, E, mode) for every non-comment row, refit rows dropped."""
    Q, E, mode = [], [], []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        if p[5] == "refit":
            continue
        Q.append([float(p[2]), float(p[3]), float(p[4])])
        E.append([float(x) for x in p[6:]])
        mode.append(p[5])
    return np.asarray(Q), np.asarray(E), np.asarray(mode)


def d2(y):
    y = np.asarray(y, dtype=float)
    if y.shape[0] < 3:
        return np.zeros((0,) + y.shape[1:])
    return np.abs(y[:-2] - 2.0 * y[1:-1] + y[2:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dat")
    ap.add_argument("--json")
    ap.add_argument("--nbranch", type=int, default=0,
                    help="branches to include in the multiplet stats (0=all)")
    ap.add_argument("--nodes", default="",
                    help="comma list of path-node INDICES to split legs at, "
                         "e.g. '0,15,23,39' for a Gamma-M-K-Gamma path.  d2 is "
                         "only meaningful INSIDE a straight segment — across a "
                         "node the direction changes and the kink is real "
                         "geometry, not roughness.  Default (empty) splits at "
                         "Gamma, which is right for the M-Gamma-K path this "
                         "campaign uses.")
    args = ap.parse_args()

    Q, E, mode = read_dat(args.dat)
    is_extra = mode == "extra"
    Qp, Ep = Q[~is_extra], E[~is_extra]
    Qx, Ex = Q[is_extra], E[is_extra]
    nb = args.nbranch or Ep.shape[1]

    # Gamma splits the path; the two legs are the two sides of it.
    iG = int(np.argmin(np.linalg.norm(Qp - np.round(Qp), axis=1)))
    if args.nodes:
        nd = [int(v) for v in args.nodes.split(",")]
        legs = {f"seg{a}-{b}": np.arange(a, b + 1)
                for a, b in zip(nd[:-1], nd[1:])}
    else:
        legs = {"Gamma-M": np.arange(0, iG + 1)[::-1],   # radiate FROM Gamma
                "Gamma-K": np.arange(iG, len(Qp))}

    out = {"dat": args.dat, "n_Q_path": int(len(Qp)), "n_extra": int(len(Qx)),
           "n_branch": int(Ep.shape[1]), "iGamma": iG,
           "E1_Gamma_eV": float(Ep[iG, 0]), "legs": {}}
    print(f"{args.dat}")
    print(f"  {len(Qp)} path Q (+{len(Qx)} extra), {Ep.shape[1]} branches, "
          f"E_1(Gamma) = {Ep[iG, 0]:.6f} eV")
    for name, idx in legs.items():
        g1 = d2(Ep[idx, 0]) * 1e3                      # meV
        gm = d2(Ep[idx][:, :nb]) * 1e3
        rec = {"n_Q": int(len(idx)),
               "E1_d2_max_meV": float(g1.max()) if g1.size else 0.0,
               "E1_d2_mean_meV": float(g1.mean()) if g1.size else 0.0,
               "multiplet_d2_max_meV": float(gm.max()) if gm.size else 0.0,
               "multiplet_d2_mean_meV": float(gm.mean()) if gm.size else 0.0,
               "E1_min_eV": float(Ep[idx, 0].min()),
               "E1_max_eV": float(Ep[idx, 0].max())}
        out["legs"][name] = rec
        print(f"  {name:9s} E_1 2nd-diff max {rec['E1_d2_max_meV']:8.2f} "
              f"mean {rec['E1_d2_mean_meV']:7.2f} meV | {nb}-branch max "
              f"{rec['multiplet_d2_max_meV']:8.2f} mean "
              f"{rec['multiplet_d2_mean_meV']:7.2f} meV")

    # ── reference-free symmetry check ────────────────────────────────────
    # Hexagonal (h,k,0) ↔ (k,h,0) is the σ_v mirror of MoS2's D3h, so any
    # extra Q that is the image of a path Q must reproduce its E_S exactly.
    if len(Qx):
        out["symmetry"] = []
        print("  symmetry images (mirror h<->k), |ΔE| vs the path point:")
        for a in range(len(Qx)):
            img = np.array([Qx[a][1], Qx[a][0], Qx[a][2]])
            d = np.linalg.norm((Qp - img) - np.round(Qp - img), axis=1)
            b = int(np.argmin(d))
            if d[b] > 1e-6:
                continue
            dE = (Ex[a] - Ep[b]) * 1e3
            rec = {"Q_extra": Qx[a].tolist(), "Q_path": Qp[b].tolist(),
                   "dE1_meV": float(dE[0]),
                   "dE_max_meV": float(np.max(np.abs(dE)))}
            out["symmetry"].append(rec)
            print(f"    ({Qx[a][0]:+.5f},{Qx[a][1]:+.5f}) vs "
                  f"({Qp[b][0]:+.5f},{Qp[b][1]:+.5f})   "
                  f"ΔE_1 = {dE[0]:+8.3f} meV   max|ΔE| = "
                  f"{np.max(np.abs(dE)):8.3f} meV")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
