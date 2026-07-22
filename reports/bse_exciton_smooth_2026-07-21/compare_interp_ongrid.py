"""interp-vs-ongrid cross-check for the arbitrary-Q exciton path.

``--vq-mode ongrid`` uses the EXACT stored production exchange tile
V_qmunu[wrap(-Q)]; ``--vq-mode interp`` builds V(Q) from the b26p arbitrary-Q
model in ``bse.vq_interp``.  On any Q that lands on the 12x12 BSE mesh both are
defined, and their difference is the only error the smooth curve's intermediate
points carry that the on-grid curve does not.  This matches the two .dat files
by wrapped Q and tabulates |ΔE_S| per branch.

usage: python3 compare_interp_ongrid.py <interp.dat> <ongrid.dat> <out.json>
"""
import json
import sys

import numpy as np


def read_dat(path):
    """Rows of an ``bse.exciton_bands`` .dat: (Q(3), E(neig))."""
    Q, E = [], []
    for line in open(path):
        if line.startswith("#"):
            continue
        p = line.split()
        if len(p) < 7:
            continue
        Q.append([float(p[2]), float(p[3]), float(p[4])])
        E.append([float(x) for x in p[6:]])
    return np.asarray(Q), np.asarray(E)


def wrap(q):
    return q - np.round(q)


def main():
    fi, fo, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    Qi, Ei = read_dat(fi)
    Qo, Eo = read_dat(fo)
    nbr = min(Ei.shape[1], Eo.shape[1])

    rows = []
    for a in range(len(Qi)):
        d = np.linalg.norm(wrap(wrap(Qi[a])[None, :] - wrap(Qo)), axis=1)
        b = int(np.argmin(d))
        if d[b] > 1e-6:
            continue                     # not an on-grid twin of the other run
        if any(r["iQ_interp"] == a for r in rows):
            continue
        dE = (Ei[a, :nbr] - Eo[b, :nbr]) * 1e3       # meV
        rows.append({
            "iQ_interp": a, "iQ_ongrid": b,
            "Q": [float(x) for x in Qi[a]],
            "E1_interp_eV": float(Ei[a, 0]),
            "E1_ongrid_eV": float(Eo[b, 0]),
            "dE1_meV": float(dE[0]),
            "dE_max_meV": float(np.max(np.abs(dE))),
            "dE_all_meV": [float(x) for x in dE],
        })

    allmax = [r["dE_max_meV"] for r in rows]
    all1 = [abs(r["dE1_meV"]) for r in rows]
    summary = {
        "interp_dat": fi, "ongrid_dat": fo,
        "n_matched_Q": len(rows), "n_branch": int(nbr),
        "dE1_max_meV": float(np.max(all1)) if rows else None,
        "dE1_median_meV": float(np.median(all1)) if rows else None,
        "dE_allbranch_max_meV": float(np.max(allmax)) if rows else None,
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)

    print(f"matched {len(rows)} on-grid Q, {nbr} branches")
    print(f"{'Qx':>9} {'Qy':>9} {'E1 interp':>11} {'E1 ongrid':>11} "
          f"{'dE1 (meV)':>10} {'max|dE| 8br':>12}")
    for r in rows:
        print(f"{r['Q'][0]:9.6f} {r['Q'][1]:9.6f} {r['E1_interp_eV']:11.6f} "
              f"{r['E1_ongrid_eV']:11.6f} {r['dE1_meV']:10.3f} "
              f"{r['dE_max_meV']:12.3f}")
    print(f"\nmax |dE1| = {summary['dE1_max_meV']:.3f} meV   "
          f"median |dE1| = {summary['dE1_median_meV']:.3f} meV   "
          f"max over all {nbr} branches = "
          f"{summary['dE_allbranch_max_meV']:.3f} meV")


if __name__ == "__main__":
    main()
