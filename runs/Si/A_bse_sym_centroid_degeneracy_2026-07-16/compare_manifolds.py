#!/usr/bin/env python3
"""Post-process results_old.json + results_sym.json into an old-vs-sym
degeneracy-splitting table.

Manifold boundaries are DEFINED by the symmetric spectrum (gap < 1 meV
clustering), because the symmetry-obeying centroids are what restore the
exact multiplet structure.  The SAME lowest-state index groups are then
applied to the old spectrum to measure how much the literal (--no-orbit)
centroids lift each would-be-degenerate manifold.

Host-only: numpy + json, no GPU / no jax.
"""
import json
import sys
import numpy as np

RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/A_bse_sym_centroid_degeneracy_2026-07-16"


def cluster(ev_eV, gap_meV=1.0):
    gap = gap_meV * 1e-3
    groups, cur = [], [0]
    for i in range(1, len(ev_eV)):
        if ev_eV[i] - ev_eV[i - 1] > gap:
            groups.append(cur); cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


def spread_ueV(ev_eV, idx):
    sub = ev_eV[idx]
    return (sub.max() - sub.min()) * 1e6


def main():
    old = json.load(open(f"{RUN}/results_old.json"))
    sym = json.load(open(f"{RUN}/results_sym.json"))
    ev_old = np.array(old["lowest_20_eV"])
    ev_sym = np.array(sym["lowest_20_eV"])

    # Manifolds defined by the symmetric spectrum.
    groups = cluster(ev_sym, gap_meV=1.0)

    rows = []
    for gi, g in enumerate(groups[:4]):
        idx = np.array(g)
        rows.append({
            "manifold": gi,
            "size": len(g),
            "sym_mean_eV": float(ev_sym[idx].mean()),
            "sym_split_ueV": float(spread_ueV(ev_sym, idx)),
            "old_mean_eV": float(ev_old[idx].mean()),
            "old_split_ueV": float(spread_ueV(ev_old, idx)),
        })

    print(f"Old centroids : {old.get('n_mu_padded')} mu (padded), "
          f"N={old['N']}, grid={old['grid']}")
    print(f"Sym centroids : {sym.get('n_mu_padded')} mu (padded), "
          f"N={sym['N']}, grid={sym['grid']}")
    print()
    print("Manifolds defined by symmetric spectrum (gap<1meV); "
          "same lowest-state index groups applied to old.")
    print(f"{'mfd':>3} {'size':>4} {'sym_mean_eV':>12} "
          f"{'sym_split_ueV':>14} {'old_split_ueV':>14}")
    for r in rows:
        print(f"{r['manifold']:>3} {r['size']:>4} {r['sym_mean_eV']:>12.6f} "
              f"{r['sym_split_ueV']:>14.3f} {r['old_split_ueV']:>14.3f}")

    print("\nLowest-8 eigenvalues (eV):")
    print("  old:", np.array2string(ev_old[:8], precision=6))
    print("  sym:", np.array2string(ev_sym[:8], precision=6))

    json.dump({"rows": rows,
               "ev_old_lowest20": ev_old.tolist(),
               "ev_sym_lowest20": ev_sym.tolist()},
              open(f"{RUN}/comparison.json", "w"), indent=2)
    print(f"\nwrote {RUN}/comparison.json")


if __name__ == "__main__":
    main()
