#!/usr/bin/env python3
"""Dense BSE eigenvalue degeneracy analysis for the sym-centroid experiment.

Reuses the SAME dense-H builder the Phase-2 gate uses (_build_dense_H from
tests/test_bse_dense_reference.py) on a gw_jax do_screened restart, then
measures intra-manifold eigenvalue splittings.

Usage:
    python3 analyze_degeneracy.py <input_file> <label> <out_json>
"""
import sys
import os
import json

import numpy as np

LROOT = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A"
sys.path.insert(0, os.path.join(LROOT, "src"))
sys.path.insert(0, os.path.join(LROOT, "tests"))

import jax
jax.config.update("jax_enable_x64", True)

from bse import bse_io
# REUSE the gate's dense-H builder (do NOT copy-paste it).
from test_bse_dense_reference import _build_dense_H

RY = 13.6056980659  # eV per Ry


def cluster_eigs(ev_eV, gap_meV=1.0):
    """Cluster sorted eigenvalues; new cluster when gap > gap_meV."""
    gap = gap_meV * 1e-3
    clusters = []
    cur = [0]
    for i in range(1, len(ev_eV)):
        if ev_eV[i] - ev_eV[i - 1] > gap:
            clusters.append(cur)
            cur = [i]
        else:
            cur.append(i)
    clusters.append(cur)
    return clusters


def main():
    inp = sys.argv[1]
    label = sys.argv[2]
    out_json = sys.argv[3]
    n_val = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    n_cond = int(sys.argv[5]) if len(sys.argv) > 5 else 2

    restart = bse_io._find_restart_file(inp)
    print(f"[{label}] restart = {restart}", flush=True)

    data = bse_io._load_ring_subset(
        restart, n_val=n_val, n_cond=n_cond, px=1, py=1, input_file=inp)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nc = int(data["psi_c"].shape[1])
    nv = int(data["psi_v"].shape[1])
    nmu = int(data["V_q0"].shape[-1])
    N = nc * nv * nkx * nky * nkz
    print(f"[{label}] grid=({nkx},{nky},{nkz}) nc={nc} nv={nv} "
          f"N={N} n_mu(padded)={nmu}", flush=True)

    H, D, Kx, Kd = _build_dense_H(data)
    Hh = 0.5 * (H + H.conj().T)
    ev = np.sort(np.linalg.eigvalsh(Hh))  # Ry, ascending
    ev_eV = ev * RY

    clusters = cluster_eigs(ev_eV, gap_meV=1.0)

    # Report the lowest few clusters (manifolds).
    manifolds = []
    for ci, cl in enumerate(clusters[:8]):
        lo = ev_eV[cl[0]]
        hi = ev_eV[cl[-1]]
        split_ueV = (hi - lo) * 1e6
        manifolds.append({
            "manifold": ci,
            "size": len(cl),
            "mean_eV": float(np.mean(ev_eV[cl])),
            "min_eV": float(lo),
            "max_eV": float(hi),
            "max_intra_split_ueV": float(split_ueV),
        })

    out = {
        "label": label,
        "restart": restart,
        "grid": [nkx, nky, nkz],
        "nc": nc, "nv": nv, "N": N, "n_mu_padded": nmu,
        "lowest_20_eV": [float(x) for x in ev_eV[:20]],
        "manifolds": manifolds,
    }
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[{label}] Lowest-8 manifolds (gap<1meV clustering):", flush=True)
    print(f"{'mfd':>3} {'size':>4} {'mean_eV':>12} "
          f"{'max_intra_split_ueV':>22}", flush=True)
    for m in manifolds:
        print(f"{m['manifold']:>3} {m['size']:>4} {m['mean_eV']:>12.6f} "
              f"{m['max_intra_split_ueV']:>22.3f}", flush=True)
    print(f"\n[{label}] wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
