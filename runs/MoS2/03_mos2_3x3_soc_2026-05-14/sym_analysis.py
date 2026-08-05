#!/usr/bin/env python3
"""Symmetry analysis for MoS2 3x3 SOC.

Reports:
- wfn.ntran                          : spatial sym op count
- len(sym.sym_mats_k)                 : with TRS, should be 2*ntran
- sum(sym.sym_idx_k >= ntran)         : k's needing TRS for unfold
- sum(sym.sym_idx_q >= ntran)         : q's needing TRS for fold
- has_inversion                       : -I in wfn.sym_matrices[:ntran]?

This run is the PR3 baseline; if has_inversion=True, stop and report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# wfn-loader symmetry maps live in the LORRAX source tree
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_B/src")

from file_io.wfn_loader import WfnLoader  # noqa: E402
from common.symmetry_maps import SymMaps  # noqa: E402


def main() -> None:
    wfn_path = Path(__file__).parent / "qe" / "nscf" / "WFN.h5"
    print(f"Loading WFN: {wfn_path}")
    wfn = WfnLoader(str(wfn_path), backend="eager")
    sym = SymMaps(wfn)

    ntran = int(wfn.ntran)
    n_total_sym = int(len(sym.sym_mats_k))
    n_trs_k = int(np.sum(sym.sym_idx_k >= ntran))
    n_trs_q = int(np.sum(sym.sym_idx_q >= ntran))
    has_inv = any(
        np.array_equal(m, -np.eye(3, dtype=int))
        for m in sym.sym_matrices
    )

    print()
    print(f"  wfn.ntran                = {ntran:4d}  (spatial sym ops, no TRS)")
    print(f"  len(sym.sym_mats_k)      = {n_total_sym:4d}  (should be 2*ntran with TRS)")
    print(f"  #k needing TRS           = {n_trs_k:4d}  (sym_idx_k >= ntran)")
    print(f"  #q needing TRS           = {n_trs_q:4d}  (sym_idx_q >= ntran)")
    print(f"  has_inversion (-I)       = {has_inv}")
    print()

    print(f"  wfn.kpoints (IBZ, {wfn.nkpts} points):")
    for ik, k in enumerate(wfn.kpoints):
        print(f"    [{ik}]  {k[0]:+.6f}  {k[1]:+.6f}  {k[2]:+.6f}")
    print()

    print(f"  unfolded full-BZ k-points ({len(sym.unfolded_kpts)} total), sym_idx_k:")
    for i, (k, sidx, irr) in enumerate(zip(sym.unfolded_kpts, sym.sym_idx_k, sym.irr_idx_k)):
        trs = "TRS" if sidx >= ntran else "    "
        print(f"    full[{i:2d}]  ({k[0]:+.4f},{k[1]:+.4f},{k[2]:+.4f})"
              f"  ← IBZ[{int(irr)}] via sym[{int(sidx)}] {trs}")
    print()

    print(f"  full-BZ q-points ({len(sym.sym_idx_q)} total), sym_idx_q:")
    for iq, sidx in enumerate(sym.sym_idx_q):
        trs = "TRS" if sidx >= ntran else "    "
        irr = int(sym.irr_idx_q[iq])
        print(f"    q[{iq:2d}]  sym[{int(sidx):2d}] {trs}  ← q_irr[{irr}]")
    print()

    print("  spatial sym_matrices (wfn convention):")
    for i, m in enumerate(sym.sym_matrices):
        flat = " ".join(f"{int(v):+d}" for v in m.flatten())
        print(f"    [{i:2d}]  {flat}")
    print()

    # PR3 prerequisites
    if has_inv:
        print("STOP: MoS2 SOC SHOWS INVERSION SYMMETRY -- not suitable for PR3 test bed.")
        sys.exit(2)
    if n_trs_k == 0:
        print("STOP: no k-points reach via TRS; PR3 fix would have NO effect here.")
        sys.exit(3)
    print(f"OK: non-inversion ({has_inv=}) and {n_trs_k} TRS-folded k-points")
    print("    --> suitable load-bearing test bed for PR3 (Agent 1 #5/#6/#7).")


if __name__ == "__main__":
    main()
