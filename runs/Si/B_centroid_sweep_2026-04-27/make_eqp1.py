"""Generate a BGW-format eqp1.dat from the centroid-sweep restart enk_full.

Applies a simple uniform scissors to push the conduction bands up by
the per-N COHSEX Σ-shift at the gap (sigTOT(CBM) - sigTOT(VBM)).  Valence
bands stay at E_DFT.  This is *not* a real GW correction but it lets BSE
produce physically sensible exciton energies (~3 eV for Si) so the
N_μ-convergence story is comparable across runs.

Output: eqp1.dat in the same directory as the input cohsex.in.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import h5py
import numpy as np


def _find_restart(input_dir: str) -> str:
    import glob
    candidates = sorted(glob.glob(os.path.join(input_dir, "tmp", "isdf_tensors_*.h5")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(input_dir, "isdf_tensors_*.h5")))
    if not candidates:
        raise FileNotFoundError("no isdf_tensors_*.h5 in restart dir")
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="centroid run dir (containing cohsex.in + tmp/)")
    ap.add_argument("--scissors-eV", type=float, required=True,
                    help="uniform conduction-band scissors in eV")
    ap.add_argument("--n-occ", type=int, default=4, help="number of occupied bands")
    ap.add_argument("--n-bands", type=int, default=60, help="bands to write")
    args = ap.parse_args()

    restart = _find_restart(args.run_dir)
    with h5py.File(restart, "r") as f:
        enk = np.asarray(f["enk_full"][:])

    # k-coords in crystal frac: read from WFN.h5
    wfn_path = os.path.join(args.run_dir, "WFN.h5")
    with h5py.File(wfn_path, "r") as f:
        kpts_frac = np.asarray(f["mf_header/kpoints/rk"][:])  # (nk, 3) crystal frac

    nk = enk.shape[0]
    nb = min(args.n_bands, enk.shape[1])
    ry2ev = 13.6056980659
    enk_dft_ev = enk[:, :nb] * ry2ev
    enk_qp_ev = enk_dft_ev.copy()
    # Conduction-band scissors: bands [n_occ, ...) get + scissors
    enk_qp_ev[:, args.n_occ:] += args.scissors_eV

    out_path = os.path.join(args.run_dir, "eqp1.dat")
    with open(out_path, "w") as f:
        for ik in range(nk):
            kx, ky, kz = kpts_frac[ik]
            f.write(f"  {kx:.9f}  {ky:.9f}  {kz:.9f}  {nb:6d}\n")
            for ib in range(nb):
                # spin=1, band=ib+1, E_DFT_eV, E_QP_eV
                f.write(f"  {1:6d}  {ib+1:6d}  {enk_dft_ev[ik, ib]:13.9f}  {enk_qp_ev[ik, ib]:13.9f}\n")

    print(f"  Wrote {out_path}: nk={nk} nb={nb} scissors={args.scissors_eV:.4f} eV", flush=True)


if __name__ == "__main__":
    main()
