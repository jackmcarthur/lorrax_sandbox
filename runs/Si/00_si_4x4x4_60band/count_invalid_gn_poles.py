#!/usr/bin/env python3
"""Count invalid GN-GPP poles (Re wtilde2 < 0) exactly as BGW sigma does.

Replicates sources/BerkeleyGW/Sigma/mtxel_cor.f90:805-844 (freq_dep=3):
    I_epsRggp(f) = delta(G,G') - epsinv_GG'(q, f)          [f = 1 static, 2 imaginary]
    wtilde2      = |dFreqBrd(2)|^2 * I2 / (I1 - I2)
    skip if all(|I_epsRggp|) < TOL_Small (= 1e-6, "no screening" cycle)
    invalid if Re(wtilde2) < 0
The |dFreqBrd(2)|^2 prefactor is real-positive and the ind/ph symmetry phases
cancel in the ratio (diagonal: |ph|^2 = 1; off-diagonal: common factor), so the
per-q invalid COUNT computed from the stored irreducible-q epsinv matrices is
exactly BGW's count for every symmetry image of that q.

Usage: python3 count_invalid_gn_poles.py eps0mat.h5 epsmat.h5
"""
import sys
import h5py
import numpy as np

TOL_SMALL = 1.0e-6


def count_file(path):
    out = []
    with h5py.File(path, "r") as f:
        freq_dep = int(f["eps_header/freqs/freq_dep"][()])
        nfreq = int(f["eps_header/freqs/nfreq"][()])
        assert freq_dep == 3 and nfreq == 2, (
            f"{path}: expected freq_dep=3 with 2 freqs, got {freq_dep}/{nfreq}")
        freqs = f["eps_header/freqs/freqs"][()]  # (nfreq, 2) in C order
        nmtx = f["eps_header/gspace/nmtx"][()]
        qpts = f["eps_header/qpoints/qpts"][()]
        mat = f["mats/matrix"]  # (nq, nmatrix, nfreq, ncol, nrow, flavor)
        nq = qpts.shape[0]
        for iq in range(nq):
            n = int(nmtx[iq])
            m = mat[iq, 0]  # (nfreq, ncol, nrow, 2)
            e1 = m[0, :n, :n, 0] + 1j * m[0, :n, :n, 1]
            e2 = m[1, :n, :n, 0] + 1j * m[1, :n, :n, 1]
            eye = np.eye(n)
            I1 = eye - e1
            I2 = eye - e2
            skip = (np.abs(I1) < TOL_SMALL) & (np.abs(I2) < TOL_SMALL)
            denom = I1 - I2
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = I2 / denom
            considered = ~skip
            invalid = considered & (np.real(ratio) < 0)
            out.append({
                "q": tuple(qpts[iq]),
                "nmtx": n,
                "pairs": int(considered.sum()),
                "invalid": int(invalid.sum()),
            })
    return out, freqs


def main():
    total_pairs = total_invalid = 0
    print(f"{'file':<12} {'q':<34} {'nmtx':>5} {'pairs':>9} {'invalid':>8} {'frac':>8}")
    for path in sys.argv[1:]:
        rows, freqs = count_file(path)
        for r in rows:
            qs = "({:+.4f},{:+.4f},{:+.4f})".format(*r["q"])
            frac = r["invalid"] / r["pairs"] if r["pairs"] else 0.0
            print(f"{path.split('/')[-1]:<12} {qs:<34} {r['nmtx']:>5} "
                  f"{r['pairs']:>9} {r['invalid']:>8} {frac:>8.4f}")
            total_pairs += r["pairs"]
            total_invalid += r["invalid"]
    print(f"\nTOTAL: {total_invalid} invalid of {total_pairs} considered (G,G') pairs "
          f"({100.0 * total_invalid / total_pairs:.2f}%) over irreducible q")


if __name__ == "__main__":
    main()
