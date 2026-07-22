"""Pack the exciton deliverables into one npz.

Arrays: the smooth 39-Q path (Q, E, signed path coordinate x with Gamma at 0),
the 11-Q interp/ongrid cross-check pair, and the run-08 on-grid reference.

usage: python3 pack_npz.py <smooth.dat> <check_interp.dat> <run08_ongrid.dat> <out.npz>
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from compare_interp_ongrid import read_dat, wrap                # noqa: E402


def main():
    smooth, check, ongrid, out = sys.argv[1:5]
    Qs, Es = read_dat(smooth)
    Qc, Ec = read_dat(check)
    Qo, Eo = read_dat(ongrid)

    iG = int(np.argmin(np.linalg.norm(wrap(Qs), axis=1)))
    QM, QK = Qs[0], Qs[-1]

    def frac(q, node):
        return float(np.dot(q, node) / np.dot(node, node))

    x = np.array([-frac(Qs[i], QM) if i <= iG else frac(Qs[i], QK)
                  for i in range(len(Qs))])

    # per-Q interp-minus-ongrid on the 11 shared mesh points
    dE = np.full((len(Qc), Ec.shape[1]), np.nan)
    for a in range(len(Qc)):
        d = np.linalg.norm(wrap(wrap(Qc[a])[None, :] - wrap(Qo)), axis=1)
        b = int(np.argmin(d))
        if d[b] < 1e-6:
            dE[a] = (Ec[a] - Eo[b]) * 1e3

    np.savez_compressed(
        out,
        smooth_Q=Qs, smooth_E_eV=Es, smooth_x=x, smooth_iGamma=iG,
        check_Q=Qc, check_E_interp_eV=Ec, check_dE_vs_ongrid_meV=dE,
        run08_ongrid_Q=Qo, run08_ongrid_E_eV=Eo,
    )
    print(f"Wrote {out}")
    print(f"  smooth {Es.shape}  check {Ec.shape}  run08 {Eo.shape}")


if __name__ == "__main__":
    main()
