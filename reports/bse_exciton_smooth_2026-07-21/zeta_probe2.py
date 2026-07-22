"""Is the regenerated FULL-BZ zeta a PERMUTATION of the (correct) IBZ zeta,
or a different fit?

Compares a thin G-slice of every slab in both files.  For each IBZ slab j we
report (a) the residual against the full-BZ slab that carries the SAME sphere
(the sphere identifies the q), and (b) the best-matching full-BZ slab anywhere.
If (b) != (a) the writer's zeta slab order and gvec/q order disagree.

usage: python3 zeta_probe2.py <zeta_full.h5> <zeta_ibz.h5> [ncols]
"""
import sys

import h5py
import numpy as np


def main():
    zfull, zibz = sys.argv[1], sys.argv[2]
    nc = int(sys.argv[3]) if len(sys.argv) > 3 else 64

    ff, fi = h5py.File(zfull, "r"), h5py.File(zibz, "r")
    Zf = ff["zeta_q_G"][:, :, :nc]
    Zi = fi["zeta_q_G"][:, :, :nc]
    gf = ff["isdf_header/gvec_components"][:, :, :nc].astype(np.int64)
    gi = fi["isdf_header/gvec_components"][:, :, :nc].astype(np.int64)
    nf, ni = Zf.shape[0], Zi.shape[0]
    print(f"full nq={nf}  ibz nq={ni}  probe cols={nc}", flush=True)

    Zf2 = Zf.reshape(nf, -1)
    Zi2 = Zi.reshape(ni, -1)
    nrm_f = np.linalg.norm(Zf2, axis=1)
    nrm_i = np.linalg.norm(Zi2, axis=1)
    print(f"slab-norm (thin) full: min {nrm_f.min():.4e} max {nrm_f.max():.4e}")
    print(f"slab-norm (thin) ibz : min {nrm_i.min():.4e} max {nrm_i.max():.4e}")

    print(f"\n{'ibz j':>6} {'same-sphere i':>14} {'relF@i':>11} "
          f"{'best i':>7} {'relF@best':>11}")
    for j in range(min(ni, 12)):
        same = [i for i in range(nf) if np.array_equal(gf[i], gi[j])]
        d = np.linalg.norm(Zf2 - Zi2[j][None, :], axis=1) / nrm_i[j]
        b = int(np.argmin(d))
        si = same[0] if same else -1
        print(f"{j:6d} {si:14d} {(d[si] if si >= 0 else np.nan):11.3e} "
              f"{b:7d} {d[b]:11.3e}", flush=True)

    # Is the full-BZ slab a SCALED version of the IBZ one at the same sphere?
    print("\nper-slab best scale (same sphere):")
    for j in range(min(ni, 6)):
        same = [i for i in range(nf) if np.array_equal(gf[i], gi[j])]
        if not same:
            continue
        i = same[0]
        a, b = Zf2[i], Zi2[j]
        s = np.vdot(b, a) / np.vdot(b, b)
        print(f"  j={j:3d} i={i:3d}  scale={s:.6f}  "
              f"resid_after_scale={np.linalg.norm(a - s*b)/np.linalg.norm(b):.3e}",
              flush=True)


if __name__ == "__main__":
    main()
