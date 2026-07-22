"""Why does makeVq-vs-disk fail?  Probe a few q with BOTH zeta files.

Rebuilds V(q) = conj(A) A^T, A = zeta~ sqrt(v), from (a) the regenerated
FULL-BZ zeta and (b) the PRODUCTION IBZ zeta, and compares each against the
production V_qmunu tile.  If (b) also disagrees, the mismatch is a convention
in the rebuild, not in the regenerated fit.

usage: python3 zeta_probe.py <restart.h5> <zeta_full.h5> <zeta_ibz.h5>
"""
import os
import sys

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "1")


def main():
    restart, zfull, zibz = sys.argv[1], sys.argv[2], sys.argv[3]
    import h5py
    from bse import vq_interp as vqi

    zx = vqi.load_zeta_coarse(restart, zfull)
    v_all = vqi.v_sphere_padded(zx)
    print(f"[probe] full-BZ zeta: nq={zx['nq']} ngk[0]={zx['ngk'][0]}", flush=True)

    fi = h5py.File(zibz, "r")
    ZG_ibz = fi["zeta_q_G"]
    ngk_ibz = fi["isdf_header/ngk"][()]
    gv_ibz = fi["isdf_header/gvec_components"][()].astype(np.int64)
    print(f"[probe] IBZ zeta:     nq={ZG_ibz.shape[0]} ngk[0]={ngk_ibz[0]}",
          flush=True)

    # map IBZ slab -> full-BZ q by matching the stored sphere G-list
    def match(j):
        n = int(ngk_ibz[j])
        gj = gv_ibz[j][:, :n]
        for i in range(zx["nq"]):
            if int(zx["ngk"][i]) != n:
                continue
            if np.array_equal(zx["gvec"][i][:, :n], gj):
                return i
        return -1

    for j in range(4):
        i = match(j)
        n = int(ngk_ibz[j])
        Vd = zx["Vqmunu"][i] if i >= 0 else None
        A_f = zx["ZG"][i][:, :n] * np.sqrt(v_all[i][:n])[None, :] if i >= 0 else None
        A_i = np.asarray(ZG_ibz[j][:, :n]) * np.sqrt(v_all[i][:n])[None, :] if i >= 0 else None
        if i < 0:
            print(f"[probe] IBZ slab {j}: no sphere match in the full-BZ file")
            continue
        Vf = np.conj(A_f) @ A_f.T
        Vi = np.conj(A_i) @ A_i.T
        nd = np.linalg.norm(Vd)
        print(f"[probe] IBZ slab {j} -> full q {i}  q={zx['qfr'][i]}  ngk={n}")
        print(f"          ||V_disk||             {nd:.6e}")
        print(f"          ||V_make(full zeta)||  {np.linalg.norm(Vf):.6e}  "
              f"relF {np.linalg.norm(Vf-Vd)/nd:.6e}")
        print(f"          ||V_make(IBZ zeta)||   {np.linalg.norm(Vi):.6e}  "
              f"relF {np.linalg.norm(Vi-Vd)/nd:.6e}")
        print(f"          zeta full-vs-IBZ relF  "
              f"{np.linalg.norm(A_f-A_i)/np.linalg.norm(A_i):.6e}", flush=True)
        # is the disagreement a pure scale?
        s = np.vdot(Vd, Vf).real / np.vdot(Vd, Vd).real
        print(f"          best scale <Vd,Vf>/<Vd,Vd> = {s:.6e}; residual after "
              f"scaling {np.linalg.norm(Vf/s-Vd)/nd:.3e}", flush=True)


if __name__ == "__main__":
    main()
