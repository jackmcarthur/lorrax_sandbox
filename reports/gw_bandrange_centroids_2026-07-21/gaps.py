"""Direct/indirect DFT and GW gaps from a LORRAX eqp1.dat (BGW eqp format).

eqp1.dat block layout (BGW convention):
  kx ky kz  nbands
  spin band E_dft(eV) E_qp(eV)     x nbands
"""
import os
import sys
import numpy as np

NVAL = int(os.environ.get("GAP_NVAL", "26"))


def parse_eqp(path):
    ks, rows, cur = [], [], None
    for line in open(path):
        if line.startswith('#'):
            continue
        p = line.split()
        if len(p) == 4 and '.' in p[0]:
            ks.append([float(v) for v in p[:3]])
            cur = {}
            rows.append(cur)
        elif len(p) == 4 and cur is not None:
            cur[int(p[1])] = (float(p[2]), float(p[3]))
    return np.asarray(ks), rows


def report(path, label):
    ks, rows = parse_eqp(path)
    nk = len(ks)
    edft = np.array([[rows[k].get(b + 1, (np.nan,) * 2)[0] for b in range(200)]
                     for k in range(nk)])
    eqp = np.array([[rows[k].get(b + 1, (np.nan,) * 2)[1] for b in range(200)]
                    for k in range(nk)])
    out = {}
    for tag, E in (("DFT", edft), ("GW ", eqp)):
        vb = E[:, NVAL - 1]
        cb = E[:, NVAL]
        direct = cb - vb
        kd = int(np.nanargmin(direct))
        ivbm, icbm = int(np.nanargmax(vb)), int(np.nanargmin(cb))
        indirect = cb[icbm] - vb[ivbm]
        out[tag] = (direct[kd], ks[kd], indirect, ks[ivbm], ks[icbm],
                    vb[ivbm], cb[icbm])
        print(f"{label:>22} {tag}: direct {direct[kd]:6.3f} eV @ "
              f"k=({ks[kd][0]:+.3f},{ks[kd][1]:+.3f})  |  indirect "
              f"{indirect:6.3f} eV  VBM@({ks[ivbm][0]:+.3f},{ks[ivbm][1]:+.3f})"
              f" CBM@({ks[icbm][0]:+.3f},{ks[icbm][1]:+.3f})")
    # direct gap at the K point (1/3,1/3) if sampled
    kk = np.argmin(np.linalg.norm(
        (ks[:, :2] - np.array([1 / 3, 1 / 3]) + 0.5) % 1.0 - 0.5, axis=1))
    if np.linalg.norm((ks[kk, :2] - np.array([1 / 3, 1 / 3]) + .5) % 1. - .5) < 1e-6:
        print(f"{label:>22}      K=(1/3,1/3) direct: DFT "
              f"{edft[kk, NVAL] - edft[kk, NVAL-1]:6.3f}  GW "
              f"{eqp[kk, NVAL] - eqp[kk, NVAL-1]:6.3f} eV   "
              f"(QP shift VBM {eqp[kk,NVAL-1]-edft[kk,NVAL-1]:+.3f}, "
              f"CBM {eqp[kk,NVAL]-edft[kk,NVAL]:+.3f})")
    kg = int(np.argmin(np.linalg.norm(ks, axis=1)))
    print(f"{label:>22}      Gamma direct:      DFT "
          f"{edft[kg, NVAL] - edft[kg, NVAL-1]:6.3f}  GW "
          f"{eqp[kg, NVAL] - eqp[kg, NVAL-1]:6.3f} eV   "
          f"(QP shift VBM {eqp[kg,NVAL-1]-edft[kg,NVAL-1]:+.3f}, "
          f"CBM {eqp[kg,NVAL]-edft[kg,NVAL]:+.3f})")
    return out


for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    report(path, label)
    print()
