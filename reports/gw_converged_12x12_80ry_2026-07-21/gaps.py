"""Direct/indirect DFT and GW gaps from a LORRAX eqp0/eqp1.dat (BGW eqp format).

Extends reports/gw_bandrange_centroids_2026-07-21/gaps.py (the established
sandbox eqp parser) with the two things this campaign has to answer:

  * WHERE the VBM sits -- Gamma or K -- and by how much, since the 6x6/30 Ry
    run put it at Gamma by 0.06 eV and a converged 12x12/80 Ry grid should
    move it to K.
  * per-k-point QP shifts at both Gamma and K, so the gap table can be read
    against the BGW anchor and the literature band.

eqp block layout (BGW convention):
  kx ky kz  nbands
  spin band E_dft(eV) E_qp(eV)     x nbands
"""
import os
import sys
import json

import numpy as np

NVAL = int(os.environ.get("GAP_NVAL", "26"))
NBMAX = int(os.environ.get("GAP_NBMAX", "400"))
K_HI = np.array([1.0 / 3.0, 1.0 / 3.0])
GAMMA = np.array([0.0, 0.0])


def parse_eqp(path):
    """-> (nk,3) k-points, list of {band(1-based): (E_dft, E_qp)} per k."""
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


def _wrap(dk):
    return (dk + 0.5) % 1.0 - 0.5


def _ik(ks, target):
    d = np.linalg.norm(_wrap(ks[:, :2] - target), axis=1)
    return int(np.argmin(d)), float(d.min())


def report(path, label, out=None):
    ks, rows = parse_eqp(path)
    nk = len(ks)
    nb = max(max(r) for r in rows)
    nb = min(nb, NBMAX)
    edft = np.full((nk, nb), np.nan)
    eqp = np.full((nk, nb), np.nan)
    for k in range(nk):
        for b, (ed, eq) in rows[k].items():
            if b <= nb:
                edft[k, b - 1] = ed
                eqp[k, b - 1] = eq

    res = {"file": path, "label": label, "nk": nk, "nb": nb}
    ikG, dG = _ik(ks, GAMMA)
    ikK, dK = _ik(ks, K_HI)
    res["k_index"] = {"Gamma": ikG, "K": ikK}
    res["k_match_err"] = {"Gamma": dG, "K": dK}

    print(f"=== {label}   ({path})")
    print(f"    nk={nk}  bands parsed 1..{nb}   "
          f"Gamma=k#{ikG} (|dk|={dG:.1e})  K=k#{ikK} (|dk|={dK:.1e})")

    for tag, E in (("DFT", edft), ("GW", eqp)):
        vb = E[:, NVAL - 1]
        cb = E[:, NVAL]
        direct = cb - vb
        kd = int(np.nanargmin(direct))
        ivbm, icbm = int(np.nanargmax(vb)), int(np.nanargmin(cb))
        indirect = cb[icbm] - vb[ivbm]

        # Which high-symmetry point owns the VBM / CBM?
        def where(i):
            dg = np.linalg.norm(_wrap(ks[i, :2] - GAMMA))
            dk = np.linalg.norm(_wrap(ks[i, :2] - K_HI))
            if dg < 1e-6:
                return "Gamma"
            if dk < 1e-6:
                return "K"
            return f"({ks[i,0]:+.3f},{ks[i,1]:+.3f})"

        # The load-bearing convergence signature: VBM(K) - VBM(Gamma).
        # > 0 means the VBM is at K (the physical monolayer answer).
        vbm_split = float(vb[ikK] - vb[ikG])
        cbm_split = float(cb[ikK] - cb[ikG])

        res[tag] = dict(
            direct=float(direct[kd]), direct_at=where(kd),
            indirect=float(indirect), vbm_at=where(ivbm), cbm_at=where(icbm),
            direct_K=float(cb[ikK] - vb[ikK]),
            direct_Gamma=float(cb[ikG] - vb[ikG]),
            vbm_K_minus_Gamma=vbm_split,
            cbm_K_minus_Gamma=cbm_split,
            vbm=float(vb[ivbm]), cbm=float(cb[icbm]),
        )
        print(f"    {tag:>3}: direct {direct[kd]:7.4f} eV @ {where(kd):>18s}"
              f"   indirect {indirect:7.4f} eV"
              f"   VBM@{where(ivbm):>18s}  CBM@{where(icbm):>18s}")
        print(f"         direct@K {cb[ikK]-vb[ikK]:7.4f}   "
              f"direct@Gamma {cb[ikG]-vb[ikG]:7.4f}   "
              f"VBM(K)-VBM(Gamma) {vbm_split:+7.4f} eV  "
              f"-> VBM at {'K' if vbm_split > 0 else 'Gamma'}")

    for nm, ik in (("Gamma", ikG), ("K", ikK)):
        dv = eqp[ik, NVAL - 1] - edft[ik, NVAL - 1]
        dc = eqp[ik, NVAL] - edft[ik, NVAL]
        print(f"    QP shift @{nm:>5s}:  VBM {dv:+7.4f}   CBM {dc:+7.4f}   "
              f"gap opening {dc-dv:+7.4f} eV")
        res[f"qp_shift_{nm}"] = {"vbm": float(dv), "cbm": float(dc),
                                 "gap_opening": float(dc - dv)}
    print()
    if out is not None:
        out.append(res)
    return res


if __name__ == "__main__":
    collected = []
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        report(path, label, collected)
    dump = os.environ.get("GAP_JSON")
    if dump:
        with open(dump, "w") as fh:
            json.dump(collected, fh, indent=1)
        print(f"wrote {dump}")
