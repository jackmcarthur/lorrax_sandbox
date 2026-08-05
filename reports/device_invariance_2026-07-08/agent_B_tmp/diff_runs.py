"""Diff LORRAX sigma_diag.dat / eqp{0,1}.dat between two run dirs (4g vs 16g)."""
import re, sys
import numpy as np

def parse_sigma_diag(path):
    """{(ik, n): dict(sigX, sigC_re, sigC_im, Eo)} for all k."""
    out = {}
    ik = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k-point (\d+):', s)
        if m:
            ik = int(m.group(1)); continue
        m = re.match(r'n=(\d+)\s+sigX=\s*([\-\d.]+)\s+sigC=\s*([\-\d.]+)\+\s*([\-\d.]+)i', s)
        if m and ik is not None:
            n = int(m.group(1))
            eo = re.search(r'Eo=\s*([\-\d.]+)', s)
            out[(ik, n)] = dict(sigX=float(m.group(2)), sigC_re=float(m.group(3)),
                                sigC_im=float(m.group(4)),
                                Eo=float(eo.group(1)) if eo else np.nan)
            continue
        # bispinor COHSEX format: sigSX / sigCOH / sigTOT
        m = re.match(r'n=(\d+)\s+sigSX=\s*([\-\d.]+)\s+sigCOH=\s*([\-\d.]+)\s+sigTOT=\s*([\-\d.]+)', s)
        if m and ik is not None:
            n = int(m.group(1))
            eo = re.search(r'Eo=\s*([\-\d.]+)', s)
            out[(ik, n)] = dict(sigX=float(m.group(2)), sigC_re=float(m.group(3)),
                                sigC_im=0.0,
                                Eo=float(eo.group(1)) if eo else np.nan)
    return out

def parse_eqp(path):
    """{(ik, n): (Eo, Eqp)}; eqp .dat: kx ky kz nb header then rows: ispin n Eo Eqp."""
    out = {}
    ik = -1
    for line in open(path):
        if line.startswith('#'): continue
        p = line.split()
        if len(p) == 4 and '.' in p[0]:
            ik += 1; continue
        if len(p) == 4:
            out[(ik, int(p[1]) - 1)] = (float(p[2]), float(p[3]))
    return out

def main(d1, d2, label):
    print(f"==== {label}: {d1} vs {d2} ====")
    s1, s2 = parse_sigma_diag(f"{d1}/sigma_diag.dat"), parse_sigma_diag(f"{d2}/sigma_diag.dat")
    keys = sorted(set(s1) & set(s2))
    dX = np.array([s2[k]['sigX'] - s1[k]['sigX'] for k in keys])
    dCre = np.array([s2[k]['sigC_re'] - s1[k]['sigC_re'] for k in keys])
    dCim = np.array([s2[k]['sigC_im'] - s1[k]['sigC_im'] for k in keys])
    print(f"sigma_diag common (k,n): {len(keys)}")
    print(f"  |d sigX |  max {np.abs(dX).max():.3e}  at {keys[np.abs(dX).argmax()]}")
    print(f"  |d sigC re| max {np.abs(dCre).max():.3e}  at {keys[np.abs(dCre).argmax()]}")
    print(f"  |d sigC im| max {np.abs(dCim).max():.3e}  at {keys[np.abs(dCim).argmax()]}")
    # top-10 sigC_re movers
    idx = np.argsort(-np.abs(dCre))[:10]
    for i in idx:
        k = keys[i]
        print(f"    (k={k[0]:2d} n={k[1]:3d}) sigC_re {s1[k]['sigC_re']:14.6f} -> {s2[k]['sigC_re']:14.6f}"
              f"  d={dCre[i]:+.6e}  sigC_im1={s1[k]['sigC_im']:.3e} Eo={s1[k]['Eo']:.3f}")
    for f in ("eqp0", "eqp1"):
        try:
            e1, e2 = parse_eqp(f"{d1}/{f}.dat"), parse_eqp(f"{d2}/{f}.dat")
        except FileNotFoundError:
            continue
        ks = sorted(set(e1) & set(e2))
        d = np.array([e2[k][1] - e1[k][1] for k in ks])
        print(f"  {f}: n={len(ks)}  |d Eqp| max {np.abs(d).max():.3e} at {ks[np.abs(d).argmax()]}"
              f"  median {np.median(np.abs(d)):.3e}  n>1meV {(np.abs(d)>1e-3).sum()}")
        idx = np.argsort(-np.abs(d))[:6]
        for i in idx:
            k = ks[i]
            print(f"    (k={k[0]:2d} n={k[1]:3d}) {e1[k][1]:14.6f} -> {e2[k][1]:14.6f}  d={d[i]:+.6e}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
