"""Compare V_H / Vxc / scissor C3-star spread across sigma_freq_debug.dat runs."""
import sys
import numpy as np

def load(path):
    cols = None; fd = {}
    for line in open(path):
        s = line.strip()
        if s.startswith("#"):
            p = s.lstrip("#").split()
            if len(p) >= 3 and p[0] == "k" and p[1] == "n":
                cols = p[2:]
            continue
        if not s or cols is None:
            continue
        p = s.split()
        if len(p) != len(cols) + 2:
            continue
        try:
            k, n = int(p[0]), int(p[1])
        except ValueError:
            continue
        fd[(k, n)] = {c: (np.nan if v == "nan" else float(v)) for c, v in zip(cols, p[2:])}
    ks = sorted({k for (k, n) in fd})
    ns = sorted({n for (k, n) in fd})
    return fd, ks, ns

def stars(fd, ks, ns):
    # group k by full E_dft spectrum (rounded), like star_decomp.py
    def spec(k):
        return tuple(round(fd[(k, n)]['E_dft'], 3) for n in ns if (k, n) in fd)
    grp = {}
    for k in ks:
        grp.setdefault(spec(k), []).append(k)
    return [v for v in grp.values() if len(v) > 1]

def worst(fd, ns, star, col):
    w = 0.0; wb = -1
    for n in ns:
        vals = [fd[(k, n)][col] for k in star if (k, n) in fd]
        vals = [v for v in vals if np.isfinite(v)]
        if len(vals) < 2:
            continue
        sp = max(vals) - min(vals)
        if sp > w:
            w = sp; wb = n
    return w, wb

for path in sys.argv[1:]:
    fd, ks, ns = load(path)
    ms = stars(fd, ks, ns)
    def vxc(k, n):
        return fd[(k, n)]['E_dft'] - fd[(k, n)]['kin_ion'] - fd[(k, n)]['V_H']
    # inject Vxc
    for (k, n) in list(fd):
        fd[(k, n)]['Vxc'] = vxc(k, n)
    gVH = gXB = gVX = gKI = 0.0
    vbm_star = None
    for star in ms:
        # find the VBM star (contains a k with E_dft(b25)≈-5.7648)
        if any(abs(fd[(k, 25)]['E_dft'] - (-5.7648)) < 0.01 for k in star if (k, 25) in fd):
            vbm_star = star
        sVH, _ = worst(fd, ns, star, 'V_H')
        sXB, _ = worst(fd, ns, star, 'x_bare')
        sVX, _ = worst(fd, ns, star, 'Vxc')
        sKI, _ = worst(fd, ns, star, 'kin_ion')
        gVH = max(gVH, sVH); gXB = max(gXB, sXB); gVX = max(gVX, sVX); gKI = max(gKI, sKI)
    print(f"\n=== {path} ===")
    print(f"  #multi-member stars: {len(ms)}")
    print(f"  WORST over all stars/bands:  V_H={gVH:8.4f}  x_bare={gXB:7.4f}  "
          f"kin_ion={gKI:7.4f}  Vxc={gVX:8.4f} eV")
    if vbm_star is not None:
        vh = [fd[(k, 25)]['V_H'] for k in vbm_star if (k, 25) in fd]
        xb = [fd[(k, 25)]['x_bare'] for k in vbm_star if (k, 25) in fd]
        vx = [fd[(k, 25)]['Vxc'] for k in vbm_star if (k, 25) in fd]
        print(f"  VBM (band26) star {vbm_star}:")
        print(f"    V_H spread={max(vh)-min(vh):.4f}  x_bare spread={max(xb)-min(xb):.4f}  "
              f"Vxc spread={max(vx)-min(vx):.4f} eV")
