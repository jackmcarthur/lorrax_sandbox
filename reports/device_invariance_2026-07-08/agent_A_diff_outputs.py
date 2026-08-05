import re, sys
import numpy as np

def parse_sigma_diag(path):
    """Parse LORRAX sigma_diag.dat -> {(ik,n): {'sigX','sigC','sigXC','Eo'}} (sigC complex)."""
    out = {}
    ik = None
    for line in open(path):
        s = line.strip()
        m = re.match(r'k-point (\d+):', s)
        if m:
            ik = int(m.group(1)); continue
        m = re.match(r'n=(\d+)\s', s)
        if m and ik is not None:
            n = int(m.group(1))
            def grab(name):
                # value is either "RE" or "RE+ IMi" / "RE+-IMi"
                mm = re.search(name + r'=\s*(-?[\d.]+)(?:\+\s*(-?[\d.]+)i)?', s)
                if mm is None:
                    return None
                re_p = float(mm.group(1))
                im_p = float(mm.group(2)) if mm.group(2) is not None else 0.0
                return complex(re_p, im_p)
            sx = grab('sigX') or grab('sigSX')
            sc = grab('sigC') or grab('sigCOH')
            st = grab('sigXC') or grab('sigTOT')
            out[(ik, n)] = {'sigX': sx.real, 'sigC': sc, 'sigXC': st, 'Eo': grab('Eo').real}
    return out

def parse_eqp(path):
    """Parse LORRAX eqp0/eqp1.dat -> {(ik,n): (Edft, Eqp)}; ik 0-indexed, n 0-indexed."""
    out = {}
    ik = -1
    for line in open(path):
        s = line.split()
        if not s or line.startswith('#'):
            continue
        if len(s) == 4 and '.' in s[0]:
            ik += 1; continue
        if len(s) == 4:
            n = int(s[1]) - 1
            out[(ik, n)] = (float(s[2]), float(s[3]))
    return out

def cmp_dir(a, b, label):
    print(f"===== {label} =====")
    for fname, kind in [('sigma_diag.dat', 'sig'), ('eqp0.dat', 'eqp'), ('eqp1.dat', 'eqp')]:
        pa, pb = f"{a}/{fname}", f"{b}/{fname}"
        try:
            if kind == 'sig':
                da, db = parse_sigma_diag(pa), parse_sigma_diag(pb)
                keys = sorted(set(da) & set(db))
                assert set(da) == set(db), "key mismatch"
                dX = np.array([db[k]['sigX'] - da[k]['sigX'] for k in keys])
                dC = np.array([db[k]['sigC'] - da[k]['sigC'] for k in keys])
                print(f"{fname}: N={len(keys)}  max|dsigX|={np.abs(dX).max():.3e}  max|Re dsigC|={np.abs(dC.real).max():.3e}  max|Im dsigC|={np.abs(dC.imag).max():.3e}")
                # worst offenders
                order = np.argsort(-np.abs(dC.real))
                for thr in (1e-6, 1e-3, 1e-1):
                    print(f"  n(|Re dsigC|>{thr:g} eV) = {(np.abs(dC.real) > thr).sum()}/{len(keys)}")
                for i in order[:8]:
                    k = keys[i]
                    print(f"  worst (ik={k[0]},n={k[1]}): sigC_a={da[k]['sigC']:.6f} sigC_b={db[k]['sigC']:.6f} d={dC[i]:.3e}  dsigX={db[k]['sigX']-da[k]['sigX']:.3e}")
                # per-k max (real part)
                iks = sorted(set(k[0] for k in keys))
                permax = {ik: max(abs((db[k]['sigC']-da[k]['sigC']).real) for k in keys if k[0]==ik) for ik in iks}
                print("  per-k max|Re dsigC|: " + " ".join(f"k{ik}:{permax[ik]:.1e}" for ik in iks))
            else:
                da, db = parse_eqp(pa), parse_eqp(pb)
                keys = sorted(set(da) & set(db))
                d = np.array([db[k][1] - da[k][1] for k in keys])
                print(f"{fname}: N={len(keys)}  max|dEqp|={np.abs(d).max():.3e}  n>1meV={(np.abs(d)>1e-3).sum()}")
                order = np.argsort(-np.abs(d))
                for i in order[:6]:
                    k = keys[i]
                    print(f"  worst (ik={k[0]},n={k[1]}): Eqp_a={da[k][1]:.6f} Eqp_b={db[k][1]:.6f} d={d[i]:.3e}  Edft={da[k][0]:.4f}")
        except FileNotFoundError as e:
            print(f"{fname}: missing ({e})")
    print()

if __name__ == "__main__":
    base = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/Z_memplanner_validation_2026-07-06"
    cmp_dir(f"{base}/A_charge/head_4g", f"{base}/A_charge/head_16g", "A_charge head 4g vs 16g")
    cmp_dir(f"{base}/B_bispinor/head_4g", f"{base}/B_bispinor/head_16g", "B_bispinor head 4g vs 16g")
