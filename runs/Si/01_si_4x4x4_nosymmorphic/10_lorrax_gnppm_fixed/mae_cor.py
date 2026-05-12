"""LORRAX sig_c(Edft) vs BGW Cor' (= SX-X + CH'), exactly as reports/3d_coulomb_si_444 did."""
import re, numpy as np, sys
from pathlib import Path
base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic/10_lorrax_gnppm_fixed')
bgw_log = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic/01_bgw_gnppm/sigma_hp.log'

lorrax = {}
cur_k = None
for raw in open(base / 'sigma_freq_debug.dat'):
    s = raw.strip()
    if s.startswith('k-point'):
        cur_k = int(s.split()[1].rstrip(':'))
        continue
    if s.startswith('#') or s.startswith('k '): continue
    if cur_k is None: continue
    p = s.split()
    if len(p) < 13: continue
    try:
        n0 = int(p[1])
        sigc_edft = p[12]
        if sigc_edft.lower() == 'nan': continue
        lorrax[(cur_k, n0)] = float(sigc_edft)
    except Exception: pass

bgw = {}
cur_k = None
for raw in open(bgw_log):
    s = raw.strip()
    m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
    if m:
        cur_k = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        continue
    if cur_k is None: continue
    p = s.split()
    if len(p) >= 15 and p[0].isdigit():
        # Cor' = SX-X (col 4) + CH' (col 10)
        n_1 = int(p[0])
        cor_p = float(p[4]) + float(p[10])
        bgw[(cur_k, n_1 - 1)] = cor_p

sys.path.insert(0, '/global/u2/j/jackm/software/lorrax_A/src')
from file_io import WFNReader
from common import symmetry_maps
w = WFNReader(str(base / 'WFN.h5'))
sym = symmetry_maps.SymMaps(w)
unfolded = np.asarray(sym.unfolded_kpts)

def find_kfull(kbgw, tol=1e-4):
    target = np.array(kbgw)
    for i, k in enumerate(unfolded):
        d = (k - target + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < tol: return i
    return None

per_k = {}
for (kbgw, n0), corp in bgw.items():
    kfull = find_kfull(kbgw)
    if kfull is None: continue
    if (kfull, n0) not in lorrax: continue
    diff = lorrax[(kfull, n0)] - corp
    per_k.setdefault(kbgw, []).append(diff)

all_d = np.array([d for ds in per_k.values() for d in ds])
print(f'Comparing LORRAX sig_c(Edft) vs BGW Cor\'  ({len(all_d)} non-NaN matches)\n')
print(f'ALL k:  MAE = {np.mean(np.abs(all_d))*1000:6.1f} meV   mean = {np.mean(all_d)*1000:+6.1f} meV   max|Δ| = {np.max(np.abs(all_d))*1000:6.1f} meV')
print('\nPer-k MAE:')
for k, ds in per_k.items():
    da = np.array(ds)
    print(f'  k={k}: n={len(da):2d}, MAE={np.mean(np.abs(da))*1000:6.1f} meV, mean={np.mean(da)*1000:+7.1f} meV, max|Δ|={np.max(np.abs(da))*1000:6.1f} meV')
