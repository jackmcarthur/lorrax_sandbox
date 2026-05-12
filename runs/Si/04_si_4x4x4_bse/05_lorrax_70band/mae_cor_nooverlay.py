"""LORRAX cohsex_ppm.in (no overlay, but with head overrides) sig_c(Edft) vs BGW Cor'."""
import re, numpy as np, sys
from pathlib import Path
base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/05_lorrax_70band')

# Use sigma_freq_debug.dat (the one written by the run we just did)
fp = base / 'sigma_freq_debug.dat'
if not fp.exists():
    print(f"NOT FOUND: {fp}"); sys.exit(1)
print(f"Reading {fp}")
lorrax = {}
cur_k = None
for raw in open(fp):
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
print(f'LORRAX entries: {len(lorrax)}')

bgw_csx_log = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/04_bgw_cohsex_70band/sigma_hp.log'
bgw_ppm_log = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/01_bgw_gn_ppm/sigma_hp.log'

def parse_corp(path):
    out = {}; cur_k = None
    for raw in open(path):
        s = raw.strip()
        m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
        if m:
            cur_k = (float(m.group(1)), float(m.group(2)), float(m.group(3))); continue
        if cur_k is None: continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n_1 = int(p[0])
            out[(cur_k, n_1 - 1)] = float(p[4]) + float(p[10])  # SX-X + CH'
    return out

bgw_csx = parse_corp(bgw_csx_log)
bgw_ppm = parse_corp(bgw_ppm_log)

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

def cmp(bgw_dict, label):
    pk = {}
    for (kbgw, n0), corp in bgw_dict.items():
        kfull = find_kfull(kbgw)
        if kfull is None: continue
        if (kfull, n0) not in lorrax: continue
        pk.setdefault(kbgw, []).append(lorrax[(kfull, n0)] - corp)
    all_d = np.array([d for ds in pk.values() for d in ds])
    if not len(all_d): print(f'{label}: no matches'); return
    print(f"\n=== sig_c(Edft) vs {label}  ({len(all_d)} matches) ===")
    print(f'ALL k:  MAE = {np.mean(np.abs(all_d))*1000:6.1f} meV   mean = {np.mean(all_d)*1000:+6.1f} meV   max|Δ| = {np.max(np.abs(all_d))*1000:6.1f} meV')
    if (0.0, 0.0, 0.0) in pk:
        dg = np.array(pk[(0.0, 0.0, 0.0)])
        print(f'Γ:      MAE = {np.mean(np.abs(dg))*1000:6.1f} meV   mean = {np.mean(dg)*1000:+6.1f} meV   max|Δ| = {np.max(np.abs(dg))*1000:6.1f} meV   n={len(dg)}')

cmp(bgw_ppm, "BGW PPM 60-band Cor' (apples-PPM-PPM)")
cmp(bgw_csx, "BGW COHSEX 70-band Cor'")
