"""Reproduce the 5-meV-MAE-at-Γ claim from reports/3d_coulomb_si_444_2026-04-05.

Compare LORRAX 10_lorrax_gnppm_fixed (working GN-PPM Si setup) to BGW
01_bgw_gnppm reference, restricted to bands where LORRAX sig_c(Edft) is
non-NaN (i.e. ε - E_F is in-grid).
"""
import re, numpy as np, h5py, sys
from pathlib import Path

base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic/10_lorrax_gnppm_fixed')
bgw_log = '/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/01_si_4x4x4_nosymmorphic/01_bgw_gnppm/sigma_hp.log'

# Parse LORRAX sigma_freq_debug.dat: cols   k n Edft-Ef E_dft kin_ion sex_0 coh_0 x_bare sig_c(0) sig_c+(w) sig_c-(w) sig_c_invld(0) sig_c(Edft)
lorrax = {}
cur_k = None
for raw in open(base / 'sigma_freq_debug.dat'):
    s = raw.strip()
    if s.startswith('k-point'):
        cur_k = int(s.split()[1].rstrip(':'))
        continue
    if s.startswith('#') or s.startswith('k '):
        continue
    if cur_k is None: continue
    p = s.split()
    if len(p) < 13: continue
    try:
        n0 = int(p[1])
        edft = float(p[3])
        x_bare = float(p[7])
        sigc_edft = p[12]
        if sigc_edft.lower() == 'nan': continue
        sigc_edft = float(sigc_edft)
        # Σ_xc = sigX (= x_bare) + Σ_c(E_DFT)
        sigxc = x_bare + sigc_edft
        lorrax[(cur_k, n0)] = (edft, x_bare, sigc_edft, sigxc)
    except Exception: pass

print(f'LORRAX entries (non-nan only): {len(lorrax)}')

# Parse BGW Sig' from sigma_hp.log
bgw = {}
cur_k = None
ik_count = 0
for raw in open(bgw_log):
    s = raw.strip()
    m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
    if m:
        cur_k = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        ik_count = int(m.group(4)) - 1
        continue
    if cur_k is None: continue
    p = s.split()
    if len(p) >= 15 and p[0].isdigit():
        n_1 = int(p[0])
        sigp = float(p[10])
        x_bgw = float(p[3])
        bgw[(cur_k, n_1 - 1)] = (sigp, x_bgw)

# Get LORRAX full-BZ k-mapping
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

print(f'BGW entries: {len(bgw)}')

# Compare per (k,n) where both sides have data and LORRAX is non-NaN
diffs_all, diffs_gamma = [], []
near_pole, kept = 0, 0
for (kbgw, n0), (sigp, x_bgw) in bgw.items():
    kfull = find_kfull(kbgw)
    if kfull is None: continue
    if (kfull, n0) not in lorrax: continue
    edft, x_l, sigc_l, sigxc_l = lorrax[(kfull, n0)]
    diff = sigxc_l - sigp
    diffs_all.append(diff)
    if max(abs(c) for c in kbgw) < 1e-4:
        diffs_gamma.append(diff)
    kept += 1

dp = np.array(diffs_all)
dg = np.array(diffs_gamma)
print(f'\nMatched (k, n) with LORRAX sig_c(Edft) non-NaN: {kept}')
print(f'\nALL k:    MAE = {np.mean(np.abs(dp))*1000:6.0f} meV   mean = {np.mean(dp)*1000:+6.0f} meV   max|Δ| = {np.max(np.abs(dp))*1000:6.0f} meV')
if len(dg):
    print(f'Γ only:   MAE = {np.mean(np.abs(dg))*1000:6.0f} meV   mean = {np.mean(dg)*1000:+6.0f} meV   max|Δ| = {np.max(np.abs(dg))*1000:6.0f} meV')
