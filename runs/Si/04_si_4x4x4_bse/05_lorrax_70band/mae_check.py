import h5py, numpy as np, re, sys
from pathlib import Path

base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/05_lorrax_70band')
ryd2ev = 13.6056980659

with h5py.File(base / 'sigma_mnk_ppm_bgw.h5', 'r') as f:
    omega_ev = f['omega_ev'][...]
    sigma_c_omega = f['sigma_c_kij_ev'][...]
    sigma_x_kij = f['sigma_sx_kij_ev'][...]

with h5py.File(base / 'WFN.h5', 'r') as f:
    el_ry = f['mf_header/kpoints/el'][0]
    nelec = int(f['mf_header/kpoints/ifmax'][0,0])
el_ev = el_ry * ryd2ev
vbm = max(np.max(el_ev[k, :nelec]) for k in range(el_ev.shape[0]))
cbm = min(np.min(el_ev[k, nelec:]) for k in range(el_ev.shape[0]))
ef = 0.5 * (vbm + cbm)

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
        if np.max(np.abs(d)) < tol:
            return i
    return None

# Find irr-k partner for each unfolded full-BZ k.  SymMaps stores kpoint_map.
print('SymMaps attrs:', [a for a in dir(sym) if not a.startswith('_') and 'k' in a.lower()][:20])

def parse_bgw_sigp(path):
    out = {}
    cur_k = None
    for raw in open(path):
        s = raw.strip()
        m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
        if m:
            cur_k = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            continue
        if cur_k is None:
            continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            sigp = float(p[10])
            out[(cur_k, n)] = sigp
    return out

bgw_ppm = parse_bgw_sigp('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/01_bgw_gn_ppm/sigma_hp.log')
bgw_csx = parse_bgw_sigp('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/04_bgw_cohsex_70band/sigma_hp.log')

print(f'BGW PPM: {len(bgw_ppm)} (k,n) entries; BGW COHSEX: {len(bgw_csx)} entries')

# Build IRR-k lookup for each full-BZ k via the WFN IRR k-point list
wfn_kpts = np.asarray(w.kpoints)  # (n_irr, 3)
def irr_index_of_full(kfull_idx):
    kf = unfolded[kfull_idx]
    for i, k in enumerate(wfn_kpts):
        d = (k - kf + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < 1e-4:
            return i
    return 0

diffs_ppm, diffs_csx = [], []
for (kbgw, n_1idx), bppm in bgw_ppm.items():
    kfull = find_kfull(kbgw)
    if kfull is None:
        continue
    n0 = n_1idx - 1
    if n0 >= sigma_c_omega.shape[2]:
        continue
    irr_k = irr_index_of_full(kfull)
    edft = el_ev[irr_k, n0]
    omega_rel = edft - ef
    sx = sigma_x_kij[kfull, n0, n0].real
    re_c = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].real))
    lppm = sx + re_c
    diffs_ppm.append(lppm - bppm)
    bcsx = bgw_csx.get((kbgw, n_1idx))
    if bcsx is not None:
        diffs_csx.append(lppm - bcsx)

dp = np.array(diffs_ppm); dc = np.array(diffs_csx)
print(f'\nMatched {len(dp)} (k,n) pairs total\n')
print('LORRAX PPM (no Sigma_c head) vs BGW PPM Sigma\' (60-band, 16 bands per k):')
print(f'  mean = {np.mean(dp):+.3f} eV   MAE = {np.mean(np.abs(dp)):.3f} eV ({np.mean(np.abs(dp))*1000:.0f} meV)   max|D| = {np.max(np.abs(dp)):.3f} eV')
print('LORRAX PPM (no Sigma_c head) vs BGW COHSEX Sigma\' (70-band, 16 bands per k):')
print(f'  mean = {np.mean(dc):+.3f} eV   MAE = {np.mean(np.abs(dc)):.3f} eV ({np.mean(np.abs(dc))*1000:.0f} meV)   max|D| = {np.max(np.abs(dc)):.3f} eV')
print(f'\nReference: previous handoff reported 1281 meV MAE vs BGW PPM (head missing).')

# Per-k MAE breakdown
print('\nPer-IRR-k MAE vs BGW PPM:')
per_k = {}
for (kbgw, n_1idx), bppm in bgw_ppm.items():
    kfull = find_kfull(kbgw)
    if kfull is None: continue
    n0 = n_1idx - 1
    if n0 >= sigma_c_omega.shape[2]: continue
    irr_k = irr_index_of_full(kfull)
    edft = el_ev[irr_k, n0]
    omega_rel = edft - ef
    sx = sigma_x_kij[kfull, n0, n0].real
    re_c = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].real))
    lppm = sx + re_c
    d = lppm - bppm
    per_k.setdefault(kbgw, []).append(d)
for k, ds in per_k.items():
    da = np.array(ds)
    print(f'  k={k}: MAE={np.mean(np.abs(da))*1000:6.0f} meV, mean={np.mean(da)*1000:+7.0f} meV')
