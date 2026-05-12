"""LORRAX 01_lorrax_bse_vcoul (60-band PPM-with-overrides on nsym=48 WFN) vs BGW PPM Cor'."""
import re, numpy as np, sys, h5py
from pathlib import Path
base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul')
ryd2ev = 13.6056980659

with h5py.File(base / 'sigma_mnk_ppm.h5', 'r') as f:
    omega_ev = f['omega_ev'][...]
    sigma_c_omega = f['sigma_c_kij_ev'][...]
    sigma_x_kij = f['sigma_sx_kij_ev'][...]
ol, oh = float(omega_ev.min()), float(omega_ev.max())
print(f'omega grid [{ol},{oh}] eV; sigma_c shape {sigma_c_omega.shape}')

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
wfn_kpts = np.asarray(w.kpoints)

def find_kfull(kbgw, tol=1e-4):
    target = np.array(kbgw)
    for i, k in enumerate(unfolded):
        d = (k - target + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < tol: return i
    return None
def irr_idx(kfull_idx):
    kf = unfolded[kfull_idx]
    for i, k in enumerate(wfn_kpts):
        d = (k - kf + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < 1e-4: return i
    return 0

def parse_corp(path):
    out = {}; cur = None
    for raw in open(path):
        s = raw.strip()
        m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
        if m:
            cur = (float(m.group(1)), float(m.group(2)), float(m.group(3))); continue
        if cur is None: continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n_1 = int(p[0])
            out[(cur, n_1 - 1)] = float(p[4]) + float(p[10])
    return out

bgw_ppm = parse_corp('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/01_bgw_gn_ppm/sigma_hp.log')

pk = {}
n_innan = 0
for (kbgw, n0), corp in bgw_ppm.items():
    kfull = find_kfull(kbgw)
    if kfull is None: continue
    if n0 >= sigma_c_omega.shape[2]: continue
    irrk = irr_idx(kfull)
    edft = el_ev[irrk, n0]
    omr = edft - ef
    if not (ol <= omr <= oh): n_innan += 1; continue
    sigc = float(np.interp(omr, omega_ev, sigma_c_omega[:, kfull, n0, n0].real))
    pk.setdefault(kbgw, []).append(sigc - corp)

all_d = np.array([d for ds in pk.values() for d in ds])
print(f"\n=== sig_c(Edft) vs BGW PPM Cor'  ({len(all_d)} matches; {n_innan} skipped NaN-equiv) ===")
print(f'ALL k:  MAE = {np.mean(np.abs(all_d))*1000:6.1f} meV   mean = {np.mean(all_d)*1000:+6.1f} meV   max|Δ| = {np.max(np.abs(all_d))*1000:6.1f} meV')
for k, ds in pk.items():
    da = np.array(ds)
    print(f'  k={k}: n={len(da):2d}, MAE={np.mean(np.abs(da))*1000:6.1f} meV, mean={np.mean(da)*1000:+7.1f} meV')
