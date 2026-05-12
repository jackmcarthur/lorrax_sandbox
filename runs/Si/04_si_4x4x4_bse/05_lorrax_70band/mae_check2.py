"""Same MAE check but reporting both raw and pole-filtered (|Im(Σ_c(E_DFT))| < 5 eV)."""
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
wfn_kpts = np.asarray(w.kpoints)

def find_kfull(kbgw, tol=1e-4):
    target = np.array(kbgw)
    for i, k in enumerate(unfolded):
        d = (k - target + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < tol:
            return i
    return None

def irr_index_of_full(kfull_idx):
    kf = unfolded[kfull_idx]
    for i, k in enumerate(wfn_kpts):
        d = (k - kf + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < 1e-4:
            return i
    return 0

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

rows = []
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
    im_c = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].imag))
    lppm = sx + re_c
    bcsx = bgw_csx.get((kbgw, n_1idx))
    rows.append((kbgw, n_1idx, edft, lppm, bppm, bcsx, im_c))

print(f'matched {len(rows)} (k,n) pairs')
arr = np.array([(r[3]-r[4], r[3]-r[5], r[6]) for r in rows], dtype=np.float64)
dp_all, dc_all, im_all = arr[:,0], arr[:,1], arr[:,2]

print(f'\n=== Raw (all bands, no filter) ===')
print(f'L-PPM vs BGW-PPM:    MAE = {np.mean(np.abs(dp_all))*1000:.0f} meV   max|Im| = {np.max(np.abs(im_all)):.1f} eV')
print(f'L-PPM vs BGW-COHSEX: MAE = {np.mean(np.abs(dc_all))*1000:.0f} meV')

for cut in [50.0, 5.0, 1.0, 0.5, 0.1]:
    mask = np.abs(im_all) < cut
    n_keep = mask.sum()
    if n_keep == 0:
        print(f"\n=== |Im(Σ_c)| < {cut} eV: 0 of {len(rows)} bands kept ===")
        continue
    print(f"\n=== |Im(Σ_c(E_DFT))| < {cut} eV: kept {n_keep}/{len(rows)} (k,n) ===")
    print(f'  L-PPM vs BGW-PPM:    mean = {np.mean(dp_all[mask])*1000:+5.0f} meV   MAE = {np.mean(np.abs(dp_all[mask]))*1000:5.0f} meV   max|Δ| = {np.max(np.abs(dp_all[mask]))*1000:5.0f} meV')
    print(f'  L-PPM vs BGW-COHSEX: mean = {np.mean(dc_all[mask])*1000:+5.0f} meV   MAE = {np.mean(np.abs(dc_all[mask]))*1000:5.0f} meV   max|Δ| = {np.max(np.abs(dc_all[mask]))*1000:5.0f} meV')

# Show the bands that get filtered
print('\n=== (k, n) with |Im(Σ_c)| > 5 eV (likely near body PPM pole — Re part suspect): ===')
for (kbgw, n_1idx, edft, lppm, bppm, bcsx, im_c) in rows:
    if abs(im_c) > 5.0:
        print(f'  k={kbgw}  n={n_1idx}  E_DFT={edft:+.3f}  L_PPM={lppm:+.3f}  BGW_PPM={bppm:+.3f}  Δ={lppm-bppm:+.3f}  Im={im_c:+.1f}')
