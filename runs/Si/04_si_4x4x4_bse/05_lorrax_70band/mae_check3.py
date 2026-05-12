"""MAE only on (k, n) where E_DFT - E_F ∈ ω-grid (so Σ_c(E_DFT) is *defined*, not extrapolated)."""
import h5py, numpy as np, re, sys
from pathlib import Path

base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/05_lorrax_70band')
ryd2ev = 13.6056980659

with h5py.File(base / 'sigma_mnk_ppm_bgw.h5', 'r') as f:
    omega_ev = f['omega_ev'][...]
    sigma_c_omega = f['sigma_c_kij_ev'][...]
    sigma_x_kij = f['sigma_sx_kij_ev'][...]

omega_lo, omega_hi = float(omega_ev.min()), float(omega_ev.max())
print(f'omega grid: [{omega_lo:.2f}, {omega_hi:.2f}] eV ({len(omega_ev)} points)')

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

# Categorize
in_range_dp, in_range_dc = [], []
out_of_range_count = 0
near_pole_count = 0  # |Im| > 5 eV
clean_dp, clean_dc = [], []
for (kbgw, n_1idx), bppm in bgw_ppm.items():
    kfull = find_kfull(kbgw)
    if kfull is None: continue
    n0 = n_1idx - 1
    if n0 >= sigma_c_omega.shape[2]: continue
    irr_k = irr_index_of_full(kfull)
    edft = el_ev[irr_k, n0]
    omega_rel = edft - ef
    if not (omega_lo <= omega_rel <= omega_hi):
        out_of_range_count += 1
        continue
    sx = sigma_x_kij[kfull, n0, n0].real
    re_c = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].real))
    im_c = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].imag))
    lppm = sx + re_c
    dp = lppm - bppm
    bcsx = bgw_csx.get((kbgw, n_1idx))
    dc = lppm - bcsx if bcsx is not None else None
    in_range_dp.append(dp)
    if dc is not None: in_range_dc.append(dc)
    if abs(im_c) > 5.0:
        near_pole_count += 1
    else:
        clean_dp.append(dp)
        if dc is not None: clean_dc.append(dc)

dp_in = np.array(in_range_dp); dc_in = np.array(in_range_dc)
dp_clean = np.array(clean_dp); dc_clean = np.array(clean_dc)

total = sum(1 for k_n in bgw_ppm.keys() if find_kfull(k_n[0]) is not None and (k_n[1]-1) < sigma_c_omega.shape[2])
print(f'\nTotal candidate (k,n): {total}')
print(f'  out-of-grid (NaN-equivalent): {out_of_range_count}')
print(f'  in-grid: {len(in_range_dp)}')
print(f'  in-grid AND |Im(Σc)|<5 eV (defined Re):  {len(clean_dp)}')

print(f'\n=== In-grid only (matches LORRAX-native sig_c(Edft) non-NaN) ===')
print(f'L-PPM vs BGW-PPM:    mean={np.mean(dp_in)*1000:+5.0f} meV  MAE={np.mean(np.abs(dp_in))*1000:5.0f} meV  max|Δ|={np.max(np.abs(dp_in))*1000:5.0f} meV')
print(f'L-PPM vs BGW-COHSEX: mean={np.mean(dc_in)*1000:+5.0f} meV  MAE={np.mean(np.abs(dc_in))*1000:5.0f} meV  max|Δ|={np.max(np.abs(dc_in))*1000:5.0f} meV')

print(f'\n=== In-grid AND |Im(Σc)|<5 eV (Re part actually trustworthy) ===')
if len(dp_clean):
    print(f'L-PPM vs BGW-PPM:    mean={np.mean(dp_clean)*1000:+5.0f} meV  MAE={np.mean(np.abs(dp_clean))*1000:5.0f} meV  max|Δ|={np.max(np.abs(dp_clean))*1000:5.0f} meV')
    print(f'L-PPM vs BGW-COHSEX: mean={np.mean(dc_clean)*1000:+5.0f} meV  MAE={np.mean(np.abs(dc_clean))*1000:5.0f} meV  max|Δ|={np.max(np.abs(dc_clean))*1000:5.0f} meV')
else:
    print('  (no clean points)')
