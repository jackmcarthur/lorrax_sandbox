"""LORRAX sig_c(Edft) vs BGW Cor'  on the 70-band PPM run.

Σ_c only — bare-X is excluded from both sides; this is what the working
60-band reports/3d_coulomb_si_444 used and got 5 meV at Γ.
"""
import re, numpy as np, sys, h5py
from pathlib import Path

base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/05_lorrax_70band')
ryd2ev = 13.6056980659

with h5py.File(base / 'sigma_mnk_ppm_bgw.h5', 'r') as f:
    omega_ev = f['omega_ev'][...]
    sigma_c_omega = f['sigma_c_kij_ev'][...]
omega_lo, omega_hi = float(omega_ev.min()), float(omega_ev.max())

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
def irr_index_of_full(kfull_idx):
    kf = unfolded[kfull_idx]
    for i, k in enumerate(wfn_kpts):
        d = (k - kf + 0.5) % 1.0 - 0.5
        if np.max(np.abs(d)) < 1e-4: return i
    return 0

def parse_bgw_corp(path):
    out = {}
    cur_k = None
    for raw in open(path):
        s = raw.strip()
        m = re.match(r'k\s*=\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+).*ik\s*=\s*(\d+)', s)
        if m:
            cur_k = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            continue
        if cur_k is None: continue
        p = s.split()
        if len(p) >= 15 and p[0].isdigit():
            n_1 = int(p[0])
            cor_p = float(p[4]) + float(p[10])  # SX-X + CH'
            out[(cur_k, n_1 - 1)] = cor_p
    return out

# Two BGW PPM references at our band counts:
# (a) BGW PPM 60-band (different band count than us)
# (b) BGW COHSEX 70-band (Cor' from COHSEX = static SX-X + CH')
bgw_ppm = parse_bgw_corp('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/00_si_4x4x4_60band/01_bgw_gn_ppm/sigma_hp.log')
bgw_csx = parse_bgw_corp('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/04_bgw_cohsex_70band/sigma_hp.log')

def compare(bgw_dict, label):
    diffs_per_k = {}
    n_match = n_innan = 0
    for (kbgw, n0), corp in bgw_dict.items():
        kfull = find_kfull(kbgw)
        if kfull is None: continue
        if n0 >= sigma_c_omega.shape[2]: continue
        irr_k = irr_index_of_full(kfull)
        edft = el_ev[irr_k, n0]
        omega_rel = edft - ef
        if not (omega_lo <= omega_rel <= omega_hi):  # NaN-equivalent
            n_innan += 1
            continue
        re_c = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].real))
        d = re_c - corp
        diffs_per_k.setdefault(kbgw, []).append(d)
        n_match += 1
    all_d = np.array([d for ds in diffs_per_k.values() for d in ds])
    if not len(all_d):
        print(f'{label}: no matches'); return
    print(f'\n=== LORRAX sig_c(Edft) vs {label}  (in-grid only, {n_match} matches; {n_innan} skipped as NaN-equivalent) ===')
    print(f'ALL k:  MAE = {np.mean(np.abs(all_d))*1000:6.1f} meV   mean = {np.mean(all_d)*1000:+6.1f} meV   max|Δ| = {np.max(np.abs(all_d))*1000:6.1f} meV')
    if (0.0, 0.0, 0.0) in diffs_per_k:
        dg = np.array(diffs_per_k[(0.0, 0.0, 0.0)])
        print(f'Γ only: MAE = {np.mean(np.abs(dg))*1000:6.1f} meV   mean = {np.mean(dg)*1000:+6.1f} meV   max|Δ| = {np.max(np.abs(dg))*1000:6.1f} meV  (n={len(dg)})')

compare(bgw_ppm, "BGW PPM 60-band Cor' (SX-X + CH')")
compare(bgw_csx, "BGW COHSEX 70-band Cor' (SX-X + CH')")

# Per-band breakdown at Γ
print('\n=== Per-band breakdown at Γ (LORRAX sig_c vs BGW Cor\') ===')
print(' n_1idx  E_DFT(eV)   ω_rel(eV)  LORRAX_sigc  BGW_Cor_PPM   diff_PPM   BGW_Cor_CSX   diff_CSX')
kbgw = (0.0, 0.0, 0.0)
kfull = find_kfull(kbgw)
irr_k = irr_index_of_full(kfull)
for n0 in range(16):
    edft = el_ev[irr_k, n0]
    omega_rel = edft - ef
    in_range = omega_lo <= omega_rel <= omega_hi
    if not in_range:
        print(f'  {n0+1:3d}  {edft:+8.3f}  {omega_rel:+8.3f}  (out of grid; sig_c=NaN)')
        continue
    sigc_l = float(np.interp(omega_rel, omega_ev, sigma_c_omega[:, kfull, n0, n0].real))
    bppm = bgw_ppm.get((kbgw, n0), float('nan'))
    bcsx = bgw_csx.get((kbgw, n0), float('nan'))
    print(f'  {n0+1:3d}  {edft:+8.3f}  {omega_rel:+8.3f}  {sigc_l:+10.3f}    {bppm:+8.3f}    {sigc_l-bppm:+7.3f}     {bcsx:+8.3f}    {sigc_l-bcsx:+7.3f}')
