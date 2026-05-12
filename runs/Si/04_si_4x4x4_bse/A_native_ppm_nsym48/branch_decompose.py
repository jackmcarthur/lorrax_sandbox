"""Decompose Σ_c at Γ into per-branch contributions via the
LORRAX_DEBUG_SKIP_BRANCH artifacts.

Σ_c(branch X) = Σ_c(baseline) − Σ_c(skip X)
"""
import h5py, numpy as np
from pathlib import Path
base = Path('/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_bse/A_native_ppm_nsym48')

def load_at_gamma(path):
    with h5py.File(path,'r') as f:
        ome = np.linspace(-10, 10, f['sigma_c_kij_ev'].shape[0])
        sig = f['sigma_c_kij_ev'][:,0,:,:]   # (n_omega, nb, nb), k=Γ=0
    return ome, sig

# Need a full-baseline file too — re-run cohsex.in once with no skip to overwrite sigma_mnk.h5
# (we'll do that in the bash before this script)
ome, sig_baseline = load_at_gamma(base / 'sigma_mnk.h5')

skip = {}
for bname in ['pos_val','pos_cond','neg_val','neg_cond']:
    fn = base / f'sigma_mnk.skip_{bname}.h5'
    if fn.exists():
        skip[bname] = load_at_gamma(fn)[1]

# Apr-6 reference values for nsym=12 LORRAX (from sigma_freq_debug.dat at Γ band 9):
#   sig_c+(w) = +1.257   ← pos_val branch (ω>0, occ-pole Σ⁺)
#   sig_c-(w) = -4.965   ← pos_cond branch (ω>0, unocc-pole Σ⁻)
#   sig_c(Edft) = -3.708 (sum)

print('Per-branch Σ_c(ω) at Γ = baseline − skip[branch], evaluated at ω = E_DFT − E_F')
print()
print(f' n  ω_rel(eV)  baseline    pos_cond   pos_val    neg_cond   neg_val    Apr6_pos_cond  Apr6_pos_val')

apr6 = {  # (ω_rel, sig_c-, sig_c+) from sigma_freq_debug.dat for 10_lorrax_gnppm_fixed at Γ
    2: (-0.386, -2.599, +3.956),
    3: (-0.386, -2.599, +3.956),
    4: (-0.338, -2.605, +3.943),
    5: (-0.338, -2.606, +3.945),
    6: (-0.338, -2.606, +3.945),
    7: (-0.338, -2.606, +3.945),
    8: (+2.163, -4.965, +1.257),
    9: (+2.163, -4.965, +1.257),
   10: (+2.198, -4.973, +1.249),
   11: (+2.198, -4.973, +1.249),
   12: (+2.198, -4.973, +1.250),
   13: (+2.198, -4.973, +1.250),
   14: (+2.994, -5.505, +1.171),
   15: (+2.994, -5.505, +1.171),
}

for n0 in range(2, 16):
    omr, apr_pcond, apr_pval = apr6.get(n0, (np.nan, np.nan, np.nan))
    base_v = float(np.interp(omr, ome, sig_baseline[:, n0, n0].real))
    cols = [base_v]
    for bname in ['pos_cond','pos_val','neg_cond','neg_val']:
        if bname in skip:
            sv = float(np.interp(omr, ome, skip[bname][:, n0, n0].real))
            contrib = base_v - sv
            cols.append(contrib)
        else:
            cols.append(np.nan)
    print(f' {n0+1:2d}  {omr:+7.3f}    {cols[0]:+8.3f}  {cols[1]:+8.3f}  {cols[2]:+8.3f}  {cols[3]:+8.3f}  {cols[4]:+8.3f}    {apr_pcond:+8.3f}      {apr_pval:+8.3f}')
