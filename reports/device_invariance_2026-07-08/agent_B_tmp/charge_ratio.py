import sys, re, numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/reports/device_invariance_2026-07-08/agent_B_tmp")
from diff_runs import parse_sigma_diag
b = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/Z_memplanner_validation_2026-07-06/A_charge"
s1, s2 = parse_sigma_diag(f"{b}/head_4g/sigma_diag.dat"), parse_sigma_diag(f"{b}/head_16g/sigma_diag.dat")
keys = sorted(set(s1) & set(s2))
d = np.array([abs(s2[k]['sigC_re'] - s1[k]['sigC_re']) for k in keys])
im = np.array([max(abs(s1[k]['sigC_im']), 1e-30) for k in keys])
big = d > 1e-3
print(f"entries: {len(keys)}, dsigC>1meV: {big.sum()}")
print(f"for dsigC>1meV: |ImSigC| range {im[big].min():.2e} .. {im[big].max():.2e} eV")
print(f"  d/|Im| ratio: median {np.median(d[big]/im[big]):.2e}  max {np.max(d[big]/im[big]):.2e}")
norm = (~big) & (im < 100.0)
print(f"normal bands (|Im|<100 eV): n={norm.sum()}  max dsigC {d[norm].max():.2e} eV  |Im| max {im[norm].max():.2e}")
mid = big & (im < 1e3)
print(f"entries with dsigC>1meV AND |Im|<1e3: {mid.sum()}")
