import sys, numpy as np
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ridge_wt/tests")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_ridge_ab_2026-07-17")
from pathlib import Path
from analyze_ridge_ab import parse_eqp, parse_sigma_diag
from harness import parse_eqp_rows
SB = "/pscratch/sd/j/jackm/lorrax_sandbox"
# MoS2 stock vs repeat
a = parse_eqp(f"{SB}/runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/02_lorrax_gnppm_stock_ridgewt/eqp0.dat")
b = parse_eqp(f"{SB}/runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/06_lorrax_gnppm_stock_repeat/eqp0.dat")
d = np.abs(a[:,1]-b[:,1])*1e3
sa = parse_sigma_diag(f"{SB}/runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/02_lorrax_gnppm_stock_ridgewt/sigma_diag.dat")
sb = parse_sigma_diag(f"{SB}/runs/MoS2/01_mos2_3x3_gnppm_gate_2026-07-02/06_lorrax_gnppm_stock_repeat/sigma_diag.dat")
print(f"MoS2 stock-vs-repeat: max|d eqp0| = {d.max():.4g} meV, "
      f"max|d sigX| = {np.abs(sa[:,0]-sb[:,0]).max()*1e3:.4g} meV, "
      f"max|d ResigC| = {np.abs(sa[:,1]-sb[:,1]).max()*1e3:.4g} meV")
# Si stock vs repeat
labels = ("sigSX","sigCOH","sigTOT")
ra = parse_eqp_rows(Path(f"{SB}/runs/Si/B_zeta_ridge_covariance_2026-07-17/work_stock/eqp_si_test.dat"), labels)
rb = parse_eqp_rows(Path(f"{SB}/runs/Si/B_zeta_ridge_covariance_2026-07-17/work_stock_repeat/eqp_si_test.dat"), labels)
print(f"Si stock-vs-repeat: max|d Sigma| = {np.abs(ra[:,2:5]-rb[:,2:5]).max()*1e3:.4g} meV")
