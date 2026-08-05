import numpy as np, sys
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ppm_sigma_reg/src")
from common import minimax as mm
RY=13.605693122994; omega_max=10.0/RY; edge=1.5
print(" xi(eV)  A_core   N   sum|a_hat|   max|a_hat|   cond~sum|a|  (physical alpha=a_hat/xi)")
for xi_ev in [0.25,0.5,1.0,2.0,4.0]:
    xi=xi_ev/RY; T=omega_max+edge*xi; A=max(2*T/xi,1e-8)
    tau,w,_n,err=mm.crossing_grids(A,1e-6,mm.G_hgl,mm.tau_max_hgl,eps_q=1e-3,N_max=500)
    print(f" {xi_ev:5.2f}  {A:7.2f}  {len(tau):3d}  {np.abs(w).sum():10.1f}  {np.abs(w).max():10.1f}   {np.abs(w).sum():.2e}   max|alpha_phys|={np.abs(w).max()/xi:.3e}")
