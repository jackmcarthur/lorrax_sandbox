import numpy as np, sys
sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ppm_sigma_reg/src")
from common import minimax as mm
RY=13.605693122994
# window params matching the MoS2 run
xi = 0.25/RY            # Ry  (sigma_regularization_ev=0.25)
edge=1.5
omega_max = 10.0/RY     # Ry
T = omega_max + edge*xi
A_core = max(2.0*T/xi, 1e-8)
print(f"xi={xi:.4f} Ry ({xi*RY:.3f} eV)  T={T:.4f} Ry  A_core={A_core:.2f}")
# solve HGL crossing quadrature (same call as code)
tau,w,_n,err = mm.crossing_grids(A_core, 1e-6, mm.G_hgl, mm.tau_max_hgl, eps_q=1e-3, N_max=500)
print(f"HGL nodes: N={len(tau)} err={err:.2e}  tau_hat range [{tau.min():.3e},{tau.max():.3f}]  alpha range [{w.min():.3e},{w.max():.3e}]")
print(f"  sum|alpha_hat| = {np.abs(w).sum():.3f}   (weights conditioning)")

def recon(u):
    # (1/xi)*sum alpha_hat sin(tau_hat*u)  == Im[coeff.sigma] for a unit real B pole
    return (1.0/xi)*np.sum(w[:,None]*np.sin(np.outer(tau, u)),axis=0)
def exact_PV(u):   # Re[1/((u)*xi + i*xi)] = Re[1/(xi(u+i))] = u/(xi(u^2+1))
    return u/(xi*(u*u+1.0))
def exact_pole_re(domega):  # Re[1/(domega + i*xi)] with domega=omega-S in Ry
    return domega/(domega*domega+xi*xi)

# scan omega-S in eV for a pole S; test reconstruction of the near-pole REAL part
print("\n  (omega-S)[eV]   recon*xi   G_hgl(u)   exactPV*xi=u/(u^2+1)   recon_ratio")
for dev in [-8,-4,-2,-1,-0.5,-0.25,0.0,0.25,0.5,1,2,4,8]:
    u=(dev/RY)/xi
    r=recon(np.array([u]))[0]
    g=mm.G_hgl(np.array([float(u)]))[0]
    ep=u/(u*u+1.0)
    print(f"   {dev:+7.2f}       {r*xi:+9.4f}  {g:+9.4f}   {ep:+9.4f}          {r*xi/(g if abs(g)>1e-9 else 1):.3f}")

# Now: sum over MANY empty-state poles (like the core does), check for spurious offset.
# core poles: S = E_A + Omega, E_A in [0, T], Omega in [0, T]. Use a grid of poles.
print("\n=== Sum over a spread of core poles (unit B each), Sigma_core(omega) ===")
np.random.seed(0)
nEA=40; nOm=30
EA=np.linspace(0.06, T, nEA)        # empty-state energies (Ry)
Om=np.linspace(0.05, T, nOm)        # low plasmon poles (Ry)
Spoles=(EA[:,None]+Om[None,:]).ravel()  # all S
Bunit=np.ones_like(Spoles)
print(f"  n_poles={Spoles.size}, S range [{Spoles.min()*RY:.2f},{Spoles.max()*RY:.2f}] eV")
for wev in [-2,0,2,5,8]:
    om=wev/RY
    u=((om-Spoles)/xi)
    contrib = (1.0/xi)*np.sum(Bunit* (np.sum(w[:,None]*np.sin(tau[:,None]*u[None,:]),axis=0)))
    # exact PV sum
    exactsum = np.sum(Bunit*(om-Spoles)/((om-Spoles)**2+xi*xi))
    print(f"  omega={wev:+5.1f} eV: recon_sum={contrib:+12.3f}  exactPV_sum={exactsum:+12.3f}  (Ry units, per unit B)")
