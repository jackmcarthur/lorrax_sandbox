"""Confirm the segni mechanism analytically + numerically.

(1) ENERGY is up<->dn swap-symmetric => GATE 0b (m=0) and the energy term-by-term
    verification CANNOT see the segni bug. E_xc = sum (nu+nd) eps(nu,nd,...) and
    pbe_xc_spin(nu,nd,...)=pbe_xc_spin(nd,nu,...) (exchange sums both channels,
    correlation depends on zeta^2 and total grad). So swapping nu<->nd (segni flip)
    leaves E unchanged -> energy is BLIND to segni. GOOD (explains established fact).

(2) POTENTIAL is NOT swap symmetric in the *4-component* sense:
    V(1)=(Vup+Vdn)/2 is swap-invariant (charge part fine) BUT
    B = (Vup-Vdn)/2 FLIPS SIGN under swap. LORRAX uses +|m| (=QE with segni=+1
    forced), so where true segni=-1, LORRAX's B has the WRONG SIGN.
    Net effect on <psi|B.sigma|psi>: B.sigma for a spin-up-majority state is
    NEGATIVE energy (stabilizing); wrong-sign B over half the cell partially
    CANCELS the correct contribution -> a band-dependent residual that grows with
    the state's spin polarization (d-states) -> ~15 meV at VBM. EXACTLY the symptom.

(3) The CHARGE part V(1)=(Vup+Vdn)/2 is ALSO corrupted, but only through the GGA
    gradient cusp of |m| (the swap leaves the LDA part of V(1) invariant; the GGA
    flux of V(1) uses grad(n_up) which has the |m|-cusp). This is the subtler,
    smaller channel.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from psp.xc import pbe_xc_spin, compute_V_xc_spin, _compute_sigma_spin

# Energy swap symmetry on random fields:
import jax
key=jax.random.PRNGKey(3)
nu=jnp.abs(jax.random.normal(key,(6,6,6)))+0.3
nd=jnp.abs(jax.random.normal(jax.random.PRNGKey(4),(6,6,6)))+0.3
suu,sud,sdd=_compute_sigma_spin(jnp.fft.fftn(nu),jnp.fft.fftn(nd),
    jnp.asarray(np.stack(np.meshgrid(*[np.fft.fftfreq(6)*6]*3,indexing='ij'),-1)))
E1=float(jnp.sum((nu+nd)*pbe_xc_spin(nu,nd,suu,sud,sdd)))
E2=float(jnp.sum((nd+nu)*pbe_xc_spin(nd,nu,sdd,sud,suu)))  # swap up<->dn
print(f"(1) E_xc swap-symmetry: E(up,dn)={E1:.8f}  E(dn,up)={E2:.8f}  diff={abs(E1-E2):.2e}")
print(f"    => energy BLIND to segni (swap) -> all energy gates pass despite the bug. CONFIRMED.\n")

# Potential swap antisymmetry of B:
G=jnp.asarray(np.stack(np.meshgrid(*[np.fft.fftfreq(6)*6]*3,indexing='ij'),-1).astype(float))
Vu,Vd=compute_V_xc_spin(nu,nd,jnp.fft.fftn(nu),jnp.fft.fftn(nd),G)
Vu_s,Vd_s=compute_V_xc_spin(nd,nu,jnp.fft.fftn(nd),jnp.fft.fftn(nu),G)
B=0.5*(Vu-Vd); B_swap=0.5*(Vu_s-Vd_s)
print(f"(2) B under swap: max|B+B_swap| (should be ~0 if B flips sign) = {float(jnp.abs(B+B_swap).max()):.2e}")
print(f"    max|B-B_swap| = {float(jnp.abs(B-B_swap).max()):.4f}  => B FLIPS SIGN under swap. CONFIRMED.")
Vbar=0.5*(Vu+Vd); Vbar_s=0.5*(Vu_s+Vd_s)
print(f"    Vbar(charge) under swap: max|Vbar-Vbar_swap| = {float(jnp.abs(Vbar-Vbar_s).max()):.2e} (invariant)")
