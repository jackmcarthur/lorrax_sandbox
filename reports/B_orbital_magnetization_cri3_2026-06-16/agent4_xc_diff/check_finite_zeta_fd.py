"""Finite-zeta correctness of compute_V_xc_spin via the VARIATIONAL identity.

V_s(r) = dE_xc/dn_s(r) must hold pointwise (it's the definition QE satisfies).
We test LORRAX's autodiff V_up,V_dn against a finite-difference functional
derivative of E_xc = sum_r (n_up+n_dn)*eps_xc_spin, INCLUDING the gradient
dependence (so we perturb n_s at a single grid point and recompute E with the
FFT-gradient sigmas re-evaluated -- this captures the divergence term).

A pass here means the spin-GGA potential assembly (df_drho - div(flux), with the
2*df_duu*gu + df_dud*gd cross structure) is internally exact at finite zeta, so
the ~15 meV residual is NOT in compute_V_xc_spin's flux algebra.
A fail localizes the bug to the cross-term / divergence at finite zeta.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from psp.xc import pbe_xc_spin, compute_V_xc_spin, _compute_sigma_spin

nx = ny = nz = 8
N = nx*ny*nz
B = np.diag([0.9, 0.9, 0.5])
fx = np.fft.fftfreq(nx)*nx; fy=np.fft.fftfreq(ny)*ny; fz=np.fft.fftfreq(nz)*nz
G = np.stack(np.meshgrid(fx,fy,fz,indexing='ij'),-1) @ B
G_cart = jnp.asarray(G)

# Build a magnetized density: n_up != n_dn, smooth, with real gradients.
key = jax.random.PRNGKey(7)
base = jnp.abs(jax.random.normal(key,(nx,ny,nz)))*0.2 + 0.6
# smooth it (low-pass) so gradients are well-resolved on the coarse grid
bG = jnp.fft.fftn(base); kk = (G**2).sum(-1)
bG = bG * jnp.exp(-0.4*kk); base = jnp.real(jnp.fft.ifftn(bG))
base = jnp.abs(base)+0.3
zeta_field = 0.5*jnp.tanh(jax.random.normal(jax.random.PRNGKey(8),(nx,ny,nz)))  # |zeta|<0.5
n_up = base*(1+zeta_field)/2 + 0.05
n_dn = base*(1-zeta_field)/2 + 0.05

def E_xc(nu, nd):
    suu,sud,sdd = _compute_sigma_spin(jnp.fft.fftn(nu), jnp.fft.fftn(nd), G_cart)
    eps = pbe_xc_spin(nu, nd, suu, sud, sdd)
    return jnp.sum((nu+nd)*eps)

# Cell-volume-free FD: dE/dn_s at a point. The discrete E is sum over grid of
# f(r); functional deriv wrt n_s(r0) is (dE/dn_s)|r0 * (1) since the sum is the
# discretized integral with unit measure here. V from autodiff is the same measure.
V_up, V_dn = compute_V_xc_spin(n_up, n_dn, jnp.fft.fftn(n_up), jnp.fft.fftn(n_dn), G_cart)

rng = np.random.default_rng(0)
pts = [tuple(rng.integers(0,[nx,ny,nz])) for _ in range(6)]
eps_fd = 1e-6
print(f"{'point':>12} {'spin':>4} {'V_auto':>14} {'V_fd':>14} {'rel':>10}")
maxrel = 0.0
for p in pts:
    for s,(V,which) in enumerate([(V_up,'up'),(V_dn,'dn')]):
        def Ep(delta):
            nu = n_up; nd = n_dn
            if which=='up':
                nu = nu.at[p].add(delta)
            else:
                nd = nd.at[p].add(delta)
            return E_xc(nu, nd)
        dEp = (Ep(eps_fd)-Ep(-eps_fd))/(2*eps_fd)
        Va = float(V[p])
        rel = abs(Va-dEp)/(abs(dEp)+1e-12)
        maxrel = max(maxrel, rel)
        print(f"{str(p):>12} {which:>4} {Va:14.6f} {float(dEp):14.6f} {rel:10.2e}")
print(f"\nMAX rel(V_autodiff vs FD dE/dn) at finite zeta = {maxrel:.2e}")
print("PASS (<1e-4): cross-term/divergence flux is self-consistent at finite zeta"
      if maxrel<1e-4 else "FAIL: V_xc_spin does NOT equal dE/dn -> flux bug at finite zeta")
