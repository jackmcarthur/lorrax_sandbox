"""GATE 0 — spin-PBE unit tests (no WFN, no GPU physics).
(a) eps_xc reduction: pbe_xc_spin(n/2,n/2,s/4,s/4,s/4) == pbe_xc(n,s)  [<1e-12]
(b) V reduction:  compute_V_xc_spin(n/2,n/2,rhoG/2,rhoG/2) -> V_up==V_dn,
                  and (V_up+V_dn)/2 == compute_V_xc(n,rhoG)            [<1e-10]
(c) bracket sanity: eps_c(rs=2, zeta) monotone, with zeta=0.5 strictly
    between zeta=0 and zeta=1 (a sign slip in the alpha_c term breaks this)."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from jax_xc_local.pbe import pbe_xc, pbe_c
from psp.xc import pbe_xc_spin, _pbe_c_spin, compute_V_xc, compute_V_xc_spin

ok = True

# (a) eps_xc reduction on random fields
key = jax.random.PRNGKey(0)
n = jnp.abs(jax.random.normal(key, (8, 8, 8))) + 0.05
s = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (8, 8, 8))) * 0.3
lhs = pbe_xc_spin(n / 2, n / 2, s / 4, s / 4, s / 4)
rhs = pbe_xc(n, s)
da = float(jnp.max(jnp.abs(lhs - rhs)))
print(f"(a) eps_xc reduction  max|spin(n/2)-unpol| = {da:.2e}  {'PASS' if da<1e-12 else 'FAIL'}")
ok &= da < 1e-12

# (b) V reduction on a real grid (need rho_G + G_cart)
nx = ny = nz = 12
B = np.diag([0.7, 0.7, 0.35])               # arbitrary reciprocal cell (bohr^-1)
fx = np.fft.fftfreq(nx) * nx; fy = np.fft.fftfreq(ny) * ny; fz = np.fft.fftfreq(nz) * nz
G = np.stack(np.meshgrid(fx, fy, fz, indexing='ij'), -1) @ B
G_cart = jnp.asarray(G)
rho = jnp.abs(jax.random.normal(jax.random.PRNGKey(2), (nx, ny, nz))) + 0.3
rho_G = jnp.fft.fftn(rho)
V_un = compute_V_xc(rho, rho_G, G_cart)
V_up, V_dn = compute_V_xc_spin(rho / 2, rho / 2, rho_G / 2, rho_G / 2, G_cart)
d_ud = float(jnp.max(jnp.abs(V_up - V_dn)))
d_mean = float(jnp.max(jnp.abs(0.5 * (V_up + V_dn) - V_un)))
print(f"(b) V reduction       max|V_up-V_dn| = {d_ud:.2e}  max|mean-unpol| = {d_mean:.2e}  "
      f"{'PASS' if (d_ud<1e-10 and d_mean<1e-10) else 'FAIL'}")
ok &= d_ud < 1e-10 and d_mean < 1e-10

# (c) bracket sanity (homogeneous, sigma=0): eps_c(rs=2, zeta)
def ec_at(n_tot, zeta):
    nu = n_tot * (1 + zeta) / 2; nd = n_tot * (1 - zeta) / 2
    return float(_pbe_c_spin(jnp.array(nu), jnp.array(nd), jnp.array(0.0)))
n2 = 3.0 / (4.0 * np.pi * 2.0 ** 3)         # rs=2 -> n
e0, e5, e1 = ec_at(n2, 0.0), ec_at(n2, 0.5), ec_at(n2, 1.0)
brack = (e1 < e5 < e0)                       # all negative, |e1|<|e0|; e5 between
print(f"(c) eps_c(rs=2): zeta=0 {e0:.5f}  0.5 {e5:.5f}  1 {e1:.5f} Ha  "
      f"bracket {'PASS' if brack else 'FAIL'}")
# also confirm unpolarized _pbe_c_spin(n/2,n/2,0)==pbe_c(n,0)
dc = float(jnp.abs(_pbe_c_spin(jnp.array(n2/2), jnp.array(n2/2), jnp.array(0.0)) - pbe_c(jnp.array(n2), jnp.array(0.0))))
print(f"    eps_c reduce at zeta=0: |spin-unpol| = {dc:.2e}  {'PASS' if dc<1e-13 else 'FAIL'}")
ok &= brack and dc < 1e-13

print("GATE 0:", "ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
