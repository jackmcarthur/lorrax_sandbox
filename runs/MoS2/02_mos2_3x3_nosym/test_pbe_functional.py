"""Test LORRAX's PBE functional against known reference values."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
import jax.numpy as jnp

from jax_xc_local.pbe import pbe_xc, pbe_x, pbe_c

# Test points: (rs, sigma/sigma_0) where sigma_0 = (4kf^2 rho^2) 
# = 4*(3pi^2*rho)^{2/3}*rho^2 
# Known PBE values from reference implementations (in Hartree per electron)

test_cases = []
for rs in [1.0, 2.0, 5.0, 10.0]:
    rho = 3.0 / (4.0 * np.pi * rs**3)
    kf = (3.0 * np.pi**2 * rho) ** (1.0/3.0)
    sigma_0 = 4.0 * kf**2 * rho**2  # s=1 point
    
    for s_val in [0.0, 0.5, 1.0, 2.0]:
        sigma = s_val**2 * sigma_0
        test_cases.append((rs, rho, sigma, s_val))

print(f"{'rs':>5} {'s':>5} {'rho':>12} {'sigma':>12}  "
      f"{'ex_LX(Ha)':>12} {'ec_LX(Ha)':>12} {'exc_LX(Ry)':>12}")
print("-" * 95)

for rs, rho, sigma, s_val in test_cases:
    rho_j = jnp.array(rho)
    sigma_j = jnp.array(sigma)
    
    ex = float(pbe_x(rho_j, sigma_j))
    ec = float(pbe_c(rho_j, sigma_j))
    exc = float(pbe_xc(rho_j, sigma_j))
    
    # Check consistency: pbe_xc should be 2*(pbe_x + pbe_c)
    exc_check = 2.0 * (ex + ec)
    assert abs(exc - exc_check) < 1e-15, f"Inconsistency: {exc} vs {exc_check}"
    
    print(f"{rs:5.1f} {s_val:5.1f} {rho:12.6e} {sigma:12.6e}  "
          f"{ex:12.8f} {ec:12.8f} {exc:12.8f}")

# Now compare with QE's PBE values computed from the MoS2 charge density
# Load the actual MoS2 density and compute V_xc statistics
from psp.qe_save_reader import CrystalData
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.get_DFT_mtxels import load_pseudopotentials

crystal = CrystalData.from_qe_save("qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("qe/nscf/MoS2.save")
fft_grid = crystal.fft_grid
_nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])

V_loc_r, rho_core_r, rho_core_G = build_ionic_and_core(
    crystal, pseudos, fft_grid, truncation_2d=False)
rho_r, _ = crystal.load_charge_density()
rho_val = jnp.asarray(rho_r, dtype=jnp.float64)

# Total density
rho_total = rho_val + rho_core_r
rho_total_np = np.asarray(rho_total)

# Compute gradient
B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
G_cart = build_G_cart(_nx, _ny, _nz, B)
rho_core_gridded = jnp.real(jnp.fft.ifftn(rho_core_G))
rho_G_total = jnp.fft.fftn(rho_total - rho_core_gridded) + rho_core_G
grad_sq = jnp.zeros_like(rho_total)
for i in range(3):
    drho = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
    grad_sq = grad_sq + drho**2
sigma = jnp.maximum(grad_sq, 0.0)
sigma_np = np.asarray(sigma)

N = _nx * _ny * _nz
vol = crystal.cell_volume
print(f"\nMoS2 density statistics:")
print(f"  rho_total: min={rho_total_np.min():.4e} max={rho_total_np.max():.4e} mean={rho_total_np.mean():.4e}")
print(f"  sigma:     min={sigma_np.min():.4e} max={sigma_np.max():.4e} mean={sigma_np.mean():.4e}")

# Check integral
print(f"  integral(rho_val) * vol/N = {np.sum(np.asarray(rho_val)) * vol / N:.4f} electrons")
print(f"  integral(rho_core) * vol/N = {np.sum(np.asarray(rho_core_r)) * vol / N:.4f} electrons")
print(f"  integral(rho_total) * vol/N = {np.sum(rho_total_np) * vol / N:.4f} electrons")
print(f"  Expected: nelec = {crystal.nelec}")

# Compute rs distribution
rho_safe = np.maximum(rho_total_np, 1e-10)
rs = (3.0 / (4.0 * np.pi * rho_safe)) ** (1.0/3.0)
print(f"  rs: min={rs.min():.2f} max={rs.max():.2f} mean(where rho>0.01)={rs[rho_total_np>0.01].mean():.2f}")

# Reduced gradient s distribution  
kf = (3.0 * np.pi**2 * rho_safe) ** (1.0/3.0)
s_sq = sigma_np / (4.0 * kf**2 * rho_safe**2 + 1e-60)
s = np.sqrt(np.maximum(s_sq, 0.0))
print(f"  s: min={s[rho_total_np>0.01].min():.4f} max={s[rho_total_np>0.01].max():.4f} "
      f"mean={s[rho_total_np>0.01].mean():.4f}")

# Compare LORRAX pbe_xc with libxc-style computation for a grid of test points
# Using Parr-Yang reference formulas
print(f"\nReference comparison (LDA only, sigma=0):")
for rs_test in [1.0, 2.0, 5.0]:
    rho_test = 3.0 / (4.0 * np.pi * rs_test**3)
    kf_test = (3.0 * np.pi**2 * rho_test) ** (1.0/3.0)
    
    # Dirac exchange (Ha)
    ex_dirac = -(3.0/(4.0*np.pi)) * kf_test
    
    # PW92 correlation (the code's parametrization)
    A = 0.031091; a1 = 0.21370
    b1 = 7.5957; b2 = 3.5876; b3 = 1.6382; b4 = 0.49294
    rsh = np.sqrt(rs_test)
    f_val = b1*rsh + b2*rs_test + b3*rs_test*rsh + b4*rs_test**2
    ec_pw92 = -2*A*(1+a1*rs_test)*np.log(1 + 1/(2*A*f_val))
    
    # Code values
    ex_code = float(pbe_x(jnp.array(rho_test), jnp.array(0.0)))
    ec_code = float(pbe_c(jnp.array(rho_test), jnp.array(0.0)))
    
    print(f"  rs={rs_test}: ex_dirac={ex_dirac:.8f} ex_code={ex_code:.8f} diff={ex_code-ex_dirac:.2e}")
    print(f"         ec_pw92={ec_pw92:.8f}  ec_code={ec_code:.8f} diff={ec_code-ec_pw92:.2e}")
    
    # Known Ry values for verification
    ex_ry_known = 2.0 * ex_dirac  # Ry (since Dirac is in Ha)
    print(f"         ex_Ry_ref={ex_ry_known:.8f} exc_code_Ry={2*(ex_code+ec_code):.8f}")

