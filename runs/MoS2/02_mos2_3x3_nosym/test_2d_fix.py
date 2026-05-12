"""Test with 2D truncation on BOTH V_loc and V_H."""
import os; os.environ['JAX_ENABLE_X64']='1'
import numpy as np, jax, jax.numpy as jnp, h5py

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials, poisson_potential_from_rhoG
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.dft_operators import build_V_scf, compute_ngkmax, setup_H_k_from_kvec, apply_H_k
from psp.davidson import davidson_k, warmup_jit
import psp.vnl_ops as vnl_ops

crystal = CrystalData.from_qe_save("qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("qe/nscf/MoS2.save")
fft = crystal.fft_grid
nx, ny, nz = int(fft[0]), int(fft[1]), int(fft[2])

# V_loc with 2D truncation
V_loc, rc_r, rc_G = build_ionic_and_core(crystal, pseudos, fft, truncation_2d=True)

# V_H with 2D truncation (bypassing the hardcoded False)
rho_r, _ = crystal.load_charge_density()
rho_val = jnp.asarray(rho_r, dtype=jnp.float64)
rho_G_ortho = jnp.fft.fftn(rho_val, norm='ortho')
V_H_r = jnp.real(poisson_potential_from_rhoG(
    rho_G_ortho,
    jnp.asarray(crystal.bdot, dtype=jnp.float64),
    jnp.asarray(crystal.bvec, dtype=jnp.float64),
    crystal.blat,
    truncation_2d=True))  # ← 2D truncation for V_H

# V_xc (same as before — no truncation needed for V_xc)
B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
G_cart = build_G_cart(nx, ny, nz, B)
from jax_xc_local.pbe import pbe_xc

rho_total = rho_val + rc_r
rho_safe = jnp.maximum(rho_total, 1e-10)
rho_core_gridded = jnp.real(jnp.fft.ifftn(rc_G))
rho_G_total = jnp.fft.fftn(rho_total - rho_core_gridded) + rc_G
grad_rho_sq = jnp.zeros_like(rho_total)
for i in range(3):
    drho = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
    grad_rho_sq += drho ** 2
sigma = jnp.maximum(grad_rho_sq, 0.0)

def E_xc_lda(rho): return jnp.sum(rho * pbe_xc(rho, jnp.zeros_like(rho)))
def E_xc_full(rho, sig): return jnp.sum(rho * pbe_xc(rho, sig))
df_drho_lda = jax.grad(E_xc_lda)(rho_safe)
df_drho_full = jax.grad(E_xc_full, argnums=0)(rho_safe, sigma)
df_dsigma = jax.grad(E_xc_full, argnums=1)(rho_safe, sigma)
gga_mask = (rho_total > 1e-6) & (grad_rho_sq > 1e-10)
df_drho = df_drho_lda + jnp.where(gga_mask, df_drho_full - df_drho_lda, 0.0)
df_dsigma = jnp.where(gga_mask, df_dsigma, 0.0)
gga_corr = jnp.zeros_like(rho_total)
for i in range(3):
    drho_ri = jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * rho_G_total))
    h_i_G = jnp.fft.fftn(df_dsigma * drho_ri)
    gga_corr += jnp.real(jnp.fft.ifftn(1j * G_cart[..., i] * h_i_G))
V_xc_r = df_drho - 2.0 * gga_corr

V_scf = build_V_scf(V_loc, V_H_r, V_xc_r)
jax.block_until_ready(V_scf)

# Diagonalize at Gamma
vnl = vnl_ops.build_vnl_setup(crystal, pseudos=pseudos, nspinor=crystal.nspinor,
                               q_max=float(np.sqrt(float(crystal.ecutwfc)))*1.01)
bdot = np.asarray(crystal.bdot, dtype=float)
k = np.array([0.,0.,0.])
ngkmax = compute_ngkmax(k.reshape(1,3), bdot, crystal.ecutwfc, fft)
warmup_jit(ngkmax, crystal.nspinor, 26)
H_k = setup_H_k_from_kvec(k, V_scf, vnl, crystal, None, V_loc_r=V_loc, ngkmax=ngkmax)
def applyH(psi, _H=H_k):
    m = _H.mask[None,None,:].astype(psi.dtype)
    box = jnp.zeros((*psi.shape[:2],nx,ny,nz), dtype=psi.dtype)
    box = box.at[:,:,_H.Gx,_H.Gy,_H.Gz].add(psi*m)
    return apply_H_k(box, _H.T_diag, _H.V_scf, _H.Gx, _H.Gy, _H.Gz, _H.vnl_Z, _H.vnl_E, _H.mask)
evals, _ = davidson_k(applyH, h_diag=H_k.h_diag, nG=ngkmax, nspinor=crystal.nspinor,
                       n_tgt=26, T_diag=H_k.T_diag, verbose=False, tol=1e-8)
evals = np.asarray(evals)

with h5py.File("qe/nscf/WFN.h5","r") as f:
    eqe = f["mf_header/kpoints/el"][0,0,:26]

diff = evals - eqe
print(f"With 2D truncation on V_loc AND V_H:")
print(f"  Offset:     {np.mean(diff)*1000:+.3f} mRy")
print(f"  MAE:        {np.mean(np.abs(diff))*1000:.3f} mRy")
print(f"  MAE-no-off: {np.mean(np.abs(diff-np.mean(diff)))*1000:.3f} mRy")
for ib in range(6):
    print(f"  band {ib}: QE={eqe[ib]:.6f} LX={evals[ib]:.6f} err={diff[ib]*1000:+.3f} mRy")
