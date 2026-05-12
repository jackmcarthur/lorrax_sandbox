"""Quick Si check: verify eigenvalue offset is negligible for Si."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp, h5py

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.dft_operators import (compute_V_H_and_V_xc, build_V_scf, compute_ngkmax,
                                setup_H_k_from_kvec, apply_H_k)
from psp.davidson import davidson_k, warmup_jit
import psp.vnl_ops as vnl_ops

crystal = CrystalData.from_qe_save("runs/Si/02_si_4x4x4_nosym/qe/nscf/silicon.save")
pseudos = load_pseudopotentials("runs/Si/02_si_4x4x4_nosym/qe/nscf/silicon.save")
fft_grid = crystal.fft_grid
_nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])

V_loc_r, rho_core_r, rho_core_G = build_ionic_and_core(
    crystal, pseudos, fft_grid, truncation_2d=False)
rho_r, _ = crystal.load_charge_density()
rho_val = jnp.asarray(rho_r, dtype=jnp.float64)
B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
G_cart = build_G_cart(_nx, _ny, _nz, B)

V_H_r, V_xc_r = compute_V_H_and_V_xc(
    rho_val, rho_core_r, rho_core_G, G_cart,
    jnp.asarray(crystal.bdot, dtype=jnp.float64),
    jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat)
V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)
jax.block_until_ready(V_scf)

rho_core_np = np.asarray(rho_core_r)
N = _nx * _ny * _nz
vol = crystal.cell_volume
core_integral = float(np.sum(rho_core_np)) * vol / N
print(f"Si: Core density integral: {core_integral:.4f} e ({core_integral/crystal.nelec:.1%} of valence)")

vnl_setup = vnl_ops.build_vnl_setup(
    crystal, pseudos=pseudos, nspinor=crystal.nspinor,
    q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)

with h5py.File("runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5", "r") as f:
    evals_qe = f["mf_header/kpoints/el"][0, 0, :]  # Gamma

nbnd = 8  # occupied bands for Si
bdot = np.asarray(crystal.bdot, dtype=float)
kpoint = np.array([0.0, 0.0, 0.0])
ngkmax = compute_ngkmax(kpoint.reshape(1,3), bdot, crystal.ecutwfc, fft_grid)
warmup_jit(ngkmax, crystal.nspinor, nbnd)

H_k = setup_H_k_from_kvec(kpoint, V_scf, vnl_setup, crystal, None,
                            V_loc_r=V_loc_r, ngkmax=ngkmax)
def apply_H(psi_G, _H=H_k):
    mask_f = _H.mask[None, None, :].astype(psi_G.dtype)
    psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
    psi_box = psi_box.at[:, :, _H.Gx, _H.Gy, _H.Gz].add(psi_G * mask_f)
    return apply_H_k(psi_box, _H.T_diag, _H.V_scf,
                      _H.Gx, _H.Gy, _H.Gz,
                      _H.vnl_Z, _H.vnl_E, _H.mask)

evals, _ = davidson_k(
    apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=crystal.nspinor,
    n_tgt=nbnd, T_diag=H_k.T_diag, verbose=False, tol=1e-8)
evals = np.asarray(evals)

diff = evals - evals_qe[:nbnd]
offset = np.mean(diff)
print(f"\nSi eigenvalue comparison (Gamma, {nbnd} occ bands):")
print(f"  Offset: {offset*1000:.3f} mRy")
print(f"  MAE: {np.mean(np.abs(diff))*1000:.3f} mRy")
print(f"  MAE-no-offset: {np.mean(np.abs(diff-offset))*1000:.3f} mRy")
for ib in range(nbnd):
    print(f"  band {ib}: QE={evals_qe[ib]:.6f} LX={evals[ib]:.6f} err={diff[ib]*1000:+.3f} mRy")
