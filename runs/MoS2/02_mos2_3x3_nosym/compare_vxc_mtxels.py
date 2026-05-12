"""Compare LORRAX V_xc matrix elements <n|V_xc|n> with QE's vxc.dat.

This separates the error into V_xc, V_loc, V_H, and V_NL contributions.
"""
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

# ---------- Setup ----------
crystal = CrystalData.from_qe_save("qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("qe/nscf/MoS2.save")
fft_grid = crystal.fft_grid
_nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
N = _nx * _ny * _nz

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

vnl_setup = vnl_ops.build_vnl_setup(
    crystal, pseudos=pseudos, nspinor=crystal.nspinor,
    q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)

# ---------- Get QE eigenvectors and eigenvalues ----------
nbnd = 26
bdot = np.asarray(crystal.bdot, dtype=float)
kpoint = np.array([0.0, 0.0, 0.0])
ngkmax = compute_ngkmax(kpoint.reshape(1,3), bdot, crystal.ecutwfc, fft_grid)

# Get LORRAX eigenvectors via Davidson with full V_scf
V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)
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

evals_lx, evecs_lx = davidson_k(
    apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=crystal.nspinor,
    n_tgt=nbnd, T_diag=H_k.T_diag, verbose=False, tol=1e-8)

# Compute <n|V_op|n> for each operator using LORRAX wavefunctions
# Using ortho FFT convention: <n|V|n> = sum_r |psi_n(r)|^2 * V(r)
# where psi_n(r) = ifftn(psi_box, 'ortho')

mask_f = H_k.mask[None, None, :].astype(jnp.complex128)
psi_box = jnp.zeros((nbnd, crystal.nspinor, _nx, _ny, _nz), dtype=jnp.complex128)
psi_box = psi_box.at[:, :, H_k.Gx, H_k.Gy, H_k.Gz].add(evecs_lx * mask_f)

# Real-space wavefunctions (ortho convention)
psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
# Density: |psi_n(r)|^2 = sum over spinor components
rho_n_r = jnp.sum(jnp.abs(psi_r)**2, axis=1)  # (nbnd, nx, ny, nz)

# Compute <n|V_op|n> for each operator
vxc_lx = jnp.sum(rho_n_r * V_xc_r[None, :, :, :], axis=(1,2,3))  # (nbnd,)
vh_lx = jnp.sum(rho_n_r * jnp.asarray(V_H_r)[None, :, :, :], axis=(1,2,3))
vloc_lx = jnp.sum(rho_n_r * jnp.asarray(V_loc_r)[None, :, :, :], axis=(1,2,3))
vscf_lx = jnp.sum(rho_n_r * jnp.asarray(V_scf)[None, :, :, :], axis=(1,2,3))

# Parse QE vxc.dat
vxc_qe = []
with open("qe/nscf/vxc.dat") as f:
    ik_count = -1
    for line in f:
        parts = line.split()
        if len(parts) == 5:
            ik_count += 1
        elif len(parts) == 4 and ik_count == 0:
            vxc_qe.append(float(parts[2]))
vxc_qe = np.array(vxc_qe[:nbnd]) / 13.6057  # eV → Ry

# Parse QE kih.dat (kinetic + ionic + Hartree)
kih_qe = []
with open("qe/nscf/kih.dat") as f:
    ik_count = -1
    for line in f:
        parts = line.split()
        if len(parts) == 5:
            ik_count += 1
        elif len(parts) == 4 and ik_count == 0:
            kih_qe.append(float(parts[2]))
kih_qe = np.array(kih_qe[:nbnd]) / 13.6057  # eV → Ry

# QE eigenvalues from WFN.h5
with h5py.File("qe/nscf/WFN.h5", "r") as f:
    evals_qe = f["mf_header/kpoints/el"][0, 0, :nbnd]

# QE decomposition: e_QE = kih_QE + vxc_QE  (total = T + V_ion + V_H + V_xc)
# So kih_QE = T + V_ion + V_H
evals_qe_check = kih_qe + vxc_qe

vxc_lx_np = np.asarray(vxc_lx)
vloc_lx_np = np.asarray(vloc_lx)
vh_lx_np = np.asarray(vh_lx)

print("=" * 80)
print("Component-wise comparison: LORRAX vs QE at Gamma")
print("=" * 80)

print(f"\n{'Band':>4} {'e_QE':>10} {'e_LX':>10} {'kih_QE':>10} {'vxc_QE':>10} {'vxc_LX':>10} {'dvxc':>10}")
print(f"{'':>4} {'(Ry)':>10} {'(Ry)':>10} {'(Ry)':>10} {'(Ry)':>10} {'(Ry)':>10} {'(mRy)':>10}")
print("-" * 80)
for ib in range(min(nbnd, 20)):
    de = (float(evals_lx[ib]) - evals_qe[ib]) * 1000
    dvxc = (vxc_lx_np[ib] - vxc_qe[ib]) * 1000
    print(f"{ib:4d} {evals_qe[ib]:10.4f} {float(evals_lx[ib]):10.4f} "
          f"{kih_qe[ib]:10.4f} {vxc_qe[ib]:10.4f} {vxc_lx_np[ib]:10.4f} {dvxc:+10.3f}")

# Summary statistics
diff_eval = np.asarray(evals_lx) - evals_qe
diff_vxc = vxc_lx_np - vxc_qe

print(f"\n{'Quantity':<20} {'Offset (mRy)':>15} {'MAE (mRy)':>15} {'MAE-no-off':>15}")
print("-" * 70)
for name, d in [("eigenvalues", diff_eval), ("V_xc mtxels", diff_vxc)]:
    off = np.mean(d) * 1000
    mae = np.mean(np.abs(d)) * 1000
    mae_no = np.mean(np.abs(d - np.mean(d))) * 1000
    print(f"{name:<20} {off:15.3f} {mae:15.3f} {mae_no:15.3f}")

# Residual: eigenvalue error minus V_xc error 
# This captures the T + V_loc + V_H + V_NL contribution to the error
residual = diff_eval - diff_vxc
print(f"{'residual (e-vxc)':<20} {np.mean(residual)*1000:15.3f} "
      f"{np.mean(np.abs(residual))*1000:15.3f} "
      f"{np.mean(np.abs(residual-np.mean(residual)))*1000:15.3f}")

