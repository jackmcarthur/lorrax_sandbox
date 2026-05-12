"""Test alpha-Z fix on both Si and MoS2."""
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

for name, save_dir, wfn_path, nbnd in [
    ("Si", "runs/Si/02_si_4x4x4_nosym/qe/nscf/silicon.save",
     "runs/Si/02_si_4x4x4_nosym/qe/nscf/WFN.h5", 8),
    ("MoS2", "runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save",
     "runs/MoS2/02_mos2_3x3_nosym/qe/nscf/WFN.h5", 26),
]:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    crystal = CrystalData.from_qe_save(save_dir)
    pseudos = load_pseudopotentials(save_dir)
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

    vnl_setup = vnl_ops.build_vnl_setup(
        crystal, pseudos=pseudos, nspinor=crystal.nspinor,
        q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)

    with h5py.File(wfn_path, "r") as f:
        evals_qe = f["mf_header/kpoints/el"][0, 0, :nbnd]

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

    diff = evals - evals_qe
    offset = np.mean(diff)
    mae = np.mean(np.abs(diff))
    mae_no_off = np.mean(np.abs(diff - offset))
    print(f"  Offset:      {offset*1000:+.3f} mRy")
    print(f"  MAE:         {mae*1000:.3f} mRy")
    print(f"  MAE-no-off:  {mae_no_off*1000:.3f} mRy")
    for ib in range(min(6, nbnd)):
        print(f"    band {ib}: err={diff[ib]*1000:+.3f} mRy")

