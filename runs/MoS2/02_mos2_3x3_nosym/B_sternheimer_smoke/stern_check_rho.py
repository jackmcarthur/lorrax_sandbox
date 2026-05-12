"""Compare my ρ_val from full-BZ sum against QE's SCF density."""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
import h5py
from common import Meta, symmetry_maps
from file_io import WFNReader, CrystalData
from psp.scf_potential import build_rho_val_from_wfn
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials
from psp.dft_operators import setup_H_k_from_kvec
from psp.h_dft import make_apply_H
from common.load_wfns import load_kpoint_fftbox
from psp.run_sternheimer import _psi_box_to_G_sphere

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)

# My ρ
rho_mine = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
print(f"My rho: integral={float(jnp.sum(rho_mine)) * float(wfn.cell_volume) / np.prod(wfn.fft_grid):.4f}")

# QE's ρ via CrystalData
crystal = CrystalData.from_qe_save('/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/02_mos2_3x3_nosym/qe/scf/MoS2.save')
rho_qe, _ = crystal.load_charge_density()
rho_qe = jnp.asarray(rho_qe, dtype=jnp.float64)
print(f"QE rho: integral={float(jnp.sum(rho_qe)) * float(wfn.cell_volume) / np.prod(wfn.fft_grid):.4f}")
print(f"shapes: mine {rho_mine.shape}  qe {rho_qe.shape}")

# Compare pointwise
diff = jnp.abs(rho_mine - rho_qe)
print(f"|ρ_mine - ρ_qe|: max={float(jnp.max(diff)):.3e}  mean={float(jnp.mean(diff)):.3e}  vs max ρ_qe={float(jnp.max(rho_qe)):.3e}")

# Try H built from QE rho + check u|H|u
pseudos = load_pseudopotentials('.')
V_scf_qe, V_loc_qe, vnl_setup_qe = build_dft_potentials(
    wfn, pseudos, rho_qe, truncation_2d=True, verbose=False)
H_Gamma_qe = setup_H_k_from_kvec(np.zeros(3), V_scf_qe, vnl_setup_qe, wfn, meta, V_loc_r=V_loc_qe)
apply_H_qe = make_apply_H(H_Gamma_qe)
Gg = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, 0), dtype=np.int32))
psi_g = load_kpoint_fftbox(wfn, sym, meta, 0, n_occ)
U_Gamma = _psi_box_to_G_sphere(psi_g, Gg)
HU = apply_H_qe(U_Gamma)
diag_qe = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(U_Gamma), HU))
eps = np.asarray(wfn.energies[0, sym.irk_to_k_map[0], :n_occ])
print(f"\nQE rho: <u|H|u>[:6] = {np.asarray(diag_qe[:6])}")
print(f"     eps[:6]        = {eps[:6]}")
print(f"     diff[:6]       = {np.asarray(diag_qe[:6]) - eps[:6]}")
