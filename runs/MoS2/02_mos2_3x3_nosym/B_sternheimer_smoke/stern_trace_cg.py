"""Trace per-iter state of my CG for band 8 alone."""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec
from psp.h_dft import make_apply_H
from psp.run_sternheimer import build_sternheimer_source, make_density_perturbation, _psi_box_to_G_sphere
from solvers.projectors import make_P_val, make_Q_kminq

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

iq, ik_full = 1, 1
ik_kminq = int(sym.kq_map[ik_full, iq])
H_kminq = setup_H_k_from_kvec(np.asarray(sym.unfolded_kpts[ik_kminq]), V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
apply_H_kminq = make_apply_H(H_kminq)
Gkminq_int = jnp.asarray(np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32))
psi_k = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
psi_p = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
U_val_kminq_G = _psi_box_to_G_sphere(psi_p, Gkminq_int)
P_val_kminq = make_P_val(U_val_kminq_G)
Q_kminq = make_Q_kminq(U_val_kminq_G)
eps_vk = jnp.asarray(wfn.energies[0, sym.irk_to_k_map[ik_full], :n_occ])
b_full = build_sternheimer_source(psi_k, Gkminq_int, make_density_perturbation(wfn.fft_grid), Q_kminq)

# Band 8 isolated
v_idx = jnp.array([8])
b = -b_full[v_idx]   # note the negation
eps_v = eps_vk[v_idx]
alpha_pv = 8.688
def apply_A(x):
    return (apply_H_kminq(x) - eps_v[:, None, None].astype(x.dtype) * x
            + alpha_pv * P_val_kminq(x))

# Manual CG, no precond
x = jnp.zeros_like(b)
r = b - apply_A(x)
z = r
p = z
rho = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(r), z))
print(f"iter  0: ||x||=0  ||r||={float(jnp.sqrt(jnp.real(jnp.einsum("vsG,vsG->v", jnp.conj(r), r))[0])):.3e}  rho={float(rho[0]):.3e}")

for it in range(1, 20):
    Ap = apply_A(p)
    pAp = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(p), Ap))
    alpha = rho / pAp
    x = x + alpha[:, None, None].astype(x.dtype) * p
    r = r - alpha[:, None, None].astype(x.dtype) * Ap
    z = r
    rho_new = jnp.real(jnp.einsum('vsG,vsG->v', jnp.conj(r), z))
    beta = rho_new / rho
    p = z + beta[:, None, None].astype(x.dtype) * p
    rho = rho_new
    xn = float(jnp.sqrt(jnp.real(jnp.einsum("vsG,vsG->v", jnp.conj(x), x))[0]))
    rn = float(jnp.sqrt(jnp.real(jnp.einsum("vsG,vsG->v", jnp.conj(r), r))[0]))
    pApv = float(pAp[0])
    av = float(alpha[0])
    bv = float(beta[0])
    print(f"iter {it:2d}: ||x||={xn:.3e}  ||r||={rn:.3e}  α={av:.3e}  β={bv:.3e}  <p,Ap>={pApv:.3e}  rho={float(rho[0]):.3e}")
