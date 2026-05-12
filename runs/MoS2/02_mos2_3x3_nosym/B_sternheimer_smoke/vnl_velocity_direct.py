"""Compare V_NL-only velocity matrix elements:

    compute_vnl_velocity_cart       (manual analytic dZ)
                 vs
    jax.jvp of apply_vnl wrt k      (autodiff through _build_vnl_kdata_core)

If these disagree, the discrepancy is inside the V_NL machinery itself.
If they agree, the ~7% gap observed in the full p + V_NL comparison must
be elsewhere (e.g., momentum operator path).
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import numpy as np
import jax, jax.numpy as jnp

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.vnl_ops import (_build_vnl_kdata_core, vnl_velocity_matrix, apply_vnl,
                         build_vnl_kdata_from_kvec)
from psp.get_dipole_mtxels import compute_vnl_velocity_cart

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
nb_cmp = min(80, int(wfn.nbands))
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=nb_cmp - n_occ, nband=nb_cmp,
                        n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(
    wfn, pseudos, rho_val, truncation_2d=True, verbose=False)

B_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
Binv = np.linalg.inv(B_cart)

ik = 4
kvec_k = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
kvec_k_j = jnp.asarray(kvec_k, dtype=jnp.float64)
H_k = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)

psi_k_box = load_kpoint_fftbox(wfn, sym, meta, ik, nb_cmp)
psi_k_G = _psi_box_to_G_sphere(psi_k_box, Gk_int)

# ── Path A: manual analytic (compute_vnl_velocity_cart) ──────────────────
vNL_manual_cart = compute_vnl_velocity_cart(
    psi_k_box, np.asarray(Gk_int), kvec_k, vnl_setup)   # (3, nb, nb) cart
# Note: this returns +(∂_q + ∂_q') V_NL. get_dipole_mtxels flips sign once more,
# so the final sign in dipole.h5 is NEGATIVE of this. Keep manual as-is here for
# direct autodiff comparison.

# ── Path B: jax.jvp through apply_vnl ────────────────────────────────────
# For each crystal direction a, compute d/dkvec_a (V_NL·ψ_n) — matrix element
# is  <m| dV_NL/dk_a |n>.  Then transform crystal → cart.
def _build_Z(k):
    kdata = _build_vnl_kdata_core(k, np.asarray(Gk_int), vnl_setup, compute_dZ=False)
    return kdata.Z

Z_primal = _build_Z(kvec_k_j)

def _apply_vnl_at(k):
    Z = _build_Z(k)
    return apply_vnl(psi_k_G, Z, jnp.asarray(H_k.vnl_E))

vNL_jvp_crys = np.zeros((3, nb_cmp, nb_cmp), dtype=np.complex128)
for a in range(3):
    dk = jnp.zeros(3, dtype=jnp.float64).at[a].set(1.0)
    _, dvNL_psi = jax.jvp(_apply_vnl_at, (kvec_k_j,), (dk,))
    # matrix element <m| dV_NL/dk_a |n>
    mtx = jnp.einsum('msG,nsG->mn', jnp.conj(psi_k_G), dvNL_psi, optimize=True)
    vNL_jvp_crys[a] = np.asarray(mtx)

# Transform crys → cart
vNL_jvp_cart = np.einsum('ia,amn->imn', Binv, vNL_jvp_crys)

print(f"\n── V_NL velocity only:  manual (dipole.h5 convention, before -1 flip) ──")
print(f"<c=26|v_NL|v=25>  (manual analytic, cart):")
for a, lbl in enumerate('xyz'):
    print(f"  α={lbl}: {vNL_manual_cart[a,26,25]:+.4e}")

print(f"\n── V_NL velocity only:  jax.jvp (autodiff through apply_vnl) ──")
print(f"<c=26|dV_NL/dk|v=25> cart:")
for a, lbl in enumerate('xyz'):
    print(f"  α={lbl}: {vNL_jvp_cart[a,26,25]:+.4e}")

# Ratio element-wise for a few key bands
print(f"\n── Ratio jvp / manual (real, c-v off-diagonal) ──")
for (c_, v_) in [(26, 25), (31, 6), (30, 3), (43, 20), (42, 21)]:
    for a, lbl in enumerate('xy'):
        m = vNL_manual_cart[a, c_, v_]
        j = vNL_jvp_cart[a, c_, v_]
        if abs(m) > 1e-6:
            r = j / m
            print(f"  (c={c_}, v={v_}), α={lbl}: manual = {m:+.3e}, jvp = {j:+.3e}, "
                  f"ratio = {float(r.real):+.4f} + {float(r.imag):+.3e}j")

mask = np.abs(vNL_manual_cart) > 1e-3
if mask.any():
    ratio = vNL_jvp_cart[mask] / vNL_manual_cart[mask]
    print(f"\n  Overall ratio stats (|manual| > 1e-3):")
    print(f"    real: median = {float(np.median(ratio.real)):.4f}, "
          f"mean = {float(np.mean(ratio.real)):.4f}, "
          f"std = {float(np.std(ratio.real)):.3e}")
    print(f"    imag: median = {float(np.median(ratio.imag)):.3e}, "
          f"std = {float(np.std(ratio.imag)):.3e}")
