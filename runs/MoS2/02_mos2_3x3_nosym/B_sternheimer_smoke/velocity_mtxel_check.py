"""Compare <c|∂H/∂kvec_p|v> (Stern-side) to dipole_cart[α, k, c, v] (dipole.h5).

If the two match to numerical precision for all (v, c, k, α), then the
residual factor of 1.64 between Stern and SoS S-tensor is a
derivation/normalisation issue, not an operator mismatch.

Both objects should be identical — k·p says
    <u_c|∂H/∂kvec_p_α|u_v>  =  <ψ_c| v_α |ψ_v>
where v_α = p_α + i·[r_α, V_NL] is the velocity operator (Peierls).
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import h5py
import numpy as np
import jax, jax.numpy as jnp

from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.scf_potential import build_dft_potentials, build_rho_val_from_wfn
from psp.dft_operators import setup_H_k_from_kvec
from psp.run_sternheimer import (_psi_box_to_G_sphere,
                                  build_sternheimer_op_at_kvec_traced)
from solvers.sternheimer_solve import _apply_A_inline
from psp.dft_operators import apply_H_k as apply_H_k_fn

# ── System setup (same as other smoke scripts) ───────────────────────────
wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
nb_tot = int(wfn.nbands)            # 82
nb_cmp = min(80, nb_tot)            # dipole.h5 has 80 bands
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=nb_cmp - n_occ, nband=nb_cmp,
                        n_rmu=0, bispinor=False)
pseudos = load_pseudopotentials('.')
rho_val = build_rho_val_from_wfn(wfn, sym, meta, n_occ, verbose=False)
V_scf, V_loc, vnl_setup = build_dft_potentials(
    wfn, pseudos, rho_val, truncation_2d=True, verbose=False)
bdot = jnp.asarray(wfn.bdot, dtype=jnp.float64)
alpha_pv_j = jnp.asarray(1.0, dtype=jnp.float64)   # dummy; not used here

# ── Load dipole.h5 reference (in same cartesian convention) ──────────────
with h5py.File('../00_lorrax_cohsex/dipole.h5', 'r') as h5:
    dipole_cart_ref = np.asarray(h5['dipole_cart'])     # (3, nk, nb, nb)
    # 'dipole_cart[α, k, m, n] = <mk|v_α|nk>'  with  v_α = p_α + i[r_α, V_NL]

# ── Compute <m|∂H/∂kvec_p_α|n> for one k-point, all m,n up to nb_cmp ─────
B_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)   # (3, 3) cart

ik = 4   # K-point index (arbitrary — (1/3, 1/3, 0))
kvec_k = np.asarray(sym.unfolded_kpts[ik], dtype=np.float64)
H_k = setup_H_k_from_kvec(kvec_k, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc)
Gk_int = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)

psi_k_box = load_kpoint_fftbox(wfn, sym, meta, ik, nb_cmp)
# gather to G-sphere
psi_k_G = _psi_box_to_G_sphere(psi_k_box, Gk_int)    # (nb, nspinor, nG)

# Dummy precond_diag and eps_v (the op tangent doesn't use them)
nG = Gk_int.shape[0]; nspinor = int(wfn.nspinor)
precond_diag = jnp.ones((n_occ, 1, nG), dtype=jnp.float64)
eps_v_dummy = jnp.zeros(n_occ, dtype=jnp.float64)
U_val_G_dummy = psi_k_G[:n_occ]   # for operator building

kvec_k_j = jnp.asarray(kvec_k, dtype=jnp.float64)

from psp.vnl_ops import _build_vnl_kdata_core

def _H_apply_at(k, x):
    """Apply H(kvec_p=k) to x — bypasses SternheimerOp's eps_v/alpha_pv which
    don't matter for the k-tangent anyway."""
    Gk_float = jnp.asarray(np.asarray(Gk_int), dtype=jnp.float64)
    kG = Gk_float + k[None, :]
    T_diag_k = jnp.einsum('gi,ij,gj->g', kG, bdot, kG)
    kdata = _build_vnl_kdata_core(k, np.asarray(Gk_int), vnl_setup, compute_dZ=False)
    return apply_H_k_fn(
        jnp.zeros((*x.shape[:2], *H_k.fft_grid), dtype=x.dtype
        ).at[:, :, H_k.Gx, H_k.Gy, H_k.Gz].add(x * H_k.mask[None, None, :].astype(x.dtype)),
        T_diag_k, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
        kdata.Z, H_k.vnl_E, H_k.mask,
    )

def dH_dk_apply(k_dir_crys, x):
    """Return  (∂H/∂kvec_p  along crystal direction k_dir) applied to x."""
    _, out = jax.jvp(lambda k: _H_apply_at(k, x),
                     (kvec_k_j,), (k_dir_crys,))
    return out

# Stern velocity in crystal coords:  <m|∂H/∂kvec_p_a|n>   for a ∈ {0,1,2}
# crystal. Then transform to cart via chain rule:
#    <m|∂H/∂q_i_cart|n>  =  Σ_a (∂q_a_crys/∂q_i_cart)·<m|∂H/∂kvec_p_a|n>
# With q_cart = q_crys · B_cart,  ∂q_a_crys/∂q_i_cart = (B_cart⁻¹)_ai  (row a col i).
Binv = np.linalg.inv(B_cart)

# Apply ∂H/∂kvec_p for each crystal direction to all nb_cmp bands at once.
# Then compute matrix elements <m|·|n>.
nb_stern = nb_cmp
all_bands_G = psi_k_G                    # (nb, ns, nG)
e_vec_crys = [jnp.zeros(3, dtype=jnp.float64).at[a].set(1.0) for a in range(3)]
vel_stern_crys = np.zeros((3, nb_stern, nb_stern), dtype=np.complex128)
for a in range(3):
    Hk_times_all = dH_dk_apply(e_vec_crys[a], all_bands_G)   # (nb, ns, nG)
    # <m|dH/dk_a|n> = Σ_{s,G} conj(psi_k_G[m,s,G]) · Hk_times_all[n,s,G]
    mtx = jnp.einsum('msG,nsG->mn',
                      jnp.conj(psi_k_G), Hk_times_all, optimize=True)
    vel_stern_crys[a] = np.asarray(mtx)

# Transform crystal → cartesian:  vel_cart_i = Σ_a (Binv)_{i,a} · vel_crys_a
# (chain rule through kvec_cart_i = Σ_a kvec_crys_a · B_{a,i}.)
vel_stern_cart = np.einsum('ia,amn->imn', Binv, vel_stern_crys)

# ── Compare with dipole.h5 for ik ──
vel_dip_cart = dipole_cart_ref[:, ik, :nb_stern, :nb_stern]

print(f"ik = {ik}   kvec_crys = {kvec_k}")
print(f"nb compared = {nb_stern}")

# Diagonal (v = c) — velocity diag matches momentum ⟨p⟩ for that band.
print(f"\n<n|v|n> (diagonal, 5 lowest) — Stern (cart) vs dipole:")
print(f"  band    Stern_x          Dipole_x       Stern_y          Dipole_y")
for n in range(5):
    print(f"  {n:>3}   {vel_stern_cart[0,n,n].real:+.4e}   {vel_dip_cart[0,n,n].real:+.4e}   "
          f"{vel_stern_cart[1,n,n].real:+.4e}   {vel_dip_cart[1,n,n].real:+.4e}")

# Off-diagonal key c-v pair (band 26 = first cond, band 25 = HOMO)
c = n_occ
v = n_occ - 1
print(f"\n<c={c}|v_α|v={v}> off-diagonal — Stern vs dipole (complex):")
for alpha, label in enumerate(['x', 'y', 'z']):
    s = vel_stern_cart[alpha, c, v]
    d = vel_dip_cart[alpha, c, v]
    rel = abs(s - d) / (abs(d) + 1e-30)
    print(f"  α={label}: Stern = {s:+.4e}  Dipole = {d:+.4e}  rel_err = {rel:.3e}")

# Aggregate: max rel err over all (m, n, α) with |dipole| > 1e-4
mask = np.abs(vel_dip_cart) > 1e-4
rel_err = np.abs(vel_stern_cart - vel_dip_cart) / np.where(mask, np.abs(vel_dip_cart), 1.0)
print(f"\nMax rel error over |dipole|>1e-4:  {float(np.max(rel_err[mask])):.3e}")
print(f"Mean rel error over |dipole|>1e-4: {float(np.mean(rel_err[mask])):.3e}")

ratio = vel_stern_cart[mask] / vel_dip_cart[mask]
print(f"\nElementwise ratio Stern/Dipole (over |dipole|>1e-4):")
print(f"  real: median = {float(np.median(ratio.real)):.4f}, std = {float(np.std(ratio.real)):.3e}")
print(f"  imag: median = {float(np.median(ratio.imag)):.4f}, std = {float(np.std(ratio.imag)):.3e}")

# Top-10 dominant off-diagonal matrix elements — is the ratio consistent?
print(f"\nTop-10 largest |dipole| elements (c-v off-diag only, in-plane):")
vel_dip_cv = vel_dip_cart[:2, n_occ:, :n_occ]    # (2, nc, nv) — xy only
vel_stern_cv = vel_stern_cart[:2, n_occ:, :n_occ]
flat_dip = vel_dip_cv.ravel()
flat_stern = vel_stern_cv.ravel()
sorted_idx = np.argsort(-np.abs(flat_dip))[:10]
shape_cv = vel_dip_cv.shape
print(f"  {'α':>2} {'c':>3} {'v':>3}   {'|Stern|':>10}  {'|Dipole|':>10}  {'|Stern|/|Dipole|':>16}  {'phase diff':>12}")
for idx in sorted_idx:
    a, c_loc, v_loc = np.unravel_index(idx, shape_cv)
    s = flat_stern[idx]; d = flat_dip[idx]
    mag_ratio = abs(s) / abs(d)
    phase = np.angle(s / d)
    print(f"  {'xy'[a]:>2} {c_loc+n_occ:>3} {v_loc:>3}   "
          f"{abs(s):10.4e}  {abs(d):10.4e}  {mag_ratio:16.4f}  {phase:+12.4e}")

# Key question: does |Stern|²/|Dipole|² average to ~0.82 (which would kill the factor of 2)?
mag_sq_ratio = np.abs(flat_stern[sorted_idx])**2 / np.abs(flat_dip[sorted_idx])**2
print(f"\nMean |Stern|²/|Dipole|² for top-10 dominant c-v elements: {float(np.mean(mag_sq_ratio)):.4f}")
print(f"(If this × 2 ≈ 1.64, then factor 2 is right and ~{(1-np.sqrt(1.64/2))*100:.0f}% of the velocity-op amplitude is missing from Stern.)")
