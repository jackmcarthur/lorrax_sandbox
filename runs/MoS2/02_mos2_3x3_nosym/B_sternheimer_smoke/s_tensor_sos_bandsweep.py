"""Scan SoS S-tensor wrt number of bands to check convergence vs Stern.

Uses existing dipole.h5 (nb=80) and applies ``compute_S_omega(ω=0)`` with a
*reduced* max-band cutoff by zeroing out the contributions from c > nb_use.
Print S_xx (cart) at each nb and 1/nb extrapolation, comparing against the
Stern value (same basis transform applied for a fair comparison).
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import numpy as np
import jax, jax.numpy as jnp
import h5py
def read_dipole_h5(path):
    with h5py.File(path, 'r') as h5:
        d = np.asarray(h5['dipole_cart'])
        e = np.asarray(h5['deltaE'])
    return d, e
from file_io import WFNReader

wfn = WFNReader('WFN.h5')
n_occ = int(wfn.nelec)
nspinor = int(wfn.nspinor)
V_cell = float(wfn.cell_volume)
nk_full = 9
B_cart = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)
Binv = np.linalg.inv(B_cart)

dipole_cart, dE = read_dipole_h5('../00_lorrax_cohsex/dipole.h5')
nk_d, nb_d = dE.shape[0], dE.shape[1]

nspin = 1
pref = 4.0 / (V_cell * nk_full * nspin * nspinor)

# Stern reference (from s_tensor_exact.py latest run) — crystal coords.
# Dropped in here to avoid rerunning; update if Stern changes.
S_stern_crys = np.array([
    [-1.079949e+00, -5.399738e-01, -3.428561e-10],
    [-5.399738e-01, -1.079952e+00, +7.249189e-11],
    [-3.428561e-10, +7.249189e-11, -2.319296e-02],
])
# Transform Stern crys → cart for apples-to-apples with SoS native output.
S_stern_cart = Binv @ S_stern_crys @ Binv.T


def sos_at_nb(nb_use: int) -> np.ndarray:
    """Compute S(0) cart from dipole with c-band cutoff at nb_use."""
    nb_use = min(nb_use, nb_d)
    c_idx = np.arange(n_occ, nb_use)
    v_idx = np.arange(0, n_occ)
    v_cvk = np.asarray(dipole_cart)[:, :, c_idx[:, None], v_idx[None, :]]
    dE_cv = np.asarray(dE)[:, c_idx[:, None], v_idx[None, :]]
    # Occupations: f_v - f_c = 1 for insulator at T=0
    # denom at ω=0: dE · (0 - dE²) = -dE³
    denom = -dE_cv ** 3
    W = 1.0 / denom   # = -1/dE³
    S = pref * np.einsum('ancv,ncv,bncv->ab',
                          np.conj(v_cvk), W, v_cvk, optimize=True)
    return S.real


print(f"n_occ = {n_occ}, nb_dipole = {nb_d}")
print(f"\nS-tensor xx/yy/zz (cartesian) vs number of bands used in SoS:")
print(f"  {'nb':>4}  {'nc':>4}   {'S_xx':>12}  {'S_yy':>12}  {'S_zz':>12}  {'Stern/SoS xx':>12}")

nb_list = [30, 40, 50, 60, 70, 76, 80]
S_xx_vs_nb = []
for nb in nb_list:
    if nb <= n_occ:
        continue
    S_sos = sos_at_nb(nb)
    nc = nb - n_occ
    ratio_xx = S_stern_cart[0, 0] / S_sos[0, 0] if abs(S_sos[0, 0]) > 1e-12 else float('nan')
    S_xx_vs_nb.append((nb, S_sos[0, 0]))
    print(f"  {nb:>4}  {nc:>4}   {S_sos[0,0]:+12.4e}  {S_sos[1,1]:+12.4e}  {S_sos[2,2]:+12.4e}  {ratio_xx:+12.4f}")

print(f"\nStern cart [0,0]:  {S_stern_cart[0,0]:+12.4e}")
print(f"Stern cart [1,1]:  {S_stern_cart[1,1]:+12.4e}")
print(f"Stern cart [2,2]:  {S_stern_cart[2,2]:+12.4e}")

# 1/nc extrapolation using the last 3 points of the sweep
if len(S_xx_vs_nb) >= 3:
    tail = S_xx_vs_nb[-3:]
    inv_nc = np.array([1.0 / (nb - n_occ) for nb, _ in tail])
    vals = np.array([v for _, v in tail])
    slope, intercept = np.polyfit(inv_nc, vals, 1)
    print(f"\n1/nc linear extrapolation of S_xx (last 3):  →  {intercept:+12.4e}")
    print(f"Stern/extrap ratio:                              {S_stern_cart[0,0]/intercept:+12.4f}")
