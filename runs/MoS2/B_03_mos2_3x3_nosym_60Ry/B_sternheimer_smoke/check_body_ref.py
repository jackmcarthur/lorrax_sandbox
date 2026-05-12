"""Verify the body-diagonal reference is converged at tighter tol."""
import sys, numpy as np, h5py
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env; set_default_env()
import jax.numpy as jnp
from psp.run_sternheimer import run_sternheimer
from file_io import WFNReader

wfn = WFNReader('WFN.h5')
qsigned = np.asarray(wfn.kpoints[4]) - np.round(np.asarray(wfn.kpoints[4]))
nx, ny, nz = (int(v) for v in wfn.fft_grid)
N = nx * ny * nz
G_pert = np.array([-1, 1, 6], dtype=int)   # what the body sweep at NG_OUT=256 picked
c_G_box = np.zeros((nx, ny, nz), dtype=np.complex128)
c_G_box[G_pert[0] % nx, G_pert[1] % ny, G_pert[2] % nz] = 1.0
V_pert_box = np.sqrt(N) * np.fft.ifftn(c_G_box, axes=(0, 1, 2), norm='ortho')

print("Body diagonal at G_pert=(0,-2,-4), |q+G_pert|² ≈ 4.73:")
for n_cond, tol, max_iter in [
    ( 20, 1e-6,  80),
    ( 20, 1e-10, 400),
    (150, 1e-6,  80),
    (150, 1e-10, 400),
    (  0, 1e-6,  80),       # no Schur warm-start at all
]:
    out = f'/tmp/body_ref_n{n_cond}_t{tol:.0e}_i{max_iter}.h5'
    run_sternheimer(
        wfn_path='WFN.h5', pseudo_dir='.',
        n_cond_bands=n_cond, iq_list=[4], ng_out=256,
        tol=tol, max_iter=max_iter, truncation_2d=True,
        output_path=out, with_derivatives=False, with_s_tensor=False,
        sos_only=False, V_pert_box=jnp.asarray(V_pert_box),
        verbose=False)
    with h5py.File(out, 'r') as f:
        chi = np.asarray(f['q_0/chi_col']); Gout = np.asarray(f['q_0/G_int'])
        for i, g in enumerate(Gout):
            if (g % np.array([nx, ny, nz]) == G_pert % np.array([nx, ny, nz])).all():
                print(f"  n_cond={n_cond:>3d}  tol={tol:.0e}  max_iter={max_iter:>3d}  "
                      f"chi_body = {chi[i]:+.6e}   |chi| = {abs(chi[i]):.4e}")
                break
