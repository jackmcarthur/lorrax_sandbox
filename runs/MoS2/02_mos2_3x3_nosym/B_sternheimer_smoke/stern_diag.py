"""Probe U_val_kminq orthonormality + build-time b projector leak."""
import os, sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env
set_default_env()

import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader

wfn = WFNReader('WFN.h5')
sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)

for iq in [0, 1]:
    qvec = wfn.kpoints[iq]
    print(f'\n=== iq={iq} q={qvec} ===')
    for ik_full in [0, 1]:
        ik_kminq = int(sym.kq_map[ik_full, iq])
        kvec_kminq = sym.unfolded_kpts[ik_kminq]

        psi_kminq_box = load_kpoint_fftbox(wfn, sym, meta, ik_kminq, n_occ)
        Gkminq = np.asarray(sym.get_gvecs_kfull(wfn, ik_kminq), dtype=np.int32)

        # gather at Gkminq
        nx, ny, nz = psi_kminq_box.shape[-3:]
        ix = np.mod(Gkminq[:,0], nx); iy = np.mod(Gkminq[:,1], ny); iz = np.mod(Gkminq[:,2], nz)
        U = np.asarray(psi_kminq_box)[..., ix, iy, iz]   # (n_occ, 2, ngk)
        # orthonormality
        UU = np.einsum('msG,nsG->mn', np.conj(U), U)
        err = float(np.max(np.abs(UU - np.eye(n_occ))))
        diag_err = float(np.max(np.abs(np.diag(UU).real - 1.0)))
        print(f'  ik_full={ik_full} ik_kminq={ik_kminq} k-q={kvec_kminq}')
        print(f'    ngk_p={Gkminq.shape[0]}  ||U U^dag - I||_inf = {err:.3e}  (diag err {diag_err:.3e})')

        # also check ik_full's own psi
        psi_k_box = load_kpoint_fftbox(wfn, sym, meta, ik_full, n_occ)
        Gk = np.asarray(sym.get_gvecs_kfull(wfn, ik_full), dtype=np.int32)
        ix = np.mod(Gk[:,0], nx); iy = np.mod(Gk[:,1], ny); iz = np.mod(Gk[:,2], nz)
        Uk = np.asarray(psi_k_box)[..., ix, iy, iz]
        UUk = np.einsum('msG,nsG->mn', np.conj(Uk), Uk)
        errk = float(np.max(np.abs(UUk - np.eye(n_occ))))
        print(f'    ngk_k={Gk.shape[0]}  ||U_k U_k^dag - I||_inf = {errk:.3e}')
