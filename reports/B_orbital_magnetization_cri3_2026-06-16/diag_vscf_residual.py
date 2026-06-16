"""V_scf acceptance gate: does the rebuilt KS H reproduce the WFN eigenvalues?
If <v|H|v> == eps_v (to ~meV) and off-diagonals ~0, V_scf is consistent with the
WFN and the Sternheimer covariant derivatives are exact."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.scf_potential import build_rho_val_from_wfn, build_dft_potentials
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]; RY2EV = 13.605693122994
trunc = ("--no-trunc" not in sys.argv)
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, sym, meta, pseudos, nspinor=int(wfn.nspinor))
rho = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=True)
V_scf, V_loc, _ = build_dft_potentials(wfn, pseudos, rho, truncation_2d=trunc, verbose=True)
ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in wfn.fft_grid)))
print(f"\ntruncation_2d={trunc}")
for ik in [0, 4]:
    kv = np.asarray(sym.unfolded_kpts[ik], float)
    k_red = int(sym.irr_idx_k[ik]); eps = np.asarray(wfn.energies[0, k_red, :nocc], float)
    box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
    H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
    Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U = _psi_box_to_G_sphere(box, Gk)[:nocc] * H_k.mask[None, None, :].astype(box.dtype)
    HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
                          H_k.vnl_Z, H_k.vnl_E, H_k.mask)
    norm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
    diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real) / norm
    resid = np.abs(diag - eps) * RY2EV * 1000   # meV
    off = np.asarray(jnp.einsum('vsG,wsG->vw', jnp.conj(U), HU))
    offmax = np.abs(off - np.diag(np.diag(off))).max()
    print(f"\nk={ik} kv={np.round(kv,3)}  <u|u> in [{norm.min():.4f},{norm.max():.4f}]")
    print(f"  max |<v|H|v> - eps_v| over {nocc} occ = {resid.max():.2f} meV "
          f"(mean {resid.mean():.2f})")
    print(f"  max |off-diagonal <v|H|w>| = {offmax:.3e} Ry")
    for v in [0, nocc//2, nocc-1]:
        print(f"    band {v:2d}: eps={eps[v]*RY2EV:+.4f} eV  <H>={diag[v]*RY2EV:+.4f} eV  "
              f"resid={resid[v]:.2f} meV")
