"""End-to-end check of the WIRED B_xc path:
  build_dft_potentials(m_vec=...) -> B_vec -> setup_H_k_from_kvec(B_vec=) -> apply.
Confirms the production API reproduces the manual gate (~60 meV on CrI3), and
that m_vec=None gives B_vec=None (non-magnetic bit-identical)."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax
from psp.scf_potential import (build_dft_potentials, build_rho_val_from_wfn,
                               build_magnetization_from_wfn)
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
RY2EV = 13.605693122994

WFN = sys.argv[1]; magnetic = (sys.argv[2] == "mag") if len(sys.argv) > 2 else True
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
fg = wfn.fft_grid
rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
m_vec = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False) if magnetic else None

V_scf, V_loc, vnl_setup, B_vec = build_dft_potentials(
    wfn, pseudos, rho_val, truncation_2d=True, m_vec=m_vec, verbose=False)
print(f"m_vec={'set' if magnetic else 'None'} -> B_vec is "
      f"{'None' if B_vec is None else f'set (max {float(jnp.abs(B_vec).max())*RY2EV:.2f} eV)'}")

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))
for ik in (0, 4):
    kv = np.asarray(sym.unfolded_kpts[ik], float); k_red = int(sym.irr_idx_k[ik])
    eps = np.asarray(wfn.energies[0, k_red, :nocc], float)
    box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
    H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc,
                              ngkmax=ngkmax, B_vec=B_vec)        # ← wired B_vec
    Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U = _psi_box_to_G_sphere(box, Gk)[:nocc] * H_k.mask[None, None, :].astype(box.dtype)
    HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz,
                          H_k.vnl_Z, H_k.vnl_E, H_k.mask, H_k.B_vec)
    nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
    diag = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real)/nrm
    sr = (diag-eps)*RY2EV*1000   # signed
    r = np.abs(sr)
    if ik == 0:
        wb = np.argsort(r)[-6:][::-1]
        print(f"  k={ik}: max={r.max():.1f} mean={r.mean():.1f} median={np.median(r):.1f} band0={r[0]:.1f} VBM={r[nocc-1]:.1f}")
        print(f"     signed: mean={sr.mean():+.1f} std={sr.std():.1f}  worst bands (idx:meV): "
              + " ".join(f"{int(b)}:{sr[int(b)]:+.0f}" for b in wb))
    else:
        print(f"  k={ik}: max={r.max():.1f} mean={r.mean():.1f}  band0={r[0]:.1f} VBM={r[nocc-1]:.1f}")
