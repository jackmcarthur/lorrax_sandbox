"""Decompose the 60 meV against QE's EXACT matrix elements (pw2bgw):
  vxc.dat = <nk|V_xc+B|nk>   kih.dat = <nk|T+V_ion+V_H|nk>   (eV)
LORRAX:  <V_xc+B> = <H_full> - <H_noxc>;  <T+Vloc+VH+VNL> = <H_noxc>.
Whichever differs from QE is where the 60 meV lives."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, apply_H_k_from_G, compute_ngkmax,
                               compute_V_H_and_V_xc, build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.xc import compute_V_xc_spin
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops
RY2EV = 13.605693122994; trunc = True

def parse_bgw(path):
    """-> dict {(kx,ky,kz) rounded: np.array(ndiag) complex eV}."""
    out = {}; toks = open(path).read().split("\n"); i = 0
    while i < len(toks):
        t = toks[i].split()
        if len(t) == 5:
            k = tuple(round(float(x), 5) for x in t[:3]); nd = int(t[3])
            vals = np.zeros(nd, complex)
            for j in range(nd):
                d = toks[i+1+j].split()
                vals[int(d[1])-1] = float(d[2]) + 1j*float(d[3])   # col0=spin, col1=band
            out[k] = vals; i += 1+nd
        else:
            i += 1
    return out

WFN = sys.argv[1]; ND = os.path.dirname(os.path.realpath(WFN))
vxc_qe = parse_bgw(f"{ND}/vxc.dat"); kih_qe = parse_bgw(f"{ND}/kih.dat")
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(ND)
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc))) * 1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
B = float(wfn.blat) * np.asarray(wfn.bvec, float)
G_cart = build_G_cart(nx, ny, nz, B); bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)
rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
mxx, myy, mzz = build_magnetization_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec, wfn.blat, truncation_2d=trunc)
n = rho_val + rho_core; amag = jnp.sqrt(mxx**2+myy**2+mzz**2)
V_up, V_dn = compute_V_xc_spin((n+amag)/2, (n-amag)/2, jnp.fft.fftn((n+amag)/2), jnp.fft.fftn((n-amag)/2), G_cart)
Vbar = 0.5*(V_up+V_dn); Bmag = 0.5*(V_up-V_dn)
inv = jnp.where(amag > 1e-12, 1.0/(amag+1e-30), 0.0)
B_vec = jnp.stack([Bmag*mxx*inv, Bmag*myy*inv, Bmag*mzz*inv], axis=0)
V_full = build_V_scf(V_loc, V_H, Vbar); V_noxc = build_V_scf(V_loc, V_H)
ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))

def diag_of(V_scf, Bv, ik):
    kv = np.asarray(sym.unfolded_kpts[ik], float)
    box = load_kpoint_fftbox(wfn, sym, meta, ik, nocc)
    H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
    Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
    U = _psi_box_to_G_sphere(box, Gk)[:nocc] * H_k.mask[None, None, :].astype(box.dtype)
    HU = apply_H_k_from_G(U, H_k.T_diag, H_k.V_scf, H_k.Gx, H_k.Gy, H_k.Gz, H_k.vnl_Z, H_k.vnl_E, H_k.mask, Bv)
    nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)
    return np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HU).real)/nrm * RY2EV  # eV

for ik in (0, 4):
    k = tuple(round(float(x), 5) for x in sym.unfolded_kpts[ik])
    if k not in vxc_qe:
        k = min(vxc_qe, key=lambda q: sum((a-b)**2 for a,b in zip(q,k)))
    vxc_q = vxc_qe[k][:nocc].real; kih_q = kih_qe[k][:nocc].real
    kr = int(sym.irr_idx_k[ik]); eps_eV = np.asarray(wfn.energies[0, kr, :nocc], float)*RY2EV
    full = diag_of(V_full, B_vec, ik); noxc = diag_of(V_noxc, None, ik)
    vxc_lx = full - noxc            # <V_xc+B>
    kih_lx = noxc                   # <T+Vloc+VH+VNL>
    dvxc = (vxc_lx - vxc_q)*1000; dkih = (kih_lx - kih_q)*1000   # meV
    print(f"k={ik} {k}:")
    print(f"   CONV chk band0: eps={eps_eV[0]:.4f}  kih_q+vxc_q={kih_q[0]+vxc_q[0]:.4f}  "
          f"full_LX={full[0]:.4f}  (full-eps={1000*(full[0]-eps_eV[0]):.1f} meV)")
    print(f"   raw band0: vxc_q={vxc_q[0]:.4f} vxc_lx={vxc_lx[0]:.4f} | kih_q={kih_q[0]:.4f} kih_lx={kih_lx[0]:.4f}")
    print(f"   <V_xc+B>  : max|LX-QE| = {np.abs(dvxc).max():7.1f} meV  rms {np.sqrt((dvxc**2).mean()):6.1f}  (band0 {dvxc[0]:+.1f}, VBM {dvxc[nocc-1]:+.1f})")
    print(f"   <T+Vl+VH+VNL>: max|LX-QE| = {np.abs(dkih).max():7.1f} meV  rms {np.sqrt((dkih**2).mean()):6.1f}  (band0 {dkih[0]:+.1f}, VBM {dkih[nocc-1]:+.1f})")
    print(f"     dkih distribution: mean {dkih.mean():+.1f}  std {dkih.std():.1f}  min {dkih.min():+.1f}  max {dkih.max():+.1f} meV  "
          f"-> {'CONSTANT (V(G=0)/alpha-Z)' if dkih.std() < 0.25*abs(dkih.mean()) else 'BAND-DEPENDENT (V_NL/shape)'}")
    # group by whether bands are deep-semicore (large |kih|) to see if shape tracks V_NL
    order = np.argsort(kih_q);
    print(f"     deepest-8 bands dkih: {np.array2string(dkih[order][:8], precision=0, floatmode='fixed')}")
    print(f"     shallowest-8 (near VBM) dkih: {np.array2string(dkih[order][-8:], precision=0, floatmode='fixed')}")
