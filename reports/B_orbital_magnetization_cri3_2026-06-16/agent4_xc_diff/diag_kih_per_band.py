"""Is the LORRAX-vs-QE kih (Kinetic+Ionic+Hartree) difference truly BAND-INDEPENDENT?

If LORRAX <T+Vloc+VH+VNL> - QE kih.dat is a CONSTANT across bands, then ALL the
band-dependent residual is in V_xc (the magnetic field/assembly).  If instead it
VARIES on the worst bands (28/35/63), the charge/kinetic channel is the culprit
and V_xc is a red herring.

kih.dat = <psi| T + V_loc(ionic) + V_H |psi>  (NO V_xc, NO V_NL? -> check).
pw2bgw write_kih uses h_psi with vrs = vltot+vhartree (vxc subtracted), so kih
INCLUDES V_NL (h_psi applies the nonlocal projectors).  So compare against
LORRAX <T> + <Vloc> + <VH> + <VNL>.
"""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax.numpy as jnp
sys.path.insert(0, "/global/u2/j/jackm/software")
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import (setup_H_k_from_kvec, compute_ngkmax, compute_V_H_and_V_xc,
                               build_V_scf, build_G_cart)
from psp.ionic_gspace import build_ionic_and_core
from psp.scf_potential import build_rho_val_from_wfn, build_magnetization_from_wfn
from psp.run_sternheimer import _psi_box_to_G_sphere
from psp.pseudos import load_pseudopotentials
import psp.vnl_ops as vnl_ops

WFN = sys.argv[1]; RY2EV = 13.605693122994; trunc = True
KIHFILE = os.path.join(os.path.dirname(os.path.realpath(WFN)), "kih.dat")
VXCFILE = os.path.join(os.path.dirname(os.path.realpath(WFN)), "vxc.dat")

def read_dat_diag(path, kx, ky, kz):
    out = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        h = lines[i].split()
        if len(h) >= 5 and abs(float(h[0])-kx) < 1e-6 and abs(float(h[1])-ky) < 1e-6 and abs(float(h[2])-kz) < 1e-6:
            ndiag = int(h[3]); noff = int(h[4])
            for j in range(ndiag):
                p = lines[i+1+j].split()
                out[int(p[1])] = float(p[2])
            return out
        ndiag = int(h[3]); noff = int(h[4]); i += 1 + ndiag + noff
    raise RuntimeError("k not found")

wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
pseudos = load_pseudopotentials(os.path.dirname(os.path.realpath(WFN)))
vnl_setup = vnl_ops.build_vnl_setup(wfn, pseudos=pseudos, nspinor=int(wfn.nspinor),
                                    q_max=float(np.sqrt(float(wfn.ecutwfc)))*1.01)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
G_cart = build_G_cart(nx, ny, nz, float(wfn.blat)*np.asarray(wfn.bvec, float))
bdot = jnp.asarray(wfn.bdot); bvec = jnp.asarray(wfn.bvec)
nb = 180

rho_val = build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)
V_loc, rho_core, rho_core_G = build_ionic_and_core(wfn, pseudos, fg, truncation_2d=trunc)
V_H, _ = compute_V_H_and_V_xc(rho_val, rho_core, rho_core_G, G_cart, bdot, bvec, wfn.blat, truncation_2d=trunc)

ngkmax = int(compute_ngkmax(np.asarray(sym.unfolded_kpts, float), np.asarray(wfn.bdot),
                            float(wfn.ecutwfc), tuple(int(x) for x in fg)))
ik = 0
kv = np.asarray(sym.unfolded_kpts[ik], float)
qe_kih = read_dat_diag(KIHFILE, kv[0], kv[1], kv[2])
qe_vxc = read_dat_diag(VXCFILE, kv[0], kv[1], kv[2])
k_red = int(sym.irr_idx_k[ik])
eps = np.asarray(wfn.energies[0, k_red, :nb], float)*RY2EV

box = load_kpoint_fftbox(wfn, sym, meta, ik, nb)
psi_r = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
V_scf = build_V_scf(V_loc, V_H, None)  # T+Vloc+VH only (no V_xc)
H_k = setup_H_k_from_kvec(kv, V_scf, vnl_setup, wfn, meta, V_loc_r=V_loc, ngkmax=ngkmax)
Gk = jnp.stack([H_k.Gx, H_k.Gy, H_k.Gz], axis=-1).astype(jnp.int32)
U = _psi_box_to_G_sphere(box, Gk)[:nb]*H_k.mask[None, None, :].astype(box.dtype)
nrm = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), U).real)

def expect_localG(field_r):
    Vp = jnp.fft.fftn(psi_r*field_r, axes=(-3, -2, -1), norm='ortho')[:, :, H_k.Gx, H_k.Gy, H_k.Gz]*H_k.mask[None, None, :]
    return np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), Vp).real)/nrm*RY2EV
T = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), H_k.T_diag[None, None, :]*U).real)/nrm*RY2EV
Vl = expect_localG(V_loc); VH = expect_localG(V_H)
P = jnp.einsum('RG,vsG->Rsv', jnp.conj(H_k.vnl_Z), U, optimize=True)
D = jnp.einsum('stRQ,Qtv->Rsv', H_k.vnl_E, P, optimize=True)
HV = jnp.einsum('RG,Rsv->vsG', H_k.vnl_Z, D, optimize=True)*H_k.mask[None, None, :]
VN = np.asarray(jnp.einsum('vsG,vsG->v', jnp.conj(U), HV).real)/nrm*RY2EV

lorrax_kih = T + Vl + VH + VN  # eV

print(f"{'b':>4} {'eps':>9} {'LRX_kih':>10} {'QE_kih':>10} {'dkih(meV)':>10} {'QE_vxc':>9} {'eps-kih_QE':>10}")
dkih = []
watch = {0, 1, 28, 29, 35, 39, 40, 63, 69, nocc-1}
for b in range(1, nb+1):
    if b not in qe_kih:
        continue
    d = (lorrax_kih[b-1] - qe_kih[b])*1000
    dkih.append(d)
    if (b-1) in watch:
        print(f"{b-1:>4} {eps[b-1]:>9.3f} {lorrax_kih[b-1]:>10.4f} {qe_kih[b]:>10.4f} {d:>10.2f} "
              f"{qe_vxc.get(b, float('nan')):>9.4f} {eps[b-1]-qe_kih[b]:>10.4f}")
dkih = np.array(dkih)
print(f"\nLORRAX_kih - QE_kih: mean={dkih.mean():.2f} std={dkih.std():.2f} "
      f"min={dkih.min():.2f} max={dkih.max():.2f} meV  (over {len(dkih)} bands)")
print("If std<<mean -> kih is band-independent (+const); residual is all in V_xc.")
print("If std is large on the worst bands -> the charge/kinetic channel is the culprit.")
# correlation of dkih with the H-residual pattern: just print worst bands' dkih
