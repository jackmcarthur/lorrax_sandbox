import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp
_SRC = "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src"
sys.path.insert(0, _SRC)
from file_io import WfnLoader
from common import symmetry_maps, Meta
from common.load_wfns import load_kpoint_fftbox
from psp.dft_operators import generate_gvectors_k, gather_psi_G_from_crys

WFN = sys.argv[1]
wfn = WfnLoader(WFN)
sym = symmetry_maps.SymMaps(wfn)
nb = 80
meta = Meta.from_system(wfn, sym, int(wfn.nelec), nb - int(wfn.nelec), nb, 0, False)
print("nspinor", int(wfn.nspinor), "nbnd_file", int(wfn.nbands), "nk_tot", int(sym.nk_tot),
      "fft_grid", tuple(int(x) for x in meta.fft_grid))

ik = 0
box = load_kpoint_fftbox(wfn, sym, meta, ik, nb)        # (nb, ns, nx,ny,nz)
print("box shape", box.shape, "box abs-sum", float(jnp.abs(box).sum()))
Gk, kp = generate_gvectors_k(ik, sym, wfn, meta)
Gk = np.asarray(Gk)
print("Gk shape", Gk.shape, "Gk min/max", Gk.min(0), Gk.max(0))
bi = np.asarray(wfn.box_index(k=[ik]))                  # where to_box PLACED coeffs
print("box_index shape", bi.shape, "first 3 rows:", bi.reshape(-1, bi.shape[-1])[:3] if bi.ndim>1 else bi[:3])
print("Gk first 3 rows:", Gk[:3])

psi = np.asarray(gather_psi_G_from_crys(box, Gk))       # (nb, ns, nG)
print("psi shape", psi.shape)
nrm = (np.abs(psi)**2).sum(axis=(1, 2))                 # per band
print("per-band norm[:6]", np.round(nrm[:6], 4), " mean", float(nrm.mean()))

# spin expectation per band (all 3 components) for ns>=2
if psi.shape[1] >= 2:
    up, dn = psi[:, 0], psi[:, 1]
    sz = (np.abs(up)**2 - np.abs(dn)**2).sum(1)
    sx = 2*np.real((np.conj(up)*dn).sum(1))
    sy = 2*np.imag((np.conj(up)*dn).sum(1))
    nocc = int(wfn.nelec)
    print(f"occ-summed <sx>={sx[:nocc].sum():+.4f} <sy>={sy[:nocc].sum():+.4f} <sz>={sz[:nocc].sum():+.4f}")
    print("per-band sz[:6]", np.round(sz[:6], 3))
