"""Decisive split: does LORRAX's WFN-reconstructed rho_val match QE's
self-consistent SCF charge density?  If they differ (esp. near the Cr nucleus)
-> the bug is the WFN->rho_val reconstruction.  If they match -> the bug is
downstream (V_xc functional / V_NL), not the density."""
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, h5py, jax.numpy as jnp
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_B/src")
from file_io import WfnLoader
from common import symmetry_maps, Meta
from psp.scf_potential import build_rho_val_from_wfn

WFN = sys.argv[1]; QESAVE = sys.argv[2]
wfn = WfnLoader(WFN); sym = symmetry_maps.SymMaps(wfn); nocc = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nocc, 0, nocc, 0, False)
fg = wfn.fft_grid; nx, ny, nz = int(fg[0]), int(fg[1]), int(fg[2])
vol = float(wfn.cell_volume)
print(f"FFT grid {nx}x{ny}x{nz}  vol={vol:.3f} bohr^3  nocc={nocc}")

# --- LORRAX reconstructed valence density (real-space, e/bohr^3) ---
rho_L = np.asarray(build_rho_val_from_wfn(wfn, sym, meta, nocc, verbose=False)).real
print(f"LORRAX rho_val: integral={rho_L.sum()*vol/(nx*ny*nz):.4f} e  "
      f"max={rho_L.max():.3f}  min={rho_L.min():.4f}")

# --- QE SCF charge density (G-space -> real, e/bohr^3) ---
f = h5py.File(os.path.join(QESAVE, "charge-density.hdf5"), "r")
mill = np.asarray(f["MillerIndices"])                 # (ngm,3)
rg = np.asarray(f["rhotot_g"]).view(np.complex128)    # (ngm,)
grid = np.zeros((nx, ny, nz), dtype=np.complex128)
ix = mill[:, 0] % nx; iy = mill[:, 1] % ny; iz = mill[:, 2] % nz
grid[ix, iy, iz] = rg
rho_Q = np.fft.ifftn(grid).real * (nx * ny * nz)      # rho(r)=sum_G rhog e^{iGr}
print(f"QE     rho_tot: integral={rho_Q.sum()*vol/(nx*ny*nz):.4f} e  "
      f"max={rho_Q.max():.3f}  min={rho_Q.min():.4f}")

# --- compare ---
d = rho_L - rho_Q
print(f"\nrho_L - rho_Q:  max|diff|={np.abs(d).max():.4f}  "
      f"L2={np.sqrt((d**2).mean()):.5f}  int|diff|={np.abs(d).sum()*vol/(nx*ny*nz):.4f} e")
# where is the biggest difference? (near nucleus => core-region reconstruction)
fi = np.unravel_index(np.argmax(np.abs(d)), d.shape)
print(f"max-diff voxel {fi}: rho_L={rho_L[fi]:.3f} rho_Q={rho_Q[fi]:.3f}  "
      f"(at the density peak rho_Q={rho_Q.max():.3f})")
# ratio at the density peak (nucleus)
pk = np.unravel_index(np.argmax(rho_Q), rho_Q.shape)
print(f"at QE density peak {pk}: rho_Q={rho_Q[pk]:.3f}  rho_L={rho_L[pk]:.3f}  "
      f"ratio L/Q={rho_L[pk]/rho_Q[pk]:.4f}")
