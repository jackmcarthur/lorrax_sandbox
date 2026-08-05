"""Probe: where do zeros enter the narrow-window (24,32) htransform load?"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input
from bandstructure.htransform import setup_wfn_and_sym
from common.meta import Meta
from common.wfn_transforms import load_centroids_band_chunked
from file_io.centroids import load_centroids
from bse.bse_w_exact import _create_mesh_xy

mesh = _create_mesh_xy(1, 1)
params = read_lorrax_input("exciton_bands_nw.in")
wfn, sym = setup_wfn_and_sym("WFN.h5")
print("nelec", wfn.nelec, "file nbands", wfn.nbands)
_, cidx, n_rmu = load_centroids("centroids_frac_640.txt",
                                tuple(int(x) for x in wfn.fft_grid))
for (nval, ncond, nband) in [(2, 6, 8), (26, 54, 80)]:
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, False)
    br = (int(wfn.nelec) - nval, int(wfn.nelec) + ncond)
    print(f"window {br} meta nband={getattr(meta,'nband',None)}")
    raw = wfn.load(bands=br, k="full_bz", bispinor=False)
    print("  raw loader norm:", float(jnp.linalg.norm(raw)), raw.shape)
    psiY, _ = load_centroids_band_chunked(wfn, sym, meta, cidx, False, mesh,
                                          band_range=br)
    print("  centroid psi norm:", float(jnp.linalg.norm(psiY)), psiY.shape)
    if nval == 2:
        # per-band norms to see which slots are dead
        pb = np.linalg.norm(np.asarray(jax.device_get(psiY)), axis=(0, 2, 3))
        print("  per-band centroid norms:", np.array2string(pb, precision=3))
