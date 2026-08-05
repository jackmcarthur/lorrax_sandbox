"""Probe 4: load_centroids_band_chunked first-call vs after a raw load
(cache pollution test), with the narrow-window Meta."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from bandstructure.htransform import setup_wfn_and_sym
from common.meta import Meta
from common.wfn_transforms import load_centroids_band_chunked
from file_io.centroids import load_centroids
from bse.bse_w_exact import _create_mesh_xy


def nrm(x):
    return float(jnp.sqrt(jnp.sum(jnp.abs(x) ** 2)))


mesh = _create_mesh_xy(1, 1)
wfn, sym = setup_wfn_and_sym("WFN.h5")
_, cidx, n_rmu = load_centroids("centroids_frac_640.txt",
                                tuple(int(x) for x in wfn.fft_grid))
meta = Meta.from_system(wfn, sym, 2, 6, 8, n_rmu, False)
br = (24, 32)
psiY, _ = load_centroids_band_chunked(wfn, sym, meta, cidx, False, mesh,
                                      band_range=br)
print("FIRST-call centroid psi norm:", nrm(psiY), flush=True)
_ = wfn.load(bands=br, k="full_bz", bispinor=False)
psiY2, _ = load_centroids_band_chunked(wfn, sym, meta, cidx, False, mesh,
                                       band_range=br)
print("after-raw-load centroid psi norm:", nrm(psiY2), flush=True)
