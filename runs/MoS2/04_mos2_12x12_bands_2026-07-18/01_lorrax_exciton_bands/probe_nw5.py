"""Probe 5: cross (meta window) x (band_range) to find which input zeroes
load_centroids_band_chunked."""
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
meta_nw = Meta.from_system(wfn, sym, 2, 6, 8, n_rmu, False)
meta_80 = Meta.from_system(wfn, sym, 26, 54, 80, n_rmu, False)
for tag, meta, br in [("meta80/br80", meta_80, (0, 80)),
                      ("meta80/br_nw", meta_80, (24, 32)),
                      ("meta_nw/br_nw", meta_nw, (24, 32)),
                      ("meta_nw/br08", meta_nw, (0, 8))]:
    psiY, _ = load_centroids_band_chunked(wfn, sym, meta, cidx, False, mesh,
                                          band_range=br)
    print(f"{tag}: centroid psi norm {nrm(psiY):.6f}", flush=True)
