"""Probe 2: is the zero from the sharded loader.load or from gflat_to_rmu?"""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.sharding import PartitionSpec as P
from bandstructure.htransform import setup_wfn_and_sym
from common.meta import Meta
from common.wfn_transforms import gflat_to_rmu
from file_io.centroids import load_centroids
from bse.bse_w_exact import _create_mesh_xy


def nrm(x):
    return float(jnp.sqrt(jnp.sum(jnp.abs(x) ** 2)))


mesh = _create_mesh_xy(1, 1)
wfn, sym = setup_wfn_and_sym("WFN.h5")
_, cidx, n_rmu = load_centroids("centroids_frac_640.txt",
                                tuple(int(x) for x in wfn.fft_grid))
cidx_np = np.asarray(cidx, dtype=np.int32)
sharding_load = P(None, ('x', 'y'), None, None)
g_index_full = wfn.box_index_dev(k="full_bz", mesh=mesh)
sym_loader = wfn._ensure_sym()
kg = np.asarray((12, 12, 1), dtype=np.float64)
kvecs = np.asarray(sym_loader.kvecs_asints, dtype=np.float64) / kg[None, :]

for br in [(24, 32), (0, 8), (0, 80)]:
    ps = wfn.load(bands=br, k="full_bz", sharding=sharding_load,
                  bispinor=False)
    out = gflat_to_rmu(ps, g_index_full, cidx_np, mesh=mesh,
                       fft_grid=wfn.fft_grid,
                       kvecs_frac=jnp.asarray(kvecs), norm="ortho",
                       chunk_size=64)
    print(f"bands {br}: sharded-load norm {nrm(ps):.6f} "
          f"-> gflat_to_rmu norm {nrm(out):.6f}", flush=True)
    if br == (24, 32):
        pb = np.sqrt((np.abs(np.asarray(jax.device_get(ps))) ** 2)
                     .sum(axis=(0, 2, 3)))
        print("  per-band G-flat norms:", np.array2string(pb, precision=3))
