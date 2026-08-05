"""Probe 3: replicate load_centroids_band_chunked internals stepwise for
band_range=(24,32); vary chunk_size incl. the memory-derived one."""
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from jax.sharding import PartitionSpec as P
from bandstructure.htransform import setup_wfn_and_sym
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
g_index_full = wfn.box_index_dev(k="full_bz", mesh=mesh)
sym_loader = wfn._ensure_sym()
kvecs = np.asarray(sym_loader.kvecs_asints, dtype=np.float64) / \
    np.array([12., 12., 1.])[None, :]
n_rtot = int(np.prod(np.asarray(wfn.fft_grid)))
cs_budget = max(1, int(36e9 // (2 * n_rtot * 16 * 4)))
print("cs_budget =", cs_budget)
ps = wfn.load(bands=(24, 32), k="full_bz",
              sharding=P(None, ('x', 'y'), None, None), bispinor=False)
ref = None
for cs in [64, 1152, cs_budget]:
    out = gflat_to_rmu(ps, g_index_full, cidx_np, mesh=mesh,
                       fft_grid=wfn.fft_grid,
                       kvecs_frac=jnp.asarray(kvecs), norm="ortho",
                       chunk_size=cs)
    o = np.asarray(jax.device_get(out))
    print(f"cs={cs}: norm {nrm(out):.6f}", flush=True)
    if ref is None:
        ref = o
    else:
        print("   max|diff| vs cs64:", float(np.max(np.abs(o - ref))))
