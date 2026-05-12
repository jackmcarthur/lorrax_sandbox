"""Minimal smoke-test: does shard_map + io_callback produce a sharded
jax.Array where each device's shard is a distinct value from its own
host-side store?  If yes the GspaceProvider mechanism works."""
from __future__ import annotations
import os, sys
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_C/src")

from gw.gw_jax import _maybe_init_jax_distributed
_maybe_init_jax_distributed()

import functools
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import io_callback
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devs = jax.devices()
assert len(devs) >= 4
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ('x', 'y'))

# Per-(x,y) host data — each device gets a distinctive numpy block
# whose values encode (x, y) so we can verify correct placement.
per_device_shape = (3, 2)  # (nk_small, bpd_small)
store = {}
for x in range(2):
    for y in range(2):
        a = np.zeros(per_device_shape, dtype=np.complex128)
        a.fill(100 * x + y)  # e.g. x=1,y=0 → all entries = 100
        store[(x, y)] = a

def _cb(x_idx, y_idx):
    return store[(int(x_idx), int(y_idx))]

out_sds = jax.ShapeDtypeStruct(per_device_shape, jnp.complex128)
spec = P(('x', 'y'), None)  # shard axis-0 flat over x,y

@jax.jit
def pull():
    @functools.partial(shard_map, mesh=mesh,
                       in_specs=(), out_specs=spec, check_rep=False)
    def _sm():
        x = jax.lax.axis_index('x')
        y = jax.lax.axis_index('y')
        return io_callback(_cb, out_sds, x, y, ordered=True)
    return _sm()


arr = pull()
arr.block_until_ready()
if jax.process_index() == 0:
    print(f"global shape={arr.shape}, sharding={arr.sharding}")
for shard in arr.addressable_shards:
    print(f"[rank {jax.process_index()}] dev={shard.device} "
          f"index={shard.index} values={np.asarray(shard.data).real.ravel()}")
