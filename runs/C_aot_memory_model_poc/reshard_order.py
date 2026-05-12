"""Test x-first vs y-first reshard orderings.

Starting: P(None, ('x','y'), None, None)  (input)
Ending:   P(None, None, None, 'y')        (output)

X-first:
    all_gather(x, axis=bands)        tile grows p_x×
    all_to_all(y, split=r, concat=b) tile stays the same
Y-first:
    all_to_all(y, split=r, concat=b) tile stays input-sized
    all_gather(x, axis=bands)        tile grows p_x× at the END

Both converge to the same output size, but peak transient may differ.
"""
import os
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/C_aot_memory_model_poc/xla_dump_order"
os.makedirs(OUT, exist_ok=True)
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={OUT} --xla_dump_hlo_as_text "
    "--xla_dump_hlo_module_re=.*reshard.*"
)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    jax.distributed.initialize(local_device_ids=[0])
except Exception:
    pass

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y"))

NK, NB, NS, BR = 64, 296, 2, 3456
sh_in = NamedSharding(mesh, P(None, ("x", "y"), None, None))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in)


def _peak(compiled):
    m = compiled.memory_analysis()
    return (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9, m


@jax.jit
def reshard_xfirst(psi):
    """all_gather(x) first, then all_to_all(y)."""
    def _local(x):
        x = jax.lax.all_gather(x, "x", axis=1, tiled=True)
        x = jax.lax.all_to_all(x, "y", split_axis=3, concat_axis=1, tiled=True)
        return x
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, ("x", "y"), None, None),
        out_specs=P(None, None, None, "y"),
        check_rep=False,
    )(psi)


@jax.jit
def reshard_yfirst(psi):
    """all_to_all(y) first, then all_gather(x)."""
    def _local(x):
        # Input tile: (nk, nb/(px*py), ns, n_rchunk)
        # all_to_all(y, split=rchunk, concat=bands, tiled=True):
        #   each y-rank splits rchunk by py and sends to other y-ranks
        #   gathers bands back.
        x = jax.lax.all_to_all(x, "y", split_axis=3, concat_axis=1, tiled=True)
        # Now: tile (nk, nb/px, ns, n_rchunk/py).  Sharding P(None,'x',None,'y').
        # all_gather(x, axis=bands):
        x = jax.lax.all_gather(x, "x", axis=1, tiled=True)
        # Now: tile (nk, nb, ns, n_rchunk/py).  Sharding P(None,None,None,'y').
        return x
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, ("x", "y"), None, None),
        out_specs=P(None, None, None, "y"),
        check_rep=False,
    )(psi)


for name, fn in [("xfirst", reshard_xfirst), ("yfirst", reshard_yfirst)]:
    print(f"\n==== {name} ====", flush=True)
    lowered = fn.lower(spec)
    compiled = lowered.compile()
    gb, m = _peak(compiled)
    print(f"  peak = {gb:.3f} GB")
    print(f"  temp={m.temp_size_in_bytes/1e9:.3f} arg={m.argument_size_in_bytes/1e9:.3f} "
          f"out={m.output_size_in_bytes/1e9:.3f}")
