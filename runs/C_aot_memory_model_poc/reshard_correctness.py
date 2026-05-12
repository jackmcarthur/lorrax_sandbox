"""Check OUTPUT sharding of each reshard variant, not just peak."""
import os
os.environ.setdefault("XLA_FLAGS", "")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    jax.distributed.initialize(local_device_ids=[0])
except Exception:
    pass

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y"))

NK, NB, NS, BR = 8, 16, 2, 256   # small for fast execution
sh_in = NamedSharding(mesh, P(None, ("x", "y"), None, None))
sh_final = NamedSharding(mesh, P(None, None, None, "y"))
sh_stage_X_rchunk_Y = NamedSharding(mesh, P(None, "x", None, "y"))
sh_stage_Y = NamedSharding(mesh, P(None, "y", None, None))


def make_input():
    x = jnp.arange(NK * NB * NS * BR, dtype=jnp.complex128).reshape(NK, NB, NS, BR)
    return jax.device_put(x, sh_in)


# --- Variants ---
@jax.jit
def xfirst_nodonate(x):
    x = jax.lax.with_sharding_constraint(x, sh_stage_Y)
    x = jax.lax.with_sharding_constraint(x, sh_final)
    return x

@partial(jax.jit, donate_argnums=(0,))
def xfirst_donate(x):
    x = jax.lax.with_sharding_constraint(x, sh_stage_Y)
    x = jax.lax.with_sharding_constraint(x, sh_final)
    return x

@jax.jit
def yfirst_nodonate(x):
    x = jax.lax.with_sharding_constraint(x, sh_stage_X_rchunk_Y)
    x = jax.lax.with_sharding_constraint(x, sh_final)
    return x

@partial(jax.jit, donate_argnums=(0,))
def yfirst_donate(x):
    x = jax.lax.with_sharding_constraint(x, sh_stage_X_rchunk_Y)
    x = jax.lax.with_sharding_constraint(x, sh_final)
    return x


for name, fn in [("xfirst_nodonate", xfirst_nodonate),
                 ("xfirst_donate", xfirst_donate),
                 ("yfirst_nodonate", yfirst_nodonate),
                 ("yfirst_donate", yfirst_donate)]:
    x = make_input()
    result = fn(x)
    spec = result.sharding.spec
    match = "OK" if spec == P(None, None, None, "y") else "MISMATCH"
    print(f"  {name:20s} output spec = {spec}  [{match}]")
