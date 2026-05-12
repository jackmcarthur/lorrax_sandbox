"""Confirm donation ACTUALLY aliases at each of the three fixed sites."""
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

def _alias(fn, *specs):
    m = fn.lower(*specs).compile().memory_analysis()
    return (m.alias_size_in_bytes / 1e9,
            (m.temp_size_in_bytes + m.argument_size_in_bytes
             + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9)


# SITE 1 — x-first + donate
NK, NB, NS, BR = 64, 296, 2, 3456
sh_in1 = NamedSharding(mesh, P(None, ("x", "y"), None, None))
stage1 = NamedSharding(mesh, P(None, "y", None, None))
final1 = NamedSharding(mesh, P(None, None, None, "y"))
spec1 = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in1)

@partial(jax.jit, donate_argnums=(0,))
def s1(x):
    x = jax.lax.with_sharding_constraint(x, stage1)
    return jax.lax.with_sharding_constraint(x, final1)

alias, peak = _alias(s1, spec1)
print(f"Site 1 (_reshard_rchunk x-first+donate):  peak {peak:.2f}  alias {alias:.3f}  "
      f"{'DONATION WORKS' if alias > 0 else '(donation no-op: shardings differ)'}")

# SITE 3 — direct + donate
NQ, MU, BR3 = 64, 2400, 12672
sh_in3 = NamedSharding(mesh, P(None, "x", "y"))
out3 = NamedSharding(mesh, P(None, None, ("x", "y")))
spec3 = jax.ShapeDtypeStruct((NQ, MU, BR3), jnp.complex128, sharding=sh_in3)

@partial(jax.jit, donate_argnums=(0,))
def s3(z):
    return jax.lax.with_sharding_constraint(z, out3)

alias, peak = _alias(s3, spec3)
print(f"Site 3 (_reshard_z + donate):             peak {peak:.2f}  alias {alias:.3f}  "
      f"{'DONATION WORKS' if alias > 0 else '(donation no-op)'}")
