"""Re-verify y-first reshard at production scale — is the output actually wrong,
or was my small-size test misleading?"""
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

# Production scale from MoS2 3x3 (what the smoke runs)
NK, NB, NS, BR = 9, 80, 2, 46080
sh_in = NamedSharding(mesh, P(None, ("x", "y"), None, None))
stage_xfirst = NamedSharding(mesh, P(None, "y", None, None))
stage_yfirst = NamedSharding(mesh, P(None, "x", None, "y"))
sh_final = NamedSharding(mesh, P(None, None, None, "y"))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in)

x = jax.device_put(jnp.ones((NK, NB, NS, BR), dtype=jnp.complex128), sh_in)


@jax.jit
def xfirst(x):
    x = jax.lax.with_sharding_constraint(x, stage_xfirst)
    return jax.lax.with_sharding_constraint(x, sh_final)

@jax.jit
def yfirst(x):
    x = jax.lax.with_sharding_constraint(x, stage_yfirst)
    return jax.lax.with_sharding_constraint(x, sh_final)


def measure(fn, x):
    # AOT
    c = fn.lower(spec).compile()
    m = c.memory_analysis()
    peak = (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9
    # Run
    y = fn(x).block_until_ready()
    return peak, y.sharding.spec, y


# Reference (x-first) produces correct output
p_x, spec_x, y_x = measure(xfirst, x)
print(f"x-first: peak {p_x:.3f} GB  output spec {spec_x}")

p_y, spec_y, y_y = measure(yfirst, x)
print(f"y-first: peak {p_y:.3f} GB  output spec {spec_y}")

# Are the VALUES the same?
diff = float(jnp.max(jnp.abs(y_x - y_y)))
print(f"max|y_x - y_y| = {diff}")

# Would downstream code see the same sharding?
print(f"specs equal? {spec_x == spec_y}")
