"""Verify site 2's chained with_sharding_constraint produces q_shard, not dropping."""
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

NQ, MU = 12, 320  # MoS2 3x3 scale, nq padded to divisible by 4
v_in = NamedSharding(mesh, P(None, "x", "y"))
q_shard = NamedSharding(mesh, P(("x", "y"), None, None))
reshard_mid = NamedSharding(mesh, P("x", None, "y"))
rep_3d = NamedSharding(mesh, P(None, None, None))
spec = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=v_in)
x = jax.device_put(jnp.ones((NQ, MU, MU), dtype=jnp.complex128), v_in)


# The production pattern — chained constraint inside a jit.
@jax.jit
def production_site2(V):
    V = jax.lax.with_sharding_constraint(V, reshard_mid)
    V = jax.lax.with_sharding_constraint(V, q_shard)
    return V

# With out_shardings to force
@partial(jax.jit, out_shardings=q_shard)
def out_shardings_site2(V):
    return jax.lax.with_sharding_constraint(V, reshard_mid)


for name, fn in [("chained hints (production site2)", production_site2),
                 ("hint + out_shardings", out_shardings_site2)]:
    c = fn.lower(spec).compile()
    m = c.memory_analysis()
    peak = (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9
    y = fn(x).block_until_ready()
    target = P(("x", "y"), None, None)
    ok = "OK" if y.sharding.spec == target else "MISMATCH"
    print(f"  {name:35s} peak {peak:.3f} GB  out_spec {y.sharding.spec}  [{ok}]")
