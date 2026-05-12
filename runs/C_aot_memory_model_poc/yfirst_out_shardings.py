"""Does partial(jit, out_shardings=...) force the final all-gather without
needing shard_map?  Simpler if so."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    jax.distributed.initialize(local_device_ids=[0])
except Exception:
    pass
devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y"))

NK, NB, NS, BR = 9, 80, 2, 46080  # MoS2 3x3 scale
sh_in = NamedSharding(mesh, P(None, ("x", "y"), None, None))
stage = NamedSharding(mesh, P(None, "x", None, "y"))
final = NamedSharding(mesh, P(None, None, None, "y"))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in)

x = jax.device_put(jnp.ones((NK, NB, NS, BR), dtype=jnp.complex128), sh_in)


# v1: hint-only — XLA drops the final constraint (known bug)
@jax.jit
def v_hints(psi):
    psi = jax.lax.with_sharding_constraint(psi, stage)
    return jax.lax.with_sharding_constraint(psi, final)

# v2: hint + out_shardings — forces XLA to actually produce final sharding
@partial(jax.jit, out_shardings=final)
def v_out_shardings(psi):
    return jax.lax.with_sharding_constraint(psi, stage)

# v3: shard_map (what I just committed)
@jax.jit
def v_sm(psi):
    def _local(psi):
        psi = jax.lax.all_to_all(psi, "y", split_axis=3, concat_axis=1, tiled=True)
        psi = jax.lax.all_gather(psi, "x", axis=1, tiled=True)
        return psi
    return shard_map(_local, mesh=mesh,
                      in_specs=P(None, ("x", "y"), None, None),
                      out_specs=P(None, None, None, "y"),
                      check_rep=False)(psi)

# v4 — hint + out_shardings + stage hint
@partial(jax.jit, out_shardings=final)
def v_stage_and_out(psi):
    psi = jax.lax.with_sharding_constraint(psi, stage)
    return jax.lax.with_sharding_constraint(psi, final)


for name, fn in [("hints only", v_hints),
                 ("out_shardings only", v_out_shardings),
                 ("shard_map", v_sm),
                 ("stage hint + out_shardings", v_stage_and_out)]:
    # AOT
    c = fn.lower(spec).compile()
    m = c.memory_analysis()
    peak = (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9
    # Run
    y = fn(x).block_until_ready()
    ok = y.sharding.spec == P(None, None, None, "y")
    print(f"  {name:30s} peak {peak:.3f} GB  out_spec {y.sharding.spec}  "
          f"{'OK' if ok else 'MISMATCH'}")
