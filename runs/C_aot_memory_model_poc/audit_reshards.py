"""AOT-measure the two main suspect reshards in gwjax, with proposed fixes."""
import os
os.environ.setdefault("XLA_FLAGS", "")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

try:
    jax.distributed.initialize(local_device_ids=[0])
except Exception:
    pass

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y"))


def peak_gb(fn, *specs):
    lowered = fn.lower(*specs)
    compiled = lowered.compile()
    m = compiled.memory_analysis()
    return (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9, m


# =============================================================================
# SUSPECT 1 — w_isdf._solve_w's rep_3d → q_shard reshard
# =============================================================================
# Current code (lines 208-211):
#   V_q = with_sharding_constraint(with_sharding_constraint(V, rep_3d), q_shard)
# Pattern: P(None,'x','y') → P(None,None,None) → P(('x','y'),None,None)
# Explicitly goes through a fully-replicated intermediate.

NQ, MU = 64, 2400
q_shard = NamedSharding(mesh, P(("x", "y"), None, None))
rep_3d = NamedSharding(mesh, P(None, None, None))
mu_shard_in = NamedSharding(mesh, P(None, "x", "y"))  # how V arrives

spec_V = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=mu_shard_in)

@jax.jit
def current_rep_then_q(V):
    V = jax.lax.with_sharding_constraint(V, rep_3d)
    V = jax.lax.with_sharding_constraint(V, q_shard)
    return V

@jax.jit
def direct_reshard(V):
    """Skip the explicit rep_3d; let XLA plan direct μ-to-q swap."""
    return jax.lax.with_sharding_constraint(V, q_shard)


print("=== SUSPECT 1: w_isdf._solve_w V/chi reshard ===")
print(f"(nq={NQ}, μ={MU}), input P(None,'x','y') → output P(('x','y'),None,None)")
for name, fn in [("current (via rep_3d)", current_rep_then_q),
                 ("direct", direct_reshard)]:
    gb, m = peak_gb(fn, spec_V)
    print(f"  {name:25s}: peak {gb:.3f} GB  (temp {m.temp_size_in_bytes/1e9:.3f} "
          f"arg {m.argument_size_in_bytes/1e9:.3f} "
          f"out {m.output_size_in_bytes/1e9:.3f})")


# =============================================================================
# SUSPECT 2 — isdf_fitting._reshard_z
# =============================================================================
# Current:  P(None,'x','y') → P(None,None,('x','y'))
# μ axis: x-sharded → replicated     (x-axis reshard, inflates by p_x)
# z axis: y-sharded → ('x','y')      (both axes now shard z)

BR = 12672
z_in_shard = NamedSharding(mesh, P(None, "x", "y"))
z_out_shard = NamedSharding(mesh, P(None, None, ("x", "y")))
spec_Z = jax.ShapeDtypeStruct((NQ, MU, BR), jnp.complex128, sharding=z_in_shard)

@jax.jit
def current_reshard_z(Z):
    return jax.lax.with_sharding_constraint(Z, z_out_shard)


print("\n=== SUSPECT 2: isdf_fitting._reshard_z ===")
print(f"(nq={NQ}, μ={MU}, Br={BR}), "
      f"input P(None,'x','y') → output P(None,None,('x','y'))")
gb, m = peak_gb(current_reshard_z, spec_Z)
print(f"  current: peak {gb:.3f} GB  (temp {m.temp_size_in_bytes/1e9:.3f} "
      f"arg {m.argument_size_in_bytes/1e9:.3f} "
      f"out {m.output_size_in_bytes/1e9:.3f})")
