"""Prototype explicit-shard_map fixes for the three flagged reshards."""
import os
os.environ.setdefault("XLA_FLAGS", "")

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


def peak(fn, *specs):
    c = fn.lower(*specs).compile()
    m = c.memory_analysis()
    return (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9


# =============================================================================
# FIX for SUSPECT 1 — w_isdf._solve_w V/chi reshard
# =============================================================================
# Goal: P(None, 'x', 'y') -> P(('x','y'), None, None)
# Conservation of bytes: input tile (nq, μ/px, μ/py) = output tile (nq/(px*py), μ, μ)
# Both are nq·μ²/(px*py) per device.  So no size change — two all_to_alls suffice.

NQ, MU = 64, 1200
mu_in = NamedSharding(mesh, P(None, "x", "y"))
q_out = NamedSharding(mesh, P(("x", "y"), None, None))
spec_V = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=mu_in)

@jax.jit
def fix_solve_w_reshard(V):
    """Two sequential all_to_alls: y on nq↔μ2, then x on nq↔μ1."""
    def _local(V):
        # Start tile: (nq, μ/px, μ/py)
        V = jax.lax.all_to_all(V, "y", split_axis=0, concat_axis=2, tiled=True)
        # Now: (nq/py, μ/px, μ)
        V = jax.lax.all_to_all(V, "x", split_axis=0, concat_axis=1, tiled=True)
        # Now: (nq/(px*py), μ, μ)
        return V
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, "x", "y"),
        out_specs=P(("x", "y"), None, None),
        check_rep=False,
    )(V)


# =============================================================================
# FIX for SUSPECT 2 — isdf_fitting._reshard_z
# =============================================================================
# Goal: P(None, 'x', 'y') -> P(None, None, ('x','y'))
# Input tile (nq, μ/px, Br/py) → output tile (nq, μ, Br/(px*py)) — same bytes.
# One all_to_all on x: split Br, concat μ.

BR = 12672
MU2 = 2400
z_in = NamedSharding(mesh, P(None, "x", "y"))
z_out = NamedSharding(mesh, P(None, None, ("x", "y")))
spec_Z = jax.ShapeDtypeStruct((NQ, MU2, BR), jnp.complex128, sharding=z_in)

@jax.jit
def fix_reshard_z(Z):
    def _local(z):
        # Start tile: (nq, μ/px, Br/py)
        # One all_to_all on x: split Br, concat μ.
        z = jax.lax.all_to_all(z, "x", split_axis=2, concat_axis=1, tiled=True)
        # Now: (nq, μ, Br/(px*py))  -- μ replicated on x, Br split further on x.
        # Current Br sharding on y is preserved implicitly.
        return z
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, "x", "y"),
        out_specs=P(None, None, ("x", "y")),
        check_rep=False,
    )(Z)


print("=== SUSPECT 1: w_isdf V/chi reshard (nq={}, μ={}) ===".format(NQ, MU))
print(f"  production (via rep_3d or direct): 11.80 GB")
print(f"  fix (shard_map, 2 all_to_alls):    {peak(fix_solve_w_reshard, spec_V):.3f} GB")

print()
print("=== SUSPECT 2: _reshard_z (nq={}, μ={}, Br={}) ===".format(NQ, MU2, BR))
print(f"  production:                        31.14 GB")
print(f"  fix (shard_map, 1 all_to_all):     {peak(fix_reshard_z, spec_Z):.3f} GB")
