"""Site 3 — try to get down to the theoretical 2× tile (input donated)."""
import os
os.environ.setdefault("XLA_FLAGS", "")

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

NQ, MU3, BR3 = 64, 2400, 12672
z_in = NamedSharding(mesh, P(None, "x", "y"))
z_out = NamedSharding(mesh, P(None, None, ("x", "y")))
z_mid = NamedSharding(mesh, P("x", None, "y"))
spec_Z = jax.ShapeDtypeStruct((NQ, MU3, BR3), jnp.complex128, sharding=z_in)


def peak(fn, *specs):
    c = fn.lower(*specs).compile()
    m = c.memory_analysis()
    total = (m.temp_size_in_bytes + m.argument_size_in_bytes
             + m.output_size_in_bytes - m.alias_size_in_bytes)
    return total / 1e9, m


# Tile = (nq, μ/px, Br/py) × 16 bytes / device
tile_gb = 16 * NQ * MU3 * BR3 / (2 * 2) / 1e9
print(f"Per-device tile size = {tile_gb:.2f} GB")
print(f"Theoretical minimum (2× tile, donated) ≈ {2*tile_gb:.2f} GB\n")


# ------------- Variant A: current hint fix (no donation) -------------
@jax.jit
def hint_no_donate(Z):
    Z = jax.lax.with_sharding_constraint(Z, z_mid)
    Z = jax.lax.with_sharding_constraint(Z, z_out)
    return Z


# ------------- Variant B: hint + donate -------------
hint_donate = jax.jit(
    lambda Z: jax.lax.with_sharding_constraint(
        jax.lax.with_sharding_constraint(Z, z_mid), z_out),
    donate_argnums=(0,))


# ------------- Variant C: shard_map + donate -------------
def _sm_body(Z):
    return jax.lax.all_to_all(Z, "x", split_axis=2, concat_axis=1, tiled=True)

sm_wrapper = shard_map(
    _sm_body, mesh=mesh,
    in_specs=P(None, "x", "y"),
    out_specs=P(None, None, ("x", "y")),
    check_rep=False,
)
sm_donate = jax.jit(sm_wrapper, donate_argnums=(0,))


# ------------- Variant D: direct via output shard constraint + donate -------------
direct_donate = jax.jit(
    lambda Z: jax.lax.with_sharding_constraint(Z, z_out),
    donate_argnums=(0,))


for name, fn in [("hint, no donate", hint_no_donate),
                 ("hint, donate", hint_donate),
                 ("shard_map, donate", sm_donate),
                 ("direct, donate", direct_donate)]:
    gb, m = peak(fn, spec_Z)
    print(f"  {name:22s} peak {gb:6.2f} GB  "
          f"(temp {m.temp_size_in_bytes/1e9:.2f} "
          f"arg {m.argument_size_in_bytes/1e9:.2f} "
          f"out {m.output_size_in_bytes/1e9:.2f} "
          f"alias {m.alias_size_in_bytes/1e9:.2f})")
