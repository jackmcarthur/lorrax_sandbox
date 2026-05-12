"""Dump HLO + buffer-assignment for reshard variants to understand the waste."""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_dump_to=/tmp/reshard_hlo_dump --xla_dump_hlo_as_text")

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
sh_xy = NamedSharding(mesh, P(None, ("x", "y"), None, None))
sh_Y = NamedSharding(mesh, P(None, "y", None, None))
sh_out = NamedSharding(mesh, P(None, None, None, "y"))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_xy)


def _peak_gb(compiled):
    m = compiled.memory_analysis()
    total = m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes
    return total / 1e9, {
        "temp": m.temp_size_in_bytes / 1e9,
        "arg": m.argument_size_in_bytes / 1e9,
        "out": m.output_size_in_bytes / 1e9,
        "alias": m.alias_size_in_bytes / 1e9,
    }


# ------------------------------------------------------------------
# Variant A — production form: 2 with_sharding_constraints, no donation
# ------------------------------------------------------------------
@jax.jit
def reshardA(psi_rchunk):
    psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, sh_Y)
    psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, sh_out)
    return psi_rchunk


# ------------------------------------------------------------------
# Variant B — same, with donation
# ------------------------------------------------------------------
@jax.jit
def _reshardB_impl(psi_rchunk):
    psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, sh_Y)
    psi_rchunk = jax.lax.with_sharding_constraint(psi_rchunk, sh_out)
    return psi_rchunk
reshardB = jax.jit(_reshardB_impl, donate_argnums=(0,))


# ------------------------------------------------------------------
# Variant C — direct: skip stage_Y, go XY -> out in one constraint
# ------------------------------------------------------------------
@jax.jit
def _reshardC_impl(psi_rchunk):
    return jax.lax.with_sharding_constraint(psi_rchunk, sh_out)
reshardC = jax.jit(_reshardC_impl, donate_argnums=(0,))


# ------------------------------------------------------------------
# Variant D — single shard_map that does a custom all-to-all
# ------------------------------------------------------------------
# Input bands sharded on both x and y; output n_rchunk on y.
# Fold the x-axis of bands into the r-chunk axis via one-shot alltoall.
def _reshardD_impl(psi_rchunk):
    # Direct with_sharding_constraint with donation; same as C but with donation.
    # Actually same as C for the AOT-lowered program.  Keep for clarity.
    return jax.lax.with_sharding_constraint(psi_rchunk, sh_out)
reshardD = jax.jit(_reshardD_impl, donate_argnums=(0,))


for name, fn in [("A_current", reshardA), ("B_donate", reshardB),
                 ("C_direct_donate", reshardC)]:
    print(f"\n==== Variant {name} ====")
    lowered = fn.lower(spec)
    compiled = lowered.compile()
    gb, parts = _peak_gb(compiled)
    print(f"  peak = {gb:.3f} GB  ({parts})")
    # Print counts of relevant stablehlo ops
    hlo = lowered.as_text("hlo")
    n_all_gather = hlo.count("all-gather")
    n_all_to_all = hlo.count("all-to-all")
    n_collective = hlo.count("collective")
    print(f"  all-gather ops: {n_all_gather}")
    print(f"  all-to-all ops: {n_all_to_all}")
    print(f"  'collective' mentions: {n_collective}")
    # Save the HLO
    with open(f"/tmp/reshard_hlo_{name}.hlo", "w") as f:
        f.write(hlo)

# Dump buffer assignment to understand peak structure
print("\nSee /tmp/reshard_hlo_dump/*-buffer-assignment.txt")
