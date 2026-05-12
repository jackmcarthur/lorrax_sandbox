"""Dump HLO text for reshard variants — save to shared dir."""
import os
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/C_aot_memory_model_poc/hlo_out"
os.makedirs(OUT, exist_ok=True)
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_dump_to={OUT} --xla_dump_hlo_as_text --xla_dump_hlo_pass_re=.*")

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

NK, NB, NS, BR = 64, 296, 2, 3456
sh_xy = NamedSharding(mesh, P(None, ("x", "y"), None, None))
sh_Y = NamedSharding(mesh, P(None, "y", None, None))
sh_out = NamedSharding(mesh, P(None, None, None, "y"))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_xy)


def _peak(compiled):
    m = compiled.memory_analysis()
    return (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9


@jax.jit
def reshardA(x):
    x = jax.lax.with_sharding_constraint(x, sh_Y)
    x = jax.lax.with_sharding_constraint(x, sh_out)
    return x


@jax.jit
def reshardC(x):
    return jax.lax.with_sharding_constraint(x, sh_out)


for name, fn in [("A_current", reshardA), ("C_direct", reshardC)]:
    lowered = fn.lower(spec)
    compiled = lowered.compile()
    gb = _peak(compiled)
    print(f"{name}: peak {gb:.3f} GB")
    with open(f"{OUT}/reshard_{name}.hlo", "w") as f:
        f.write(lowered.as_text("hlo"))
    with open(f"{OUT}/reshard_{name}.stablehlo", "w") as f:
        f.write(lowered.as_text("stablehlo"))

print("HLO + stablehlo saved to", OUT)
