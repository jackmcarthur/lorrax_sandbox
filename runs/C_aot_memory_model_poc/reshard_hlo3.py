"""HLO dump v3 — XLA_FLAGS set BEFORE any jax import, writing to pscratch."""
import os
import sys

OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/C_aot_memory_model_poc/xla_dump_v3"
os.makedirs(OUT, exist_ok=True)

# MUST be set before the first `import jax`.
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={OUT} "
    "--xla_dump_hlo_as_text "
    "--xla_dump_hlo_as_dot "
    "--xla_dump_hlo_module_re=.*reshard.*"
)

# Now safe to import JAX.
import jax                       # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp          # noqa: E402
import numpy as np               # noqa: E402
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P  # noqa: E402

try:
    jax.distributed.initialize(local_device_ids=[0])
except Exception:
    pass

print(f"[rank {jax.process_index()}] XLA_FLAGS = {os.environ.get('XLA_FLAGS','')}", flush=True)

devs = jax.devices()
mesh = Mesh(np.array(devs[:4]).reshape(2, 2), ("x", "y"))

NK, NB, NS, BR = 64, 296, 2, 3456
sh_xy = NamedSharding(mesh, P(None, ("x", "y"), None, None))
sh_out = NamedSharding(mesh, P(None, None, None, "y"))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_xy)

@jax.jit
def reshard_dump(x):
    return jax.lax.with_sharding_constraint(x, sh_out)

lowered = reshard_dump.lower(spec)
compiled = lowered.compile()  # this triggers the dump if backend sees flag
m = compiled.memory_analysis()
print(f"peak = {(m.temp_size_in_bytes + m.argument_size_in_bytes + m.output_size_in_bytes - m.alias_size_in_bytes)/1e9:.3f} GB", flush=True)
print(f"temp={m.temp_size_in_bytes/1e9:.3f} arg={m.argument_size_in_bytes/1e9:.3f} out={m.output_size_in_bytes/1e9:.3f}", flush=True)

# Only rank 0 lists files (others would race).
if jax.process_index() == 0:
    import glob
    files = sorted(glob.glob(f"{OUT}/*"))
    print(f"\n{len(files)} files in {OUT}:", flush=True)
    for f in files[:30]:
        print(f"  {os.path.basename(f)}", flush=True)
