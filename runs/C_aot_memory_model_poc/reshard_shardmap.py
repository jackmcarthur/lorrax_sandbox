"""Test an explicit-shard_map reshard vs the production with_sharding_constraint pair.

Production form (load_wfns._reshard_rchunk):
    psi = with_sharding_constraint(psi, P(None,'y',None,None))    # all-gather-x
    psi = with_sharding_constraint(psi, P(None,None,None,'y'))    # all-to-all-y

Observed problem: XLA's SPMD partitioner does an Involuntary full
rematerialization.  The "stage_Y" intermediate is ignored and a fully
replicated c128[nk, nb, ns, n_rchunk] intermediate is materialized.

Candidate fix: force the two collectives explicitly with shard_map.
"""
import os
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/C_aot_memory_model_poc/xla_dump_shardmap"
os.makedirs(OUT, exist_ok=True)
os.environ["XLA_FLAGS"] = (
    f"--xla_dump_to={OUT} --xla_dump_hlo_as_text "
    "--xla_dump_hlo_module_re=.*reshard.*"
)

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
sh_in = NamedSharding(mesh, P(None, ("x", "y"), None, None))
sh_out = NamedSharding(mesh, P(None, None, None, "y"))
spec = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in)


def _peak(compiled):
    m = compiled.memory_analysis()
    total = (m.temp_size_in_bytes + m.argument_size_in_bytes
             + m.output_size_in_bytes - m.alias_size_in_bytes)
    return total / 1e9, m


# ----------------------------------------------------------------
# Production (baseline)
# ----------------------------------------------------------------
_stage_Y = NamedSharding(mesh, P(None, "y", None, None))
_final_Y = NamedSharding(mesh, P(None, None, None, "y"))

@jax.jit
def reshard_production(x):
    x = jax.lax.with_sharding_constraint(x, _stage_Y)
    x = jax.lax.with_sharding_constraint(x, _final_Y)
    return x


# ----------------------------------------------------------------
# Proposed: explicit shard_map with all_gather('x') + all_to_all('y')
# ----------------------------------------------------------------
@jax.jit
def reshard_shardmap(psi_rchunk):
    """From P(None,('x','y'),None,None) to P(None,None,None,'y') in
    exactly two collectives — one per mesh axis — with no intermediate
    full-replication.

    Inside shard_map, input local shape is
        (nk, nb_pad/(p_x·p_y), ns, n_rchunk)
    After all_gather('x', axis=1, tiled=True):
        (nk, nb_pad/p_y, ns, n_rchunk)       -- bands still y-sharded
    After all_to_all('y', split=3, concat=1, tiled=True):
        (nk, nb_pad, ns, n_rchunk/p_y)       -- bands replicated, rchunk on y
    """
    def _local(psi_local):
        psi = jax.lax.all_gather(psi_local, "x", axis=1, tiled=True)
        psi = jax.lax.all_to_all(psi, "y", split_axis=3, concat_axis=1,
                                  tiled=True)
        return psi
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, ("x", "y"), None, None),
        out_specs=P(None, None, None, "y"),
        check_rep=False,
    )(psi_rchunk)


for name, fn in [("production", reshard_production),
                 ("shardmap", reshard_shardmap)]:
    print(f"\n==== {name} ====", flush=True)
    lowered = fn.lower(spec)
    compiled = lowered.compile()
    gb, m = _peak(compiled)
    print(f"  peak = {gb:.3f} GB", flush=True)
    print(f"  temp={m.temp_size_in_bytes/1e9:.3f} arg={m.argument_size_in_bytes/1e9:.3f} "
          f"out={m.output_size_in_bytes/1e9:.3f}", flush=True)
    # Save HLO for inspection.
    if jax.process_index() == 0:
        with open(f"{OUT}/hlo_{name}.txt", "w") as f:
            f.write(lowered.as_text("hlo"))

if jax.process_index() == 0:
    import glob
    files = sorted(glob.glob(f"{OUT}/*"))
    print(f"\n{len(files)} files in {OUT}", flush=True)
