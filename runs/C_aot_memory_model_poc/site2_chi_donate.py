"""Site 2: can we donate chi_flat (argnum=1) safely for extra savings?"""
import os
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

# Match production sizes; this is _solve_w taking V, chi, pref.
NQ, MU = 64, 1200   # tractable size for test
v_in = NamedSharding(mesh, P(None, "x", "y"))
q_shard = NamedSharding(mesh, P(("x", "y"), None, None))
reshard_mid = NamedSharding(mesh, P("x", None, "y"))
rep_3d = NamedSharding(mesh, P(None, None, None))

spec_V = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=v_in)
spec_chi = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=v_in)
spec_pref = jax.ShapeDtypeStruct((), jnp.complex128)


def _body(V_flat, chi_flat, pref):
    """Mimics _solve_w reshard only (not the actual solve — just the reshards)."""
    nq_local = V_flat.shape[0]
    chi_scaled = pref * chi_flat
    V_q = jax.lax.with_sharding_constraint(
        jax.lax.with_sharding_constraint(V_flat, reshard_mid), q_shard)
    chi_q = jax.lax.with_sharding_constraint(
        jax.lax.with_sharding_constraint(chi_scaled, reshard_mid), q_shard)
    # Simulate a trivial solve that uses both and returns (nq, μ, μ).
    W = V_q - chi_q
    return jax.lax.with_sharding_constraint(W, rep_3d)


variants = {
    "no donate": jax.jit(_body),
    "donate V only (0)":    jax.jit(_body, donate_argnums=(0,)),
    "donate chi only (1)":  jax.jit(_body, donate_argnums=(1,)),
    "donate both (0,1)":    jax.jit(_body, donate_argnums=(0, 1)),
}

def peak(fn):
    c = fn.lower(spec_V, spec_chi, spec_pref).compile()
    m = c.memory_analysis()
    total = (m.temp_size_in_bytes + m.argument_size_in_bytes
             + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9
    alias = m.alias_size_in_bytes / 1e9
    return total, alias

for name, fn in variants.items():
    gb, alias = peak(fn)
    print(f"  {name:22s} peak {gb:.3f} GB   alias {alias:.3f} GB")
