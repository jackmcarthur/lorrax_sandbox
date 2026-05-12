"""Measure peak + output sharding for the CORRECT variants of each fix."""
import os
os.environ.setdefault("XLA_FLAGS", "")

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


def measure(fn, *specs):
    lowered = fn.lower(*specs)
    c = lowered.compile()
    m = c.memory_analysis()
    peak = (m.temp_size_in_bytes + m.argument_size_in_bytes
            + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9
    # Check output sharding via an actual execution
    shape = specs[0].shape
    dtype = specs[0].dtype
    x = jax.device_put(jnp.ones(shape, dtype=dtype), specs[0].sharding)
    y = fn(x)
    return peak, m, y.sharding.spec


# ========================================================================
# SITE 1: load_wfns._reshard_rchunk (x-first, with donation)
# ========================================================================
NK, NB, NS, BR = 64, 296, 2, 3456
sh_in1 = NamedSharding(mesh, P(None, ("x", "y"), None, None))
sh_final1 = NamedSharding(mesh, P(None, None, None, "y"))
sh_stage1 = NamedSharding(mesh, P(None, "y", None, None))
spec1 = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in1)

@jax.jit
def s1_original(x):  # production form (x-first, no donate)
    x = jax.lax.with_sharding_constraint(x, sh_stage1)
    x = jax.lax.with_sharding_constraint(x, sh_final1)
    return x

@partial(jax.jit, donate_argnums=(0,))
def s1_donate(x):  # x-first WITH donation
    x = jax.lax.with_sharding_constraint(x, sh_stage1)
    x = jax.lax.with_sharding_constraint(x, sh_final1)
    return x

print("=== SITE 1: _reshard_rchunk (nb=296, Br=3456) ===")
for name, fn in [("original (x-first)", s1_original),
                 ("x-first + donate", s1_donate)]:
    peak, m, spec = measure(fn, spec1)
    ok = "OK" if spec == P(None, None, None, "y") else "MISMATCH"
    print(f"  {name:25s} peak {peak:.3f} GB  alias {m.alias_size_in_bytes/1e9:.3f} GB  [{ok}] out={spec}")

# ========================================================================
# SITE 2: w_isdf._solve_w V/chi reshard
# ========================================================================
NQ2, MU2 = 64, 1200
sh_in2 = NamedSharding(mesh, P(None, "x", "y"))
sh_out2 = NamedSharding(mesh, P(("x", "y"), None, None))
sh_mid2 = NamedSharding(mesh, P("x", None, "y"))
sh_rep2 = NamedSharding(mesh, P(None, None, None))
spec2 = jax.ShapeDtypeStruct((NQ2, MU2, MU2), jnp.complex128, sharding=sh_in2)

@jax.jit
def s2_via_rep(V):  # original
    V = jax.lax.with_sharding_constraint(V, sh_rep2)
    V = jax.lax.with_sharding_constraint(V, sh_out2)
    return V

@jax.jit
def s2_hint(V):  # fix without donate
    V = jax.lax.with_sharding_constraint(V, sh_mid2)
    V = jax.lax.with_sharding_constraint(V, sh_out2)
    return V

@partial(jax.jit, donate_argnums=(0,))
def s2_hint_donate(V):  # fix with donate
    V = jax.lax.with_sharding_constraint(V, sh_mid2)
    V = jax.lax.with_sharding_constraint(V, sh_out2)
    return V

print("\n=== SITE 2: _solve_w V/chi (nq=64, μ=1200) ===")
for name, fn in [("original (via rep)", s2_via_rep),
                 ("hint P('x',None,'y')", s2_hint),
                 ("hint + donate", s2_hint_donate)]:
    peak, m, spec = measure(fn, spec2)
    ok = "OK" if spec == P(("x", "y"), None, None) else "MISMATCH"
    print(f"  {name:25s} peak {peak:.3f} GB  alias {m.alias_size_in_bytes/1e9:.3f} GB  [{ok}] out={spec}")

# ========================================================================
# SITE 3: isdf_fitting._reshard_z
# ========================================================================
NQ3, MU3, BR3 = 64, 2400, 12672
sh_in3 = NamedSharding(mesh, P(None, "x", "y"))
sh_out3 = NamedSharding(mesh, P(None, None, ("x", "y")))
spec3 = jax.ShapeDtypeStruct((NQ3, MU3, BR3), jnp.complex128, sharding=sh_in3)

@jax.jit
def s3_direct(z):
    return jax.lax.with_sharding_constraint(z, sh_out3)

@partial(jax.jit, donate_argnums=(0,))
def s3_direct_donate(z):
    return jax.lax.with_sharding_constraint(z, sh_out3)

print("\n=== SITE 3: _reshard_z (μ=2400, Br=12672) ===")
for name, fn in [("original (direct)", s3_direct),
                 ("direct + donate", s3_direct_donate)]:
    peak, m, spec = measure(fn, spec3)
    ok = "OK" if spec == P(None, None, ("x", "y")) else "MISMATCH"
    print(f"  {name:25s} peak {peak:.3f} GB  alias {m.alias_size_in_bytes/1e9:.3f} GB  [{ok}] out={spec}")
