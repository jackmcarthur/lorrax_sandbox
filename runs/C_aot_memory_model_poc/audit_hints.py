"""Try fix variants using with_sharding_constraint hints (NOT shard_map).

If a single additional intermediate hint works as well as explicit
shard_map, the production change is a 1-line diff rather than a
collective rewrite.
"""
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


def peak_and_warn(fn, *specs, name=""):
    import sys, io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        c = fn.lower(*specs).compile()
    m = c.memory_analysis()
    gb = (m.temp_size_in_bytes + m.argument_size_in_bytes
          + m.output_size_in_bytes - m.alias_size_in_bytes) / 1e9
    warns = buf.getvalue()
    # Note: the SPMD rematerialization warning goes to C++ stderr, which
    # this Python redirect cannot capture.  Instead, infer from peak
    # magnitude vs input size.
    return gb, m


# =============================================================================
# SITE 1: reshard_rchunk — try y-first via hints
# =============================================================================
NK, NB, NS, BR = 64, 296, 2, 3456
sh_in = NamedSharding(mesh, P(None, ("x", "y"), None, None))
spec1 = jax.ShapeDtypeStruct((NK, NB, NS, BR), jnp.complex128, sharding=sh_in)

stage_Y = NamedSharding(mesh, P(None, "y", None, None))
stage_X_rchunk_Y = NamedSharding(mesh, P(None, "x", None, "y"))
final_rchunk_Y = NamedSharding(mesh, P(None, None, None, "y"))

@jax.jit
def s1_current_xfirst_hints(x):
    x = jax.lax.with_sharding_constraint(x, stage_Y)          # all-gather x
    x = jax.lax.with_sharding_constraint(x, final_rchunk_Y)   # all-to-all y
    return x

@jax.jit
def s1_yfirst_hints(x):
    """Hint through P(None,'x',None,'y'): bands on x, rchunk on y (y-first)."""
    x = jax.lax.with_sharding_constraint(x, stage_X_rchunk_Y) # all-to-all y (split rchunk, concat bands)
    x = jax.lax.with_sharding_constraint(x, final_rchunk_Y)   # all-gather x
    return x

@jax.jit
def s1_yfirst_shardmap(x):
    def _local(x):
        x = jax.lax.all_to_all(x, "y", split_axis=3, concat_axis=1, tiled=True)
        x = jax.lax.all_gather(x, "x", axis=1, tiled=True)
        return x
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, ("x", "y"), None, None),
        out_specs=P(None, None, None, "y"),
        check_rep=False,
    )(x)


print("=== SITE 1: _reshard_rchunk (nk={}, nb={}, ns={}, Br={}) ===".format(NK, NB, NS, BR))
for name, fn in [("current x-first (hints)", s1_current_xfirst_hints),
                 ("y-first (hints only)", s1_yfirst_hints),
                 ("y-first (shard_map)", s1_yfirst_shardmap)]:
    gb, m = peak_and_warn(fn, spec1, name=name)
    print(f"  {name:30s} peak {gb:.3f} GB (temp {m.temp_size_in_bytes/1e9:.3f})")

# =============================================================================
# SITE 2: _solve_w V reshard — try 2-step hints
# =============================================================================
NQ, MU = 64, 1200
v_in = NamedSharding(mesh, P(None, "x", "y"))
v_rep = NamedSharding(mesh, P(None, None, None))
v_q_out = NamedSharding(mesh, P(("x", "y"), None, None))
# Try intermediates that only change one mesh axis at a time:
v_mid_yq_x = NamedSharding(mesh, P("y", "x", None))
v_mid_xq_y = NamedSharding(mesh, P("x", None, "y"))

spec_V = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=v_in)

@jax.jit
def s2_current_via_rep(V):
    V = jax.lax.with_sharding_constraint(V, v_rep)
    V = jax.lax.with_sharding_constraint(V, v_q_out)
    return V

@jax.jit
def s2_two_step_yq_first(V):
    """Hint through P('y','x',None): y-axis moves nq↔μ2 first, then x."""
    V = jax.lax.with_sharding_constraint(V, v_mid_yq_x)
    V = jax.lax.with_sharding_constraint(V, v_q_out)
    return V

@jax.jit
def s2_two_step_xq_first(V):
    """Hint through P('x',None,'y'): x-axis moves nq↔μ1 first, then y."""
    V = jax.lax.with_sharding_constraint(V, v_mid_xq_y)
    V = jax.lax.with_sharding_constraint(V, v_q_out)
    return V

@jax.jit
def s2_shardmap(V):
    def _local(V):
        V = jax.lax.all_to_all(V, "y", split_axis=0, concat_axis=2, tiled=True)
        V = jax.lax.all_to_all(V, "x", split_axis=0, concat_axis=1, tiled=True)
        return V
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, "x", "y"),
        out_specs=P(("x", "y"), None, None),
        check_rep=False,
    )(V)


print()
print("=== SITE 2: _solve_w V/chi reshard (nq={}, μ={}) ===".format(NQ, MU))
for name, fn in [("current (via rep_3d)", s2_current_via_rep),
                 ("hint P('y','x',None) first", s2_two_step_yq_first),
                 ("hint P('x',None,'y') first", s2_two_step_xq_first),
                 ("shard_map (2 all_to_alls)", s2_shardmap)]:
    gb, m = peak_and_warn(fn, spec_V, name=name)
    print(f"  {name:30s} peak {gb:.3f} GB (temp {m.temp_size_in_bytes/1e9:.3f})")

# =============================================================================
# SITE 3: _reshard_z — try intermediate hints
# =============================================================================
MU3, BR3 = 2400, 12672
z_in = NamedSharding(mesh, P(None, "x", "y"))
z_out = NamedSharding(mesh, P(None, None, ("x", "y")))
# For _reshard_z the transform moves x from μ-axis to Br-axis while y
# stays on Br.  Test a few one-axis-at-a-time intermediate hints.
z_mid_rep_mu = NamedSharding(mesh, P(None, None, "y"))       # μ replicated, Br on y
z_mid_x_q = NamedSharding(mesh, P("x", None, "y"))           # x on nq, y on Br
z_mid_yx_reversed = NamedSharding(mesh, P(None, None, ("y", "x")))  # axes reversed

spec_Z = jax.ShapeDtypeStruct((NQ, MU3, BR3), jnp.complex128, sharding=z_in)

@jax.jit
def s3_current(Z):
    return jax.lax.with_sharding_constraint(Z, z_out)

@jax.jit
def s3_via_rep_mu(Z):
    """Intermediate: μ replicated, Br on y only."""
    Z = jax.lax.with_sharding_constraint(Z, z_mid_rep_mu)
    Z = jax.lax.with_sharding_constraint(Z, z_out)
    return Z

@jax.jit
def s3_via_x_nq(Z):
    """Intermediate: P('x', None, 'y') — x parks on nq, then moves to Br."""
    Z = jax.lax.with_sharding_constraint(Z, z_mid_x_q)
    Z = jax.lax.with_sharding_constraint(Z, z_out)
    return Z

@jax.jit
def s3_via_yx_reversed(Z):
    """Intermediate: P(None,None,('y','x')) — same output axes in reversed order."""
    Z = jax.lax.with_sharding_constraint(Z, z_mid_yx_reversed)
    Z = jax.lax.with_sharding_constraint(Z, z_out)
    return Z

@jax.jit
def s3_shardmap(Z):
    def _local(z):
        z = jax.lax.all_to_all(z, "x", split_axis=2, concat_axis=1, tiled=True)
        return z
    return shard_map(
        _local, mesh=mesh,
        in_specs=P(None, "x", "y"),
        out_specs=P(None, None, ("x", "y")),
        check_rep=False,
    )(Z)


print()
print("=== SITE 3: _reshard_z (nq={}, μ={}, Br={}) ===".format(NQ, MU3, BR3))
for name, fn in [("current (direct)", s3_current),
                 ("hint via P(None,None,'y')", s3_via_rep_mu),
                 ("hint via P('x',None,'y')", s3_via_x_nq),
                 ("hint via P(None,None,('y','x'))", s3_via_yx_reversed),
                 ("shard_map (1 all_to_all)", s3_shardmap)]:
    try:
        gb, m = peak_and_warn(fn, spec_Z, name=name)
        print(f"  {name:30s} peak {gb:.3f} GB (temp {m.temp_size_in_bytes/1e9:.3f})")
    except Exception as e:
        print(f"  {name:30s} FAILED: {type(e).__name__}: {str(e)[:80]}")
