"""AOT-lower solve_q on a 3x3 CPU mesh to break p_x*p_y = 4 collinearity.

XLA_FLAGS=--xla_force_host_platform_device_count=9 makes 9 virtual CPU
devices.  The SPMD partitioner runs the same compiler passes (sharding
+ collective insertion) as on GPU; memory_analysis() still reports
per-device predicted bytes.  We don't execute — purely lower/compile."""
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=9"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

print(f"devices: {jax.devices()}")
assert len(jax.devices()) == 9, "need 9 virtual CPU devices"

# Also keep 2x2 and 1x4 for comparison
meshes = {
    "2x2": Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y")),
    "1x4": Mesh(np.array(jax.devices()[:4]).reshape(1, 4), ("x", "y")),
    "3x3": Mesh(np.array(jax.devices()[:9]).reshape(3, 3), ("x", "y")),
    "2x3": Mesh(np.array(jax.devices()[:6]).reshape(2, 3), ("x", "y")),
    "3x1": Mesh(np.array(jax.devices()[:3]).reshape(3, 1), ("x", "y")),
}

# Match the solve_q production: L arrives sharded on (x,y), gather to rep
# inside, then triangular solve with vmap.  Replicate the kernel body.
def solve_body(mesh, nq, mu, Br):
    L_batch_rep_shard = NamedSharding(mesh, P(None, None, None))
    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, None, None), P(None, None, ("x", "y"))),
             out_specs=P(None, None, ("x", "y")))
    def _sharded_cho_solve_batch(L_batch, Z_batch):
        def _solve_single(L, Z):
            y = jax.scipy.linalg.solve_triangular(L, Z, lower=True)
            return jax.scipy.linalg.solve_triangular(L.conj().T, y, lower=False)
        return jax.vmap(_solve_single)(L_batch, Z_batch)

    @jax.jit
    def _solve_all_at_once(L_q_sharded, Z_col):
        L_full_rep = jax.lax.with_sharding_constraint(L_q_sharded, L_batch_rep_shard)
        return _sharded_cho_solve_batch(L_full_rep, Z_col)
    return _solve_all_at_once


# Use dims divisible by every needed P value (2,3,4,6,9,12).
# nq=36 (2·2·3·3), μ=120 (all small factors), Br=360 (divisible by all).
NQ, MU, BR = 36, 120, 360

print("\n=== solve_q AOT peak vs mesh (nq=36, μ=120, Br=360) ===")
samples = []
for name, mesh in meshes.items():
    px, py = mesh.shape["x"], mesh.shape["y"]
    P_tot = px*py
    # These must divide cleanly.
    if NQ % P_tot or BR % P_tot or MU % px or MU % py:
        print(f"  {name:5s}: SKIP — dims not divisible by mesh")
        continue
    sh_L = NamedSharding(mesh, P(None, "x", "y"))
    sh_Z = NamedSharding(mesh, P(None, None, ("x", "y")))
    spec_L = jax.ShapeDtypeStruct((NQ, MU, MU), jnp.complex128, sharding=sh_L)
    spec_Z = jax.ShapeDtypeStruct((NQ, MU, BR), jnp.complex128, sharding=sh_Z)
    fn = solve_body(mesh, NQ, MU, BR)
    c = fn.lower(spec_L, spec_Z).compile()
    m = c.memory_analysis()
    peak_B = (m.temp_size_in_bytes + m.argument_size_in_bytes
              + m.output_size_in_bytes - m.alias_size_in_bytes)
    T_Lrep = 16 * NQ * MU * MU
    T_Lshard = T_Lrep / P_tot
    T_Zcol = 16 * NQ * MU * BR / P_tot
    samples.append((name, px, py, peak_B, T_Lrep, T_Lshard, T_Zcol))
    print(f"  {name:5s} (p={px}x{py}, P={P_tot}): peak {peak_B/1e6:7.2f} MB   "
          f"T_Lrep={T_Lrep/1e6:6.2f}  T_Lshard={T_Lshard/1e6:6.2f}  T_Zcol={T_Zcol/1e6:6.2f}")

# Now fit peak = β_Lrep·T_Lrep + β_Lshard·T_Lshard + β_Zcol·T_Zcol + intercept
from scipy.optimize import nnls
X_rows = []
y = []
for (_, _, _, peak_B, T_Lrep, T_Lshard, T_Zcol) in samples:
    X_rows.append([1.0, T_Lrep, T_Lshard, T_Zcol])
    y.append(peak_B)
X = np.asarray(X_rows)
y = np.asarray(y)
coefs, rnorm = nnls(X, y)
print(f"\n  fit with {len(samples)} samples (intercept, β_Lrep, β_Lshard, β_Zcol):")
print(f"    intercept   = {coefs[0]/1e6:.2f} MB")
print(f"    β[Lrep]     = {coefs[1]:.3f}")
print(f"    β[Lshard]   = {coefs[2]:.3f}")
print(f"    β[Zcol]     = {coefs[3]:.3f}")
print(f"    residual RMS = {rnorm/np.sqrt(len(samples))/1e6:.3f} MB")
