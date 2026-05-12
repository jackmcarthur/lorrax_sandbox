"""Determinism test for compute_pair_density_spin_traced + compute_CCT_from_left_right
+ compute_L_q_from_CCT on the *same* psi inputs at different mesh shapes.

If the L_q output differs by O(1e-7) between meshes here, we've reproduced the
Si v1/v2 issue in isolation, and the bug is somewhere in the pair-density →
CCT → L_q chain (not Cholesky_2d, which we've already exonerated).
"""
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np, os, sys
from jax import config; config.update("jax_enable_x64", True)
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from runtime import init_jax_distributed
init_jax_distributed()

n_dev = len(jax.devices())
proc_idx = jax.process_index()
def _print(*a, **k):
    if proc_idx == 0: print(*a, **k, flush=True)

mesh_x = int(os.environ.get('TEST_MESH_X', '0'))
mesh_y = int(os.environ.get('TEST_MESH_Y', '0'))
if mesh_x * mesh_y != n_dev:
    if n_dev == 1: mesh_x, mesh_y = 1, 1
    elif n_dev == 4: mesh_x, mesh_y = 2, 2
    elif n_dev == 8: mesh_x, mesh_y = 2, 4
    elif n_dev == 16: mesh_x, mesh_y = 4, 4
    else: raise SystemExit(f"set TEST_MESH_X / TEST_MESH_Y for n_dev={n_dev}")

dev_grid = np.array(jax.devices()).reshape(mesh_x, mesh_y)
mesh_xy = Mesh(dev_grid, axis_names=('x', 'y'))
_print(f"jax devices: {n_dev}; mesh ({mesh_x}, {mesh_y})")

sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src')
from common.isdf_fitting import (
    compute_pair_density_spin_traced,
    compute_CCT_from_left_right,
    compute_L_q_from_CCT,
)

# --- Build synthetic centroid wavefunctions ---
nk_grid = (4, 4, 4); nkx, nky, nkz = nk_grid
nk = nkx * nky * nkz                      # 64
n_rmu = 1440
nb = 16            # band axis
ns = 1             # spinor axis (Si is non-spinor for cohsex.in)

# Same RNG seed → same psi on every mesh.
rng = np.random.default_rng(seed=42)
psi_full = (rng.standard_normal((nk, nb, ns, n_rmu))
            + 1j * rng.standard_normal((nk, nb, ns, n_rmu))).astype(np.complex128)
# Normalize per-band per-k to magnitude ~1.
psi_full /= np.sqrt(n_rmu)

# Convert to the two sharding shapes the API wants:
#   psi_rmu_Y:  (nk, nb, ns, n_rmu)  with P(None, None, None, 'y')
#   psi_rmuT_X: (nk, n_rmu, nb, ns)  with P(None, 'x', None, None), CONJUGATED
psi_rmu_Y = jax.device_put(jnp.asarray(psi_full),
                            NamedSharding(mesh_xy, P(None, None, None, 'y')))
psi_rmuT_X = jax.device_put(jnp.asarray(np.conj(psi_full).transpose(0, 3, 1, 2)),
                             NamedSharding(mesh_xy, P(None, 'x', None, None)))

# Pair density.
P_k = compute_pair_density_spin_traced(psi_rmuT_X, psi_rmu_Y, mesh_xy)
P_k.block_until_ready()
_print(f"P_k shape {P_k.shape} dtype {P_k.dtype}")

# Save P_k to disk for cross-mesh comparison.
out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_pruning_sharding_bug_2026-05-04/scripts/pd_out"
os.makedirs(out_dir, exist_ok=True)
P_k_host = jax.experimental.multihost_utils.process_allgather(P_k, tiled=False)
if P_k_host.ndim == 4 and P_k_host.shape[0] != nk:
    P_k_host = P_k_host[0]   # leading allgather batch axis
if proc_idx == 0:
    np.save(f"{out_dir}/P_k_mesh{mesh_x}x{mesh_y}.npy", np.asarray(P_k_host))

# CCT — call pair-density twice so we get two distinct, donatable buffers.
P_k_a = compute_pair_density_spin_traced(psi_rmuT_X, psi_rmu_Y, mesh_xy)
P_k_b = compute_pair_density_spin_traced(psi_rmuT_X, psi_rmu_Y, mesh_xy)
P_k_a.block_until_ready(); P_k_b.block_until_ready()
C_q = compute_CCT_from_left_right(P_k_a, P_k_b, nk_grid, mesh_xy)
C_q.block_until_ready()
nq = nkx * nky * nkz
C_q_flat = C_q.reshape(nq, n_rmu, n_rmu)
flat_shard = NamedSharding(mesh_xy, P(None, 'x', 'y'))
C_q_flat = jax.lax.with_sharding_constraint(C_q_flat, flat_shard)

# Symmetrize (would normally happen in a Hermitian sense; here C_q is generic
# from a synthetic PSD-without-extra-symmetry. Skip — compute_L_q_from_CCT
# handles its own ridge in the 1×1 path).
C_q_host = jax.experimental.multihost_utils.process_allgather(C_q_flat, tiled=False)
if C_q_host.ndim == 4: C_q_host = C_q_host[0]
if proc_idx == 0:
    np.save(f"{out_dir}/C_q_mesh{mesh_x}x{mesh_y}.npy", np.asarray(C_q_host))

# L_q via the same multi-device path the production code takes.
# Note: mesh.devices.size may be 1, so we may hit the single-device path.
L_q = compute_L_q_from_CCT(C_q_flat, mesh_xy)
L_q.block_until_ready()
L_q_host = jax.experimental.multihost_utils.process_allgather(L_q, tiled=False)
if L_q_host.ndim == 4: L_q_host = L_q_host[0]
if proc_idx == 0:
    np.save(f"{out_dir}/L_q_mesh{mesh_x}x{mesh_y}.npy", np.asarray(L_q_host))

if proc_idx == 0:
    nrm = float(np.linalg.norm(P_k_host))
    _print(f"  P_k norm = {nrm:.6e}")
    _print(f"  C_q norm = {float(np.linalg.norm(C_q_host)):.6e}")
    _print(f"  L_q norm = {float(np.linalg.norm(L_q_host)):.6e}")
