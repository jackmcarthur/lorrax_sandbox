"""Direct determinism test of cholesky_2d_batched across mesh shapes.

Build the SAME synthetic Hermitian PSD C_q on every mesh, run the sharded
Cholesky, gather L back, and check ‖L_mesh1 - L_mesh2‖_F / ‖L_mesh1‖_F.

If this is at fp64 noise (~1e-14), cholesky_2d is fine and v1/v2 ζ diff
comes from upstream (pair-density / FFT). If this is at ~1e-7, the
Cholesky itself is the culprit.

Tests two flavors:
  1. Well-conditioned (κ ≈ 100, like the real Si ζ at q=0)
  2. Ill-conditioned (κ ≈ 1e10) — checks if any conditioning-amplification
     would happen if the input were singular.
"""
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from jax import config; config.update("jax_enable_x64", True)
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from runtime import init_jax_distributed
init_jax_distributed()

n_dev = len(jax.devices())
proc_idx = jax.process_index()
def _print(*a, **k):
    if proc_idx == 0:
        print(*a, **k, flush=True)

_print(f"jax devices: {n_dev}; process count {jax.process_count()}")

# Pick a mesh shape from env so the same script runs at multiple meshes.
import os
mesh_x = int(os.environ.get('TEST_MESH_X', '0'))
mesh_y = int(os.environ.get('TEST_MESH_Y', '0'))
if mesh_x * mesh_y != n_dev:
    if n_dev == 1:
        mesh_x, mesh_y = 1, 1
    elif n_dev == 4:
        mesh_x, mesh_y = 2, 2
    elif n_dev == 8:
        mesh_x, mesh_y = 2, 4
    elif n_dev == 16:
        mesh_x, mesh_y = 4, 4
    else:
        raise SystemExit(f"set TEST_MESH_X / TEST_MESH_Y for n_dev={n_dev}")

dev_grid = np.array(jax.devices()).reshape(mesh_x, mesh_y)
mesh_xy = Mesh(dev_grid, axis_names=('x', 'y'))
_print(f"mesh: ({mesh_x}, {mesh_y})")

import sys
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_A/src')
from common.cholesky_2d import (
    cholesky_2d_batched, dense_to_tiles, tiles_to_dense,
)
from common.fft_helpers import compute_block_size_for_2d_cholesky

# --- Build a deterministic Hermitian PSD C on rank 0 then broadcast to all ---
n_rmu = 1440
nq = 4   # small batch — we just want determinism, not speed
rng = np.random.default_rng(seed=2026)

def make_psd(n, kappa, rng):
    """Random Hermitian PSD with condition number ~kappa.
    eigvals span [1, kappa], log-spaced. Spectrum is fully under our control.
    """
    eig = np.geomspace(1.0, kappa, n)
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    M = (Q * eig[None, :]) @ Q.conj().T
    M = 0.5 * (M + M.conj().T)
    return M.astype(np.complex128)

# Two regimes
C_well_np = np.stack([make_psd(n_rmu, 100.0, rng) for _ in range(nq)])
C_ill_np  = np.stack([make_psd(n_rmu, 1e10, rng)  for _ in range(nq)])

# Identical bytes across processes — make on every rank with same seed.
def run_chol_on_mesh(C_np, label):
    Pr, Pc = mesh_xy.shape['x'], mesh_xy.shape['y']
    block_size, J = compute_block_size_for_2d_cholesky(n_rmu, Pr, Pc)
    _print(f"  [{label}] block={block_size}, J={J} (n_rmu={n_rmu}, Pr={Pr}, Pc={Pc})")
    chol_fn = cholesky_2d_batched(mesh_xy, J, block_size)

    # Match isdf_fitting flow exactly: dense → dense_to_tiles → constraint
    C_jax = jax.device_put(jnp.asarray(C_np),
                           NamedSharding(mesh_xy, P(None, 'x', 'y')))
    C_tiles = dense_to_tiles(C_jax, block_size)
    tiles_sh = NamedSharding(mesh_xy, P(None, 'x', 'y', None, None))
    C_tiles = jax.lax.with_sharding_constraint(C_tiles, tiles_sh)

    L_tiles = chol_fn(C_tiles)
    L_tiles.block_until_ready()
    L_dense_sh = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    L_dense = tiles_to_dense(L_tiles, block_size)
    # Gather to rank 0
    L_host = np.asarray(jax.experimental.multihost_utils.process_allgather(L_dense, tiled=False))
    return L_host[0] if L_host.ndim == 4 else L_host  # (nq, n, n) on rank 0

L_well = run_chol_on_mesh(C_well_np, "well")
L_ill  = run_chol_on_mesh(C_ill_np,  "ill")

# Save to disk so we can compare across mesh runs.
out_dir = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/zeta_pruning_sharding_bug_2026-05-04/scripts/chol2d_out"
os.makedirs(out_dir, exist_ok=True)
if proc_idx == 0:
    np.save(f"{out_dir}/L_well_mesh{mesh_x}x{mesh_y}.npy", np.asarray(L_well))
    np.save(f"{out_dir}/L_ill_mesh{mesh_x}x{mesh_y}.npy",  np.asarray(L_ill))

    # Also reference: replicated jnp.linalg.cholesky for ground truth comparison.
    L_well_ref = np.linalg.cholesky(C_well_np)
    L_ill_ref  = np.linalg.cholesky(C_ill_np)
    # Reconstruction errors
    err_well = np.linalg.norm(L_well - L_well_ref) / np.linalg.norm(L_well_ref)
    err_ill  = np.linalg.norm(L_ill - L_ill_ref) / np.linalg.norm(L_ill_ref)
    _print(f"\n  cholesky_2d({mesh_x}x{mesh_y}) vs np.linalg.cholesky:")
    _print(f"    well-conditioned (κ=100):  ‖ΔL‖/‖L‖ = {err_well:.3e}")
    _print(f"    ill-conditioned  (κ=1e10): ‖ΔL‖/‖L‖ = {err_ill:.3e}")
    # Reconstruction: ‖L L† - C‖ / ‖C‖
    rec_well = np.linalg.norm(L_well @ L_well.conj().swapaxes(-1, -2) - C_well_np) / np.linalg.norm(C_well_np)
    rec_ill  = np.linalg.norm(L_ill @ L_ill.conj().swapaxes(-1, -2) - C_ill_np) / np.linalg.norm(C_ill_np)
    _print(f"    well: ‖L L† - C‖ / ‖C‖ = {rec_well:.3e}")
    _print(f"    ill:  ‖L L† - C‖ / ‖C‖ = {rec_ill:.3e}")
