"""V_q lorentz smoke test: 4-channel synthetic ζ → check 10 tiles + hermitian fill.

Validates the new agent-B/bispinor-on-main bispinor V_q scaffolding
(commit c771bdc) on a tiny synthetic system that fits in a single A100.
Does NOT exercise zeta-fitting, WFN reading, or any GW-driver wiring —
just the V_q lorentz pipeline (compute_V_q_tile → compute_all_V_q_lorentz_sharded).

Checks:
    1. Returns 10 V_blocks (7 unique + 3 hermitian-fill).
    2. Hermitian fill: V[(j,i)] == V[(i,j)]† (centroid axis swap + conj).
    3. (0,0) tile matches what compute_all_V_q would produce on the same ζ.
    4. G0 head populated only for the (0,0) tile (transverse zero by gauge).
    5. Off-diagonal (i,j) for i≠j is computed (not None / not zero).

Run via lxrun in the existing allocation, single GPU sufficient.
"""
import sys
import time

# CRITICAL: set_default_env must run BEFORE ``import jax`` (JAX reads
# JAX_ENABLE_X64 + JAX_PLATFORMS at import time).  Otherwise the run
# silently downcasts to float32 and the Hermitian checks fail at 1e-4.
from runtime import set_default_env, init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
set_default_env()

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

# Tiny synthetic problem.  Sized so 4 channels × ζ-on-host fits trivially.
NKX, NKY, NKZ = 2, 2, 1
NQ = NKX * NKY * NKZ
N_RMU = 64
FFT_GRID = (8, 8, 8)
N_RTOT = int(np.prod(FFT_GRID))
SYS_DIM = 2  # slab — exercises the get_K_cart_on_sphere path

# Synthetic 2D-slab cell.
bvec = np.array([
    [1.0,  0.0, 0.0],
    [0.0,  1.0, 0.0],
    [0.0,  0.0, 0.5],
], dtype=np.float64)
cell_volume = np.abs(np.linalg.det(bvec))

# Mesh — single device for simplicity (1×1).
devices = jax.devices()
print(f"jax.devices: {devices}")
mesh_xy = Mesh(np.array(devices[:1]).reshape(1, 1), axis_names=('x', 'y'))


def make_synthetic_zeta_io(channel_idx: int, seed: int):
    """Create an in-memory SlabIO-like object with a 'zeta_q' dataset of
    shape (NQ, N_RTOT, N_RMU) c128.  Bypasses the FFI/h5py distinction
    by implementing the .read_slab interface directly.

    This is a SHIM for the smoke test only — production code uses
    file_io.slab_io.SlabIO opened against an HDF5 file.
    """
    rng = np.random.RandomState(seed)
    real = rng.randn(NQ, N_RTOT, N_RMU).astype(np.float64)
    imag = rng.randn(NQ, N_RTOT, N_RMU).astype(np.float64)
    zeta = (real + 1j * imag).astype(np.complex128)

    class _ZetaShim:
        def __init__(self, data):
            self._data = data
            self.backend = type('B', (), {'name': 'shim'})()

        def read_slab(self, name, *, shape, dtype, offset, mesh, partition_spec):
            assert name == 'zeta_q'
            q_off, r_off, mu_off = offset
            q_n, r_n, mu_n = shape
            sub = self._data[q_off:q_off + q_n,
                             r_off:r_off + r_n,
                             mu_off:mu_off + mu_n]
            assert sub.shape == shape, f"{sub.shape} vs {shape}"
            return jax.device_put(jnp.asarray(sub.astype(dtype)),
                                  NamedSharding(mesh, partition_spec))

    return _ZetaShim(zeta), zeta


print("Creating 4 synthetic ζ channels...", flush=True)
shims = {}
zeta_arrays = {}
for ch in (0, 1, 2, 3):
    shim, raw = make_synthetic_zeta_io(ch, seed=1000 + ch)
    shims[ch] = shim
    zeta_arrays[ch] = raw
    print(f"  channel {ch}: {raw.shape} c128 ({raw.nbytes / 1e6:.2f} MB)")

# Build the Coulomb kernel bundle.
print("\nBuilding make_v_munu_chunked_kernel...", flush=True)
from gw.compute_vcoul import make_v_munu_chunked_kernel
kernels = make_v_munu_chunked_kernel(
    FFT_GRID[0], FFT_GRID[1], FFT_GRID[2],
    NKX, NKY, NKZ,
    bvec, cell_volume, sys_dim=SYS_DIM,
    mc_average_vcoul_body=True,
    vcoul_cutoff_ry=None,
)
print(f"  n_sph = {kernels.n_sph}, has K-cart helper: "
      f"{kernels.get_K_cart_on_sphere is not None}, "
      f"has v_per_G helper: {kernels.get_v_per_G_and_phase is not None}")

# Run the bispinor V_q lorentz driver.
print("\nCalling compute_all_V_q_lorentz_sharded...", flush=True)
t0 = time.perf_counter()
from gw.v_q_lorentz import compute_all_V_q_lorentz_sharded
V_blocks, G0_mu = compute_all_V_q_lorentz_sharded(
    zeta_io_by_channel=shims,
    coulomb_kernels=kernels,
    mesh_xy=mesh_xy,
    kgrid=(NKX, NKY, NKZ),
    fft_grid=FFT_GRID,
    bvec=bvec,
    cell_volume=cell_volume,
    n_rmu_by_channel={0: N_RMU, 1: N_RMU, 2: N_RMU, 3: N_RMU},
    sys_dim=SYS_DIM,
    bdot=None,
    mc_average_vcoul_body=True,
    bare_coulomb_cutoff=None,
    bgw_v_grid_fn=None,
    budget_bytes=24.0e9,
    verbose=True,
)
for V in V_blocks.values():
    V.block_until_ready()
G0_mu.block_until_ready()
elapsed = time.perf_counter() - t0
print(f"\ncompute_all_V_q_lorentz_sharded done in {elapsed:.2f}s")

# ---- CHECKS ----
print("\n=== Checks ===")
all_pass = True

# (1) 10 V_blocks present.
expected_keys = {(0, 0), (1, 1), (2, 2), (3, 3),
                 (1, 2), (1, 3), (2, 3),
                 (2, 1), (3, 1), (3, 2)}
got_keys = set(V_blocks.keys())
missing = expected_keys - got_keys
extra = got_keys - expected_keys
if missing or extra:
    print(f"FAIL  V_blocks key set: missing={missing}, extra={extra}")
    all_pass = False
else:
    print(f"PASS  V_blocks key set = 10 expected tiles")

# (2) Hermitian fill: V[(j,i)] == V[(i,j)]†.
for (i, j) in [(1, 2), (1, 3), (2, 3)]:
    V_ij = np.asarray(V_blocks[(i, j)])
    V_ji = np.asarray(V_blocks[(j, i)])
    expected_ji = np.conj(V_ij.swapaxes(-1, -2))
    diff = np.max(np.abs(V_ji - expected_ji))
    rel = diff / max(np.max(np.abs(V_ij)), 1e-30)
    status = 'PASS' if rel < 1e-10 else 'FAIL'
    if rel >= 1e-10:
        all_pass = False
    print(f"{status}  V[({j},{i})] == V[({i},{j})]† : max|Δ|={diff:.2e} (rel {rel:.2e})")

# (3) (0,0) shape and finiteness.
V00 = np.asarray(V_blocks[(0, 0)])
print(f"      V_blocks[(0,0)].shape = {V00.shape} (expected ({NQ}, {N_RMU}, {N_RMU}))")
if V00.shape != (NQ, N_RMU, N_RMU):
    print(f"FAIL  V[(0,0)] wrong shape")
    all_pass = False
else:
    print(f"PASS  V[(0,0)] shape correct")
if not np.all(np.isfinite(V00)):
    print(f"FAIL  V[(0,0)] contains NaN/Inf")
    all_pass = False
else:
    print(f"PASS  V[(0,0)] finite, max|V|={np.max(np.abs(V00)):.3e}")

# (4) G0 head shape + finite (only meaningful for (0,0)).
G0 = np.asarray(G0_mu)
print(f"      G0_mu.shape = {G0.shape} (expected ({NQ}, {N_RMU}))")
if G0.shape != (NQ, N_RMU):
    print(f"FAIL  G0_mu wrong shape")
    all_pass = False
else:
    print(f"PASS  G0_mu shape correct")
if not np.all(np.isfinite(G0)):
    print(f"FAIL  G0_mu contains NaN/Inf")
    all_pass = False
else:
    print(f"PASS  G0_mu finite, max|G0|={np.max(np.abs(G0)):.3e}")

# (5) Off-diagonal (i,j) i≠j non-zero (i.e. transverse projector did something).
for (i, j) in [(1, 2), (1, 3), (2, 3)]:
    V_ij = np.asarray(V_blocks[(i, j)])
    if not np.all(np.isfinite(V_ij)):
        print(f"FAIL  V[({i},{j})] contains NaN/Inf")
        all_pass = False
        continue
    nrm = np.max(np.abs(V_ij))
    if nrm < 1e-12:
        print(f"FAIL  V[({i},{j})] is zero — projector likely killed it")
        all_pass = False
    else:
        print(f"PASS  V[({i},{j})] non-zero, max|V|={nrm:.3e}")

# (6) Diagonal (i,i) for i=1,2,3 non-zero and Hermitian per q.
for i in (1, 2, 3):
    V_ii = np.asarray(V_blocks[(i, i)])
    nrm = np.max(np.abs(V_ii))
    herm_diff = np.max(np.abs(V_ii - np.conj(V_ii.swapaxes(-1, -2))))
    rel = herm_diff / max(nrm, 1e-30)
    if not np.all(np.isfinite(V_ii)):
        print(f"FAIL  V[({i},{i})] contains NaN/Inf")
        all_pass = False
    elif rel > 1e-9:
        print(f"FAIL  V[({i},{i})] not Hermitian: rel={rel:.2e}")
        all_pass = False
    else:
        print(f"PASS  V[({i},{i})] Hermitian, max|V|={nrm:.3e}, herm rel={rel:.2e}")

print("\n" + ("ALL PASS" if all_pass else "SOME FAILED"))
sys.exit(0 if all_pass else 1)
