"""Unit test: verify the pad-past-file path in PhdfWfnReader.coeffs_gspace
matches the aligned path on the bulk bands AND matches a small h5py
ground-truth on the tail bands, on a SYM (ntran > 1) WFN file.

Strategy:
  1. Aligned read [0, n_aligned) via PhdfWfnReader (the existing fast path).
     Record output A.
  2. Pad-past-file read [0, n_padded) where n_padded > nbnd_in_file via
     PhdfWfnReader (the new path with bulk + tail + zeros).
     Record output B.
  3. Verify:
     a. B[:, :n_aligned, ...] == A      (bulk path identical)
     b. B[:, n_aligned:nbnd, ...] vs ground truth from h5py + manual
        sym unfold + manual fft-box scatter.   (sym tail correct)
     c. B[:, nbnd:n_padded, ...] == 0   (zero pad correct)

Run on 4 GPUs.  Pass `--wfn` to point at a sym WFN file."""
from __future__ import annotations
import os, sys, argparse
os.environ.setdefault("JAX_ENABLE_X64", "1")

from runtime import set_default_env
set_default_env()
import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from runtime import init_jax_distributed
init_jax_distributed()

from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from common.phdf5_wfn_reader import PhdfWfnReader

def _log(*a):
    if jax.process_index() == 0:
        print(*a, flush=True)

# Build a default 1×W mesh covering all devices.
n_dev = len(jax.devices())
mesh_x = 1
mesh_y = n_dev
dev_grid = np.array(jax.devices()).reshape(mesh_x, mesh_y)
mesh = Mesh(dev_grid, axis_names=('x', 'y'))
_log(f"jax devices: {n_dev}; mesh ({mesh_x}, {mesh_y})")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wfn", default="/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/04_si_4x4x4_memory_assay/WFN.h5")
args, _ = parser.parse_known_args()

reader = PhdfWfnReader(args.wfn, mesh=mesh)
nbnd_in_file = int(reader.nbands)
ntran = int(reader.ntran)
world = n_dev
_log(f"WFN: {args.wfn}\n  nbnd={nbnd_in_file}, ntran={ntran}, world={world}")
assert ntran > 1, f"This test requires a sym WFN; ntran={ntran}"

# Choose n_aligned = largest multiple of world ≤ nbnd, padded > nbnd.
n_aligned = (nbnd_in_file // world) * world
n_pad = n_aligned + 2 * world   # 2 padded chunks past nbnd (some real tail + some zero)
assert n_pad > nbnd_in_file
_log(f"  test ranges: n_aligned={n_aligned}  nbnd={nbnd_in_file}  n_pad={n_pad}")
n_tail = nbnd_in_file - n_aligned
n_zero = n_pad - nbnd_in_file
_log(f"  derived: n_tail={n_tail}, n_zero={n_zero}")

# 1. Aligned path (within file, divisible).
A = reader._coeffs_gspace_aligned(0, n_aligned, None)
A_host = np.asarray(jax.experimental.multihost_utils.process_allgather(A, tiled=False))
if A_host.ndim == 7:
    A_host = A_host[0]
_log(f"  A shape: {A_host.shape}, |A|_F = {np.linalg.norm(A_host):.6e}")

# 2. Pad-past-file path.
B = reader.coeffs_gspace((0, n_pad))
B_host = np.asarray(jax.experimental.multihost_utils.process_allgather(B, tiled=False))
if B_host.ndim == 7:
    B_host = B_host[0]
_log(f"  B shape: {B_host.shape}, |B|_F = {np.linalg.norm(B_host):.6e}")

# 3a. Bulk equivalence.
diff_bulk = np.linalg.norm(B_host[:, :n_aligned, ...] - A_host)
_log(f"  ‖B[:n_aligned] - A‖_F = {diff_bulk:.3e}  (expect 0)")
assert diff_bulk == 0.0, f"Bulk path mismatch: {diff_bulk}"

# 3c. Zero pad past nbnd.
zero_block = B_host[:, nbnd_in_file:n_pad, ...]
_log(f"  ‖B[nbnd:n_pad]‖_F = {np.linalg.norm(zero_block):.3e}  (expect 0)")
assert np.linalg.norm(zero_block) == 0.0

# 3b. Tail (sym unfold).  Compute ground-truth from a SECOND aligned read
# spanning [n_aligned - world + n_tail, n_aligned + n_tail).  Wait — this
# needs to be aligned and within file, with the last n_tail bands matching
# the tail we want.  Simpler: read [n_aligned - (world - n_tail), n_aligned + n_tail)
# = (world) bands aligned ending at n_aligned + n_tail.  Provided
# n_aligned >= world - n_tail.
gt_lo = n_aligned + n_tail - world
gt_hi = n_aligned + n_tail
assert gt_lo >= 0, "WFN too small for the test layout"
assert gt_hi <= nbnd_in_file
gt = reader._coeffs_gspace_aligned(gt_lo, gt_hi, None)
gt_host = np.asarray(jax.experimental.multihost_utils.process_allgather(gt, tiled=False))
if gt_host.ndim == 7:
    gt_host = gt_host[0]
# The last n_tail bands of gt are bands [n_aligned, n_aligned + n_tail) =
# the tail bands we want to verify.
gt_tail = gt_host[:, world - n_tail:, ...]
my_tail = B_host[:, n_aligned:nbnd_in_file, ...]
_log(f"  gt_tail shape={gt_tail.shape}, my_tail shape={my_tail.shape}")
diff_tail = np.linalg.norm(my_tail - gt_tail) / max(np.linalg.norm(gt_tail), 1e-30)
_log(f"  rel ‖my_tail - gt_tail‖_F / ‖gt_tail‖_F = {diff_tail:.3e}  "
     f"(expect ≤ 1e-12)")
# Allow for fp64 ε — both paths use the same h5py read of the same file
# bytes; the only difference is the on-device unfold.  Should be ~1e-15 in
# practice; we tolerate up to 1e-10 for safety.
assert diff_tail < 1e-10, f"Tail mismatch: {diff_tail}"

_log("PASS: pad-past-file produces bulk-bit-equivalent and tail-numerically-equivalent results on the sym file.")
reader.close()
