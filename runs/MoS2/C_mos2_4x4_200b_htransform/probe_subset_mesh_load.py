"""Smoke test: does load_centroids_band_chunked work with a submesh that
covers only a subset of the global JAX cluster's devices?

If it works, the docs-blessed Pattern B' for htransform is viable:
    1. Call loader with mesh_xy_one (one k-row's devices)
    2. device_put the result onto the full 3D mesh, replicated on 'k'

If it fails, we know we need to add a 'k_axis' argument to the loaders
(modify load_wfns.py).

Run with: LORRAX_NGPU=4 lxrun python3 -u probe_subset_mesh_load.py
"""
from __future__ import annotations
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import traceback

# Required for the lorrax python path / runtime setup
from runtime import init_jax_distributed
init_jax_distributed()

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io import WFNReader
from file_io.centroids import load_centroids
from common import symmetry_maps
from common import Meta
from common.load_wfns import load_centroids_band_chunked


def banner(msg, rank):
    if rank == 0:
        print(f"\n{'='*60}\n{msg}\n{'='*60}", flush=True)


def main():
    rank = jax.process_index()
    nproc = jax.process_count()
    ndev = jax.device_count()

    banner(f"probe: nproc={nproc}, ndev={ndev}, devices={jax.devices()}", rank)

    # Set up WFN + SymMaps + Meta exactly as gw_jax does.
    wfn = WFNReader("WFN.h5")
    sym = symmetry_maps.SymMaps(wfn)
    nval, ncond, nband = 26, 34, 60
    _, centroid_indices, n_rmu = load_centroids("centroids_frac_480.txt", wfn.fft_grid)
    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor=False)

    # Build the 3D mesh. For 4 GPUs on a 4x4 k-grid: pk=4, px=py=1.
    # In real production with ≥16 GPUs we'd factor differently.
    pk = min(meta.nk_tot, ndev)
    px = py = 1
    while pk * px * py < ndev:
        # absorb extra GPUs into px/py for future scalability; today this
        # just keeps the loop happy when ndev is unusual.
        if px <= py:
            px *= 2
        else:
            py *= 2
    pk = ndev // (px * py)  # truncate; for 4 GPUs this gives pk=4, px=py=1

    devs = np.asarray(jax.devices()[:pk * px * py]).reshape(pk, px, py)
    mesh_3d = Mesh(devs, ('k', 'x', 'y'))
    banner(f"mesh_3d: shape=(pk={pk}, px={px}, py={py})", rank)

    # Pattern A: each proc gets its OWN single-device 2D mesh (its local device).
    # Every proc loads the full data redundantly (cheap for psi_rmu).
    my_dev = jax.local_devices()[0]
    mesh_xy_local = Mesh(np.asarray([my_dev]).reshape(1, 1), ('x', 'y'))
    if rank == 0:
        print(f"  rank {rank} local devices: {jax.local_devices()}", flush=True)
        print(f"  rank {rank} mesh_xy_local devices: {list(mesh_xy_local.devices.flatten())}",
              flush=True)

    # === Test 1: each proc calls loader on its own local-device mesh ===
    banner("Test 1: load_centroids_band_chunked(mesh_xy_local) — Pattern A redundant load", rank)
    try:
        psi_rmu_local, _ = load_centroids_band_chunked(
            wfn, sym, meta, centroid_indices, bispinor=False,
            mesh_xy=mesh_xy_local, band_range=(0, nband),
            band_chunk_size=64,
        )
        banner(
            f"  ✓ rank {rank}: shape={psi_rmu_local.shape}, "
            f"sharding={psi_rmu_local.sharding}",
            rank,
        )
        if rank == 0:
            print(f"  rank 0 addressable: "
                  f"{[s.data.shape for s in psi_rmu_local.addressable_shards]}",
                  flush=True)
    except Exception as e:
        banner(f"  ✗ Loader raised: {type(e).__name__}: {e}", rank)
        if rank == 0:
            traceback.print_exc()
        return 1

    # === Test 2: construct global 3D-meshed Array via make_array_from_process_local_data ===
    banner("Test 2: assemble global Array on mesh_3d via make_array_from_process_local_data", rank)
    try:
        # Pull this proc's data to host as numpy, then re-place onto its local
        # device with the global sharding spec. Each proc contributes its OWN
        # full copy as its addressable shard; sharding is replicated-on-k.
        host_data = jax.device_get(psi_rmu_local)
        global_sharding = NamedSharding(mesh_3d, P(None, None, None, 'y'))
        psi_rmu_full = jax.make_array_from_process_local_data(
            global_sharding, host_data, host_data.shape)
        jax.block_until_ready(psi_rmu_full)
        banner(
            f"  ✓ Global Array built. shape={psi_rmu_full.shape}, "
            f"sharding={psi_rmu_full.sharding}",
            rank,
        )
        if rank == 0:
            print(f"  rank 0 addressable: "
                  f"{[s.data.shape for s in psi_rmu_full.addressable_shards]}",
                  flush=True)
    except Exception as e:
        banner(f"  ✗ make_array raised: {type(e).__name__}: {e}", rank)
        if rank == 0:
            traceback.print_exc()
        return 2

    # === Test 3: verify all procs see consistent data ===
    banner("Test 3: cross-process consistency check", rank)
    try:
        # Pull a deterministic small slice (the first centroid sample for band 0,
        # k=0, spin 0) — should be identical on every proc.
        # Have to gather to a specific device to read.
        check_val = jax.device_get(psi_rmu_full[0, 0, 0, 0])
        # Compare across procs via broadcast_one_to_all
        from jax.experimental.multihost_utils import broadcast_one_to_all
        check_arr = np.array([check_val.real, check_val.imag], dtype=np.float64)
        check_from_0 = broadcast_one_to_all(check_arr)
        max_diff = float(np.max(np.abs(check_arr - check_from_0)))
        banner(
            f"  rank {rank}: psi_rmu_full[0,0,0,0]={check_val:.6e}, "
            f"|diff vs rank 0|={max_diff:.3e}",
            rank,
        )
        if max_diff > 1e-12:
            banner(f"  ✗ Disagreement across procs (max diff {max_diff:.3e})", rank)
            return 3
        banner("  ✓ All procs agree", rank)
    except Exception as e:
        banner(f"  ✗ consistency check raised: {type(e).__name__}: {e}", rank)
        if rank == 0:
            traceback.print_exc()
        return 4

    banner("ALL TESTS PASSED — Pattern A (per-proc redundant load + replicated assembly) works", rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
