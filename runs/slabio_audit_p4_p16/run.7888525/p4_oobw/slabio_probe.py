"""SlabIO read/write audit probe — one case per process launch.

Extends the method certified in /scratch2/08271/jackmc/phdf5_padrank
(RESULTS.md): every rank's stdout and stderr go to its OWN file, and
LORRAX_PHDF5_WRITE_DEBUG=1 makes the writer announce its per-call decision,
so a bounds/emptiness verdict is OBSERVED rather than inferred.  stderr
written just before process death is lost under srun+apptainer, which is why
the merged log is not the instrument.

Geometry is the production one in kind: dataset (n_q, mu, nG) LOGICAL,
buffer (n_q, mu_padded, nG) PHYSICAL, sharded P(None, ('x','y'), None).
``mu = P + 1`` so ranks with ``r*loc >= mu`` own a WHOLLY-PADDED block --
the shape that killed jobs 7885953.

Cases (SLABIO_CASE):

  rt   cross-backend round-trip matrix.  Write the same reference with each
       available backend, read every file back with every backend, in two
       request shapes: (a) logical + replicated, (b) PADDED physical shape
       with valid_shape=logical + sharded on mu, which is the shape that
       puts wholly-padded ranks in the READ path.  Also: replica_dup
       (sharding consumes only one mesh axis), and the fill-time contract
       for a created-but-never-written dataset.
  raw  read-after-write on ONE open handle.  write_slab only ENQUEUES on
       the FFI backend; the read that follows must see the written bytes.
  oob  the logical slab overruns the dataset extent WHILE wholly-padded
       ranks exist.  SLABIO_OOB=write|read.  The refusal must fire on
       EVERY rank; if it fires on the non-empty ranks only, the empty ranks
       enter the collective alone and the job hangs.  Bound this case with
       an external timeout.

rc=0 iff every check in the case passed.
"""
import os
import sys
import traceback

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.experimental import multihost_utils                  # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,  # noqa: E402
                                resolve_mesh)
from file_io.slab_io import SlabIO                            # noqa: E402
from gw.gw_config import SlabIOBackend                        # noqa: E402

CASE = os.environ.get("SLABIO_CASE", "rt")
OOB_KIND = os.environ.get("SLABIO_OOB", "write")
N_Q = int(os.environ.get("SLABIO_NQ", "2"))
N_G = int(os.environ.get("SLABIO_NG", "8"))
DIR = os.environ.get("SLABIO_DIR", ".")

RANK, WORLD = process_rank(), process_count()
P0 = print if RANK == 0 else (lambda *a, **k: None)

_FAILS = []


def check(label, ok, detail=""):
    """Record one verdict.  Every rank evaluates; rank 0 prints."""
    if not ok:
        _FAILS.append(f"{label}: {detail}")
    P0(f"  [{'PASS' if ok else 'FAIL'}] {label}"
       f"{('  ' + detail) if detail else ''}")
    sys.stdout.flush()


def max_over_ranks(x: float) -> float:
    """Reduce a per-rank scalar to its global max (small, cheap)."""
    g = multihost_utils.process_allgather(
        jnp.asarray([float(x)], dtype=jnp.float64), tiled=True)
    return float(np.asarray(g).max())


def host_backends():
    """(name, enum) for every backend usable in this process."""
    out = [("ffi", SlabIOBackend.PHDF5_FFI),
           ("allgather", SlabIOBackend.H5PY_ALLGATHER)]
    try:
        import mpi4py  # noqa: F401
        import h5py
        if h5py.get_config().mpi:
            out.insert(1, ("host", SlabIOBackend.PHDF5_HOST))
        else:
            P0("  [note] h5py has no MPI support -> PHDF5_HOST skipped")
    except Exception as e:                                   # noqa: BLE001
        P0(f"  [note] PHDF5_HOST unavailable ({type(e).__name__}: {e})")
    return out


def geometry():
    mu = WORLD + 1
    loc = -(-mu // WORLD)
    mu_pad = loc * WORLD
    pure_pad = [r for r in range(WORLD) if r * loc >= mu]
    return mu, loc, mu_pad, pure_pad


def reference(mu, mu_pad):
    ref = (np.arange(N_Q * mu * N_G, dtype=np.float64).reshape(N_Q, mu, N_G)
           + 1.0).astype(np.complex128)
    ref = ref + 1j * ref
    buf = np.zeros((N_Q, mu_pad, N_G), dtype=np.complex128)
    buf[:, :mu, :] = ref
    return ref, buf


def unlink(path):
    if RANK == 0 and os.path.exists(path):
        os.remove(path)
    multihost_utils.sync_global_devices(f"unlink/{os.path.basename(path)}")


# ---------------------------------------------------------------------------
def write_with(name, backend, path, A, mesh, mu, spec_desc="mu-sharded"):
    unlink(path)
    with SlabIO(path, mode="w", mesh=mesh, backend=backend) as io:
        io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                          dtype=jnp.complex128)
        io.write_slab("zeta_like", A, offset=(0, 0, 0),
                      global_shape=(N_Q, mu, N_G),
                      valid_shape=(N_Q, mu, N_G))
    multihost_utils.sync_global_devices(f"written/{name}")


def read_logical(backend, path, mesh, mu):
    """Replicated read at the LOGICAL shape — every rank gets the whole slab."""
    with SlabIO(path, mode="r", mesh=mesh, backend=backend) as io:
        arr = io.read_slab("zeta_like", shape=(N_Q, mu, N_G),
                           dtype=np.complex128, offset=(0, 0, 0),
                           mesh=mesh, partition_spec=P(None, None, None))
        return np.asarray(jax.device_get(arr))


def read_padded_sharded(backend, path, mesh, mu, mu_pad):
    """PADDED physical shape, valid_shape=logical, sharded on mu.

    This is the read that puts WHOLLY-PADDED ranks in the collective —
    the read-side twin of the write defect.  Returns this rank's local
    block and the global index of its first row.
    """
    with SlabIO(path, mode="r", mesh=mesh, backend=backend) as io:
        arr = io.read_slab("zeta_like", shape=(N_Q, mu_pad, N_G),
                           dtype=np.complex128, offset=(0, 0, 0),
                           valid_shape=(N_Q, mu, N_G),
                           mesh=mesh,
                           partition_spec=P(None, ("x", "y"), None))
        shards = arr.addressable_shards
        blk = np.asarray(shards[0].data)
        start = shards[0].index[1].start or 0
        return blk, int(start)


# ---------------------------------------------------------------------------
def case_rt(mesh):
    mu, loc, mu_pad, pure_pad = geometry()
    ref, buf = reference(mu, mu_pad)
    P0(f"[slabio] world={WORLD} mesh={tuple(mesh.devices.shape)} mu={mu} "
       f"mu_padded={mu_pad} loc={loc} pad_rows={mu_pad - mu}")
    P0(f"[slabio] WHOLLY-PADDED ranks: {pure_pad if pure_pad else 'none'}")
    if not pure_pad:
        P0("[slabio] REFUSING: no wholly-padded rank at this world size, "
           "the case would gate nothing.")
        return 2

    A = jax.device_put(jnp.asarray(buf),
                       NamedSharding(mesh, P(None, ("x", "y"), None)))
    backends = host_backends()
    P0(f"[slabio] backends under test: {[n for n, _ in backends]}")

    paths = {}
    for name, be in backends:
        path = os.path.join(DIR, f"xrt_{name}.h5")
        P0(f"\n--- WRITE with {name} -> {os.path.basename(path)}")
        write_with(name, be, path, A, mesh, mu)
        paths[name] = path
        if RANK == 0:
            import h5py
            with h5py.File(path, "r") as f:
                got = np.asarray(f["zeta_like"])
            check(f"serial-h5py readback of {name}-written file",
                  got.shape == ref.shape and not np.count_nonzero(got != ref),
                  f"shape={got.shape} maxdelta="
                  f"{float(np.abs(got - ref).max()) if got.shape == ref.shape else float('nan'):.3e}")
    multihost_utils.sync_global_devices("xrt_all_written")

    P0("\n--- CROSS-BACKEND READ MATRIX (writer x reader) ---")
    for wname in paths:
        for rname, rbe in backends:
            # (a) logical + replicated
            got = read_logical(rbe, paths[wname], mesh, mu)
            e = (float(np.abs(got - ref).max())
                 if got.shape == ref.shape else float("inf"))
            check(f"{wname:>9} -> {rname:<9} logical/replicated",
                  e == 0.0, f"max|delta|={e:.3e} shape={got.shape}")
            # (b) padded physical shape, valid_shape=logical, mu-sharded.
            #     Every rank checks its OWN block against the padded
            #     reference, including the wholly-padded ranks whose block
            #     must come back EXACTLY zero.
            blk, start = read_padded_sharded(rbe, paths[wname], mesh, mu,
                                             mu_pad)
            want = buf[:, start:start + blk.shape[1], :]
            local_e = (float(np.abs(blk - want).max())
                       if blk.shape == want.shape else float("inf"))
            e = max_over_ranks(local_e)
            check(f"{wname:>9} -> {rname:<9} padded/mu-sharded "
                  f"(wholly-padded ranks {pure_pad})",
                  e == 0.0, f"max|delta| over ranks={e:.3e}")

    # ---- replica_dup: sharding consumes only mesh axis 'x'; 'y' is a
    # replica axis, so every rank with coord_y != 0 must drop to a null
    # selection and still join the collective.  mu is padded to px here.
    px = int(mesh.shape["x"])
    mu_r = px + 1
    loc_r = -(-mu_r // px)
    mu_r_pad = loc_r * px
    ref_r, buf_r = reference(mu_r, mu_r_pad)
    P0(f"\n--- replica_dup: spec P(None,'x',None) on {px}x"
       f"{int(mesh.shape['y'])} mesh, mu={mu_r} padded={mu_r_pad}")
    Ar = jax.device_put(jnp.asarray(buf_r),
                        NamedSharding(mesh, P(None, "x", None)))
    rpath = os.path.join(DIR, "replica.h5")
    unlink(rpath)
    with SlabIO(rpath, mode="w", mesh=mesh,
                backend=SlabIOBackend.PHDF5_FFI) as io:
        io.create_dataset("zeta_like", shape=(N_Q, mu_r, N_G),
                          dtype=jnp.complex128)
        io.write_slab("zeta_like", Ar, offset=(0, 0, 0),
                      global_shape=(N_Q, mu_r, N_G),
                      valid_shape=(N_Q, mu_r, N_G))
    multihost_utils.sync_global_devices("replica_written")
    if RANK == 0:
        import h5py
        with h5py.File(rpath, "r") as f:
            got = np.asarray(f["zeta_like"])
        check("replica_dup write round-trip (FFI)",
              got.shape == ref_r.shape and not np.count_nonzero(got != ref_r),
              f"shape={got.shape}")
    multihost_utils.sync_global_devices("replica_checked")

    # ---- fill-time contract.  The FFI dcpl sets H5D_FILL_TIME_NEVER +
    # H5D_ALLOC_TIME_EARLY, so a dataset that is created and never written
    # returns whatever the filesystem hands back, NOT an HDF5-guaranteed
    # zero.  tagged_arrays.write_restart_state_to_h5(init_W0=True) relies
    # on exactly this to advertise an "all-zeros W0_qmunu placeholder".
    fpath = os.path.join(DIR, "fillnever.h5")
    unlink(fpath)
    with SlabIO(fpath, mode="w", mesh=mesh,
                backend=SlabIOBackend.PHDF5_FFI) as io:
        io.create_dataset("never_written", shape=(N_Q, mu, N_G),
                          dtype=jnp.complex128)
    multihost_utils.sync_global_devices("fillnever_written")
    if RANK == 0:
        import h5py
        with h5py.File(fpath, "r") as f:
            got = np.asarray(f["never_written"])
        nz = int(np.count_nonzero(got))
        check("created-but-never-written dataset reads back zero "
              "(FILL_TIME_NEVER; a filesystem property, not an HDF5 one)",
              nz == 0, f"{nz} nonzero of {got.size}")
    multihost_utils.sync_global_devices("fillnever_checked")
    return 0


# ---------------------------------------------------------------------------
def case_raw(mesh):
    """read_slab on a handle with writes still QUEUED on the dispatcher.

    ``write_slab`` on the FFI backend only ENQUEUES: the jitted dispatch
    runs on a Python worker thread and the H5Dwrite on the C++ writer
    thread.  The read that follows must (a) observe the written bytes and
    (b) not put a second thread inside HDF5/MPI-IO on the same file handle.
    The precondition -- a non-empty queue at the moment read_slab is
    entered -- is MEASURED here (``dispatcher.pending``) rather than
    assumed, because a read issued after the queue happens to have drained
    exercises nothing.  Writes go one q-slab at a time and the read asks
    for the LAST one, the slab most likely still in flight.
    """
    mu, loc, mu_pad, pure_pad = geometry()
    ref, buf = reference(mu, mu_pad)
    P0(f"[slabio] read-after-write on ONE handle; mu={mu} padded={mu_pad} "
       f"nG={N_G} nq={N_Q} wholly-padded ranks={pure_pad}")
    path = os.path.join(DIR, "raw.h5")
    unlink(path)
    pend_at_read = -1
    with SlabIO(path, mode="w", mesh=mesh,
                backend=SlabIOBackend.PHDF5_FFI) as io:
        io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                          dtype=jnp.complex128)
        for iq in range(N_Q):
            Aq = jax.device_put(
                jnp.asarray(buf[iq:iq + 1]),
                NamedSharding(mesh, P(None, ("x", "y"), None)))
            io.write_slab("zeta_like", Aq, offset=(iq, 0, 0),
                          global_shape=(N_Q, mu, N_G),
                          valid_shape=(1, mu, N_G))
        # NO drain, NO close.
        pend_at_read = int(io._backend._dispatcher.pending)
        P0(f"[slabio] dispatcher.pending at read_slab entry = "
           f"{pend_at_read}")
        sys.stdout.flush()
        arr = io.read_slab("zeta_like", shape=(N_Q, mu, N_G),
                           dtype=np.complex128, offset=(0, 0, 0),
                           mesh=mesh, partition_spec=P(None, None, None))
        got = np.asarray(jax.device_get(arr))
    e = float(np.abs(got - ref).max()) if got.shape == ref.shape else float("inf")
    e = max_over_ranks(e)
    check("the read-with-writes-in-flight window is reachable "
          "(dispatcher.pending > 0 at read_slab entry)",
          max_over_ranks(pend_at_read) > 0,
          f"max pending over ranks={int(max_over_ranks(pend_at_read))}")
    check("read_slab on a handle with queued writes returns the written data",
          e == 0.0, f"max|delta| over ranks={e:.3e}")
    return 0


# ---------------------------------------------------------------------------
def case_oob(mesh):
    """Logical slab overruns the dataset WHILE wholly-padded ranks exist.

    Every rank must reach the same verdict.  On a build whose bounds test
    is per-rank, only the ranks with a non-empty selection refuse; the
    wholly-padded ranks enter the collective alone and the job HANGS.
    """
    mu, loc, mu_pad, pure_pad = geometry()
    ref, buf = reference(mu, mu_pad)
    P0(f"[slabio] OOB/{OOB_KIND}: mu={mu} padded={mu_pad} "
       f"wholly-padded ranks={pure_pad}")
    if not pure_pad:
        P0("[slabio] REFUSING: no wholly-padded rank, the asymmetry cannot "
           "be exercised at this world size.")
        return 2
    path = os.path.join(DIR, f"oob_{OOB_KIND}.h5")

    if OOB_KIND == "write":
        A = jax.device_put(jnp.asarray(buf),
                           NamedSharding(mesh, P(None, ("x", "y"), None)))
        unlink(path)
        try:
            with SlabIO(path, mode="w", mesh=mesh,
                        backend=SlabIOBackend.PHDF5_FFI) as io:
                io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                                  dtype=jnp.complex128)
                # offset 1 on the mu axis: the LOGICAL slab now ends at
                # 1 + mu > mu = the dataset extent.  Ranks whose block is
                # wholly padding still compute file_count = 0.
                P0("[slabio] issuing the overrunning write "
                   f"(offset=(0,1,0) valid=(., {mu}, .) vs extent {mu}) …")
                sys.stdout.flush()
                io.write_slab("zeta_like", A, offset=(0, 1, 0),
                              global_shape=(N_Q, mu_pad, N_G),
                              valid_shape=(N_Q, mu, N_G))
        except Exception as exc:                             # noqa: BLE001
            print(f"[slabio rank={RANK}] REFUSED as expected: "
                  f"{type(exc).__name__}: {exc}", flush=True)
            check("overrunning write refuses on this rank", True,
                  "(the verdict that matters is that ALL ranks got here — "
                  "see each rank's own file)")
            return 0
        check("overrunning write refuses", False,
              "the write was ACCEPTED; the dataset extent was overrun")
        return 0

    # read
    A = jax.device_put(jnp.asarray(buf),
                       NamedSharding(mesh, P(None, ("x", "y"), None)))
    unlink(path)
    with SlabIO(path, mode="w", mesh=mesh,
                backend=SlabIOBackend.PHDF5_FFI) as io:
        io.create_dataset("zeta_like", shape=(N_Q, mu, N_G),
                          dtype=jnp.complex128)
        io.write_slab("zeta_like", A, offset=(0, 0, 0),
                      global_shape=(N_Q, mu, N_G),
                      valid_shape=(N_Q, mu, N_G))
    multihost_utils.sync_global_devices("oob_read_written")
    try:
        with SlabIO(path, mode="r", mesh=mesh,
                    backend=SlabIOBackend.PHDF5_FFI) as io:
            P0("[slabio] issuing the overrunning read "
               f"(offset=(0,1,0) valid=(., {mu}, .) vs extent {mu}) …")
            sys.stdout.flush()
            io.read_slab("zeta_like", shape=(N_Q, mu_pad, N_G),
                         dtype=np.complex128, offset=(0, 1, 0),
                         valid_shape=(N_Q, mu, N_G), mesh=mesh,
                         partition_spec=P(None, ("x", "y"), None))
    except Exception as exc:                                 # noqa: BLE001
        print(f"[slabio rank={RANK}] REFUSED as expected: "
              f"{type(exc).__name__}: {exc}", flush=True)
        check("overrunning read refuses on this rank", True,
              "(all ranks must reach here — see each rank's own file)")
        return 0
    check("overrunning read refuses", False,
          "the read was ACCEPTED past the dataset extent")
    return 0


# ---------------------------------------------------------------------------
def main():
    if WORLD < 2:
        P0("[slabio] SKIP: needs P>1")
        return 0
    mesh = resolve_mesh()
    P0(f"===== SlabIO audit probe: case={CASE} world={WORLD} =====")
    sys.stdout.flush()
    try:
        rc = {"rt": case_rt, "raw": case_raw, "oob": case_oob}[CASE](mesh)
    except KeyError:
        P0(f"[slabio] unknown SLABIO_CASE={CASE!r}")
        return 2
    except Exception:                                        # noqa: BLE001
        print(f"[slabio rank={RANK}] UNCAUGHT:", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return 1
    if rc:
        return rc
    n_bad = len(_FAILS)
    P0(f"\n[slabio] VERDICT case={CASE}: "
       f"{'PASS' if n_bad == 0 else f'FAIL ({n_bad})'}")
    for f in _FAILS:
        P0(f"    {f}")
    sys.stdout.flush()
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    finalize_process(main())
