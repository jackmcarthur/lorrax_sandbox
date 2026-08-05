"""Fast 16-GPU distributed probe: validate init + host->sharded placement.

Run via run11.sh; every process executes this.  Answers, from REAL logs:
  * jax.distributed init worked (device_count==16, process_count==?, mesh 4x4)
  * whether jax.device_put(host_numpy, sharded NamedSharding) is
    multi-process correct in JAX 25.04 (the exciton driver's vq_interp
    setup uses this pattern), by gathering back and byte-comparing
  * device_put(jnp.asarray(...)) single-device -> sharded
  * replicated P() device_put + device_get round-trip (rank0 output path)
"""
from runtime import set_default_env
set_default_env()

import numpy as np
import jax
import jax.numpy as jnp

from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend
init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()
jax.config.update("jax_enable_x64", True)

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils

RANK = jax.process_index()


def log(*a):
    if RANK == 0:
        print(*a, flush=True)


log(f"[probe] device_count={jax.device_count()} process_count={jax.process_count()} "
    f"local_device_count={jax.local_device_count()}")

from bse.bse_w_exact import _create_mesh_xy
mesh = _create_mesh_xy(4, 4)
log(f"[probe] _create_mesh_xy(4,4).shape={dict(mesh.shape)} axes={mesh.axis_names}")


def test(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        log(f"[probe] {name}: FAILED {type(e).__name__}: {e}")


# --- (a) host numpy -> P('x','y') sharded (V_q0 / eval_vq tile pattern) ---
def _t_xy():
    A = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)  # identical all procs
    Ad = jax.device_put(A, NamedSharding(mesh, P("x", "y")))
    Ag = np.asarray(multihost_utils.process_allgather(Ad, tiled=True))
    log(f"[probe] device_put(numpy, P('x','y')): OK "
        f"local_shards={[s.data.shape for s in Ad.addressable_shards]} "
        f"gather_match={bool(np.allclose(Ag, A))}")
test("device_put(numpy,P('x','y'))", _t_xy)


# --- (b) host numpy -> P(('x','y'),None,None) (prepare_coarse eigh chunk) ---
def _t_qbatch():
    B = (np.arange(48 * 4 * 4) + 1j).astype(np.complex128).reshape(48, 4, 4)
    Bd = jax.device_put(B, NamedSharding(mesh, P(("x", "y"), None, None)))
    Bg = np.asarray(multihost_utils.process_allgather(Bd, tiled=True))
    log(f"[probe] device_put(numpy, P(('x','y'),None,None)): OK "
        f"gather_match={bool(np.allclose(Bg, B))}")
test("device_put(numpy,P(('x','y'),..))", _t_qbatch)


# --- (c) jnp.asarray (single-device jax.Array) -> sharded ---
def _t_jnp():
    A = np.arange(16 * 16, dtype=np.float64).reshape(16, 16)
    Cd = jax.device_put(jnp.asarray(A), NamedSharding(mesh, P("x", "y")))
    Cg = np.asarray(multihost_utils.process_allgather(Cd, tiled=True))
    log(f"[probe] device_put(jnp.asarray(A), P('x','y')): OK "
        f"gather_match={bool(np.allclose(Cg, A))}")
test("device_put(jnp.asarray,P('x','y'))", _t_jnp)


# --- (d) replicated P() device_put + device_get (the rank0 .dat gather) ---
def _t_rep():
    A = np.arange(40 * 8, dtype=np.float64).reshape(40, 8)  # evs_all shape-ish
    Rd = jax.device_put(A, NamedSharding(mesh, P()))
    Rg = np.asarray(jax.device_get(Rd))  # replicated -> full on every proc
    log(f"[probe] device_put(numpy,P())+device_get: OK match={bool(np.allclose(Rg, A))}")
test("replicated device_get", _t_rep)


# --- (e) jnp.stack(list of global sharded) + device_put (V_stack pattern) ---
def _t_stack():
    tiles = []
    for i in range(3):
        t = jax.device_put(np.full((16, 16), float(i)),
                           NamedSharding(mesh, P("x", "y")))
        tiles.append(t)
    S = jax.device_put(jnp.stack(tiles),
                       NamedSharding(mesh, P(None, "x", "y")))
    Sg = np.asarray(multihost_utils.process_allgather(S, tiled=True))
    ok = all(np.allclose(Sg[i], float(i)) for i in range(3))
    log(f"[probe] jnp.stack(global tiles)+device_put(P(None,'x','y')): OK match={ok}")
test("jnp.stack+device_put", _t_stack)


# --- (f) small jit eigh -> replicated evals (block_lanczos final eigh) ---
def _t_eigh():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((32, 32))
    H = jnp.asarray(X + X.T)
    ev = jax.jit(jnp.linalg.eigvalsh)(H)
    log(f"[probe] jit eigvalsh sharding={ev.sharding} fully_addressable={ev.is_fully_addressable}")
test("jit eigvalsh replicated", _t_eigh)

log("[probe] PROBE DONE")
