"""P1 stability sweep of the block-cyclic distributed-linalg FFIs.

Runs every (op x dtype x size-case) cell on the CURRENT process mesh and
prints one machine-parseable line per cell:

    CELL op=<op> mesh=<PxQ> dtype=<dt> case=<name> status=<PASS|FAIL|RAISE|ERROR> \
        res=<max resid> det=<bit|DIFF|-> note=...

Launch (from sources/lorrax_D, inside lxattach'd allocation):

    LORRAX_NGPU=1 lxrun python3 -u <this file> --mesh 1x1
    lxrun python3 -u <this file> --mesh 2x2       # 4 tasks x 1 GPU
    lxrun python3 -u <this file> --mesh 4x1
    lxrun python3 -u <this file> --mesh 1x4

Ops filter: --ops mp_chol,mp_lu,mp_eigh,mg_eigh,blasmp_gemm,blasmp_wsolve,
                  slate_chol,slate_batched,slate_eigh
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_DIST = "_LORRAX_JAX_DISTRIBUTED_DONE"
if not os.environ.get(_DIST) and int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    try:
        from ffi.common.ffi_loader import get_lib
        get_lib().lrx_slate_init_mpi()
    except Exception as e:
        print(f"slate_init_mpi skipped: {e}", flush=True)
    jax.distributed.initialize(local_device_ids=[0])
    os.environ[_DIST] = "1"

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import multihost_utils


def log(s):
    if jax.process_index() == 0:
        print(s, flush=True)


def gather(x):
    if jax.process_count() == 1:
        return np.asarray(x)
    return np.asarray(multihost_utils.process_allgather(
        x, tiled=True))


def put(np_arr, mesh, spec):
    return jax.device_put(np_arr, NamedSharding(mesh, P(*spec)))


def put_rep(np_arr, mesh):
    """Fully-replicated device_put — used by the RAISE cells so the
    WRAPPER's own shape validation fires (a NamedSharding put of a
    non-divisible extent fails in device_put before the wrapper runs)."""
    return jax.device_put(np_arr, NamedSharding(mesh, P()))


def rng_complex(rng, shape):
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def make_hpd(rng, nq, n, dtype):
    if np.dtype(dtype).kind == "c":
        z = rng_complex(rng, (nq, n, n))
    else:
        z = rng.standard_normal((nq, n, n))
    a = 0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))
    a = a + n * np.eye(n)[None]
    return a.astype(dtype)


def make_herm(rng, nq, n, dtype):
    """Hermitian indefinite (no diagonal shift)."""
    if np.dtype(dtype).kind == "c":
        z = rng_complex(rng, (nq, n, n))
    else:
        z = rng.standard_normal((nq, n, n))
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))).astype(dtype)


def make_rhs(rng, shape, dtype):
    if np.dtype(dtype).kind == "c":
        return rng_complex(rng, shape).astype(dtype)
    return rng.standard_normal(shape).astype(dtype)


RESULTS = []


def cell(op, mesh_str, dtype, case, fn):
    """Run one cell; catch python exceptions; print a CELL line."""
    try:
        out = fn()  # dict: status/res/det/note
    except ValueError as e:
        out = {"status": "RAISE", "note": f"ValueError: {str(e)[:120]}"}
    except Exception as e:
        out = {"status": "ERROR",
               "note": f"{type(e).__name__}: {str(e)[:160]}"}
        if jax.process_index() == 0:
            traceback.print_exc()
    row = (f"CELL op={op} mesh={mesh_str} dtype={dtype} case={case} "
           f"status={out.get('status')} res={out.get('res', '-')} "
           f"det={out.get('det', '-')} note={out.get('note', '')}")
    log(row)
    RESULTS.append(row)


def check(res, tol, det_ok=None, note=""):
    status = "PASS" if res < tol else "FAIL"
    det = "-"
    if det_ok is not None:
        det = "bit" if det_ok else "DIFF"
        if not det_ok:
            status = "FAIL"
    return {"status": status, "res": f"{res:.3e}", "det": det, "note": note}


# ---------------------------------------------------------------- cusolvermp
def run_mp_chol(mesh, mesh_str, Px, Py):
    from ffi.cusolvermp import (batched_distributed_cholesky,
                                batched_distributed_potrs,
                                cholesky_handle_to_natural_L)

    def one(dtype, case, nq, n, mrhs):
        rng = np.random.default_rng(7)
        A_np = make_hpd(rng, nq, n, dtype)
        B_np = make_rhs(rng, (nq, n, mrhs), dtype)

        def solve():
            A = put(A_np, mesh, (None, "x", "y"))
            B = put(B_np, mesh, (None, "x", "y"))
            L = batched_distributed_cholesky(A, mesh=mesh)
            X = batched_distributed_potrs(L, B, mesh=mesh)
            Lnat = cholesky_handle_to_natural_L(L)
            return gather(X), gather(Lnat)

        X1, Ln = solve()
        X2, _ = solve()
        det_ok = np.array_equal(X1, X2)
        res_x = max(
            np.linalg.norm(A_np[q] @ X1[q] - B_np[q]) /
            max(np.linalg.norm(B_np[q]), 1.0) for q in range(nq))
        res_l = max(
            np.linalg.norm(Ln[q] @ np.conj(Ln[q].T) - A_np[q]) /
            max(np.linalg.norm(A_np[q]), 1.0) for q in range(nq))
        return check(max(res_x, res_l), 1e-11, det_ok,
                     note=f"res_x={res_x:.1e} res_L={res_l:.1e}")

    for dt in ("complex128", "float64"):
        cell("mp_chol", mesh_str, dt, "n64_rhs96",
             lambda dt=dt: one(dt, "n64_rhs96", 3, 64, 96))
    # NRHS < N on a 2D grid: the documented cuSOLVERMp 0.6.0 silent-wrong bug.
    cell("mp_chol", mesh_str, "complex128", "n64_rhs16_smallNRHS",
         lambda: one("complex128", "small_nrhs", 3, 64, 16))
    # small edge: one row per rank on the larger axis
    n_edge = 2 * max(Px, Py)
    cell("mp_chol", mesh_str, "complex128", f"edge_n{n_edge}",
         lambda: one("complex128", "edge", 2, n_edge, 2 * Py))
    # non-divisible N -> wrapper must refuse (RAISE is the pass condition)
    if max(Px, Py) > 1:
        cell("mp_chol", mesh_str, "complex128", "nondiv_n66_expectRAISE",
             lambda: one("complex128", "nondiv", 2, 66, 32))
    # non-divisible NRHS -> refuse
    if Py > 1:
        cell("mp_chol", mesh_str, "complex128", "nondiv_rhs%d_expectRAISE" % (Py + 1),
             lambda: one("complex128", "nondiv_rhs", 2, 64, Py + 1))


def run_mp_lu(mesh, mesh_str, Px, Py):
    from ffi.cusolvermp import batched_distributed_solve_lu

    def one(dtype, nq, n, nrhs, herm=True):
        rng = np.random.default_rng(11)
        if herm:
            A_np = make_herm(rng, nq, n, dtype)   # indefinite Hermitian
        else:
            A_np = make_rhs(rng, (nq, n, n), dtype) + n * np.eye(n)[None]
        B_np = make_rhs(rng, (nq, n, nrhs), dtype)

        def solve():
            A = put(A_np, mesh, (None, "x", "y"))
            B = put(B_np, mesh, (None, "x", "y"))
            return gather(batched_distributed_solve_lu(A, B, mesh=mesh))

        X1 = solve()
        X2 = solve()
        det_ok = np.array_equal(X1, X2)
        res = max(
            np.linalg.norm(A_np[q] @ X1[q] - B_np[q]) /
            max(np.linalg.norm(B_np[q]), 1.0) for q in range(nq))
        return check(res, 1e-10, det_ok)

    for dt in ("complex128", "float64"):
        cell("mp_lu", mesh_str, dt, "herm_indef_n64_rhs32",
             lambda dt=dt: one(dt, 3, 64, 32))
    cell("mp_lu", mesh_str, "complex128", "general_n64_rhs32",
         lambda: one("complex128", 3, 64, 32, herm=False))
    n_edge = 2 * max(Px, Py)
    cell("mp_lu", mesh_str, "complex128", f"edge_n{n_edge}",
         lambda: one("complex128", 2, n_edge, 2 * Py))
    if Py > 1:
        cell("mp_lu", mesh_str, "complex128", "nondiv_rhs_expectRAISE",
             lambda: one("complex128", 2, 64, Py + 1))


def _eigh_interpretations(A_np, W_np, Q_np):
    """Try Q as-is / transposed / conj / conj-T; return dict of residuals."""
    outs = {}
    for name, Qi in (("asis", Q_np), ("T", Q_np.T),
                     ("conj", np.conj(Q_np)), ("H", np.conj(Q_np.T))):
        r = (np.linalg.norm(A_np @ Qi - Qi * W_np[None, :]) /
             max(np.linalg.norm(A_np), 1.0))
        outs[name] = r
    return outs


def run_mp_eigh(mesh, mesh_str, Px, Py):
    from ffi.cusolvermp import distributed_eigh

    def one(dtype, n):
        rng = np.random.default_rng(13)
        A_np = make_herm(rng, 1, n, dtype)[0]

        def solve():
            A = put(A_np, mesh, ("x", "y"))
            W, Q = distributed_eigh(A, mesh=mesh)
            return gather(W)[:n], gather(Q)

        W1, Q1 = solve()
        W2, Q2 = solve()
        det_ok = np.array_equal(W1, W2) and np.array_equal(Q1, Q2)
        W_ref = np.linalg.eigvalsh(A_np)
        ev_err = float(np.max(np.abs(W1 - W_ref)))
        interp = _eigh_interpretations(A_np, W1, Q1)
        best = min(interp, key=interp.get)
        note = (f"ev={ev_err:.1e} Q[asis]={interp['asis']:.1e} "
                f"Q[T]={interp['T']:.1e} Q[conj]={interp['conj']:.1e} "
                f"Q[H]={interp['H']:.1e} best={best}")
        # PASS criterion: eigenvalues only (documented contract today).
        return check(ev_err, 1e-9, det_ok, note=note)

    for dt in ("complex128", "float64"):
        cell("mp_eigh", mesh_str, dt, "n64",
             lambda dt=dt: one(dt, 64))


def run_mg_eigh(mesh, mesh_str, Px, Py):
    if jax.process_count() != 1:
        return
    from ffi.cusolvermg import eigh_mg

    def one(n, tile):
        rng = np.random.default_rng(17)
        A_np = make_herm(rng, 1, n, "float64")[0]

        def solve():
            A = jax.device_put(A_np)
            W, Q = eigh_mg(A, tile_size=tile)
            return np.asarray(W), np.asarray(Q)

        W1, Q1 = solve()
        W2, Q2 = solve()
        det_ok = np.array_equal(W1, W2) and np.array_equal(Q1, Q2)
        W_ref = np.linalg.eigvalsh(A_np)
        ev_err = float(np.max(np.abs(W1 - W_ref)))
        interp = _eigh_interpretations(A_np, W1, Q1)
        note = (f"ev={ev_err:.1e} Q[asis]={interp['asis']:.1e} "
                f"Q[T]={interp['T']:.1e}")
        return check(ev_err, 1e-9, det_ok, note=note)

    cell("mg_eigh", mesh_str, "float64", "n64_tile32", lambda: one(64, 32))
    cell("mg_eigh", mesh_str, "float64", "n100_tile32", lambda: one(100, 32))


# ----------------------------------------------------------------- cublasmp
def run_blasmp_gemm(mesh, mesh_str, Px, Py):
    from ffi.cublasmp import batched_distributed_gemm

    def one(dtype, transa, transb, alpha, beta):
        rng = np.random.default_rng(19)
        nq, m, k, n = 2, 64, 32, 48
        # op(A)=(m,k), op(B)=(k,n)
        A_shape = (nq, m, k) if transa == "N" else (nq, k, m)
        B_shape = (nq, k, n) if transb == "N" else (nq, n, k)
        A_np = make_rhs(rng, A_shape, dtype)
        B_np = make_rhs(rng, B_shape, dtype)
        C_np = make_rhs(rng, (nq, m, n), dtype)

        def opx(X, t):
            if t == "N":
                return X
            if t == "T":
                return np.swapaxes(X, -1, -2)
            return np.conj(np.swapaxes(X, -1, -2))

        def solve():
            A = put(A_np, mesh, (None, "x", "y"))
            B = put(B_np, mesh, (None, "x", "y"))
            C = put(C_np, mesh, (None, "x", "y"))
            return gather(batched_distributed_gemm(
                A, B, C, mesh=mesh, alpha=alpha, beta=beta,
                transa=transa, transb=transb))

        D1 = solve()
        D2 = solve()
        det_ok = np.array_equal(D1, D2)
        D_ref = alpha * np.einsum("qij,qjk->qik", opx(A_np, transa),
                                  opx(B_np, transb)) + beta * C_np
        res = (np.linalg.norm(D1 - D_ref) /
               max(np.linalg.norm(D_ref), 1.0))
        return check(res, 1e-12, det_ok)

    for (ta, tb) in (("N", "N"), ("T", "N"), ("C", "N"), ("N", "C")):
        cell("blasmp_gemm", mesh_str, "complex128", f"{ta}{tb}",
             lambda ta=ta, tb=tb: one("complex128", ta, tb,
                                      1.3 - 0.7j, 0.4 + 0.2j))
    cell("blasmp_gemm", mesh_str, "float64", "TN",
         lambda: one("float64", "T", "N", 1.5, 0.5))


def run_blasmp_wsolve(mesh, mesh_str, Px, Py):
    from ffi.cublasmp import batched_fused_w_solve

    def one(dtype, nq, n, pref):
        rng = np.random.default_rng(23)
        V_np = make_hpd(rng, nq, n, dtype)
        # chi Hermitian negative semidefinite: -(C C^H), scaled small
        C = make_rhs(rng, (nq, n, n), dtype)
        chi_np = (-(C @ np.conj(np.swapaxes(C, -1, -2))) / n).astype(dtype)

        def solve():
            V = put(V_np, mesh, (None, "x", "y"))
            chi = put(chi_np, mesh, (None, "x", "y"))
            return gather(batched_fused_w_solve(V, chi, pref, mesh=mesh))

        W1 = solve()
        W2 = solve()
        det_ok = np.array_equal(W1, W2)
        eye = np.eye(n)
        res = 0.0
        for q in range(nq):
            W_ref = np.linalg.solve(eye - pref * V_np[q] @ chi_np[q], V_np[q])
            res = max(res, np.linalg.norm(W1[q] - W_ref) /
                      max(np.linalg.norm(W_ref), 1.0))
        return check(res, 1e-11, det_ok)

    cell("blasmp_wsolve", mesh_str, "complex128", "n64",
         lambda: one("complex128", 2, 64, 0.37))
    cell("blasmp_wsolve", mesh_str, "float64", "n64",
         lambda: one("float64", 2, 64, 0.37))


# -------------------------------------------------------------------- slate
def run_slate_chol(mesh, mesh_str, Px, Py):
    from ffi.slate import distributed_cholesky, distributed_trsm

    def one(dtype, n, m):
        rng = np.random.default_rng(29)
        A_np = make_hpd(rng, 1, n, dtype)[0]
        B_np = make_rhs(rng, (n, m), dtype)

        def solve():
            A = put(A_np, mesh, ("x", "y"))
            B = put(B_np, mesh, ("x", "y"))
            L = distributed_cholesky(A, mesh=mesh)
            Xf = distributed_trsm(L, B, mesh=mesh, op="N")
            Xa = distributed_trsm(L, B, mesh=mesh, op="C")
            return gather(L.to_jax_lower()), gather(Xf), gather(Xa)

        L1, Xf1, Xa1 = solve()
        L2, Xf2, Xa2 = solve()
        det_ok = (np.array_equal(L1, L2) and np.array_equal(Xf1, Xf2)
                  and np.array_equal(Xa1, Xa2))
        nrmA = max(np.linalg.norm(A_np), 1.0)
        nrmB = max(np.linalg.norm(B_np), 1.0)
        res_l = np.linalg.norm(L1 @ np.conj(L1.T) - A_np) / nrmA
        res_f = np.linalg.norm(L1 @ Xf1 - B_np) / nrmB
        res_a = np.linalg.norm(np.conj(L1.T) @ Xa1 - B_np) / nrmB
        return check(max(res_l, res_f, res_a), 1e-12, det_ok,
                     note=f"L={res_l:.1e} fwd={res_f:.1e} adj={res_a:.1e}")

    for dt in ("complex128", "float64"):
        cell("slate_chol", mesh_str, dt, "n64_m64",
             lambda dt=dt: one(dt, 64, 64))
    cell("slate_chol", mesh_str, "complex128", "n64_m32",
         lambda: one("complex128", 64, 32))
    n_edge = 2 * max(Px, Py)
    cell("slate_chol", mesh_str, "complex128", f"edge_n{n_edge}",
         lambda: one("complex128", n_edge, n_edge))
    if max(Px, Py) > 1:
        cell("slate_chol", mesh_str, "complex128", "nondiv_n66_expectRAISE",
             lambda: one("complex128", 66, 64))


def run_slate_batched(mesh, mesh_str, Px, Py):
    from ffi.slate import (batched_distributed_cholesky,
                           batched_distributed_trsm)

    def one(dtype, nbatch, n, nrhs):
        rng = np.random.default_rng(31)
        A_np = make_hpd(rng, nbatch, n, dtype)
        B_np = make_rhs(rng, (nbatch, n, nrhs), dtype)

        def solve():
            A = put(A_np, mesh, ("x", None, "y"))
            B = put(B_np, mesh, ("x", None, "y"))
            L = batched_distributed_cholesky(A, mesh=mesh)
            X = batched_distributed_trsm(L, B, mesh=mesh, op="N")
            return gather(X)

        X1 = solve()
        X2 = solve()
        det_ok = np.array_equal(X1, X2)
        # forward solve residual: L X = B with L = chol(A)
        res = 0.0
        for qq in range(nbatch):
            L_ref = np.linalg.cholesky(A_np[qq])
            res = max(res, np.linalg.norm(L_ref @ X1[qq] - B_np[qq]) /
                      max(np.linalg.norm(B_np[qq]), 1.0))
        return check(res, 1e-12, det_ok)

    for dt in ("complex128", "float64"):
        cell("slate_batched", mesh_str, dt, "nb4_n32_rhs32",
             lambda dt=dt: one(dt, 4, 32, 32))


def run_slate_eigh(mesh, mesh_str, Px, Py):
    if Px != Py:
        return
    from ffi.slate import distributed_eigh

    def one(dtype, n, kind):
        rng = np.random.default_rng(37)
        if kind == "diag":
            A_np = np.diag(np.arange(1.0, n + 1)).astype(dtype)
        else:
            A_np = make_herm(rng, 1, n, dtype)[0]

        def solve():
            A = put(A_np, mesh, ("x", "y"))
            W, Q = distributed_eigh(A, mesh=mesh)
            return gather(W)[:n], gather(Q)

        W1, Q1 = solve()
        W2, Q2 = solve()
        det_ok = np.array_equal(W1, W2) and np.array_equal(Q1, Q2)
        W_ref = np.linalg.eigvalsh(A_np)
        ev_err = float(np.max(np.abs(W1 - W_ref)))
        interp = _eigh_interpretations(A_np, W1, Q1)
        best = min(interp, key=interp.get)
        note = (f"ev={ev_err:.1e} Q[asis]={interp['asis']:.1e} "
                f"Q[T]={interp['T']:.1e} Q[conj]={interp['conj']:.1e} "
                f"Q[H]={interp['H']:.1e} best={best}")
        return check(ev_err, 1e-9, det_ok, note=note)

    for dt in ("complex128", "float64"):
        cell("slate_eigh", mesh_str, dt, "n64_random",
             lambda dt=dt: one(dt, 64, "random"))
    cell("slate_eigh", mesh_str, "complex128", "n64_diag",
         lambda: one("complex128", 64, "diag"))


OPS = {
    "mp_chol": run_mp_chol,
    "mp_lu": run_mp_lu,
    "mp_eigh": run_mp_eigh,
    "mg_eigh": run_mg_eigh,
    "blasmp_gemm": run_blasmp_gemm,
    "blasmp_wsolve": run_blasmp_wsolve,
    "slate_chol": run_slate_chol,
    "slate_batched": run_slate_batched,
    "slate_eigh": run_slate_eigh,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=None, help="PxQ, e.g. 2x2")
    ap.add_argument("--ops", default=",".join(OPS))
    args = ap.parse_args()

    world = jax.process_count()
    if args.mesh:
        Px, Py = (int(v) for v in args.mesh.lower().split("x"))
    else:
        Px, Py = world, 1
    if Px * Py != world:
        log(f"mesh {Px}x{Py} != world {world}; abort")
        return 2
    mesh = Mesh(np.asarray(jax.devices()).reshape(Px, Py), ("x", "y"))
    mesh_str = f"{Px}x{Py}"
    log(f"== sweep mesh={mesh_str} world={world} "
        f"devices={len(jax.devices())} ==")

    for name in args.ops.split(","):
        name = name.strip()
        if not name:
            continue
        OPS[name](mesh, mesh_str, Px, Py)

    log(f"== done: {len(RESULTS)} cells ==")
    return 0


if __name__ == "__main__":
    sys.exit(main())
