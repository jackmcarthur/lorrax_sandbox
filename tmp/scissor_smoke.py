"""Smoke test for src/gw/scissor.py.

Exercises:
1. fit_scissor on synthetic (E, ΔE) with known v/c slopes+intercepts.
2. extrapolate_delta_e: in-grid passthrough + out-of-grid fit substitution.
3. add_diag_to_H_kmn on a (None, 'x', 'y')-sharded Hamiltonian on 4 GPUs.

Run under lxrun so JAX sees 4 A100s; exits non-zero on any mismatch.
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import sys
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# Initialize distributed if launched multi-process (srun -n >1).
def _maybe_init_dist():
    proc_count = int(os.environ.get("SLURM_NTASKS", "1"))
    if proc_count > 1:
        jax.distributed.initialize()

_maybe_init_dist()

from gw.scissor import (
    ScissorFit,
    fit_scissor,
    extrapolate_delta_e,
    add_diag_to_H_kmn,
)


def _build_mesh():
    devs = np.asarray(jax.devices())
    total = devs.size
    gx = int(np.sqrt(total))
    while gx > 1 and total % gx != 0:
        gx -= 1
    return Mesh(devs.reshape(gx, total // gx), ('x', 'y'))


def _assert_close(name, a, b, atol):
    a = np.asarray(a)
    b = np.asarray(b)
    err = float(np.max(np.abs(a - b)))
    ok = err < atol
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: maxabs err = {err:.3e} (tol {atol:.0e})", flush=True)
    return ok


def test_fit_and_predict():
    print("== test_fit_and_predict ==", flush=True)
    rng = np.random.default_rng(0)
    nk, nb = 4, 20
    nocc = 8

    alpha_v, beta_v = 0.10, -1.5
    alpha_c, beta_c = 0.05, +0.7

    # Synthetic energies: valence in [-8, 0], conduction in [0, 12], noisy.
    E = np.zeros((nk, nb), dtype=np.float64)
    E[:, :nocc] = rng.uniform(-8.0, 0.0, size=(nk, nocc))
    E[:, nocc:] = rng.uniform(0.0, 12.0, size=(nk, nb - nocc))

    vmask = np.zeros((nk, nb), dtype=bool)
    vmask[:, :nocc] = True

    noise = 1.0e-4 * rng.standard_normal((nk, nb))
    dE = np.where(vmask, alpha_v * E + beta_v, alpha_c * E + beta_c) + noise

    # Use ALL points as fit points.
    fit = fit_scissor(E, dE, vmask, np.ones_like(vmask, dtype=bool))
    print(f"  {fit.summary()}", flush=True)
    ok = True
    ok &= _assert_close("slope_v",     fit.slope_v,     alpha_v, 1e-3)
    ok &= _assert_close("intercept_v", fit.intercept_v, beta_v,  1e-3)
    ok &= _assert_close("slope_c",     fit.slope_c,     alpha_c, 1e-3)
    ok &= _assert_close("intercept_c", fit.intercept_c, beta_c,  1e-3)
    ok &= _assert_close("n_fit_v", fit.n_fit_v, nk * nocc,          0.5)
    ok &= _assert_close("n_fit_c", fit.n_fit_c, nk * (nb - nocc),   0.5)

    # predict() should reproduce the noiseless law.
    pred = fit.predict(E, vmask)
    expected = np.where(vmask, alpha_v * E + beta_v, alpha_c * E + beta_c)
    ok &= _assert_close("predict", pred, expected, 5e-3)
    return ok


def test_extrapolate_in_grid_passthrough():
    print("== test_extrapolate_in_grid_passthrough ==", flush=True)
    rng = np.random.default_rng(1)
    nk, nb = 3, 10
    E = rng.uniform(-10.0, 10.0, size=(nk, nb))
    dE_measured = rng.standard_normal((nk, nb))
    vmask = np.array([True] * 5 + [False] * 5, dtype=bool)[None, :].repeat(nk, axis=0)
    in_grid = np.abs(E) < 5.0  # mimic a ±5 eV Σ(ω) grid

    # Fit from in-grid only — doesn't matter for this test; we use a fit with
    # slope=0, intercept=99 so the substituted values are unambiguous.
    fit = ScissorFit(
        slope_v=0.0, intercept_v=-99.0,
        slope_c=0.0, intercept_c=+99.0,
        n_fit_v=1, n_fit_c=1, rmse_v_ev=0.0, rmse_c_ev=0.0,
    )
    out = extrapolate_delta_e(E, dE_measured, vmask, in_grid, fit)

    # In-grid points keep their measured ΔE.
    ok = _assert_close("in_grid passthrough", out[in_grid], dE_measured[in_grid], 1e-14)
    # Out-of-grid: valence → -99, conduction → +99.
    expect_out = np.where(vmask, -99.0, +99.0)
    ok &= _assert_close("out_of_grid substitution",
                        out[~in_grid], expect_out[~in_grid], 1e-14)
    return ok


def test_add_diag_sharded():
    print("== test_add_diag_sharded ==", flush=True)
    mesh = _build_mesh()
    gx, gy = mesh.devices.shape
    print(f"  mesh shape = ({gx}, {gy}) on {jax.device_count()} devices", flush=True)

    # Pick nb divisible by both gx and gy, small enough that we can compare to
    # a replicated reference.
    nk, nb = 3, 4 * gx * gy
    rng = np.random.default_rng(2)
    H = (rng.standard_normal((nk, nb, nb)) +
         1j * rng.standard_normal((nk, nb, nb))).astype(np.complex128)
    d = (rng.standard_normal((nk, nb)) +
         1j * rng.standard_normal((nk, nb))).astype(np.complex128)

    # Reference: add to diagonal via numpy.
    ref = H.copy()
    idx = np.arange(nb)
    ref[:, idx, idx] += d

    # Shard H as P(None, 'x', 'y'); d replicated.
    H_j = jax.device_put(jnp.asarray(H), NamedSharding(mesh, P(None, 'x', 'y')))
    with mesh:
        H_out = add_diag_to_H_kmn(H_j, d, mesh)
    H_out_np = np.asarray(
        jax.experimental.multihost_utils.process_allgather(H_out, tiled=False)
    )

    ok = _assert_close("sharded diag add", H_out_np, ref, 1e-12)

    # Check sharding preserved.
    shard_spec = H_out.sharding
    print(f"  output sharding: {shard_spec}", flush=True)
    return ok


def test_add_diag_divisibility_error():
    print("== test_add_diag_divisibility_error ==", flush=True)
    mesh = _build_mesh()
    gx, gy = mesh.devices.shape
    if gx * gy == 1:
        print("  SKIP (single device; every nb is divisible)", flush=True)
        return True
    nk, nb = 2, gx * gy + 1  # intentionally not divisible
    H_j = jnp.zeros((nk, nb, nb), dtype=jnp.complex128)
    d = np.zeros((nk, nb), dtype=np.complex128)
    try:
        add_diag_to_H_kmn(H_j, d, mesh)
    except ValueError as e:
        msg = str(e)
        ok = "divisible" in msg
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] raised ValueError: {msg}", flush=True)
        return ok
    print("  [FAIL] expected ValueError; none raised", flush=True)
    return False


def main():
    results = []
    results.append(("fit_and_predict", test_fit_and_predict()))
    results.append(("extrapolate_in_grid_passthrough", test_extrapolate_in_grid_passthrough()))
    results.append(("add_diag_sharded", test_add_diag_sharded()))
    results.append(("add_diag_divisibility_error", test_add_diag_divisibility_error()))

    print("=" * 60, flush=True)
    all_ok = True
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}", flush=True)
        all_ok = all_ok and ok
    print("=" * 60, flush=True)
    if jax.process_index() == 0:
        print(f"OVERALL: {'PASS' if all_ok else 'FAIL'}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
