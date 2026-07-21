"""Stage-3/4 experiment driver: gw.gw_jax with named internals rebound in-process.

ONE driver for every knob this campaign has to move that is a module constant
rather than a cohsex.in key.  Nothing here edits shipped source -- each knob is
rebound on the imported module, so the worktree stays clean and every other run
in it is unaffected.

  --cap-gib G   raise `isdf.core._REPLICATED_CHOL_MAX_STACK_BYTES` (default 4 GiB).
                ONLY the replicated route carries the rank-truncation cure
                (23af6b9) and the mesh-invariant replicated factor (ca78008):

                    if _replicate_charge_ok(nq, n_rmu):
                        return ('replicated_rank_truncate' if ... else
                                'replicated_cholesky')
                    return 'cusolvermp_cholesky' if is_2d else 'sharded_cholesky'

                and the cap is `nq * n_rmu**2 * 16 <= 4 GiB`.  The converged
                campaign is ABOVE it (nq=74, n_rmu=2416 -> 6.44 GiB), so both
                cures silently disengage -- the code's own comment says "above
                the replication cap it silently uses the distributed Cholesky".
                Raising the cap restores the intended default route.

  --lift-xi     disable the Sigma_c crossing-quadrature xi floor by setting
                `_CROSSING_A_MAX` to a huge value in BOTH `gw.ppm_windows` and
                `gw.ppm_sigma` (ppm_sigma does `from .ppm_windows import
                _CROSSING_A_MAX, crossing_regularization_floor` at import time,
                so it holds its own references and both must be rebound).
                The input's `sigma_regularization_ev` is then used as written.

usage: python3 -u gw_probe.py -i cohsex.in [--cap-gib 8] [--lift-xi] [--a-max 1e9]
"""
import sys

CAP_GIB = None
LIFT_XI = False
A_MAX = 1.0e9
argv = []
it = iter(sys.argv[1:])
for a in it:
    if a == "--cap-gib":
        CAP_GIB = float(next(it))
    elif a == "--lift-xi":
        LIFT_XI = True
    elif a == "--a-max":
        A_MAX = float(next(it))
    else:
        argv.append(a)

# Import gw.gw_jax FIRST: its module body runs `set_default_env()` before
# `import jax` and `init_jax_distributed()` before any backend call.  Importing
# isdf.core or gw.ppm_windows ahead of it initialises the XLA backend too early
# ("jax.distributed.initialize() must be called before ... jax.devices").
from gw.gw_jax import main                          # noqa: E402

# Patch AFTER the import.  Every site below reads its module global at CALL
# time, so rebinding now takes effect for the whole run.
if CAP_GIB is not None:
    from isdf import core as isdf_core              # noqa: E402
    _old = isdf_core._REPLICATED_CHOL_MAX_STACK_BYTES
    isdf_core._REPLICATED_CHOL_MAX_STACK_BYTES = int(CAP_GIB * 1024 ** 3)
    print(f"[probe] _REPLICATED_CHOL_MAX_STACK_BYTES "
          f"{_old / 1024**3:.2f} -> {CAP_GIB:.2f} GiB; the charge zeta-solve "
          f"now takes the replicated route and honours charge_zeta_solve.",
          flush=True)

if LIFT_XI:
    from gw import ppm_windows, ppm_sigma           # noqa: E402

    def _no_floor(omega_max_ry: float, edge_factor: float) -> float:
        """xi floor with the lifted ceiling (identical formula, A_max huge)."""
        denom = A_MAX - 2.0 * float(edge_factor)
        if denom <= 1.0 or float(omega_max_ry) <= 0.0:
            return 0.0
        return 2.0 * float(omega_max_ry) / denom

    _old_a = ppm_windows._CROSSING_A_MAX
    ppm_windows._CROSSING_A_MAX = A_MAX
    ppm_sigma._CROSSING_A_MAX = A_MAX
    ppm_windows.crossing_regularization_floor = _no_floor
    ppm_sigma.crossing_regularization_floor = _no_floor
    print(f"[probe] _CROSSING_A_MAX {_old_a} -> {A_MAX} in ppm_windows AND "
          f"ppm_sigma (+ crossing_regularization_floor rebound in both); "
          f"sigma_regularization_ev is now used as written.", flush=True)

sys.argv = [sys.argv[0]] + argv
raise SystemExit(main())
