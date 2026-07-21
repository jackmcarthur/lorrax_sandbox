"""Stage 4(a): is the HGL crossing quadrature still stressed at 80 Ry / n_mu=2412?

The xi-floor (gw/ppm_windows.py, _CROSSING_A_MAX = 24) exists because the
crossing-window minimax sin-fit becomes ill-conditioned at large dimensionless
bandwidth A_core = 2*omega_max/xi + 2*edge: the weight sum Sigma|alpha_hat|
jumps from O(1) at A<=24 to O(1e4-1e5), and those near-cancelling weights
amplify ANY perturbation of the per-tau operand sigma(tau) -- which carries the
screened W and is mesh-sensitive.  The floor raises xi so A_core <= 24.

Two things are worth separating, and the report needs both:

  (1) The CONDITIONING of the quadrature, Sigma|alpha_hat|(A).  This is a
      property of the minimax fit ALONE -- it depends only on A_core, i.e. on
      the Sigma omega-grid and edge_factor.  It does NOT depend on the cutoff,
      n_mu, or the ISDF at all.  So it cannot have improved at 80 Ry, and this
      script measures it to say so with numbers instead of asserting it.

  (2) The AMPLIFIED QUANTITY, i.e. how big the perturbation of sigma(tau) that
      those weights multiply actually is.  THAT is what the converged run can
      have improved (rank-truncated CCT + band-range centroids -> better
      conditioned W).  Measuring it needs the GW itself; this script only sizes
      the amplifier.

Run under the LORRAX shifter env (needs gw.minimax_screening).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get(
    "LORRAX_SRC",
    "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gw_converged/src"))

from gw.ppm_windows import _CROSSING_A_MAX, crossing_regularization_floor  # noqa
from gw.minimax_screening import solve_phase_minimax_bandwidth            # noqa

RYD_TO_EV = 13.6056980659

# The production Sigma grid for this campaign (cohsex.in).
OMEGA_MAX_EV = float(os.environ.get("XI_OMEGA_MAX_EV", "10.0"))
EDGE = float(os.environ.get("XI_EDGE", "1.5"))       # sigma_window_edge_factor
TARGET_ERR = float(os.environ.get("XI_TARGET_ERR", "1e-6"))
MAX_NODES = int(os.environ.get("XI_MAX_NODES", "500"))
EPS_Q = float(os.environ.get("XI_EPS_Q", "1e-3"))

omega_max_ry = OMEGA_MAX_EV / RYD_TO_EV
xi_floor_ry = crossing_regularization_floor(omega_max_ry, EDGE)

print(f"Sigma omega grid   : +/- {OMEGA_MAX_EV} eV  "
      f"(omega_max = {omega_max_ry:.6f} Ry)")
print(f"edge_factor        : {EDGE}")
print(f"_CROSSING_A_MAX    : {_CROSSING_A_MAX}")
print(f"xi floor           : {xi_floor_ry:.6f} Ry = "
      f"{xi_floor_ry*RYD_TO_EV:.4f} eV")
print(f"requested xi (in)  : 0.25 eV -> floored to "
      f"{max(0.25, xi_floor_ry*RYD_TO_EV):.4f} eV")
print()
print(f"{'xi (eV)':>9} {'xi (Ry)':>10} {'A_core':>9} {'n_tau':>6} "
      f"{'sum|alpha|':>13} {'fit err':>11}  regime")
print("-" * 76)

rows = []
_XI_LIST = os.environ.get("XI_LIST", "0.25,0.50,0.75,1.00,1.50,2.00")
for xi_ev in sorted(set([float(x) for x in _XI_LIST.split(",")]
                        + [round(xi_floor_ry * RYD_TO_EV, 6)])):
    xi_ry = xi_ev / RYD_TO_EV
    A = 2.0 * omega_max_ry / xi_ry + 2.0 * EDGE
    try:
        q = solve_phase_minimax_bandwidth(
            A, target_error=TARGET_ERR, max_nodes=MAX_NODES, eps_q=EPS_Q,
            target_kind="hgl", use_shipped_tables=True)
        s = float(np.abs(np.asarray(q.alpha)).sum())
        n = int(q.node_count)
        err = float(q.max_error)
    except Exception as exc:                          # pragma: no cover
        print(f"{xi_ev:9.3f} {xi_ry:10.5f} {A:9.2f}  solver failed: {exc}")
        continue
    tag = ("OK" if s < 10 else "STRESSED" if s < 1e3 else "ILL-CONDITIONED")
    at_floor = " <- xi floor" if abs(xi_ev - xi_floor_ry * RYD_TO_EV) < 1e-9 else ""
    print(f"{xi_ev:9.3f} {xi_ry:10.5f} {A:9.2f} {n:6d} {s:13.4e} "
          f"{err:11.3e}  {tag}{at_floor}")
    rows.append((xi_ev, A, n, s, err))

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "xi_probe.npz"),
         rows=np.array(rows), omega_max_ev=OMEGA_MAX_EV, edge=EDGE,
         xi_floor_ev=xi_floor_ry * RYD_TO_EV, a_max=_CROSSING_A_MAX)
print()
print("Sigma|alpha_hat| is the amplification factor applied to any perturbation")
print("of the per-tau operand sigma(tau).  A_core depends ONLY on (omega_max,")
print("xi, edge_factor) -- not on ecutwfc, n_mu, or the ISDF -- so this table is")
print("identical at 30 Ry and 80 Ry.  What convergence can change is the SIZE of")
print("the perturbation being amplified, not the amplifier.")
