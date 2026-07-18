#!/usr/bin/env python3
"""Golden-gate subset with zeta_ridge_eps ON — record deltas vs the
FROZEN references (no re-freezing).

Cases (the non-bispinor Tier-1 pins; bispinor+ridge is rejected at
config parse by design):
  * cohsex       (2D WFNsmall COHSEX, frozen atol 1e-6)
  * si_cohsex_3d (bulk Si COHSEX, BGW-anchored, frozen atol 1e-3 eV)
  * gnppm        (MoS2 3×3 GN-PPM, frozen atol 1e-6)

For each case × ε_rel ∈ {1e-4, 1e-5, 1e-6}: copy fixture → append the
knob → run gw.gw_jax subprocess → parse output + frozen ref with the
harness's own parse_eqp_rows → report per-column max|Δ| (eV) and
whether the gate's frozen atol would PASS/FAIL.  Usage:
    python3 golden_gates_ridge_on.py <workdir>
"""
import json
import sys
from pathlib import Path

import numpy as np

WT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_ridge_wt")
sys.path.insert(0, str(WT / "tests"))
sys.path.insert(0, str(WT / "src"))
from harness import REG, copy_fixture, parse_eqp_rows, run_gw_jax  # noqa: E402

CASES = [
    ("cohsex", REG / "cohsex_debug", "cohsex_test.in", "eqp_test.dat",
     "eqp_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-6),
    ("si_cohsex_3d", REG / "si_cohsex_debug", "cohsex_si_test.in",
     "eqp_si_test.dat", "eqp_si_ref.dat", ("sigSX", "sigCOH", "sigTOT"), 1e-3),
    ("gnppm", REG / "gnppm_debug", "gnppm_test.in",
     "sigma_diag_gnppm_test.dat", "sigma_diag_gnppm_ref.dat",
     ("sigX", "sigC", "sigXC"), 1e-6),
]
EPS = ["1e-4", "1e-5", "1e-6"]


def main():
    workdir = Path(sys.argv[1])
    results = {}
    for case_id, case_dir, input_name, out_name, ref_name, labels, atol in CASES:
        ref_rows = parse_eqp_rows(case_dir / ref_name, labels)
        for eps in EPS:
            run_dir = workdir / f"{case_id}_ridge{eps.replace('-', 'm')}"
            copy_fixture(case_dir, run_dir)
            with open(run_dir / input_name, "a") as f:
                f.write(f"\nzeta_ridge_eps = {eps}\n")
            res = run_gw_jax(run_dir, input_name)
            if res.returncode != 0:
                results[f"{case_id}@{eps}"] = {
                    "status": "RUN_FAILED",
                    "tail": res.stdout[-2000:] + res.stderr[-1000:]}
                print(f"[{case_id}@{eps}] RUN FAILED", flush=True)
                continue
            out_rows = parse_eqp_rows(run_dir / out_name, labels)
            assert out_rows.shape == ref_rows.shape
            d = np.abs(out_rows[:, :6] - ref_rows[:, :6])
            per_col = d.max(axis=0)
            results[f"{case_id}@{eps}"] = {
                "status": "OK",
                "max_abs_delta_eV": float(d.max()),
                "per_col_max_eV": [float(x) for x in per_col],
                "frozen_atol": atol,
                "gate_would_pass": bool(d.max() <= atol),
            }
            print(f"[{case_id}@{eps}] max|Δ|={d.max():.3e} eV "
                  f"(frozen atol {atol:g} → "
                  f"{'PASS' if d.max() <= atol else 'FAIL — expected, do not re-freeze'})",
                  flush=True)
    out = workdir / "golden_gates_ridge_on.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
