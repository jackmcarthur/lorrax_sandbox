# g0w0_sc_toggle_audit — compile-count runs for G0W0_SC_TOGGLE_DESIGN.md

Inputs copied from tests/regression/{cohsex_debug,gnppm_debug} in sources/lorrax_D
(WFN/kin_ion/dipole h5 deleted after the runs to keep the report light — re-copy
from the fixture dirs to reproduce). run_ts.py timestamps every print() so
per-SC-iteration wall times can be read alongside JAX_LOG_COMPILES lines.

Env per run: JAX_LOG_COMPILES=1 LORRAX_SC_MAX_ITER={3,4} LORRAX_SC_TOL_EV=1e-10
LORRAX_SC_ACCEL=linear LORRAX_SC_MIXING=1.0 LORRAX_NGPU=1 lxrun python3 -u ...
