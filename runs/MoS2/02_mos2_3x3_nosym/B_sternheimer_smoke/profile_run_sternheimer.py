"""Profile cold-start of psp.run_sternheimer end-to-end.

Captures the JAX compile log + xprof trace for one full run so we can
find shape-polymorphic recompiles.  Run with:

    PF_OUT=profile_runstern \
        LORRAX_NGPU=1 lxrun python3 -u profile_run_sternheimer.py

Then analyse with:

    python3 /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/analyze_compile_log.py \
        profile_runstern/compile.log --top 40 --out-md profile_runstern/report.md
"""
from __future__ import annotations
import os, sys

# ── pf wiring (must come before any jax import) ──
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
sys.path.insert(0, '/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling')

import pf
out = os.environ.get('PF_OUT', 'profile_runstern')
pf.setup_env(out)
pf.attach_compile_log(os.path.join(out, 'compile.log'))

# Now safe to import lorrax / jax.
from runtime import set_default_env; set_default_env()
from common.jax_compile_cache import ensure_jax_compile_cache
ensure_jax_compile_cache()

from psp.run_sternheimer import run_sternheimer

# A small but representative cold-start: q=0 + 2 nonzero q's, ng_out=1
# (chi-only, no derivatives, no S-tensor — they trace through the same
# JIT graphs as derivs/S-tensor anyway).  Schur warm-start on by default.
run_sternheimer(
    wfn_path='WFN.h5', pseudo_dir='.',
    iq_list=[0, 4, 8], ng_out=1,
    n_cond_bands=20, tol=1e-6, max_iter=80,
    truncation_2d=True,
    output_path=os.path.join(out, 'sternheimer_profiled.h5'),
    with_derivatives=False, with_s_tensor=False,
    verbose=True,
)
