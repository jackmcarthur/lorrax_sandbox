"""Print allocator env vars then delegate to gw.gw_jax.

Used by Agent O (allocator audit) to verify that --env= overrides from the
launcher actually reach the JAX import inside the Shifter container.  We must
print BEFORE importing jax (which gw.gw_jax does indirectly).  Therefore we
print first, then execve into the canonical entrypoint.

This is a thin diagnostic wrapper — no code is changed in LORRAX itself.
"""

import os
import sys

# Print rank-0 only to keep the log readable; SLURM_PROCID is the rank.
_rank = os.environ.get("SLURM_PROCID", "0")
if _rank == "0":
    print("[agent_o env audit] rank 0 sees:", flush=True)
    for k in (
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "TF_GPU_ALLOCATOR",
        "LORRAX_MEM_DEBUG",
        "LORRAX_RCHUNK_DEBUG",
        "LORRAX_MAX_RCHUNKS",
        "LORRAX_EXIT_AFTER_ZETA",
        "LORRAX_FORCE_FULL_BZ",
    ):
        print(f"  {k}={os.environ.get(k, '<unset>')}", flush=True)
    print("[agent_o env audit] handing off to gw.gw_jax", flush=True)

# Delegate to gw.gw_jax with original argv (drop sys.argv[0])
# Use runpy so the module's __name__ == '__main__' guard executes.
import runpy
sys.argv = ["gw.gw_jax"] + sys.argv[1:]
runpy.run_module("gw.gw_jax", run_name="__main__")
