"""Timestamp every print() then exec gw.gw_jax.main — analysis wrapper only.

Usage: python3 run_ts.py -i cohsex.in
Prefixes each stdout print with [epoch-seconds] so per-SC-iteration wall
times can be read off the log alongside JAX_LOG_COMPILES' own timestamps.
"""
import builtins
import sys
import time

_orig_print = builtins.print


def _ts_print(*args, **kwargs):
    _orig_print(f"[{time.time():.3f}]", *args, **kwargs)


builtins.print = _ts_print

from gw.gw_jax import main  # noqa: E402

sys.exit(main(sys.argv[1:]))
