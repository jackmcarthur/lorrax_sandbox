#!/usr/bin/env python3
import os
import sys
import time
import traceback
from pathlib import Path

import jax

os.environ.setdefault("LORRAX_MEM_PROFILE", "1")
sys.argv = ["gw_jax", "-i", str(Path("cohsex.in").resolve())]
from gw import gw_jax  # noqa: E402

stamp = time.strftime("%Y%m%d-%H%M%S")
trace_dir = Path("profiles") / "xprof" / f"mem16-full-{stamp}"
trace_dir.mkdir(parents=True, exist_ok=True)
print(f"[xprof] trace_dir={trace_dir.resolve()}", flush=True)

rc = 1
try:
    with jax.profiler.trace(str(trace_dir)):
        rc = int(gw_jax.main())
except Exception:
    print("[xprof] gw_jax raised exception; keeping traceback for diagnosis", flush=True)
    traceback.print_exc()
    rc = 1
finally:
    print(f"[xprof] done rc={rc}", flush=True)

raise SystemExit(rc)
