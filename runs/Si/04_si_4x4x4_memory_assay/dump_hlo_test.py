#!/usr/bin/env python3
"""Minimal test: can we get XLA to dump HLO protos from inside Shifter?

Sets XLA_FLAGS before any JAX import, runs a tiny JIT, checks for output.
"""
import os
import sys
import glob

# The dump directory — must be on the shared filesystem (not /tmp)
DUMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profiles", "hlo_test")
os.makedirs(DUMP_DIR, exist_ok=True)

# Set XLA_FLAGS BEFORE importing JAX — this is the only way to ensure
# the XLA backend picks them up during initialization.
xla_flags = (
    f"--xla_dump_to={DUMP_DIR}"
    f" --xla_dump_hlo_as_proto=true"
    f" --xla_dump_hlo_module_re=.*"
)
# Append to existing XLA_FLAGS if any
existing = os.environ.get("XLA_FLAGS", "")
os.environ["XLA_FLAGS"] = f"{existing} {xla_flags}".strip()

print(f"XLA_FLAGS = {os.environ['XLA_FLAGS']}")
print(f"Dump dir: {DUMP_DIR}")

# Now import JAX (triggers XLA backend init)
import jax
import jax.numpy as jnp

# Distributed init for multi-process
proc_count = int(os.environ.get("SLURM_NTASKS", "1"))
if proc_count > 1:
    jax.distributed.initialize()

if jax.process_index() == 0:
    print(f"Processes: {jax.process_count()}, Devices: {jax.device_count()}")

# Run a tiny JIT to trigger HLO compilation + dump
@jax.jit
def tiny_matmul(a, b):
    return a @ b

a = jnp.ones((4, 4), dtype=jnp.float32)
b = jnp.ones((4, 4), dtype=jnp.float32)
c = tiny_matmul(a, b)
c.block_until_ready()

if jax.process_index() == 0:
    print(f"Computation result: {c[0, 0]}")

    # Check for dumped files
    all_files = glob.glob(os.path.join(DUMP_DIR, "*"))
    print(f"\nFiles in dump dir: {len(all_files)}")
    for f in sorted(all_files)[:20]:
        sz = os.path.getsize(f)
        print(f"  {os.path.basename(f):<80s} {sz:>10d} bytes")

    pb_files = [f for f in all_files if f.endswith(".pb")]
    txt_files = [f for f in all_files if f.endswith(".txt")]
    print(f"\n.pb files: {len(pb_files)}, .txt files: {len(txt_files)}")
    if not all_files:
        print("\nNO FILES DUMPED. XLA_FLAGS may not be reaching the XLA backend.")
        print("Try: export XLA_FLAGS before srun, not inside Python.")
