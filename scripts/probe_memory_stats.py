"""Probe: what does jax.local_devices()[0].memory_stats() return on A100?"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)

cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
n_local = len(cv.split(",")) if cv else 1
if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize(local_device_ids=list(range(n_local)))

d = jax.local_devices()[0]
print(f"Local device 0: {d!r}")

# Try BEFORE any allocation
stats0 = d.memory_stats()
if jax.process_index() == 0:
    print(f"\nmemory_stats() before allocation:")
    if stats0 is None:
        print("  None")
    else:
        for k, v in sorted(stats0.items()):
            print(f"  {k:30s} = {v}")

# Force an allocation
import jax.numpy as jnp
x = jnp.zeros((1024, 1024, 128), dtype=jnp.complex128)  # ~1 GB
x = x + 1.0
x.block_until_ready()

stats1 = d.memory_stats()
if jax.process_index() == 0:
    print(f"\nmemory_stats() after 1 GB alloc:")
    if stats1 is None:
        print("  None")
    else:
        for k, v in sorted(stats1.items()):
            print(f"  {k:30s} = {v}")

    # Try alternative APIs
    print(f"\nother device APIs:")
    for attr in ("get_memory_info", "memory_usage", "memory"):
        if hasattr(d, attr):
            fn = getattr(d, attr)
            try:
                print(f"  {attr}(): {fn() if callable(fn) else fn}")
            except Exception as e:
                print(f"  {attr}(): EXC {e}")
        else:
            print(f"  {attr}: absent")

    # Try nvidia-smi as a fallback
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader"],
            text=True, timeout=5)
        print(f"\nnvidia-smi:\n{out}")
    except Exception as e:
        print(f"\nnvidia-smi failed: {e}")
