import sys
sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling")
import pf
pf.setup_env("/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke")
pf.attach_compile_log("/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke/compile.log")
import jax, jax.numpy as jnp

@jax.jit
def kernel(x, y):
    return (x @ y.T).sum(axis=-1)

x = jnp.ones((8, 16))
y = jnp.ones((8, 16))

with pf.trace_profile("/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke"):
    with pf.region("kernel_exec"):
        for _ in range(3):
            kernel(x, y).block_until_ready()

pf.snapshot_memory("/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke/memprof/end.prof", label="end")
r = pf.aot_report(
    kernel, x, y,
    out="/pscratch/sd/j/jackm/lorrax_sandbox/tmp/pf_smoke/aot/kernel",
    timing_runs=3,
)
print("AOT keys:", list(r.keys()))
