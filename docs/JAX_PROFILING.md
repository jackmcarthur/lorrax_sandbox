# JAX Profiling & JIT Debugging

## Cache miss diagnosis

The single most useful tool for understanding JIT recompilation:

```python
jax.config.update("jax_explain_cache_misses", True)
```

This logs every tracing cache miss with the exact function, the argument that
changed, and the old vs new type signature. Use it when:
- A k-loop is slower than expected (per-k recompilation)
- Warmup doesn't seem to help
- You suspect shape/dtype polymorphism issues

Example output:
```
TRACING CACHE MISS at h_dft.py:37 (_apply) because:
  for apply_H_k at dft_operators.py:577
  never seen input type signature
  closest has 1 mismatch:
    * at psi_box, seen c128[1,2,24,24,24], but now given c128[32,2,24,24,24]
```

This tells you the FFT batch dimension changed between warmup (1) and actual
use (32), triggering a retrace. Fix: warmup with all batch sizes Davidson uses.

## Common recompilation causes

| Symptom | Cause | Fix |
|---------|-------|-----|
| First 2-3 k-points slow, rest fast | Python loop creates new JAX arrays per k | JIT the entire per-k assembly |
| Every k-point retraces | Closure captures different data per k | Pass H_k data as explicit args to a shared JIT kernel |
| Batch dimension mismatch | Warmup at wrong batch size | Warmup at all Davidson subspace sizes |
| `float(x)` inside JIT | Python value extraction breaks tracing | Use `x.astype(jnp.float64)` or pass as static |
| `jnp.asarray(numpy_array)` inside JIT | Creates new tracer each call | Convert once at caller, pass JAX array |

## XLA HLO dumps

For low-level compilation analysis:

```bash
# Dump HLO for a specific function
XLA_FLAGS="--xla_dump_to=/tmp/xla_dumps --xla_dump_hlo_as_text" \
  python3 -u -m psp.run_nscf ...

# Then inspect the .txt files in /tmp/xla_dumps/
# Look for: module size, number of fusions, memory usage
```

## xprof traces (Perlmutter)

```python
# In code:
jax.profiler.start_trace("/tmp/jax_trace")
# ... run the code to profile ...
jax.profiler.stop_trace()

# Then view:
# tensorboard --logdir=/tmp/jax_trace
```

## Timing individual operations

```python
import time
jax.block_until_ready(result)  # force GPU sync before timing
t0 = time.perf_counter()
result = my_jit_fn(x)
jax.block_until_ready(result)
print(f"  {time.perf_counter()-t0:.3f}s")
```

Always `block_until_ready` — JAX dispatches async by default.
