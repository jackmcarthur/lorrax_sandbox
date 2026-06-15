# `_to_host` design review — is the 4-case shape switch worth keeping?

**Date:** 2026-05-20
**Scope:** `sources/lorrax_B/src/file_io/_slab_io_allgather.py` `_to_host` helper,
the three `tiled=True` patches on branch `agent/jax-09-cpu-compat`, the ~15
adjacent sites that still call `process_allgather(tiled=False)`.
**Status:** evaluation only. No source changes proposed in this report.

---

## 1. Verdict

**The 4-case shape switch in `_to_host` is accidental complexity.** Two of its
four branches are dead code under multi-process LORRAX (cases 2 and 3 cannot
fire — `process_allgather(tiled=True)` on a non-fully-addressable JAX Array
returns *exactly* `A.shape`, the case-1 shape), and the 4th branch defends
against a hypothetical JAX-future-restoration that the deprecation message
explicitly does not promise. The code is not *wrong* (the live branch handles
the only shape that occurs), but the docstring and three of the four branches
encode a model of `process_allgather` that does not match what the JAX
implementation actually does. The user's intuition — "this feels like it's in
the weeds" — is correct. The user's *alternative* (always-tiled with explicit
tile-size) is the wrong fix for the wrong problem: the issue isn't tile size,
it's that we don't need shape dispatch at all.

## 2. What `tiled=True` actually does — read from JAX source

`.venv/.../jax/experimental/multihost_utils.py:97-135`, function
`_handle_array_process_allgather(inp, tiled)`:

```python
def _handle_array_process_allgather(inp, tiled):
  if isinstance(inp, array.ArrayImpl) and not inp.is_fully_addressable:
    if not tiled:
      raise ValueError(...)                                       # (A)
    if isinstance(inp.sharding, sharding_impls.NamedSharding):
      reps = inp.sharding.update(spec=P())
    else:
      reps = sharding_impls.GSPMDSharding.get_replicated(...)
    out = jax.jit(_identity_fn, out_shardings=reps)(inp)          # (B)
  else:
    # All inputs here will be fully addressable.
    if jax.process_count() == 1:
      out = np.asarray(inp)
      return np.expand_dims(out, axis=0) if not tiled else out    # (C)
    ...                                                           # (D)
  return np.asarray(out.addressable_data(0))                      # (E)
```

There are exactly **two** semantically distinct paths:

**Path (B) — non-fully-addressable JAX Array.** The function resharding-jits
the input to *fully replicated* (`out_shardings=P()`) on the input's existing
mesh. Then it returns `np.asarray(out.addressable_data(0))`. From
`jax/_src/array.py:533-537`:

```python
def addressable_data(self, index: int) -> ArrayImpl:
  if self.is_fully_replicated:
    return self._fully_replicated_shard()
  return self._arrays[index]
```

A fully-replicated shard has the **global shape** of the array. So under Path
(B), the return is always `np.ndarray` with shape exactly `A.shape`. The
`tiled` flag is *only* consulted to error out if `False` — it does not affect
the return shape.

**Path (D) — fully-addressable input under `process_count() > 1`.** This is
the "every host holds an identical local numpy" branch (e.g. someone passed a
plain `np.ndarray` that was lifted, or a JAX array with `SingleDeviceSharding`
or a NamedSharding spec'd to `P()` on a *single-host* sub-mesh). Here `tiled`
matters: the per-host local arrays are stacked on a new leading "processes"
axis and reduced by identity-jit to `P()`. With `tiled=True`, no
`expand_dims` is inserted, so the global stacked shape is
`(world * A_local.shape[0], *A_local.shape[1:])`. With `tiled=False`, an
`expand_dims(axis=0)` happens first, so the result is `(world, *A_local.shape)`.

**Summary table — return shape of `process_allgather(A, tiled=True)`:**

| input regime | path | return shape |
|---|---|---|
| `process_count() == 1` (never hit; we early-return at line 63) | (C) | `A.shape` (since `tiled=True`) |
| multi-process, JAX Array, **not** fully addressable | (B) | **exactly `A.shape`** — independent of sharding spec |
| multi-process, fully-addressable (e.g. plain numpy on every host) | (D) | `(world * A_local.shape[0], *A_local.shape[1:])` |
| multi-process, scalar | (D) with `ndim==0` branch | `(world,)` |

The crucial finding: under LORRAX's normal multi-process regime, **every JAX
Array that has any sharding on the `('x','y')` mesh is non-fully-addressable**.
A `NamedSharding(mesh, P())` (fully replicated) on a multi-process mesh is
*also* non-fully-addressable — `jax.Array.is_fully_addressable` docstring
(`jax/_src/array.py:430-433`):

> Note that fully replicated is not equal to fully addressable i.e.
> a jax.Array which is fully replicated can span across multiple hosts and is
> not fully addressable.

So **path (B) catches every interesting LORRAX case**, and path (B) always
returns `A.shape`. Cases 2 and 3 in `_to_host`'s docstring describe a Path (D)
phenomenon that doesn't apply to the inputs `_to_host` actually receives.

## 3. The case-analysis taxonomy LORRAX actually produces

Concrete PartitionSpecs grepped from `src/file_io/` and `src/gw/`:

| call site | typical spec | what arr.shape looks like |
|---|---|---|
| `zeta_loader.py:29` | `P(None, None, ('x','y'))` | `(nq, n_rmu, n_G)` |
| `zeta_reader.py:206` | `P(None, None, ('x','y'))` | `(nq, n_rmu, n_G)` |
| `zeta_reader.py:320` | `P(None, ('x','y'), None)` | `(nq, n_rmu, n_G)` |
| `tagged_arrays.py:200` | `P(None, None, None, 'y')` | `(nk, nb, ns, n_rmu)` |
| `wfn_loader.py:763` | `P(('x','y'), None, None, None)` | `(nk, ns, nb, n_G)` |
| `gw_init.py:1123` (G0) | replicated or single-axis | `(nq, n_rmu)` |
| `isdf_fitting.py:2691` (gflat_acc) | sharded on `('x','y')` | `(nq, n_rmu, n_G_sph)` |
| `compute_vcoul.py:551` | `P('x', None)` or `P(('x','y'), None)` | `(nq, n_rmu)` |
| `head_wing_schur.py:53` | `P(None, 'y')` | `(nq, n_rmu_Y)` |

All of these are NamedSharded jax.Arrays whose `sharding` lives on the
process-spanning `mesh_xy`. All are non-fully-addressable. All hit Path (B).
All return shape `A.shape`. **None of them hit cases 2, 3, or 4 of `_to_host`.**

If you wanted to exercise case 2 / case 3 you would need to construct (say) a
fully-replicated `np.ndarray` and feed it through `process_allgather` —
something LORRAX does not do, because `_to_host`'s first `isinstance(A,
np.ndarray)` short-circuits to a cheap cast.

**Empirical anchor:** the Si μ=384 CPU x_only run on 1, 2, and 4 processes
(`CPU_VALIDATION_2026-05-20.md` headline table) matches GPU `eqp0.dat` to 9
sig figs, exercising both the `_slab_io_allgather` `write_slab` path (Σ
output) and the `isdf_fitting` zeta-write path. The fact that the run *works*
is consistent with Path (B) being the only path used: cases 2–4 are simply
never visited.

## 4. Alternative designs

### 4a. Status quo: shape-based dispatch (4 cases)

```python
def _to_host(A):
    if isinstance(A, np.ndarray):                                return A
    if jax.process_count() == 1: return np.asarray(jax.device_get(A))
    gathered = multihost_utils.process_allgather(A, tiled=True)
    host = np.asarray(jax.device_get(gathered))
    expected = tuple(A.shape)
    if host.shape == expected:                                   return host        # case 1
    world = jax.process_count()
    if (host.ndim == len(expected) and host.shape[1:] == expected[1:]
        and host.shape[0] == world * expected[0]):
        return host[:expected[0]]                                              # case 2/3
    if (host.ndim == len(expected) + 1 and host.shape[0] == world
        and host.shape[1:] == expected):
        return host[0]                                                         # case 4
    raise RuntimeError(...)
```

- **LOC:** ~25 in the helper body + 12 lines of docstring rationale for the
  four cases.
- **Ergonomics at call sites:** perfect — no extra arguments.
- **JAX-version stability:** robust to future API restorations (defensive
  case 4) and to a hypothetical regression where Path (D) starts to be hit
  (case 2/3).
- **What breaks if assumption violated:** nothing — every observed shape is
  caught and an unobserved one raises. **The cost is interpretability:** a
  reader has to verify each case against JAX internals to satisfy themselves
  the code is correct. Three of the four cases are dead in practice, which
  invites future maintainers to "simplify" them away unsafely.

### 4b. Always tiled + explicit `tile_size` arg (the user's hypothesis)

```python
def _to_host(A, *, global_shape=None):
    ...
    gathered = multihost_utils.process_allgather(A, tiled=True)
    host = np.asarray(jax.device_get(gathered))
    if global_shape is None:
        global_shape = tuple(A.shape)        # trust A.shape
    if host.shape == global_shape:
        return host
    # truncate replicated tiling
    if host.shape[1:] == global_shape[1:] and host.shape[0] % global_shape[0] == 0:
        return host[:global_shape[0]]
    raise RuntimeError(...)
```

- **LOC:** similar.
- **Ergonomics at call sites:** the user's question is whether threading a
  `global_shape` (their "tile size") through ~20 call sites is feasible.
  Inventory of `_to_host` consumers within the allgather backend:

  | site | shape source |
  |---|---|
  | `_slab_io_allgather.py:184` (write_slab) | already has `gshape` in scope from `_normalize_slab_request` |
  | `_slab_io_allgather.py:294` (accumulate_slab) | already has `local_shape = tuple(host.shape)` post-gather |

  And the inline gathers that *could* call `_to_host`:

  | site | shape source |
  |---|---|
  | `gw_init.py:1123` (G0 write) | `G0_all.shape` directly available |
  | `isdf_fitting.py:2691` (gflat) | `gflat_acc.shape` directly available |
  | `davidson.py:59` | `arr.shape` directly available |
  | `bse_davidson_helpers.py:53` | `arr.shape` directly available |
  | `davidson_absorption.py:48` | `arr.shape` directly available |

  In every case the global shape is `A.shape`. The user's worry — "I'd need to
  thread a tile size that isn't super available" — would be valid for a
  helper that's called *inside a jit'd kernel* where the source array's shape
  is partially traced. But `_to_host` is called from eager Python only (it's
  the boundary between a JAX gather and an h5py write); `A.shape` is *always*
  statically available. So this alternative doesn't actually need a new
  argument — the helper can just `expected = tuple(A.shape)`, which is what
  4a already does.

- **JAX-version stability:** identical to 4a.
- **Net assessment:** the user's hypothesis encodes a real concern (don't
  dispatch on a shape that's ambiguous between cases 2 and 3) but the fix —
  add a parameter — is unnecessary because `A.shape` is already an
  unambiguous source of truth. The redundant parameter would be ignored
  everywhere.

### 4c. Sharding-metadata-driven dispatch

`Array.sharding.is_fully_replicated`, `Array.is_fully_addressable`, and
`Array.sharding` are all available pre-gather on every jax.Array
(`jax/_src/named_sharding.py:225`, `jax/_src/array.py:384/423`,
documented since at least JAX 0.4). They are stable public API.

```python
def _to_host(A):
    if isinstance(A, np.ndarray):
        return A
    if jax.process_count() == 1:
        return np.asarray(jax.device_get(A))
    # multi-process JAX Array.
    if A.is_fully_replicated:
        # Every host already has the full data; cheaper than allgather.
        return np.asarray(A.addressable_data(0))
    # Non-replicated, non-fully-addressable: path (B) of process_allgather
    # returns shape A.shape after an identity-jit reshard.
    gathered = multihost_utils.process_allgather(A, tiled=True)
    return np.asarray(jax.device_get(gathered))
```

- **LOC:** ~10, no shape arithmetic, no defensive branches.
- **Correctness argument:** for the replicated branch, `A.is_fully_replicated`
  is `True` iff every device holds the full array (`named_sharding.py:225-233`
  — `num_partitions == 1`). For such an array, `A.addressable_data(0)` returns
  one of the (identical) local shards and that has shape `A.shape`. This
  bypasses the gather entirely. For the non-replicated branch, we are
  guaranteed to hit Path (B) of `_handle_array_process_allgather` (Path (D)
  requires fully-addressable, and a non-replicated multi-process NamedSharded
  array is non-fully-addressable). Path (B) returns `A.shape`.
- **What breaks if assumption violated:** what if JAX changes Path (B) to
  return a different shape in some future version? Then 4c would need a
  shape check too — but so would 4a and 4b. 4c is no *worse* on that axis,
  and the dispatch is on stable metadata (sharding properties) rather than
  on the shape of a value returned by an opaque jit.
- **JAX-version stability:** `is_fully_replicated` and `addressable_data`
  are part of the documented `jax.Array` API, not the experimental
  `multihost_utils` module. They are at least as stable as
  `process_allgather` itself.
- **Bonus:** the replicated short-circuit saves an identity-jit launch +
  whatever the underlying reshard does (in JAX 0.9, replicated→replicated
  *should* be a no-op but it still goes through `jax.jit(_identity_fn,
  out_shardings=reps)`, which is non-zero overhead).

### 4d. Brute-force: drop the helper, use what JAX recommends directly

The JAX deprecation message says verbatim: pass `tiled=True` for
non-fully-addressable arrays. The JAX team's recommended migration is exactly
to flip the flag — nothing about post-process. Inline:

```python
host = np.asarray(multihost_utils.process_allgather(A, tiled=True))
```

This is two LOC. It's what `pivoted_cholesky.py:448`,
`cusolvermp_eigh_test.py:97`, and every test under `src/common/slate_*` and
`src/common/cusolvermp_*` already do (often without even the `np.asarray`).
The downside: no defensive guard against a future regression. The upside:
zero dead code; the helper is literally a noop wrapper.

This is essentially "4c minus the replicated short-circuit." Worth
considering if we want to be maximally faithful to the JAX docs.

### Comparison

| design | LOC | dead branches | reshard cost on replicated | reader effort |
|---|---:|---:|---|---|
| 4a (status quo) | ~25 | 2–3 of 4 | identity-jit always | high (verify 4 cases) |
| 4b (tile_size) | ~22 | same as 4a | same | medium |
| 4c (metadata) | ~10 | 0 | bypassed if replicated | low |
| 4d (inline) | ~2 | 0 | identity-jit always | trivial |

## 5. Recommendation

**Adopt 4c (sharding-metadata-driven dispatch).** Concrete proposed body
already written above. Rationale, in order of weight:

1. **Eliminates the docstring/code mismatch in 4a.** The current docstring
   asserts cases 2–4 are reachable; the JAX source shows they are not for
   LORRAX inputs. Future maintainers will trip on that gap.
2. **Dispatches on stable, public API** (`is_fully_replicated`,
   `addressable_data`) rather than on the opaque return shape of an
   experimental utility.
3. **Cheaper replicated case** — `A.addressable_data(0)` skips the
   identity-jit reshard that `process_allgather` does even when nothing
   changes.
4. **Half the lines, no dead branches, no shape arithmetic.** Easier to audit
   against the next JAX version bump.
5. **Subsumes every other inline gather in the codebase.** Once 4c is in,
   the `_gather_to_host` in `davidson.py`, `bse_davidson_helpers.py`,
   `davidson_absorption.py`, and the inline gathers in `gw_init.py:1123` and
   `isdf_fitting.py:2691` can all delete their local helpers and import
   `_to_host`. They have identical semantics (gather a possibly-sharded
   array to host as numpy); they don't need different code.
6. **Doesn't accept the user's framing.** The user proposed always-tiled +
   tile-size. That proposal solves "we want one code path"; it doesn't
   address "the shape switch is encoding stale assumptions about
   `process_allgather`". 4c addresses the *actual* defect.

The recommendation is NOT to delete the helper (4d) because the replicated
short-circuit is meaningful on multi-process, multi-host runs where
`is_fully_replicated=True` is common for small metadata (`omega_ev`,
mu_indices, etc.) and the identity-jit reshard is not free.

### Migration sketch (not for implementation in this report — just the
shape of it):

1. Replace `_to_host` body in `_slab_io_allgather.py` with the 4c body.
2. In `solvers/davidson.py`, `bse/bse_davidson_helpers.py`, and
   `bse/davidson_absorption.py`, delete the duplicate `_to_host` /
   `_gather_to_host` and `from file_io._slab_io_allgather import _to_host`.
3. In `gw_init.py:1123-1126`, replace with `G0_gathered = _to_host(G0_all)`
   and drop the `ndim==5 and shape[0]==1` postprocess (which was guarding
   against the V_q bispinor leading-1 axis, NOT a multi-process tile —
   verify in code review that V_q raw shape is `(nq, n_rmu, n_rmu)` not
   `(1, npol, npol, nq, n_rmu, n_rmu)`).
4. In `isdf_fitting.py:2691-2700`, replace with `_g = _to_host(gflat_acc)`
   and drop the `_g.ndim == 4 and _g.shape[0] == 1` postprocess (same
   caveat — that may be from a since-removed leading-batch axis).
5. Leave the `tiled=False` sites that LORRAX never actually exercises under
   multi-process (e.g. `eigh_benchmark.py` is a single-process benchmark)
   alone, but audit each: most of `src/common/*_test.py` and `*_bench.py`
   are single-process anyway and the `tiled=False` calls go through the
   `jax.process_count() == 1` branch on line 111, which expand-dims's and
   returns shape `(1, *A.shape)`. If those tests pass today they don't need
   the patch.

Steps (3) and (4) deserve a careful read before flipping. The "leading-1
axis" pattern in those inline guards looks like a leftover from a previous
shape convention, not a multi-process artifact, but I have not chased the
provenance. Anyone executing the migration should verify by printing
`G0_all.shape` and `gflat_acc.shape` in a working run before deleting.

## 6. What I am not confident about

- **Whether 4c's `if A.is_fully_replicated` branch is hit in practice.** All
  the call sites I inspected pass arrays that are *sharded* on `mesh_xy`
  axes; the replicated short-circuit may never fire on the production code
  path. If so, 4c reduces to 4d in practice. That's still better than 4a
  (still half the LOC), but the "cheaper replicated case" argument is
  hypothetical until measured.
- **The leading-1 axes in `gw_init.py:1125` and `isdf_fitting.py:2695`.** I
  did not chase whether those `shape[0] == 1` guards are an old V_q bispinor
  shape convention (in which case they should stay), a multi-process
  artifact (in which case 4c subsumes them), or dead defensive code (in
  which case they should be deleted regardless). Migration step (3)/(4)
  needs that verification.
- **Path (B) of `_handle_array_process_allgather` is itself an experimental
  internal.** The function's contract — "returns shape `A.shape` for
  non-fully-addressable inputs" — is what the source code does as of the
  JAX 0.9 pinned to lorrax_B's venv. The public docstring on
  `process_allgather` (lines 138-155) hedges: "If the input is a
  non-fully addressable jax.Array, then the data is fully replicated."
  "Fully replicated" doesn't explicitly say "with no leading axis" — that's
  an implementation detail. If JAX 0.10 changes the implementation to add a
  leading process axis here (analogous to Path (D) with `tiled=False`), 4c
  *and* the live branch of 4a both break in exactly the same way. The
  defensive cases 2/3/4 of 4a would NOT save us — they'd misfire on the new
  shape because the case-2/3 check requires `host.shape[0] == world *
  expected[0]`, which a leading-process-axis `(world, *A.shape)` would not
  satisfy (it would hit case 4 only if `host.shape[0] == world`). So 4a's
  defensive value is real only against a specific narrow regression. 4c is
  no worse here, and a future shape check could be added if/when that regression
  ships.
- **I did not run any experiment.** Two CPU validation runs (1- and 4-proc
  Si μ=384, table in §3) already exercised the live branch of 4a in
  production. No additional run is needed to recommend 4c — the equivalence
  is mechanical from the JAX source. An empirical check would be a
  smoke-test that 4c produces bit-identical Σ output to 4a on the existing
  4-proc CPU Si μ=384 input (cheap, ~17 s wall, no new SLURM allocation
  needed since `JID 54408957` from the prior validation is reusable or a
  fresh `lxalloc` is trivial). I would propose running that test only when
  the user accepts the recommendation and starts the implementation pass.
- **JAX migration-guide reference.** I did not find a written migration
  guide entry for the `tiled=False` deprecation. The recommendation is
  encoded in the `ValueError` message itself (line 100-102 of
  `multihost_utils.py`): "Gathering global non-fully-addressable arrays
  only supports tiled=True." That is the only "what should users do"
  guidance I can locate in the installed package. The deprecation likely
  has a corresponding GitHub PR/issue but I did not fetch the web to find
  it. If the user wants a paper trail before merging, they may want me to
  WebFetch the JAX changelog as a follow-up.
