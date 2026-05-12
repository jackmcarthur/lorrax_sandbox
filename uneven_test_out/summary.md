# Uneven-sharding XLA partitioner microbench

Mesh: 4x4 (16 devices)
Input shape: (9, 2419, μ).  Tested μ: 672, 668, 661.

## Test 1 — top-level input on `P(None, None, ('x','y'))`

```
case                   mu  mu%16  result
```

  A_mu672_div16         672      0  compiled.  ag=0  ag-start=0  a2a=0  cperm=0  remat=0
  B_mu668_div4          668     12  device_put error: ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('x': 4, 'y': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, None, ('x', 'y')), memory_kind=device), which implies that the global size of its dimension 2 should be divisible by 16, but it is equal to 668 (full shape: (9, 2419, 668))
  C_mu661_prime         661      5  device_put error: ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('x': 4, 'y': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, None, ('x', 'y')), memory_kind=device), which implies that the global size of its dimension 2 should be divisible by 16, but it is equal to 661 (full shape: (9, 2419, 661))

## Test 2 — input on `P(None, None, 'x')`, wsc inside jit to `P(None, None, ('x','y'))`

```
case                   mu  mu%16  result
```
  A_mu672_div16         672      0  compiled.  ag=0  ag-start=0  a2a=0  cperm=0  remat=0
  B_mu668_div4          668     12  compiled.  ag=0  ag-start=0  a2a=0  cperm=6  remat=0
  C_mu661_prime         661      5  device_put error: ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('x': 4, 'y': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, None, 'x'), memory_kind=device), which implies that the global size of its dimension 2 should be divisible by 4, but it is equal to 661 (full shape: (9, 2419, 661))

## Test 3 — input padded inside jit, then wsc to product

Padded path: input on `P(None, None, 'x')` at logical μ; inside jit jnp.pad to mesh-divisible padded μ, wsc to `P(None, None, ('x','y'))`, sum.  Compares Test 2's uneven-wsc cost against an explicit pad.

```
case                   mu  mu%16  result
```
  A_mu672_div16         672      0  compiled  μ_pad=672.  ag=0  ag-start=0  a2a=0  cperm=0  remat=0
  B_mu668_div4          668     12  compiled  μ_pad=672.  ag=0  ag-start=0  a2a=0  cperm=6  remat=0
  C_mu661_prime         661      5  device_put error: ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('x': 4, 'y': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, None, 'x'), memory_kind=device), which implies that the global size of its dimension 2 should be divisible by 4, but it is equal to 661 (full shape: (9, 2419, 661))

## Test 4 — V_q-tile-like kernel body, μ=672 (mesh-divisible)

Body: WSC chain `P(None,None,('x','y'))` → `P(None,('x','y'),None)` → `P(None,'x',None)` & `P(None,'y',None)` → einsum → `P(None,'x','y')` → DUS into V_acc.  Mirrors the V_q tile inner kernel.

```
case                       mu  mu%16  result
```
  A_mu672_div16             672      0  compiled.  ag=6  ag-start=6  a2a=0  cperm=6  remat=0

## Test 5 — V_q-tile-like body, INPUT padded inside jit (logical 668 → 672)

Input arrives single-axis sharded at logical μ; jit pads to 672; kernel body runs on padded shape.  Tests the 'pad-inside-jit + clean kernel' pattern the helper module would expose at top-level boundaries.

```
case                       mu  mu%16  result
```
  B_mu668_div4              668     12  compiled  μ_pad=672.  ag=6  ag-start=6  a2a=0  cperm=12  remat=0
  C_mu661_prime             661      5  error: ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('x': 4, 'y': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, None, 'x'), memory_kind=device), which implies that the global size of its dimension 2 should be divisible by 4, but it is equal to 661 (full shape: (9, 2419, 661))

## Test 6 — V_q-tile-like body with INTERMEDIATES on uneven product spec

This is the user's worry case: WSC inside jit to product spec on logical 668 (NOT padded).  XLA must reshape an uneven product axis on every intermediate.  Compares HLO collective count to Test 5 (padded path).

```
case                       mu  mu%16  result
```
  B_mu668_div4              668     12  compiled  no-pad path.  ag=6  ag-start=6  a2a=0  cperm=25  remat=0
  C_mu661_prime             661      5  error: ValueError: One of device_put args was given the sharding of NamedSharding(mesh=Mesh('x': 4, 'y': 4, axis_types=(Auto, Auto)), spec=PartitionSpec(None, None, 'x'), memory_kind=device), which implies that the global size of its dimension 2 should be divisible by 4, but it is equal to 661 (full shape: (9, 2419, 661))
