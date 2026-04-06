# `load_wfns.py` Architecture Review and Refactor Proposal

**File**: `src/common/load_wfns.py` (1955 lines)
**Purpose**: Load wavefunctions from WFN.h5, rotate via SymMaps, place in FFT
boxes, transform to real space, and extract centroid values for ISDF.

## Current structure

The file has ~30 functions serving at least four distinct roles that have
accumulated organically. Here's a map:

### Role 1: G-space → FFT box → GPU (wavefunction ingestion)

| Function | Lines | What it does |
|----------|-------|-------------|
| `read_Gvecs_to_devices` | 198-310 | HDF5 → SymMaps rotation → scatter to FFT box → `device_put` → shard. Returns `(global_psi_Gtot, nb)` |
| `load_kpoint_fftbox` | 125-136 | Single k-point version of the above. Used by centroid generation. |
| `load_gspace_for_bands` | 1445-1496 | Yet another variant: loads G-space coefficients into cache for r-chunk processing. |

**Problem**: Three functions that do variations of the same thing (read WFN, rotate,
scatter to grid). They share no code and use different data layouts.

### Role 2: FFT + phase + centroid extraction (the JIT kernel)

| Function | Lines | What it does |
|----------|-------|-------------|
| `get_sharded_wfns` | 317-421 | JIT kernel: FFT → phase → normalize → flatten → reshard → centroid gather → transpose. Returns `(psi_rtot_Y, psi_rmu_Y, psi_rmuT_X)` |
| `get_sharded_wfns_centroids` | 1770-1872 | Similar but centroid-only (no rtot output). |
| `get_sharded_wfns_rchunk_slice` | 1498-1614 | R-space chunk variant. Returns wavefunctions at a slice of r-points. |
| `get_psi_rchunk_from_cached` | 1616-1669 | Uses cached G-space to produce r-chunk. |
| `get_psi_rchunk` | 1672-1767 | Orchestrates get_psi_rchunk_from_cached. |

**Problem**: Five functions for extracting real-space wavefunctions, each with
different output shapes and sharding strategies. The core FFT+phase+centroid
logic is duplicated across them.

### Role 3: ISDF fitting (pair densities, Cholesky, zeta solve)

| Function | Lines | What it does |
|----------|-------|-------------|
| `compute_pair_density_spin_traced` | 440-481 | P_k(μ,ν) = Σ_{n,s} ψ*·ψ |
| `compute_pair_density_spin_matrix` | 483-535 | P_k,ab(μ,ν) for full spin matrix |
| `compute_CCT_from_left_right` | 537-598 | C = P_l^T P_r → Cholesky |
| `compute_CCT_from_left_right_spin_matrix` | 600-640 | Same, spin-matrix variant |
| `compute_ZCT_from_left_right_zchunk` | 642-703 | Z = P(r_chunk) → ZCT pipeline |
| `compute_ZCT_from_left_right_zchunk_spin_matrix` | 705-770 | Same, spin-matrix |
| `compute_L_q_from_CCT` | 772-838 | L = chol(C) with 2D block-cyclic |
| `solve_zeta_from_L_q` | 840-977 | ζ = L^{-1} Z via forward substitution |
| `fit_zeta_chunked_to_h5` | 979-1443 | Master ISDF fitting: orchestrates all above + H5 I/O |

**Problem**: This is the bulk of the file (700 lines) and is conceptually
separate from wavefunction loading. Each function exists in both spin-traced
and spin-matrix variants — a classic case for a strategy pattern or a single
parameterized function.

### Role 4: FFT helper functions

| Function | Lines |
|----------|-------|
| `make_sharded_ifftn_3d` | 81-104 |
| `make_sharded_fftn_3d` | 106-122 |
| `compute_block_size_for_2d_cholesky` | 25-78 |

### Role 5: Utility functions

| Function | Lines |
|----------|-------|
| `get_enk_bandrange` | 139-179 |
| `get_small_psi_component` | 181-196 |
| `load_centroids_band_chunked` | 1874-1955 |

## What's messy

1. **Three independent wavefunction loading paths** that don't share code:
   `read_Gvecs_to_devices`, `load_kpoint_fftbox`, `load_gspace_for_bands`.
   Each reimplements the HDF5→SymMaps→scatter pipeline differently.

2. **Five real-space extraction functions** that duplicate the FFT+phase core:
   `get_sharded_wfns`, `get_sharded_wfns_centroids`,
   `get_sharded_wfns_rchunk_slice`, `get_psi_rchunk_from_cached`, `get_psi_rchunk`.

3. **Spin-traced vs spin-matrix duplication**: `compute_pair_density_*`,
   `compute_CCT_*`, `compute_ZCT_*` all exist in two variants that differ
   only in the spin contraction.

4. **ISDF fitting mixed with wavefunction loading**: 700 lines of Cholesky
   factorization and zeta solving live in `load_wfns.py` alongside HDF5 I/O
   and FFT transforms. These are independent algorithms.

5. **Global mutable caches** (`_get_sharded_wfns_cache`, `_compute_pair_density_cache`,
   `_isdf_pipeline_cache`) scattered throughout the file, keyed by tuples of
   metadata. Hard to reason about invalidation.

6. **The OOM-causing monolithic JIT**: `_finalize` in `get_sharded_wfns` processes
   all k-points simultaneously. This was natural for small grids but breaks for
   production sizes.

## Proposed refactor

### Split into three files

```
src/common/
    load_wfns.py        → wavefunction I/O + FFT extraction (slimmed)
    isdf_fitting.py     → pair density, CCT, L_q, zeta solve, fit_zeta_chunked_to_h5
    fft_helpers.py      → make_sharded_ifftn_3d, make_sharded_fftn_3d, block sizes
```

### Unify the wavefunction loading paths

Replace the three loading functions with one:

```python
def load_wfns_gspace(
    wfn: WFNReader,
    sym: SymMaps,
    meta: Meta,
    band_range: tuple[int, int],
    mesh: Mesh,
    *,
    bispinor: bool = False,
) -> jax.Array:
    """Load wavefunctions from HDF5 into a sharded G-space FFT box.
    
    Returns: global_psi_Gtot of shape (nk, nb_padded, nspinor, *fft_grid),
             sharded over bands across the device mesh.
    """
```

The single-k-point `load_kpoint_fftbox` becomes a special case (nk=1).

### Make the FFT+centroid extraction k-chunked

Replace the monolithic `_finalize` JIT with a k-chunked pipeline:

```python
def extract_centroids_chunked(
    psi_Gtot: jax.Array,       # (nk, nb, nspinor, *fft_grid) — sharded on bands
    kpts: jax.Array,           # (nk, 3)
    centroid_indices: jax.Array,
    *,
    k_chunk_size: int = 64,    # process this many k-points per JIT call
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """FFT + phase + centroid extraction, k-chunked to control memory.
    
    Returns: (psi_y, psi_x) at centroid points only.
    """
    # The inner JIT kernel operates on k_chunk_size k-points at a time.
    # Each chunk: psi_Gtot[k_start:k_end] → FFT → phase → gather centroids.
    # The centroid outputs are small (n_rmu << n_rtot) and accumulate cheaply.
```

This solves the OOM: each chunk's FFT intermediates are ~0.4 GB (64 k-pts ×
60 bands × 2 spinors × 13824 r × 16 bytes / 4 GPUs), well within device memory.

### Parameterize spin contraction instead of duplicating

```python
def compute_pair_density(psi_l, psi_r, *, spin_mode: str = "traced"):
    """Compute pair density P(μ,ν).
    
    spin_mode: "traced" for Σ_s ψ*_s ψ_s, "matrix" for P_{ab}(μ,ν).
    """
```

One function with a flag instead of six copy-pasted functions.

### Move ISDF fitting to its own module

The 700 lines of `compute_CCT`, `compute_L_q`, `solve_zeta`, and
`fit_zeta_chunked_to_h5` are pure linear algebra with no wavefunction I/O
dependency. They belong in `isdf_fitting.py` (or `isdf/fitting.py`).

## Impact on the OOM fix

The k-chunked `extract_centroids_chunked` directly solves the 10×10×10 OOM.
With `k_chunk_size=64`, the per-chunk XLA buffer is ~8 GB (same as the working
4×4×4 case), regardless of total nk. The `psi_Gtot` can also be loaded in
k-chunks from host, avoiding the 26.5 GB host→device transfer entirely.

The most minimal OOM fix (without full refactor) would be to add k-chunking
to the existing `_finalize` function in `get_sharded_wfns`. But the full
refactor is recommended to prevent re-accumulating the same complexity.
