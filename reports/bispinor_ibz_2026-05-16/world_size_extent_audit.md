# World-size × extent slicing audit (2026-05-16)

Read-only audit of LORRAX for places where `world_size`-padded ranges flow into per-rank slicing that can overshoot file extent. Triggered by the bispinor 16-GPU gate's HDF5 read failure (band axis) and continued because the bug class is general.

## HIGH-risk sites

### 1. `src/gw/v_q_g_flat.py:258` — μ-axis padded request in `ZetaReader.read_zeta_G_slab` (FFI path)

- **Axis**: μ (centroid axis)
- **Extent source**: `zeta_loader.n_rmu` (logical) vs caller-padded `n_rmu_padded` (rounded to world_size)
- **Request path**: `_make_read_all_ibz` → `read_zeta_G_slab(mu_offset=0, mu_count=n_rmu_padded, valid_mu=n_rmu_logical)` (lines 362-363 build `n_rmu_padded = _pad(n_rmu_L)`)
- **Trigger conditions**: `world_size ∤ n_rmu_logical`
- **CrI3 6×6 30Ry bispinor concrete**: n_rmu=300 (charge), 298 (transverse); 16 ∤ 300, 16 ∤ 298 — fires on charge AND TT tiles
- **Failure mode**: H5Dread "selection + offset not within extent" or silent wrong data
- **Severity**: HIGH — active on every V_q G-flat read at world_size > 2 unless n_rmu happens to divide

## MEDIUM-risk sites

### 2. `src/file_io/wfn_loader.py:579-582` — band-axis per-rank count in `_phdf5_build`

- **Axis**: band
- **Pattern**: `bands_per_rank = nb_padded // world` and per-rank `counts = [bands_per_rank, ...]`
- **Currently guarded** by check at line 590 (raises "pad-past-file") but the per-rank arithmetic is fragile if the cap is removed
- This is the bug currently being patched by Agent 1 (commit pending)

## LOW-risk (likely safe)

### 3. `src/common/load_wfns.py:418` — n_rmu round_up for sharding constraints
Padding is for `with_sharding_constraint` only; no I/O reader consumes the padded extent.

### 4. `src/common/meta.py:109, 127-129, 133` — band/μ paddings stored
Callers respect `valid_shape=` clip on writes; risk is procedural, not algorithmic.

### 5. `src/file_io/zeta_reader.py:220-228` — μ/q axes in `read_zeta_r_slab`
Calls `SlabIO.read_slab` with `valid_shape=` clip; SlabIO validates against the logical extent at line 225.

### 6. `src/file_io/slab_io.py` — allgather backend
Sequential gather-to-rank-0, per-rank overshoot can't happen in this path. Only FFI path is at risk.

## Recommendation

Two fixes needed for the 16-GPU bispinor gate to clear:
1. Band-axis: `wfn_loader._phdf5_build` (Agent 1, in flight).
2. μ-axis: `ZetaReader.read_zeta_G_slab` / FFI handler at v_q_g_flat.py:258.

If the FFI C++ handler is shared between band and μ readers, a single clamp+pad fix in the C++ layer would address both. If not, two Python-side patches needed.
