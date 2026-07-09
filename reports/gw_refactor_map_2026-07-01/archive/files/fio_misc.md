# file_io misc group — deep-read notes (gw refactor map, 2026-07-01)

Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`. Line numbers from that checkout.
Caller evidence: grepped function names across `src/`, `tests/`, `tools/`, `scripts/` (grep commands noted per finding).

---

## src/file_io/sigma_output.py (392 LOC)

**Purpose.** Writers for all Σ self-energy outputs: human-readable diagonal tables (`sigma_diag.dat`, `eqp0.dat`-style), per-(k,n) frequency-debug decomposition tables, and the canonical `sigma_mnk.h5` frequency-dependent Σ HDF5 (via SlabIO), plus a Ry→eV converter for the streamed PPM accumulator file. Category: **I/O: sigma output stage**.

### Functions

| Function | Lines | Role |
|---|---|---|
| `write_sigma_to_file` | 9–79 | Text table of diagonal Σ components per (k,n) in eV. Columns: `sigSX`, optional `sigCOH`, `sigTOT = sigSX + sigCOH`, optional `VH`. Labels configurable (`sx_label`/`corr_label`/`total_label`) so the same writer serves COHSEX (SX/COH) and Σ⁺/Σ⁻ decompositions. Imaginary part printed only if \|Im\| > 1e-10. Callers: `src/gw/gw_output.py:261`. |
| `write_eqp_g0w0` | 82–123 | Text table: per (k,n) `E_DFT`, `Re/Im[H0 + Σ_xc(E_DFT)]` — the G0W0 eqp0 diagnostic (physics: E_qp⁰ = ⟨n|T+V_ion+V_H+Σ_xc(E_DFT)|n⟩). Callers: `src/gw/gw_output.py:352`. |
| `write_sigma_freq_debug_table` | 126–203 | Generic per-(k,n) tab-separated table from `(name, (nk,nb) array)` pairs; complex → Re/Im sub-columns; all eV (caller converts, "internals in Ry, eV only at print"). Column width magic constant `col_w = 16`. Callers: `src/gw/gw_jax.py:806/907` (gated on `config.debug.sigma_freq_debug_output` and rank 0). |
| `write_chunked_complex_dataset_h5` | 206–251 | Standalone chunked complex128 HDF5 writer for 3D/4D arrays, k-chunked (`k_chunk_size=16` default), plain h5py. **Zero callers** (grep `write_chunked_complex_dataset_h5` over the whole repo: only its definition and the `__init__.py:38` re-export). Dead suspect. |
| `write_sigma_omega_h5` | 254–337 | Canonical `sigma_mnk.h5` writer via `SlabIO` (backend selectable H5PY_ALLGATHER vs PHDF5_FFI; legacy `use_ffi_io` bool alias still accepted). Datasets: `omega_ev (n_omega,)`, `sigma_total_kij_ev (n_omega,nk,nb,nb)`, optional `sigma_c_kij_ev`, `sigma_sx_kij_ev (nk,nb,nb)`, `hartree_kij_ev (nk,nb,nb)`. If total not passed it is derived: `total = c + sx[None,...] + hartree[None,...]` (Σ_tot(ω) = Σ_c(ω) + Σ_SX + V_H). Callers: `src/gw/ppm_pipeline.py:261` (inside `_write_sigma_omega_h5` wrapper, itself called by `ppm_pipeline.py:387` and `sc_iteration.py:674`). |
| `copy_sigma_kij_h5_to_omega_h5` | 340–392 | Converts the `_StreamedH5Accumulator` transient file (`sigma_c_kij_ry` dataset, Ry, written by `ppm_sigma`) into canonical `sigma_mnk.h5` layout in eV, ω-batched reads (`omega_batch_size=4`) so host never holds full (n_ω,nk,nb,nb). Rank-0-only, serial h5py; caller must gate. Multiplies by `RYD_TO_EV`. Callers: `src/gw/ppm_pipeline.py:270` (rank-0 gated, streamed path); referenced in comment `src/gw/ppm_sigma.py:1609`. |

### Cross-module deps
`common.provenance.provenance_header`, `file_io.slab_io.SlabIO`, `common.units.RYD_TO_EV`, h5py, numpy.

### Flags consumed (via callers)
`sigma_diag_file`, `eqp0_file`, `eqp1_file`, `sigma_omega_h5_file` (default `sigma_mnk.h5`), `sigma_kij_h5_file`, `sigma_freq_debug_output` (debug), backend selection `cfg.backend.slab_io` threaded through `backend=`.

### I/O
- Writes text: sigma diag tables (`sigma_diag.dat`/`eqp0.dat`/`eqp1.dat` per config names), G0W0 eqp table, freq-debug table. All prefixed with `provenance_header()`.
- Writes HDF5: `sigma_mnk.h5` (`omega_ev`, `sigma_total_kij_ev`, `sigma_c_kij_ev`, `sigma_sx_kij_ev`, `hartree_kij_ev`) via SlabIO.
- Reads HDF5: streamed accumulator file dataset `sigma_c_kij_ry` (Ry).

### Suspects
- **dead**: `write_chunked_complex_dataset_h5` — grep over repo shows only definition + `__init__` export; no caller in src/tests/tools/scripts.
- **weird**: `write_sigma_freq_debug_table._val` (lines 175–182) contains a "should never reach here" complex fallback that silently drops the imaginary part. `copy_sigma_kij_h5_to_omega_h5` re-imports h5py locally (line 360) though it is already a module import (line 4). Chunk-size magic numbers: `k_chunk_size=16`, `max(1, min(4, nk))` in copy path.
- **redundancy**: three-way overlap between `write_chunked_complex_dataset_h5` (dead, plain h5py chunked), `write_sigma_omega_h5` (SlabIO), and `copy_sigma_kij_h5_to_omega_h5` (plain h5py rank-0) — two live parallel Σ(ω)→h5 paths selected by whether ppm_sigma streamed or not (`ppm_pipeline._write_sigma_omega_h5` dispatches).

---

## src/file_io/qp_wfn.py (187 LOC)

**Purpose.** QP-wavefunction outputs for self-consistent GW: (a) a small companion HDF5 with QP rotation matrices U and QP energies, and (b) a full BGW-compatible `WFN.h5` with ψ already rotated into the QP basis and energies replaced (drop-in for BSE/restart). Category: **I/O: QP-WFN writer (SC-GW output stage)**.

### Functions

| Function | Lines | Role |
|---|---|---|
| `write_qp_rotations_h5` | 16–82 | Writes HDF5 with `U_mnk (nk,nb,nb)` where `U[k,m,n] = <m_DFT|n_QP>` (rotation: `c_qp[n,G] = Σ_m U[k,m,n] c_dft[m,G]`, i.e. `c_qp = U^T @ c_dft`), `E_qp_nk_hartree (nk,nb)`, `E_qp_nk_rydberg` (= ×2.0), `band_range` (0-based half-open), `kpoints_crys`, `kgrid`, optional `kpoints_reduced`, `kirr_to_kfull`; documentation attrs. Callers: `src/gw/gw_output.py:361`, `src/gw/sc_iteration.py:727`. Reader: `src/postprocess/rotate_wfn_to_qp.py:91` (`f_rot['U_mnk']`). |
| `write_qp_wfn_h5` | 89–186 | Writes full BGW-compatible WFN.h5 via `WFNWriter`: rotates the active band block `[band_start, band_stop)` per k with einsum **`"mn,msg->nsg"`** (`c_qp[n,s,G] = Σ_m U[k,m,n]·c_dft[m,s,G]`; spinor axis s untouched), replaces energies in the active window (`enk_full_ry[:, b0:b1] = enk_active_qp_ry`), copies all other bands/energies through. IBZ-only: output on the same irreducible-k grid, `mtrx`/`tnp` copied through. Opens a fresh mesh-less `WfnLoader(wfn._filename)` (eager backend) so the rank-0 write needs no collective (top-level loader's `.load` is collective on phdf5 backend); re-slurps ALL bands host-side: `psi_all (nk,nb,ns,ngkmax)` via `loader.load(bands=(0,nbands), k="ibz", sharding=None)`, `gvecs (nk,ngkmax,3)`, `ngk_valid (nk,)`. Callers: `src/gw/gw_jax.py:761`, `src/gw/sc_iteration.py:722`. |

### Cross-module deps
`file_io.wfn_writer.WFNWriter`, `file_io.wfn_loader.WfnLoader` (local imports), h5py, numpy. Reads `wfn` (WFNReader/WfnLoader) attributes: `nkpts, nbands, energies, kpoints, kweights, kgrid, shift, _filename`.

### Flags consumed
None directly; call sites gate on SC-iteration / `write_qp_wfn` behavior in `gw_jax`/`sc_iteration` config.

### I/O
- Writes: `*_rotations.h5`-style companion (datasets above); BGW `WFN.h5` (via WFNWriter, full BGW mf_header layout).
- Reads: source `WFN.h5` via fresh `WfnLoader` (all bands, IBZ, host).

### Suspects
- **redundancy**: `src/postprocess/rotate_wfn_to_qp.py` is a standalone postprocess tool that reads the rotations file and re-implements the same WFN rotation (U_mnk read at its line 91, rotation applied ~line 158) — a parallel old path to `write_qp_wfn_h5` (module docstring lines 3–10 acknowledges both writers exist). Classic old/new pair.
- **weird**: `E_qp_nk_rydberg = E_qp * 2.0` — assumes input is Hartree; two unit copies of the same dataset in one file (line 56–57). `write_qp_wfn_h5` reaches into private `wfn._filename` (line 160). `nosym=False` hard-coded (line 176). Occupations caveat documented (lines 123–131): `ifmax = nelec` inherited — wrong for metals / gap-closure after SC update, silently. Only `write_qp_rotations_h5` is re-exported by `file_io/__init__.py` (line 42); `write_qp_wfn_h5` imported by full path at call sites — inconsistent export surface.

---

## src/file_io/tagged_arrays.py (341 LOC)

**Purpose.** Canonical GW/BSE restart-state HDF5 I/O ("restart format v2"): writes/reads `isdf_tensors_{n_rmu}.h5` holding V_qmunu, W0_qmunu, S_qmunu, V0_noG0_munu, G0_mu_nu, psi_full_y, enk_full, plus the q→0 head scalars (vhead/whead), and a per-process shard dump. Name "tagged_arrays" is historical — nothing is tagged. Category: **I/O: restart / ISDF-tensor state (GW↔BSE seam)**.

### Functions

| Function | Lines | Role |
|---|---|---|
| `write_restart_state_to_h5` | 14–100 | Writes any provided subset of {V_qmunu, S_qmunu, V0_noG0_munu, G0_mu_nu, psi_full_y, enk_full, W0_qmunu} via SlabIO; `mode="w"` truncates + writes attr `restart_format_version=2` and optional `kgrid` (int64[3], lets BSE recover (nkx,nky,nkz) from flat-q V without re-reading WFN). `init_W0=True` pre-allocates all-zeros `W0_qmunu` sized from V_qmunu with dataset attr `W0_ready=False` (placeholder for bse_io); passing real `W0_qmunu` sets `W0_ready=True`. `W0_ready` attr written rank-0-only *after* SlabIO closes, then `sync_global_devices("restart_W0_ready_flag")` in try/except-pass. Callers: `src/gw/gw_init.py:1267` (V/G0/enk + init_W0, mode="w") and `:1297` (append ψ, mode="a"). |
| `write_w0_qmunu_to_h5` | 103–124 | Append/overwrite `W0_qmunu` in an existing restart file, sets `W0_ready=True` (same rank-0 + sync pattern). Caller: `src/gw/gw_jax.py:310` (after W solve, gated on `config.do_screened` and file existence). |
| `write_head_scalars_to_h5` | 127–171 | Rank-0 h5py write of q=0 Coulomb head: `vhead` scalar complex128 (v(q→0, G=G'=0) in Ry, BGW convention), `whead (n_omega,)` (len 1 static COHSEX, len 2 GN-PPM: static + iω_p), optional `omega_grid` as attr on `whead`. Consumed by `src/bse/bse_io.py:476–486` (`_load_ring_subset`, with cohsex.in overrides taking priority) via `head_correction.apply_q0_head_rank1`. Caller: `src/gw/gw_jax.py:329`. Non-rank-0 processes early-return after the sync barrier. |
| `read_restart_state_from_h5` | 174–190 | Eager full-file h5py read → jnp arrays: V_qmunu (required), S_qmunu/V0_noG0_munu/G0_mu_nu/enk_full optional, psi_full_y required (raises with "Regenerate restart tensors" message if missing). NOT via SlabIO — every rank reads the full file (replicated read; asymmetric with the sharded SlabIO write path). Only caller: `load_restart_state_from_h5` (internal). |
| `load_restart_state_from_h5` | 193–257 | Restart entry point: reads then reshapes/reshards for `gw.wavefunction_bundle.build_wavefunctions`. Back-compat: collapses old 8-D restarts `V[0,0,0].reshape(-1, μ, μ)` and 6-D `V[0,0,0]` to flat-q `(nq, μ, μ)`; if `G0_mu_nu.ndim > 1` takes row `[0]` (old (nqz, n_rmu) layout). Shardings: V → `P(None,'x','y')`; S → `P(None,None,None,'x','y')`; V0 → `P('x','y')`; G0 → `P('y')`; `psi_rmu_Y (nk,nb,ns,n_rmu)` → `P(None,None,None,'y')`; `psi_rmuT_X = conj(psi).transpose(0,3,1,2)` → `P(None,'x',None,None)` (single y→x all-to-all on μ; conjugated ψ* matches `load_centroids_band_chunked` pair-density convention); enk replicated. `band_slices` arg is dead (`del band_slices`, "retained for call-site compatibility"). Returns SimpleNamespace. Caller: `src/gw/gw_init.py:1310` (restart branch). |
| `_mesh_coords_for_local_process` | 260–266 | Finds (cx,cy) of this process's first local device in the 2-D mesh via `np.argwhere(devices_2d == target)`. |
| `save_restart_state_per_proc` | 269–341 | Writes per-process shard files `{prefix}.rank{R}.x{cx}.y{cy}.h5` with hand-computed block slices `_block_slice(n, parts, idx) = (n·idx//parts, n·(idx+1)//parts)` on the trailing sharded axes of V/S/V0/psi (duplicating SlabIO's slab math), plus global-shape attrs. `del meta` (arg ignored). Caller: `src/gw/gw_init.py:1302` — unconditionally on the ISDF compute path, with `S_qmunu=None`. **No reader anywhere** (grep `psi_full_local|V_local|V0_noG0_local` over src/tests/tools/scripts: only this file + unrelated `w_isdf.py` local variable names). Write-only output. |

### Cross-module deps
`file_io.slab_io.SlabIO`, jax (`process_index`, `with_sharding_constraint`, NamedSharding/PartitionSpec, `multihost_utils.sync_global_devices`), h5py; consumers: `src/gw/gw_init.py`, `src/gw/gw_jax.py`, `src/bse/bse_io.py` (reads `V_qmunu`, `W0_qmunu` + `W0_ready` attr + `kgrid` attr — see bse_io.py:316, 476; locates file via glob `isdf_tensors_*.h5`, bse_io.py:757).

### Flags consumed
`restart` (gw_config default True → selects compute-vs-load branch in gw_init), `cfg.backend.slab_io` (SlabIO backend), `do_screened` (gates W0 write in gw_jax). File name: `tmp/isdf_tensors_{n_rmu}.h5` (gw_jax.py:141).

### I/O
- Writes/reads HDF5 `isdf_tensors_{n_rmu}.h5`: attrs `restart_format_version=2`, `kgrid`; datasets `V_qmunu (nq,μ,μ)` flat-q, `S_qmunu`, `V0_noG0_munu (μ,μ)`, `G0_mu_nu (n_rmu,)`, `psi_full_y (nk,nb,ns,n_rmu)`, `enk_full`, `W0_qmunu` (+dataset attr `W0_ready`), `vhead` scalar, `whead (n_ω,)` (+attr `omega_grid`).
- Writes HDF5 per-proc shards `{prefix}.rank{R}.x{cx}.y{cy}.h5` (`V_local`, `S_local`, `V0_noG0_local`, `psi_full_local`, `enk_full`, grid/coord attrs).

### Suspects
- **dead**: `save_restart_state_per_proc` output — written every ISDF run, no reader found (grep evidence above); pure disk cost. Also dead arg `band_slices` in `load_restart_state_from_h5` and dead arg `meta` in `save_restart_state_per_proc`.
- **redundancy**: `save_restart_state_per_proc` re-implements slab block-slicing that SlabIO already owns; `read_restart_state_from_h5` (eager replicated h5py) is a parallel read path next to the SlabIO write path — the read side never uses SlabIO, so restart reads are unsharded full-file loads on every rank.
- **weird**: three copies of the `try: sync_global_devices(...) except Exception: pass` pattern (lines 96–100, 120–124, 150–154/167–171) — swallowed exceptions around a collective barrier; `W0_ready` written outside SlabIO "to stay compatible with that reader" (comment lines 90–92); back-compat 8-D/6-D collapse (lines 226–230) and `G0_mu_nu[0]` row-pick (240) encode two dead legacy formats inline; module name/docstring mismatch ("tagged arrays" vs restart state).

---

## src/file_io/kin_ion.py (86 LOC)

**Purpose.** Loads the band sub-window of the precomputed one-body Hamiltonian matrix elements `kin_ion = H_DFT − V_xc` (kinetic + local/nonlocal ionic; Hartree only if added at generation time) from `kin_ion.h5`, fully replicated on the device mesh. Category: **I/O: GW input loader (H0 matrix elements)**.

### Functions

| Function | Lines | Role |
|---|---|---|
| `load_kin_ion_submatrix` | 17–86 | Validates half-open `[band_start, band_stop)` window, peeks dataset shape with h5py (dataset key `"kin_ion"`, shape (nk, nb_total, nb_total), must be square), then `SlabIO.read_slab("kin_ion", shape=(nk,nb,nb), offset=(0,band_start,band_start), dtype=complex128, partition_spec=P(None,None,None))` → replicated jax.Array. Diagonal-block read: same offset on both band axes. Physics: ⟨mk\|T + V_ion\|nk⟩ = ⟨mk\|H_DFT − V_xc\|nk⟩ (docstring warns not to subtract V_xc again downstream). Caller: `src/gw/gw_jax.py:467`. |

### Cross-module deps
`file_io.slab_io.SlabIO` (backend parity with zeta_q.h5 / sigma_omega.h5 stack), h5py shape-peek, jax mesh/PartitionSpec.

### Flags consumed
`kin_ion_file` (cohsex.in key, default `kin_ion.h5`, resolved by `paths.resolve_input_paths`); `backend` = `cfg.backend.slab_io` at the gw_jax call site.

### I/O
Reads `kin_ion.h5`, dataset `kin_ion (nk, nb_total, nb_total)` complex128. (Generation side lives elsewhere — psp/nscf tooling.)

### Suspects
- none dead; single caller, single purpose.
- **weird**: mixed access pattern — raw h5py open for the shape peek then SlabIO for the actual read (comment says FFI backend needs explicit shape); replicated read routed through the slab machinery purely for "backend parity", i.e. a (nk,nb,nb) fully-replicated array doesn't need slab I/O at all.

---

## src/file_io/paths.py (29 LOC)

**Purpose.** Resolves relative file paths in the parsed `cohsex.in` parameter dict against the input file's directory. Category: **I/O: input-file path plumbing**.

### Functions

| Function | Lines | Role |
|---|---|---|
| `resolve_input_paths` | 5–28 | In-place (and returned) resolution of the fixed key list `["wfn_file", "centroids_file", "kin_ion_file", "sigma_diag_file", "eqp0_file", "eqp1_file"]`: relative → `os.path.join(input_dir, path)`. Callers: `src/gw/gw_config.py:849` (inside cohsex.in parsing), `scripts/checks/sigma_direct_check.py:374`, `scripts/checks/w_from_eps0_0d_check.py:432`, `tests/archive/test_chunked_wfn_loading.py:714`. |

### Suspects
- **weird**: hard-coded key list must be kept in sync with gw_config path-type keys by hand — output paths (`sigma_omega_h5_file`, `sigma_kij_h5_file`, `eqp1_file` vs restart tmp dir, zeta paths, etc.) are NOT in the list, so some path-typed config keys resolve relative to input dir and others relative to cwd. Silent inconsistency risk; candidate to fold into gw_config's schema.

---

## src/file_io/centroids.py (27 LOC)

**Purpose.** Loads ISDF centroid fractional coordinates from a text file and converts to integer FFT-grid indices with a periodic wrap. Category: **I/O: ISDF centroid loader (preprocessing input)**.

### Functions

| Function | Lines | Role |
|---|---|---|
| `load_centroids` | 5–26 | `np.loadtxt(centroids_file)` → `centroids_frac (n_rmu, 3)`; `centroid_indices = round(frac * fft_grid)`; wrap: only `index == fft_grid[i]` → 0 (handles frac exactly 1.0; NOT a general `mod` — negative or >1 fractions would pass through as out-of-range indices). Returns `(centroids_frac, centroid_indices, n_rmu)`. Callers: `src/gw/gw_jax.py:138` (charge centroids), `src/gw/gw_init.py:772` and `:963` (transverse/bispinor centroid set), `src/bandstructure/htransform.py:588`, `tests/archive/test_chunked_wfn_loading.py:727`. |

### Flags consumed (by callers)
`centroids_file` (cohsex.in), bispinor transverse centroid file key in gw_init.

### I/O
Reads whitespace text file of fractional coords (kmeans/centroid generator output).

### Suspects
- **redundancy**: `src/bandstructure/htransform.py:473` defines a private `_load_centroids` that duplicates this (with `ndmin=2`, empty-file check, and a more robust `np.mod` wrap) — and htransform ALSO imports the shared loader at line 578/588; the private one at 473 has no remaining call site in htransform (grep `_load_centroids` in htransform: def only + the aliased shared import). Dead duplicate to delete; if merging, keep the `np.mod` + `ndmin=2` behavior.
- **weird**: single-row centroid files break (`np.loadtxt` without `ndmin=2` returns shape (3,) → `n_rmu=3`, garbage); wrap handles only the ==fft_grid edge case.

---

## src/file_io/__init__.py (46 LOC)

**Purpose.** Package facade for `file_io`: re-exports loaders/writers/readers and carries the `WFNReader = WfnLoader` back-compat alias. Category: **I/O: package facade / back-compat shim**.

### Contents
- Imports/re-exports: `WfnLoader`, `ZetaLoader`, `ZetaReader` (comment: "legacy slab-only reader (kept for now)"), `WFNReader = WfnLoader` alias (lines 17–22, comment says a follow-up commit will sweep call sites to `WfnLoader`), `CrystalData`, `WFNWriter`, `EPSReader`, six `tagged_arrays` functions, six `sigma_output` functions, `write_qp_rotations_h5`, `load_kin_ion_submatrix`, `load_centroids`, `resolve_input_paths`, `read_bgw_vcoul` / `fill_v_grid_for_q` / `BGWVcoulTable`.
- Docstring module list (lines 4–11) is stale: omits `slab_io`, `zeta_loader`/`zeta_reader`, `read_bgw_vcoul`, `sigma_output`, `paths`.

### Suspects
- **redundancy (self-documented)**: `WFNReader` alias and legacy `ZetaReader` both explicitly marked as transitional; refactor should sweep the ~20 `WFNReader` import sites (e.g. `tests/archive/*`, `scripts/checks/*`, `src/gw/gw_jax.py:21`) and delete the alias, and decide `ZetaReader`'s fate.
- **weird**: `write_qp_wfn_h5` not exported although its sibling is — call sites use `from file_io.qp_wfn import write_qp_wfn_h5`; export surface is inconsistent (`write_chunked_complex_dataset_h5` IS exported despite being dead).
