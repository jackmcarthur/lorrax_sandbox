# src/bse/bse_io.py (932 LOC) + write_eigenvectors.py (236 LOC) + eigenvectors.h5.spec (172 LOC) — deep-read notes

Audit date: 2026-07-15, lorrax_D checkout. Stated base: agent/slate-linalg-ffi @ e18d0e5.
Working tree was on agent/ppm-fit-conditioning @ adc2197 at audit time, but
`git diff --stat e18d0e5 HEAD -- src/bse/ src/file_io/tagged_arrays.py src/runtime/padding.py src/gw/head_correction.py`
is **empty** — the audited files are bit-identical to e18d0e5.

## Purpose

BSE data-plane I/O: (a) loaders that turn the canonical gw_jax restart bundle
(`isdf_tensors_*.h5`: `V_qmunu`, `W0_qmunu`, `psi_full_y`, `enk_full`,
`G0_mu_nu`, `vhead`, `whead`, `kgrid`) into the ψ/ε/V/W tensors the BSE
solvers consume — one sharded per-slab loader for the (x,y) 2-D mesh, one
whole-array single-device "ring" loader; (b) BGW `eqp1.dat` QP-correction
readers/appliers; (c) two writers of BGW-format `eigenvectors.h5`
(streaming writer in bse_io.py, legacy in-memory writer in
write_eigenvectors.py); (d) the verbatim BGW spec file.

Physics as written in code:

```
q=0 head reinstatement (compute_vcoul zeroed v(q=0, G=G'=0); BGW BSE uses the
mini-BZ-averaged 1/q² there; reinstated rank-1 in the ISDF centroid basis,
bse_io.py:804-836 / 463-513 → gw/head_correction.py:766-775, 805-814):
  V_q0[μ,ν]          += v_scalar · conj(G0[μ]) · G0[ν]      G0[μ] = ζ(q=0, μ, G=0)
  W_q[μ,ν,0,0,0]     += w_scalar · conj(G0[μ]) · G0[ν]      (q=0 slice only)
  v_scalar/w_scalar from vhead/whead (Ry, BGW wcoul0 convention) and 1/V_cell.
  Source priority: cohsex.in `vhead`/`whead_0freq` override restart datasets.

EQP substitution (apply_eqp_corrections, bse_io.py:698-753):
  enk_qp[k_full, b] = e_qp_ibz[ sym.irr_idx_k[k_full], b ] / 13.6056980659   (eV→Ry)
  for b < nb_eqp where e_qp_ibz is not NaN; fallback (no input_file): fuzzy
  IBZ match by max|e_dft| difference < 0.01 eV.

Band slicing (both loaders):
  val_indices  = arange(n_occ - n_val, n_occ)      # v=0 is the DEEPEST valence
  cond_indices = arange(n_occ, n_occ + n_cond)     # c=0 is the lowest conduction
  n_occ from resolve_n_occ: explicit n_occ  >  WFN ifmax (WfnLoader.nelec =
  max(ifmax), wfn_loader.py:162)  >  count(mean_k enk < fermi_energy).

W layout transform, flat-q → 3-D-k ("the ONE place the 3-D-k form
materialises inside BSE", bse_io.py:906-912; sharded analog 321-324):
  W_q[μ, ν, i, j, l] = W_flat[ q = i·nky·nkz + j·nkz + l , μ, ν ]
  (reshape(nkx,nky,nkz,μ,ν) then transpose(3,4,0,1,2))

BGW-compat on eigenvector write (write_eigenvectors_stream, bse_io.py:23-105):
  eigenvalues_file        = eigenvalues_Ry[:n_write] · 13.6056980659      (Ry→eV)
  file[0, i, k, c, v, s]  = vec_i[k, c, nv-1-v]                            (valence flip:
  v_file=0 = highest valence, BGW BSE/input_fi.f90:407; conduction and k axes unflipped)
  dataset shape (C order) = (nQ=1, nevecs, nk, nc, nv, ns=1, 2) — matches the
  Fortran spec dims [2, ns, nv, nc, nk, nevecs, nQ] reversed (spec lines 121-132).
```

Category: **pipeline stage (BSE input assembly + BGW-compat output)**.

## Entry points (grep over src/, tests/, tools/, scripts/, docs/ at HEAD; plus sandbox runs/, skills/, scripts/)

| symbol | callers (grep evidence) |
|---|---|
| `load_bse_data_from_restart_sharded` | `bse_jax.py:227`, `bse_feast.py:24`, `bse_pseudopoles.py:35`, `feast_sweep.py:26`, `feast_zolo_sweep.py:24`, `feast_ellipse_mixed_sweep.py:19`, `bse_kpm.py:29`, `bse_w_exact.py:14`, `bse_ring_comm.py:24`, `absorption_haydock.py:50`, `davidson_absorption.py:34-37`, `test_davidson_bse.py:40` |
| `_load_ring_subset` (private) | `bse_jax.py:27` (used at :296, single-device Lanczos), `bse_ring_comm.py:24`; production evidence: `runs/Si/04_si_4x4x4_bse/01_lorrax_bse_vcoul/profile_bse/compile.log:2` traces `(_load_ring_subset)` |
| `_find_restart_file` (private) | same 10+ importers as the two loaders (`grep "_find_restart_file"`) |
| `write_eigenvectors_stream` | `bse_jax.py:39`, called at `bse_jax.py:335` (`--write-eigs`); documented workflow `src/bse/BGW_COMPARE.md:59-62` (`python -m bse.bse_jax ... --write-eigs 100`) |
| `apply_eqp_corrections` | `bse_jax.py:243`, `davidson_absorption.py:34`, `absorption_haydock.py:178`, `test_davidson_bse.py:77` |
| `resolve_n_occ` | `bse_jax.py:243`, `davidson_absorption.py:36`, `absorption_haydock.py:175` |
| `read_bgw_eqp` | internal (`apply_eqp_corrections`); `tests/test_eqp_bgw.py:78,83`; named in `src/gw/eqp_bgw.py:51` docstring |
| `_pad_axis_to_multiple` (private) | `bse_jax.py:259`, `davidson_absorption.py:105`, `absorption_haydock.py:178`, `test_davidson_bse.py:77` (all discard the 2nd retval) + both loaders here |
| `BSEData` | **NONE FOUND** (grep `BSEData` across src/tests/tools/scripts/docs and sandbox runs/skills/scripts → only the definition, bse_io.py:18) |
| `write_eigenvectors_h5` | `test_bse.py:39,361` only; CLI `python -m bse.write_eigenvectors <npz>` (main, write_eigenvectors.py:176-235) — no run-dir/skill invocation found (sandbox grep hits for "write_eigenvectors" are the unrelated BGW `absorption.inp` keyword) |
| `generate_kpts_grid` | `bse_io.py:34,42` (stream writer — production), `test_bse.py:39,360` |

`python -m bse.bse_jax` is the CLI that reaches everything here
(`bse_jax.py:349 __name__ == "__main__"`); `python -m bse.test_bse` is the
documented test driver (test_bse.py docstring line 4). No skills/ or
sandbox scripts/ reference these modules.

## Function table

### `BSEData(SimpleNamespace)` — bse_io.py:18-20
Empty container class. Zero users anywhere (see entry table). Dead.

### `write_eigenvectors_stream(output_file, eigenvalues, eigenvectors, n_val, n_cond, nkx, nky, nkz, n_write)` — bse_io.py:23-105
- The production BGW-compliant `eigenvectors.h5` writer: eigenvalues Ry→eV
  (comment cites BGW BSE/diag.f90, lines 36-40), valence axis flipped on
  write (lines 95-100, cites BSE/input_fi.f90:407), streams one eigenvector
  per `jax.device_get` (line 88) so host memory = one vector.
- Accepts per-vector shape `(1, nc, nv, nk)` (sharded path, leading block
  axis squeezed at 92-93) or `(nc, nv, nk)`; transposes `(2,0,1)` → `(nk,nc,nv)`.
- Hardcodes: `ns=1`, `nQ=1`, `flavor=2`, `spin_kernel=3`, `use_tda=1`
  (lines 44-47, 71). k-points regenerated via `generate_kpts_grid`, not read
  from the restart — assumes the solver's flat-k order is the ix-outer /
  iz-inner Γ-centered [0,1) MP grid.
- **Shape contract hazard**: dataset is sized `(1, n_write, nk, n_cond, n_val, ns, 2)`
  from the *caller's* n_val/n_cond (bse_jax passes the user-requested values,
  bse_jax.py:335-345), but the vectors carry the solver dims
  `(n_eig, bs=1, nc_pad, nv_pad, nk)` (bse_lanczos.py:229). Any clamp
  (loader reduced n_val to what is available, bse_io.py:426) or band pad
  (mesh-divisibility, bse_io.py:449-453) makes the h5py assignment at
  102-103 raise on shape mismatch. Works only when requested == available
  == padded (the usual power-of-two case).

### `_pad_last_axis` / `_pad_last_two_axes` / `_pad_first_two_axes` — bse_io.py:108-136
Zero-pad helpers (μ axes) up to `target`; no-ops when already big enough.

### `_pad_axis_to_multiple(x, axis, multiple) -> (padded, size)` — bse_io.py:139-146
Pads `axis` up to the next multiple. **Returns the PRE-pad size** in both
branches (`size = x.shape[axis]` captured before `jnp.pad`, returned
verbatim at 143 and 146). See Suspects — the two loaders bind this to
`n_val_pad`/`n_cond_pad`, which downstream reads as the *padded* extent.

### `_get_local_mesh_coords` / `_get_local_axis_coords` / `_assert_local_block` — bse_io.py:149-171
Map `jax.local_devices()` onto the 2-D mesh; require the local device set
to be a full Cartesian x×y block (raise otherwise, 168-171). Contiguity of
local coords is NOT checked — `make_array_from_process_local_data` with a
process owning non-adjacent x-rows would interleave silently (multi-host
edge case; single-host always fine).

### `_read_psi_mu_sharded(dset, band_indices, mu_per_shard, axis, mesh_xy, n_rmu_pad, dtype, trim)` — bse_io.py:174-218
- Reads `psi_full_y` `(nk, nb, nspinor, n_rmu)` per local μ-slab:
  `dset[:, band_indices, :, mu_start:mu_end]` per owned axis coord, zero-pads
  the last slab to `mu_per_shard`, assembles
  `jax.make_array_from_process_local_data` with `P(None,None,None,axis)`.
- Full k and full band-slice per process; only μ is sharded on read.
- `trim=True` default would slice back to logical n_rmu; **every call site
  passes `trim=False`** (bse_io.py:446-447) — dead branch in practice.

### `_read_vq0_sharded(dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad, dtype, trim)` — bse_io.py:221-265
- Reads the q=0 bare-Coulomb block into `(n_rmu_pad, n_rmu_pad)` `P("x","y")`.
- **Hardwired to the legacy 8-D layout**: `n_rmu = dset.shape[6]`,
  `n_rnu = dset.shape[7]` (233-234) and the slab read is
  `dset[0, 0, 0, 0, 0, 0, mu_start:mu_end, nu_start:nu_end]` (250) — i.e. it
  assumes `(1, npol, npol, nkx, nky, nkz, μ, ν)` and q=0 at k-index (0,0,0).
  For the CURRENT flat-q 3-D `V_qmunu` (`(nq, μ, μ)` — gw_init.py:277
  docstring, gw_jax.py:264 "flat-q (nq, μ, μ) — compute and restart alike")
  `dset.shape[6]` raises IndexError. See Suspects (bug).

### `_read_wq_sharded(dset, mu_per_x, nu_per_y, mesh_xy, n_rmu_pad, dtype, trim)` — bse_io.py:268-355
- Reads W (or V fallback) into `(n_rmu_pad, n_rmu_pad, nkx, nky, nkz)`
  `P("x","y",None,None,None)`, converting on read via the layout shim:
  8-D legacy → `np.transpose(dset[0,0,0,:,:,:,μ,ν], (3,4,0,1,2))` (291-292:
  out[m,n,i,j,l] = dset[0,0,0,i,j,l,mu0+m,nu0+n]); 6-D transitional strips
  `[0,0,0]`; 3-D flat-q reshapes as per the Purpose formula (321-324).
- For 6-D/3-D, kgrid must come from `dset.attrs['kgrid']` (309-310), else
  ValueError (312-317). **The writer never sets that dataset attr** — it
  writes `kgrid` as a top-level rank-0 dataset (tagged_arrays.py:81-82 via
  `io.write_attr`, which is "Write a small rank-0-only dataset",
  slab_io.py:190-196); the outer loader duly reads `f['kgrid']` (398-401)
  but doesn't forward it here. The in-code comment (313-317) itself flags
  this as an open follow-up. See Suspects.
- Error text and the comment at 388 name `_load_per_axis_padded_w_block`, a
  function that exists nowhere in the repo (grep) — stale pre-rename name.

### `load_bse_data_from_restart_sharded(restart_file, n_val=4, n_cond=4, fermi_energy=0.0, mesh_xy=None, pad_bands=True, *, input_file=None, cell_volume=None, n_occ=None) -> dict` — bse_io.py:358-536
- The multi-GPU loader (used whenever `jax.device_count() > 1`,
  bse_jax.py:222-236). Steps:
  1. `W0_qmunu` chosen over `V_qmunu` only if its `W0_ready` attr is truthy
     (376-379); silent fallback to bare V otherwise (RPA-with-V screening,
     no warning printed).
  2. kgrid resolution for flat-q: `vq_dset.attrs['kgrid']` → top-level
     `f['kgrid']` dataset → WFN via `input_file`'s `wfn_file` (389-411).
  3. `resolve_n_occ` (415-418; `fermi_energy=0.0` means "unset" — a true
     Fermi level of exactly 0.0 Ry cannot be passed), clamp n_val/n_cond to
     availability with printed warnings (423-427).
  4. μ re-pad to the ONE in-memory convention:
     `n_rmu_pad = padded_mu_extent(n_rmu, grid_x*grid_y)` (442,
     runtime/padding.py:67 = round_up; disk stores LOGICAL extent per
     tests/test_restart_pad_roundtrip.py docstring).
  5. ψ loads: both valence and conduction read with `mu_per_x` on axis "x"
     (446-447), then **Y-copies made by `with_sharding_constraint` to
     `P(None,None,None,"y")`** (457-458) — a device-side reshard, not a
     second disk read.
  6. Band pad to mesh multiples: valence→grid_y, conduction→grid_x
     (450-453; eps padded with **zeros**). `pad_bands=False` branch has no
     caller anywhere (grep "pad_bands" → only this file + unrelated
     psi_G_store/wfn_transforms locals).
  7. `V_q0` and `W_q` sharded reads (460-461) — crash on current-format
     files via `_read_vq0_sharded` (see Suspects).
  8. q=0 head: `G0_mu_nu` zero-padded to n_rmu_pad and device_put **twice**
     — `g0_X` under `P("x")`, `g0_Y` under `P("y")` (474-477) so the rank-1
     `conj(g0_X)[μ_loc]·g0_Y[ν_loc]` is local on every proc (comment
     463-467); `apply_q0_head_rank1_sharded` adds `v_scalar·g0g0` to `V_q0`
     and `w_scalar·g0g0` to `W_q[:,:,0,0,0]` (head_correction.py:805-814).
     cohsex.in `vhead`/`whead_0freq` override restart values (487-492);
     cell_volume from arg or WFN (495-501), head silently skipped if
     unresolvable.
- Returns dict: `psi_{c,v}_{X,Y}`, `eps_c/eps_v` (replicated), `W_q`,
  `V_q0`, `g0_X/g0_Y`, `nkx/nky/nkz`, `n_rmu`, `n_rmu_pad`,
  `n_val/n_cond`, `n_val_pad/n_cond_pad`, `fermi_energy`.

### `read_bgw_eqp(eqp_file)` — bse_io.py:539-581
BGW `eqp1.dat` parser: per-k header `kx ky kz nb`, band rows with
`cols[2]=E_dft(eV)`, `cols[3]=E_qp(eV)`; NaN-padded ragged blocks.
(BGW-convention file; energies stay eV until `apply_eqp_corrections`
divides by 13.6056980659.)

### `_parse_wfn_path(input_file)` — bse_io.py:584-600
Extracts `wfn_file` from cohsex.in (`key = value` lines, default `WFN.h5`),
resolves relative to the input file's directory.

### `resolve_n_occ(enk_full, *, n_occ, input_file, fermi_energy)` — bse_io.py:603-664
Resolution order: explicit `n_occ` → `WfnLoader(wfn).nelec`
(= `int(np.max(ifmax))`, wfn_loader.py:162 — a count of occupied BANDS,
not electrons, despite the name; matches the docstring's "ifmax —
authoritative") → `count(mean_k enk < fermi_energy)` sanity-bounded to
[1, nb-1]. Raises otherwise; docstring records that the old "largest gap /
< 0" auto-detect silently broke on QE reference levels (e.g. Si semicore).

### `_parse_head_overrides(input_file)` — bse_io.py:667-695
Parses cohsex.in `vhead` and `whead_0freq` as `complex(float(val))` —
NB a genuinely complex string ("3303.7+0.5j") hits `ValueError` and is
silently ignored (693-694). Ry units, BGW wcoul0 convention
(STATUS.md fair-compare table: `vhead = 3303.748` = BGW's wcoul0).

### `apply_eqp_corrections(enk_full, eqp_file, input_file, ry_to_ev)` — bse_io.py:698-753
Formula in Purpose. With `input_file`: exact IBZ unfold via
`SymMaps.irr_idx_k` with asserts `nk_tot == nk_full`, `nk_ibz == nk_red`
(717-720). Without: fuzzy eigenvalue matching (tol 0.01 eV); the `matched`
bool array (729, 748) is written but never read — dead diagnostics.

### `_find_restart_file(input_file)` — bse_io.py:756-764
Globs `tmp/isdf_tensors_*.h5` then `isdf_tensors_*.h5` next to the input
file; returns the lexicographically first hit — multiple restart
generations in one dir resolve to the oldest name, silently.

### `_load_ring_subset(restart_file, n_val, n_cond, px, py, eqp_file, n_occ, input_file) -> dict` — bse_io.py:767-932
- Single-device path (bse_jax.py:296 with px=py=1). Loads **everything
  whole** onto the default device: `jnp.asarray(f["V_qmunu"][:])`,
  full `psi_full_y`, `W0_qmunu` (779-796) — no sharding, no slab reads.
- EQP applied to host enk before slicing (799-800).
- q=0 head injection (811-836) — **before** the axis-shape shim (846-874):
  `apply_q0_head_rank1` updates `V_qmunu.at[..., 0, 0, 0, :, :]`
  (head_correction.py:773), i.e. the last five axes must be
  `(nkx, nky, nkz, μ, ν)`-like. Fine for 8-D legacy; for 6-D it silently
  indexes `(npol, npol, nq)` instead (correct only because npol=1); for the
  CURRENT 3-D flat-q it is 5 indices on 3 axes → IndexError. See Suspects.
- Shim (846-874): 8-D → `[0,0,0].reshape(-1, μ, μ)`; 6-D → `[0,0,0]`; 3-D
  passthrough; kgrid from shape (8-D) or WFN via input_file (861-863;
  raises if input_file is None — the top-level `f['kgrid']` dataset the
  sharded loader consults is NOT checked here).
- μ re-pad `padded_mu_extent(n_rmu, px*py)` (877), band pads to px/py
  multiples (899-902; same stale-size binding as the sharded loader),
  `V_q0 = V_qmunu[0]` (904), W built per the Purpose reshape (911) from
  `W0_qmunu` or V fallback (906).
- Seeds a random trial block `X` with **the same PRNG key for real and
  imaginary parts** (914-917): `real(X) == imag(X)` elementwise, i.e. a
  constant-phase e^{iπ/4} start vector — harmless for spectra but loses the
  imaginary DOF of the start block.

### write_eigenvectors.py

#### `write_eigenvectors_h5(output_file, eigenvalues, eigenvectors, kpts, n_val, n_cond, nkx, nky, nkz, version, exciton_Q_shifts)` — lines 21-153
- Legacy in-memory writer. Accepts `(n_eig, nc, nv, nk)` or
  `(n_eig, ns, nc, nv, nk)`; per-element reorder (line 96):
  `evecs_reordered[e, k, c, v, s] = eigenvectors[e, s, c, v, k]`, then
  `[np.newaxis]` for nQ and re/im stack → C-shape
  `(1, n_eig, nk, nc, nv, ns, 2)` — axis order matches the spec (reversed
  Fortran dims), same as the stream writer.
- **No valence flip and no Ry→eV conversion**: eigenvalues are stored raw
  (docstring line 39: "exciton energies in Ry") and `v` keeps the LORRAX
  internal order (v=0 = deepest valence). Both violate the BGW file
  conventions this file claims to implement (STATUS.md:27-34, which states
  the *stream* writer is the compliant one). Any BGW-convention consumer of
  a file from this writer reads energies 13.6× off and mismatched valence
  indices. Test/CLI-only reachable (test_bse.py:361; `python -m
  bse.write_eigenvectors`).
- Computes `flavor` from `iscomplexobj` (75), hardcodes `spin_kernel=3`,
  `use_tda=1`; imports `jax.numpy as jnp` (line 18) that is never used.

#### `generate_kpts_grid(nkx, nky, nkz)` — lines 156-173
Γ-centered [0,1) MP grid, flat order ix-outer / iz-inner
(`k = ix·nky·nkz + iy·nkz + iz`) — must (and does) match the flat-q order
assumed by the W reshape in bse_io. The `iz / nkz if nkz > 0 else 0.0`
guard (171) is dead: `range(nkz)` is empty when nkz == 0.

#### `main()` — lines 176-235
argparse CLI: positional `input_npz` (needs `eigenvalues` + `eigenvectors`
arrays), `-o/--output` (default `eigenvectors.h5`), `--n-val` (required),
`--n-cond` (required), `--nkx/--nky/--nkz` (default 1).

### eigenvectors.h5.spec — 172 lines
Verbatim BGW spec, Fortran dim order (header warning lines 1-2). Documents
the four eigenvector datasets (`eigenvectors`, `eigenvectors_left`,
`eigenvectors_deexcitation`, `eigenvectors_deexcitation_left`); LORRAX
writers only ever emit the first (TDA-only), plus `exciton_Q_shifts`
semantics (valence-shifted, exciton momentum −Q).

## Sharding / PartitionSpec assumptions

| array | shape | spec | where set |
|---|---|---|---|
| `psi_v_X` / `psi_c_X` | `(nk, nb_pad, ns, n_rmu_pad)` | `P(None,None,None,"x")` | bse_io.py:213 |
| `psi_v_Y` / `psi_c_Y` | same | `P(None,None,None,"y")` via `with_sharding_constraint` | bse_io.py:457-458 |
| `V_q0` | `(n_rmu_pad, n_rmu_pad)` | `P("x","y")` | bse_io.py:260 |
| `W_q` | `(n_rmu_pad, n_rmu_pad, nkx, nky, nkz)` | `P("x","y",None,None,None)` | bse_io.py:350 |
| `g0_X` / `g0_Y` | `(n_rmu_pad,)` | `P("x")` / `P("y")` (dual copies for local rank-1) | bse_io.py:474-477 |
| `eps_c` / `eps_v` | `(nk, nb_pad)` | unconstrained (`jnp.asarray`) | bse_io.py:436-437 |
| ring path (`_load_ring_subset`) | all | unsharded, single default device | bse_io.py:779-917 |

Mesh contract: 2-D `Mesh` with axes `("x","y")`; per-process local devices
must form a Cartesian block (bse_io.py:164-171). Band axes must end up
divisible by grid_x (conduction) / grid_y (valence) for the solver-side
`P(None,"x","y",None)` trial vectors (bse_ring_comm.py:736-737 enforces).

## Host-vs-device residency

- Sharded loader: h5py slab reads per local μ-block into host numpy →
  `device_put` per process → `make_array_from_process_local_data`; the
  global tensors never materialize on one host. ψ reads are full-k,
  full-band-slice per process (only μ sharded on read).
- Ring loader: `f["V_qmunu"][:]` / full `psi_full_y` straight to one
  device — deliberate small-system path.
- `write_eigenvectors_stream`: one `jax.device_get(eigenvectors[i])` per
  vector (bse_io.py:88) — host peak is a single (nc,nv,nk) vector.
  `write_eigenvectors_h5` holds all n_eig vectors + a re/im-stacked copy in
  host memory.

## TDA vs full-BSE

Loaders are kernel-agnostic (they only provide ψ/ε/V/W). Both writers
hardcode `use_tda=1` and write only the right-eigenvector dataset; the
spec's non-TDA datasets (`eigenvectors_left`, `*_deexcitation*`) are never
produced, even though `bse_jax.py` exposes `--tda` with "Default is full
non-TDA" for the FEAST path. `evec_sz = bse_hamiltonian_size` (TDA value)
is likewise hardcoded (bse_io.py:48-49, write_eigenvectors.py:71-72).

## Spin / nspinor

- Loaders carry the `psi_full_y` nspinor axis opaquely
  (`nspinor = dset.shape[2]`, bse_io.py:190; ring path keeps axis 2) — no
  spin-dependent logic anywhere in this file.
- Writers hardcode `ns=1` + `spin_kernel=3` ("spinor calculations" per spec
  line 52). For non-spinor runs (the Si 4×4×4 BGW comparisons) the file's
  spin metadata is wrong-but-harmless for eigvec comparison tooling;
  a consumer branching on `spin_kernel` (singlet=1 vs spinor=3) would
  misinterpret.

## Coupling to gw/ and isdf/ modules

- **Producer contract** (gw side): `gw_init.py:661-667` writes the restart
  bundle via `file_io.tagged_arrays.write_restart_state_to_h5` — flat-q
  `V_qmunu (nq, μ, μ)` (gw_init.py:277), `G0_mu_nu`, `enk_full`,
  `psi_full_y` (appended, :693-698), zeros `W0_qmunu` placeholder with
  `W0_ready` attr (tagged_arrays.py:115-120), `kgrid` as a top-level
  rank-0 dataset (tagged_arrays.py:81-82). vhead/whead appended by the
  Phase-B writer (tagged_arrays.py:160-191).
- `gw.head_correction.apply_q0_head_rank1` (:743) / `_sharded` (:779) —
  the actual rank-1 head math.
- `file_io.WfnLoader` — kgrid, cell_volume, `nelec = max(ifmax)`.
- `common.symmetry_maps.SymMaps` — `irr_idx_k` for the eqp IBZ unfold.
- `runtime.padding.padded_mu_extent` — the ONE in-memory μ-pad convention
  (mesh-product round-up; honors test-only `LORRAX_EXTRA_MU_PAD`).
- No direct isdf/ imports; the ISDF coupling is entirely through the
  restart file contract (μ = ISDF centroid index throughout).

## Flags / config keys consumed by these files

- cohsex.in `wfn_file` — path to WFN.h5, relative to the input dir — default `WFN.h5` (bse_io.py:584-600).
- cohsex.in `vhead` — v(q→0, G=G'=0) in Ry (BGW wcoul0); overrides restart `vhead` (bse_io.py:689).
- cohsex.in `whead_0freq` — W head at ω=0 in Ry; overrides restart `whead[0]` (bse_io.py:691).
- `write_eigenvectors.py` CLI: `input_npz` (positional), `-o/--output` (default `eigenvectors.h5`), `--n-val` (required), `--n-cond` (required), `--nkx/--nky/--nkz` (default 1).
- No env vars read directly (`LORRAX_EXTRA_MU_PAD` is read inside `runtime.padding`, not here).
- Everything else (`--n-val/--n-cond/--eqp/--n-occ/--write-eigs/...`) is
  bse_jax.py's argparse, threaded in as function arguments.

## Suspects

### Bugs (confirmed by explicit index math at HEAD)

1. **`n_val_pad`/`n_cond_pad` are the UNPADDED sizes whenever padding
   actually occurs** — bse_io.py:139-146 returns `size = x.shape[axis]`
   captured *before* `jnp.pad`, in both branches; the loaders bind that to
   `n_val_pad`/`n_cond_pad` (bse_io.py:450-451, 899-900) and export them in
   the data dict. Every downstream consumer treats these as the padded
   trial-vector extents: `bse_lanczos.py:140-148`
   (`shape = (bs, nc_pad, nv_pad, nk)`), `bse_feast.py:558`,
   `bse_kpm.py:79`, `davidson_absorption.py:115-116`, and
   `bse_ring_comm.py:736-737` raises unless `n_cond_pad % px == 0`.
   Failure math: mesh 2×2, `n_val = 5` → `psi_v_X` padded 5→6 (grid_y=2)
   but `data["n_val_pad"] = 5`; `bse_ring_comm.py:737` raises
   "n_cond_pad and n_val_pad must be divisible by px/py" even though the
   loader *just* padded ψ to be divisible; the "simple" matvec path instead
   hits an einsum dim mismatch (X band dim 5 vs ψ band dim 6). Net effect:
   the whole `pad_bands` mechanism can never deliver a working
   non-divisible run — band counts must divide the mesh exactly. Masked to
   date because standard runs use nv=nc=4 or 8 on 1/2/4-GPU meshes.
   Plausibly the "CrI3 small-nbnd band-sharding death mode" flagged in the
   sandbox memory.

2. **`_read_vq0_sharded` is hardwired to the legacy 8-D layout while the
   current restart format is flat-q 3-D** — bse_io.py:233-234
   (`n_rmu = dset.shape[6]`) and :250
   (`dset[0, 0, 0, 0, 0, 0, μ0:μ1, ν0:ν1]`). The current GW writer emits
   `V_qmunu` as `(nq, μ, μ)` (gw_init.py:277 "V_qmunu has shape (nq, μ, μ)
   (flat-q)"; gw_jax.py:264 "flat-q (nq, μ, μ) — compute and restart
   alike"). Failure: `load_bse_data_from_restart_sharded` on any
   current-format restart → `IndexError: tuple index out of range` at
   :233, *after* the outer function's own compat shim (389-411) has
   successfully resolved kgrid for exactly this layout. The multi-GPU BSE
   path is dead against current restarts; only pre-flat-q files (e.g. the
   April `runs/Si/04_si_4x4x4_bse` era) load. No test covers this reader
   (tests/test_restart_pad_roundtrip.py exercises
   `file_io.load_restart_state_from_h5`, not bse_io).

3. **`_read_wq_sharded` flat-q branch can't see the kgrid the writer
   actually stores** — bse_io.py:309-317 accepts kgrid only as
   `dset.attrs['kgrid']` and raises otherwise, but the writer stores kgrid
   as a top-level rank-0 *dataset* (tagged_arrays.py:81-82 via
   `io.write_attr`, which per slab_io.py:190-196 writes "a small
   rank-0-only dataset"), which the outer loader duly reads at
   bse_io.py:398-401 and then fails to forward. Once bug 2 is fixed, a
   flat-q `W0_qmunu` read raises the :312 ValueError despite kgrid being
   resolvable two frames up. (The error text also names the nonexistent
   `_load_per_axis_padded_w_block` — see cruft.)

4. **`_load_ring_subset` injects the q=0 head before the layout shim** —
   head call at bse_io.py:828 on the raw on-disk array; the shim that
   normalizes 8-D/6-D/3-D runs later (846-874).
   `apply_q0_head_rank1` updates `V_qmunu.at[..., 0, 0, 0, :, :]`
   (head_correction.py:773) — per-element only correct when the last five
   axes are `(nkx, nky, nkz, μ, ν)`: correct for 8-D; for 6-D
   `(1, npol, npol, nq, μ, ν)` it indexes `(npol=0, npol=0, nq=0)` —
   right answer only because npol=1; for the CURRENT 3-D flat-q it is
   5 indices on a 3-axis array → IndexError. Failure: single-device BSE on
   a current-format restart carrying `G0_mu_nu` + `vhead` (which the
   current writer always produces) crashes at :828. Fix direction: inject
   after the shim on `V_qmunu_flat` (q=0 is `[0]`).

5. **`write_eigenvectors_h5` writes a BGW-format-claiming file with Ry
   eigenvalues and unflipped valence axis** — write_eigenvectors.py:141
   stores `eigenvalues` raw (docstring :39 says Ry) where BGW convention is
   eV (STATUS.md:30; bse_io.py:36-40), and no `v → nv-1-v` flip anywhere
   (contrast bse_io.py:100; STATUS.md:29 "BGW iv=1 is the highest valence").
   A consumer applying BGW conventions reads energies 13.6× off and
   valence-mirrored amplitudes with no error raised. Reachable via
   `test_bse.py --write-eigenvectors` and `python -m bse.write_eigenvectors`.

6. **`write_eigenvectors_stream` shape contract breaks under clamp/pad** —
   dataset sized from caller-supplied `n_val/n_cond` (bse_io.py:83) while
   vectors carry solver dims `(bs=1, nc_pad, nv_pad, nk)`
   (bse_lanczos.py:229); bse_jax.py:335-345 passes the *user-requested*
   n_val/n_cond, not the loader-clamped/padded ones. Requested 8 valence
   with 4 available → loader clamps (bse_io.py:426), solver returns nv=4,
   h5py assignment at :102 raises on shape mismatch. Loud crash, not silent
   corruption — but the writer should consume `data["n_val"]`/pad-stripped
   vectors (slice `[:n_cond, :n_val]` BEFORE the `::-1` flip).

### Dead

- `BSEData` (bse_io.py:18-20): zero references anywhere at HEAD (src,
  tests, tools, scripts, docs, sandbox runs/skills/scripts).
- `trim=True` default of the three `_read_*_sharded` readers: all four call
  sites pass `trim=False` (bse_io.py:446-447, 460-461).
- `pad_bands=False` branch (bse_io.py:454-456): no caller passes
  `pad_bands` (grep across src → only this file).
- `apply_eqp_corrections` `matched` array (bse_io.py:729, 748): written,
  never read.
- `generate_kpts_grid` `nkz > 0` guard (write_eigenvectors.py:171):
  unreachable (empty range at nkz=0).
- `import jax.numpy as jnp` in write_eigenvectors.py:18: unused.

### Redundancy

- **Two `eigenvectors.h5` writers with divergent conventions**:
  `bse_io.write_eigenvectors_stream` (eV, v-flip, streaming — the compliant
  one per STATUS.md:34) vs `write_eigenvectors.write_eigenvectors_h5` (Ry,
  no flip, in-memory, more general ns/nQ handling that nothing uses).
  Classic parallel old/new path; the old one is test/CLI-only. Fold or
  delete (bug 5 goes away with it).
- **Three restart loaders**: `load_bse_data_from_restart_sharded`,
  `_load_ring_subset`, plus a third private copy `load_bse_data_from_restart`
  living in `test_bse.py:42`. The two in this file duplicate n_occ
  resolution, clamping/warning, head-override parsing, and band-pad logic
  almost line-for-line (423-434 vs 879-891; 486-513 vs 811-836).
- The 8-D/6-D/3-D layout shim exists **three times** with different
  completeness: `_read_wq_sharded` (287-324, full), outer sharded loader
  (388-411, kgrid-only), `_load_ring_subset` (838-874) — and
  `_read_vq0_sharded` got none (bug 2).

### Weird

- The streaming writer lives in bse_io.py while the module named
  `write_eigenvectors.py` holds the legacy writer — file naming inverted
  relative to importance; stream writer also reaches back into
  write_eigenvectors for `generate_kpts_grid` (bse_io.py:34).
- `wq_dset = vq_dset` / `W_src = V_qmunu` fallback when W0 isn't ready
  (bse_io.py:376-379, 906) builds the BSE kernel with the **bare Coulomb as
  W** with no printed warning — physically an unscreened-direct-term run
  that looks like a normal completion.
- `_load_ring_subset` trial block `X` uses the same PRNG key for real and
  imaginary parts (bse_io.py:914-917) → `imag(X) == real(X)` elementwise.
- `_find_restart_file` returns the lexicographically first
  `isdf_tensors_*.h5` (bse_io.py:759-763) — stale-file hazard in reused run
  dirs.
- `fermi_energy=0.0` sentinel (bse_io.py:417) makes an exact-0.0-Ry Fermi
  hint unpassable.
- kpts written to the h5 are regenerated from `generate_kpts_grid`, not
  read from the restart/WFN — silently assumes the solver flat-k order is
  the Γ-centered ix-outer MP grid (true today; single point of failure if
  the k-order convention ever changes).
