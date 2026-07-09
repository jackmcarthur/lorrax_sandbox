# file_io group: BGW-format readers/writers (fio_bgw)

Deep-read catalog for the GW refactor map, 2026-07-01. Repo root: `/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D`.
Files: `src/file_io/read_bgw_vcoul.py`, `src/file_io/epsreader.py`, `src/file_io/qe_save_reader.py`, `src/file_io/mf_header.py`, `src/file_io/wfn_writer.py`.

Grep scope used throughout: `src/`, `tests/`, `tools/`, `scripts/` (Python files), excluding `src/lorrax.egg-info/`.

---

## 1. src/file_io/read_bgw_vcoul.py (194 LOC)

**Purpose.** Parser for BerkeleyGW's text `vcoul` file (written by `write_vcoul` in sigma/epsilon). Provides an IBZ q-point table of MC-averaged `v(q+G) = <8π/|q+G+δq|²>_miniBZ` (Ry) and a helper that scatters one q-point's values onto the LORRAX dense FFT grid in LORRAX's `v_scaled = v/Ω` units. Used to bypass LORRAX's approximate G=0-only mini-BZ average so BGW comparisons are bit-reproducible.

**Category.** I/O: BGW-comparison override (diagnostic Coulomb-interaction ingestion).

**Entry points / callers.**
- `read_bgw_vcoul` <- `gw/gw_driver_helpers.py:206` (`build_bgw_v_grid_fn`), exported in `file_io/__init__.py:46`.
- `fill_v_grid_for_q` <- `gw/gw_driver_helpers.py:230` (closure `bgw_v_grid_fn(q_frac_tuple)` passed to `compute_V_q`), exported in `__init__.py`.
- `BGWVcoulTable` — exported in `__init__.py`; only instantiated inside this module. `find_q_index` only called internally by `fill_v_grid_for_q`.

**Function table.**

| name | lines | role |
|---|---|---|
| `BGWVcoulTable` (dataclass) | 19-81 | `q_fracs (nq,3) f64`, `G_miller_per_q` list of `(nG_q,3) i32`, `vcoul_per_q` list of `(nG_q,) f64`. All host numpy. |
| `BGWVcoulTable.find_q_index(q_frac, tol=1e-4, sym_mats_k=None)` | 33-81 | Find stored q sym-equivalent to requested q. Convention: `q_frac = S_k · q_table[iq] + kg0` (BGW/LORRAX `k_full = S·k_bar + kg0`, see `symmetry_maps.SymMaps._get_umklapp_vector`). Tries direct match first (S=identity), then loops all `sym_mats_k`. Returns `(iq, S_k int32 3x3, kg0 int32 3)`. Raises ValueError if unmatched. |
| `read_bgw_vcoul(path)` | 84-130 | `np.loadtxt` of 7-column text (qx qy qz Gx Gy Gz vcoul). Groups rows into contiguous q-blocks; q rounded via `np.round(np.mod(q,1.0)*1e8)` int64 keys; **stops at first repeated q key** — sigma re-emits the q-list every outer-k iteration with a fresh mini-BZ MC draw, so later blocks are MC-noise siblings; only the first outer-k iteration's blocks are trusted (matches BGW internal behaviour at outer-k=rk(1)). |
| `fill_v_grid_for_q(table, q_frac, fft_grid, cell_volume, tol=1e-4, sym_mats_k=None)` | 133-194 | Builds `(fft_nx,fft_ny,fft_nz) f64` host array. Physics: `v_scaled(G) = vcoul_BGW(q_tbl, G_miller) / Ω` scattered at `G_input = S_k @ G_miller − kg0` (comment cites the same integer formula as `common/symmetry_maps.py:get_gvecs_kfull` lines 689-690: `G_full = sym_krep @ G_irr − kg0`). einsum VERBATIM: `np.einsum('ij,gj->gi', S_k.astype(np.int32), G_miller) - kg0[None, :]`. Miller indices wrapped by `np.mod(...,fft_n)`. `v_scaled[0,0,0]` zeroed **only when q wraps to 0** (`np.mod(q+0.5,1)-0.5` all < tol) — the true head is handled by the separate rank-1 head correction; for q≠0 the G=0 entry `8π/|q|²` is a finite body contribution and preserved. Unmatched grid entries stay 0. |

**Flags consumed (by its caller `build_bgw_v_grid_fn`, gw_config.py):** `head.use_bgw_vcoul` (default False), `head.bgw_vcoul_file`, `head.bgw_vcoul_sym_wfn` (aux sym-reduced WFN to supply the 48 crystal ops when the main WFN is nosym; `gw_driver_helpers.py:221` reads `mf_header/symmetry/mtrx` from it and transposes → `sym_mats_k`).

**I/O.** Reads: BGW `vcoul` plain text, one `(q,G)` per line, 7 columns. Writes: nothing.

**Boundary arrays.** All host numpy. Output of `fill_v_grid_for_q` is a dense f64 FFT-grid cube per q, later consumed by `compute_V_q`.

**Dead suspects.** None — all three exports have live callers (grepped `read_bgw_vcoul|BGWVcoulTable|fill_v_grid_for_q` over src/tests/tools/scripts; hits in gw_driver_helpers + __init__).

**Redundancy suspects.**
- `find_q_index`'s symmetry-unfold + umklapp logic re-implements the `k_full = S·k_bar + kg0` matching that `common/symmetry_maps.SymMaps` already owns (the docstring itself points at `SymMaps._get_umklapp_vector`). Per the "unified sym action" principle this is a parallel sym-matching path.
- The G-mapping einsum `'ij,gj->gi' ... - kg0` duplicates `common/symmetry_maps.py:get_gvecs_kfull` (explicitly acknowledged in comment at lines 170-175).
- `epsreader.unfold_eps_comps` (below) implements the same S@G − Gq map with the same einsum shape — three copies of one integer transform across two files.

**Weird code.**
- Line 108: magic constant `1e8` q-rounding key; q reconstructed from the key (`np.asarray(key)/1e8`, line 121), so stored `q_fracs` are quantized to 1e-8 and wrapped into [0,1) — original signed fractional q is discarded.
- Lines 113-119: "stop at first repeated q" heuristic assumes q-blocks are contiguous and the first outer-k iteration covers the IBZ; if BGW ever interleaved q's the reader would silently truncate.
- Line 176: `S_k.astype(np.int32)` on a matrix already `np.rint(...).astype(np.int32)` in `find_q_index` — harmless double cast.

---

## 2. src/file_io/epsreader.py (161 LOC)

**Purpose.** Eager reader for BGW `epsmat.h5` / `eps0mat.h5` dielectric-matrix files. Loads the full `eps_header` plus the **entire** `mats/matrix` dataset into memory at construction and exposes per-q complex ε⁻¹ matrices and the q→0 head element `epshead`. In the current pipeline it is used almost only for `epshead` (head-correction wcoul0 source) and by diagnostic check scripts.

**Category.** I/O: BGW epsilon reader (diagnostic / head-correction input).

**Entry points / callers.**
- `EPSReader` <- `gw/head_correction.py:130-133` (`from_epshead` path, uses only `.epshead`); `scripts/checks/sigma_direct_check.py:71,167` (uses `.epshead`); `scripts/checks/w_from_eps0_0d_check.py:60,439` (uses `.get_eps_matrix`, header fields, `gind_eps2rho`, `comps`); exported `file_io/__init__.py:25`.
- `get_eps_matrix` <- `scripts/checks/w_from_eps0_0d_check.py:494,498,557` only.
- `.epshead` attribute <- `gw/head_correction.py:136`, `scripts/checks/sigma_direct_check.py:174`.
- `if __name__ == "__main__"` demo at 153-162 (hardcoded `"epsmat.h5"` cwd path).

**Function table.**

| name | lines | role |
|---|---|---|
| `__init__(filename)` | 5-80 | Opens h5py file handle (kept open for object lifetime). Reads `eps_header/{versionnumber,flavor}`; `params/{matrix_type(0=ε⁻¹,1=ε,2=χ0), has_advanced, nmatrix, matrix_flavor, icutv, ecuts, nband, efermi}`; optional `params/{subsampling,subspace}`; `qpoints/{nq,qpts,qgrid,qpt_done}`; `freqs/{freq_dep,nfreq,nfreq_imag,freqs}`; `gspace/{nmtx,nmtx_max,ekin,gind_eps2rho,gind_rho2eps,vcoul}` — gind arrays get `-1` for Fortran→C indexing; `mf_header/gspace/components[:gvec_ind_max]` → `.comps`; optional `eps_header/subspace/*`. **Eagerly slurps `mats/matrix[:]`** (shape (nq, nmatrix, nfreq, ng, ng, 2) — can be GB-scale) and `mats/matrix-diagonal[:]`. Sets `epshead = matrix[0,0,0,0,0,0] + 1j*matrix[0,0,0,0,0,1]` (static ε⁻¹ head at first q). Optional `mats/{matrix_subspace, matrix_eigenvec, matrix_fulleps0}` if subspace. |
| `__del__` | 82-85 | Closes file handle. |
| `get_eps_matrix(iq, ifreq=0, imatrix=0)` | 87-100 | Returns complex `(nmtx[iq], nmtx[iq])` from `matrix[iq, imatrix, ifreq, :n, :n, 0] + 1j*[...,1]`. |
| `get_eps_minus_delta_matrix(iq, ifreq=0, imatrix=0)` | 102-116 | Copy-paste of `get_eps_matrix` then `mat.flat[::nmtx_q+1] -= 1.0` (subtracts identity: ε⁻¹ − δ_GG'). |
| `unfold_eps_comps(iqbar, S, Gq)` | 118-138 | Symmetry unfold of the ε G-basis: for `q_1 = qbar{S|tau} + Gq`, per Deslippe 2012 §7.2 `epsinv_GG'(q_1) = exp(-i(G-G')τ) epsinv_[(G+Gq)S⁻¹,(G'+Gq)S⁻¹](qbar)`, so one can evaluate `sum_GG' M*_q1(GS−Gq) epsinv_GG'(qbar) M_q1(GS−Gq)`. Comment: "NO SUPPORT FOR TAU (FRAC TRANS) CURRENTLY". Builds `G_comps_qbar = comps[gind_eps2rho[iqbar,:nmtx[iqbar]],:]` then einsum VERBATIM: `np.einsum('ij,kj->ki',S.astype(np.int32),G_comps_qbar) - Gq[np.newaxis,:]`. Commented-out `Sinv`/matmul alternative at 131,135. |
| `get_eps_diagonal(iq)` | 141-151 | `matrix_diagonal[:, :nmtx[iq], iq]` → `diag[0] + 1j*diag[1]`. |

**Flags consumed.** None directly. Callers gate it via cohsex.in keys `wcoul0_source` ("s_tensor" | "epshead", gw_config.py:285/533) and the hardwired path `{input_dir}/eps0mat.h5` in `head_correction.py:120`.

**I/O.** Reads BGW `epsmat.h5`/`eps0mat.h5`: groups `eps_header/{params,qpoints,freqs,gspace,subspace}`, `mf_header/gspace/components`, `mats/{matrix, matrix-diagonal, matrix_subspace, matrix_eigenvec, matrix_fulleps0}`. Writes nothing. (Format spec: `docs/docs_bgw/epsmat.h5.spec`.)

**Dead suspects** (grepped each name over src/tests/tools/scripts, excluding this file):
- `get_eps_minus_delta_matrix` — zero callers.
- `unfold_eps_comps` — zero callers.
- `get_eps_diagonal` — zero callers.
- Subspace attributes (`matrix_subspace` etc.) — zero consumers.

**Redundancy suspects.**
- `get_eps_minus_delta_matrix` is a verbatim copy of `get_eps_matrix` + one diagonal-subtract line (classic "fetch_X_dyn next to fetch_X" pattern).
- `unfold_eps_comps` duplicates the S@G − Gq integer transform of `read_bgw_vcoul.fill_v_grid_for_q` and `symmetry_maps.get_gvecs_kfull` (transposed-index einsum variant `'ij,kj->ki'` vs `'ij,gj->gi'` — same operation).

**Weird code.**
- Line 70: `# you should only really want this for eps0. TODO: frequency dep.` — `epshead` is static-only, frequency dependence unimplemented; `head_correction.py:127` papers over it with a "static-only; using epshead(0)" warning.
- Line 67: eager full-matrix load in `__init__` — for a production `epsmat.h5` this is potentially many GB pulled to host even when the caller (head_correction) only needs one complex scalar.
- Lines 121-122, 131, 135: commented-out asserts and the `Sinv` alternative path left in place.
- `__del__`-based file closing (85) — fragile lifetime management; no context manager.
- Line 54: `comps` truncated at `gvec_ind_max = max(gind_eps2rho)` — silently assumes gind covers the needed component range.

---

## 3. src/file_io/qe_save_reader.py (401 LOC)

**Purpose.** Reads crystal structure, symmetry ops (with fractional translations and TR markers), electronic parameters from a QE `.save/data-file-schema.xml`, plus the valence charge density from `charge-density.hdf5`. `CrystalData` deliberately duck-types `WFNReader` for structure queries so the standalone-DFT path (`CrystalData → setup_H_k_from_kvec → Davidson`) and the centroid charge-density path can run without a WFN.h5. Also contains a faithful reimplementation of QE's `kpoint_grid.f90` MP-grid → IBZ reduction.

**Category.** I/O: QE .save reader + preprocessing (standalone-DFT/NSCF front-end); contains embedded symmetry machinery (IBZ k-grid reduction).

**Entry points / callers.**
- `CrystalData.from_qe_save` <- `psp/run_nscf.py:510`, `psp/kpm_dos.py:337`, `centroid/charge_density.py:70` (`rho_from_qe_save`), `psp/tests/test_dft_hamiltonian.py:65`; exported `file_io/__init__.py:23`.
- `build_kgrid` <- `psp/run_nscf.py:78`, `psp/kpm_dos.py:180`.
- `load_charge_density` <- `centroid/charge_density.py:71`, `psp/run_nscf.py:61` (via `build_dft_potentials`), `psp/kpm_dos.py:165`, `psp/tests/test_dft_hamiltonian.py:98`.
- `validate_against_wfn` <- `psp/tests/test_dft_hamiltonian.py:67` only.
- Duck-typed consumers of the instance: `psp/scf_potential.py`, `psp/dft_operators.py` (needs bdot/ecutwfc/fft_grid), `psp/vnl_ops.py` (atom_crys/bvec/blat/cell_volume), `file_io/wfn_writer.WFNWriter`.
- `_reduce_mp_to_ibz` — module-private, called only from `build_kgrid`.

**Function table.**

| name | lines | role |
|---|---|---|
| `_SYMBOL_TO_Z` / `_ELEMENTS` | 51-60 | Hand-rolled periodic table to Z≤86 (Rn). Elements beyond Rn would KeyError. |
| `_text`, `_all`, `_vec` | 67-77 | Namespace-stripping XML helpers (`tag.split("}")[-1]`), first-match text / all elements / float vector parse. |
| `CrystalData` (dataclass) | 84-322 | Fields documented inline: alat, blat, avec (=avec_bohr/alat), bvec (/blat), bdot `= bvec @ bvec.T * blat²` [bohr⁻²], cell_volume, nat, atom_crys (crystal coords), atom_types (Z), ntran, sym_matrices `(48,3,3) i32` (padded), translations `(48,3) f64` **stored as τ_crys × (−2π) to match BGW tnp**, sym_time_rev `(48,) bool`, nelec, nspin, nspinor, ecutwfc/ecutrho (Ha→Ry via ×2), fft_grid, nbands, nkpts (always 0!), kgrid, assume_isolated ("none"|"2D"), `_save_dir`. |
| `from_qe_save(save_dir)` | 123-216 | Parses XML. Crystal coords via `τ_crys = inv(avec_bohr.T) @ pos` (line 146-148). Symmetry: reads `<symmetry>/<rotation>` + optional `<fractional_translation>`; sign flip `frac_trans.append(-tau * 2π)` (line 163, "BGW convention: opposite sign from QE XML, × 2π"); TR flag from `<info time_reversal="true">`. nspinor=2 if noncolin; nspin=2 if (lsda and not noncolin). ecut×2 (Ha→Ry). Raises on assume_isolated not in {none, 2D}. |
| `build_kgrid(nk, nosym, noinv, no_t_rev, force_symmorphic)` | 219-277 | Γ-centred MP grid → IBZ, matching pw.x flags: nosym→identity only; force_symmorphic drops ops with τ≠0; no_t_rev drops rotation+TR ops; `time_reversal = not noinv` folds k↔−k. Delegates to `_reduce_mp_to_ibz`. Returns `(kpoints (n_ibz,3) f64, weights summing to 1)`. |
| `load_charge_density()` | 280-298 | Reads `charge-density.hdf5`: `MillerIndices` + interleaved real/imag `rhotot_g`; scatters `rho_G[mi%nx, mi%ny, mi%nz] = rho_ri[0::2]+1j*rho_ri[1::2]`; `rho_r = Re(ifftn(rho_G)) * N` (QE forward-FFT normalization). Returns `(rho_r, rho_G)` host complex/real cubes. NLCC excluded (valence only). |
| `validate_against_wfn(wfn, atol=1e-6)` | 301-322 | Asserts alat/blat/celvol/avec/bvec/bdot/atom_crys/atom_types/nelec/nspinor/fft_grid/ntran/sym_matrices match a WFNReader. **Does NOT check translations** (see weird code). Prints "all checks passed". |
| `_reduce_mp_to_ibz(nk, sym_matrices, time_reversal=True)` | 329-399 | QE kpoint_grid.f90 clone: (1) enumerate grid with dir-3 fastest, `n = k + j*nk3 + i*nk2*nk3`, `xkg[n] = [i/nk1, j/nk2, k/nk3]`; (2) for each unclaimed k apply all S (`xkr = S @ xkg[n]`, wrap by `-= round`), plus −xkr if TR; forward-only equivalence `if n_eq > nk_idx and equiv[n_eq]==n_eq`; (3) extract IBZ in discovery order, weights normalized. `_EPS = 1e-5` off-grid tolerance. Pure O(nkr·nsym) Python loops. |

**Flags consumed.** No LorraxConfig / cohsex.in keys. `build_kgrid` kwargs are driven by CLI args of `run_nscf.py` / `kpm_dos.py` (mirroring pw.x `nosym/noinv/no_t_rev/force_symmorphic`).

**I/O.** Reads: `{save}/data-file-schema.xml` (QE XML schema), `{save}/charge-density.hdf5` (datasets `MillerIndices` (ng,3), `rhotot_g` (2·ng,) interleaved). Writes nothing.

**Dead suspects.** None hard-dead. `validate_against_wfn` is test-only (single caller `psp/tests/test_dft_hamiltonian.py:67`).

**Redundancy suspects.**
- `CrystalData` mirrors ~15 `WFNReader`/`MfHeader` fields by construction (intentional duck-typing, but it is a second definition of the mf-header field set; a refactor could make `MfHeader` the single schema and have both sources emit it).
- `_reduce_mp_to_ibz` is a second IBZ-reduction implementation next to the `common/symmetry_maps.SymMaps` machinery used by the GW path (different purpose — grid generation vs. matching an existing WFN k-list — but overlapping symmetry math).
- `_SYMBOL_TO_Z` hand-rolled periodic table (would duplicate any element table elsewhere; grep found no other, so currently unique).

**Weird code.**
- Line 1: docstring says `psp/qe_save_reader.py` but the file lives in `src/file_io/` — stale header from a directory move (same in wfn_writer.py; `psp/dev_status.md:16-17` still lists both under psp with stale LOC counts).
- Lines 24-26 (docstring): "Note: 24/48 non-symmorphic translations have a sign mismatch vs pw2bgw; rotation matrices match exactly." — **documented known sign discrepancy** in the −τ·2π convention (line 163). `validate_against_wfn` conspicuously omits a `translations` check (lines 301-322), consistent with the mismatch being unresolved. Hypothesis: QE XML stores S/τ in a convention where half the ops need the opposite τ sign after pw2bgw's own transformation; any consumer of non-symmorphic translations from CrystalData (glide/screw phases) is at risk.
- Line 214: `nkpts=0` always — the duck-type advertises a field it never fills; consumers that read `.nkpts` off a CrystalData would silently get 0.
- Line 137: `alat = |a1|` — assumes QE's alat equals |a1| (true for QE ibrav conventions used here, but not a general invariant if celldm differs).
- Magic constants: `_EPS = 1e-5` (line 345, QE's eps in kpoint_grid.f90), `atol=1e-8` symmorphic test (line 265), tolerance `0.5` used as "integer equality" in `_chk` calls (315-321).

---

## 4. src/file_io/mf_header.py (191 LOC)

**Purpose.** Canonical read path + verbatim copy helper for the BGW-style `mf_header` HDF5 group that travels with every mean-field-derived artifact in LORRAX (`WFN.h5`, `zeta_q.h5`, `kin_ion.h5`, `V_qmunu.h5`, `sigma_omega.h5`). `MfHeader` is a pure 1:1 NamedTuple mirror of the on-disk group; the module deliberately exposes no derived quantities (vbm, efermi, kpt_starts are computed by consumers). This is the single-source-of-truth schema module the docstring says it is.

**Category.** I/O: mf_header schema (shared header for all LORRAX HDF5 artifacts).

**Entry points / callers.**
- `read_mf_header(path)` <- `file_io/zeta_reader.py:79`; tests `tests/test_mf_isdf_header_roundtrip.py:118,170-171,265`.
- `read_mf_header_from_file(f)` <- `file_io/wfn_loader.py:153` (WFNReader/WfnLoader init), `file_io/zeta_loader.py:106`.
- `copy_mf_header(src,dst)` <- `common/isdf_fitting.py:2292,2342` (stamps `zeta_q.h5` from the source WFN); tests `test_mf_isdf_header_roundtrip.py`, `test_zeta_reader.py:35,138`, `test_zeta_loader.py:29,108`, `test_compute_all_V_q_g_flat.py:82`, `test_compute_V_q_bispinor_g_flat.py:80`.
- `MfHeader` class <- type surface for `WfnLoader`/`ZetaReader`/`ZetaLoader` attribute mirroring; imported in `tests/test_mf_isdf_header_roundtrip.py:16`.
- `__all__` at 186-191 matches exactly the four public names.

**Function table.**

| name | lines | role |
|---|---|---|
| `MfHeader` (NamedTuple) | 30-82 | Fields with BGW dataset names in comments: version(versionnumber), flavor; kpoints: nspin, nspinor, nkpts(nrk), nbands(mnband), ngkmax, ecutwfc, kgrid (3,)i, shift (3,)f, ngk (nkpts,)i, ifmin/ifmax (nspin,nkpts)i, kweights(w), kpoints(rk) (nkpts,3), energies(el) (nspin,nkpts,nbands), occs(occ); gspace: ng, ecutrho, fft_grid(FFTgrid) (3,)i32; symmetry: ntran, cell_symmetry, sym_matrices(mtrx) (48,3,3)i, translations(tnp) (48,3)f **BGW convention 2π·τ_frac**; crystal: cell_volume(celvol), recip_volume(recvol), alat, blat, nat, avec, bvec, adot, bdot, atom_types(atyp), atom_positions(apos, cartesian in units of alat). |
| `_read_group(f)` | 89-132 | Straight 1:1 dataset reads from open `h5py.File['mf_header']`. Only transform: `fft_grid` cast to int32. |
| `read_mf_header(path)` | 135-143 | Open/read/close wrapper. |
| `read_mf_header_from_file(f)` | 146-148 | Same on an open handle (used inside `WFNReader.__init__`). |
| `copy_mf_header(src_path, dst_path, dst_mode='a')` | 155-183 | `h5py.File.copy('mf_header', fd)` — verbatim, preserves dtypes/chunking/attrs. **Refuses to overwrite** an existing `mf_header` in dst (ValueError, explicit design: header overwrite is destructive). |

**Flags consumed.** None.

**I/O.** Reads/copies the `mf_header/{versionnumber,flavor,kpoints/*,gspace/*,symmetry/*,crystal/*}` group of any LORRAX/BGW HDF5 file. Writes only via verbatim copy into destination artifacts.

**Dead suspects.** None — all four `__all__` names have live callers (grep above).

**Redundancy suspects.**
- Field set triplicated across `MfHeader`, `CrystalData` (qe_save_reader), and `WFNWriter._write_header` (wfn_writer) — three hand-maintained copies of the same schema; a schema change must touch all three plus BGW spec docs.
- Note `file_io/__init__.py:16-22`: `WFNReader = WfnLoader` back-compat alias with a promised follow-up sweep — not this module's fault but part of the same header-consumer surface.

**Weird code.**
- Nothing suspicious; this is the cleanest file of the group. The only latent trap: `_read_group` reads eagerly with `[()]`/`[:]` and no shape validation — a malformed artifact fails at consumer time, not read time.

---

## 5. src/file_io/wfn_writer.py (248 LOC)

**Purpose.** Writes BGW/LORRAX-compatible `WFN.h5` files from a `CrystalData` (or duck-typed WFNReader) plus eigen-solutions. Two modes: streaming `WFNWriter` (header up front, `wfns/coeffs` pre-allocated, one k written at a time — for parallel Davidson) and batch `write_wfn_h5` (legacy wrapper). Used by the standalone NSCF path and by the QP-wavefunction dump (`WFN_qp.h5`) at the end of GW.

**Category.** I/O: WFN.h5 writer (NSCF output / QP-WFN artifact).

**Entry points / callers.**
- `WFNWriter` <- `psp/run_nscf.py:37,265,383` (main NSCF WFN + pseudobands WFN, rank-0 only), `file_io/qp_wfn.py:133,169` (`write_qp_wfn_h5`, which is called from `gw/gw_jax.py:761` under `config.debug.write_wfn_h5` and from `dump_qp_wfn_artifacts` in the SC path at gw_jax.py:574); exported `file_io/__init__.py:24`.
- `write_wfn_h5` (batch) — grep over src/tests/tools/scripts finds NO caller of the *function* (the `config.debug.write_wfn_h5` hits in gw_jax/gw_config are the identically-named config flag, and `psp/dev_status.md:168` mentions it in stale docs). Dead suspect.
- `_build_gspace_components` — private, called only from `_write_header`.

**Function table.**

| name | lines | role |
|---|---|---|
| `_build_gspace_components(crystal)` | 30-48 | Charge-density G-sphere in QE convention: per-axis range `[-N//2+1, N//2]`, filter `G·bdot·G ≤ ecutrho`, sort key `(round(|G|²×1e8), g1, g2, g3)` via `np.lexsort((G_f[:,2], G_f[:,1], G_f[:,0], G2_int))` — "Matches QE ggen.f90". einsum VERBATIM: `np.einsum("gi,ij,gj->g", G_all.astype(float), bdot, G_all.astype(float))`. |
| `WFNWriter.__init__` | 71-105 | Stores nk, nbands, nspin, nspinor, `ngk` from `gvecs_per_k`, cumulative `_offsets`; allocates host `_el`/`_occ` `(nspin, nk, nbands)`; **`n_occ = int(crystal.nelec)`; `_occ[0, :, :n_occ] = 1.0`** (see weird code); opens file `"w"`, writes header. |
| `_write_header` | 107-187 | Writes full `mf_header` tree: versionnumber=1, flavor=2 (complex) hardcoded; kpoints group incl. placeholder el/occ (rewritten at close); `ifmin=1`, `ifmax=n_occ=int(nelec)` everywhere; gspace from `_build_gspace_components`; symmetry: if `nosym` → identity-only ntran=1, else copies `crystal.{ntran,sym_matrices,translations}`; `cell_symmetry=0` hardcoded in **both** branches; crystal group with `apos = atom_crys @ avec` (units of alat), `adot = avec@avec.T * alat²`, `recvol = (2π)³/celvol`; `wfns/gvecs` = concat of per-k gvecs; `wfns/coeffs` pre-allocated `(nbands, nspinor, ngktot, 2)` f64 fillvalue 0. |
| `write_k(ik, eigenvalues, coeffs=None)` | 189-206 | `_el[0, ik] = eigenvalues` (Ry); if coeffs given, writes real/imag slabs into `wfns/coeffs[:, :, off:off+ng_k, :]`. coeffs shape `(nbands, nspinor, ngk_ik)` complex128, host. **Spin index hardcoded 0** — no nspin=2 write path despite nspin in the header. |
| `close()` / `__enter__` / `__exit__` | 208-218 | Rewrites el/occ datasets, closes. Context-manager support. |
| `write_wfn_h5(...)` | 225-248 | Batch wrapper: constructs WFNWriter, loops `write_k`. Accepts an `occupations` kwarg that is **silently ignored** (never touched in the body). |

**Flags consumed.** `config.debug.write_wfn_h5` (gw_config.py:343 default True, field 657) gates the *callers* in gw_jax.py (574, 757) — note the flag shares its name with the dead batch function. run_nscf's `nosym` comes from its CLI.

**I/O.** Writes `WFN.h5` (and `WFN_qp.h5`, pseudobands WFN): full `mf_header` tree + `wfns/gvecs (ngktot,3) i32` + `wfns/coeffs (nbands, nspinor, ngktot, 2) f64`. Spec: `docs/docs_bgw/wfn.h5.spec`.

**Dead suspects.**
- `write_wfn_h5` (function, lines 225-248): grepped `write_wfn_h5` over src/tests/tools/scripts — every hit is the `config.debug.write_wfn_h5` flag (gw_jax.py:574,757; gw_config.py:343,657,1052) or stale doc `psp/dev_status.md:168`. Zero call sites of the function. Its `occupations` parameter is additionally a no-op.

**Redundancy suspects.**
- `_build_gspace_components` is a line-for-line duplicate of `psp/gvec_utils.build_master_gvec_list` (same ranges, same einsum, same 1e8 lexsort key; gvec_utils also returns G2). Two copies of QE ggen.f90 logic — exactly the "no redundancy" violation class. run_nscf.py imports gvec_utils for per-k gvecs, then WFNWriter regenerates the master list internally.
- Header-writing duplicates the `MfHeader` schema by hand (third copy of the field set, see mf_header.py notes).

**Weird code.**
- Line 2: docstring says `psp/wfn_writer.py`; file lives in `src/file_io/` (stale move, same as qe_save_reader).
- Lines 99-100 + 135-137: occupation/ifmax convention `n_occ = int(crystal.nelec)` marks the first **nelec** bands occupied with occ=1.0. Correct only for nspinor=2 (one electron per band). For nspinor=1 spin-unpolarized (nelec = 2×n_occ_bands) this doubles the occupied count; `file_io/qp_wfn.py:124` explicitly warns downstream about "WFNWriter (which sets ifmax = nelec on every k)". Hypothesis: written for the bispinor path and never generalized; a scalar-relativistic WFN written by this class carries wrong occ/ifmin/ifmax for BGW consumers.
- `write_k` line 199: `_el[0, ik]` — spin channel 0 only; an nspin=2 header would carry zero eigenvalues in spin 1.
- Line 155/160: `cell_symmetry=0` hardcoded (cubic) regardless of actual lattice; BGW reads this field.
- Line 116: flavor=2 hardcoded (complex wavefunctions only).
- Magic constant `1e8` shell-discretization key (line 46, shared with gvec_utils).
- `write_wfn_h5`'s ignored `occupations` kwarg (line 235) — signature promises functionality the body never implements.

---

## Cross-file observations

1. **Three copies of the integer sym-transform `G' = S@G − G0`**: `read_bgw_vcoul.fill_v_grid_for_q` (einsum `'ij,gj->gi'`), `epsreader.unfold_eps_comps` (einsum `'ij,kj->ki'`, dead), `common/symmetry_maps.get_gvecs_kfull` (the canonical one). Refactor should route the first through SymMaps and delete the second.
2. **Three hand-maintained copies of the mf_header field schema**: `MfHeader` (reader), `CrystalData` (QE-side producer), `WFNWriter._write_header` (writer). One schema, three definitions.
3. **QE ggen.f90 G-sphere generation duplicated**: `wfn_writer._build_gspace_components` ≡ `psp/gvec_utils.build_master_gvec_list`.
4. **Known unresolved convention issue**: qe_save_reader's non-symmorphic translation sign (24/48 mismatch vs pw2bgw), documented in its own docstring and dodged by `validate_against_wfn`.
5. `epsreader` is effectively a diagnostic-only module in the production path (only `.epshead` is consumed by gw); its eager full-matrix load and three dead methods make it a prime candidate for a slim-down.
