# Reader Cleanup — Execution Plan

Repo: `sources/lorrax_D`, branch `agent/memplanner-cleanup`. MAP §4 #6/#7.
Primary goal: **maintainability via single-sourced shared backends** across the `file_io/`
reader family. Every consolidation below survived a DIFF-THE-CORES pass (cores read at
file:line, not summarized). PLANNING ONLY here — no edits are described as done.

Guiding rule (heeded throughout): "N parallel X = redundant" has been wrong repeatedly.
Only cores that are **byte/value-identical** get single-sourced. Conceptually-similar cores
on a **different object / sharding / dtype / device / provenance** stay separate.

---

## 0. Gate context (prereqs)

The pytest suite is trustworthy. Coverage relevant to this cleanup:

| Gate | Covers | Guards which steps |
|------|--------|--------------------|
| pytest `-q` (full) | reader unit surface, header binds, slab math | every step |
| IBZ-equivalence gate | ψ unfold + ζ unfold read paths (B4, S2, zeta merge) | S2, §3, and load_wfns cutover |
| charge gate | scalar ζ / V_q charge screening read path | §3 (zeta), S3 |
| **bispinor gate** | B3 lift + 4-spinor path | **S1 only** — must run before/after the lift move |

Rule of thumb: T-tier (pure boilerplate) and load_wfns cutover are covered by pytest +
IBZ/charge gates and are **gated-now**. **S1 (bispinor lift move) additionally needs a
bispinor gate** — do not land S1 on green pytest alone.

---

## 1. Single-source consolidations (user's priority)

Ordered by (value × safety). "Home" = the one module the backend lives in after the change;
"Adopters" = readers that import it instead of holding a private copy.

### TRULY-REDUNDANT → single-source (value-identical cores)

**T3 — `_rank0()` / `_barrier(tag)` process helpers.**  *(warm-up; zero-risk)*
- Copies: `_slab_io_allgather.py:31-39` vs `_slab_io_mpi_host.py:58-70` — **byte-identical**
  bodies (`jax.process_index()==0`; `try: sync_global_devices(tag) except: pass`); only a
  `_barrier` docstring differs.
- **Home:** new `file_io/_slab_io_common.py` (or fold into `_slab_io_ffi.py`, which both
  files already import).
- **Adopters:** `_slab_io_allgather.py`, `_slab_io_mpi_host.py`.

**T1 — mf_header 35-field attribute mirror.**  *(biggest boilerplate win: 3 copies × ~35 L)*
- Copies: `wfn_loader.py:154-188` vs `zeta_loader.py:117-151` vs `zeta_reader.py:85-119` —
  identical field set, identical order; only the local var name (`hdr.`/`mf.`) differs.
- **Home:** `file_io/mf_header.py::bind_mf_attrs(obj, mf)` (or `MfHeader.bind_to(self)`).
- **Adopters:** `wfn_loader`, `zeta_loader`, `zeta_reader`.
- **CAVEAT that must survive:** `wfn_loader` alone appends a *derived tail* the zeta readers
  lack — `atom_crys` (190-191), `nelec/vbm/cbm/efermi` (195-207). The helper binds only the
  35 raw fields; wfn_loader keeps its derived tail **local**. Do not fold the tail in.

**T2 — isdf_header attribute mirror.**  *(2 copies × ~13 L; dissolves into §3)*
- Copies: `zeta_loader.py:154-166` vs `zeta_reader.py:121-133` — same 11 fields. Only real
  diff: loader does `int(isdf.n_rmu)`; reader keeps `isdf.n_rmu` (an int-returning
  `@property`) → value-identical.
- **Home:** `file_io/isdf_header.py::bind_isdf_attrs(obj, isdf)`.
- **Adopters:** `zeta_loader`, `zeta_reader` (both classes merge in §3 anyway).

**T4 — cumulative-ngk (`kpt_starts`) prefix sum.**  *(trivial; low priority)*
- Copies: `wfn_loader.py:214-216` vs `postprocess/rotate_wfn_to_qp.py:136-138` — identical
  arithmetic. Grep confirms **no** copy in zeta_loader/zeta_reader/epsreader/centroids (the
  reader-map over-guessed its spread).
- **Home:** `file_io/mf_header.py::kpt_starts(ngk)` (module docstring already names it).
- **Adopters:** `wfn_loader`, `rotate_wfn_to_qp`.

### SHARED-SEAM → extract the core, keep thin per-reader wrappers

**S1 — B3 bispinor small-component lift (constant + σ·p block).**  *(needs bispinor gate)*
- `wfn_loader.py` `_get_bispinor_lift_jit._kernel:1050-1072` (+ `_HALFALPHA=0.00364867628215`
  at 1035) vs `common/bispinor_init.py::get_small_psi_component:30-41`. **Numerics
  value-identical** (same constant, same Pauli block, same `halfalpha·σ·p·ψ_L`). Differences
  are wrapping only: leading `k` batch + `nb` axis; lru-cached jit keyed on output sharding
  with `with_sharding_constraint`; concat `[psi_2, psi_S]` into a 4-spinor (1069).
- **Home:** `common/bispinor_init.py` — the `_HALFALPHA` constant + σ·p contraction live once
  there (generalize `get_small_psi_component` to accept an optional leading k-axis, or add a
  `lift_to_4spinor` sibling).
- **Adopter:** `wfn_loader` imports it; keeps loader-only concerns (jit-cache-by-sharding
  wrapper + the 4-spinor concat). The duplicated constant + Pauli block is the defect.

**S2 — B4 ψ symmetry-unfold *rule* (TRS-augmented U + τ-phase).**
- Host `common/symmetry_maps.py::unfold_psi:692-725` vs device
  `wfn_loader.py::_ensure_phdf5_static:587-620` (+ apply half `_phdf5_unfold_kernel:1142-1145`).
  **Rule element-for-element identical:** τ-phase `exp(-i (S·G_kbar)·τ)`; TRS spinor
  `U_eff = iσ_y·conj(U_spatial)`; `where(tr_mask, conj, id)` selection.
- **But a genuinely different object:** host = per-single-k numpy on the IBZ G-axis; device =
  k-vectorized replicated tables consumed inside a `shard_map`. **You cannot call `unfold_psi`
  inside the shard_map — do NOT delete the phdf5 copy.**
- **Home:** `common/symmetry_maps.py` gains two named helpers both callers invoke:
  `trs_augment_U(U_spatial, sym_idx, n_tran)` and the τ-phase builder. The eager path already
  single-sources via `unfold_psi`; this makes the **rule** (not the object) the single source
  so the phdf5 table-build stops copy-pasting it.
- **Also:** fold the G-side `gvecs()` unfold branch's private `sym._get_umklapp_vector`
  (`wfn_loader.py:377,379`) into a **public** `symmetry_maps` accessor (private `_`-method
  leaking into the loader).

### SUPERFICIAL → **NOT merging** (forced merge breaks correctness or just couples readers)

- **L1 — zeta on-disk shape probe / attr derivation.** `zeta_loader.py:105-181` vs
  `zeta_reader.py:143-162`. Only the ~2-line ds-name+`shape` read is common; the derived
  attrs **genuinely diverge** (loader `n_rtot_disk=None`, adds `n_q_full`/`q_layout`; reader
  computes `n_rtot_disk=nx·ny·nz`, tracks `n_G_sph_disk` loader lacks). Extract at most a
  2-line `probe_zeta_dataset(f, isdf)->(name, shape)` — and only as part of §3, where the
  divergence resolves. Standalone value is low. **Do not blind-merge.**
- **L2 — G-vector "rotate-by-sym − umklapp" einsum.** `wfn_loader.py:379` vs
  `epsreader.py:136` vs `read_bgw_vcoul.py:176`. Same integer identity, but **umklapp
  provenance differs in every copy** (`sym._get_umklapp_vector` / caller-supplied `Gq` from
  the eps file / `find_q_index` BGW-table search) on three unrelated G-lists in three
  unrelated readers. A shared helper would couple readers and buy nothing. **Leave.** Max
  warranted: a docstring cross-ref between `read_bgw_vcoul._umklapp`/`find_q_index` and
  `SymMaps._get_umklapp_vector`.
- **L3 — SymMaps-from-header construction.** `wfn_loader.py:294-319` (builds a
  `SimpleNamespace` stub deliberately, to avoid a circular ref to the loader) vs
  `zeta_loader.py:400-409` (`SymMaps(self)`). Different construction by design. **Leave this
  pass;** once T1's `bind_mf_attrs` exists both could pass the bound object — that's the
  `feedback_unified_sym_action` follow-up, tracked not done.
- **L4 — `_default_sharding` / `_pad_to`.** `wfn_loader.py:511-543` / `545-548`. Repo-wide
  grep found **no second copy** (the reader-map's "certainly duplicated somewhere" is wrong).
  Single-copy → **not a dedup target.** Do not manufacture a utils module for a one-off.

### Confirmed non-findings (already single-sourced — do NOT touch)
- `_do_disk_to_G` (r-space→G FFT + sphere gather): one def in `zeta_reader.py:354+`, imported
  by `zeta_loader.py:343`. Already shared; just relocate on the §3 merge.
- FFT-box / g_index build: `wfn_loader.box_index:413-442` and `load_wfns` both delegate to
  `common.gvec_fft_box.build_g_index_for_fft_box`. Correct pattern.
- SlabIO hyperslab plumbing: all zeta/kin_ion/sigma_output reads route through
  `file_io/slab_io.py`; the three `_slab_io_{allgather,ffi,mpi_host}` backends are **distinct
  transports** (rank-0 h5py gather / CUDA-FFI MPI-IO / mpi4py-parallel) — correctly separate.

---

## 2. `common/load_wfns.py` → WfnLoader cutover, then delete the facade

**Reality check (recurring lesson).** `load_wfns.py` is **not** a redundant facade over
WfnLoader methods. `WfnLoader.load()` returns G-flat ψ only; **none** of the 5 live functions
has a WfnLoader equivalent — 4 are `loader.load()` + a `common/wfn_transforms.py` primitive
(FFT-box / r-chunk / centroid), and the 5th is pure energy extraction. So the deletion blocker
is **where the code lives**, not what object it receives (every consumer already passes a
`WfnLoader`). "Inline `wfn.load()` at call sites" is **wrong** and would under-pad the spinor
axis.

**Relocation targets** (recommend `common/wfn_transforms.py` alongside
`to_box`/`to_rchunk`/`gflat_to_rmu`, or new `WfnLoader` methods):

| # | Function | Relocate to |
|---|----------|-------------|
| 1 | `get_enk_bandrange` (L46) | `WfnLoader.enk_bandrange(sym, …)` or `common/band_energies.py` |
| 2 | `load_kpoint_fftbox` (L17) | `WfnLoader.load_box(k=ik, bands=…)` (single-device, replicated) |
| 3 | `read_Gvecs_to_devices` (L118) | same `load_box` (multi-k / band-sharded / bispinor form) |
| 4 | `iter_psi_rchunk_bandwise` (L182) | `wfn_transforms.py` gen (or `WfnLoader.iter_rchunk`) |
| 5 | `load_centroids_band_chunked` (L293) | `wfn_transforms.py` (or `WfnLoader.load_centroids`) |

**Consumer-by-consumer cutover (22 import sites + 3 test files):**

- `get_enk_bandrange` — 10 sites/6 files + 1 re-export:
  `common/__init__.py:2` (re-export — update to new home, public surface),
  `gw/gw_jax.py:381,488,728,749`, `gw/gw_init.py:1000,1101`, `gw/ppm_pipeline.py:130,186`,
  `gw/sc_iteration.py:153`, `bandstructure/htransform.py:424`.
- `load_kpoint_fftbox` — `gw/kin_ion_io.py:195`, `psp/scf_potential.py:151`,
  `psp/dft_operators.py:1113`, `psp/run_sternheimer.py:820`;
  test `psp/tests/test_sternheimer_jvp.py:87`; archive `psp/archive/charge_density.py:106`
  (**confirm dead before touching**).
- `read_Gvecs_to_devices` — `psp/get_DFT_mtxels.py:800`, `psp/get_dipole_mtxels.py:541`;
  test `psp/tests/test_dft_hamiltonian.py:137`.
- `iter_psi_rchunk_bandwise` — `bandstructure/htransform.py:162` (single consumer).
- `load_centroids_band_chunked` — `bandstructure/htransform.py:87`,
  `centroid/pivoted_cholesky.py:959,975` (drop dead `use_phdf5=` kwarg),
  `gw/gw_init.py:679,1062`.

**Behavior that must survive relocation (do NOT assume equivalence):**
1. **meta-driven nspinor zero-pad** (`load_kpoint_fftbox:32-34`, `read_Gvecs_to_devices:162-165`)
   — pads spinor axis to `meta.nspinor`; `load()` does not. Conflating this with `bispinor=True`
   injects wrong physics (the lift is σ·(k+G), not a zero-pad). Keep the explicit pad.
2. **`get_enk_bandrange` file-short sentinel fill** (L83-91): fills `max(E)+1 Ry` (not ∞) so
   `f_n=0` and PPM resolvents stay finite; `np.repeat(…, nspinor)` weight expansion;
   **intentional host-numpy** (compile-cache, commit 31b5961). Preserve verbatim — do not jnp-ify.
3. **`load_centroids_band_chunked` band contracts** (L459-478, L525-533): past-file cap +
   zero-pad to `nb_total`, user-band-pad zeroing keyed on `meta.b_id_4_user`.
4. **`load_kpoint_fftbox` single-device/replicated** output (L38-43) — preserve if merged into
   the band-sharded path.
5. Drop dead `use_phdf5` kwarg at `pivoted_cholesky` sites (facade already `del`s it).

**Doc-bug to fix when touching the file:** `load_wfns.py:127` cites
`read_Gvecs_to_devices_legacy` which **does not exist** (grep-confirmed). Source doc bug, not a
KNOWN_SANDBOX_ERRORS item.

**Deletion:** after relocating #1–#5 and updating the 22 sites + 3 tests + `common/__init__.py`
re-export, delete `common/load_wfns.py`. Decide #2-vs-#3 merge only **after** verifying
single-device/replicated + singleton-strip semantics (do not assume).

---

## 3. `zeta_loader.py` / `zeta_reader.py` unification (keep new, delete old, kill shim)

**Framing:** unfinished old→new migration. The nominal "new" `ZetaLoader` (WfnLoader-shaped
`.load()`) is **not** production; the nominal "legacy" `ZetaReader` is the reader-of-record in
`gw_init`, and `ZetaLoader` imports `_do_disk_to_G` **from** `zeta_reader` (reverse dependency).
`ZetaLoader`'s one unique feature (ζ-level unfold) currently **raises `NotImplementedError` on
G_flat** (`zeta_loader.py:305-311`) — i.e. dead for the only files the writer emits.

**Survivor = `ZetaLoader`'s `.load()` API with `ZetaReader`'s production-proven G-flat read body.**

- **Merge the two slab cores (S3):** keep ZetaLoader's parametrized + non-contiguous
  `_read_g_flat_disk:537-563` and `_read_r_space:583-611` (strict supersets); **fold in**
  ZetaReader's sphere-mismatch guard (`read_zeta_G_slab:322-329`). Same dataset, same slab
  arithmetic (`zeta_q_G` `(q,μ,ngkmax)` off `(q,μ,0)`; `zeta_q` `(q,n_rtot,μ)` off `(q,0,μ)`).
- **Preserve per-layout default shardings** (consumers depend on the difference):
  G-flat `P(None,('x','y'),None)` (μ-sharded) vs r-space `P(None,None,('x','y'))` (r-sharded).
  Do not collapse to one spec.
- **`_do_disk_to_G`:** move the single definition into the surviving module; drop the reverse
  import. (Note: the whole r_space→G FFT path is **dead in production** — writer only emits
  `G_flat`; flag for **separate deletion**, not part of this merge.)
- **ζ-level unfold:** leave **as-is and separate** — it is a *different object* from the
  live V_q-level `symmetry_maps.unfold_v_q` (linear single-gather ζ vs bilinear centroid
  double-permute + L-phase + TRS conj-wrap). **Do NOT fuse** with `unfold_v_q`. Either wire it
  for G_flat later or delete the scaffolding; do not merge cores.
- **Kill the duck-typing shim:** delete `v_q_g_flat._make_read_all_ibz` (`v_q_g_flat.py:242-262`,
  which sniffs `has_load` vs `has_read_zeta_G_slab`) and the `ZetaLoader | ZetaReader` type
  unions (`v_q_g_flat.py:272`, `v_q_bispinor.py:155-156`).
- **Cut over consumers** from `read_zeta_G_slab` → `.load(layout='G_flat')`:
  `gw_init.py:936-939` (scalar), `gw_init.py:880-888` (bispinor, 4 handles), and
  `v_q_bispinor` (receives handles by arg).
- **Delete** class `ZetaReader` / `zeta_reader.py`, drop the `file_io/__init__.py:16` legacy
  export.

---

## 4. Physics un-smear out of `wfn_loader.py` + make the eager slurp lazy

- **B3 lift → `common/bispinor_init.py`** (see S1). Constant + σ·p block live once; loader
  imports and keeps the jit-cache-by-sharding wrapper + 4-spinor concat.
- **B4 ψ-unfold rule → `common/symmetry_maps.py`** (see S2). Extract `trs_augment_U` + τ-phase
  builder; host `unfold_psi` and the phdf5 device table-build both call them. Keep the phdf5
  apply-kernel (can't run inside shard_map otherwise). Promote `_get_umklapp_vector` to public.

- **Eager whole-array slurp (latent OOM), `wfn_loader.__init__:211-216`.** Executed
  **unconditionally for every backend**:
  `self._coeffs_raw = self._file["wfns/coeffs"][:]` — the whole
  `(nbands, nspinor, ngktot, 2)` f64 array (many GB at CrI3 scale). Usage audit (grep):
  `_coeffs_raw` is read in exactly **two places, both in `_eager_build`** (L992, L1009), each a
  `[b_lo:b_hi,:,start:end,:]` sub-slice. **The phdf5 backend never touches it** (reads coeffs
  via the collective FFI `read_kchunk_union_sharded:766`) — so on the phdf5 path the multi-GB
  slurp is pure waste + a double read on the exact multi-rank path where memory is tightest.
  - **Fix (minimal, high-value):**
    1. **Guard the slurp on `self.backend == "eager"`** — removes the OOM on phdf5 outright.
    2. **For eager, redirect the two `_eager_build` slices at the open h5 dataset**
       (`self._file["wfns/coeffs"][b_lo:b_hi,:,start:end,:]`, an h5py hyperslab) instead of the
       in-RAM array. Reads only the requested block; no io_callback needed for the single-process
       host path.
    3. Keep the small `_gvecs_raw` (`(ngktot,3)`) + `_kpt_starts` eager — index metadata, cheap,
       used by both backends.
  - Optional (per the "io_callback for big host caches" memory): wrap the per-`(b_lo,b_hi,k)`
    hyperslab read in `jax.experimental.io_callback` so ψ(G) is never a jit arg. Not required
    for the OOM fix — the backend guard + slice redirect is the whole defect.

---

## 5. Gating & commit boundaries

Ordered by (value × safety). **Gated-now** = pytest + IBZ/charge gates suffice.

| # | Step | Gate | Commit boundary |
|---|------|------|-----------------|
| 1 | **T3** `_rank0`/`_barrier` → `_slab_io_common` | pytest | commit A (warm-up) |
| 2 | **T1 + T2** mf/isdf binders (verify wfn_loader derived tail stays local) | pytest + IBZ + charge | commit B |
| 3 | **T4** `kpt_starts` util | pytest | commit B or C (trivial) |
| 4 | **B4/S2** ψ-unfold rule → symmetry_maps (+ public umklapp) | pytest + **IBZ-equivalence** | commit C |
| 5 | **Eager slurp lazy** (backend guard + slice redirect) | pytest + IBZ | commit C (with S2, both touch wfn_loader) |
| 6 | **B3/S1** bispinor lift → `bispinor_init` | pytest + **bispinor gate** | commit D (isolate; only gate that needs bispinor) |
| 7 | **load_wfns cutover** (§2) — relocate #1–5, update 22 sites + 3 tests, delete facade | pytest + IBZ + charge | commit E (may split: relocate → cutover → delete) |
| 8 | **zeta merge** (§3) — S3 cores + `_do_disk_to_G` relocate + shim/union kill + consumer cutover + delete `zeta_reader.py` | pytest + IBZ + **charge** | commit F (the big one; split relocate vs delete) |
| 9 | Separate: delete dead r_space→G FFT path | pytest + charge | commit G (optional, later) |

**Gated-now:** #1–#5, #7, #8 (green pytest + the IBZ/charge gates already cover ψ/ζ read paths).
**Needs a bispinor gate:** **#6 only** — run a bispinor gate before landing and after.

Recommended: land #1–#3 as one small "boilerplate single-source" commit, #4–#5 as a
"wfn_loader un-smear + lazy read" commit, #6 alone behind the bispinor gate, then the two large
structural commits (#7 load_wfns, #8 zeta) each split into relocate/cutover/delete sub-commits so
a bisect can localize any regression.

---

## 6. How far to go (blunt)

**Do:** single-source the value-identical boilerplate (T1–T4), extract the two physics seams as
**free functions** in their existing home modules (S1→`bispinor_init`, S2→`symmetry_maps`), and
collapse the two zeta classes into one. That is the maximal maintainability win.

**Do NOT:**
- Introduce a reader **base class / hierarchy**. The shared pieces are a handful of free binders
  (`bind_mf_attrs`, `bind_isdf_attrs`), two physics helpers, and tiny utils (`kpt_starts`,
  `_rank0`/`_barrier`). None of that is removed by an ABC — a base class would *add* an
  abstraction layer for human readers to carry (`feedback_no_new_api_layers`) while deleting zero
  duplication. Free functions in the natural home module are the right altitude.
- Merge the SUPERFICIAL cores (L1–L4): the umklapp einsum (different provenance per copy), the
  SymMaps-from-header construction (stub-vs-self by design), `_default_sharding`/`_pad_to`
  (single-copy), and the zeta shape-probe attr derivation (genuinely divergent). Forcing these
  couples unrelated readers or breaks correctness.
- Fuse the ζ-level unfold with `unfold_v_q`, or delete the phdf5 unfold copy. Different objects.
- Manufacture a `wfn_transforms` util module *just* for `_pad_to` — relocate #4/#5 there only
  because `to_box`/`to_rchunk` already live there.

The minimal-but-complete cleanup is: **~4 free binders/utils + 2 physics helpers + 1 class
merge + 1 facade deletion + 1 lazy-read guard.** No new classes, no new packages beyond the tiny
`_slab_io_common.py`.
