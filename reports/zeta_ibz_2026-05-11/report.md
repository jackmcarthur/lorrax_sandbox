# ζ_q.h5 IBZ refactor + G-flat reader — design (2026-05-11)

**Status**: design proposal, awaiting sign-off. No code changes yet.

**Scope (per session goals)**:
1. Make `zeta_q.h5` self-describing: copy `mf_header` from the source `WFN.h5`
   verbatim; add a new `isdf_header` group for ζ-specific metadata.
2. Write only the **irreducible q wedge** (factor ~`ntran` disk reduction).
3. Introduce a `wfn_reader`-shaped `ZetaReader` that exposes per-q μ-slab
   reads in **G-flat** layout (FFT + per-q phase + sphere gather moved out
   of `compute_V_q_tile` and into the reader).
4. Migrate `compute_V_q_tile` to consume G-flat ζ directly (no FFT in
   the kernel).
5. Keep ζ on disk in r-space for this phase. Disk → G-flat is Phase 2.

**Out of scope** (deferred to Phase 2):
- Writing ζ in G-flat on disk. The current writer's per-r-chunk solve
  produces ζ at r ∈ chunk, so a forward FFT can only happen after all
  chunks are done. That's a separate refactor (q-batched post-loop FFT
  + sphere-gather, peak-memory model rebuild).
- Migration of `V_qmunu.h5` and any other intermediate to share the
  same header module.

---

## 1. Schema

### 1.1 `mf_header` — verbatim copy from `WFN.h5`

The exact group tree (groups + dataset names + dtypes + shapes) under
`mf_header/` of the source `WFN.h5` is duplicated by the writer. This
includes:

```
mf_header/versionnumber, flavor
mf_header/kpoints/{nspin, nspinor, nrk, mnband, ngkmax, ecutwfc,
                   kgrid, shift, ngk, ifmin, ifmax, w, rk, el, occ}
mf_header/gspace/{ng, ecutrho, FFTgrid}
mf_header/symmetry/{ntran, cell_symmetry, mtrx, tnp}
mf_header/crystal/{celvol, recvol, alat, blat, nat, avec, bvec,
                   adot, bdot, atyp, apos}
```

Rationale: a `WFNReader` (or any consumer) can read this header from
`zeta_q.h5` and get the same crystal/symmetry/k-grid/G-grid view that
the producing `WFN.h5` exposed. The reader's mf_header attributes work
identically to `WFNReader`'s.

`mf_header` is **not** ζ-specific. The user intends to attach this same
header verbatim to other h5 files in the future (kin_ion.h5,
V_qmunu.h5, sigma_omega.h5, …). Hence the writer/reader lives in a
shared `file_io/mf_header.py` module — see §3 below.

### 1.2 `isdf_header` — new group, ζ-specific metadata

Per your direction: **no redundant symmetry arrays**. Anything derivable
from `mf_header/symmetry/{mtrx,tnp}` + `kpoints/kgrid` + `gspace/FFTgrid`
is computed on the fly via `common.symmetry_maps.SymMaps` and
`centroid.orbit_syms` at read time. The header only stores what *cannot*
be derived:

```
isdf_header/
  density            scalar str   ('scalar' | 'current' | 'mu_L=<int>')
  vertex_mu_L        scalar int   (0 = charge; 1,2,3 = transverse Lorentz)
  centroids/
    r_mu_fft_idx     (n_rmu, 3)   int32    FFT-grid indices (0..N_a)
    r_mu_crystal     (n_rmu, 3)   float64  fractional coords
                                            (= r_mu_fft_idx / FFTgrid;
                                             carried for human-readable
                                             inspection, not for compute)
```

Derived on the fly at reader / writer time (no on-disk redundancy):

| quantity                          | derived from                              | helper                                        |
|-----------------------------------|-------------------------------------------|-----------------------------------------------|
| q-IBZ list (`q_irr_kgrid_int`)    | `mf_header/{symmetry, kpoints/kgrid}`     | `SymMaps.find_irreducible_qpoints()` (new)    |
| q_full → q_irr index + sym op     | same                                      | same                                          |
| centroid orbit perm π_s           | `r_mu_fft_idx`, `symmetry`, `gspace/FFTgrid` | `orbit_syms.compute_centroid_sym_perm()` (new) |
| G-sphere indices                  | `gspace/{FFTgrid, ecutrho}` + bvec        | already in `gw_init.compute_V_q`              |
| G-sphere sym permutation g_s      | sphere indices + `mf_header/symmetry`     | new helper, used by Phase 2 G-flat unfold     |

The reader caches these derived tables on the `ZetaReader` instance
(host-side) the first time they're needed. Cost: `find_irreducible_qpoints`
is O(`n_q_full · n_sym`) integer compares — microseconds.

The `isdf_header` module is intentionally tiny — three datasets + two
scalars. It lives in `file_io/isdf_header.py`. Both header modules
expose:

- `read_mf_header(path) -> MfHeader` / `copy_mf_header(wfn_path, out_path)`
- `read_isdf_header(path) -> IsdfHeader` / `write_isdf_header(path, hdr)`

with `MfHeader` exposing the same attribute names `WFNReader` does — so
the existing `WFNReader.__init__` is refactored to delegate to
`read_mf_header` (single source of truth).

### 1.3 `zeta_q` dataset

```
zeta_q   shape = (n_qpt_irr, n_rtot, n_rmu)
         dtype = complex128
         chunks = (1, n_rchunk, n_rmu)
```

Same layout as today, **except the leading axis is `n_qpt_irr` instead
of `n_q_full`**. The r-chunk innermost-contiguous ordering is preserved
for fast per-r-chunk writes (see existing comment in
`isdf_fitting.py:1657-1667`).

For 6×6×1 CrI3 with `ntran=12` and `n_qpt_irr=4` (typical) this is a
**9× disk reduction**: ~1080 GB → ~120 GB per channel.

### 1.4 What about full-BZ q on the consumer side?

`compute_V_q_tile` today computes `V_q` at every full-BZ q. We exploit
the fact that **V_q has no τ-phase under {S|τ}** — only a centroid
permutation:

```
V_{Sq, π_s(μ), π_s(ν)} = V_{q, μ, ν}           (§2.2 derivation)
```

So the V_q orchestrator loops over IBZ q only (n_qpt_irr tiles, not
n_q_full), and a tiny post-loop kernel unfolds IBZ V_q → full-BZ V_q by
double-permuting the centroid axes. Compute and read I/O drop by
`n_q_full / n_qpt_irr` (e.g. 9× for CrI3 6×6×1).

This is strictly better than the current full-BZ loop. The unfold is
~`n_q_full · n_rmu²` complex multiply-adds per channel, dominated by
the V_q kernel itself by orders of magnitude.

---

## 2. Symmetry transformation derivation

### 2.1 Convention

WFN sym ops `{S | τ}`:
- `S = wfn.sym_matrices[s]` is the integer rotation in **crystal coords**
  (`mtrx`, BGW convention).
- `R_grid[s] := S` acts on G-vectors (column convention):
  `G' = S G` means `G' = R_grid[s] @ G` (Bohr-1 if cartesian, integer if
  crystal — both are equivalent under `S G_crys = (S G_cart)_crys`).
- `Rinv_grid[s] := S^{-1}` acts on real-space coordinates:
  `r' = Rinv_grid[s] @ r + τ` for the cartesian or fractional `r`.
- `τ = wfn.translations[s] / (2π)` is the fractional translation
  (BGW sign convention; see `symmetry_maps.py:341`).

We use Bloch phase convention `ψ_{nk}(r) = Σ_G c_{nk}(G) e^{i(k+G)·r}`
throughout. The "z" variant `z_{q,μ}(r) := e^{-iq·r} ζ_{q,μ}(r)` (current
ISDF docstring) is for the umklapp-free FFT in `v_q_tile.py`; the
on-disk ζ_{q,μ}(r) is the Bloch form (no per-q phase baked in).

### 2.2 Pair density transformation

Bloch coefficient: `ρ_{q}(G) = ∫ d³r e^{-i(q+G)·r} ρ_q(r) / V`.

Under `{S|τ}`: `r → r' = S^{-1}(r - τ)`. Substituting,

```
ρ_{Sq}(SG) = ∫ d³r e^{-i(Sq + SG)·r} ρ_q(S^{-1}(r-τ)) / V
           = ∫ d³r' e^{-i(Sq+SG)·(Sr'+τ)} ρ_q(r') / V          (r = Sr' + τ)
           = e^{-i(Sq+SG)·τ} ∫ d³r' e^{-i S(q+G) · Sr'} ρ_q(r') / V
           = e^{-i(Sq+SG)·τ} ρ_q(G)                            (S is unitary)
```

So **the pair density transforms with one τ-phase on (Sq, SG)** and no
phase on the rotation itself. Note `(Sq + SG)·τ = S(q+G)·τ` and equally
`(q+G)·(S^T τ) = (q+G)·(S^{-1} τ)` since S is orthogonal in crystal
coords with the appropriate metric — both forms appear in BGW source.

### 2.3 ζ transformation rule

ζ is fitted to interpolate ρ at the centroid r_μ:
`ρ_q(r) ≈ Σ_μ ζ_{q,μ}(r) ρ_q(r_μ)`.

With centroids closed under `{S|τ}` (orbit-aware k-means + `snap_orbits_to_grid`
ensure `S r_μ + τ ≡ r_{π_s(μ)}` mod 1), the same transformation law
applies to ζ with a centroid permutation on the μ leg:

```
ζ_{Sq, π_s(μ)}(SG) = e^{-i(Sq + SG)·τ} · ζ_{q,μ}(G)            (eq. 1)
```

Equivalently, **to unfold an IBZ ζ to a full-BZ q**: given `q_full = S · q_irr`,

```
ζ_{q_full, ν}(G_target) = e^{-i(q_full + G_target)·τ_s} · ζ_{q_irr, π_{s^{-1}}(ν)}(S^{-1} G_target)
                                                                (eq. 2)
```

(`s` here is the sym index witnessing `q_full = S q_irr`; the centroid
permutation needed is the **inverse** of `π_s` because we want "what
on-disk μ value maps to the requested output μ".)

### 2.4 V_q transformation rule (the IBZ-loop justification)

```
V_{q, μν} = Σ_G ζ*_{q,μ}(G) · v(q+G) · ζ_{q,ν}(G)
```

Apply eq. 1 to both ζ's, substitute G' = S^{-1} G_new:

```
V_{Sq, π_s(μ), π_s(ν)}
  = Σ_{G_new} ζ*_{Sq, π_s(μ)}(G_new) · v(Sq + G_new) · ζ_{Sq, π_s(ν)}(G_new)
  = Σ_{G_new} [e^{-i(Sq + G_new)·τ} ζ_{q,μ}(S^{-1} G_new)]* · v(Sq + G_new)
                                          · e^{-i(Sq + G_new)·τ} ζ_{q,ν}(S^{-1} G_new)
  = Σ_{G_new} e^{+i(Sq + G_new)·τ} e^{-i(Sq + G_new)·τ}
                                   ζ*_{q,μ}(S^{-1} G_new) · v(Sq + G_new) · ζ_{q,ν}(S^{-1} G_new)
  = Σ_{G'} ζ*_{q,μ}(G') · v(q + G') · ζ_{q,ν}(G')              (rename G' = S^{-1} G_new)
  = V_{q, μν}
```

τ-phases cancel (V is bilinear in ζ); `v(q+G)` is invariant because S
preserves the metric (`|Sq+SG|² = |q+G|²`).

**Conclusion**: V_q at full-BZ q is a pure centroid-axis double-permute
of V_q at IBZ q:

```
V_{q_full, μ', ν'} = V_{q_irr[i(q_full)], π_{s(q_full)^{-1}}(μ'),
                                          π_{s(q_full)^{-1}}(ν')}    (eq. 3)
```

So we **compute V_q on the IBZ only** and unfold cheaply post-loop.

---

## 3. Module layout

### 3.1 `file_io/mf_header.py` (new)

Single source of truth for `mf_header/` group r/w. Used by `ZetaReader`
now, and by future kin_ion/V_qmunu/sigma_omega readers as a drop-in.

```python
@dataclass
class MfHeader:
    # exposes the same set of attributes WFNReader has — verbatim
    version, flavor
    nspin, nspinor, nkpts, nbands, ngkmax, ecutwfc
    kgrid, shift, ngk, ifmin, ifmax, kweights, kpoints, energies, occs
    ng, ecutrho, fft_grid
    ntran, cell_symmetry, sym_matrices, translations
    cell_volume, recip_volume, alat, blat, nat, avec, bvec, adot, bdot
    atom_types, atom_positions
    # plus derived: kpt_starts, atom_crys (same as WFNReader)

def read_mf_header(h5_path: str) -> MfHeader: ...
def copy_mf_header(wfn_path: str, out_path: str,
                   out_mode: str = 'a') -> None: ...
```

`WFNReader` is refactored (small, contained change) to populate its
header fields by calling `read_mf_header` internally. This guarantees
consumer parity.

### 3.2 `file_io/isdf_header.py` (new)

Tiny — only the irreducible content. Sym-derivable arrays live in
`SymMaps` / `orbit_syms` and are recomputed at read time.

```python
@dataclass
class IsdfHeader:
    density: str                 # 'scalar' | 'current' | 'mu_L=<int>'
    vertex_mu_L: int
    r_mu_fft_idx: np.ndarray     # (n_rmu, 3) int32 — primary representation
    r_mu_crystal: np.ndarray     # (n_rmu, 3) float64 — convenience / inspection

def read_isdf_header(h5_path: str) -> IsdfHeader: ...
def write_isdf_header(h5_path: str, header: IsdfHeader,
                      mode: str = 'a') -> None: ...
```

### 3.3 `file_io/zeta_reader.py` (new)

```python
class ZetaReader:
    """wfn_reader-shaped reader for zeta_q.h5.

    Eager:  mf_header attributes (same surface as WFNReader),
            isdf_header attributes (centroids, q-IBZ, perms).
    Lazy:   per-(q, μ-slab) ζ reads.

    G-flat is the primary read API; r-space is kept for the
    legacy compute_V_q path until it's removed."""

    # === eager (set in __init__) ===
    # all of WFNReader's mf_header attributes
    # + density, vertex_mu_L, n_rmu, n_rmu_padded,
    #   r_mu_fft_idx, r_mu_crystal
    #
    # === derived on first access, cached on self ===
    # _sym: SymMaps  (built from mf_header sym tables, kgrid)
    # _q_irr_table: (q_irr_kgrid_int, full_to_irr_idx, full_to_irr_sym)
    #               — from self._sym.find_irreducible_qpoints()
    # _centroid_sym_perm: (n_sym, n_rmu) int32
    #               — from orbit_syms.compute_centroid_sym_perm(...)

    # === lazy ===
    def read_zeta_q_slab_G(
        self,
        q_irr_indices: np.ndarray,        # (Q,) IBZ indices
        mu_start: int, mu_end: int,
        *, mesh: Mesh,
        partition_spec_in=P(None, None, ('x', 'y')),
        partition_spec_out=P(None, ('x', 'y'), None),
    ) -> jax.Array:
        """Return G-flat ζ:  (Q, μ_end - μ_start, n_G_sph) c128, sharded
        per partition_spec_out.

        Steps (all inside a jit'd shard_map):
          1. SlabIO read r-space (Q, n_rtot, μ) — async dispatched.
          2. Transpose to (Q, μ, r); reshape to (Q, μ, nx, ny, nz);
             multiply by per-q phase exp(+i q·r).      ←  was inside kernel
          3. 3D FFT over the (nx, ny, nz) axes (μ-sharded).
          4. Sphere gather: (Q, μ, n_G_sph).
        """
        ...

    # legacy: same I/O contract as today's SlabIO.read_slab on 'zeta_q'.
    # Kept until the only consumer of r-space ζ (currently
    # compute_V_q_tile) migrates to G-flat.
    def read_zeta_q_slab_r(self, q_irr_indices, mu_start, mu_end, *,
                           mesh, partition_spec) -> jax.Array: ...
```

Notes:
- `read_zeta_q_slab_G` reads **IBZ q only**. No symmetry unfolding
  happens in the reader. The orchestrator iterates IBZ; unfolding
  to full-BZ V_q is done post-loop via the centroid double-permute
  (§2.4).
- The FFT moved here is identical to today's `_zeta_disk_to_G` in
  `v_q_tile.py:666`. Same shardings, same per-q phase fn, same sphere
  gather. Just relocated.

### 3.4 Writer changes — `common/isdf_fitting.py`

Minimal surgery to `fit_zeta_to_h5`:

1. After computing C_q / L_q, build `IsdfHeader` host-side and write
   `mf_header` (copied from `wfn._filename`) + `isdf_header` into the
   output file **before** the chunk loop. The `SlabIO.create_dataset`
   call for `zeta_q` runs after these are in place.
2. The leading axis of `zeta_q` becomes `n_qpt_irr` instead of `nq`.
   The chunk loop's q-axis indexing changes from full-BZ to IBZ — the
   IBZ map is `sym.q_irr_indices` (a new helper, §4.3). Inside the
   fit, **C_q and Z_q are already computed at all q's via FFT** (q lives
   on the k-grid, FFT delivers full-BZ q's), so we just **select the
   IBZ subset** before the writes.
3. The async writer thread / FFI write path slices the q axis to IBZ
   indices in the same place we read C_q from. No other change.

### 3.5 V_q kernel changes — `gw/v_q_tile.py`

1. Drop `_zeta_disk_to_G` from the kernel. The kernel's input becomes
   G-flat ζ directly:
   ```
   def _kernel_body(V_acc, g0_acc, zeta_L_G, zeta_R_G_or_None,
                    v_per_G_batch, q_lo_dyn, mu_lo_dyn, nu_lo_dyn):
       zeta_mu_X = w_sc(zeta_L_G, blk_x_sh)
       zeta_nu_Y = w_sc(zeta_R_G_or_alias, blk_y_sh)
       zeta_mu_X = w_sc(zeta_mu_X * v_per_G_batch[:, None, :], blk_x_sh)
       V_block = einsum('qmG,qnG->qmn', conj(zeta_mu_X), zeta_nu_Y)
       ...
   ```
   `phase_batch` and `n_rtot`-related args are gone. `g0` is now read
   directly off the (Q, μ, G=0) entry of `zeta_L_G` rather than the
   `(:,:,0)` r-space corner of the FFT box.
2. The `compute_V_q_tile` outer driver replaces `SlabIO.read_slab(..., 'zeta_q')`
   with `zeta_reader.read_zeta_q_slab_G(...)`. The q-batch construction
   iterates IBZ indices only.
3. AOT chooser: the per-rank peak shrinks because the FFT input + plan
   workspace are gone from the kernel boundary. `_aot_full_kernel_peak`
   gets re-run on the new HLO; the memory model rebuild is straightforward
   (axes shrink — strictly less peak).
4. Post-loop unfold: a small jitted op that takes V_q at IBZ
   (`n_qpt_irr, n_rmu, n_rmu`) and a centroid-perm tensor
   (`n_q_full, n_rmu`) → V_q at full-BZ (`n_q_full, n_rmu, n_rmu`) via
   `V_q_full[q_full, μ', ν'] = V_q_irr[i(q_full), perm[q_full, μ'], perm[q_full, ν']]`.
   Cost: 9× smaller compute and read than today's full-BZ kernel loop.

### 3.6 g0 head correction

The Coulomb head term written into `g0_work` today equals
`ζ_{q,μ}(G=0)` for q=0. In G-flat that's just `zeta_L_G[q=q0, μ, 0]`,
read directly. No FFT-side hack. The `wants_g0` / `write_g0` paths
become trivial reads.

---

## 4. New helpers to land

### 4.1 `centroid/orbit_syms.py::compute_centroid_sym_perm`

Given `r_mu_fft_idx (n_rmu, 3)`, `R_grid (n_sym, 3, 3)`, `tau (n_sym, 3)`,
and `fft_grid (3,)`, returns `sym_perm (n_sym, n_rmu) int32` such that
`r_{sym_perm[s, μ]} ≡ S_s r_μ + τ_s` on the FFT grid.

Validation: for each s, `sym_perm[s]` is a permutation (every value in
`[0, n_rmu)` appears exactly once). If validation fails the centroid
file is not orbit-closed under the WFN sym set — the writer raises with
a pointer to `kmeans_cli --no-orbit ... ` (see KNOWN_SANDBOX_ERRORS).

This generalizes the orbit machinery already in `orbit_syms.py` — the
existing helpers compute orbit closure / canonical reps but don't return
the explicit permutation per sym op.

### 4.2 `common/symmetry_maps.py::find_irreducible_qpoints`

Mirror of the k-IBZ reduction already implicit in WFN.h5 (k IBZ list is
stored; q IBZ is currently not exposed). Returns:
- `q_irr_kgrid_int (n_qpt_irr, 3) int32`
- `full_to_irr_idx (n_q_full,) int32`
- `full_to_irr_sym (n_q_full,) int32`

This is a small ~40 LOC host-side helper (BZ-fold via R_grid; standard).

### 4.3 `gw/v_q_tile.py::_unfold_v_q_ibz_to_full`

Post-loop unfold kernel. ~30 LOC. Pure index-gather; no phase.

---

## 5. Phasing

Three commits, each independently testable:

**C1 — `mf_header` module + `IsdfHeader` schema + writer wires them in.**
Files: `file_io/mf_header.py` (new), `file_io/isdf_header.py` (new),
`centroid/orbit_syms.py` (+ `compute_centroid_sym_perm`),
`common/symmetry_maps.py` (+ `find_irreducible_qpoints`),
`common/isdf_fitting.py` (writes headers; on-disk q axis still full-BZ
for this commit to keep changes minimal).
Test: pytest unit tests for header round-trip on a captured WFN.

**C2 — IBZ-only q axis on disk + V_q orchestrator loops IBZ.**
Files: `common/isdf_fitting.py` (zeta_q axis n_qpt_irr),
`gw/v_q_tile.py::compute_V_q_tile` (q-batch list = IBZ, post-loop
unfold), `gw/gw_init.py` (driver glue).
Test: MoS2 3×3 COHSEX end-to-end; V_q must match prior bit-for-bit on
the V_q,full[q] outputs.

**C3 — `ZetaReader` + G-flat kernel input.**
Files: `file_io/zeta_reader.py` (new), `gw/v_q_tile.py::_make_V_q_tile_kernel`
(drop FFT + phase + sphere from inside; consume G-flat), AOT chooser
rebuild.
Test: MoS2 3×3 COHSEX bit-identical vs C2; CrI3 6×6 smoke.

`compute_V_q_bispinor` consumes the new reader the same way (drop-in).

---

## 6. Risks / things to verify

1. **Phase convention sign**. The current `_zeta_disk_to_G` uses
   `phase_batch = exp(±i q·r)` from `phase_fn`. The reader must match
   that sign exactly. Quickest way to lock it down: capture the
   `phase_fn` output for q=(1/3, 0, 0) on a known MoS2 file and compare
   to a fresh ZetaReader G-flat read for the same q. Bit-equal or fix
   the sign.
2. **g0 from G-flat vs r-corner**. Today `g0_blk = zeta_box[:, :, 0]`
   after the FFT — this is the `r=0` corner of the FFT box, which is
   `Σ_G ζ_{q,μ}(G)` (DC of the IFFT-back), used by the head correction.
   In G-flat the equivalent is `zeta_q_G[q, μ, G_sph_idx_of_zero]`
   provided q=0 (the head only matters at q→0). Need to confirm the
   sphere indexing — `sphere_idx[0]` is conventionally G=(0,0,0) but
   verify in `gw/gw_init.py:compute_V_q` where `sphere_idx` is built.
3. **Centroid orbit closure**. If the centroid file was produced with
   `kmeans_cli --no-orbit` (the recommended path post-2026-05-07 per
   KNOWN_SANDBOX_ERRORS), orbit closure is NOT guaranteed. We must
   either:
     (a) re-run kmeans in orbit-aware mode for files used with the new
         writer, OR
     (b) restrict the IBZ-only writer to identity sym op (n_qpt_irr =
         n_q_full) when orbit closure fails — i.e. fall back to the
         current full-BZ layout. The writer detects this in
         `compute_centroid_sym_perm`.
   Recommend (b) as the safe fallback; (a) for the production runs
   we care about.
4. **WFN file lifetime**. The writer copies `mf_header` from
   `wfn._filename` — but `wfn` may be a `PhDF5WFN` reading from a
   different path, or a fresh QP rotation. We pass the WFN.h5 path
   explicitly to the writer, not the WFNReader object's internal
   handle, to avoid lock contention.
5. **Bispinor four-channel files**. `zeta_q.h5` (μ_L=0),
   `zeta_q_mu1.h5`, `zeta_q_mu2.h5`, `zeta_q_mu3.h5`. All four get the
   same `mf_header` (same WFN source) but different `isdf_header`
   (`vertex_mu_L` field + possibly different centroid file → different
   `r_mu_crystal` and `sym_perm`). The writer takes `vertex_mu_L` and
   the centroid path as inputs; nothing else to handle.

---

## 7. What this does NOT change

- ζ on disk remains r-space, layout `(n_qpt_irr, n_rtot, n_rmu)`.
- FFT count per V_q run is the same as today (the FFT moved from
  inside the kernel to inside the reader, but it's still done once per
  (q-batch, μ-tile) read).
- Phase 2 (G-flat on disk) eliminates the FFT entirely from the
  read+kernel path. That's the medium-term big win — 16-20× disk
  reduction (r-space → G-sphere) on top of the 9× from IBZ-only. Total
  ~150× vs today for CrI3.

---

## 8. Sign-off checklist

If this design holds:
- [ ] Eq. 1 (`ζ_{Sq, π_s(μ)}(SG) = e^{-i(Sq+SG)·τ} ζ_{q,μ}(G)`) is
      correct in our WFN sign convention.
- [ ] Eq. 3 (V_q is centroid-double-permute under {S|τ}) is correct;
      no τ-phase needed for V_q unfold.
- [ ] Schema in §1 acceptable; group/dataset names ok.
- [ ] Module split `file_io/mf_header.py` + `file_io/isdf_header.py`
      is the right factoring (rather than collapsing into one).
- [ ] V_q kernel migration in C3 (drop FFT from kernel; consume G-flat
      from reader) is the right shape vs. an alternative where the FFT
      stays in the kernel and the reader returns r-space.

Once the above are confirmed, I'll proceed to C1 → C2 → C3 in three
small commits on branch `agent/zeta-ibz-header`.
