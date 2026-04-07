# Quasiparticle Input File Construction

This document tells you how to construct the complete set of input files and directory
tree for a GW quasiparticle calculation, given a material specification (CIF, POSCAR,
or natural-language description) and the user's stated goals. A calculation may use any
subset of the three codes — Quantum ESPRESSO (QE), BerkeleyGW (BGW), and GWJAX — and
you must determine which pipeline applies before generating anything.

## Determine the pipeline

Ask the user (or infer from context) which of these pipelines they want:

| Pipeline | Steps | When to use |
|----------|-------|-------------|
| QE only | SCF, NSCF, pw2bgw | Ground-state wavefunctions only; no GW |
| QE → BGW | SCF, NSCF, NSCFq, pw2bgw, pw2bgwq, epsilon, sigma | Standard BerkeleyGW GW calculation |
| QE → GWJAX | SCF, NSCF, pw2bgw, ISDF + GW | GWJAX GW calculation (ISDF-compressed screening) |
| QE → BGW + GWJAX | All of the above | Matched comparison between BGW and GWJAX |

GWJAX does not need WFNq (no separate shifted-grid calculation). BGW always needs both
WFN and WFNq, except for 0D molecules at Γ only (no WFNq needed; epsilon uses only
eps0mat.h5). Both codes need the same WFN.h5 from the unshifted NSCF.

## Step 0: Determine system essentials

Before writing any input file, establish these quantities. Report them to the user
and confirm before proceeding.

### Chemical identity and structure

Parse the CIF/POSCAR/description to obtain: lattice vectors, atomic species and
positions, and space group. Choose a `prefix` for the calculation — use the chemical
formula if obvious (e.g. `MoS2`, `GaSb`), or something short and descriptive otherwise.

### Number of valence electrons

Read the pseudopotential `.upf` files for each atom type from `assets/pseudos_standard`
(or `pseudos_stringent` if semicore states are requested). Grep for the line containing
`z_valence=` and sum over all atoms:

```
N_val = sum over atom types: (number of atoms of that type) × (z_valence)
```

Report `N_val` to the user. For a quick test, suggest `ncond = N_val` conduction bands;
for production, suggest `ncond = 2 × N_val`. The total band count for the GW calculation
is `nband = N_val + ncond`. The WFN.h5 must contain at least `nband + 2` bands (the +2
accounts for frequent 2-fold degeneracies at the band edges, which BGW needs clearance
around).

### Metal or insulator

Determine whether the system is likely metallic or insulating. This matters because
metals require much finer k-grids (at least 12 points in periodic directions) to
resolve the Fermi surface, while insulators can use coarser k-grids (3×3 or 6×6 for
initial tests). If the user requests a metal, confirm they understand the cost
implications. BGW requires `screening_metal` in `sigma.inp` for metallic systems.

### System dimensionality

Determine `sys_dim` (used by both BGW Coulomb truncation and GWJAX `sys_dim`):

- **3D** (bulk): k-grid extends in all three directions (e.g. 3×3×3), z lattice vector
  is not vacuum-padded.
- **2D** (slab): z-axis lattice vector is along cartesian z only, k-grid is NxNx1, the
  z lattice parameter exceeds 12 Å, and the fractional z-coordinates of atoms are
  confined to less than 40% of the periodic z-range:
  ```python
  def zwidth(frac_z):
      z = sorted(x % 1.0 for x in frac_z)
      return 1 - max(b - a for a, b in zip(z, z[1:] + [z[0] + 1]))
  ```
  `zwidth < 0.40` → 2D.
- **1D** (wire): zwidth < 0.40 on both x and y crystal axes but not z, k-grid is 1×1×N.
  Warn the user that 1D is not properly implemented in GWJAX.
- **0D** (molecule): zwidth < 0.50 in all three dimensions, single k-point at Γ.
  No WFNq is needed for BGW (only eps0mat.h5 is produced, no epsmat.h5).

### Coulomb truncation flags

| Dim | QE (`assume_isolated`) | BGW | GWJAX (`sys_dim`) |
|-----|------------------------|-----|--------------------|
| 3D  | (none) | (none) | 3 |
| 2D  | `'2D'` | `cell_slab_truncation` | 2 |
| 1D  | (none; no QE support) | `cell_wire_truncation` | 1 |
| 0D  | (none; no QE support) | `cell_box_truncation` | 0 |

## Step 1: QE SCF input (`scf.in`)

Start from the `scf.in` template in `assets/templates/`. The template contains the
`&electrons` settings (convergence threshold, mixing parameters, diagonalization) and
structural cards that should not be changed without good reason. Modify only the
fields listed below.

Key parameters to set per-system:
- `prefix` — the chosen prefix
- `ecutwfc = 30.0` for quick tests; `80.0` for production
- `nat`, `ntyp` — total atoms, distinct atom types
- `nbnd` — only needs enough for SCF convergence: `N_val + 8` is sufficient
- `assume_isolated` — set per dimensionality table above
- `ATOMIC_SPECIES` — e.g. `Mo 95.94 'Mo.upf'` (masses are unused by `pw.x`
  but required syntactically)
- `CELL_PARAMETERS` — lattice vectors from the CIF/POSCAR
- `ATOMIC_POSITIONS` — fractional coordinates from the CIF/POSCAR
- `K_POINTS automatic Nk Nk Nz 0 0 0` — coarse unshifted grid (e.g. 3×3×1 for 2D)

Parameters that must always be set (already in template, verify they are present):
- `noncolin = .true.`, `lspinorb = .true.` — spinor wavefunctions (nspinor=2),
  currently required for all GWJAX calculations
- `no_t_rev = .true.` — the BGW WFN.h5 format cannot represent symmetries involving
  time reversal; both BGW and GWJAX read WFN.h5, so this is always required
- `pseudo_dir = './'` — pseudopotentials are symlinked or copied into the working directory

Pseudopotential files live in `assets/pseudos_standard/` and should be copied or
symlinked to the working directory.

## Step 2: NSCF inputs

Each NSCF overwrites `<prefix>.save/`, so pw2bgw + wfn2hdf must run **immediately**
after each NSCF before starting the next `pw.x` call.

### Unshifted grid (`nscf.in`) → WFN

- `calculation = 'nscf'`
- `nbnd` — must be ≥ `nband + 2` (the total bands BGW/GWJAX will use, plus clearance).
- `K_POINTS` — the full (unfolded) GW k-grid. For BGW-compatible calculations, generate
  this with `data-file2kgrid.py` + `kgrid.x` and use `K_POINTS crystal` with the
  explicit k-point list from kgrid.out. For GWJAX-only calculations,
  `K_POINTS automatic Nk Nk Nz 0 0 0` is acceptable.
- All other system parameters (`ecutwfc`, `noncolin`, etc.) must match `scf.in`.
- Add a symlink to the SCF save directory: `ln -sf ../<scf_dir>/<prefix>.save .`

### Shifted grid (`nscfq.in`) → WFNq (BGW pipelines only, not 0D)

Same as `nscf.in` but every k-point is offset by a small shift. Generate the shifted
grid with `data-file2kgrid.py --qshift 0.001 0.0 0.0` + `kgrid.x`, and use the
resulting `K_POINTS crystal` block in `nscfq.in`. The crystal-coordinate shift for a
Nk×Nk×Nz grid is 0.001/Nk along b₁.

### BGW-compatible k-grid generation

For explicit BGW runs, generate k-grids with `data-file2kgrid.py` + `kgrid.x` rather
than hand-typing them, to ensure compatibility with BGW symmetries:

```bash
# Unshifted grid for WFN
python data-file2kgrid.py --kgrid Nk Nk Nz data-file-schema.xml kgrid.inp
kgrid.x kgrid.inp kgrid.out kgrid.log

# Shifted grid for WFNq (not needed for 0D)
python data-file2kgrid.py --kgrid Nk Nk Nz --qshift 0.001 0.0 0.0 \
    data-file-schema.xml kgridq.inp
kgrid.x kgridq.inp kgridq.out kgridq.log
```

The `data-file-schema.xml` is produced by the SCF calculation (inside `<prefix>.save/`).
Use the resulting k-point lists in `nscf.in` and `nscfq.in` respectively.

## Step 3: pw2bgw inputs

### Unshifted (`pw2bgw.in`)

- `wfng_file = 'WFN'`, `wfng_flag = .true.`
- `wfng_nk1, nk2, nk3` — must match the k-grid dimensions
- `wfng_dk1, dk2, dk3` — all 0.0 (unshifted)
- `real_or_complex = 2` — required for spinor wavefunctions
- `vxc_flag = .true.` — writes `vxc.dat`, needed by BGW sigma
- `kih_flag = .true.` — writes `kih.dat`, needed by GWJAX for QP energy output
- `vxc_diag_nmin = 1`, `vxc_diag_nmax = nbnd` — range of bands for matrix elements

### Shifted (`pw2bgwq.in`) (BGW pipelines only, not 0D)

- `wfng_file = 'WFNq'`, `wfng_flag = .true.`
- `wfng_nk1, nk2, nk3` — must match the k-grid dimensions (same as unshifted)
- `wfng_dk1 = {0.001/Nk}`, `wfng_dk2 = 0.0`, `wfng_dk3 = 0.0`
- `real_or_complex = 2` — required for spinor wavefunctions (same as unshifted)
- `vxc_flag = .false.`, `kih_flag = .false.` — only needed from the unshifted grid

## Step 4: BGW epsilon input (`epsilon.inp`)

- `epsilon_cutoff` — should equal `ecutwfc` for comparison purposes. If ecutwfc > 50.0,
  confirm with the user due to memory constraints.
- `number_bands` — must equal `nband` (and match sigma.inp and GWJAX). Must be ≤ nbnd - 2.
- `frequency_dependence 3` — Godby-Needs GPP (two frequency points: static + imaginary).
  Use `0` for static COHSEX, `2` for full-frequency.
- `broadening 0.001`
- `use_wfn_hdf5` — use HDF5 wavefunctions for faster I/O.
- `degeneracy_check_override` — useful for testing.
- Coulomb truncation flag per dimensionality table.

### q-points block

The q-points come from the WFN k-grid. To construct the block:

1. Copy the k-point list from the WFN `nscf.in` (or from `kgrid.out`).
2. Set all weights (fourth column) to 1.0, and add a fifth column of all 0's.
3. Replace the Γ point with the shifted WFNq point in crystal coordinates,
   and change its fifth column to 1 (indicating epsilon should read WFNq for this point).

For a 3×3×1 grid: 1 shifted + 8 regular = 9 q-points. For 0D (single Γ point): only
the shifted q→0 point (1 q-point), and only `eps0mat.h5` is produced.

## Step 5: BGW sigma input (`sigma.inp`)

- `screened_coulomb_cutoff` — should match `epsilon_cutoff`.
- `number_bands` — must match epsilon.
- `band_index_min / band_index_max` — which bands to compute Σ for (1-indexed, inclusive).
  Default: 8 valence + 4 conduction bands around the Fermi level. For N_val valence
  electrons, `band_index_min = N_val - 7`, `band_index_max = N_val + 4`. Ask the user
  if they want different coverage.
- `use_wfn_hdf5`
- `dont_use_vxcdat` + `use_kihdat` — use kih.dat (KIH = Kinetic + Ionic + Hartree)
  instead of vxc.dat for QP energy construction. Preferred over vxc.dat because it
  supports all functionals (LDA/GGA/meta-GGA/hybrid). Both flags are required together.
  Note: the flag is `use_kihdat` (no underscore), not `use_kih_dat`.
- Coulomb truncation flag — must match epsilon.
- `frequency_dependence` — must match epsilon.

For GN-GPP (`frequency_dependence 3`):
- `exact_static_ch 0`
- `invalid_gpp_mode 2` — overwrite imaginary poles to 2 Ry
- `gpp_broadening 0.00001`
- `gpp_sexcutoff 1000.0`

For static COHSEX (`frequency_dependence 0`):
- `exact_static_ch 0`

### k-points block

Same k-point list as epsilon's q-points, but: use `begin kpoints` / `end`, keep only
the first four columns (drop the fifth), and restore the first point to Γ = (0, 0, 0).

## Step 6: GWJAX input (`cohsex.in`)

INI-style with a `[cohsex]` section.

- `nval` — number of valence electrons (= N_val)
- `ncond` — number of conduction bands (user's choice; N_val for quick test)
- `nband` — must match BGW `number_bands` if running a comparison
- `sys_dim` — per dimensionality table
- `wfn_file = WFN.h5`
- `output_file = eqp0.dat`
- `sigma_omega_h5_file = sigma_mnk.h5` — HDF5 output for full Σ(ω) matrix
- `centroids_file = centroids_frac_<N>.txt` where N = 8 × nband for standard accuracy,
  12 × nband for high accuracy. Generated by `centroid.kmeans_isdf` (see Execute Workflow step 5a).
- `kin_ion_file = kin_ion.h5` — kinetic + ionic matrix elements (T + V_loc + V_NL),
  shape (nk, nb, nb). Generated by `gw.kin_ion_io_chunked -i cohsex.in` (see Execute
  Workflow step 5c). Required for QP energy output. Must be generated after WFN.h5 and
  pseudopotentials are in the working directory.
- `restart = false` — first run must build ISDF tensors. Set to `true` for reruns with
  the same geometry and band count.
- `bispinor = false` for standard spinor calculations (the ISDF code handles spinors
  internally; this flag controls a different feature).
- `use_chunked_isdf = true` — memory-efficient algorithm.

For GN-PPM dynamic sigma (matching BGW `frequency_dependence 3`):
- `use_ppm_sigma = true`
- `ppm_omega_p = 2.0` (Ry, matches BGW default imaginary frequency probe)
- `sigma_at_dft_energies = true`
- `sigma_debug_split_contrib = true`
- `sigma_freq_debug_output = true` — writes `sigma_freq_debug.dat`

For static COHSEX (matching BGW `frequency_dependence 0`):
- `use_ppm_sigma = false`
- `x_only = false`, `do_screened = true`

### Mini-BZ Coulomb head overrides

GWJAX and BGW use slightly inequivalent head computation methods (GWJAX uses
analytic forms of the q→0 limit, BGW fits a model function) that can cause
Sigma errors in the extreme coarse k-grid limit. For matched comparisons,
extract BGW's mini-BZ heads from `frequency_dependence 3` epsilon.out and
pass as overrides in cohsex.in. See `PARSE_OUTPUTS.md` for the parser.

```ini
vhead = 315.0137
whead_0freq = 55.6197
whead_imfreq = 246.1099
```

### K_POINTS block

GWJAX uses the same `K_POINTS {crystal_b}` format as QE band-structure calculations.
For a comparison run, include the high-symmetry path appropriate to the lattice.

## Step 7: Comparison script

If the pipeline includes both BGW and GWJAX, include `compare_bgw_gwjax.py` in the
directory. The script matches BGW's symmetry-reduced k-points to GWJAX's full-grid
k-points by comparing crystal coordinates mod G. It reads primed Cor' from
`sigma_hp.log` and `sigC_EDFT` from `eqp0.dat`. See `PARSE_OUTPUTS.md` for column
definitions and parser functions.

## Constructing variant input files

When creating a variant (e.g. changing k-grid size or number of bands), do not
reconstruct all input files from scratch. Instead:

1. Copy only the input files that need to change from the parent run.
2. Modify only the fields listed in the variant's `changes` (see AGENTS.md for the
   variant manifest format).
3. Regenerate any derived blocks — e.g. if k-grid changes, regenerate the k-point
   lists in nscf.in/nscfq.in via kgrid.x, and reconstruct the q-points/k-points
   blocks in epsilon.inp/sigma.inp from the new grid.
4. Symlink unchanged files from the parent.

## The handoff manifest (`manifest.yaml`)

After constructing all input files, write a `manifest.yaml` in the run directory root.
This is the handoff document read by the runner agent to know what to execute and in
what order. It also records the scientific parameters so that variant runs can be
constructed by modifying only the changed fields.

```yaml
system:
  formula: MoS2
  prefix: MoS2
  dimensionality: 2
  n_val: 26
  n_cond: 44
  n_band: 80
  ecutwfc: 30.0
  kgrid: [3, 3, 1]
  metal: false

pipeline: qe+bgw+gwjax    # one of: qe, qe+bgw, qe+gwjax, qe+bgw+gwjax

platform: perlmutter       # or: laptop

files:
  scf: scf.in
  nscf: nscf.in
  nscfq: nscfq.in          # absent if pipeline lacks BGW, or if 0D
  pw2bgw: pw2bgw.in
  pw2bgwq: pw2bgwq.in      # absent if pipeline lacks BGW, or if 0D
  epsilon: epsilon.inp      # absent if pipeline lacks BGW
  sigma: sigma.inp          # absent if pipeline lacks BGW
  cohsex: cohsex.in         # absent if pipeline lacks GWJAX
  compare: compare_bgw_gwjax.py  # present only for qe+bgw+gwjax

pseudopotentials:
  - Mo.upf
  - S.upf

gwjax:
  n_centroids: 640          # 8 × nband
  centroids_file: centroids_frac_640.txt
  use_ppm_sigma: true

bgw:
  frequency_dependence: 3
  band_index_min: 19
  band_index_max: 30
  epsilon_cutoff: 30.0

status:
  scf: pending
  nscf: pending
  pw2bgw: pending
  nscfq: pending
  epsilon: pending
  sigma: pending
  gwjax_centroids: pending
  gwjax_dipole: pending
  gwjax_kin_ion: pending
  gwjax_gw: pending
  compare: pending
```
