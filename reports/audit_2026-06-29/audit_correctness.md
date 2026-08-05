# Audit — Correctness & regressions lens (2026-06-29)

READ-ONLY audit. Lens: would any change break a real run or install? Every assertion
below was checked against the actual diff lines and, where relevant, the live filesystem
($HOME/software trees) and the parsed pyproject/uv.lock.

## Scope reviewed
- Migration commit `dc5f7f7` (FFI defaults $SCRATCH → $HOME/software)
- S1 path-strips in `ca840ec` (run_shifter.sh, CMakeLists.txt, site_config.sh, pyproject.toml)
- `pyproject.toml` extras structure + jax/jaxlib pin coherence
- `uv.lock` (`ea1ea3c`) consistency with pyproject
- Sandbox run scripts (`runs/**/*.sh,*.sbatch`) + `skills/execute_workflow/SKILL.md`

---

## Verified-correct (no defect)

**Migration dc5f7f7 — mount targets & LD paths byte-identical.** Full `diff
main..branch` of `run_shifter.sh` shows ONLY: NVHPC_HOST default (line 44), the two
PHDF5_DEFAULT lines (54, 61), LORRAX_SRC/LORRAX_SITE block (95-108), and
LORRAX_FFI_SLATE_DIR + SLATE_INSTALL_HOST defaults (118-119) changed. Container mount
targets (`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate`), every LD_LIBRARY_PATH entry
(incl. `/lorrax_nvhpc/25.5_cuda12.9/...`), LD_PRELOAD, MPICH_GPU_SUPPORT_ENABLED, and
`LORRAX_MPI_*` are unchanged. site_config.sh: only the 3 `*_DEFAULT` lines + comment
changed. Intent achieved with no collateral.

**All 4 relocated dep trees exist at $HOME with the referenced subpaths.** Verified on
disk: `$HOME/software/lorrax_nvhpc/{0.7.2_cuda12.9,25.5_cuda12.9}`,
`lorrax_phdf5_cray/stage`, `lorrax_phdf5_openmpi/stage`, `lorrax_slate_cray/stage`
(incl. `lib/libmpi_gtl_cuda.so.0` for the LD_PRELOAD), `slate/install/lib64`. Both nvhpc
subpaths referenced anywhere (0.7.2 in SKILL/VI3, 25.5 in run_shifter/Si scripts) are
present, so no dangling subpath.

**No missing libcal under the 0.7.2 subpath (looked, not a bug).** The SKILL/VI3
LD_LIBRARY_PATH lists only `.../0.7.2_cuda12.9/math_libs/12.9/lib64`, which contains
*only* `libcusolverMp.so*` (no libcal). `readelf -d libcusolverMp.so.0.7.2.0` shows it
links `libnccl.so.2` directly with NO libcal NEEDED — this is exactly the "CAL→NCCL ABI
fix" in the site_config comment. libnccl is in the container. Consistent with the
validated 4-GPU run.

**pyproject.toml valid + internally consistent.** Parses with `tomllib`. jax/jaxlib both
`>=0.5.3,<0.6` across core deps, `[cuda12]` extra, and `jax` dependency-group — all agree.
mkdocs-* moved into `[project.optional-dependencies] docs`; new `[cuda12]` extra. No
remaining reference to dropped `pybind11`/`pdoc`/`pydoc-markdown` anywhere in tracked
files (git grep clean). Dropping pybind11 is safe: no `pybind11/` include exists; the FFI
`api.cc`/`ffi_loader.py` are ctypes plain-C ("no pybind/nanobind dependency").

**uv.lock consistent with pyproject.** jax==0.5.3, jaxlib==0.5.3. Zero `cu13` strings;
60 cu12 entries. pybind11/pdoc/pydoc-markdown absent from lock. mkdocs present only under
`extra == 'docs'` markers. `[package.metadata]` requires-dist + requires-dev exactly
mirror pyproject (docs/cuda12 extras; build group sans pybind11; jax group with cuda12).
`provides-extras = ["docs", "cuda12"]`.

**run_shifter.sh LORRAX_SRC auto-derivation is correct.** `../../..` from
`src/ffi/common/cpp/` resolves to `.../src` (verified). Empty LORRAX_SITE → PYTHONPATH is
just `${LORRAX_SRC}` (guarded by the `if [[ -n ... ]]`), no trailing-colon bug. The FFI
test scripts that actually invoke run_shifter.sh (slate_*/cusolvermg_* in src/common)
import none of h5py/scipy/matplotlib, so the now-empty LORRAX_SITE default does not break
them. CMakeLists SLATE default `$ENV{HOME}/software/slate/install` resolves to a real dir.

**SKILL.md $SEL/$INC wiring correct.** `LORRAX_SRC` (148) defined before `VOL`(149)/
`SHIFTER`(152)/`SEL`(165)/`INC`(166); all used only in Steps 5/6 (189+). Order
`$SEL $SHIFTER ... $INC python3` matches run_shifter.sh's
`"${SRUN_WRAPPER[@]}" "${SHIFTER_ARGS[@]}" "${IN_CONTAINER}"`. Both wrapper scripts exist,
are +x, and are `exec "$@"` chains (SEL sets CUDA_VISIBLE_DEVICES=$SLURM_LOCALID on host
before shifter; INC re-asserts MPICH_GPU_SUPPORT_ENABLED inside). All Step-5/6 GWJAX srun
commands carry both wrappers (the QE/BGW sruns correctly do not). Step 6 command is valid
bash and runs as written: `srun ... $SEL $SHIFTER --env=... $INC python3 -m gw.gw_jax -i
$(pwd)/cohsex.in 2>&1 | tee gw.out`.

**SKILL.md `kmeans_isdf` → `kmeans_cli` + drop of `--no-plot` is a real fix, not a break.**
`centroid/kmeans_isdf.py` has no `__main__`/`main` (would fail `python -m`);
`centroid/kmeans_cli.py` has `def main()` and is the `lorrax-centroids` entrypoint. Its
parser has `--plot` (default off) and NO `--no-plot` — so keeping `--no-plot` would have
raised "unrecognized arguments". `--seed` and `--density-mode current` are valid.

**Run scripts: --volume sources fully migrated, targets/stage intact.** Every `.sh`/
`.sbatch` under `runs/` had its 3-or-2 `--volume` SOURCEs repathed to $HOME/software with
the `:/lorrax_*` targets unchanged. Zero unmigrated scratch sources remain in any
executable script (the only residual `/pscratch/.../lorrax_nvhpc` hits are in historical
`*.out`/`*.log` records). Stage selection is internally consistent per script: Si
`C_vqsph_*` are the OpenMPI variant (`--module=gpu`, `phdf5_openmpi/stage`, `--mpi=pmix`,
`/opt/hpcx/ompi/lib`, 2 volumes, no slate, nvhpc 25.5); VI3 + Si_pseudobands are the
Cray-MPICH variant (3 volumes, slate, gtl LD_PRELOAD). VI3 scripts define
SEL/INC/SITE/VOL/LDP/SHIFTER before use and apply `$SEL $SHIFTER $INC`.

**Installation docs updated to match migration.** `docs/installation/perlmutter.md:43`
references the new `$HOME/software/lorrax_{nvhpc,phdf5_cray/stage,slate_cray/stage}`.

---

## Findings

### SHOULD-FIX 1 — Staging scripts still write to $SCRATCH; consumers now read $HOME/software (migration half-done on the write side)
- **Location:** `src/ffi/phdf5/scripts/stage_cray.sh:31`, `src/ffi/slate/scripts/stage_cray.sh:23`
  (and prose in `src/ffi/slate/README.md:146`). NOT touched by dc5f7f7/ca840ec.
- **Problem:** The migration moved the *read-side* defaults (run_shifter.sh, site_config.sh)
  to `$HOME/software/lorrax_*/stage`, but the *write-side* stagers still default
  `LORRAX_FFI_{PHDF5,SLATE}_DIR` to `/pscratch/sd/${USER:0:1}/${USER}/lorrax_*_cray/stage`.
  A fresh user who follows the docs — run the stagers, then the launcher, without setting
  the override — stages into $SCRATCH but the launcher looks in $HOME, so the phdf5/slate
  bind-mount is silently skipped (`[[ -d ... ]]` false) and the run loses parallel HDF5 /
  SLATE. The migration commit claims it relocated "the 4 FFI dep dirs" but only relocated
  where they are *read from*.
- **Why not a BLOCKER:** the validated workflow's deps already exist at $HOME, and both
  stagers honor the `LORRAX_FFI_*_DIR` env override, so the live session is unaffected.
- **Fix:** change the two `: "${LORRAX_FFI_*_DIR:=/pscratch/...}"` defaults (and the slate
  README line) to `$HOME/software/lorrax_*_cray/stage` to match the consumers.

### NIT 2 — `nanobind` left in the `build` group though it is as unused as the dropped `pybind11`
- **Location:** `pyproject.toml:80` (`build` group).
- **Problem:** ca840ec dropped `pybind11` with rationale "the FFI is plain C". By the same
  evidence (`api.cc`/`ffi_loader.py` both say "no pybind/nanobind"; no `NB_MODULE`/
  `find_package(nanobind)`/`nanobind_add_*` anywhere), `nanobind` is equally unused —
  it survives only in pyproject + two comments that say it is NOT used.
- **Impact:** none at runtime/install; pure dead build dep. Not a regression (it was
  unused before too).
- **Fix:** drop `nanobind>=2.0.0` from the build group, or add a one-line note if it is
  intentionally reserved for a future binding.

### NIT 3 — `run_shifter.sh` empty-LORRAX_SITE default is a behavior change for any future Python use needing site-packages
- **Location:** `src/ffi/common/cpp/run_shifter.sh:103` (`: "${LORRAX_SITE:=}"`), commit ca840ec.
- **Problem:** previously LORRAX_SITE defaulted to the isdf_site path (h5py/scipy/matplotlib
  the NVIDIA container lacks); now empty. The current consumers (FFI smoke tests) don't
  import those, so nothing breaks today (verified). But if run_shifter.sh is later pointed
  at a Python step that needs h5py, it will ImportError unless the caller sets LORRAX_SITE.
- **Impact:** none for the validated FFI-test use case; this is the intended
  de-personalization. Flagged only so the behavior change is on record.
- **Fix (optional):** none required; consider a one-line comment that Python steps needing
  site-packages must export LORRAX_SITE (the comment at 100-102 already points to
  site_config; adequate).

---

## Pre-existing issues surfaced (NOT introduced by these changes)

- **nvhpc subpath split between launchers.** `run_shifter.sh` LD_LIBRARY_PATH uses
  `25.5_cuda12.9` (cuSolverMp 0.6.0, which site_config:81-86 documents as giving *wrong*
  Px>1∧Py>1 answers), while the SKILL/VI3 production path uses `0.7.2_cuda12.9`. Both
  subpaths predate this session (unchanged in the diffs) and both dirs exist; production
  uses 0.7.2 per VALIDATED FACTS. Pre-existing inconsistency, not a migration regression.
- **`SKILL.md:136` claims "JAX 0.7.2"** for the nvcr.io/.../jax:25.04-py3 image. This line
  is in committed HEAD (not in this session's diff) and conflicts with the pyproject 0.5.3
  pin — likely a confusion with cuSolverMp 0.7.2. Out of scope (pre-existing), but worth a
  later correction.

---

## Verdict
No correctness BLOCKER: the migration is byte-clean (targets/LD/LD_PRELOAD untouched), all
relocated deps exist, pyproject/uv.lock are coherent, and the SKILL + run-script $SEL/$INC
wiring is well-formed and matches the validated recipe — the one real gap is that the
staging scripts still default to $SCRATCH while their consumers now read $HOME/software
(SHOULD-FIX), plus two NITs (dead `nanobind`, behavior-change note).
