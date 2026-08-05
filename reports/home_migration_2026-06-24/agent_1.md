# Agent 1 — Run scripts & shell/env linkages

Lens: every run script / launcher / env var / config default in the sandbox that
references one of the four scratch dep dirs (`lorrax_nvhpc`, `lorrax_phdf5_cray`,
`lorrax_phdf5_openmpi`, `lorrax_slate_cray`). Read-only sweep; only `--volume`
SOURCE paths (left of the colon) change. Docs (`.md`) are Agent 2's; profile dumps
(`hlo_summary.*`, `xla_dump/`, `*.mlir`) excluded.

Proposed target root: `/global/homes/j/jackm/software/` (literal — Shifter does NOT
expand `$HOME`). So the four SOURCE rewrites are:

| current SOURCE (scratch) | proposed SOURCE ($HOME) |
|---|---|
| `/pscratch/sd/j/jackm/lorrax_nvhpc` | `/global/homes/j/jackm/software/lorrax_nvhpc` |
| `/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage` | `/global/homes/j/jackm/software/lorrax_phdf5_cray/stage` |
| `/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage` | `/global/homes/j/jackm/software/lorrax_phdf5_openmpi/stage` |
| `/pscratch/sd/j/jackm/lorrax_slate_cray/stage` | `/global/homes/j/jackm/software/lorrax_slate_cray/stage` |

---

## 0. THE KEY STRUCTURAL FACT — only the SOURCE changes

Every script bind-mounts the dep dirs at **container-stable mount targets** that are
uniform across the whole sandbox:

| host SOURCE (changes) | container MOUNT target (UNCHANGED) |
|---|---|
| `…/lorrax_nvhpc` | `/lorrax_nvhpc` |
| `…/lorrax_phdf5_{cray,openmpi}/stage` | `/lorrax_phdf5` |
| `…/lorrax_slate_cray/stage` | `/lorrax_slate` |

**Consequence:** `LD_LIBRARY_PATH`, `LD_PRELOAD`, `LORRAX_MPI_INCLUDE_DIR`,
`LORRAX_MPICH_LIB_DIR` all reference the in-container paths (`/lorrax_slate/lib`,
`/lorrax_phdf5/lib`, `/lorrax_phdf5/include`, `/lorrax_nvhpc/<ver>/math_libs/...`,
`/lorrax_slate/lib/libmpi_gtl_cuda.so.0`). **None of these need to change** — they
are container-internal and independent of where the host tree lives. I verified this
holds in *every* script below. The **only** edits required are the `--volume=SOURCE:`
left-hand sides (and the equivalent `*_HOST`/`*_DEFAULT` shell vars in the FFI
infra). I found **no exception** to this rule anywhere.

The `LD_LIBRARY_PATH` lines also contain `/global/homes/j/jackm/software/slate/install/lib64`
— that is the host SLATE install, already on `$HOME`, NOT one of the four dep dirs.
Leave it. `/global/common/software/nersc9/darshan/default/lib` is a NERSC system path
— leave it.

---

## 1. Run scripts (`runs/**`) — the live runs

Grouped by run. "current" = the SOURCE substring(s) on that line; mount target stays.
Every `--volume` SOURCE on these lines gets the §0 rewrite. All paths absolute,
relative to repo root `/pscratch/sd/j/jackm/lorrax_sandbox/`.

### CrI3 — `runs/CrI3/04_gw_6x6_600b_2026-06-17/`
| file:line | current SOURCE(s) | nvhpc ver in LD_LIBRARY_PATH |
|---|---|---|
| `run_cri3_now.sh:13` | `…/lorrax_nvhpc`, `…/lorrax_phdf5_cray/stage`, `…/lorrax_slate_cray/stage` (VOL=) | `0.7.2_cuda12.9` (line 14, no edit) |
| `run_cri3_gw.sbatch:22` | same three (VOL=) | `0.7.2_cuda12.9` (line 23, no edit) |

### CrI3 sweep — `runs/CrI3/M_6x6_80Ry_2026-05-07/sweep_B_workdir_2026-05-16/`
| file:line | current SOURCE(s) |
|---|---|
| `run_one.sh:76` | `--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc` |
| `run_one.sh:77` | `--volume=/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage:/lorrax_phdf5` |
| `run_one.sh:78` | `--volume=/pscratch/sd/j/jackm/lorrax_slate_cray/stage:/lorrax_slate` |
| — lines 56/57/60 are in-container (`/lorrax_*`) — NO edit. nvhpc ver `0.7.2_cuda12.9` (line 56). |

### VI3 — `runs/VI3/04_gw_6x6_600b_2026-06-17/` (six scripts, identical VOL= pattern)
| file:line | current SOURCE(s) | nvhpc ver |
|---|---|---|
| `run_vi3_gwonly.sh:9`  | three (VOL=) | `0.7.2_cuda12.9` (L10) |
| `run_vi3_lorrax.sh:10` | three (VOL=) | `0.7.2_cuda12.9` (L11) |
| `run_vi3_orbmag.sh:10` | three (VOL=) | `0.7.2_cuda12.9` (L11) |
| `run_vi3_cont.sh:9`    | three (VOL=) | `0.7.2_cuda12.9` (L10) |
| `run_vi3_now.sh:11`    | three (VOL=) | `0.7.2_cuda12.9` (L12) |
| `run_vi3_gw.sbatch:22` | three (VOL=) | `0.7.2_cuda12.9` (L23) |
("three" = `lorrax_nvhpc` + `lorrax_phdf5_cray/stage` + `lorrax_slate_cray/stage`,
all on one `VOL="…"` line.)

### Si (`C_vqsph_*`, OpenMPI stack) — `runs/Si/`
| file:line | current SOURCE(s) | nvhpc ver |
|---|---|---|
| `C_vqsph_si10_ref/run.sh:13`    | `--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc` | — |
| `C_vqsph_si10_ref/run.sh:14`    | `--volume=/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage:/lorrax_phdf5` | — |
| `C_vqsph_si10_ref/run.sh:19`    | (LD_LIBRARY_PATH, in-container only) | `25.5_cuda12.9` — NO edit |
| `C_vqsph_si10_sphere/run.sh:13` | `…/lorrax_nvhpc` | — |
| `C_vqsph_si10_sphere/run.sh:14` | `…/lorrax_phdf5_openmpi/stage` | — |
| `C_vqsph_si10_sphere/run.sh:19` | (LD_LIBRARY_PATH) | `25.5_cuda12.9` — NO edit |
NB: Si uses `lorrax_phdf5_openmpi` (not `_cray`) and NO slate mount. PYTHONPATH here
points at `/global/homes/j/jackm/software/lorrax_C/src` — already on `$HOME`, not a
dep dir, leave it.

### Si_pseudobands — `runs/Si_pseudobands/00_si_2x2x2_60Ry/`
| file:line | current SOURCE(s) | nvhpc ver |
|---|---|---|
| `run_sweep_w10.sh:2` | three on one `SHIFTER='…'` line (`lorrax_nvhpc`,`lorrax_phdf5_cray/stage`,`lorrax_slate_cray/stage`) | `25.5_cuda12.9` (same line) — NO edit |
| `D_lorrax_canonical_gnppm_test/run.sh:3` | same three on one `SHIFTER='…'` line | `25.5_cuda12.9` (same line) — NO edit |

**Run-script SOURCE-path edit count: 13 files.** (CrI3 ×2, CrI3-sweep ×1, VI3 ×6,
Si ×2, Si_pseudobands ×2.)

---

## 2. nvhpc VERSION subdirs that must be copied

`ls /pscratch/sd/j/jackm/lorrax_nvhpc/` shows **all four exist on disk**:
`0.7.0_cuda12.9`, `0.7.2_cuda12.9`, `0.8.0_cuda12.9`, `25.5_cuda12.9`.

Versions actually pinned by the LD_LIBRARY_PATH of a run script:
- **`0.7.2_cuda12.9`** — all CrI3 + all VI3 + the CrI3 sweep_B (current June production).
- **`25.5_cuda12.9`** — both Si `C_vqsph_*` runs + both Si_pseudobands launchers.
- `0.7.0_cuda12.9`, `0.8.0_cuda12.9` — NOT referenced by any run script's
  LD_LIBRARY_PATH, but they ARE referenced by name in the cuSolverMp LU-bug test
  logs (`runs/cache_audit_shared/lu_bug_test/v0{70,80}_*.log`) and discussed in the
  FFI `site_config.sh` comments. They are the version-comparison fallbacks.

**Recommendation: copy the WHOLE `lorrax_nvhpc/` tree (all four subdirs, 422 MB).**
A selective copy of only `0.7.2`+`25.5` would break any future shell that sets
`LORRAX_NVHPC_SUBPATH=0.7.0…/0.8.0…` for the LU-bug regression check. The total is
~660 MB — well under the 40 GB `$HOME` quota — so there is no reason to prune.

---

## 3. FFI build/staging infra (`sources/**`) — the second tier

These are NOT run scripts but they (a) set the **default `--volume` SOURCE** for the
`run_shifter.sh` build/smoke-test driver, and (b) the live `select_gpu.sh` /
`in_container.sh` that the run scripts invoke live under
`sources/lorrax_D/src/ffi/common/cpp/`. Each `lorrax_*` source worktree carries its
own copy: **`lorrax_D` (live), `lorrax_D_old`, `lorrax_B_orbmag_wt`, and three
`blitz_workspaces/{blitz4-distinit,blitz5,blitz6}`.** Two flavors of default:

**(a) `$SCRATCH`-parameterized** (in `config/perlmutter/site_config.sh`, every copy):
```
LORRAX_FFI_NVHPC_DIR_DEFAULT="$SCRATCH/lorrax_nvhpc"
LORRAX_FFI_PHDF5_DIR_DEFAULT="$SCRATCH/lorrax_phdf5_cray/stage"
LORRAX_FFI_SLATE_DIR_DEFAULT="$SCRATCH/lorrax_slate_cray/stage"
```
On Perlmutter `$SCRATCH` = `/pscratch/sd/j/jackm`, so these resolve to scratch and
**also need rewriting** to `/global/homes/j/jackm/software/…` (or, cleaner, introduce
a `LORRAX_FFI_ROOT_DEFAULT="$HOME/software"` and build the three from it). Files:
`sources/{lorrax_D,lorrax_D_old,lorrax_B_orbmag_wt,blitz_workspaces/blitz4-distinit,blitz_workspaces/blitz5,blitz_workspaces/blitz6}/config/perlmutter/site_config.sh`
lines ~100–102 (D_old: 85–87). Note site_config also already has
`LORRAX_SLATE_INSTALL_DIR_DEFAULT="$HOME/software/slate/install"` — the new dep root
should match that `$HOME/software` convention.

**(b) Hard-coded literal `/pscratch/sd/j/jackm/…` fallbacks** (in
`src/ffi/common/cpp/run_shifter.sh`, the `lorrax_D` / `lorrax_D_old` /
`lorrax_B_orbmag_wt` copies):
| file:line | current |
|---|---|
| `…/run_shifter.sh:44` | `NVHPC_HOST="${LORRAX_FFI_NVHPC_DIR:-/pscratch/sd/j/jackm/lorrax_nvhpc}"` |
| `…/run_shifter.sh:54` | `PHDF5_DEFAULT="/pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage"` |
| `…/run_shifter.sh:61` | `PHDF5_DEFAULT="/pscratch/sd/j/jackm/lorrax_phdf5_cray/stage"` |
| `…/run_shifter.sh:107` | `: "${LORRAX_FFI_SLATE_DIR:=/pscratch/sd/j/jackm/lorrax_slate_cray/stage}"` |
(The blitz4/5/6 copies use `${USER}`-parameterized `/pscratch/sd/${USER}/…` forms at
lines ~59–79/109/133 — same rewrite, but those are older snapshots; only `lorrax_D`
is on the live run path.)

The `stage_*.sh` staging scripts (`stage_nvhpc.sh`, `stage_pypi.sh`,
`stage_cray.sh`, `stage_openmpi.sh`) default `LORRAX_FFI_*_DIR` to
`/pscratch/sd/${USER:0:1}/${USER}/lorrax_*` — these WRITE the staged trees. If you
re-stage in future you'd want them pointing at the new root too, but for a one-time
`cp -a` migration they're not on the critical path. Flag, don't block.

**Priority:** Only the **`lorrax_D`** copy (live `run_shifter.sh` + `site_config.sh`)
and the run scripts in §1 are load-bearing for current runs. The other source
worktrees are historical; update for consistency but they won't break June runs.

---

## 4. `scratchperl` symlink form

**No run script references any of the four dep dirs via the
`/global/homes/j/jackm/scratchperl/...` symlink.** The only `scratchperl` use in run
scripts is `SITE=/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site` (the
ISDF site-packages on PYTHONPATH) — that is a *different* directory, not one of our
four, and is out of scope. So there is no hidden symlink-form reference to chase for
the dep-dir migration. (Confirmed: `grep 'scratchperl/lorrax_\(nvhpc\|phdf5\|slate\)'`
over `runs/` returns nothing.)

---

## 5. Ready-to-run sweep proposal

**Step 0 — copy the trees (preserve perms/timestamps):**
```bash
mkdir -p /global/homes/j/jackm/software
cp -a /pscratch/sd/j/jackm/lorrax_nvhpc          /global/homes/j/jackm/software/
cp -a /pscratch/sd/j/jackm/lorrax_phdf5_cray     /global/homes/j/jackm/software/
cp -a /pscratch/sd/j/jackm/lorrax_phdf5_openmpi  /global/homes/j/jackm/software/
cp -a /pscratch/sd/j/jackm/lorrax_slate_cray     /global/homes/j/jackm/software/
```
(Copies the `/stage` subdirs and all four nvhpc version dirs intact. ~660 MB.
A plain `cp -a` reproduces them — these stages are flat copies of libs/headers, no
build step needed at the destination; see
`reports/lorrax_install_maintain_blitz_2026-05-13/` for how they were originally
staged.)

**Step 1 — rewrite the run-script SOURCE paths (the common pattern):**
The string `/pscratch/sd/j/jackm/lorrax_` immediately followed by one of
`nvhpc`/`phdf5_cray`/`phdf5_openmpi`/`slate_cray` is, in EVERY run-script hit, a
`--volume` SOURCE — never anything else. So a single anchored sed is safe over the
run scripts:
```bash
cd /pscratch/sd/j/jackm/lorrax_sandbox
FILES=$(grep -rl -E '/pscratch/sd/j/jackm/lorrax_(nvhpc|phdf5_cray|phdf5_openmpi|slate_cray)' \
          --include='*.sh' --include='*.sbatch' runs/)
for f in $FILES; do
  sed -i -E 's#/pscratch/sd/j/jackm/(lorrax_(nvhpc|phdf5_cray|phdf5_openmpi|slate_cray))#/global/homes/j/jackm/software/\1#g' "$f"
done
```
This is correct because (i) the in-container `/lorrax_nvhpc`,`/lorrax_phdf5`,
`/lorrax_slate` mount targets do NOT contain the `/pscratch/...` prefix, so they are
untouched; (ii) the only other `/pscratch/sd/j/jackm/` occurrences in these scripts
are `lorrax_sandbox/...` (PYTHONPATH/RUN/LROOT) and `.jax_cache` — neither matches the
`lorrax_(nvhpc|phdf5_cray|phdf5_openmpi|slate_cray)` alternation, so they're safe.
**Verify after:** `grep -rn '/pscratch/sd/j/jackm/lorrax_\(nvhpc\|phdf5\|slate\)' runs/`
should return nothing.

**Step 2 — FFI infra (optional, for consistency):** same sed over
`sources/lorrax_D/{config/perlmutter/site_config.sh,src/ffi/common/cpp/run_shifter.sh}`
PLUS hand-edit the two `$SCRATCH/...` lines in site_config.sh (sed won't catch
`$SCRATCH` — see below). If you want the other worktrees consistent, widen `FILES`
to `sources/` but exclude `.venv`.

### Scripts that DON'T fit the common sed (hand-edit)
1. **`sources/**/config/perlmutter/site_config.sh`** — uses `"$SCRATCH/lorrax_nvhpc"`
   etc., literal `$SCRATCH` not `/pscratch/...`. The anchored sed misses these. Edit
   by hand (or change `$SCRATCH` → `$HOME/software`). 6 copies (D, D_old at L85–87,
   B_orbmag_wt, blitz4/5/6).
2. **`sources/blitz_workspaces/{4,5,6}/**/{run_shifter,stage_*}.sh`** — use
   `${USER:0:1}/${USER}` and `${USER}` parameterized prefixes
   (`/pscratch/sd/${USER}/lorrax_…`), not the literal `j/jackm`. The §5-step1 sed is
   keyed to `j/jackm` and won't touch them. Either widen the sed regex to
   `/pscratch/sd/[^/]+/[^/]+/lorrax_(…)` or hand-edit. (Low priority — historical.)
3. **`stage_*.sh` staging scripts** (all worktrees) — these are the re-stage entry
   points, not run-path; rewrite only if you plan to re-stage rather than `cp -a`.

---

## 6. Open questions / risks

1. **`/global/common/software/<project>` alternative.** CONTEXT floats this if an
   allocation exists. The sbatch headers show account **`m2651`**; if `m2651` has a
   `/global/common/software/m2651` dir it would be the more robust home (shared,
   never purged, read-fast on compute nodes). If chosen, the rewrite target in every
   table above changes accordingly — but the *set of file:line edits is identical*.
   Worth one `ls /global/common/software/m2651` before committing to `$HOME/software`.
2. **`$HOME` read performance from compute nodes.** `$HOME` (GPFS) is fine for the
   ~660 MB of `.so` libs loaded once at container start, but it is metadata-slow under
   many concurrent stat()s. With 16 ranks each dlopen-ing the same libs at job start
   this should be negligible; flag only if a 4-node job shows slow startup.
3. **Shifter `--volume` and `$HOME`/symlinks.** Confirmed Shifter won't expand `$HOME`,
   which is why every script must use the literal `/global/homes/j/jackm/software/...`.
   Also verify Shifter accepts a `--volume` SOURCE on GPFS `$HOME` (it bind-mounts the
   *resolved* host path; should be fine, but the original choice of scratch may have
   been deliberate — worth a 1-rank smoke test before trusting a 4-node production run).
   Note the existing `--volume=/global/homes/j/jackm/software/slate/install/...`
   equivalent works today (slate install is already `$HOME`), which is strong evidence
   `$HOME` bind-mounts are fine.
4. **Doc vs reality divergence (CONTEXT wrinkle).** Confirmed from my side: the run
   scripts use the heavy 3-mount shifter prefix; `skills/execute_workflow/SKILL.md`
   lines 143–146 use a LIGHT prefix with NO dep-dir mounts at all
   (`PYTHONPATH=/global/u2/j/jackm/software/lorrax/src` only). So the SKILL surface
   has **nothing to migrate** for these four dirs — but the divergence means the doc
   doesn't actually describe how production GW runs (which need cuSolverMp/phdf5/slate)
   are launched. That's Agent 2's call; flagging it confirms the dep-dir edits are
   confined to `runs/**` + `sources/**` FFI infra.
5. **Copy fidelity of the `stage/` dirs.** The phdf5/slate `stage/` trees may contain
   symlinks (e.g. `.so` → `.so.0`). Use `cp -a` (preserves symlinks/perms) not
   `cp -rL`; a `cp -rL` would dereference and bloat/break them. Verify with
   `find <dest> -type l` after copy that the symlink count matches the source.
6. **The `0.7.0`/`0.8.0` nvhpc subdirs** are referenced only in LU-bug test logs and
   site_config comments, never in a live LD_LIBRARY_PATH — but copy them anyway (see
   §2). If someone selectively copies and later runs the LU-bug regression with
   `LORRAX_NVHPC_SUBPATH=0.8.0_cuda12.9/...`, a pruned copy breaks it silently.
