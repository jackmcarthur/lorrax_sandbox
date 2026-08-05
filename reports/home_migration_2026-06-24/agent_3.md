# Agent 3 — Copy correctness, provenance, NERSC platform gotchas

Lens: *will the relocation actually WORK and keep runs byte-for-byte functional?*
All facts below are from read-only inspection of the live dep dirs, the run scripts,
the NERSC `udiRoot.conf`, and the 2026-05-13 install/maintain reports.

**Headline go/no-go:** Copy-and-repath is sufficient. **No library needs RPATH patching.**
But there is ONE real blocker the orchestrator must resolve first: **HOME is at 97.3 %
quota (1.08 GiB free) and the full footprint is 659 MB** — it *technically* fits but
leaves a dangerous ~0.4 GiB margin. Either prune unused nvhpc versions (saves 141 MB)
or — strongly preferred — target `/global/common/software` instead (no per-user quota,
already in the shifter siteFs). Details in item 5.

---

## Item 1 — Internal symlinks & soname chains

**Finding: every relocation-relevant symlink is RELATIVE; zero symlinks point into
`/pscratch`. A plain `cp -a` (or `cp -rp`) preserves them perfectly.**

Full symlink inventory across all four dirs (`find -type l -printf '%p -> %l'`):

- **openmpi `stage/lib`** (17 symlinks) — all relative soname chains, e.g.
  `libhdf5.so -> libhdf5.so.310.5.1`, `libhdf5.so.310 -> libhdf5.so.310.5.1`,
  `libaec.so.0 -> libaec.so.0.1.3`. Bare names, no leading `/` → survive a move.
- **nvhpc** (16 symlinks) — all relative, e.g.
  `0.7.2_cuda12.9/.../libcusolverMp.so -> libcusolverMp.so.0`,
  `libcusolverMp.so.0 -> libcusolverMp.so.0.7.2.0`. Relative → survive.
- **phdf5_cray `stage`** — the `.so`/`.so.200`/`.so.200.1.1` triples are NOT symlinks
  at all; they are **three independent file copies** (distinct inodes
  `882724765364690799/...800/...801`, identical 440848-byte size, `-links 1`). `cp -a`
  copies all three as-is. Only one symlink in this dir (see item 2).
- **slate_cray `stage`** — one symlink (see item 2).

**Two ABSOLUTE symlinks exist, but both point OUTSIDE the dep tree into the *container*
filesystem, so they are independent of the host location:**
- `phdf5_cray/stage/lib/libmpi_gnu_123.so.12 -> /opt/udiImage/modules/mpich/libmpi.so.12`
  (resolved inside the shifter container; that path is the same regardless of host dir).
- `slate_cray/stage/lib/libreadline.so.7 -> /usr/lib/x86_64-linux-gnu/libreadline.so.8`
  (the readline-7→8 ABI shim that 2026-05-13 `agent_2.md` D9 documented; resolves inside
  the container). Neither targets `/pscratch` or `/global`, so the move does not touch them.

**`find ... -type l -lname '*pscratch*'` returned EMPTY** → nothing dangles after the move.

**No hardlinks** (`find -type f -links +1` empty in every dir), so `cp -a` won't silently
de-duplicate or split anything.

> ⚠️ `cp -rL` (dereference) would be WRONG here: it would turn the two absolute
> container-symlinks into broken copies of host files and explode every relative soname
> chain into duplicate real files. **Use `-a` / `-P`, never `-L`.**

## Item 2 — RPATH / RUNPATH baked into the libs

**Finding: EXHAUSTIVE `readelf -d` scan of every `.so`/`.so.*` in all four dirs shows
ZERO host-absolute RPATH/RUNPATH. Every entry that exists is `$ORIGIN`-relative.**
The libraries are fully relocatable. **No `patchelf` step is required.**

Complete RPATH/RUNPATH inventory (only libs that have one at all):

| Lib | RPATH/RUNPATH |
|---|---|
| `nvhpc/0.7.2_cuda12.9/.../libcusolverMp.so.0.7.2.0` | `$ORIGIN:$ORIGIN/../../nccl/lib:$ORIGIN/../../cuda_runtime/lib:$ORIGIN/../../cublas/lib:$ORIGIN/../../cusolver/lib` |
| `nvhpc/0.7.0/...` & `0.8.0/...libcusolverMp` | same `$ORIGIN`-relative pattern |
| `nvhpc/25.5/.../libnvshmem_host.so.3.2.5`, `nvshmem_bootstrap_uid.so.3.0.0` | `${ORIGIN}` |
| `nvhpc/25.5/.../libcublasmp.so.0.4.0` | RUNPATH `$ORIGIN` |
| `phdf5_openmpi/stage/lib/*` (8 libs: libhdf5, libsz, libaec, hl, cpp, fortran…) | `$ORIGIN/.` |
| `phdf5_cray/stage/lib/*` | **no RPATH at all** (resolves purely via `LD_LIBRARY_PATH`) |
| `slate_cray/stage/lib/*` | **no RPATH at all** |

`$ORIGIN` resolves at load time to the directory containing the `.so`, so it is
*defined by where the file physically sits* — it works identically at
`/global/u2/...` as at `/pscratch/...`. The cross-directory hops in cusolverMp's RPATH
(`$ORIGIN/../../cuda_runtime/lib` etc.) are satisfied by NVIDIA's pip-wheel layout that
sits inside the *JAX container*, not the dep dir — and the dep dir's lib65 layout is
copied intact, so these stay valid.

The phdf5_cray and slate libs have NO RPATH, so they rely entirely on `LD_LIBRARY_PATH`
(which Agent 1 owns). **LD_LIBRARY_PATH override IS sufficient for those.** For the
nvhpc/openmpi libs that DO have `$ORIGIN` RPATH, no override is even needed — but note
that an `$ORIGIN` RPATH is searched *before* `LD_LIBRARY_PATH` for non-RUNPATH (RPATH)
entries, which is fine because `$ORIGIN` self-resolves correctly after the move.

**Conclusion: GO. No lib needs RPATH patching. The move is RPATH-transparent.**

## Item 3 — Shifter `--volume` source constraints (the load-bearing question)

This is where I corrected a premise. The 2026-05-13 reports say the stage scripts exist
because *"Shifter forbids `--volume` from `/opt/cray`"* (`agent_2.md` D8 line 194-200,
`round2_agent_1.md` A9). The restriction is on the **system OS source `/opt/cray`**, NOT
a blanket "only `/pscratch` is an allowed `--volume` source" rule. `$SCRATCH` was merely
the convenient *destination* for libs extracted out of `/opt/cray`.

**Authoritative evidence — NERSC `/etc/shifter/udiRoot.conf` `siteFs` (lines 221-231)**
auto-bind-mounts these host trees into every container:
```
/dvs_ro/cfs, /global/cfs, /global/common, /global/dna,
/global/u1, /global/u2, /pscratch
```
So `/global/u2` (and `/global/common`) are first-class, container-visible host trees,
exactly like `/pscratch`. A `--volume=<src>:<dst>` whose `src` is under `/global/u2`
or `/global/common` is therefore allowed, the same as `/pscratch`.

**The `$HOME` non-expansion gotcha — now fully explained:**
`$HOME = /global/homes/j/jackm`, and `readlink -f` shows it resolves to
`/global/u2/j/jackm`. The siteFs list contains **`/global/u2`** (the real path), NOT
`/global/homes` (a symlink that does not exist inside the container namespace). That is
*why* `execute_workflow/SKILL.md:149` says "use `/global/u2/j/jackm/...` paths (Shifter
may not expand `$HOME`)" — an *in-container* path must be `/global/u2/...`.

For the `--volume` SOURCE specifically: the host resolves the symlink before the mount,
so both `/global/homes/j/jackm/...` and `/global/u2/j/jackm/...` work as the source
(left side of `SRC:DST`). **The existing run scripts already prove `$HOME` works on
compute nodes** — e.g. `runs/Si/C_vqsph_si10_ref/run.sh:15,21` uses
`PYTHONPATH=/global/homes/j/jackm/software/lorrax_C/src` and
`git -C /global/homes/j/jackm/software/lorrax_C`, and
`runs/CrI3/.../run_cri3_now.sh:14` puts
`LD_LIBRARY_PATH=/global/homes/j/jackm/software/slate/install/lib64:...` inside the
container *with no explicit `--volume`* — i.e. read through the auto-mounted home tree.

**Which literal form do existing scripts use?** Two forms coexist:
- run scripts (`runs/**/run*.sh`): `/global/homes/j/jackm/...`
- `execute_workflow/SKILL.md:144,149`: `/global/u2/j/jackm/...`

**Recommendation for the new `--volume` lines: use the `/global/u2/j/jackm/...` form**
(matches what the SKILL already declares correct, and is robust because it is the literal
siteFs entry). The `/global/homes` form also works for the *source* side but is one
symlink-resolution away from the documented-safe path; do not use `/global/homes` for any
*in-container destination* or env path.

## Item 4 — Provenance / rebuild path

Source of truth: `reports/lorrax_install_maintain_blitz_2026-05-13/`. The four dep dirs
are produced by `src/ffi/*/scripts/stage_*.sh` in the LORRAX source tree
(`stage_pypi.sh` → nvhpc from a pip wheel; `phdf5/scripts/stage_cray.sh` →
HDF5-vs-Cray-MPICH; `slate/scripts/stage_cray.sh` → Cray GTL/libsci;
`stage_openmpi.sh` → HDF5-vs-OpenMPI). They are *staging copies of vendor libraries*,
not from-scratch builds of project code (`agent_2.md` D8: "copy 80 MB of vendor libs
into `$SCRATCH`"). The cray-phdf5 stage even bakes in a `libreadline.so.7→.8` ABI shim
and the `libmpi_gnu_123.so.12 -> /opt/udiImage/.../libmpi.so.12` container symlink
(`agent_2.md` D9) — artifacts a naive rebuild on a different Cray-driver release might
reproduce *differently*.

**Verdict: copy-in-place is the correct and safest option, not rebuild.**
Re-running the stage scripts would re-derive the libs from whatever vendor/driver state
exists *today*, which risks soname/ABI drift vs. the binaries the current CrI3/VI3/Si
runs were validated against. A `cp -a` is byte-for-byte; the stage scripts are not
guaranteed to be. The goal here is "keep runs byte-for-byte functional," so copy wins.
(The stage scripts can still be re-pointed to emit into `$HOME/software` *for future*
re-stages — a doc change, not part of this migration.)

**Partial-migration note:** `/global/homes/j/jackm/software/` ALREADY exists and is the
de-facto staging home (`slate/`, `lorrax_A/B/C`, `BerkeleyGW`, `qe-7.4.1`, etc.). There
is even a `software/lorrax_phdf5_openmpi/` — but it contains only `cache/{hdf5,libaec}.conda`
(the conda *build inputs*), NOT the runtime `stage/`. So **none of the four runtime trees
are in HOME yet**; the copy still has to happen for all four.

## Item 5 — Filesystem availability & quota

**Compute-node visibility: CONFIRMED.** Per `udiRoot.conf` siteFs (item 3), `/global/u2`
(= `$HOME`) and `/global/common` are bind-mounted into every shifter container on every
node, including compute. Empirically confirmed: `runs/Si/C_vqsph_si10_ref/run.sh`
already reads `software/lorrax_C` and `software/slate/install/lib64` from HOME inside a
4-GPU `srun ... shifter` job. A HOME-resident dep dir IS visible to compute nodes. ✅

**Quota — THE RISK.** `myquota` right now:
```
home   38.92GiB / 40.00GiB  (97.3%)   →  only ~1.08 GiB free
```
Footprint to copy: nvhpc 422M + phdf5_cray 169M + slate 56M + phdf5_openmpi 12M = **659 MB**.
Copying all four → HOME ~39.58 GiB, ~0.4 GiB headroom. That is *technically* under quota
but perilously tight — HOME also absorbs git objects, `.cache`, conda, etc., and hitting
40 GiB will start failing unrelated writes. **CONTEXT.md's "fits easily in the 40 GB
quota" is inaccurate as of today — the relevant number is free space (1.08 GiB), not
total.**

Mitigations, in order of preference:
1. **Target `/global/common/software` instead of `$HOME/software`.** It is in the
   shifter siteFs (`/global/common` line 227), has **no per-user 40 GiB quota**, and is
   the standard NERSC location for project-shared software. This is the cleanest fix and
   sidesteps the quota entirely. (Requires write access to a project subdir under
   `/global/common/software/<project>`; the user has m-allocations — worth checking which.)
2. **If staying in `$HOME`: copy only the referenced nvhpc versions.** Only `0.7.2`
   (110M) and `25.5` (172M) are referenced by current run scripts
   (`grep 'lorrax_nvhpc/[0-9]'` over `runs/**/*.sh`). `0.7.0` (64M) and `0.8.0` (77M) are
   **unused** → skip them, saving **141 MB**. New footprint ≈ 518 MB, leaving ~0.56 GiB
   headroom. Still tight but workable.
3. Free unrelated HOME space first.

`/global/cfs` (CFS, large quota) is also siteFs-mounted and is another viable home for a
660 MB dep tree if neither of the above is clean.

---

## EXACT recommended copy commands

Assuming target `DEST=/global/u2/j/jackm/software` (substitute
`/global/common/software/<project>` if option 5.1 is taken). `cp -a` = archive:
preserves symlinks-as-symlinks, perms, timestamps; recurses; never dereferences.

```bash
DEST=/global/u2/j/jackm/software
mkdir -p "$DEST"

# nvhpc — copy the WHOLE versioned tree (mounts the dir; run scripts pick the subpath).
# To respect quota, copy only referenced versions instead of the whole 422M dir:
mkdir -p "$DEST/lorrax_nvhpc"
cp -a /pscratch/sd/j/jackm/lorrax_nvhpc/0.7.2_cuda12.9 "$DEST/lorrax_nvhpc/"
cp -a /pscratch/sd/j/jackm/lorrax_nvhpc/25.5_cuda12.9  "$DEST/lorrax_nvhpc/"
#   (omit 0.7.0_cuda12.9 and 0.8.0_cuda12.9 — unused, saves 141 MB.
#    If quota is not a concern, just: cp -a /pscratch/sd/j/jackm/lorrax_nvhpc "$DEST/")

# phdf5 (cray) — the run scripts mount the .../stage subdir, so preserve the
# lorrax_phdf5_cray/stage layout. Do NOT copy bench_read16k.h5 (0-byte setuid leftover).
mkdir -p "$DEST/lorrax_phdf5_cray"
cp -a /pscratch/sd/j/jackm/lorrax_phdf5_cray/stage "$DEST/lorrax_phdf5_cray/"

# slate (cray)
mkdir -p "$DEST/lorrax_slate_cray"
cp -a /pscratch/sd/j/jackm/lorrax_slate_cray/stage "$DEST/lorrax_slate_cray/"

# phdf5 (openmpi)
mkdir -p "$DEST/lorrax_phdf5_openmpi"
cp -a /pscratch/sd/j/jackm/lorrax_phdf5_openmpi/stage "$DEST/lorrax_phdf5_openmpi/"
```

Post-copy verification (must all pass before repointing run scripts):
```bash
# 1. No dangling symlinks introduced by the move:
find "$DEST"/lorrax_* -xtype l        # expect EMPTY
# 2. soname chains intact (spot-check):
readlink "$DEST/lorrax_phdf5_openmpi/stage/lib/libhdf5.so"   # -> libhdf5.so.310.5.1
# 3. RPATHs still $ORIGIN-relative (unchanged by cp -a, but confirm):
readelf -d "$DEST/lorrax_nvhpc/0.7.2_cuda12.9/math_libs/12.9/lib64/libcusolverMp.so.0.7.2.0" | grep -i rpath
# 4. byte-identical (paranoia, on a couple of libs):
cmp /pscratch/sd/j/jackm/lorrax_slate_cray/stage/lib/libmpi_gtl_cuda.so.0 \
    "$DEST/lorrax_slate_cray/stage/lib/libmpi_gtl_cuda.so.0"
```

The two ABSOLUTE container-symlinks (`libmpi_gnu_123.so.12`, `libreadline.so.7`) will show
as dangling on the *login node* (their `/opt/udiImage` / `/usr/lib/x86_64-linux-gnu`
targets only exist inside the container) — that is EXPECTED and identical to their state
in the current `/pscratch` copy. Don't "fix" them; check `find -xtype l` *relative to the
move*, i.e. compare the dangling set is unchanged from the source (it is: exactly those two).

**`--volume` line edits Agent 1 will need (form recommendation):** change
`--volume=/pscratch/sd/j/jackm/lorrax_nvhpc:/lorrax_nvhpc` →
`--volume=/global/u2/j/jackm/software/lorrax_nvhpc:/lorrax_nvhpc` (and the three siblings).
Container-side mountpoints (`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate`) and the
in-container `LD_LIBRARY_PATH` subpaths stay EXACTLY the same — so no RPATH/soname concern
flows downstream. (Path-edit enumeration is Agents 1 & 2's contract; I only assert the
target-path *form*.)

---

## Open questions / risks

1. **[BLOCKER-CLASS] HOME quota.** 1.08 GiB free vs 659 MB footprint. Resolve *before*
   copying: prefer `/global/common/software/<m-project>` (no quota, siteFs-mounted), else
   prune unused nvhpc `0.7.0`/`0.8.0` (saves 141 MB). Needs a human decision +
   confirmation of which `/global/common/software/<project>` dir is writable.
2. **siteFs config is host-state, not repo-state.** I read the *current* login-node
   `/etc/shifter/udiRoot.conf`. If NERSC changes the siteFs list (rare), `/global/u2` /
   `/global/common` mounting could change. Low risk, but it's the one fact that isn't
   pinned in the repo. The empirical proof (existing runs reading HOME on compute nodes)
   is the stronger guarantee.
3. **`/global/homes` vs `/global/u2` form discipline.** Mixed usage exists today (run
   scripts use `/global/homes`, the SKILL uses `/global/u2`). For `--volume` *sources*
   both work; for any in-container path only `/global/u2` is safe. If the orchestrator
   standardizes, pick `/global/u2/j/jackm` everywhere to match the SKILL and the literal
   siteFs entry.
4. **Performance.** `$HOME`/CFS is GPFS-class, not the Lustre `$SCRATCH` these libs sit on
   now. For ~660 MB of `.so` files `dlopen`'d once at startup this is negligible (and they
   get page-cached), but worth a one-line note since the *reason* they were on `$SCRATCH`
   may partly have been I/O. No striping needed for a read-only lib dir.
5. **The 0-byte `bench_read16k.h5` setuid/setgid/sticky leftover** at
   `lorrax_phdf5_cray/bench_read16k.h5` (perms `-rwsr-sr-t`) sits *beside* `stage/`, not
   in it. The recommended `cp -a .../stage` excludes it correctly — do not sweep the whole
   `lorrax_phdf5_cray/` dir or you'll drag a setuid artifact into HOME.
6. **Verification depth.** I confirmed RPATH/symlink/inode structure but did NOT run an
   actual shifter job from the new path (read-only mandate; no allocation spun up). The
   post-copy `find -xtype l` + `cmp` checks above are the substitute; a single smoke run
   of one CrI3/VI3 script with the repointed `--volume` is the real go/no-go gate.
