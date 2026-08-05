# Consolidated migration checklist — relocate lorrax runtime deps off $SCRATCH

Synthesis of `agent_1.md` (run scripts), `agent_2.md` (docs/skills), `agent_3.md`
(copy correctness / platform). Orchestrator-reconciled 2026-06-24.

## Two corrections to the original premise (READ FIRST)

1. **`$HOME` is NOT a viable target — it's 97.3% full (~1.08 GiB free).** The 659 MB
   would fit by the quota number but leaves home dangerously full, and `cp -a` of the
   nvhpc tree alone is 422 MB. **Recommend `/global/common/software/m2651`** (the
   account is `m2651`, per the `salloc` line in `execute_workflow/SKILL.md`): no quota,
   purge-exempt, and auto-mounted into every shifter container via siteFs. This is also
   the canonical NERSC location for staged software. (Agent 3.)

2. **Use the `/global/u2/j/jackm/...` or `/global/common/...` path form, NOT
   `/global/homes/...`.** Shifter's siteFs auto-mounts `/global/u2`, `/global/common`,
   and `/pscratch` — but NOT `/global/homes` (which resolves to `/global/u2` anyway).
   A `--volume=/global/homes/...` source would fail to mount. (Agent 3, corrects the
   CONTEXT note about "$HOME not expanding".)

Net: target `/global/common/software/m2651/lorrax_{nvhpc,phdf5_cray,phdf5_openmpi,slate_cray}`.

## The copy itself — copy-only, no rebuild, no RPATH patching (Agent 3)

All four trees are vendor-lib staging copies. Every `.so→.so.N` chain is a RELATIVE
symlink; zero symlinks or RPATHs point into `/pscratch`; every ELF RPATH is
`$ORIGIN`-relative or absent. So a straight `cp -a` reproduces them byte-faithfully
and they remain dynamically resolvable. **Use `cp -a` (preserves symlinks); never
`cp -rL` (would explode the soname chains).**

```bash
DEST=/global/common/software/m2651        # confirm you can write here first
cp -a /pscratch/sd/j/jackm/lorrax_nvhpc          $DEST/
cp -a /pscratch/sd/j/jackm/lorrax_phdf5_cray     $DEST/
cp -a /pscratch/sd/j/jackm/lorrax_phdf5_openmpi  $DEST/
cp -a /pscratch/sd/j/jackm/lorrax_slate_cray     $DEST/
# verify no dangling symlinks after copy:
find $DEST/lorrax_{nvhpc,phdf5_cray,phdf5_openmpi,slate_cray} -xtype l
```
Optional prune: only `lorrax_nvhpc/0.7.2_cuda12.9` and `25.5_cuda12.9` are referenced
by run scripts; `0.7.0`/`0.8.0` (141 MB) are unused EXCEPT a LU-bug regression — keep
them unless space-constrained. (Agents 1 & 3.) Skip the 0-byte setuid
`lorrax_phdf5_cray/bench_read16k.h5` leftover — don't carry it over.

## Surface 1 — run scripts (13 files, Agent 1). ONLY the --volume SOURCE changes.

Container mount targets (`/lorrax_nvhpc`, `/lorrax_phdf5`, `/lorrax_slate`) are stable,
so `LD_LIBRARY_PATH`, `LD_PRELOAD`, `LORRAX_MPI_INCLUDE_DIR`, `LORRAX_MPICH_LIB_DIR`
need **no edit** (verified, zero exceptions). Only the left-of-colon source path moves.

Files (all under `runs/`): CrI3 `run_cri3_now.sh`, `run_cri3_gw.sbatch`;
CrI3 sweep `M_6x6_80Ry_2026-05-07/sweep_B_workdir_2026-05-16/run_one.sh`;
VI3 `04_gw_6x6_600b_2026-06-17/run_vi3_{gwonly,orbmag,lorrax,cont,now}.sh` +
`run_vi3_gw.sbatch`; Si `C_vqsph_si10_{ref,sphere}/run.sh`;
Si_pseudobands `00_si_2x2x2_60Ry/run_sweep_w10.sh` +
`D_lorrax_canonical_gnppm_test/run.sh`. (Exact file:line in `agent_1.md`.)

Anchored sweep (does NOT touch PYTHONPATH/.jax_cache/mount-targets):
```bash
grep -rlZ -E '/pscratch/sd/j/jackm/lorrax_(nvhpc|phdf5_cray|phdf5_openmpi|slate_cray)' \
  runs/ | xargs -0 sed -i -E \
  's#/pscratch/sd/j/jackm/(lorrax_(nvhpc|phdf5_cray|phdf5_openmpi|slate_cray))#/global/common/software/m2651/\1#g'
```
No run script references the deps via the `scratchperl` symlink form. (Agent 1.)

## Surface 2 — the modulefile / `lxrun` path (Agent 2). THE SILENT-BREAK RISK.

The `runs/**/run*.sh` scripts are not the only place the mounts are defined. A base
`LORRAX_SHIFTER` definition lives in the source checkouts'
`sources/lorrax_*/config/modulefiles/` and is spliced verbatim by
`sources/.../lorrax_agent/1.0.lua:133`. And `sources/lorrax_*/docs/
ENVIRONMENT_COMPREHENSIVE.md` documents `$LORRAX_FFI_{NVHPC,PHDF5,SLATE}_DIR` env
defaults currently set to `$SCRATCH/...`. **If you fix only `runs/`, anyone using the
`lxrun`/module path keeps the scratch mounts.** The `$LORRAX_FFI_*_DIR` defaults are
the cleanest single lever — change them once and both the modulefile and FFI build
inherit the new path. NOTE: these live in the source repo (`sources/lorrax_D/...`),
not the sandbox proper — only the live `lorrax_D` worktree is load-bearing; the other
worktrees (`lorrax_A/B/C`, phdf5 worktrees) have their own parameterized
`$SCRATCH`/`${USER}` defaults the anchored sed won't catch — hand-edit those. (Agents 1 & 2.)

## Surface 3 — docs (Agent 2). Essentially ONE sandbox file.

`AGENTS.md`, `agents_xprof.md`, `PARSE_OUTPUTS.md`, `KNOWN_SANDBOX_ERRORS.md`, all
templates, and every SKILL except one have **zero** scratch dep-path references.

- **`skills/execute_workflow/SKILL.md:139-151`** — the `SHIFTER=` prefix. Already stale
  in a *different* way: it has NO nvhpc/phdf5/slate mounts at all and its
  `PYTHONPATH=/global/u2/j/jackm/software/lorrax/src` points at a path that doesn't
  exist on disk. It's outdated (predates the distributed-GPU/FFI stack becoming
  default), not an alternate single-GPU code path. Replace with a prefix that matches
  the real run scripts AND the new dep location. (Agent 2 has a drop-in.)
- **Leave alone (historical, date-guarded):** `CHANGELOG.md:2398-2401` (dated 2026-04-16)
  and the whole `reports/lorrax_install_maintain_blitz_2026-05-13/` campaign — they use
  container-internal mount paths or `$SCRATCH/.jax_cache` (a cache, not a dep).

## Gotcha to verify before trusting the $HOME tree

`$HOME/software/lorrax_phdf5_openmpi` ALREADY EXISTS but contains only `cache/`, NOT
the `stage/` the run scripts mount. Don't assume any partial home tree is complete —
the copy must land the `stage/` subdirs. (Agent 2.)

## Recommended execution order

1. Pick + confirm write access to `/global/common/software/m2651` (or fall back to a
   pruned `$HOME/software` only if common/software is unavailable — but mind the 1 GiB
   free).
2. `cp -a` the four trees; verify with `find -xtype l` (no dangling) and a spot `cmp`.
3. Sed-sweep `runs/` sources; hand-edit the non-conforming `sources/lorrax_D` FFI
   infra + the `$LORRAX_FFI_*_DIR` defaults / modulefile.
4. Update `execute_workflow/SKILL.md:139-151`.
5. **Smoke test before deleting scratch originals:** one 1-rank shifter run with the new
   `--volume` source, confirm libs resolve (`ldd` inside container / job completes).
6. Only then remove the scratch dep dirs.

## Open questions for the user
- Target `/global/common/software/m2651` (recommended) or insist on `$HOME/software`?
- Do you own/maintain the `sources/lorrax_D/config/modulefiles/` + `ENVIRONMENT_
  COMPREHENSIVE.md` env defaults, or is that a separate repo you'd rather not touch
  from here? (Determines whether Surface 2 is in-scope.)
- Prune unused nvhpc `0.7.0`/`0.8.0`, or copy the whole tree?
