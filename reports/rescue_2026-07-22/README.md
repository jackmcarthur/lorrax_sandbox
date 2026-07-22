# lorrax_D rescue, 2026-07-22 (pre-outage)

Two pieces of work were sitting **uncommitted** in the `sources/lorrax_D`
checkout hours before the 2026-07-22→08-06 Perlmutter maintenance outage.
Both are preserved here as patches. Neither has been reviewed or merged.

| patch | origin | on GitHub as a branch? |
|---|---|---|
| `lorrax-D-worktree-edits.patch` | uncommitted `AGENTS.md` + `docs/ENVIRONMENT_COMPREHENSIVE.md` edits and `test_jvp_verify.py` | yes — `rescue/lorrax-D-worktree-2026-07-22` (`fb831c6`) |
| `lorrax-D-stash-head-wing-fix.patch` | `git stash` "On agent-D/head-wing-fix: head-wing-fix WIP 2026-05-04" | **no** — see below |

## Why the stash is a patch and not a branch

It IS committed locally as `rescue/lorrax-D-stash-head-wing-fix` (`db9a885`,
parent `424e0dc`, which is already on origin — so only 1 commit / 159 small
objects need transferring). Pushing it nevertheless hangs indefinitely, and so
does `git bundle create`. Both stall in the same place:

    trace: run_command: git pack-objects --all-progress-implied --revs \
                        --stdout --thin --delta-base-offset -q

SSH auth and `git ls-remote` to the same remote both succeed, and other pushes
to `lorrax.git` (including `rescue/lorrax-D-worktree-2026-07-22` and several
to `main`) went through normally. So this is not network, not auth, and not
data volume — it is the object-graph walk in this particular checkout, which
has 23 local branches and, apparently, a packfile/loose-object state that
makes `pack-objects` pathological.

**To finish the job after the outage:** run `git gc --aggressive` (or at least
`git repack -ad`) in `sources/lorrax_D`, then
`git push origin rescue/lorrax-D-stash-head-wing-fix`. If that still hangs,
apply `lorrax-D-stash-head-wing-fix.patch` onto `424e0dc` in a fresh clone.

The patch is the authoritative copy; recover from it if the branch is lost.
