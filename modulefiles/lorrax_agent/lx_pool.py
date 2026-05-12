#!/usr/bin/env python3
"""lx_pool — pool-aware coordinator for shared LORRAX interactive allocations.

Backs the lxrun / lxstatus / lxattach shell functions defined in the
`lorrax_agent` Lmod module. All state is read live from SLURM (squeue,
scontrol); the only on-disk state is per-agent heartbeat files at
~/.lorrax/agents/<LETTER>.heartbeat, used purely for "when did agent X
last invoke a tool" diagnostics.

Subcommands
-----------
  status                    Print the pool table for $SLURM_JOBID.
  status --agents           Same, plus per-agent heartbeat ages.
  prelaunch <N>             Print banner to stderr; on success print N free
                            node names to stdout (space separated).
                            Exit 1 with guidance if pool can't satisfy.
  reap [--yes]              Cancel current agent's stale steps (over the
                            time threshold). --yes skips the confirmation.
  attach                    Find an existing lx-alloc-$USER allocation and
                            print "export SLURM_JOBID=...". If 0 → exit 1.
                            If >1 → list and exit 1.
  heartbeat                 Touch this agent's heartbeat file. Used by the
                            shell functions on every call.
  other-allocs              Print other lx-alloc-$USER allocations (one per
                            line: JID NODES TIMELEFT). Used by prelaunch on
                            pool-full.

Environment
-----------
  SLURM_JOBID     active allocation (required for status / prelaunch / reap)
  LORRAX_AGENT    A/B/C/D... letter (set by `module load lorrax_agent`)
  USER            for filtering lx-alloc-$USER jobs

Design notes
------------
* squeue / scontrol are the source of truth for "what's running where" —
  we never maintain a parallel state file for that.
* Heartbeat files are advisory: they tell you whether an agent's shell is
  alive recently. They never affect capacity calculations.
* All node selection is by name (nidXXXXXX). We pick lexicographically;
  --immediate=10 on srun resolves any race with another agent.
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# --- constants --------------------------------------------------------------

# Steps older than this age are flagged as suspicious in the banner and
# eligible for `lxstatus --reap`. 4h matches the Perlmutter interactive QOS
# upper bound, so any step older than 4h is past its useful life.
STALE_STEP_SECONDS = 4 * 3600

HEARTBEAT_DIR = Path.home() / ".lorrax" / "agents"

# ANSI colors. Disabled via LORRAX_NO_COLOR=1 or non-tty stderr.
def _color_enabled() -> bool:
    if os.environ.get("LORRAX_NO_COLOR"):
        return False
    return sys.stderr.isatty()

C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_CYAN = "\033[36m"

def _c(s: str, color: str) -> str:
    if not _color_enabled():
        return s
    return f"{color}{s}{C_RESET}"


# --- subprocess helpers -----------------------------------------------------

def _run(cmd: list[str], check: bool = True) -> str:
    """Run a command and return stdout, stripped. On failure, exit with
    a clear message rather than letting a stack trace leak to the user."""
    try:
        out = subprocess.run(
            cmd, check=check, capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired:
        print(f"lx_pool: timeout running {' '.join(shlex.quote(c) for c in cmd)}",
              file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        print(f"lx_pool: command not found: {cmd[0]}", file=sys.stderr)
        sys.exit(2)
    if check and out.returncode != 0:
        print(f"lx_pool: {' '.join(shlex.quote(c) for c in cmd)} failed:",
              file=sys.stderr)
        if out.stderr.strip():
            print(out.stderr.strip(), file=sys.stderr)
        sys.exit(2)
    return out.stdout.strip()


def _expand_nodelist(compressed: str) -> list[str]:
    """nid[001234,001236-001237] → ['nid001234', 'nid001236', 'nid001237']."""
    if not compressed:
        return []
    out = _run(["scontrol", "show", "hostnames", compressed])
    return [n for n in out.splitlines() if n]


# --- data model -------------------------------------------------------------

@dataclass
class Step:
    step_id: str        # e.g. "1234567.42"
    name: str           # e.g. "lx-A-7f3"
    nodelist: str       # compressed
    runtime_s: int      # elapsed seconds
    state: str          # RUNNING, COMPLETING, etc.

    @property
    def nodes(self) -> list[str]:
        return _expand_nodelist(self.nodelist)

    @property
    def owner(self) -> str:
        # job-name convention: lx-<LETTER>-<suffix>; default "?".
        if self.name.startswith("lx-") and len(self.name) >= 5:
            return self.name[3]
        return "?"

    @property
    def is_lorrax(self) -> bool:
        return self.name.startswith("lx-")


@dataclass
class Allocation:
    jobid: str
    name: str
    nodelist: str       # compressed
    time_left: str      # e.g. "1:23:45"
    nodes: list[str]    # expanded


# --- queries ----------------------------------------------------------------

def _parse_runtime(s: str) -> int:
    """Parse SLURM time formats (e.g. "1:23", "1:23:45", "1-12:34:56") to seconds."""
    if not s or s in ("INVALID", "NOT_SET", "0:00", "00:00"):
        return 0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [int(p) for p in parts]
    if len(parts) == 1:
        sec = parts[0]
    elif len(parts) == 2:
        sec = parts[0] * 60 + parts[1]
    elif len(parts) == 3:
        sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
    else:
        sec = 0
    return days * 86400 + sec


def get_steps(jobid: str) -> list[Step]:
    """Return currently-active *user* steps within `jobid`.

    SLURM auto-creates `<jobid>.extern` (and `<jobid>.batch` for sbatch
    jobs) to hold the allocation alive — those are not user work and
    must not count against pool capacity, or lxrun on a fresh
    `lxalloc 1` refuses with "0/1 free" because the only step it sees
    is the extern. Filter by step-ID suffix; user steps are
    always `<jobid>.<int>`.
    """
    out = _run(
        ["squeue", "-j", jobid, "-s", "--noheader",
         "-o", "%i|%j|%N|%M|%T"],
        check=False,
    )
    SYSTEM_SUFFIXES = (".extern", ".batch", ".interactive")
    steps = []
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            sid, name, nodelist, runtime, state = line.split("|")
        except ValueError:
            continue
        sid = sid.strip()
        if any(sid.endswith(s) for s in SYSTEM_SUFFIXES):
            continue
        steps.append(Step(
            step_id=sid,
            name=name.strip(),
            nodelist=nodelist.strip(),
            runtime_s=_parse_runtime(runtime.strip()),
            state=state.strip(),
        ))
    return steps


def get_allocation(jobid: str) -> Allocation | None:
    """Return the salloc-level Allocation record for `jobid`, or None if gone."""
    out = _run(
        ["squeue", "-j", jobid, "--noheader", "-o", "%i|%j|%N|%L"],
        check=False,
    )
    if not out:
        return None
    for line in out.splitlines():
        try:
            jid, name, nodelist, time_left = line.split("|")
        except ValueError:
            continue
        if jid.strip() == str(jobid):
            return Allocation(
                jobid=jid.strip(),
                name=name.strip(),
                nodelist=nodelist.strip(),
                time_left=time_left.strip(),
                nodes=_expand_nodelist(nodelist.strip()),
            )
    return None


def get_user_allocations() -> list[Allocation]:
    """Return all running interactive allocations owned by $USER, named lx-alloc-*."""
    out = _run(
        ["squeue", "--me", "-h", "-t", "RUNNING",
         "-o", "%i|%j|%N|%L"],
        check=False,
    )
    allocs = []
    for line in out.splitlines():
        try:
            jid, name, nodelist, time_left = line.split("|")
        except ValueError:
            continue
        name = name.strip()
        if not name.startswith("lx-alloc-"):
            continue
        allocs.append(Allocation(
            jobid=jid.strip(),
            name=name,
            nodelist=nodelist.strip(),
            time_left=time_left.strip(),
            nodes=_expand_nodelist(nodelist.strip()),
        ))
    return allocs


# --- heartbeat --------------------------------------------------------------

def heartbeat_path(letter: str) -> Path:
    return HEARTBEAT_DIR / f"{letter}.heartbeat"


def touch_heartbeat(letter: str) -> None:
    if not letter:
        return
    HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    p = heartbeat_path(letter)
    p.touch(exist_ok=True)
    os.utime(p, None)


def heartbeat_age_seconds(letter: str) -> int | None:
    p = heartbeat_path(letter)
    if not p.exists():
        return None
    return int(time.time() - p.stat().st_mtime)


# --- formatting -------------------------------------------------------------

def _fmt_age(seconds: int | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 86400}d{(seconds % 86400) // 3600}h"


def _fmt_clock(epoch_seconds: float) -> str:
    return time.strftime("%H:%M", time.localtime(epoch_seconds))


def _state_label(busy: bool, has_stale: bool) -> str:
    if has_stale:
        return _c("STALE ", C_RED)
    if busy:
        return _c("busy  ", C_YELLOW)
    return _c("free  ", C_GREEN)


def render_banner(
    alloc: Allocation,
    steps: list[Step],
    incoming_nodes: list[str] | None = None,
    incoming_owner: str | None = None,
    incoming_cmd: str | None = None,
    show_agents: bool = False,
) -> str:
    """Build the multi-line banner string."""
    incoming_nodes = incoming_nodes or []
    busy_by_node: dict[str, list[Step]] = {n: [] for n in alloc.nodes}
    for s in steps:
        for n in s.nodes:
            busy_by_node.setdefault(n, []).append(s)

    free_count = sum(1 for n in alloc.nodes if not busy_by_node[n])
    total = len(alloc.nodes)
    header_parts = [
        f"JID {alloc.jobid}",
        f"{total} nodes",
        f"{alloc.time_left} left",
        f"{free_count}/{total} free",
    ]
    if incoming_owner:
        header_parts.append(f"incoming: lx-{incoming_owner}")
    header = "[lxrun] " + " · ".join(header_parts)

    lines = [_c(header, C_BOLD)]
    now = time.time()
    for idx, node in enumerate(alloc.nodes, start=1):
        steps_here = busy_by_node[node]
        if steps_here:
            s = steps_here[0]  # primary step (--overlap can stack; pick first)
            stale = s.runtime_s > STALE_STEP_SECONDS
            label = _state_label(True, stale)
            owner = s.name
            if len(steps_here) > 1:
                owner += f" (+{len(steps_here)-1})"
            runtime = _fmt_age(s.runtime_s)
            started_clock = _fmt_clock(now - s.runtime_s)
            row = (f"  {idx}  {node}  {label} {owner:<14} "
                   f"{runtime:<6} started {started_clock}")
            if stale:
                row += _c("  ⚠ exceeds 4h", C_RED)
        else:
            label = _state_label(False, False)
            row = f"  {idx}  {node}  {label} {'—':<14} {'—':<6} —"
            if node in incoming_nodes:
                inc_label = f"lx-{incoming_owner}" if incoming_owner else "incoming"
                if incoming_cmd:
                    inc_label += f"  {incoming_cmd[:24]}"
                row += _c(f"  ← starting {inc_label}", C_CYAN)
        lines.append(row)

    if show_agents:
        lines.append("")
        lines.append(_c("  agents (heartbeat = last lxrun in any shell):", C_DIM))
        # Discover agent letters from heartbeat dir + any seen in steps.
        letters = set()
        if HEARTBEAT_DIR.exists():
            for p in HEARTBEAT_DIR.glob("*.heartbeat"):
                letters.add(p.stem)
        for s in steps:
            if s.owner != "?":
                letters.add(s.owner)
        for letter in sorted(letters):
            age = heartbeat_age_seconds(letter)
            running = [s for s in steps if s.owner == letter]
            running_str = ", ".join(s.name for s in running) if running else "—"
            self_marker = ""
            if letter == os.environ.get("LORRAX_AGENT"):
                self_marker = _c("  (you)", C_CYAN)
            lines.append(
                f"    agent {letter}  last lxrun {_fmt_age(age):<6}  "
                f"steps: {running_str}{self_marker}"
            )

    return "\n".join(lines)


# --- subcommands ------------------------------------------------------------

def _require_jobid() -> str:
    jid = os.environ.get("SLURM_JOBID")
    if not jid:
        print("lx_pool: SLURM_JOBID not set. Run `lxalloc` to create one, "
              "or `lxattach` to use an existing lx-alloc allocation.",
              file=sys.stderr)
        sys.exit(1)
    return jid


def cmd_status(args: argparse.Namespace) -> int:
    jid = _require_jobid()
    alloc = get_allocation(jid)
    if alloc is None:
        print(f"lx_pool: SLURM_JOBID={jid} is not a running allocation. "
              f"Did it expire? Re-run `lxalloc` or `lxattach`.",
              file=sys.stderr)
        return 1
    steps = get_steps(jid)
    print(render_banner(alloc, steps, show_agents=args.agents), file=sys.stderr)
    touch_heartbeat(os.environ.get("LORRAX_AGENT", ""))
    return 0


def cmd_prelaunch(args: argparse.Namespace) -> int:
    """Print banner + chosen nodes, or fail with guidance."""
    jid = _require_jobid()
    alloc = get_allocation(jid)
    if alloc is None:
        print(f"lx_pool: SLURM_JOBID={jid} is not a running allocation.",
              file=sys.stderr)
        return 1

    steps = get_steps(jid)
    busy_nodes = set()
    for s in steps:
        for n in s.nodes:
            busy_nodes.add(n)
    free = [n for n in alloc.nodes if n not in busy_nodes]

    n_needed = args.nodes
    letter = os.environ.get("LORRAX_AGENT", "?")
    cmd_hint = (args.cmd or "")[:32]

    if len(free) >= n_needed:
        chosen = free[:n_needed]
        print(render_banner(alloc, steps,
                            incoming_nodes=chosen,
                            incoming_owner=letter,
                            incoming_cmd=cmd_hint),
              file=sys.stderr)
        # stdout: space-separated nodes for the shell wrapper
        print(" ".join(chosen))
        touch_heartbeat(letter)
        return 0

    # Pool can't satisfy. Render banner anyway so user sees the picture,
    # then print guidance on options.
    print(render_banner(alloc, steps,
                        incoming_owner=letter,
                        incoming_cmd=cmd_hint),
          file=sys.stderr)
    print("", file=sys.stderr)
    print(_c(f"[lxrun] insufficient capacity: need {n_needed} free node(s), "
             f"have {len(free)}", C_RED), file=sys.stderr)

    # Suggest alternatives.
    print("", file=sys.stderr)
    print("  Options:", file=sys.stderr)

    # (a) other allocations the user owns
    others = [a for a in get_user_allocations() if a.jobid != jid]
    if others:
        print("    (a) Use an allocation you already own:", file=sys.stderr)
        for a in others:
            print(f"          export SLURM_JOBID={a.jobid}   "
                  f"# {a.name}, {len(a.nodes)} nodes, {a.time_left} left",
                  file=sys.stderr)
    else:
        print("    (a) [no other lx-alloc-$USER allocations found]",
              file=sys.stderr)

    # (b) spawn another
    print("    (b) Open another interactive allocation:", file=sys.stderr)
    print(f"          lxalloc {n_needed}", file=sys.stderr)

    # (c) wait for the earliest step to free its node(s).
    if steps:
        # Approximate "earliest free": the step with the *smallest* remaining
        # default time. We don't know step --time, so just report the oldest
        # currently-running step as the most-likely-to-finish-soon.
        oldest = max(steps, key=lambda s: s.runtime_s)
        clock_started = _fmt_clock(time.time() - oldest.runtime_s)
        print(f"    (c) Wait — oldest running step is {oldest.name} on "
              f"{oldest.nodelist} (started {clock_started}, "
              f"running {_fmt_age(oldest.runtime_s)})", file=sys.stderr)

    touch_heartbeat(letter)
    return 1


def cmd_reap(args: argparse.Namespace) -> int:
    jid = _require_jobid()
    letter = os.environ.get("LORRAX_AGENT", "")
    if not letter:
        print("lx_pool: LORRAX_AGENT not set; refusing to reap "
              "(can only reap your own steps).", file=sys.stderr)
        return 1

    steps = get_steps(jid)
    candidates = [
        s for s in steps
        if s.owner == letter and s.runtime_s > STALE_STEP_SECONDS
    ]
    if not candidates:
        print(f"lx_pool: no stale steps for agent {letter} "
              f"(threshold {STALE_STEP_SECONDS // 3600}h).", file=sys.stderr)
        return 0

    print(f"Stale steps for agent {letter} (over "
          f"{STALE_STEP_SECONDS // 3600}h):", file=sys.stderr)
    for s in candidates:
        print(f"  {s.step_id}  {s.name}  on {s.nodelist}  "
              f"running {_fmt_age(s.runtime_s)}", file=sys.stderr)

    if not args.yes:
        try:
            ans = input("Cancel these steps? [y/N] ").strip().lower()
        except EOFError:
            ans = "n"
        if ans not in ("y", "yes"):
            print("aborted.", file=sys.stderr)
            return 1

    for s in candidates:
        _run(["scancel", s.step_id], check=False)
        print(f"  cancelled {s.step_id}", file=sys.stderr)
    return 0


def cmd_attach(args: argparse.Namespace) -> int:
    """Find an lx-alloc-$USER allocation and print `export SLURM_JOBID=N`."""
    allocs = get_user_allocations()
    if not allocs:
        print("lx_pool: no lx-alloc-$USER allocations found. "
              "Run `lxalloc` to create one.", file=sys.stderr)
        return 1
    if len(allocs) > 1:
        print("lx_pool: multiple lx-alloc-$USER allocations found. "
              "Pick one and `export SLURM_JOBID=...` explicitly:",
              file=sys.stderr)
        for a in allocs:
            print(f"  JID {a.jobid}  {a.name}  "
                  f"nodes={a.nodelist}  time_left={a.time_left}",
                  file=sys.stderr)
        return 1
    a = allocs[0]
    print(f"export SLURM_JOBID={a.jobid}")
    print(f"lx_pool: attached to JID {a.jobid} "
          f"({a.name}, {len(a.nodes)} nodes, {a.time_left} left)",
          file=sys.stderr)
    touch_heartbeat(os.environ.get("LORRAX_AGENT", ""))
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    letter = os.environ.get("LORRAX_AGENT", "")
    if not letter:
        return 0  # silent no-op
    touch_heartbeat(letter)
    return 0


def cmd_other_allocs(args: argparse.Namespace) -> int:
    jid = os.environ.get("SLURM_JOBID", "")
    others = [a for a in get_user_allocations() if a.jobid != jid]
    for a in others:
        print(f"{a.jobid}\t{a.name}\t{a.nodelist}\t{a.time_left}")
    return 0


# --- entrypoint -------------------------------------------------------------

EPILOG = """
Status banner columns
  index  nid        state  owner          runtime  started
   1     nid001234  busy   lx-A-7f3       23m      13:49

  state:    free | busy | STALE (running > 4h, probably wedged)
  owner:    --job-name of the running step (lx-<letter>-<suffix>)

Environment knobs (set before lxrun)
  LORRAX_NNODES     nodes for next lxrun (default 1)
  LORRAX_NGPU       GPUs per node (default 4)
  LORRAX_IMMEDIATE  srun --immediate seconds, fail-fast budget (default 10)
  LORRAX_MPI_TYPE   --mpi= flag (default cray_shasta; pmix for FFI/cusolvermp/phdf5)
  LORRAX_NO_COLOR   set to 1 to disable banner colors

Troubleshooting
  module load lorrax_agent fails ("not found"):
    → module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
  lxrun says SLURM_JOBID not set:
    → lxattach (or lxalloc N if no allocation exists yet)
  lxattach finds 0 allocations:
    → lxalloc N (in a backgrounded shell), then lxattach again
  lxattach finds >1 allocation:
    → pick one and `export SLURM_JOBID=<jid>` directly
  lxrun says "insufficient capacity":
    → use the menu it prints: another existing alloc, lxalloc N, or wait
  banner shows STALE step that's yours:
    → `lx_pool.py reap` to cancel it (only acts on your own steps)
  STALE step belongs to another agent:
    → flag it to the user; you cannot cancel it
"""


def main() -> int:
    ap = argparse.ArgumentParser(prog="lx_pool",
        description=__doc__, epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("status", help="print pool table")
    p.add_argument("--agents", action="store_true",
                   help="include per-agent heartbeat info")
    p.set_defaults(func=cmd_status)

    p = sp.add_parser("prelaunch",
                      help="banner + free node selection for lxrun")
    p.add_argument("nodes", type=int, help="number of free nodes needed")
    p.add_argument("--cmd", default="", help="incoming command (for banner)")
    p.set_defaults(func=cmd_prelaunch)

    p = sp.add_parser("reap", help="cancel current agent's stale steps")
    p.add_argument("--yes", action="store_true",
                   help="skip confirmation prompt")
    p.set_defaults(func=cmd_reap)

    p = sp.add_parser("attach", help="find existing lx-alloc-$USER allocation")
    p.set_defaults(func=cmd_attach)

    p = sp.add_parser("heartbeat", help="touch agent heartbeat (silent)")
    p.set_defaults(func=cmd_heartbeat)

    p = sp.add_parser("other-allocs",
                      help="list other lx-alloc-$USER allocations (TSV)")
    p.set_defaults(func=cmd_other_allocs)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
