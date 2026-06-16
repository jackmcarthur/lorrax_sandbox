-- -*- lua -*-
-- lorrax_agent 1.0 — pool-aware overlay for shared LORRAX interactive sessions.
--
-- This module is sandbox-local dev tooling, *not* part of the upstream
-- LORRAX module (which lives in sources/lorrax_X/config/modulefiles/lorrax/).
-- Load it on top of a base lorrax_X module to coordinate multiple agents
-- (A/B/C/D) sharing a single salloc allocation.
--
-- Setup (once per shell)
--   module use $LORRAX_SANDBOX/modulefiles
--   module load lorrax_A           # or B / C / D — your assigned variant
--   module load lorrax_agent       # adds pool-aware lxrun + lxstatus etc.
--
-- What it does
--   * Derives LORRAX_AGENT (A/B/C/D) from the loaded base module's path.
--   * Overrides lxrun: prelaunch banner, free-node selection, --immediate,
--     --job-name=lx-${LORRAX_AGENT}-<suffix>, multi-node via LORRAX_NNODES.
--   * Overrides lxalloc: tags job as lx-alloc-${USER} so other shells can
--     find it via `lxattach`.
--   * Overrides lxshell: adds --overlap so an open interactive shell on a
--     compute node doesn't block other agents' lxrun on the same node.
--   * Adds lxstatus / lxstatus --agents / lxreap / lxattach.
--   * Touches ~/.lorrax/agents/${LORRAX_AGENT}.heartbeat on every call.
--
-- All complex logic (squeue parsing, banner rendering, free-node search,
-- reap, attach) lives in tools/agent/lx_pool.py.  This Lua file is just
-- shell-function glue.

help([[
lorrax_agent 1.0 — pool-aware overlay for shared LORRAX interactive sessions

This module makes lxrun safe to use when multiple agents (A/B/C/D) share
a single salloc allocation.  Each lxrun queries the pool, picks free
nodes, fails fast on contention, and prints a 4-line table so every
launch shows the state of the world.

Quick start — GPU (default)
  module load lorrax_A           # base module (your letter)
  module load lorrax_agent       # this overlay
  lxattach                       # find shared allocation (or `lxalloc N`)
  lxrun python3 -u -m gw.gw_jax -i cohsex.in

Quick start — CPU MPI (Milan partition, no GPUs)
  module load lorrax_B lorrax_agent
  LORRAX_PARTITION=cpu lxalloc 1 02:00:00 &
  lxattach
  LORRAX_PARTITION=cpu lxrun python3 -u -m gw.gw_jax -i cohsex.in
  # ranks: LORRAX_NRANKS (default 4); threads: LORRAX_CPUS_PER_TASK (default 8)
  # Needs cray-hdf5-parallel + cray-mpich + mpi4py + h5py-parallel in the venv
  # (one-time build, see docs/ENVIRONMENT_COMPREHENSIVE.md §3.5).

Multi-node from one agent
  LORRAX_NNODES=2 lxrun python3 -u -m gw.gw_jax -i cohsex.in

Diagnostics
  lxstatus                       # 4-line pool table
  lxstatus --agents              # + per-agent heartbeat ages
  lxreap                         # cancel your own steps over 4h
  lxattach                       # find existing lx-alloc-$USER allocation

Knobs (env vars)
  LORRAX_PARTITION     gpu (default) | cpu — both lxalloc and lxrun branch
  LORRAX_NNODES        nodes to request for next lxrun (default 1)
  LORRAX_NGPU          GPUs per node (default = base module's gpus_per_node)
                       — GPU partition only
  LORRAX_NRANKS        ranks per node on CPU partition (default 4)
  LORRAX_CPUS_PER_TASK cores per rank on CPU partition (default 8)
  LORRAX_NO_COLOR      disable ANSI colors in banner (set to 1)
  LORRAX_IMMEDIATE     srun --immediate seconds (default 10)
]])

whatis("Name:        lorrax_agent")
whatis("Version:     1.0")
whatis("Description: Pool-aware overlay for shared LORRAX interactive sessions")

-- =========================================================================
--  Prereq: a base lorrax_X must be loaded.  family("lorrax") in the base
--  module guarantees only one is loaded at a time.
-- =========================================================================

local lorrax_root = os.getenv("LORRAX_ROOT")
if not lorrax_root or lorrax_root == "" then
    LmodError([[
lorrax_agent requires a base lorrax module to be loaded first, e.g.:
  module load lorrax_A
LORRAX_ROOT is not set.
]])
end

-- Derive the agent letter from the LORRAX_ROOT path.  Base module installs
-- live at $HOME/software/lorrax_<LETTER>, so the trailing component is
-- "lorrax_A" / "lorrax_B" / ...
local agent_letter = lorrax_root:match("/lorrax_([A-Z])$")
if not agent_letter then
    LmodError("lorrax_agent: could not derive agent letter from "
              .. "LORRAX_ROOT=" .. lorrax_root
              .. " (expected path ending in /lorrax_<LETTER>)")
end
setenv("LORRAX_AGENT", agent_letter)

-- =========================================================================
--  Sandbox tooling location (this modulefile's repo root).
-- =========================================================================

local this_file = myFileName()
local sandbox_root = this_file:match("(.+)/modulefiles/lorrax_agent/.*$")
if not sandbox_root then
    LmodError("lorrax_agent: could not derive sandbox_root from "
              .. "myFileName()=" .. this_file)
end
setenv("LORRAX_SANDBOX", sandbox_root)

-- lx_pool.py lives next to this modulefile. We expose it on PATH so
-- agents (and our own shell functions) can call it as `lx_pool.py`.
local module_dir = this_file:match("(.+)/[^/]+%.lua$")
local lx_pool    = pathJoin(module_dir, "lx_pool.py")

prepend_path("PATH", module_dir)

-- =========================================================================
--  Reused paths from the base module.  We rebuild the srun line ourselves
--  rather than wrapping the base lxrun, because the base bakes -N 1 and
--  has no node-pinning hook.  We mirror its structure exactly:
--    srun ... select_gpu.sh shifter ${shifter args} in_container.sh "$@"
-- =========================================================================

local select_gpu_sh   = pathJoin(lorrax_root, "src/ffi/common/cpp/select_gpu.sh")
local in_container_sh = pathJoin(lorrax_root, "src/ffi/common/cpp/in_container.sh")

-- LORRAX_SHIFTER is exported by the base module as the full
--   "shifter --image=... --module=... --volume=... --env=..."
-- string.  We splice it in verbatim.
local shifter_cmd = "$LORRAX_SHIFTER"

-- gpus_per_node default — base module sets nothing in env for this, so we
-- rely on the user's LORRAX_NGPU override or fall back to 4 (Perlmutter).
-- Reading site_config.sh would tie us to install layout; 4 is a safe
-- default since the only site we run on is Perlmutter A100.
local default_ngpu = "4"

-- =========================================================================
--  lxalloc override — adds -J lx-alloc-${USER} so other shells can find
--  this allocation via `lxattach`.
-- =========================================================================
--
-- The base lxalloc blocks the calling shell on `bash -c "sleep 100000"`
-- for the lifetime of the allocation.  We preserve that behavior; the
-- only addition is the job-name tag and a heartbeat touch on entry.
--
-- For Claude Code / harness use, lxalloc is typically run via run_in_background.
-- The agent then calls `lxattach` from its working shell to discover the JID.

set_shell_function("lxalloc", string.format([[
    %s heartbeat 2>/dev/null || true
    local nodes="${1:-1}"
    local time="${2:-02:00:00}"
    local partition="${LORRAX_PARTITION:-gpu}"
    if [ "${partition}" = "cpu" ]; then
        # Perlmutter Milan CPU partition.  No --gres=gpu, no --gpus.
        # The CPU partition is shared so we don't reserve all sockets;
        # the user controls -n / -c via LORRAX_NRANKS / LORRAX_CPUS_PER_TASK
        # at lxrun time.
        echo "lxalloc: ${nodes} CPU node(s), ${time} (constraint=cpu) — tag lx-alloc-${USER}"
        salloc --nodes=${nodes} --qos=interactive --time=${time} \
               --constraint=cpu \
               --account=m2651 \
               -J "lx-alloc-${USER}" \
               bash -c "sleep 100000"
    else
        local gpus=$((nodes * 4))
        echo "lxalloc: ${nodes} node(s), ${gpus} GPU(s), ${time} — tag lx-alloc-${USER}"
        salloc --nodes=${nodes} --qos=interactive --time=${time} \
               --constraint=gpu --gpus=${gpus} \
               --account=m2651 \
               -J "lx-alloc-${USER}" \
               bash -c "sleep 100000"
    fi
]], lx_pool), "")

-- =========================================================================
--  lxrun override — pool-aware launcher.
-- =========================================================================
--
-- Flow:
--   1. Touch heartbeat.
--   2. nnodes = ${LORRAX_NNODES:-1}.
--   3. Call `lx_pool.py prelaunch <nnodes>`:
--        - prints banner to stderr (always)
--        - on success, prints space-separated free node names to stdout
--        - on failure (pool full), prints guidance + exits 1
--   4. Build --nodelist=nidA,nidB,... and srun line.
--   5. srun --immediate=${LORRAX_IMMEDIATE:-10} --job-name=lx-A-<suffix>.
--
-- Job-name suffix uses HHMMSS-PID for human-readable uniqueness; even with
-- two agents in the same shell PID, the timestamp prevents collisions.

set_shell_function("lxrun", string.format([[
    # Lustre pre-stripe (preserved from base lxrun)
    if [ -z "${LORRAX_NO_PRESTRIPE:-}" ] && command -v lfs >/dev/null 2>&1; then
        mkdir -p "$PWD/tmp" 2>/dev/null
        lfs setstripe -c "${LORRAX_LUSTRE_STRIPE_COUNT:-16}" \
                      -S "${LORRAX_LUSTRE_STRIPE_SIZE:-4M}" \
                      "$PWD/tmp" >/dev/null 2>&1 || true
    fi

    if [ -z "${SLURM_JOBID:-}" ]; then
        echo "lxrun: SLURM_JOBID is not set." >&2
        echo "  Run 'lxalloc' to create an allocation, or 'lxattach' to" >&2
        echo "  attach to an existing lx-alloc-${USER} allocation." >&2
        return 1
    fi

    local nnodes="${LORRAX_NNODES:-1}"
    local partition="${LORRAX_PARTITION:-gpu}"
    local ngpu_per_node="${LORRAX_NGPU:-%s}"

    # Prelaunch: banner + free-node selection.  stderr is the banner;
    # stdout is "nid001234 nid001235" or empty on failure.
    local nodes_str
    if ! nodes_str=$(%s prelaunch "${nnodes}" --cmd "$*"); then
        return 1
    fi

    # Convert "nid001234 nid001235" → "nid001234,nid001235"
    local nodelist
    nodelist=$(echo "${nodes_str}" | tr ' ' ',')

    local mpitype="${LORRAX_MPI_TYPE:-cray_shasta}"
    local mpiflag=""
    if [ "${mpitype}" != "none" ]; then
        mpiflag="--mpi=${mpitype}"
    fi

    local immediate="${LORRAX_IMMEDIATE:-10}"
    # Suffix uses BASHPID (subshell-aware) and $RANDOM so two lxrun calls
    # backgrounded from the same loop still get distinct job names.
    local suffix="$(date +%%H%%M%%S)-${BASHPID:-$$}-${RANDOM}"
    local jobname="lx-${LORRAX_AGENT:-X}-${suffix}"

    if [ "${partition}" = "cpu" ]; then
        # CPU partition launch: native srun, no Shifter container, no
        # --gres=gpu, no select_gpu.sh, no in_container.sh.  The user
        # controls -n (ranks per node) and -c (cores per rank) via
        # LORRAX_NRANKS (default 4) and LORRAX_CPUS_PER_TASK (default 8).
        # These match the validated CPU MPI configuration on Perlmutter
        # Milan (see docs/ENVIRONMENT_COMPREHENSIVE.md §3.5).
        local nranks_per_node="${LORRAX_NRANKS:-4}"
        local cpus_per_task="${LORRAX_CPUS_PER_TASK:-8}"
        local total_ranks=$((nnodes * nranks_per_node))
        # MPICH_GPU_SUPPORT_ENABLED=0 is required on CPU nodes; the
        # default GPU-aware MPICH aborts if loaded without a GPU.
        export MPICH_GPU_SUPPORT_ENABLED=0
        srun --jobid="$SLURM_JOBID" $mpiflag \
            --nodelist="${nodelist}" -N "${nnodes}" \
            -n "${total_ranks}" -c "${cpus_per_task}" \
            --cpu-bind=cores \
            --immediate="${immediate}" \
            --job-name="${jobname}" \
            "$@"
        return $?
    fi

    # GPU path (default): Shifter + select_gpu.sh + in_container.sh.
    local total_ranks=$((nnodes * ngpu_per_node))
    srun --jobid="$SLURM_JOBID" $mpiflag \
        --nodelist="${nodelist}" -N "${nnodes}" -n "${total_ranks}" \
        --gres="gpu:${ngpu_per_node}" \
        --overlap \
        --immediate="${immediate}" \
        --job-name="${jobname}" \
        %s \
        %s \
        %s \
        "$@"
]], default_ngpu, lx_pool, select_gpu_sh, shifter_cmd, in_container_sh), "")

-- =========================================================================
--  lxshell override — single-rank pty shell with --overlap.
-- =========================================================================
--
-- Without --overlap, srun --pty $SHELL takes exclusive ownership of the
-- node's resources (per SLURM 20.11+ docs), blocking any other agent's
-- lxrun on that node until the shell exits.  In a shared session this is
-- a footgun.  --overlap lets the shell coexist with other steps on the
-- same node.
--
-- We still pin to a free node where possible, but a free-node lookup that
-- fails should fall through to "wherever has space" rather than hard-fail
-- — interactive shells need to be reachable.

set_shell_function("lxshell", string.format([[
    %s heartbeat 2>/dev/null || true
    local ngpu="${LORRAX_NGPU:-1}"

    if [ -z "${SLURM_JOBID:-}" ]; then
        echo "lxshell: SLURM_JOBID is not set; run lxalloc or lxattach first." >&2
        return 1
    fi

    # Try to pin to a free node; if pool full, fall through to --overlap
    # without --nodelist (SLURM picks).
    local nodelist=""
    local free_node
    if free_node=$(%s prelaunch 1 --cmd "shell" 2>/dev/null); then
        nodelist="--nodelist=${free_node}"
    else
        echo "lxshell: pool full; entering with --overlap on whichever node SLURM picks." >&2
    fi

    local suffix="$(date +%%H%%M%%S)-${BASHPID:-$$}-${RANDOM}"
    local jobname="lx-${LORRAX_AGENT:-X}-shell-${suffix}"

    srun --jobid="$SLURM_JOBID" --pty --overlap \
        ${nodelist} -N 1 -n 1 --gres="gpu:${ngpu}" \
        --job-name="${jobname}" \
        %s \
        %s \
        bash -l
]], lx_pool, lx_pool, select_gpu_sh, shifter_cmd), "")

-- =========================================================================
--  lxpre override — same as base, but each step goes through pool-aware
--  lxrun under the hood (so it picks a free node, prints banner, etc).
-- =========================================================================

set_shell_function("lxpre", [[
    local input="${1:?Usage: lxpre <cohsex.in> <n_centroids>}"
    local ncentroids="${2:?Usage: lxpre <cohsex.in> <n_centroids>}"
    local abs_input="$(cd "$(dirname "$input")" && pwd)/$(basename "$input")"

    LORRAX_NGPU=1 LORRAX_NNODES=1 lxrun \
        python3 -u -m centroid.kmeans_cli "$ncentroids" --seed 42 \
        || { echo "FAILED: centroid generation"; return 1; }

    LORRAX_NGPU=1 LORRAX_NNODES=1 lxrun \
        python3 -u -m psp.get_dipole_mtxels -i "$abs_input" \
        || { echo "FAILED: dipole matrix elements"; return 1; }

    LORRAX_NGPU=1 LORRAX_NNODES=1 lxrun \
        python3 -u -m gw.kin_ion_io -i "$abs_input" \
        || { echo "FAILED: kin_ion"; return 1; }

    echo "=== Preprocessing complete ==="
    ls -lh centroids_frac_*.txt dipole.h5 kin_ion.h5 2>/dev/null
]], "")

-- =========================================================================
--  lxstatus / lxreap / lxattach — thin shell wrappers around lx_pool.py.
-- =========================================================================

set_shell_function("lxstatus", string.format([[
    %s status "$@"
]], lx_pool), "")

set_shell_function("lxreap", string.format([[
    %s reap "$@"
]], lx_pool), "")

-- lxattach: query lx_pool.py for an existing lx-alloc-$USER allocation,
-- and `eval` only its single `export SLURM_JOBID=N` line.  Strict guard
-- (parses out the integer and re-builds the assignment) so we never eval
-- attacker-controlled input — even though the source is our own script.
--
-- Lua delimiters are `[=[ ... ]=]` so the embedded shell `[[ ... ]]` and
-- `]]` glob-bracket forms don't terminate the Lua string.
set_shell_function("lxattach", string.format([=[
    local out
    out=$(%s attach) || return $?
    local first_line
    first_line=$(printf '%%s' "$out" | head -n1)
    case "$first_line" in
        "export SLURM_JOBID="[0-9]*)
            local jid="${first_line#export SLURM_JOBID=}"
            case "$jid" in
                ''|*[!0-9]*)
                    echo "lxattach: bad JID in output: '$first_line'" >&2
                    return 1 ;;
            esac
            export SLURM_JOBID="$jid"
            echo "[lxattach] SLURM_JOBID=$SLURM_JOBID" >&2
            %s status
            ;;
        *)
            echo "lxattach: refusing to eval unexpected output: $first_line" >&2
            return 1
            ;;
    esac
]=], lx_pool, lx_pool), "")

-- =========================================================================
--  On unload: remove the function definitions Lmod doesn't auto-restore.
--  (Lmod handles this for set_shell_function we set, but not for the base
--   module's lxrun/lxalloc which we shadowed — those will be the base
--   module's again on unload, which is what we want.)
-- =========================================================================
