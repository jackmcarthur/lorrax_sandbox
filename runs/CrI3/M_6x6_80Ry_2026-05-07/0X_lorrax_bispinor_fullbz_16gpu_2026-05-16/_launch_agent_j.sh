#!/bin/bash -l
# Agent J Round-3 cross-config verification launcher.
# Usage: _launch_agent_j.sh <sandbox_dir> <slurm_jobid> <out_file> [extra_env]
#   extra_env: optional comma-separated NAME=VAL pairs added to shifter --env=...
#              and a special token "INCLUDE_V_Q" removes LORRAX_EXIT_AFTER_ZETA.

set -u

SB="${1:?sandbox dir}"
JID="${2:?slurm jobid}"
OUT="${3:?output file}"
EXTRA="${4:-}"

cd "$SB"

# Module init (login shell already sourced lmod init).
module purge 2>&1 >/dev/null || true
module load lorrax_B 2>&1 >/dev/null
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent 2>&1 >/dev/null

export SLURM_JOBID="$JID"
export LORRAX_NNODES=4
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120  # give lxrun 2 min to acquire nodes (handles stale step cleanup)

# Default env probes (always on for this run).
SHIFTER_ENVS="--env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_RCHUNK_DEBUG=1"

# By default exit after zeta; INCLUDE_V_Q token disables LORRAX_EXIT_AFTER_ZETA
# only (keep MAX_RCHUNKS=3 so V_q runs on partial ζ — buffer sizes still
# match the full-BZ allocation since they're shape-driven not value-driven).
SHIFTER_ENVS="$SHIFTER_ENVS --env=LORRAX_MAX_RCHUNKS=3"
if [[ "$EXTRA" != *"INCLUDE_V_Q"* ]]; then
    SHIFTER_ENVS="$SHIFTER_ENVS --env=LORRAX_EXIT_AFTER_ZETA=1"
fi

# Splice any extra NAME=VAL pairs from EXTRA (excluding INCLUDE_V_Q).
for kv in $(echo "$EXTRA" | tr ',' ' '); do
    if [[ "$kv" == "INCLUDE_V_Q" ]] || [[ -z "$kv" ]]; then continue; fi
    SHIFTER_ENVS="$SHIFTER_ENVS --env=$kv"
done

LORRAX_SHIFTER="$LORRAX_SHIFTER $SHIFTER_ENVS" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 \
  | tee "$OUT"
