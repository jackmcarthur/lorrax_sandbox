#!/bin/bash -l
# Agent L Round-5 live verification launcher.
# Usage: _launch_agent_l.sh <sandbox_dir> <slurm_jobid> <out_file>
set -u

SB="${1:?sandbox dir}"
JID="${2:?slurm jobid}"
OUT="${3:?output file}"

cd "$SB"

# Module init (login shell already sourced lmod init).
module purge 2>&1 >/dev/null || true
module load lorrax_B 2>&1 >/dev/null
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent 2>&1 >/dev/null

export SLURM_JOBID="$JID"
export LORRAX_NNODES=4
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

# Probes & limits per task: zeta-only verification, 3 r-chunks, full BZ
SHIFTER_ENVS="--env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_RCHUNK_DEBUG=1 \
  --env=LORRAX_MAX_RCHUNKS=3 \
  --env=LORRAX_EXIT_AFTER_ZETA=1"

LORRAX_SHIFTER="$LORRAX_SHIFTER $SHIFTER_ENVS" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 \
  | tee "$OUT"
