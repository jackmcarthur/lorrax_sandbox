#!/bin/bash -l
# HLO-dump launcher for V3 config (r=24576, b=32, cs=100).
# Uses XLA_FLAGS via shifter --env to enable per-module HLO + memory-usage-report.
set -u

cd "$(dirname "$0")/v3_hlo_sb"

module purge 2>&1 >/dev/null || true
module load lorrax_B 2>&1 >/dev/null
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent 2>&1 >/dev/null

export SLURM_JOBID=53075115
export LORRAX_NNODES=4
export LORRAX_NGPU=4
export LORRAX_IMMEDIATE=120

# Dump HLO + buffer-assignment to a per-rank-aware path.  We use only rank-0
# dumping by suffixing per-process is automatic when running multi-rank.
XLA_DUMP_DIR="$PWD/xla_dump"
mkdir -p "$XLA_DUMP_DIR"
XLA_FLAGS_VAL="--xla_dump_to=$XLA_DUMP_DIR --xla_dump_hlo_as_text --xla_dump_hlo_as_proto --xla_dump_include_timestamp=false"

SHIFTER_ENVS="--env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_RCHUNK_DEBUG=1 \
  --env=LORRAX_EXIT_AFTER_ZETA=1 \
  --env=LORRAX_MAX_RCHUNKS=2 \
  --env=XLA_FLAGS=$XLA_FLAGS_VAL"

LORRAX_SHIFTER="$LORRAX_SHIFTER $SHIFTER_ENVS" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 \
  | tee "$PWD/../agent_j_v3_hlo.out"
