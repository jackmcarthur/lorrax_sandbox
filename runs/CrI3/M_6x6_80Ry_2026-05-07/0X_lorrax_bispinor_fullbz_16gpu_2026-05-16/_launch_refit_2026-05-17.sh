#!/bin/bash
# Memory-model refit baseline verify on JID 53075110 (4 nodes, 16 GPUs hbm80g).
# Runs ~3 r-chunks with HLO dump + per-r-chunk timing prints.

cd "$(dirname "$0")"

# `module purge` emits a cpe-restore warning whose return code is non-zero;
# leave -e off around module ops, enable -u only afterwards.
module purge || true
module load lorrax_B
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent

set -u

# Stay on JID 53075110 (the other allocation 53075115 belongs to a sibling agent).
export SLURM_JOBID=53075110

# 16-GPU mandatory for CrI3 production [[feedback-cri3-always-16-gpus]].
export LORRAX_NNODES=4
export LORRAX_NGPU=4   # per-node; total = 4 nodes × 4 = 16 ranks

# Shifter env passthrough per KNOWN_SANDBOX_ERRORS 2026-05-16 entries.
# Needed:  force full-BZ (avoid silent IBZ cascade), exit after ζ-fit
# (skip downstream sigma so we focus on ζ), max 3 r-chunks (cheap baseline),
# per-r-chunk timing debug, XLA HLO dump (text form) at $PWD/xla_dump.
export LORRAX_SHIFTER_OVERRIDE="$LORRAX_SHIFTER \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_EXIT_AFTER_ZETA=1 \
  --env=LORRAX_MAX_RCHUNKS=3 \
  --env=LORRAX_RCHUNK_DEBUG=1 \
  --env=XLA_FLAGS=--xla_dump_to=$PWD/xla_dump \
  --env=XLA_DUMP_HLO_AS_TEXT=true"

LORRAX_SHIFTER="$LORRAX_SHIFTER_OVERRIDE" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex.in" 2>&1 | tee gw_refit_2026-05-17.out
