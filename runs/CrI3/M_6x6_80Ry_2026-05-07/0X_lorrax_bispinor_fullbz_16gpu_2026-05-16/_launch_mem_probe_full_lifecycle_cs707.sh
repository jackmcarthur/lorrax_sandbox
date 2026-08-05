#!/bin/bash
# Round-1 jax.live_arrays() probe — full ζ+V_q lifecycle map.
# Extends Round-0 (3 r-chunk-loop probes) with P0 zeta_fit_start,
# P1 pre_rchunk_loop, P3 zeta_fit_end, P4 pre_v_q, P5 post_v_q.
# cs=707 (safe regime) so we get through ζ-fit AND V_q without OOM.
# NO LORRAX_EXIT_AFTER_ZETA — we need V_q probes to fire.
# LORRAX_MAX_RCHUNKS=4 to limit the r-chunk loop (steady-state by chunk 2).

cd "$(dirname "$0")"

module purge || true
module load lorrax_B
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles
module load lorrax_agent

set -u

# Use one of the two alive 4-node hbm80g allocations (both alive ~1h+).
export SLURM_JOBID=53075115

# 16-GPU mandatory for CrI3 production [[feedback-cri3-always-16-gpus]].
export LORRAX_NNODES=4
export LORRAX_NGPU=4   # per-node; total = 16 ranks

# Shifter env passthrough per KNOWN_SANDBOX_ERRORS 2026-05-16 entries.
# NOTE: no LORRAX_EXIT_AFTER_ZETA — V_q must run for P4/P5 probes.
export LORRAX_SHIFTER_OVERRIDE="$LORRAX_SHIFTER \
  --env=LORRAX_MEM_DEBUG=1 \
  --env=LORRAX_FORCE_FULL_BZ=1 \
  --env=LORRAX_MAX_RCHUNKS=4 \
  --env=LORRAX_RCHUNK_DEBUG=1"

LORRAX_SHIFTER="$LORRAX_SHIFTER_OVERRIDE" \
  lxrun python3 -u -m gw.gw_jax -i "$PWD/cohsex_mem_probe_cs707.in" 2>&1 \
  | tee mem_probe_full_lifecycle_cs707.out
