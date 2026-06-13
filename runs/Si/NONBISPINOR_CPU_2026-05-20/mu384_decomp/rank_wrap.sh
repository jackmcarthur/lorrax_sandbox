#!/bin/bash
# Per-rank wrapper:
#   * /usr/bin/time -v captures final Maximum resident set size (whole process)
#   * rss_main.py background sampler captures per-rank RSS timeline + peak
#
# Usage:
#   srun --jobid=... -N 1 -n 4 -c 8 --cpu-bind=cores \
#        ./rank_wrap.sh -i cohsex.in
#
# Reads:
#   RSS_TAG      scenario tag (passed through to rss_main and used in time log filename)
#   PY           absolute path to the python interpreter
set -e
RANK=${SLURM_PROCID:-0}
TAG=${RSS_TAG:-default}
TIMEOUT="time_rank${RANK}_${TAG}.log"
exec /usr/bin/time -v -o "$TIMEOUT" "$PY" -u rss_main.py "$@"
