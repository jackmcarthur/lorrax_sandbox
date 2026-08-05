#!/bin/bash
# Background nvidia-smi sampler — appends timestamps + memory.used MB
# for ALL GPUs on the local node every 0.2s.  Run alongside the LORRAX
# job to catch transient cuFFT/NCCL allocations the discrete _mem_probe
# samples miss.
OUT="${1:?out file}"
INTERVAL="${2:-0.2}"
> "$OUT"
while true; do
  TS=$(date +%s.%N)
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -v ts="$TS" '{printf "%s,%s\n", ts, $0}' >> "$OUT"
  sleep "$INTERVAL"
done
