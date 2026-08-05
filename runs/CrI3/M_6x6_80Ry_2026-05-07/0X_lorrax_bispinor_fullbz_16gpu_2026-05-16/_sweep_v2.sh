#!/bin/bash
# Bispinor 80 Ry SOC memory-model sweep on 80 GB hbm80g hardware (JID 53058291)
# Sweep A: small + medium r_chunk (4096, 8192) x (16, 32, 64) bands.
set -u

cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16/
source ./_setup.sh

TSV=/tmp/bispinor_80ry_sweep_v2_A.tsv
PROG=/tmp/bispinor_80ry_sweep_v2_progress.log

# Initialise TSV header if not present.
if [ ! -s "$TSV" ]; then
  printf "config\tr_chunk\tband_chunk\tgflat_chunk\tstatus\tmean_s_per_rchunk\tn_chunks_sampled\tpeak_FFT_box_GB\tpeak_zeta_acc_GB\tplanner_HWM_GB\tbottleneck\tnotes\n" > "$TSV"
fi

echo "=== SWEEP_A_V2 BEGIN $(date) ===" | tee -a "$PROG"

# Config table: name r b
CONFIGS=(
  "A1 4096 16"
  "A2 4096 32"
  "A3 4096 64"
  "A4 8192 16"
  "A5 8192 32"
  "A6 8192 64"
)

run_one() {
  local CONF=$1 R=$2 B=$3
  local GFLAT=360

  echo "=== SWEEP_A_V2 CONFIG $CONF r=$R b=$B START $(date) ===" | tee -a "$PROG"

  # Edit cohsex.in
  sed -i "s/^r_chunk_size = .*/r_chunk_size = $R/" cohsex.in
  sed -i "s/^band_chunk_size = .*/band_chunk_size = $B/" cohsex.in
  # ensure gflat_chunk_size stays 360
  sed -i "s/^gflat_chunk_size = .*/gflat_chunk_size = $GFLAT/" cohsex.in

  # Clean
  rm -rf tmp/ gw.out
  mkdir -p tmp/

  local T0=$(date +%s)

  # Launch in background
  lxrun python3 -u -m gw.gw_jax -i $PWD/cohsex.in > gw.out 2>&1 &
  local PID=$!

  # Wait for either "Started zeta fitting" or an error or 10 min timeout
  while true; do
    if [ -s gw.out ] && grep -q "Started zeta fitting" gw.out 2>/dev/null; then break; fi
    if [ $(($(date +%s) - T0)) -gt 600 ]; then break; fi
    if [ -s gw.out ] && grep -qE "Error|Traceback|FAILED|OOM|Killed|fatal|srun: error" gw.out 2>/dev/null; then break; fi
    sleep 5
  done

  local STATUS="unknown"
  if grep -q "Started zeta fitting" gw.out 2>/dev/null; then
    # Wait for r-chunk 10-14 or 15 min cap.
    while true; do
      if grep -qE "r-chunk +(1[0-9]|[2-9][0-9]+) / " gw.out 2>/dev/null; then break; fi
      if [ $(($(date +%s) - T0)) -gt 900 ]; then break; fi
      if grep -qE "Error|Traceback|FAILED|OOM|Killed|fatal" gw.out 2>/dev/null; then break; fi
      sleep 10
    done
  fi

  # Kill the run and any orphans
  kill $PID 2>/dev/null
  # The python and shifter still inside srun should be killed via scancel of the lx-* step
  scancel -u $USER --jobid=53058291 --name=$(grep "^lx-" gw.out 2>/dev/null | head -1) 2>/dev/null
  # Generic: cancel any lx- job step in our JID owned by us, with non-extern name
  scancel --jobid=53058291 --signal=KILL --name="lx-B-*" 2>/dev/null
  # Cancel any srun steps initiated by this lxrun in our JID
  squeue -s -j 53058291 -o "%i %j" 2>/dev/null | awk '$2 ~ /^lx-/ {print $1}' | xargs -r scancel 2>/dev/null
  wait $PID 2>/dev/null
  sleep 3

  local WALL=$(($(date +%s) - T0))

  # === Parse output ===
  # 1. Progress-bar lines: "[ HH:MM:SS | ... |  N% ] r-chunk  K / TOT · ETA ..."
  # Extract HH:MM:SS and r-chunk K
  # 2. Memory diagnostics
  local FFT_GB ZETA_GB HWM_GB BOTTLENECK MEAN N_SAMPLED

  FFT_GB=$(grep "G-flat ζ accumulator" gw.out 2>/dev/null | head -1 | grep -oE "per-iter FFT box [0-9.]+ GB" | awk '{print $4}')
  ZETA_GB=$(grep "G-flat ζ accumulator" gw.out 2>/dev/null | head -1 | grep -oE "chunk_size=[0-9]+ → per-iter FFT box [0-9.]+ GB" | awk '{print $4}')
  # planner block
  HWM_GB=$(grep "HWM estimate" gw.out 2>/dev/null | tail -1 | grep -oE "= +[0-9.]+ GB" | head -1 | awk '{print $2}')
  BOTTLENECK=$(grep "HWM estimate" gw.out 2>/dev/null | tail -1 | grep -oE "bottleneck: [A-Za-z_]+" | awk '{print $2}')

  # Mean s/r-chunk: parse the progress-bar lines.
  # Format: "[ HH:MM:SS | ... | XX% ] r-chunk  K / TOT ..."
  python3 <<PYEOF > /tmp/parse_$$.txt 2>&1
import re,sys
times=[]
chunks=[]
with open("gw.out") as f:
    for line in f:
        m=re.search(r'\[ (\d{2}):(\d{2}):(\d{2}) \|.*\| +\d+% \] r-chunk +(\d+) / \d+', line)
        if m:
            h,mi,s,k=map(int,m.groups())
            t=h*3600+mi*60+s
            times.append(t); chunks.append(k)
# Use consecutive samples to compute (dt / dchunk).
if len(times)>=2:
    diffs=[]
    for i in range(1,len(times)):
        dt=times[i]-times[i-1]
        if dt < 0:  # midnight wrap
            dt += 86400
        dk=chunks[i]-chunks[i-1]
        if dk>0 and dt>0:
            diffs.append(dt/dk)
    mean = sum(diffs)/len(diffs) if diffs else 0.0
    print(f"MEAN={mean:.3f}")
    print(f"NSAMPLED={chunks[-1]}")
else:
    print("MEAN=0.0")
    print(f"NSAMPLED={chunks[-1] if chunks else 0}")
PYEOF
  MEAN=$(grep "^MEAN=" /tmp/parse_$$.txt | head -1 | cut -d= -f2)
  N_SAMPLED=$(grep "^NSAMPLED=" /tmp/parse_$$.txt | head -1 | cut -d= -f2)
  rm -f /tmp/parse_$$.txt
  : "${MEAN:=0.0}"
  : "${N_SAMPLED:=0}"

  # Determine status
  if [ -z "$N_SAMPLED" ] || [ "$N_SAMPLED" = "0" ]; then
    # No progress at all
    if grep -qE "RESOURCE_EXHAUSTED|out of memory|OOM" gw.out 2>/dev/null; then
      STATUS="OOM"
    elif grep -q "Started zeta fitting" gw.out 2>/dev/null; then
      STATUS="fit_timeout"
    elif grep -qE "Error|Traceback|FAILED|fatal" gw.out 2>/dev/null; then
      STATUS="crash"
    else
      STATUS="compile_timeout"
    fi
  elif [ "$N_SAMPLED" -ge 8 ]; then
    STATUS="ok"
  else
    STATUS="ok_partial"
  fi

  : "${FFT_GB:=}"
  : "${ZETA_GB:=}"
  : "${HWM_GB:=}"
  : "${BOTTLENECK:=}"

  # Notes: capture wall, last r-chunk line, any error
  local NOTES="wall=${WALL}s"
  if grep -qE "RESOURCE_EXHAUSTED|out of memory" gw.out 2>/dev/null; then
    NOTES="$NOTES;oom"
  fi
  local LAST_RCHUNK=$(grep -E "r-chunk +[0-9]+ / " gw.out 2>/dev/null | tail -1 | grep -oE "r-chunk +[0-9]+ / [0-9]+" | tr -s ' ' '_')
  if [ -n "$LAST_RCHUNK" ]; then
    NOTES="$NOTES;${LAST_RCHUNK}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$CONF" "$R" "$B" "$GFLAT" "$STATUS" "$MEAN" "$N_SAMPLED" "$FFT_GB" "$ZETA_GB" "$HWM_GB" "$BOTTLENECK" "$NOTES" >> "$TSV"

  # Archive gw.out for this config
  cp gw.out "_prior_attempts/gw.out.${CONF}_r${R}_b${B}_jid53058291"

  echo "=== SWEEP_A_V2 CONFIG $CONF r=$R b=$B DONE wall=${WALL}s status=$STATUS mean=${MEAN}s ===" | tee -a "$PROG"
}

for c in "${CONFIGS[@]}"; do
  read -r NAME R B <<<"$c"
  run_one "$NAME" "$R" "$B"
done

echo "=== SWEEP_A_V2 ALL DONE $(date) ===" | tee -a "$PROG"
