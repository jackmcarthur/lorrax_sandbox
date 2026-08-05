#!/bin/bash
# Sweep v5: 12-config bispinor 80Ry SOC memory-model param sweep on JID 53058709
# Robust per-config cleanup, polling for SLURM-step disappearance.
set -u

JID=53058709
RUNDIR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/M_6x6_80Ry_2026-05-07/0X_lorrax_bispinor_fullbz_16gpu_2026-05-16
LOGDIR=/tmp/sweep_v5_logs
TSV=/tmp/bispinor_80ry_sweep_v5.tsv
PROG=/tmp/bispinor_80ry_sweep_v5_progress.log

mkdir -p "$LOGDIR"
cd "$RUNDIR"

# ---- env / modules (must use module purge first) ----
module purge 2>/dev/null
module use /pscratch/sd/j/jackm/lorrax_sandbox/modulefiles 2>/dev/null
module use $HOME/modulefiles 2>/dev/null
module load lorrax_B 2>/dev/null
module load lorrax_agent 2>/dev/null
export SLURM_JOBID=$JID
export LORRAX_NNODES=4
export LORRAX_NGPU=4
# Full-BZ flag MUST go through shifter
export LORRAX_SHIFTER="$LORRAX_SHIFTER --env=LORRAX_FORCE_FULL_BZ=1"

# ---- TSV header ----
if [ ! -s "$TSV" ]; then
  printf "config\tr_chunk\tband_chunk\tgflat\tstatus\tmean_s_per_rchunk\tn_chunks\twall_s\tplanner_HWM_GB\tnotes\n" > "$TSV"
fi

echo "=== SWEEP_V5 BEGIN $(date) JID=$JID ===" | tee -a "$PROG"

CONFIGS=(
  "A1 4096 16"
  "A2 4096 32"
  "A3 4096 64"
  "A4 8192 16"
  "A5 8192 32"
  "A6 8192 64"
  "B1 16384 16"
  "B2 16384 32"
  "B3 16384 64"
  "B4 24576 16"
  "B5 24576 32"
  "B6 24576 64"
)

wait_for_clean() {
  local timeout=120 t0=$(date +%s)
  while true; do
    local n=$(squeue -j $JID -s -h -o '%.30i' 2>/dev/null | awk '$1 !~ /\.extern$/ && NF>0' | wc -l)
    if [ "$n" -eq 0 ]; then return 0; fi
    if [ $(($(date +%s) - t0)) -gt $timeout ]; then
      echo "    wait_for_clean: TIMEOUT after ${timeout}s, n_steps=$n" | tee -a "$PROG"
      squeue -j $JID -s 2>/dev/null | tee -a "$PROG"
      return 1
    fi
    sleep 5
  done
}

SRUN_BUSY_STREAK=0

run_one() {
  local CONF=$1 R=$2 B=$3
  local GFLAT=360
  local TAG="${CONF}_r${R}_b${B}"

  echo "=== SWEEP_V5 CONFIG $CONF r=$R b=$B START $(date) ===" | tee -a "$PROG"

  sed -i "s/^r_chunk_size = .*/r_chunk_size = $R/" cohsex.in
  sed -i "s/^band_chunk_size = .*/band_chunk_size = $B/" cohsex.in
  sed -i "s/^gflat_chunk_size = .*/gflat_chunk_size = $GFLAT/" cohsex.in

  rm -rf tmp/ gw.out
  mkdir -p tmp/

  local T0=$(date +%s)
  lxrun python3 -u -m gw.gw_jax -i $PWD/cohsex.in > gw.out 2>&1 &
  local PID=$!

  # Poll loop: success = saw r-chunk 8+; fail = error keywords; cap 360s
  while true; do
    local ELAPSED=$(($(date +%s) - T0))
    if [ $ELAPSED -gt 360 ]; then break; fi
    if grep -qE "Traceback|fatal|FAILED|cuFFT|nodes are busy|RESOURCE_EXHAUSTED|out of memory|OOM|Killed" gw.out 2>/dev/null; then break; fi
    if grep -q "Started zeta fitting" gw.out 2>/dev/null; then
      local LAST_RC=$(grep -oE "r-chunk +[0-9]+ /" gw.out | tail -1 | grep -oE "[0-9]+" | head -1)
      if [ "${LAST_RC:-0}" -ge 8 ]; then break; fi
    fi
    if ! kill -0 $PID 2>/dev/null; then break; fi
    sleep 5
  done

  local WALL=$(($(date +%s) - T0))

  # Save snapshot BEFORE killing (to preserve last lines)
  cp gw.out "$LOGDIR/${TAG}.out"

  # ---- Parse before killing ----
  local STATUS="unknown"
  local MEAN="0.0"
  local N_SAMPLED="0"
  local HWM_GB=""
  local NOTES="wall=${WALL}s"

  # Parse mean s/r-chunk and last sampled chunk
  python3 <<PYEOF > /tmp/parse_v5_$$.txt 2>&1
import re
times=[]; chunks=[]
try:
    with open("gw.out") as f:
        for line in f:
            m=re.search(r'\[ (\d{2}):(\d{2}):(\d{2}) \|.*\| +\d+% \] r-chunk +(\d+) / \d+', line)
            if m:
                h,mi,s,k=map(int,m.groups())
                t=h*3600+mi*60+s
                times.append(t); chunks.append(k)
except Exception as e:
    print(f"PARSE_ERR={e}")
diffs=[]
for i in range(1,len(times)):
    dt=times[i]-times[i-1]
    if dt < 0:
        dt += 86400
    dk=chunks[i]-chunks[i-1]
    if dk>0 and dt>0:
        diffs.append(dt/dk)
mean = sum(diffs)/len(diffs) if diffs else 0.0
print(f"MEAN={mean:.3f}")
print(f"NSAMPLED={chunks[-1] if chunks else 0}")
PYEOF
  MEAN=$(grep "^MEAN=" /tmp/parse_v5_$$.txt 2>/dev/null | head -1 | cut -d= -f2)
  N_SAMPLED=$(grep "^NSAMPLED=" /tmp/parse_v5_$$.txt 2>/dev/null | head -1 | cut -d= -f2)
  rm -f /tmp/parse_v5_$$.txt
  : "${MEAN:=0.0}"
  : "${N_SAMPLED:=0}"

  # Parse HWM
  HWM_GB=$(grep "HWM estimate" gw.out 2>/dev/null | tail -1 | grep -oE "= +[0-9.]+ GB" | head -1 | awk '{print $2}')
  : "${HWM_GB:=}"

  # Get total n_chunks for projected wall computation later
  local N_CHUNKS_TOTAL=$(grep -oE "Zeta fitting: +[0-9]+ r-chunks" gw.out 2>/dev/null | head -1 | grep -oE "[0-9]+" | head -1)
  : "${N_CHUNKS_TOTAL:=0}"

  # Determine status
  if grep -qE "RESOURCE_EXHAUSTED|out of memory" gw.out 2>/dev/null; then
    STATUS="OOM"
  elif grep -q "nodes are busy" gw.out 2>/dev/null; then
    STATUS="srun_busy"
    SRUN_BUSY_STREAK=$((SRUN_BUSY_STREAK+1))
  elif [ "${N_SAMPLED:-0}" -ge 8 ]; then
    STATUS="ok"
    SRUN_BUSY_STREAK=0
  elif [ "${N_SAMPLED:-0}" -gt 0 ]; then
    STATUS="ok_partial"
    SRUN_BUSY_STREAK=0
  elif grep -q "Started zeta fitting" gw.out 2>/dev/null; then
    STATUS="fit_timeout"
  elif grep -qE "Traceback|fatal|FAILED" gw.out 2>/dev/null; then
    STATUS="crash"
  else
    STATUS="compile_timeout"
  fi

  # Append last r-chunk to notes
  local LAST_RCHUNK=$(grep -oE "r-chunk +[0-9]+ / [0-9]+" gw.out 2>/dev/null | tail -1 | tr -s ' ' '_')
  if [ -n "$LAST_RCHUNK" ]; then
    NOTES="$NOTES;$LAST_RCHUNK"
  fi
  if [ "$N_CHUNKS_TOTAL" -gt 0 ]; then
    NOTES="$NOTES;total=$N_CHUNKS_TOTAL"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$CONF" "$R" "$B" "$GFLAT" "$STATUS" "$MEAN" "$N_SAMPLED" "$WALL" "$HWM_GB" "$NOTES" >> "$TSV"

  # ---- Kill and wait for SLURM steps to clear ----
  kill $PID 2>/dev/null
  wait $PID 2>/dev/null

  # Also try to scancel non-extern steps for this JID (best-effort)
  squeue -j $JID -s -h -o '%i' 2>/dev/null | awk '$1 !~ /\.extern$/ && NF>0' | xargs -r scancel 2>/dev/null

  if ! wait_for_clean; then
    echo "=== SWEEP_V5 WAIT_FOR_CLEAN FAILED after $CONF ===" | tee -a "$PROG"
  fi

  echo "=== SWEEP_V5 CONFIG $CONF r=$R b=$B DONE wall=${WALL}s status=$STATUS mean=${MEAN}s rcs=$N_SAMPLED ===" | tee -a "$PROG"

  # Abort if two consecutive srun_busy
  if [ "$SRUN_BUSY_STREAK" -ge 2 ]; then
    echo "=== SWEEP_V5 ABORT $(date) reason=srun_busy_streak ===" | tee -a "$PROG"
    return 99
  fi
  return 0
}

for c in "${CONFIGS[@]}"; do
  read -r NAME R B <<<"$c"
  run_one "$NAME" "$R" "$B"
  rc=$?
  if [ $rc -eq 99 ]; then
    exit 1
  fi
done

echo "=== SWEEP_V5 ALL DONE $(date) ===" | tee -a "$PROG"
