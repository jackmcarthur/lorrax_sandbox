#!/bin/bash
# Phase-A sweep: HLO + mem + warm (+perterm for ring/full), both meshes, both
# regimes, capturing stderr (donation warnings). Each config -> own log+hlo.
# Runs on login node; each config is one srun step into the held allocation.
JID="${JID:?set JID}"
RAW=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_refactor_map_2026-07-15/archive/matvec_efficiency_audit/raw_traces
mkdir -p $RAW/logs $RAW/hlo
FILT='grep -vE "^(WARNING|W[0-9]{4}|INFO|I[0-9]{4}|libtpu|Platform|absl|E[0-9]{4})"'

run() {  # kind regime px py ntrials extra_flags...
  local kind=$1 regime=$2 px=$3 py=$4 nt=$5; shift 5
  local ngpu=$((px*py))
  local tag="${kind}_${regime}_${px}x${py}_nt${nt}"
  echo "===== $tag ====="
  JID=$JID NGPU=$ngpu timeout 500 $RAW/runlx.sh $RAW \
    python3 -u matvec_profile.py $kind $regime $px $py --ntrials $nt \
      --mem --hlo $RAW/hlo/${tag}.hlo --reps 15 "$@" \
    > $RAW/logs/${tag}.log 2>&1
  echo "  exit=$? -> logs/${tag}.log"
  grep -E "^\[(cfg|mem|cost|warm)\]|donated|not usable" $RAW/logs/${tag}.log | head -8
}

# --- STACK (production TDA) ---
run stack inflated 2 2 8            # COMMS#1 collective inventory, LAYOUT#1/#5
run stack inflated 1 1 8            # 1x1 baseline
run stack inflated 2 2 1
run stack inflated 1 1 1
run stack fixture 2 2 8             # donation, fixture 2x2
run stack fixture 2 2 1
run stack fixture 1 1 1
run stack fixture 1 1 8

# --- RING (legacy TDA, ppermute) ---
run ring inflated 2 2 1 --perterm  # COMMS#5 all_gather-vs-ppermute inflated
run ring inflated 1 1 1 --perterm
run ring fixture 2 2 1 --perterm    # COMMS#3 ppermute count
run ring fixture 1 1 1 --perterm

# --- FULL (non-TDA, the ~20ms SOLVE structure) ---
run full fixture 2 2 1 --perterm
run full inflated 1 1 1 --perterm

echo "ALL DONE"
