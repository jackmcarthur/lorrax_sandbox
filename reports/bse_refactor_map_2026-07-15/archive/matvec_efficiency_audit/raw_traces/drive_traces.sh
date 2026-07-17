#!/bin/bash
# Phase-B: xprof traces (device timeline) for the key configs + ring nt8
# (memory-vs-time tradeoff) + the flag experiment (latency-hiding scheduler).
JID="${JID:?set JID}"
RAW=/pscratch/sd/j/jackm/lorrax_sandbox/reports/bse_refactor_map_2026-07-15/archive/matvec_efficiency_audit/raw_traces
mkdir -p $RAW/traces $RAW/logs

trace() {  # kind regime px py nt
  local kind=$1 regime=$2 px=$3 py=$4 nt=$5
  local ngpu=$((px*py)); local tag="${kind}_${regime}_${px}x${py}_nt${nt}"
  echo "===== TRACE $tag ====="
  JID=$JID NGPU=$ngpu timeout 500 $RAW/runlx.sh $RAW \
    python3 -u matvec_profile.py $kind $regime $px $py --ntrials $nt \
      --trace $RAW/traces/$tag --reps 10 \
    > $RAW/logs/trace_${tag}.log 2>&1
  echo "  exit=$? -> traces/$tag"
}

# --- traces: single-matvec device timeline, both regimes, 1x1 + 2x2 ---
trace stack inflated 2 2 8      # COMMS#2 primary: overlap + per-collective timeline
trace stack inflated 1 1 1      # regime baseline (dossier single-call)
trace stack inflated 2 2 1
trace stack fixture 2 2 8       # fixture latency-bound timeline
trace stack fixture 1 1 1
trace ring   inflated 2 2 1     # COMMS#5 ppermute vs all_gather timeline
trace ring   fixture 2 2 1      # COMMS#3 ppermute launch timeline
trace full   fixture 2 2 1      # ~20ms SOLVE structure

# --- ring nt8 for the memory-vs-time tradeoff (stack scans, ring batches) ---
echo "===== RING nt8 mem/warm ====="
for cfg in "inflated 2 2 8" "inflated 1 1 8" "fixture 2 2 8" "fixture 1 1 8"; do
  set -- $cfg; regime=$1 px=$2 py=$3 nt=$4; ngpu=$((px*py))
  tag="ring_${regime}_${px}x${py}_nt${nt}"
  JID=$JID NGPU=$ngpu timeout 500 $RAW/runlx.sh $RAW \
    python3 -u matvec_profile.py ring $regime $px $py --ntrials $nt --mem --reps 15 \
    > $RAW/logs/${tag}.log 2>&1
  echo "  $tag exit=$?"; grep -E "^\[(mem|warm)\]" $RAW/logs/${tag}.log
done

echo "ALL DONE TRACES"
