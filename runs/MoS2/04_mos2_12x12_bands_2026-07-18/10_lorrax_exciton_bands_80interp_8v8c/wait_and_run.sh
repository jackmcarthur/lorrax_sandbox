#!/bin/bash
# Block until SLURM job $JID is RUNNING (up to MAXWAIT sec), then exec run10.sh
# with the given args.  Lets a tracked background job own both the queue-wait and
# the compute (no foreground idle-wait).
#   usage: JID=<jid> NGPU=<n> MAXWAIT=<sec> ./wait_and_run.sh <workdir> <args...>
set -uo pipefail
JID="${JID:?set JID}"; MAXWAIT="${MAXWAIT:-2700}"
D="$(cd "$(dirname "$0")" && pwd)"
t=0
while [ $t -lt "$MAXWAIT" ]; do
  st=$(squeue -j "$JID" -h -o %T 2>/dev/null || echo GONE)
  [ "$st" = "RUNNING" ] && break
  sleep 10; t=$((t+10))
done
st=$(squeue -j "$JID" -h -o %T 2>/dev/null || echo GONE)
if [ "$st" != "RUNNING" ]; then
  echo "[wait_and_run] JID=$JID not RUNNING after ${t}s (state=$st)"; exit 3
fi
echo "[wait_and_run] JID=$JID RUNNING after ${t}s on $(squeue -j "$JID" -h -o %N); launching: $*"
exec "$D/run10.sh" "$@"
