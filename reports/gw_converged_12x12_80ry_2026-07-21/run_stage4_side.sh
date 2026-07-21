#!/bin/bash
# Stage-4 Sigma-only variants on the 1-node side allocation (2x2 mesh).
# They restart from 00b's ISDF tensors, so no zeta fit -- ~2 min each.
# Reading a 4x4-written restart on a 2x2 mesh works (verified: "Loaded restart
# tensors from H5", zero r-chunk lines).
#   usage: JID=<jid> ./run_stage4_side.sh tag [tag...]
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21
RPT=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_converged_12x12_80ry_2026-07-21
SRCRUN=$R/00b_lorrax_gw_2400c_ranktrunc
for TAG in "$@"; do
  RD=$R/_stage4/$TAG
  ln -sfn $SRCRUN/tmp        $RD/tmp
  ln -sfn $SRCRUN/kin_ion.h5 $RD/kin_ion.h5
  ln -sfn $SRCRUN/dipole.h5  $RD/dipole.h5
  EXTRA=""; [ "$TAG" = a_xi_lifted ] && EXTRA="--lift-xi"
  srun --jobid=$JID --overlap --immediate=60 -N1 -n1 --gres=gpu:4 \
       bash -c 'pkill -9 -u $USER -f gw_probe; exit 0' >/dev/null 2>&1
  sleep 3
  echo "=== $TAG start $(date +%T) extra='$EXTRA'"
  JID=$JID NNODES=1 NTASKS=4 GRES=4 $R/run_shifter.sh $RD \
    python3 -u $RPT/gw_probe.py -i $RD/cohsex.in --cap-gib 8 $EXTRA > $RD/gw.out 2>&1
  echo "=== $TAG end $(date +%T) rc=$?"
  grep -aE "crossing conditioning|Computing L_q|Scissor fit|QSGW:|Diagonal SC" $RD/gw.out | head -4
done
