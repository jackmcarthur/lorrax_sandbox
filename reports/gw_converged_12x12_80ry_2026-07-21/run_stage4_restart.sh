#!/bin/bash
# Stage-4 variants that change ONLY the Sigma stage: they restart from the
# stage-3b ISDF tensors (symlinked tmp/), so each costs ~2 min instead of ~12.
# The restart path in gw_init is read-only -- everything that writes tmp/ is
# inside `if not cfg.restart:` -- so sharing one tmp/ across variants is safe.
#   usage: JID=<jid> [SRCRUN=<dir with tmp/>] ./run_stage4_restart.sh tag [tag...]
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21
RPT=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_converged_12x12_80ry_2026-07-21
S4=$R/_stage4
SH=$R/run_shifter.sh
SRCRUN="${SRCRUN:-$R/00b_lorrax_gw_2400c_ranktrunc}"

for TAG in "$@"; do
  RD=$S4/$TAG
  ln -sfn "$SRCRUN/tmp"        "$RD/tmp"
  ln -sfn "$SRCRUN/kin_ion.h5" "$RD/kin_ion.h5"
  ln -sfn "$SRCRUN/dipole.h5"  "$RD/dipole.h5"

  EXTRA=""
  [ "$TAG" = a_xi_lifted ] && EXTRA="--lift-xi"

  srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:4 \
       bash -c 'pkill -9 -u $USER -f gw_probe; pkill -9 -u $USER -f gw.gw_jax; exit 0' >/dev/null 2>&1
  sleep 3
  echo "=== $TAG start $(date +%T)  extra='$EXTRA'"
  JID=$JID NNODES=4 NTASKS=16 GRES=4 "$SH" "$RD" \
    python3 -u "$RPT/gw_probe.py" -i "$RD/cohsex.in" --cap-gib 8 $EXTRA > "$RD/gw.out" 2>&1
  echo "=== $TAG end $(date +%T) rc=$?"
  grep -aE "Computing L_q|crossing conditioning|Scissor fit|QSGW:|Diagonal SC" "$RD/gw.out" | head -4
done
