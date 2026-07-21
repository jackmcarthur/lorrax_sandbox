#!/bin/bash
# Stage 3 — the converged MoS2 G0W0 on the 80 Ry / 12x12 / 400-band reference.
# 16 x A100-80GB (4 nodes), n_mu = 2412, nband = 326, QP bands 1..80.
#   usage: JID=<jid> [RDIR=<subdir>] [IN=<input file>] ./run_gw.sh
set -uo pipefail
JID="${JID:?set JID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21
RD="${RDIR:-$R/00_lorrax_gw_2400c}"
IN="${IN:-$RD/cohsex.in}"
SH=$R/run_shifter.sh

# GPU zombies from a previous GW step masquerade as a phdf5 "negative offset"
# crash -- reap before every run (KNOWN_SANDBOX_ERRORS).
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:4 \
     bash -c 'pkill -9 -u $USER -f gw.gw_jax; exit 0' >/dev/null 2>&1
sleep 3

echo "=== GW start $(date +%F' '%T)  rundir=$RD  input=$IN"
JID=$JID NNODES=4 NTASKS=16 GRES=4 "$SH" "$RD" \
  python3 -u -m gw.gw_jax -i "$IN" > "$RD/gw.out" 2>&1
rc=$?
echo "=== GW end $(date +%F' '%T) rc=$rc"
tail -5 "$RD/gw.out"
ls -la "$RD"/eqp0.dat "$RD"/eqp1.dat "$RD"/sigma_diag.dat 2>&1 | tail -3
exit $rc
