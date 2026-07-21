#!/bin/bash
# Post-fix A/B of 02_lorrax_gw_d3h_16gpu: IDENTICAL cohsex.in / WFN / 1496
# recovered-D3h centroids / 16 GPU; the ONLY difference is the LORRAX source
# (agent/gw-conduction-postfix = xi-floor crossing quadrature d011a36 +
# rank-revealing charge zeta-solve, zeta_rcond default 1e-6).
#   usage: JID=<jid> ./run_gw.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
RD="$(cd "$(dirname "$0")" && pwd)"
RUN="$(cd "$RD/.." && pwd)"
SH="$RUN/run_shifter_postfix.sh"

echo "=== kin_ion start $(date +%s) $(date)"
JID=$JID NNODES=1 NTASKS=1 GRES=1 "$SH" "$RD" \
  python3 -u -m gw.kin_ion_io -i cohsex.in > kin_ion.log 2>&1
echo "=== kin_ion end   $(date +%s) $(date)  rc=$?"

echo "=== dipole start $(date +%s) $(date)"
JID=$JID NNODES=1 NTASKS=1 GRES=1 "$SH" "$RD" \
  python3 -u -m psp.get_dipole_mtxels -i cohsex.in > dipole.log 2>&1
echo "=== dipole end   $(date +%s) $(date)  rc=$?"
ls -la kin_ion.h5 dipole.h5 2>&1 | tail -2

echo "=== gw_jax start $(date +%s) $(date)"
JID=$JID NNODES=4 NTASKS=16 GRES=4 "$SH" "$RD" \
  python3 -u -m gw.gw_jax -i "$RD/cohsex.in" > gw.out 2>&1
GWRC=$?
echo "=== gw_jax end   $(date +%s) $(date)  rc=$GWRC"
echo "--- gw.out tail ---"; tail -20 gw.out
ls -la eqp0.dat eqp1.dat sigma_diag.dat tmp/zeta_q.h5 2>&1 | tail -6
exit $GWRC
