#!/bin/bash
# GN-PPM G0W0 producer: kin_ion + dipole (1 GPU each) -> gw_jax (16 GPU / 4 node).
# dipole.h5 IS required: the q=0 Coulomb head (default wcoul0_source=s_tensor) is
# built from the dipole S(omega) tensor (gw/head_correction.py:152).  Uses the
# run-root run_shifter.sh.
#   usage: JID=<jid> ./run_gw.sh
set -uo pipefail
JID="${JID:?set JID to the salloc job id}"
RD="$(cd "$(dirname "$0")" && pwd)"
RUN="$(cd "$RD/.." && pwd)"
SH="$RUN/run_shifter.sh"

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
