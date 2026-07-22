#!/bin/bash
# STEP 0 — regenerate ONLY the ISDF zeta tensor on the FULL BZ (144 q).
#
# WHY: the converged production GW (runs/MoS2/07_.../00b_lorrax_gw_2400c_ranktrunc)
# stores zeta IBZ-only (74 of 144 q) because the 2412 D3h centroids close under
# the density point group -> the IBZ cascade activates.  ``bse.vq_interp`` (the
# b26p arbitrary-Q V_Q model) requires FULL-BZ zeta (nq == nk) and refuses
# IBZ-only storage.  ``--vq-mode ongrid`` was the previous run's way around it,
# and it is exactly what limited that run to the 13 on-grid Q of the 12x12 mesh.
#
# NOTHING ELSE IS REGENERATED.  ``LORRAX_EXIT_AFTER_ZETA=1`` stops the driver
# immediately after ``gw_init.fit_zeta``, before ``compute_V_q`` and before any
# restart flush, so the production restart (psi_full_y / V_qmunu / W0_qmunu /
# enk_full / Sigma / eqp1.dat) is untouched and reused byte-for-byte.  The zeta
# fit is a deterministic least-squares fit of the SAME wavefunctions to the SAME
# 2412 centroids -- the only difference is that it is evaluated at all 144 q
# instead of the 74 IBZ representatives.  ``run_zeta_gate.sh`` proves that by
# rebuilding V_qmunu from the new zeta and diffing against the production tiles.
#
# THE CAP.  ``charge_zeta_solve = rank_truncate`` (the production setting, and
# the cure for the near-singular n_mu=2412 charge CCT) is honoured ONLY on the
# REPLICATED solve route, which ``isdf.core._resolve_solver_kind_charge`` picks
# only while nq*n_mu^2*16 <= _REPLICATED_CHOL_MAX_STACK_BYTES.  The converged
# campaign already had to raise that cap 4 -> 8 GiB for its IBZ stack
# (nq=74 -> 6.9 GiB; KNOWN_SANDBOX_ERRORS 2026-07-21 §1).  Doubling the q axis
# to the full BZ doubles the stack to 13.4 GiB, so 8 GiB is no longer enough and
# the route silently falls back to cusolvermp_cholesky again.  Measured on the
# first attempt (logs/zeta_fullbz_BAD_cholesky_fallback.log): zeta comes out
# ~4.5x too large in norm and rebuilding V_q from it misses the production
# V_qmunu tiles by relF 16-32 (vs 1e-15 for the production IBZ zeta).
# --cap-gib 16 covers 13.36 GiB.  VERIFY ``path=replicated_rank_truncate`` in
# the log; a ``path=cusolvermp_cholesky`` line means the cap is still too low.
#
#   usage: JID=<jid> [CAP_GIB=16] ./run_zeta_fullbz.sh
set -uo pipefail
JID="${JID:?set JID}"
CAP_GIB="${CAP_GIB:-16}"
PROBE=/pscratch/sd/j/jackm/lorrax_sandbox/reports/gw_converged_12x12_80ry_2026-07-21/gw_probe.py
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/09_mos2_exciton_smooth_2026-07-21
RD=$R/00_zeta_fullbz
SH=$R/run_shifter.sh
NN=4; NT=16

# GPU zombies masquerade as a phdf5 "negative offset" crash -- reap first.
srun --jobid=$JID --overlap --immediate=60 -N4 -n4 --gres=gpu:1 \
     bash -c 'pkill -9 -u $USER -f gw_jax; pkill -9 -u $USER -f exciton_bands; exit 0' >/dev/null 2>&1
sleep 3

echo "=== zeta full-BZ fit start $(date +%T)  ${NT} GPU / ${NN} nodes  cap=${CAP_GIB} GiB"
JID=$JID NNODES=$NN NTASKS=$NT GRES=4 \
  EXTRA_ENV="--env=LORRAX_FORCE_FULL_BZ=1 --env=LORRAX_EXIT_AFTER_ZETA=1" \
  "$SH" "$RD" python3 -u "$PROBE" -i cohsex.in --cap-gib "$CAP_GIB" \
  > "$RD/zeta_fullbz.log" 2>&1
rc=$?
echo "=== zeta full-BZ fit end $(date +%T) rc=$rc"
grep -aE "Zeta output|Finished zeta|Elapsed|n_q_disk|Traceback|Error|EXIT_AFTER_ZETA|path=|replicated-cap|probe\]" \
     "$RD/zeta_fullbz.log" | sort -u | tail -20
ls -la "$RD/tmp/"
exit $rc
