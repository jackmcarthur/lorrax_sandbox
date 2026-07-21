#!/bin/bash
# Build the stage-4 variant run dirs.
#
# Variants that only change the Sigma stage (xi floor, far-band scissor,
# omega grid) set `restart = true` and SHARE the stage-3 ISDF restart
# (tmp/zeta_q.h5 + tmp/isdf_tensors_*.h5) by symlink -- the restart path in
# gw_init is read-only (everything that writes tmp/ lives inside
# `if not cfg.restart:`), so this is safe and saves the 467 s zeta fit + the
# V_q build per variant.
#
# Variants that change the zeta fit itself (charge_zeta_solve, zeta_rcond)
# CANNOT restart: they get restart = false and their own tmp/.
set -uo pipefail
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21
SRC=$R/00_lorrax_gw_2400c
S4=$R/_stage4

mk () {   # $1 = tag, $2 = restart(true|false)
  local RD=$S4/$1
  mkdir -p "$RD"
  ln -sfn ../../qe/nscf/WFN.h5 "$RD/WFN.h5"
  ln -sfn "$SRC/kin_ion.h5"    "$RD/kin_ion.h5"
  ln -sfn "$SRC/dipole.h5"     "$RD/dipole.h5"
  cp -f "$SRC/centroids.txt"   "$RD/centroids.txt"
  cp -f "$SRC/Mo.upf" "$SRC/S.upf" "$RD/"
  if [ "$2" = true ]; then
    ln -sfn "$SRC/tmp" "$RD/tmp"
  else
    rm -f "$RD/tmp"; mkdir -p "$RD/tmp"
  fi
  echo "  built $RD (restart=$2)"
}

echo "stage-4 variant dirs:"
mk a_xi_lifted      true
mk c_scissor_extrap true
mk c_wide_omega     true
mk b_cholesky       false
mk b_rcond_1e-6     false
mk b_rcond_1e-4     false
ls -d $S4/*/
