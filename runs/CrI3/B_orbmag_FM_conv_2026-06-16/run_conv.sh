#!/bin/bash
# FM CrI3 convergence study: one magnetic SCF, reused for NSCFs at several grids.
# Requires SLURM_JOBID (JID) from an lxattach'd 2-node allocation.
JID="${SLURM_JOBID:?need SLURM_JOBID}"
R=/pscratch/sd/j/jackm/lorrax_sandbox/runs/CrI3/B_orbmag_FM_conv_2026-06-16
module load espresso/7.5-libxc-7.0.0-gpu berkeleygw/4.0-nvhpc-23.9

run_nscf () {   # $1=grid dir   $2=npools   $3=nodes
  local d=$1 np=$2 nn=$3
  echo "===== NSCF $d (npools=$np, nodes=$nn) ====="
  cd $R/qe/$d
  rm -rf CrI3.save WFN WFN.h5; cp -r $R/qe/scf/CrI3.save .
  OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 --overlap -N$nn -n$((nn*4)) -c16 \
      pw.x -npools $np -i nscf.in > nscf.out 2>&1
  if grep -qi "Non magnetic calculation" nscf.out; then echo "!! NON-MAGNETIC $d"; fi
  grep -iE "Noncollinear calculation|number of k points" nscf.out | head -2
  if ! grep -qi "JOB DONE" nscf.out; then echo "!! NSCF FAILED $d"; tail -8 nscf.out; return 1; fi
  MPICH_GPU_SUPPORT_ENABLED=0 srun --jobid=$JID --gres=gpu:1 --overlap -N1 -n1 \
      pw2bgw.x -i pw2bgw.in > pw2bgw.out 2>&1
  srun --jobid=$JID --gres=gpu:1 --overlap -N1 -n1 wfn2hdf.x BIN WFN WFN.h5 > wfn2hdf.out 2>&1
  ls -la WFN.h5 && echo "OK $d"
}

echo "############ SCF (magnetic, 3x3) ############"
cd $R/qe/scf
OMP_NUM_THREADS=16 srun --jobid=$JID --gres=gpu:4 --overlap -N1 -n4 -c16 \
    pw.x -npools 4 -i scf.in > scf.out 2>&1
grep -E "convergence has been achieved|total magnetization" scf.out | tail -3
grep -qi "convergence has been achieved" scf.out || { echo "!! SCF not converged"; tail -5 scf.out; }

run_nscf nscf_6x6sym 4 1
run_nscf nscf_8x8    8 2
run_nscf nscf_10x10  8 2
echo "############ ALL DONE ############"
