#!/bin/bash
set -e
JID=52567301
D=/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si_pseudobands/00_si_2x2x2_60Ry/qe_nosym
cd $D

LOAD='module load PrgEnv-gnu cudatoolkit craype-accel-nvidia80 espresso berkeleygw 2>/dev/null'

echo "[$(date +%H:%M:%S)] NSCF nosym (8 kpts, nb=40)"
HDF5_USE_FILE_LOCKING=FALSE srun --jobid=$JID --gres=gpu:4 -N 1 -n 8 -c 16 \
    bash -c "$LOAD; pw.x -nk 8 -in $D/nscf.in" > $D/nscf.out 2>&1
grep -E "JOB DONE|ERROR" $D/nscf.out | head -2
echo "[$(date +%H:%M:%S)] pw2bgw"
HDF5_USE_FILE_LOCKING=FALSE srun --jobid=$JID --gres=gpu:1 -N 1 -n 1 -c 16 \
    bash -c "$LOAD; pw2bgw.x -in $D/pw2bgw.in" > $D/pw2bgw.out 2>&1
ls -la WFN VSC VKB kih.dat vxc.dat 2>&1 | head
echo "[$(date +%H:%M:%S)] parabands → WFN_pb.h5 (target: nb=4200)"
cat > $D/parabands.inp << 'EOF'
input_wfn_file WFN
output_wfn_file WFN_pb.h5
vsc_file VSC
vkb_file VKB
number_bands 4200
solver_algorithm 0
verbosity 2
EOF
HDF5_USE_FILE_LOCKING=FALSE srun --jobid=$JID --gres=gpu:4 -N 4 -n 16 -c 16 \
    bash -c "$LOAD; parabands.cplx.x" < $D/parabands.inp > $D/parabands.out 2>&1
grep -E "TOTAL|ERROR|nbnd" $D/parabands.out | head -5
ln -sf WFN_pb.h5 WFN.h5
ln -sf WFN_pb.h5 WFN_inner.h5
echo "[$(date +%H:%M:%S)] DONE"
