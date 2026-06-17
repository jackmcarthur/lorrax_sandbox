#!/bin/bash
# pw2bgw -> wfn2hdf -> patch kgrid, for the VI3 band-path WFN. Run on the live alloc.
set -e
cd /pscratch/sd/j/jackm/lorrax_sandbox/runs/VI3/05_bandpath_orbmag_2026-06-17/qe/nscf_bandpath
JID=$(squeue -u jackm -h -t RUNNING -o "%i %j" | awk '/lx-alloc/{print $1;exit}')
module load espresso/7.5-libxc-7.0.0-gpu 2>/dev/null
PW2BGW=/global/common/software/nersc9/espresso/7.5-libxc-7.0.0-gpu/bin/pw2bgw.x
WFN2HDF=/global/common/software/nersc9/berkeleygw/zen3-ampere80/nvidia23.9/mpich/berkeleygw/BerkeleyGW-4.0/bin/wfn2hdf.x
echo "[pw2bgw] $(date)"
srun --jobid=$JID --overlap -N1 -n1 --gres=gpu:1 $PW2BGW < pw2bgw.in > pw2bgw.out 2>&1
echo "[pw2bgw] exit=$? ; $(grep -c 'JOB DONE' pw2bgw.out) JOB DONE"
echo "[wfn2hdf] $(date)"
srun --jobid=$JID --overlap -N1 -n1 --gres=gpu:1 $WFN2HDF BIN WFN WFN.h5 > wfn2hdf.out 2>&1
echo "[wfn2hdf] exit=$? ; WFN.h5 -> $(ls -la WFN.h5 2>/dev/null | awk '{print $5}')"
python3 - <<'PY'
import h5py, numpy as np
with h5py.File("WFN.h5","r+") as f:
    kg=f["mf_header/kpoints/kgrid"]; print("kgrid before:",kg[()])
    kg[...]=np.array([240,240,1],dtype=kg.dtype); print("kgrid after :",kg[()])
PY
echo "[done] $(date)"
