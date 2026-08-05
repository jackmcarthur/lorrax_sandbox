. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_audit/src_fix/src:$PYTHONPATH
export PADRANK_CASE=read_pad
export PADRANK_PATH=/scratch2/08271/jackmc/slabio_audit/gate.7888537/read_pad/padrank_gate.h5
cd /scratch2/08271/jackmc/slabio_audit/gate.7888537/read_pad
taskset -c $(( ${SLURM_LOCALID:-0} * 28 ))-$(( ${SLURM_LOCALID:-0} * 28 + 27 )) \
  "$LORRAX_PY" -u -m phdf5_padded_rank_write \
    > /scratch2/08271/jackmc/slabio_audit/gate.7888537/read_pad/rank_${SLURM_PROCID}.out 2> /scratch2/08271/jackmc/slabio_audit/gate.7888537/read_pad/rank_${SLURM_PROCID}.err
RC=$?
echo "rc=$RC rank=${SLURM_PROCID}" >> /scratch2/08271/jackmc/slabio_audit/gate.7888537/read_pad/rc.txt
exit $RC
