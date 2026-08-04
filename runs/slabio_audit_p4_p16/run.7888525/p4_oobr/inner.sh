. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_audit/src_fix/src:$PYTHONPATH
export LORRAX_PHDF5_WRITE_DEBUG=1
export SLABIO_CASE=oob
export SLABIO_OOB=read
export SLABIO_DIR=/scratch2/08271/jackmc/slabio_audit/run.7888525/p4_oobr
export SLABIO_NQ=2
export SLABIO_NG=8
cd /scratch2/08271/jackmc/slabio_audit/run.7888525/p4_oobr
taskset -c $(( ${SLURM_LOCALID:-0} * 28 ))-$(( ${SLURM_LOCALID:-0} * 28 + 27 )) \
  "$LORRAX_PY" -u -m slabio_probe \
    > /scratch2/08271/jackmc/slabio_audit/run.7888525/p4_oobr/rank_${SLURM_PROCID}.out 2> /scratch2/08271/jackmc/slabio_audit/run.7888525/p4_oobr/rank_${SLURM_PROCID}.err
RC=$?
echo "rc=$RC rank=${SLURM_PROCID}" >> /scratch2/08271/jackmc/slabio_audit/run.7888525/p4_oobr/rc.txt
exit $RC
