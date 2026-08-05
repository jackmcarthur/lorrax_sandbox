. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_audit/src_base/src:$PYTHONPATH
export LORRAX_PHDF5_WRITE_DEBUG=1
export SLABIO_CASE=raw
export SLABIO_OOB=write
export SLABIO_DIR=/scratch2/08271/jackmc/slabio_audit/run.7888528/p4_raw
export SLABIO_NQ=12
export SLABIO_NG=32768
cd /scratch2/08271/jackmc/slabio_audit/run.7888528/p4_raw
taskset -c $(( ${SLURM_LOCALID:-0} * 28 ))-$(( ${SLURM_LOCALID:-0} * 28 + 27 )) \
  "$LORRAX_PY" -u -m slabio_probe \
    > /scratch2/08271/jackmc/slabio_audit/run.7888528/p4_raw/rank_${SLURM_PROCID}.out 2> /scratch2/08271/jackmc/slabio_audit/run.7888528/p4_raw/rank_${SLURM_PROCID}.err
RC=$?
echo "rc=$RC rank=${SLURM_PROCID}" >> /scratch2/08271/jackmc/slabio_audit/run.7888528/p4_raw/rc.txt
exit $RC
