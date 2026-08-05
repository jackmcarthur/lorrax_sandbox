. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_padding/src_fix/src:$PYTHONPATH
export PADRANK_CASE=repro
export PADRANK_PATH=/scratch2/08271/jackmc/slabio_padding/gate.7888650/padrank_repro/padrank_gate.h5
cd /scratch2/08271/jackmc/slabio_padding/gate.7888650/padrank_repro
taskset -c $(( ${SLURM_LOCALID:-0} * 28 ))-$(( ${SLURM_LOCALID:-0} * 28 + 27 )) \
  "$LORRAX_PY" -u -m phdf5_padded_rank_write \
    > /scratch2/08271/jackmc/slabio_padding/gate.7888650/padrank_repro/rank_${SLURM_PROCID}.out 2> /scratch2/08271/jackmc/slabio_padding/gate.7888650/padrank_repro/rank_${SLURM_PROCID}.err
RC=$?
echo "rc=$RC rank=${SLURM_PROCID}" >> /scratch2/08271/jackmc/slabio_padding/gate.7888650/padrank_repro/rc.txt
exit $RC
