. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/slabio_padding/src_fix/src:$PYTHONPATH
export SLABIO_CASE=raw
export SLABIO_OOB=write
export SLABIO_DIR=/scratch2/08271/jackmc/slabio_padding/gate16.7888651/probe_raw
export LORRAX_PHDF5_WRITE_DEBUG=1
cd /scratch2/08271/jackmc/slabio_padding/gate16.7888651/probe_raw
taskset -c $(( ${SLURM_LOCALID:-0} * 28 ))-$(( ${SLURM_LOCALID:-0} * 28 + 27 )) \
  "$LORRAX_PY" -u -m slabio_probe \
    > /scratch2/08271/jackmc/slabio_padding/gate16.7888651/probe_raw/rank_${SLURM_PROCID}.out 2> /scratch2/08271/jackmc/slabio_padding/gate16.7888651/probe_raw/rank_${SLURM_PROCID}.err
RC=$?
echo "rc=$RC rank=${SLURM_PROCID}" >> /scratch2/08271/jackmc/slabio_padding/gate16.7888651/probe_raw/rc.txt
exit $RC
