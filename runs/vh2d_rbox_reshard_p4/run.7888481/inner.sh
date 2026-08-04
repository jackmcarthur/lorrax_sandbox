. /scratch2/08271/jackmc/b600_p64/harness/inner_common.sh
export PYTHONPATH=/scratch2/08271/jackmc/vh2d/lorrax_frozen/src:$PYTHONPATH
cd /scratch2/08271/jackmc/vh2d/run.7888481
taskset -c $(( ${SLURM_LOCALID:-0} * 28 ))-$(( ${SLURM_LOCALID:-0} * 28 + 27 )) \
  "$LORRAX_PY" -u -m vh2d_check > /scratch2/08271/jackmc/vh2d/run.7888481/rank_${SLURM_PROCID}.out 2> /scratch2/08271/jackmc/vh2d/run.7888481/rank_${SLURM_PROCID}.err
RC=$?
echo "rc=$RC rank=${SLURM_PROCID}" >> /scratch2/08271/jackmc/vh2d/run.7888481/rc.txt
exit $RC
