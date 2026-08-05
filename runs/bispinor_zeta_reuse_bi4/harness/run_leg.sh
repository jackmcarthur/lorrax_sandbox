# ===========================================================================
# run_leg.sh -- one instrumented driver launch.  Source from an sbatch body
# AFTER env_common.sh, with $RUN set to the run directory.
#
#   run_leg <tag> <python -m module> [argv...]
#
# Transcribed from the b600 bispinor harness with two changes:
#
#  1. PER-RANK stdout AND stderr go to separate files under
#     $R/logs/rank_<tag>.<jobid>/.  Under srun+apptainer the aggregate
#     stderr is LOST at teardown (the d935ce7 investigation lost the
#     writer's own oob= diagnostic that way), so every rank's streams are
#     captured to disk before the container tears down.  Rank 0's stdout
#     is ALSO tee'd to the leg log so the banners/timing table stay in
#     one place.
#  2. The inner script ends in `exit $RC` -- a trailing echo would mask
#     the driver's exit code with the echo's.
# ===========================================================================
run_leg () {
  local TAG=$1 MOD=$2; shift 2
  local LOG=$R/logs/${TAG}.$SLURM_JOB_ID.log
  local HWM=$R/logs/hwm_${TAG}.$SLURM_JOB_ID
  local RANKD=$R/logs/rank_${TAG}.$SLURM_JOB_ID
  rm -rf "$HWM" "$RANKD"; mkdir -p "$HWM" "$RANKD"
  local INNER=$RUN/inner_$TAG.$SLURM_JOB_ID.sh
  cat > "$INNER" <<EOS
. $R/harness/inner_common.sh
export LORRAX_HWM_DIR=$HWM
${LORRAX_LEG_ENV:-}
cd $RUN
P=\${SLURM_PROCID:-0}
if [ "\$P" = "0" ]; then
  lorrax_run $MOD $* 2> $RANKD/rank0.err | tee $RANKD/rank0.out
  RC=\${PIPESTATUS[0]}
else
  lorrax_run $MOD $* > $RANKD/rank\$P.out 2> $RANKD/rank\$P.err
  RC=\$?
fi
exit \$RC
EOS
  echo "##### [$TAG] $MOD $* -- $(date) #####"
  local t0=$(date +%s)
  srun --mpi=pmi2 -N "$SLURM_NNODES" --ntasks-per-node=2 -n "$SLURM_NTASKS" \
      apptainer exec --bind "$BIND" "$LORRAX_SIF" \
      bash "$INNER" > "$LOG" 2>&1
  local rc=$?
  local wall=$(( $(date +%s) - t0 ))
  echo "[$TAG rc=$rc wall=${wall}s log=$LOG]"
  echo "$wall" > $R/logs/wall_${TAG}.$SLURM_JOB_ID.txt
  echo "--- $TAG per-rank VmHWM ---"
  for f in $HWM/rank_*.hwm; do
    [ -s "$f" ] || continue
    awk '{for(i=1;i<=NF;i++) if($i=="VmHWM:" && $(i+1)>m) m=$(i+1)} END{if(m) print m}' "$f"
  done 2>/dev/null | sort -nr \
    | awk '{s+=$1; n++; if(NR==1)mx=$1; mn=$1}
           END{if(n) printf "  n_ranks=%d  max=%.2f GiB  min=%.2f GiB\n", n, mx/1048576, mn/1048576;
               else print "  (no samples)"}'
  echo "--- $TAG stderr (non-empty per-rank files) ---"
  for f in $RANKD/*.err; do
    [ -s "$f" ] || continue
    echo "  == $(basename $f) ($(wc -l < $f) lines) =="
    tail -25 "$f" | sed 's/^/    /'
  done
  echo "--- $TAG banners ---"
  grep -aE "libfabric provider|collectives implementation|\[stage\]|Backend:|device mesh|SlabIO|slab_io|Traceback|Error|ERROR|REFUS|Out of memory|RESOURCE_EXHAUSTED|zeta reuse|REUSING|bispinor\]|library_provenance|\[ffi\]" \
       "$LOG" 2>/dev/null | head -40
  echo "--- $TAG end ($TAG rc=$rc wall=${wall}s) ---"
  rm -f "$INNER"
  return $rc
}
