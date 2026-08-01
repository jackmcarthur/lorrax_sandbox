# ===========================================================================
# run_leg.sh -- one instrumented driver launch.  Source from an sbatch body
# AFTER env_common.sh, with $RUN set to the run directory.
#
#   run_leg <tag> <python -m module> [argv...]
#
# It writes a container-side inner script that sources harness/inner_common.sh
# (the certified gw_dev.sbatch env block) and calls lorrax_run, then srun's it
# over the whole allocation at 2 ranks/node.  Per leg it emits:
#   * $R/logs/<tag>.<jobid>.log         full driver stdout/stderr
#   * $R/logs/hwm_<tag>.<jobid>/        per-rank /proc VmHWM samples
#   * rc, wall, transport banners, and the driver's own "--- Timing ---" table
#
# WHY NOT config/frontera/templates/gw_dev.sbatch for the GW leg: the template
# `exec`s python, which leaves no process to sample /proc/<pid>/status.  The
# scorecard (AY, L9828) is explicit that sacct MaxRSS undersamples brief peaks
# by up to 3.3x (5.6 GiB reported vs 18.6 GiB real) and that the per-rank
# VmHWM sampler is the authoritative instrument.  inner_common.sh is a
# verbatim transcription of the template's env block; the only change is
# child-instead-of-exec.  sacct MaxRSS is still collected as a cross-check.
# ===========================================================================
run_leg () {
  local TAG=$1 MOD=$2; shift 2
  local LOG=$R/logs/${TAG}.$SLURM_JOB_ID.log
  local HWM=$R/logs/hwm_${TAG}.$SLURM_JOB_ID
  rm -rf "$HWM"; mkdir -p "$HWM"
  local INNER=$RUN/inner_$TAG.$SLURM_JOB_ID.sh
  cat > "$INNER" <<EOS
. $R/harness/inner_common.sh
export LORRAX_HWM_DIR=$HWM
cd $RUN
lorrax_run $MOD $*
EOS
  echo "##### [$TAG] $MOD $* -- $(date) #####"
  local t0=$(date +%s)
  srun --mpi=pmi2 -N "$SLURM_NNODES" --ntasks-per-node=2 -n "$SLURM_NTASKS" \
      apptainer exec --bind "$BIND" "$LORRAX_SIF" \
      bash "$INNER" > "$LOG" 2>&1
  local rc=$?
  local wall=$(( $(date +%s) - t0 ))
  echo "[$TAG rc=$rc wall=${wall}s log=$LOG]"
  echo "--- $TAG per-rank VmHWM ---"
  # one value per RANK = max over that rank's samples (VmHWM is monotone)
  for f in $HWM/rank_*.hwm; do
    [ -s "$f" ] || continue
    awk '{for(i=1;i<=NF;i++) if($i=="VmHWM:" && $(i+1)>m) m=$(i+1)} END{if(m) print m}' "$f"
  done 2>/dev/null | sort -nr \
    | awk 'NR<=3{printf "  rank-peak %d kB (%.2f GiB)\n", $1, $1/1048576}
           {s+=$1; n++; if(NR==1)mx=$1; mn=$1}
           END{if(n) printf "  n_ranks=%d  max=%.2f GiB  min=%.2f GiB  mean=%.2f GiB\n", n, mx/1048576, mn/1048576, s/n/1048576;
               else print "  (no samples -- driver exited before the first 5 s tick)"}'
  echo "--- $TAG banners ---"
  grep -aE "libfabric provider|cpu_collectives|collectives implementation|\[stage\]|\[mpi_transport_env\]|Backend:|Mesh:|device mesh|TIER resolved|tier:|SlabIO|slab_io|Traceback|Error|ERROR|REFUS|Out of memory|RESOURCE_EXHAUSTED" \
       "$LOG" 2>/dev/null | head -30
  echo "--- $TAG timing table ---"
  sed -n '/^--- .*[Tt]iming.*---/,$p' "$LOG" 2>/dev/null | head -70
  echo "--- $TAG end ($TAG rc=$rc wall=${wall}s) ---"
  rm -f "$INNER"
  return $rc
}
