#!/bin/bash
B=/sys/fs/cgroup/system.slice/slurmstepd.scope/job_$SLURM_JOB_ID
echo "== $(hostname)"
for f in memory.max memory.high memory.current memory.swap.max; do
  [ -r "$B/$f" ] && echo "  job/$f = $(cat $B/$f)"
done
[ -r "$B/memory.events" ] && echo "  job/memory.events: $(tr '\n' ' ' < $B/memory.events)"
