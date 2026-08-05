#!/bin/bash -l
# Run all 8 GW configs (4 dirs x 2 allocator variants).
# Re-runs any config whose gw_{variant}.out doesn't end with "exiting cleanly"
# (the LORRAX_EXIT_AFTER_ZETA marker).
set -u

cd "$(dirname "${BASH_SOURCE[0]}")"

for d in 3x3x3_nb100 3x3x3_nb200 4x4x4_nb100 4x4x4_nb200; do
  for v in platform_false bfc_pre95; do
    OUT="${d}/gw_${v}.out"
    SUCCESS_TAIL=0
    if [ -f "$OUT" ]; then
      if tail -n 5 "$OUT" 2>/dev/null | grep -q "exiting cleanly after fit_zeta"; then
        SUCCESS_TAIL=1
      fi
    fi
    if [ "$SUCCESS_TAIL" = "1" ]; then
      echo "[$d/$v] skip (already complete)"
      continue
    fi
    echo "[$d/$v] running..."
    rm -rf "${d}/tmp"
    ./_run_gw.sh "$d" "$v" > "gw_${d}_${v}.log" 2>&1
    EC=$?
    if [ "$EC" != "0" ]; then
      echo "[$d/$v] WARN: exit=$EC"
    else
      echo "[$d/$v] done"
    fi
  done
done
echo "ALL_GW_DONE" > gw_all_done.flag
echo "ALL_GW_DONE"
