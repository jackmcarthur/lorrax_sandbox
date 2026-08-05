#!/bin/bash
# Check out COMMIT in the bisect worktree, guarantee the FFI .so is present
# (build/ is gitignored so it survives checkout; re-seed from figures worktree
# if a checkout wiped it), then run the 1496 orbit-closed fixture.
#   usage: JID=<jid> ./bisect_one.sh <commit> <label> [FFB=0] [CENT=centroids_frac_1496.txt]
set -uo pipefail
COMMIT="${1:?commit}"; LABEL="${2:?label}"; FFB="${3:-0}"; CENT="${4:-centroids_frac_1496.txt}"
JID="${JID:?}"
WT=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_gnppm_bisect
BR=/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/_gnppm_bisect
SEED_SO=/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/lorrax_A_figures/src/ffi/common/cpp/build/liblorrax_ffi.so
BUILD=$WT/src/ffi/common/cpp/build

cd "$WT"
git checkout -q --detach "$COMMIT" 2>&1 | tail -2
echo "=== CHECKED OUT $(git rev-parse --short HEAD) : $(git log -1 --pretty=%s | cut -c1-70)"
mkdir -p "$BUILD"
[ -f "$BUILD/liblorrax_ffi.so" ] || cp -f "$SEED_SO" "$BUILD/liblorrax_ffi.so"

JID=$JID SRC=$WT/src LABEL="$LABEL" CENT="$CENT" FFB="$FFB" bash "$BR/run_bisect.sh"
