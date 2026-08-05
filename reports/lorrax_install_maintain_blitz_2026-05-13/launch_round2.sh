#!/usr/bin/env bash
# Launch 4-agent round-2 (reconciliation) team in a separate tmux session.
# Independent of the round-1 session `lorrax_install_team`.
#
# Attach with:  tmux attach -t lorrax_install_round2
# Kill with:    tmux kill-session -t lorrax_install_round2

set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
REPORT=$SBOX/reports/lorrax_install_maintain_blitz_2026-05-13
PROMPTS=$REPORT/prompts
SESSION=lorrax_install_round2

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Kill it first with: tmux kill-session -t $SESSION"
  exit 1
fi

# Sanity check: all four round-1 drafts and four round-2 prompts must exist.
for n in 1 2 3 4; do
  if [ ! -f "$REPORT/agent_${n}.md" ]; then
    echo "Missing round-1 draft: agent_${n}.md — round 2 cannot start."
    exit 2
  fi
  if [ ! -f "$PROMPTS/round2_agent_${n}.md" ]; then
    echo "Missing round-2 prompt: round2_agent_${n}.md"
    exit 2
  fi
done

# Create detached session with pane 0
tmux new-session -d -s "$SESSION" -x 240 -y 80 -c "$SBOX"

# Split into 4 panes total; tile them 2x2
tmux split-window -t "$SESSION":0 -c "$SBOX"
tmux split-window -t "$SESSION":0 -c "$SBOX"
tmux split-window -t "$SESSION":0 -c "$SBOX"
tmux select-layout -t "$SESSION":0 tiled

# Visible pane titles
tmux set-option -t "$SESSION" pane-border-status top
tmux set-option -t "$SESSION" pane-border-format " #{pane_index}: #{pane_title} "

PANE_NAMES=(
  "R2 Agent 1: Build/FFI"
  "R2 Agent 2: MPI/Shifter"
  "R2 Agent 3: Runtime/Env"
  "R2 Agent 4: Docs/Synthesis"
)
for i in 0 1 2 3; do
  tmux select-pane -t "$SESSION":0.$i -T "${PANE_NAMES[$i]}"
done

# Launch claude in each pane.
for i in 0 1 2 3; do
  agent=$((i + 1))
  cmd="claude --permission-mode bypassPermissions --model opus --effort high --name 'R2 Agent ${agent}' 'Your round-2 assignment is at ${PROMPTS}/round2_agent_${agent}.md - read it and follow the instructions in it exactly.'"
  tmux send-keys -t "$SESSION":0.$i "$cmd" Enter
done

echo
echo "Launched round-2 4-pane session '$SESSION'."
echo "Round-1 session '$SBOX' is unaffected (still alive if you left it running)."
echo
echo "Attach with:  tmux attach -t $SESSION"
echo "Detach from inside tmux with: Ctrl-b d"
echo "Kill all panes with:  tmux kill-session -t $SESSION"
