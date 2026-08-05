#!/usr/bin/env bash
# Launch 4-agent LORRAX install/maintain blitz team in a tmux session.
# Each pane runs an interactive claude with a self-contained prompt.
#
# Attach with:  tmux attach -t lorrax_install_team
# Kill with:    tmux kill-session -t lorrax_install_team

set -euo pipefail

SBOX=/pscratch/sd/j/jackm/lorrax_sandbox
REPORT=$SBOX/reports/lorrax_install_maintain_blitz_2026-05-13
PROMPTS=$REPORT/prompts
SESSION=lorrax_install_team

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Kill it first with: tmux kill-session -t $SESSION"
  exit 1
fi

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

# Pane order from `tmux list-panes` after a series of splits depends on
# the split direction; the tiled layout assigns logical indices 0..3
# left-to-right, top-to-bottom. We set titles using those indices.
PANE_NAMES=(
  "Agent 1: Build/FFI"
  "Agent 2: MPI/Shifter"
  "Agent 3: Runtime/Env"
  "Agent 4: Docs/Synthesis"
)
for i in 0 1 2 3; do
  tmux select-pane -t "$SESSION":0.$i -T "${PANE_NAMES[$i]}"
done

# Launch claude in each pane. Use bypassPermissions so they don't stall
# on tool prompts; the user is watching and can interrupt at any time.
# Pass a one-line seed pointing each agent at its own prompt file.
for i in 0 1 2 3; do
  agent=$((i + 1))
  cmd="claude --permission-mode bypassPermissions --model opus --effort high --name 'Agent ${agent}' 'Your assignment is at ${PROMPTS}/agent_${agent}.md - read it and follow the instructions in it exactly.'"
  tmux send-keys -t "$SESSION":0.$i "$cmd" Enter
done

echo
echo "Launched 4-pane session '$SESSION'."
echo "Attach with:  tmux attach -t $SESSION"
echo "Detach from inside tmux with: Ctrl-b d"
echo "Kill all panes with:  tmux kill-session -t $SESSION"
