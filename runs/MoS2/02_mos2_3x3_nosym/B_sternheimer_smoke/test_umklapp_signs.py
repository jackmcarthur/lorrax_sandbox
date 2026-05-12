"""Try each G_wrap sign convention on q=(0, 2/3) to find the right one."""
import sys, subprocess
# Just toggle the sign by editing the driver and running.
# Configs to test:
#   (1) Both source and density use +G_wrap (current)
#   (2) Source only
#   (3) Density only
#   (4) Source only with -G_wrap
#   (5) Density only with -G_wrap
# Expected: χ(q=(0, 2/3), 0) should be REAL and NEGATIVE (not 0.0032 - 0.0093j).
# Print results for each.
print("Manual: try toggles; report which gives q=2 REAL NEGATIVE")
