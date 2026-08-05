#!/usr/bin/env python3
"""Plot orbital-moment vs #bands convergence from orbmag_nb*.out files.
Usage: python3 plot_orbmag.py <label> <run_dir> [<label2> <run_dir2> ...]"""
import sys, re, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

args = sys.argv[1:]
fig, ax = plt.subplots(figsize=(6.2, 4.4))
for i in range(0, len(args), 2):
    label, d = args[i], args[i+1]
    pts = []
    for f in sorted(glob.glob(os.path.join(d, "orbmag_nb*.out"))):
        nb = int(re.search(r"nb(\d+)", f).group(1))
        txt = open(f).read()
        # m_orb at midgap from the mu-scan block
        m = re.search(r"midgap.*?m_z = ([-+0-9.]+)", txt, re.S)
        if not m:  # fallback: the single m_z line
            m = re.search(r"orbital moment along spin axis: ([-+0-9.]+)", txt)
        if m: pts.append((nb, float(m.group(1))))
    if not pts:
        print(f"no data for {label} in {d}"); continue
    pts.sort()
    xs, ys = zip(*pts)
    ax.plot(xs, ys, "o-", label=f"{label}  (m_orb @ {xs[-1]}b = {ys[-1]:.3f} μB)")
    print(f"{label}: " + ", ".join(f"{x}:{y:.3f}" for x,y in pts))
ax.set_xlabel("# bands in orbital-moment sum"); ax.set_ylabel(r"$m_{orb}$ along spin axis (μB/cell)")
ax.set_title("Orbital magnetic moment vs band count (6×6, midgap)")
ax.axhline(0, color="0.8", lw=0.7); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("orbmag_convergence.png", dpi=160)
print("wrote orbmag_convergence.png")
