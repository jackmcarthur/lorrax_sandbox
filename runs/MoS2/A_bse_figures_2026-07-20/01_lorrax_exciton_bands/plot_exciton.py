"""Publication figure: MoS2 exciton bandstructure E_S(Q) along Gamma-M-K-Gamma.

Reads the exciton_bands driver output (mos2_exciton_bands.dat):
  # iQ  s_path  Qx Qy Qz  mode  E_1..E_neig (eV)
Plots the lowest few exciton branches vs the path, annotates the Q=0 binding
energy relative to the (DFT) direct gap the BSE is built on.

Usage: python plot_exciton.py <dat> <out.png> [dft_direct_gap_eV]
"""
import sys, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

DAT = sys.argv[1] if len(sys.argv) > 1 else "mos2_exciton_bands.dat"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../figures/mos2_exciton_bandstructure.png"
DFT_GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 1.732   # DFT direct gap (eV)

# ---- parse ----
rows = []
with open(DAT) as fh:
    for line in fh:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        # iQ s_path Qx Qy Qz mode E1..
        iQ = int(p[0]); spath = float(p[1]); Q = np.array([float(p[2]), float(p[3]), float(p[4])])
        evs = np.array([float(x) for x in p[6:]])
        rows.append((iQ, spath, Q, evs))
rows.sort(key=lambda r: r[0])
s_path = np.array([r[1] for r in rows])
Qs = np.array([r[2] for r in rows])
E = np.array([r[3] for r in rows])            # (nQ, n_eig)
nQ, neig = E.shape
print(f"[exciton] nQ={nQ} n_eig={neig} s_path[0,-1]=({s_path[0]:.3f},{s_path[-1]:.3f})")

# ---- high-sym nodes: Gamma, M(0,1/2), K(1/3,1/3), Gamma ----
nodes_frac = [np.array([0,0,0.]), np.array([0,0.5,0.]), np.array([1/3,1/3,0.]), np.array([0,0,0.])]
def wrap(q): return (q + 0.5) % 1.0 - 0.5
node_idx = []
seen = -1
for tgt in nodes_frac:
    d = np.linalg.norm(wrap(Qs - tgt), axis=1)
    cand = [i for i in np.argsort(d) if i > seen]
    j = cand[0] if cand else int(np.argmin(d))
    node_idx.append(j); seen = j
node_x = [s_path[i] for i in node_idx]

# ---- binding energy at Gamma ----
iG = node_idx[0]
E1_gamma = E[iG, 0]
binding = DFT_GAP - E1_gamma
print(f"[exciton] E_1(Gamma)={E1_gamma:.4f} eV  DFT_direct_gap={DFT_GAP:.3f}  binding={binding*1e3:.0f} meV")
print(f"[exciton] E_1 min over path = {E[:,0].min():.4f} at s={s_path[int(np.argmin(E[:,0]))]:.3f}")
np.savez(OUT.replace(".png", ".npz"), s_path=s_path, E=E, node_idx=node_idx,
         E1_gamma=E1_gamma, binding=binding, dft_gap=DFT_GAP)

# ---- plot ----
plt.rcParams.update({"font.size": 12, "axes.linewidth": 1.0, "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.4, 6.2), dpi=200)
nshow = min(6, neig)
cmap = plt.cm.viridis(np.linspace(0.12, 0.85, nshow))
for b in range(nshow):
    ax.plot(s_path, E[:, b], color=cmap[b], lw=1.7 if b == 0 else 1.2,
            label=(f"S$_{b+1}$" if b < 4 else None), zorder=3)
# DFT direct gap reference (free-pair onset the BSE is built on)
ax.axhline(DFT_GAP, color="0.45", lw=1.1, ls="--", zorder=1,
           label="DFT direct gap (free e-h)")
for x in node_x:
    ax.axvline(x, color="0.85", lw=0.7, zorder=0)
ax.set_xticks(node_x); ax.set_xticklabels(["$\\Gamma$", "M", "K", "$\\Gamma$"])
ax.set_xlim(s_path[0], s_path[-1])
ax.set_ylabel("Exciton energy E$_S$(Q) (eV)")
ax.set_title("MoS$_2$ monolayer — BSE exciton bandstructure\n"
             "12$\\times$12 via bse_k_grid (6$\\times$6 GW $\\to$ 12$\\times$12), 8v8c, coarse-W pad",
             fontsize=11.5)
# binding annotation at Gamma
ax.annotate("", xy=(node_x[0], E1_gamma), xytext=(node_x[0], DFT_GAP),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
ax.text(node_x[0] + 0.01*(s_path[-1]-s_path[0]), 0.5*(E1_gamma+DFT_GAP),
        f"E$_b$ = {binding*1e3:.0f} meV", color="#c0392b", fontsize=10.5, va="center")
ax.text(0.02, 0.03, f"E$_1(\\Gamma)$ = {E1_gamma:.3f} eV", transform=ax.transAxes,
        fontsize=10.5, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
ax.legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=2)
fig.tight_layout(); fig.savefig(OUT)
print(f"SAVED {OUT}")
