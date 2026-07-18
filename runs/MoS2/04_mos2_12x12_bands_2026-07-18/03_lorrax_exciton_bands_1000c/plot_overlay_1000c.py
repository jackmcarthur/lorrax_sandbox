"""Overlay: 640-centroid vs 1000-centroid exciton bands (identical 12x12
restart physics, identical 40-pt GMKG path, identical driver window —
ONLY the ISDF centroid basis differs).

Owner question: does a more converged ISDF basis smooth the bands?
Annotation contract (from 05_htransform_spbands): the iQ 6/9/16-17 dips
are htransform (24,32)-window cache artifacts (D_min A/B, max 317 meV at
iQ 9) — they are NOT ISDF-basis error, so they should survive the basis
upgrade; differences elsewhere are the true basis effect."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C640, C1000, C_MARK = "0.55", "#2a78d6", "#eda100"
OLD = "../01_lorrax_exciton_bands/exciton_bands_12x12_GMKG.dat"
NEW = "exciton_bands_1000c.dat"
OUT = "exciton_bands_640c_vs_1000c_GMKG.png"
ARTIFACT_IQ = [6, 9, 16, 17]        # window-cache artifact rows (task-1 A/B)


def load(path):
    rows, nodes = [], None
    with open(path, encoding="utf8") as fh:
        for ln in fh:
            if ln.startswith("# nodes:"):
                nodes = [(int(t.split(":")[0]), t.split(":")[1])
                         for t in ln.split()[2:]]
            if ln.startswith("#") or not ln.strip():
                continue
            t = ln.split()
            if t[5] == "interp":
                rows.append((int(t[0]), float(t[1]),
                             [float(x) for x in t[6:]]))
    rows.sort()
    return (np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]), nodes)


s_o, E_o, nodes = load(OLD)
s_n, E_n, nodes_n = load(NEW)
assert np.allclose(s_o, s_n, atol=1e-5)
dE = np.abs(E_n - E_o) * 1e3
print("per-state |dE(1000c-640c)| (meV): "
      f"median {np.median(dE):.2f}, mean {dE.mean():.2f}, max {dE.max():.2f} "
      f"@iQ {int(dE.max(axis=1).argmax())}")
mask = np.ones(len(s_o), bool)
mask[ARTIFACT_IQ] = False
print(f"  excluding artifact rows {ARTIFACT_IQ}: median "
      f"{np.median(dE[mask]):.2f}, max {dE[mask].max():.2f} meV")
print(f"  on artifact rows only: median {np.median(dE[~mask]):.2f}, max "
      f"{dE[~mask].max():.2f} meV")

fig, ax = plt.subplots(figsize=(7.4, 5.0))
for i in ARTIFACT_IQ:
    lo = 0.5 * (s_n[i - 1] + s_n[i]) if i > 0 else s_n[i]
    hi = 0.5 * (s_n[i] + s_n[i + 1]) if i + 1 < len(s_n) else s_n[i]
    ax.axvspan(lo, hi, color=C_MARK, alpha=0.12,
               label="window-cache artifact rows (iQ 6/9/16-17)"
               if i == ARTIFACT_IQ[0] else None)
for b in range(E_o.shape[1]):
    ax.plot(s_o, E_o[:, b], lw=1.0, ls="--", color=C640,
            label="640 centroids" if b == 0 else None)
for b in range(E_n.shape[1]):
    ax.plot(s_n, E_n[:, b], lw=1.4, color=C1000,
            label="1000 centroids" if b == 0 else None)
node_x = [s_n[i] for i, _ in nodes_n]
for xv in node_x:
    ax.axvline(xv, color="k", lw=0.6, alpha=0.25)
ax.set_xticks(node_x, [l for _, l in nodes_n])
ax.set_xlim(s_n[0], s_n[-1])
ax.set_ylabel("$E_S(Q)$ (eV)")
ax.set_title("MoS$_2$ exciton bands (TDA, 12$\\times$12): "
             "640 vs 1000 ISDF centroids", fontsize=11)
ax.legend(loc="lower right", fontsize="small", framealpha=0.9)
ax.grid(axis="y", ls="--", lw=0.4, alpha=0.3)
fig.text(0.5, -0.05,
         "Identical restart physics, path, and htransform window — only the "
         "ISDF centroid basis differs.\nShaded rows: htransform "
         "window-cache artifacts (05_htransform_spbands $D_\\min$ A/B) — "
         "structure shared by both curves there is NOT basis error.\n"
         "Differences outside the shading are the true ISDF-basis effect.",
         ha="center", fontsize=8)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(f"Wrote {OUT}")
