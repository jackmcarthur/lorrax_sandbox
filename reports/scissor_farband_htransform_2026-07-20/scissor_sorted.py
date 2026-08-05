"""Disambiguate: is the jagged scissor(k,n) a QP-reordering/labeling artifact
(fixed by per-k energy-sorting, a la gw.scissor sort-and-pair) or genuine
per-(k,n) Sigma noise? Also locate the eqp1 Z-factor blowups.
"""
import os
import numpy as np

RUN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_bse_figures_2026-07-20/02_lorrax_gw_d3h_16gpu"
OUT = "/pscratch/sd/j/jackm/lorrax_sandbox/reports/scissor_farband_htransform_2026-07-20"
d = np.load(os.path.join(OUT, "scissor_data.npz"), allow_pickle=True)
Edft, Eqp0, Eqp1 = d["Edft"], d["Eqp0"], d["Eqp1"]   # (nk, nb)
Erel, EF = d["Erel"], float(d["EF"])
idx_ij = d["idx_ij"]
nk, nb = Edft.shape
kg = 6

# k adjacency
pos2k = {tuple(ij): k for k, ij in enumerate(idx_ij)}
edges = []
for k, (i, j) in enumerate(idx_ij):
    for di, dj in ((1, 0), (0, 1)):
        edges.append((k, pos2k[((i+di) % kg, (j+dj) % kg)]))
edges = np.array(edges)


def nn_jump(field_kn):
    return np.abs(field_kn[edges[:, 0], :] - field_kn[edges[:, 1], :]).max(axis=0)


# ---- LABELED scissor (as-is, DFT-band-label n) ----
sc0_lab = Eqp0 - Edft
sc1_lab = Eqp1 - Edft

# ---- SORTED scissor: sort each k's Edft and Eqp independently, pair by rank ----
Edft_s = np.sort(Edft, axis=1)
Eqp0_s = np.sort(Eqp0, axis=1)
Eqp1_s = np.sort(Eqp1, axis=1)
sc0_sorted = Eqp0_s - Edft_s
sc1_sorted = Eqp1_s - Edft_s

print("=== NN-jump smoothness: LABELED vs SORTED scissor (max over adjacent k) ===")
print(f"{'metric':<22}{'labeled meV':>14}{'sorted meV':>14}")
for name, lab, srt in [
    ("kink0 mean (all b)", nn_jump(sc0_lab).mean(), nn_jump(sc0_sorted).mean()),
    ("kink0 median (all b)", np.median(nn_jump(sc0_lab)), np.median(nn_jump(sc0_sorted))),
    ("kink1 mean (all b)", nn_jump(sc1_lab).mean(), nn_jump(sc1_sorted).mean()),
    ("kink1 median (all b)", np.median(nn_jump(sc1_lab)), np.median(nn_jump(sc1_sorted))),
]:
    print(f"{name:<22}{lab*1e3:>14.1f}{srt*1e3:>14.1f}")

# focus on valence (sorted rank 0..25) and near-gap sorted ranks
print("\n=== sorted-rank scissor smoothness for top-valence & low-cond ranks ===")
k0 = nn_jump(sc0_sorted)
k1 = nn_jump(sc1_sorted)
for r in [0, 5, 11, 12, 20, 24, 25, 26, 27, 30, 40, 45, 46, 60, 99]:
    reln = Erel[:, :].mean(axis=0)  # not rank-based; report sorted Edft rel
    er = (Edft_s[:, r] - EF)
    print(f"  rank {r:>3}: Erel[{er.min():+7.2f},{er.max():+7.2f}]  "
          f"kink0={k0[r]*1e3:>8.1f} meV  kink1={k1[r]*1e3:>9.1f} meV")

# ---- the VBM as sorted top-valence across k ----
# valence = 26 lowest DFT states (ranks 0..25); VBM = rank 25 sorted
vbm_dft = Edft_s[:, 25]
vbm_qp0 = Eqp0_s[:, 25]
vbm_qp1 = Eqp1_s[:, 25]
print("\n=== VBM (sorted valence rank 25) across k ===")
print(f"  Edft   range [{vbm_dft.min():.3f},{vbm_dft.max():.3f}]  NNjump={nn_jump(vbm_dft[:,None])[0]*1e3:.1f} meV")
print(f"  Eqp0   range [{vbm_qp0.min():.3f},{vbm_qp0.max():.3f}]  NNjump={nn_jump(vbm_qp0[:,None])[0]*1e3:.1f} meV")
print(f"  Eqp1   range [{vbm_qp1.min():.3f},{vbm_qp1.max():.3f}]  NNjump={nn_jump(vbm_qp1[:,None])[0]*1e3:.1f} meV")
print(f"  scissor0(VBM sorted) range [{(vbm_qp0-vbm_dft).min():+.3f},{(vbm_qp0-vbm_dft).max():+.3f}]")

# ---- locate eqp1 blowups vs eqp0 (Z-factor poles) ----
zblow = np.abs(Eqp1 - Eqp0)                       # |eqp1 - eqp0| = |Z-1|*|Sig-Vxc|
print("\n=== largest |eqp1 - eqp0| (Z-factor amplification) ===")
flat = np.argsort(zblow.ravel())[::-1][:12]
for f in flat:
    k, b = divmod(f, nb)
    print(f"  k={k:>2} band={b:>2} Erel={Erel[k,b]:+7.2f}  Edft={Edft[k,b]:8.3f}  "
          f"eqp0={Eqp0[k,b]:9.3f}  eqp1={Eqp1[k,b]:10.3f}  |dz|={zblow[k,b]:9.2f}")

# how many (k,b) have |eqp1-eqp0| > 5 eV, and where (in/out window)?
big = zblow > 5.0
print(f"\n|eqp1-eqp0|>5eV: {big.sum()} of {nk*nb} ({big.mean()*100:.1f}%); "
      f"of these in-window(|Erel|<10): {(big & (np.abs(Erel)<10)).sum()}")
