"""Plot QP-gap (Σ_xc-only contribution) and timing vs N_μ for Si 4×4×4 COHSEX."""
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_centroid_sweep_2026-04-27")
RESULTS = json.loads((ROOT / "sweep_results.json").read_text())

# Per-N timings (sec) — pulled from gw.out
TIMING = {
    "336": dict(load=1.305, zeta=8.112, vq=3.913, chiw=1.216, sigma=1.796, total=16.587, kmeans=13.8),
    "480": dict(load=1.337, zeta=9.952, vq=4.471, chiw=1.216, sigma=3.291, total=20.499, kmeans=12.8),
    "624": dict(load=1.330, zeta=11.581, vq=5.051, chiw=1.390, sigma=3.023, total=22.613, kmeans=12.5),
    "768": dict(load=1.238, zeta=13.491, vq=5.650, chiw=1.432, sigma=3.181, total=25.212, kmeans=12.7),
}

Ns = sorted(int(N) for N in RESULTS.keys())
gap_gamma = [RESULTS[str(N)]["sig_gap_gamma"] for N in Ns]
gap_x = [RESULTS[str(N)]["sig_gap_x"] for N in Ns]
gap_l = [RESULTS[str(N)]["sig_gap_l"] for N in Ns]

zeta_t = [TIMING[str(N)]["zeta"] for N in Ns]
vq_t = [TIMING[str(N)]["vq"] for N in Ns]
chiw_t = [TIMING[str(N)]["chiw"] for N in Ns]
sig_t = [TIMING[str(N)]["sigma"] for N in Ns]
total_t = [TIMING[str(N)]["total"] for N in Ns]
kmeans_t = [TIMING[str(N)]["kmeans"] for N in Ns]

fig, (ax_gap, ax_time) = plt.subplots(1, 2, figsize=(11, 4.4))

# Plot 1: Σ_xc gap shift relative to N=480
ref_g = RESULTS["480"]["sig_gap_gamma"]
ref_x = RESULTS["480"]["sig_gap_x"]
ref_l = RESULTS["480"]["sig_gap_l"]

ax2 = ax_gap.twinx()  # secondary axis with absolute Σ-gap values

ax_gap.axhline(0, color="0.7", lw=0.5)
ax_gap.plot(Ns, [(g - ref_g) * 1000 for g in gap_gamma], "o-", label=r"Γ (Σ-gap shift)")
ax_gap.plot(Ns, [(g - ref_x) * 1000 for g in gap_x], "s-", label=r"X (Σ-gap shift)")
ax_gap.plot(Ns, [(g - ref_l) * 1000 for g in gap_l], "^-", label=r"L (Σ-gap shift)")
ax_gap.axvline(480, color="C3", ls="--", alpha=0.5, label="baseline (480)")
ax_gap.set_xlabel(r"$N_\mu$ (number of ISDF centroids)")
ax_gap.set_ylabel(r"$\Delta(\Sigma^{\rm CBM}-\Sigma^{\rm VBM})$ vs $N_\mu=480$ (meV)")
ax_gap.set_title(
    "Si 4×4×4 COHSEX: gap-shift convergence\n(Σ_xc only; V_xc independent of $N_\\mu$)"
)
ax_gap.legend(fontsize=8)
ax_gap.grid(alpha=0.3)

# Plot 2: timings
ax_time.plot(Ns, total_t, "ko-", label="GW total", lw=2)
ax_time.plot(Ns, zeta_t, "C0o-", label="ζ-fit (ISDF)")
ax_time.plot(Ns, vq_t, "C1s-", label="V_q")
ax_time.plot(Ns, sig_t, "C2^-", label="Σ")
ax_time.plot(Ns, chiw_t, "C3v-", label="χ₀ + W")
ax_time.plot(Ns, kmeans_t, "C4d:", label="k-means (preproc)", alpha=0.6)
ax_time.set_xlabel(r"$N_\mu$ (number of ISDF centroids)")
ax_time.set_ylabel("Wall time (s)")
ax_time.set_title("Si 4×4×4 GW timing per stage\n(1 node × 4 A100, single-shot)")
ax_time.legend(fontsize=8, ncol=2)
ax_time.grid(alpha=0.3)
ax_time.axvline(480, color="C3", ls="--", alpha=0.5)

fig.tight_layout()
out = ROOT.parent.parent.parent / "reports/centroid_sweep_2026-04-27/centroid_sweep.png"
fig.savefig(out, dpi=140)
print(f"saved {out}")
