"""Build the BSE-on-top centroid-convergence plot for the
B_centroid_sweep_2026-04-27 run.  Saves bse_centroid_sweep.png."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWEEP = Path("/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/B_centroid_sweep_2026-04-27")
REPORT = Path("/pscratch/sd/j/jackm/lorrax_sandbox/reports/centroid_sweep_2026-04-27")


_eig_re = re.compile(r"Lowest \d+ eigenvalues \(eV\):\s*\[([^\]]+)\]")


def parse_eigs(bse_out: Path) -> np.ndarray:
    text = bse_out.read_text()
    # bse_out has 4 copies (one per rank); take the first match.
    m = _eig_re.search(text)
    if m is None:
        raise RuntimeError(f"no eigenvalues in {bse_out}")
    # The matched group only contains the first line; eigenvalues span
    # multiple lines.  Simpler: re-search and join everything between
    # '[' and ']' after the header.
    idx = text.find("Lowest 12 eigenvalues (eV):")
    if idx < 0:
        idx = text.find("Lowest 10 eigenvalues (eV):")
    start = text.find("[", idx)
    end = text.find("]", start)
    raw = text[start + 1 : end]
    raw = raw.replace("\n", " ")
    return np.array([float(x) for x in raw.split()])


def main():
    Ns = [336, 480, 624, 768]
    eigs = {N: parse_eigs(SWEEP / f"N_{N}" / "bse.out") for N in Ns}
    for N, e in eigs.items():
        print(f"  N={N}: lowest 5 (eV) = {e[:5].tolist()}")

    # GW gap from sweep_results.json (the GW Σ-shifted gap at Γ used as scissors)
    sweep_results = json.loads((SWEEP / "sweep_results.json").read_text())
    gw_gap = {N: sweep_results[str(N)]["sig_gap_gamma"] for N in Ns}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel A: lowest 5 exciton peaks vs N_mu, deltas relative to N=480
    ax = axes[0]
    ref = eigs[480]
    colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.85, 5))
    for ii in range(5):
        deltas_meV = np.array([(eigs[N][ii] - ref[ii]) * 1000.0 for N in Ns])
        ax.plot(Ns, deltas_meV, "o-", color=colors[ii], label=f"E$_{{{ii+1}}}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"N$_\mu$ (centroids)")
    ax.set_ylabel(r"E$_{ex}$(N) - E$_{ex}$(480) [meV]")
    ax.set_title("BSE exciton peaks vs N$_\\mu$ (relative to 480)")
    ax.set_xticks(Ns)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel B: lowest exciton energy and GW gap at Γ overlaid (absolute eV)
    ax = axes[1]
    e1 = np.array([eigs[N][0] for N in Ns])
    g_gap = np.array([gw_gap[N] for N in Ns])
    # Add the COHSEX correction back to absolute scale (DFT direct gap was
    # ~0.05 eV so we're looking at correction effects only).  Since BSE
    # eigenvalues already include scissors=GW Σ-shift, plot as-is.
    ax.plot(Ns, e1, "o-", color="C0", label="lowest exciton (BSE+scissors)")
    ax.plot(Ns, g_gap, "s--", color="C3", label="GW Σ-shift at Γ (sweep_results)")
    ax.set_xlabel(r"N$_\mu$ (centroids)")
    ax.set_ylabel(r"E [eV]")
    ax.set_title("Lowest exciton vs GW Σ-shift")
    ax.set_xticks(Ns)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out = REPORT / "bse_centroid_sweep.png"
    fig.savefig(out, dpi=140)
    print(f"  Saved {out}")

    # also dump JSON for the report
    out_json = REPORT / "bse_eigenvalues.json"
    out_json.write_text(json.dumps({str(N): eigs[N].tolist() for N in Ns}, indent=2))
    print(f"  Saved {out_json}")


if __name__ == "__main__":
    main()
