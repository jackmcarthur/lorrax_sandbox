"""Scissor-shift post-analysis for Si 4x4x4 pseudobands G0W0.

Reads:
  - WFN.h5: E_DFT per (k, n)
  - sigma_mnk.h5: Sigma_c(omega, k, m, n) and Sigma_sx(k, m, n)

Computes:
  - Sigma_xc(E_DFT, k, n) via linear interpolation of the dynamic tensor at each
    band's DFT energy (diagonal only).  In-grid bands get the true interpolated
    value; out-of-grid bands get NaN (to be overwritten by scissor).
  - Scissor fit on (E_DFT - E_F, Re Sigma_xc(E_DFT)) restricted to in-grid
    bands, with separate valence / conduction lines.
  - Reconstructed Sigma_xc across the full bandrange (measured in-grid,
    scissor-extrapolated out-of-grid).
  - E_QP(k, n) = E_DFT(k, n) + Re Sigma_xc.
  - Plots: E_QP vs band index per k, raw-vs-scissor-replaced, band-resolved
    QP correction vs E_DFT with the fitted line overlaid.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/global/homes/j/jackm/software/lorrax_A/src")
from gw.scissor import fit_scissor  # noqa: E402

RYD2EV = 13.6056980659
SIGMA_OMEGA_MIN_EV = -5.0
SIGMA_OMEGA_MAX_EV = +5.0


def load_enk(wfn_path: str) -> np.ndarray:
    """Return (nk, nb) DFT eigenvalues in eV."""
    with h5py.File(wfn_path, "r") as f:
        # mf_header/kpoints/el: (nspin, nk, nb) in Ry
        el = np.asarray(f["mf_header/kpoints/el"], dtype=np.float64)
    # Use first spin channel (spinor wfns have nspin=1 at this layer).
    return el[0] * RYD2EV


def load_sigma_mnk(h5_path: str):
    """Return omega_ev (n_omega,), sigma_c_diag_omega (n_omega, nk, nb),
    sigma_sx_diag (nk, nb), all in eV."""
    with h5py.File(h5_path, "r") as f:
        omega_ev = np.asarray(f["omega_ev"], dtype=np.float64)
        sc = np.asarray(f["sigma_c_kij_ev"], dtype=np.complex128)
        ssx = np.asarray(f["sigma_sx_kij_ev"], dtype=np.complex128)
    n_omega, nk, nb, _ = sc.shape
    sc_diag = np.empty((n_omega, nk, nb), dtype=np.complex128)
    for iw in range(n_omega):
        sc_diag[iw] = np.diagonal(sc[iw], axis1=1, axis2=2)
    ssx_diag = np.diagonal(ssx, axis1=1, axis2=2)
    return omega_ev, sc_diag, ssx_diag


def interp_sigma_at_edft(
    sigma_c_diag_omega: np.ndarray,
    omega_ev: np.ndarray,
    edft_rel_ev: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of Sigma_c(omega) at each (k, n)'s E_DFT - E_F.

    Points outside [omega_ev[0], omega_ev[-1]] get np.nan (not clamped).
    """
    nk, nb = edft_rel_ev.shape
    out = np.full((nk, nb), np.nan + 1j * np.nan, dtype=np.complex128)
    inside = (edft_rel_ev >= omega_ev[0]) & (edft_rel_ev <= omega_ev[-1])
    for ik in range(nk):
        for ib in range(nb):
            if inside[ik, ib]:
                out[ik, ib] = complex(
                    np.interp(edft_rel_ev[ik, ib], omega_ev, np.real(sigma_c_diag_omega[:, ik, ib])),
                    np.interp(edft_rel_ev[ik, ib], omega_ev, np.imag(sigma_c_diag_omega[:, ik, ib])),
                )
    return out


def main():
    run_dir = os.path.dirname(os.path.abspath(__file__))
    enk_ev = load_enk(os.path.join(run_dir, "WFN.h5"))
    omega_ev, sigma_c_diag_omega, sigma_sx_diag = load_sigma_mnk(
        os.path.join(run_dir, "sigma_mnk.h5")
    )
    nk, nb = enk_ev.shape
    print(f"nk = {nk}, nb = {nb}, omega grid = {omega_ev[0]:.2f}..{omega_ev[-1]:.2f} eV "
          f"(step {omega_ev[1] - omega_ev[0]:.3f} eV, {omega_ev.size} pts)")

    # Fermi level (midgap, 8 valence) from DFT eigenvalues.
    n_val = 8
    vbm_ev = float(np.max(enk_ev[:, :n_val]))
    cbm_ev = float(np.min(enk_ev[:, n_val:]))
    ef_ev = 0.5 * (vbm_ev + cbm_ev)
    print(f"E_F(DFT, midgap) = {ef_ev:.4f} eV (VBM={vbm_ev:.4f}, CBM={cbm_ev:.4f})")

    edft_rel = enk_ev - ef_ev

    # Interpolate diagonal Sigma_c at each band's E_DFT; Sigma_x is freq-independent.
    sigma_c_at_dft = interp_sigma_at_edft(sigma_c_diag_omega, omega_ev, edft_rel)
    sigma_x_at_dft = np.broadcast_to(sigma_sx_diag, (nk, nb)).astype(np.complex128)
    sigma_xc_at_dft = sigma_c_at_dft + sigma_x_at_dft

    in_grid = (edft_rel >= omega_ev[0]) & (edft_rel <= omega_ev[-1])
    n_in = int(in_grid.sum())
    print(f"In-grid (k,n) points: {n_in}/{enk_ev.size}")

    occ = np.zeros_like(enk_ev, dtype=bool)
    occ[:, :n_val] = True

    # --- Scissor fit on the QP correction: delta E = Re Sigma_xc(E_DFT)
    # (approximates E_QP - E_DFT up to the DFT exchange-correlation potential,
    # which LORRAX does not currently subtract out — constant offset doesn't
    # change the slope or continuity of the out-of-grid extrapolation.)
    delta_e = np.real(sigma_xc_at_dft)
    fit = fit_scissor(
        edft_rel,
        np.where(in_grid, delta_e, 0.0),  # out-of-grid values ignored by fit_mask
        valence_mask_kn=occ,
        fit_mask_kn=in_grid,
    )
    print(f"Scissor: {fit.summary()}")

    # Apply scissor to out-of-grid bands.
    extrap = fit.predict(edft_rel, occ)
    delta_e_full = np.where(in_grid, delta_e, extrap)

    # E_QP = E_DFT + delta_e.  Both in eV, ABSOLUTE scale.
    eqp_ev = enk_ev + delta_e_full

    # ================================================================
    # Plots
    # ================================================================

    # Panel 1: fit itself — delta_e vs E_DFT_rel, split v/c + line overlays.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    ax = axes[0]
    x_v = edft_rel[occ & in_grid]
    y_v = delta_e[occ & in_grid]
    x_c = edft_rel[(~occ) & in_grid]
    y_c = delta_e[(~occ) & in_grid]
    ax.scatter(x_v, y_v, s=10, alpha=0.6, color='C0', label=f"valence in-grid (n={x_v.size})")
    ax.scatter(x_c, y_c, s=10, alpha=0.6, color='C3', label=f"conduction in-grid (n={x_c.size})")

    # Line overlays across the full bandrange.
    e_lo = float(np.min(edft_rel))
    e_hi = float(np.max(edft_rel))
    xs = np.linspace(e_lo, e_hi, 400)
    ax.plot(xs, fit.slope_v * xs + fit.intercept_v, '-', color='C0', lw=1.5,
            label=f"val fit: α={fit.slope_v:+.3f}, β={fit.intercept_v:+.3f} eV")
    ax.plot(xs, fit.slope_c * xs + fit.intercept_c, '-', color='C3', lw=1.5,
            label=f"cond fit: α={fit.slope_c:+.3f}, β={fit.intercept_c:+.3f} eV")
    ax.axvspan(omega_ev[0], omega_ev[-1], color='gray', alpha=0.08, label='Σ(ω) grid')
    ax.set_xlabel("E_DFT - E_F (eV)")
    ax.set_ylabel("ΔE = Re Σ_xc(E_DFT) (eV)")
    ax.set_title("Scissor fit: in-grid QP corrections + fitted affine law")
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.2)

    # Panel 2: E_QP vs band index per k, showing scissor-replaced points.
    ax = axes[1]
    # Sort bands by DFT energy for display; same band index per k.
    # Use k=0 only for clarity.
    ik = 0
    ib_range = np.arange(nb)
    ax.plot(ib_range, enk_ev[ik], '-o', color='gray', ms=3, lw=0.8, label="E_DFT")
    ax.plot(ib_range[in_grid[ik]], eqp_ev[ik, in_grid[ik]], 'o', color='C2', ms=6, label="E_QP (in-grid)")
    ax.plot(ib_range[~in_grid[ik]], eqp_ev[ik, ~in_grid[ik]], 'x', color='C1', ms=6, label="E_QP (scissor)")
    ax.axhline(ef_ev, color='k', ls='--', lw=0.6)
    ax.set_xlabel("Band index (k=0)")
    ax.set_ylabel("Energy (eV)")
    ax.set_title("QP bandstructure at k=0 (in-grid vs scissor-extrapolated)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    out_png = os.path.join(run_dir, "scissor_analysis.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")

    # Panel 3: continuity view — E_QP vs E_DFT over all (k, n), colored by in/out of grid.
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.scatter(enk_ev[in_grid], eqp_ev[in_grid], s=8, alpha=0.5, color='C2', label='in-grid')
    ax.scatter(enk_ev[~in_grid], eqp_ev[~in_grid], s=8, alpha=0.5, color='C1', label='scissor')
    mn = float(min(enk_ev.min(), eqp_ev.min()))
    mx = float(max(enk_ev.max(), eqp_ev.max()))
    ax.plot([mn, mx], [mn, mx], 'k--', lw=0.7, alpha=0.5)
    ax.set_xlabel("E_DFT (eV, absolute)")
    ax.set_ylabel("E_QP (eV, absolute)")
    ax.set_title("E_QP vs E_DFT — check continuity across scissor boundary")
    ax.legend()
    ax.grid(alpha=0.2)
    out_png2 = os.path.join(run_dir, "scissor_continuity.png")
    fig.savefig(out_png2, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png2}")

    # Also dump a numpy file with the bands for later.
    np.savez(
        os.path.join(run_dir, "scissor_bands.npz"),
        enk_ev=enk_ev, eqp_ev=eqp_ev, delta_e_full=delta_e_full,
        in_grid=in_grid, ef_ev=ef_ev,
        slope_v=fit.slope_v, intercept_v=fit.intercept_v,
        slope_c=fit.slope_c, intercept_c=fit.intercept_c,
    )
    print("Wrote scissor_bands.npz")


if __name__ == "__main__":
    main()
