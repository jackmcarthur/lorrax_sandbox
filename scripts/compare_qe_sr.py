"""Compare QE vs LORRAX SR form factor computation at individual q points.

Tests: Simpson vs rab integration, msh cutoff, and the form factor values.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from scipy.special import erf as scipy_erf

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import _extract_species_radial_data
from psp.radial_jax import make_local_sr_table, make_uniform_q_grid, radial_weights

def simpson(n, f, rab):
    """QE-style Simpson integration: ∫ f(r) dr ≈ Σ c_i f_i rab_i."""
    result = 0.0
    for i in range(0, n - 2, 2):
        result += (f[i] * rab[i] + 4 * f[i+1] * rab[i+1] + f[i+2] * rab[i+2]) / 3.0
    return result

crystal = CrystalData.from_qe_save("runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save")
vol = crystal.cell_volume
species_data = _extract_species_radial_data(pseudos, vol)

for sd in species_data:
    elem = sd["elem"]
    r = sd["r"]
    rab = sd["rab"]
    vloc = sd["vloc_r"]
    z_val = sd["z_valence"]
    e2 = 2.0

    # QE-style msh: cutoff at ~10 bohr, force odd
    msh = len(r)
    for ir in range(len(r)-1, -1, -1):
        if r[ir] <= 10.0:
            msh = ir + 1
            break
    if msh % 2 == 0:
        msh -= 1

    print(f"\n=== {elem} (Z_v={z_val}) ===")
    print(f"  Full grid: n_r={len(r)}, r_max={r[-1]:.2f}")
    print(f"  QE msh: {msh}, r_cut={r[msh-1]:.2f}")

    # Compare at several q values
    print(f"\n  {'q':>6} {'QE_simpson':>14} {'LX_rab':>14} {'LX_simpson':>14} {'diff_rab':>14} {'diff_simp':>14}")
    print("  " + "-" * 80)

    for q in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        # QE-style computation (Simpson, msh cutoff)
        aux_qe = np.zeros(msh)
        for ir in range(msh):
            ri = r[ir]
            if q > 1e-10 and ri > 1e-12:
                aux_qe[ir] = (ri * vloc[ir] + z_val * e2 * scipy_erf(ri)) * np.sin(q * ri) / q
            elif q <= 1e-10:
                aux_qe[ir] = ri * (ri * vloc[ir] + z_val * e2 * scipy_erf(ri))
            else:
                aux_qe[ir] = ri * (ri * vloc[ir] + z_val * e2 * 2/np.sqrt(np.pi)) if q <= 1e-10 else 0.0

        vloc_qe = simpson(msh, aux_qe, rab[:msh]) * 4 * np.pi / vol

        # LORRAX-style (rab weights, full grid) 
        n_full = len(r)
        aux_lx = np.zeros(n_full)
        for ir in range(n_full):
            ri = r[ir]
            if q > 1e-10 and ri > 1e-12:
                aux_lx[ir] = (ri * vloc[ir] + z_val * e2 * scipy_erf(ri)) * np.sin(q * ri) / q
            elif q <= 1e-10:
                aux_lx[ir] = ri * (ri * vloc[ir] + z_val * e2 * scipy_erf(ri))

        vloc_lx_rab = np.sum(aux_lx * rab) * 4 * np.pi / vol
        vloc_lx_simp = simpson(n_full if n_full % 2 == 1 else n_full - 1, 
                                aux_lx[:n_full if n_full % 2 == 1 else n_full - 1],
                                rab[:n_full if n_full % 2 == 1 else n_full - 1]) * 4 * np.pi / vol

        diff_rab = (vloc_lx_rab - vloc_qe) * 1000  # mRy
        diff_simp = (vloc_lx_simp - vloc_qe) * 1000

        print(f"  {q:6.2f} {vloc_qe:14.6f} {vloc_lx_rab:14.6f} {vloc_lx_simp:14.6f} "
              f"{diff_rab:+14.3f} {diff_simp:+14.3f}")

    # Alpha-Z comparison
    aux_az_qe = np.zeros(msh)
    aux_az_lx = np.zeros(len(r))
    for ir in range(msh):
        aux_az_qe[ir] = r[ir] * (r[ir] * vloc[ir] + z_val * e2)
    for ir in range(len(r)):
        aux_az_lx[ir] = r[ir] * (r[ir] * vloc[ir] + z_val * e2)

    az_qe = simpson(msh, aux_az_qe, rab[:msh]) * 4 * np.pi / vol
    az_lx = np.sum(aux_az_lx * rab) * 4 * np.pi / vol

    print(f"\n  Alpha-Z: QE={az_qe:.6f}, LX={az_lx:.6f}, diff={(az_lx-az_qe)*1000:+.3f} mRy")

