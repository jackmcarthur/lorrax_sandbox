"""Verify V_loc_sr + V_loc_lr = V_loc_full at several G points."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from scipy.special import erf as scipy_erf, spherical_jn
from scipy.integrate import quad

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import _extract_species_radial_data
from psp.radial_jax import make_local_sr_table, make_uniform_q_grid, radial_weights

crystal = CrystalData.from_qe_save("runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save")
vol = crystal.cell_volume
species_data = _extract_species_radial_data(pseudos, vol)

for sd in species_data:
    elem = sd["elem"]
    r = sd["r"]
    rab = sd["rab"]
    z_val = sd["z_valence"]
    vloc = sd["vloc_r"]
    e2 = 2.0

    # Build the sr table
    q_grid = make_uniform_q_grid(20.0, 4000)
    tab = make_local_sr_table(r, vloc, z_val, q_grid, radial_weights(r, rab, scheme="rab"))

    # Compute H₀[vloc_full](G) via direct numerical integration for G > 0
    # H₀[f](G) = int_0^inf f(r) * sin(Gr)/(Gr) * r^2 dr
    def H0_full(G):
        """Full V_loc Hankel transform via trapezoidal rule on the pseudopotential grid."""
        if G < 1e-10:
            return None  # divergent at G=0
        j0 = np.sin(G * r) / (G * r)
        j0[0] = 1.0  # limit of sin(x)/x at x=0
        integrand = vloc * j0 * r**2 * rab
        return np.sum(integrand)

    # The lr form factor: -Z*e² * exp(-G²/4) / G² 
    # (This is what the code adds to V_sr)
    def f_lr(G):
        return -z_val * e2 * np.exp(-G**2/4) / G**2

    # V_sr from the table (at exact G, not interpolated)
    def v_sr_from_table(G):
        return np.interp(G, q_grid, tab.values)

    print(f"\n=== {elem} (Z_v={z_val}) ===")
    print(f"{'G':>6} {'H0[vloc]':>14} {'v_sr(G)':>14} {'f_lr(G)':>14} {'sr+lr':>14} {'err':>14}")
    print("-" * 85)
    for G in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        h0 = H0_full(G)
        vsr = v_sr_from_table(G)
        flr = f_lr(G)
        total = vsr + flr
        err = total - h0
        print(f"{G:6.1f} {h0:14.6f} {vsr:14.6f} {flr:14.6f} {total:14.6f} {err:14.6f}")

