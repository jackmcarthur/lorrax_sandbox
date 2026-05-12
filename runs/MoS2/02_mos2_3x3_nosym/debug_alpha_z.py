"""Debug alpha-Z computation in ionic_gspace.py."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import _extract_species_radial_data
from psp.radial_jax import (make_local_sr_table, make_uniform_q_grid, radial_weights)

crystal = CrystalData.from_qe_save("qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("qe/nscf/MoS2.save")
vol = crystal.cell_volume
print(f"Cell volume: {vol:.4f} bohr^3")

species_data = _extract_species_radial_data(pseudos, vol)

for sd in species_data:
    elem = sd["elem"]
    r = sd["r"]
    rab = sd["rab"]
    z_val = sd["z_valence"]
    vloc_r = sd["vloc_r"]
    e2 = 2.0
    
    # Alpha-Z integral
    integrand = r * (r * vloc_r + z_val * e2)
    alpha_z = (4.0 * np.pi) * np.sum(integrand * rab) / vol
    
    # Check convergence: the integrand should → 0 at large r
    print(f"\n=== {elem} (Z_v={z_val}) ===")
    print(f"  r range: [{r[0]:.4e}, {r[-1]:.4f}] bohr, n_r={len(r)}")
    print(f"  vloc_r[0]={vloc_r[0]:.4f}, vloc_r[-1]={vloc_r[-1]:.6f}")
    print(f"  Expected vloc_r[-1] ≈ -Z*e²/r[-1] = {-z_val*e2/r[-1]:.6f}")
    print(f"  integrand[0]={integrand[0]:.6e}, integrand[-1]={integrand[-1]:.6e}")
    print(f"  |integrand[-1]|/max|integrand| = {abs(integrand[-1])/np.max(np.abs(integrand)):.2e}")
    print(f"  alpha_Z = {alpha_z:.6f} Ry (per atom, contributes to V_loc G=0)")
    
    # Make the vloc_sr table for comparison
    q_max = 20.0  # typical
    n_q = 4000
    q_grid = make_uniform_q_grid(q_max, n_q)
    weights_rab = radial_weights(r, rab, scheme="rab")
    tab = make_local_sr_table(r, vloc_r, z_val, q_grid, weights_rab)
    
    # The table at q=0 (before alpha-Z override)
    vloc_sr_at_q0 = tab.values[0]
    print(f"  vloc_sr_table[q=0] = {vloc_sr_at_q0:.6f} (before alpha-Z)")
    print(f"  alpha_Z * vol/(4pi) = {alpha_z * vol / (4*np.pi):.6f} (the override value)")
    print(f"  Difference: {alpha_z * vol/(4*np.pi) - vloc_sr_at_q0:.6f}")

# QE-style alpha-Z total
total_alpha_z = 0.0
for sd in species_data:
    # Count atoms of this species in the cell
    from psp.get_DFT_mtxels import _symbol_to_Z
    z_atomic = _symbol_to_Z(sd["elem"])
    natoms = sum(1 for t in crystal.atom_types if int(t) == z_atomic)
    
    r, rab, z_val, vloc_r = sd["r"], sd["rab"], sd["z_valence"], sd["vloc_r"]
    integrand = r * (r * vloc_r + z_val * 2.0)
    alpha_z_s = (4.0 * np.pi) * np.sum(integrand * rab) / vol
    total_alpha_z += natoms * alpha_z_s
    print(f"\n{sd['elem']}: natoms={natoms}, alpha_Z_per_atom={alpha_z_s:.6f}, "
          f"contribution={natoms*alpha_z_s:.6f}")

print(f"\nTotal alpha_Z eigenvalue shift = {total_alpha_z:.6f} Ry = {total_alpha_z*1000:.3f} mRy")
print(f"Observed MoS2 offset = 594.459 mRy")

