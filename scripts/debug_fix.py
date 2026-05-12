"""Debug: verify the alpha-Z fix is being applied."""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, jax, jax.numpy as jnp

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials, _symbol_to_Z
from psp.ionic_gspace import (build_fft_G_data, _extract_species_radial_data,
                               species_structure_factors, accumulate_species_on_G)
from psp.radial_jax import (make_local_sr_table, make_uniform_q_grid, radial_weights)

crystal = CrystalData.from_qe_save("runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save")
pseudos = load_pseudopotentials("runs/MoS2/02_mos2_3x3_nosym/qe/nscf/MoS2.save")
fft_grid = crystal.fft_grid
nx, ny, nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])
N = nx * ny * nz
vol = crystal.cell_volume
sqrtN = np.sqrt(float(N))

G_crys_flat, G_norm_flat = build_fft_G_data(fft_grid, crystal.bvec, crystal.blat)
q_max = float(np.max(G_norm_flat))
n_q = 4000
q_grid = make_uniform_q_grid(q_max, n_q)
q0, dq = float(q_grid[0]), float(q_grid[1] - q_grid[0])

species_data = _extract_species_radial_data(pseudos, vol)

# Build tables WITHOUT alpha-Z override
vloc_tables = np.zeros((len(species_data), n_q))
for i, sd in enumerate(species_data):
    if sd["has_vloc"]:
        r, rab = sd["r"], sd["rab"]
        tab = make_local_sr_table(r, sd["vloc_r"], sd["z_valence"], q_grid,
                                   radial_weights(r, rab, scheme="rab"))
        vloc_tables[i] = tab.values

print(f"Table values at q=0:")
for i, sd in enumerate(species_data):
    print(f"  {sd['elem']}: vloc_table[0] = {vloc_tables[i,0]:.6f}, vloc_table[1] = {vloc_tables[i,1]:.6f}")

# Accumulate V_sr without alpha-Z override
# ... (need structure factors etc.)
# Let's just compute alpha-Z and compare
print(f"\nAlpha-Z values:")
total_az = 0.0
for sd in species_data:
    if not sd["has_vloc"]:
        continue
    r, rab = sd["r"], sd["rab"]
    z_val = sd["z_valence"]
    integrand = r * (r * sd["vloc_r"] + z_val * 2.0)
    az = (4*np.pi) * np.sum(integrand * rab) / vol
    z_at = _symbol_to_Z(sd["elem"])
    n_at = int(np.sum(np.asarray(crystal.atom_types, dtype=int) == z_at))
    total_az += n_at * az
    # What the table gives at q=0 (through accumulator):
    table_g0_contrib = (4*np.pi/vol) * n_at * vloc_tables[species_data.index(sd), 0]
    alpha_z_contrib = n_at * az
    print(f"  {sd['elem']}: n_at={n_at}, alpha_Z_per_atom={az:.6f}, "
          f"table_q0_contrib={table_g0_contrib:.6f}, alpha_Z_contrib={alpha_z_contrib:.6f}")

print(f"\nTotal table_q0_eigenvalue = (4π/vol) * sum = {(4*np.pi/vol)*sum(vloc_tables[i,0]*int(np.sum(np.asarray(crystal.atom_types)==_symbol_to_Z(sd['elem']))) for i, sd in enumerate(species_data) if sd['has_vloc']):.6f} Ry")
print(f"Total alpha_Z_eigenvalue = {total_az:.6f} Ry")
print(f"Difference = {(total_az - (4*np.pi/vol)*sum(vloc_tables[i,0]*int(np.sum(np.asarray(crystal.atom_types)==_symbol_to_Z(sd['elem']))) for i, sd in enumerate(species_data) if sd['has_vloc']))*1000:.3f} mRy")

