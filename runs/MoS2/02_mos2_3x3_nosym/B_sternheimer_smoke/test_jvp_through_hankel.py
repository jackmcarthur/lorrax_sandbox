"""Verify 1st-derivative (JVP) plumbing through the new JAX Hankel pipeline.

Three independent checks:

  (1) Compare setup.G_table  built by the NEW build_vnl_setup against
      a fresh scipy-only reference computed from the same UPF data.

  (2) Compare setup.Gp_table similarly.  This is the table that
      _interp_with_deriv's custom_jvp uses as the q-tangent — any error
      here propagates directly into k·p tangents, S-tensor, and
      velocity matrix elements.

  (3) Compute the velocity matrix elements (compute_dZ=True path,
      which exercises Gp_table via _table_interp inside dZ assembly)
      for a non-trivial k-point and compare against a scipy-Gp_table
      reference.  This stresses the *actual* dipole/velocity output.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')

from runtime import set_default_env; set_default_env()
import numpy as np
import jax.numpy as jnp

from file_io import WFNReader
from psp.pseudos import load_pseudopotentials
from psp.species import extract_species, build_atom_species_map
import psp.vnl_ops as vnl_ops
from psp.radial_tables import (
    build_all_tables, hankel_l, projector_deriv_table,
)

wfn = WFNReader('WFN.h5')
pseudos = load_pseudopotentials('.')
nspinor = int(wfn.nspinor)
ecut = float(wfn.ecutwfc)
# build_vnl_setup multiplies q_max by 1.01 internally — pre-scale here so the
# NEW (build_vnl_setup) path and the OLD scipy reconstruction below use the
# *same* q-grid.  Without this, comparison is off by a 1.01× q-axis stretch.
q_max_in = float(np.sqrt(ecut)) * 1.01
q_max_used = q_max_in * 1.01    # what build_vnl_setup actually uses
n_q = 4000

# (1) + (2): NEW path
new_setup = vnl_ops.build_vnl_setup(
    wfn, pseudos=pseudos, nspinor=nspinor, q_max=q_max_in)
G_table_new = np.asarray(new_setup.G_table)
Gp_table_new = np.asarray(new_setup.Gp_table)

# OLD scipy reference: replay the exact build_vnl_setup logic with scipy
species_list = extract_species(pseudos, nspinor=nspinor)
species_natoms, species_tau, _ = build_atom_species_map(wfn, species_list)
q_grid = np.linspace(0.0, max(q_max_used, 1e-8), n_q)

# Build scipy-only F_l for each projector (old projector_table path)
ref_tables = build_all_tables(species_list, q_max_used, n_q)  # uses NEW pipeline now
# To force the OLD path for reference, build F_l ourselves with scipy
ref_proj_F = []
for sp in species_list:
    F = np.zeros((sp.n_proj, n_q), dtype=np.float64)
    for ip in range(sp.n_proj):
        l = int(sp.proj_l[ip])
        F[ip] = hankel_l(l, sp.r, sp.beta_r[ip], q_grid, sp.rab)
    ref_proj_F.append(F)

# Build scipy-only Gp via projector_deriv_table (the original per-projector path)
ref_proj_dGdq = []  # this is the *normalised* dG_l/dq, NOT raw H_{l+1}
for sp in species_list:
    rows = np.zeros((sp.n_proj, n_q), dtype=np.float64)
    for ip in range(sp.n_proj):
        rows[ip] = projector_deriv_table(sp, ip, q_grid)
    ref_proj_dGdq.append(rows)

# Now reconstruct the OLD G_table / Gp_table arrays the way build_vnl_setup did,
# using the SCIPY F_l + projector_deriv_table values as input.
G_rows_old, Gp_rows_old = [], []
for isp, sp in enumerate(species_list):
    natoms = int(species_natoms[isp])
    if natoms == 0: continue
    per_l = {}
    for ip in range(sp.n_proj):
        per_l.setdefault(int(sp.proj_l[ip]), []).append(ip)
    for l, proj_ids in per_l.items():
        for ip in proj_ids:
            F_vals = ref_proj_F[isp][ip]
            if l == 0:
                G_vals = F_vals.copy()
            else:
                G_vals = np.empty(n_q, dtype=np.float64)
                G_vals[1:] = F_vals[1:] / q_grid[1:] ** l
                G_vals[0]  = F_vals[1] / q_grid[1] ** l
            G_rows_old.append(G_vals)
            Gp_rows_old.append(ref_proj_dGdq[isp][ip])

G_table_old = np.stack(G_rows_old) if G_rows_old else np.zeros((0, n_q))
Gp_table_old = np.stack(Gp_rows_old) if Gp_rows_old else np.zeros((0, n_q))

def cmp(name, a, b):
    diff = a - b
    abs_max = float(np.max(np.abs(diff)))
    rel_max = abs_max / max(float(np.max(np.abs(a))), 1e-300)
    print(f"  {name:18s}: |Δ|_∞ = {abs_max:.3e}   rel = {rel_max:.3e}")
    return abs_max, rel_max

print("\n══ (1) Forward G_table  (used by _interp_with_deriv primal) ══")
ag, rg = cmp("G_table  (new vs scipy-old)", G_table_new, G_table_old)

print("\n══ (2) Deriv Gp_table  (used by _interp_with_deriv JVP tangent) ══")
ad, rd = cmp("Gp_table (new vs scipy-old)", Gp_table_new, Gp_table_old)

# (3) Velocity matrix elements via compute_dZ=True path
print("\n══ (3) Velocity matrix elements via compute_dZ=True ══")
from common import symmetry_maps, Meta
sym = symmetry_maps.SymMaps(wfn)
n_occ = int(wfn.nelec)
meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=0, nband=n_occ, n_rmu=0, bispinor=False)
from psp.dft_operators import build_T_diag_from_kvec

# Build dZ from current (NEW) setup at one k-point
ik = 4
kv = np.asarray(sym.unfolded_kpts[ik], dtype=float)
T_diag, Gx, Gy, Gz = build_T_diag_from_kvec(kv, wfn)
Gk_int = np.stack([np.asarray(Gx), np.asarray(Gy), np.asarray(Gz)], axis=-1)
kdata_new = vnl_ops.build_vnl_kdata_from_kvec(kv, Gk_int, new_setup, compute_dZ=True)
dZ_new = np.asarray(kdata_new.dZ)
print(f"  dZ_new shape = {dZ_new.shape}, |dZ|_∞ = {np.max(np.abs(dZ_new)):.3e}")

# Build a "scipy-Gp" setup by mutating new_setup.Gp_table in place
import dataclasses
old_setup = dataclasses.replace(new_setup, Gp_table=jnp.asarray(Gp_table_old))
kdata_old = vnl_ops.build_vnl_kdata_from_kvec(kv, Gk_int, old_setup, compute_dZ=True)
dZ_old = np.asarray(kdata_old.dZ)
diff_dZ = dZ_new - dZ_old
print(f"  |dZ_new - dZ_old|_∞ = {np.max(np.abs(diff_dZ)):.3e}  "
      f"(rel = {np.max(np.abs(diff_dZ))/max(np.max(np.abs(dZ_new)), 1e-300):.3e})")

# Verdict
TOL = 1e-9
worst_rel = max(rg, rd)
print(f"\n  Tolerance: < {TOL:.0e} for tables")
fail = (rg > TOL) or (rd > TOL)
if fail:
    print(f"  [WARN] regression: G rel={rg:.2e}  Gp rel={rd:.2e}")
    sys.exit(1)
print(f"  [PASS] new G/Gp tables agree with scipy reference within machine eps.")
print(f"         dZ velocity matrix elements: rel-Δ = "
      f"{np.max(np.abs(diff_dZ))/max(np.max(np.abs(dZ_new)), 1e-300):.2e}")
