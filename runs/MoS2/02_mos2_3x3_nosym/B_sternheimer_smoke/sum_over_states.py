"""Adler–Wiser finite-q sum-over-states for χ_{00}(q, ω=0).

Uses all bands in WFN.h5 (occupied + conduction) to construct:

    χ_{00}(q, 0) = (2 · spin_factor / N_k) Σ_{v,c,k}
                    |⟨u_{c, k-q} | u_{v,k}⟩_cell|² / (ε_{v,k} − ε_{c,k-q})

The factor of 2 lumps the +ω and -ω poles at ω=0 (per Adler–Wiser).
spin_factor = 2 for nspinor=1, 1 for nspinor=2.

The cell overlap ⟨u_{c,k-q} | u_{v,k}⟩_cell = (1/V) ∫_cell u_c^*(r) u_v(r) dr
is done via ortho-FFT-box convention:  with u_v box-scatter at k and u_c
box-scatter at k-q, both on the same FFT grid, the cell overlap is just
Σ_j (1/N) ψ_c^*(r_j) ψ_v(r_j) = Σ_G c_c^*(G) c_v(G)  — but the G's
live on DIFFERENT spheres, so we go through real space via IFFT.

Cross-check target: my Sternheimer χ_{00}(q=1, 0) = -3.958e-02.
"""
import sys
sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env
set_default_env()
import jax, jax.numpy as jnp, numpy as np
from common import Meta, symmetry_maps
from common.load_wfns import load_kpoint_fftbox
from file_io import WFNReader

wfn = WFNReader('WFN.h5'); sym = symmetry_maps.SymMaps(wfn)
n_occ  = int(wfn.nelec)                 # 26
n_band = int(wfn.nbands)                # 80 on MoS2 3×3 nosym
n_cond = n_band - n_occ
nspinor = int(wfn.nspinor)
spin_factor = 2 if nspinor == 1 else 1
print(f"n_occ={n_occ}  n_cond={n_cond}  nspinor={nspinor}")

meta = Meta.from_system(wfn, sym, nval=n_occ, ncond=n_cond, nband=n_band, n_rmu=0, bispinor=False)

# Pre-load all bands at all k
nk_full = int(sym.nk_tot)
print(f"Loading all {n_band} bands at {nk_full} k-points...")
psi_box_full = jnp.stack([load_kpoint_fftbox(wfn, sym, meta, ik, n_band)
                          for ik in range(nk_full)], axis=0)
# (nk, nb, ns, nx, ny, nz) G-space scatter.  IFFT → cell-periodic u(r).
# Ortho IFFT convention: Σ_j |u_r(r_j)|² = 1.
psi_r_full = jnp.fft.ifftn(psi_box_full, axes=(-3,-2,-1), norm='ortho')

# Energies at each full-BZ k
irk_to_k = np.asarray(sym.irk_to_k_map)
en_full = jnp.asarray(wfn.energies[0, irk_to_k, :n_band], dtype=jnp.float64)

# Loop: for each reduced q, compute χ_{00}
print(f"\nRunning sum-over-states...")
for iq in [1, 2, 4, 5]:   # pick representative q-points
    qvec = wfn.kpoints[iq]
    chi = 0.0 + 0.0j
    for ik in range(nk_full):
        ik_kminq = int(sym.kq_map[ik, iq])
        u_vk   = psi_r_full[ik,       :n_occ]        # (n_occ, ns, nx, ny, nz)
        u_ckq  = psi_r_full[ik_kminq, n_occ:n_band]  # (n_cond, ns, nx, ny, nz)
        eps_v  = en_full[ik,       :n_occ]           # (n_occ,)
        eps_c  = en_full[ik_kminq, n_occ:n_band]     # (n_cond,)
        # Cell overlap:  M_vc = (1/N) Σ_j Σ_s u_c^*(r_j) u_v(r_j)
        # With ortho IFFT, Σ_j  ⟨u|u⟩ = 1 per band (= norm 1), so the sum
        # over grid points IS the overlap directly.
        M_vc = jnp.einsum('csxyz,vsxyz->vc', jnp.conj(u_ckq), u_vk)
        delta_E = eps_v[:, None] - eps_c[None, :]     # (n_occ, n_cond) = ε_v − ε_c < 0
        chi_vc = jnp.abs(M_vc) ** 2 / delta_E
        chi += complex(jnp.sum(chi_vc))
    # Factor 2 from +ω,-ω combination at ω=0; spin_factor;  average over k
    chi = 2.0 * spin_factor * chi / nk_full
    print(f"  q[{iq}] = {np.asarray(qvec)}  χ_{{00}} = {chi:.6e}")
