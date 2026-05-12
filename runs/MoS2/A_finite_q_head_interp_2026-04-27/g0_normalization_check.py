"""Sanity-check: does my (1/N_FFT) Σ_r e^{-iG·r} zeta_q[q,r,μ] formula match
the persisted g0_mu(q) at q=Γ AND give a reasonable A_x = z_{Γ,μ}(G=b_x)?"""
import os; os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np, h5py

HERE = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/A_finite_q_head_interp_2026-04-27"
with h5py.File(f"{HERE}/fine_8x8/tmp/zeta_q.h5", 'r') as f:
    zeta_q0 = f["zeta_q"][0, :, :].astype(np.complex128)        # (n_FFT=46080, n_μ=640) at q=Γ
    g0_persisted = f["g0_mu"][0, 0, 0, :].astype(np.complex128)  # (n_μ,)
with h5py.File(f"{HERE}/fine_8x8/WFN.h5", 'r') as f:
    fft_grid = tuple(int(x) for x in f["mf_header/gspace/FFTgrid"][...])
    cell_volume = float(f["mf_header/crystal/celvol"][()])
N_x, N_y, N_z = fft_grid
n_FFT = N_x * N_y * N_z

# Try several normalization conventions and find which matches the persisted g0_mu(Γ).
ratios = {}
for tag, factor in [("(1/N_FFT)", 1.0/n_FFT), ("(1/sqrt(N_FFT))", 1.0/np.sqrt(n_FFT)),
                    ("(1)", 1.0), ("(N_FFT)", float(n_FFT)),
                    ("V_cell/N_FFT", cell_volume / n_FFT),
                    ("V_cell", cell_volume), ("1/V_cell", 1.0/cell_volume),
                    ("V_cell/sqrt(N_FFT)", cell_volume/np.sqrt(n_FFT))]:
    g0_my = zeta_q0.sum(axis=0) * factor                          # (n_μ,)
    rel_err = np.linalg.norm(g0_my - g0_persisted) / np.linalg.norm(g0_persisted)
    ratios[tag] = (np.linalg.norm(g0_my), rel_err)

print(f"persisted ‖g0_mu(Γ)‖_F = {np.linalg.norm(g0_persisted):.6e}")
print(f"persisted g0_mu(Γ, μ=0..3) = {g0_persisted[:3]}")
print(f"FFT grid = {fft_grid},  V_cell = {cell_volume:.3f},  n_FFT = {n_FFT}")
print()
print("Trying normalization conventions for g0_my = (factor) · Σ_r ζ_Γ,μ(r):")
print(f"{'factor':>22}  {'‖g0_my‖_F':>13}  {'rel ‖g0_my − g0_persisted‖_F':>30}")
for tag, (norm, rel) in ratios.items():
    print(f"  {tag:>20}  {norm:>13.4e}  {rel:>30.3e}")

# Now compute z_{Γ,μ}(G=b_x) using whichever factor matched, and report magnitude.
print()
print("=== z_{Γ,μ}(G=b_x) under each convention ===")
phase_x = np.exp(-2j * np.pi * np.arange(N_x) / N_x)
zeta_q0_3d = zeta_q0.reshape(N_x, N_y, N_z, -1)
# Σ_r e^{-iG_bx·r} ζ_Γ,μ(r) = Σ_xyz ζ_3d[x,y,z,μ] · phase_x[x]
sum_with_phase = np.einsum('xyzm,x->m', zeta_q0_3d, phase_x)
for tag, factor in [("(1/N_FFT)", 1.0/n_FFT), ("(1/sqrt(N_FFT))", 1.0/np.sqrt(n_FFT)),
                    ("(1)", 1.0), ("V_cell/N_FFT", cell_volume / n_FFT),
                    ("V_cell/sqrt(N_FFT)", cell_volume/np.sqrt(n_FFT))]:
    A = sum_with_phase * factor
    print(f"  {tag:>20}: ‖z_Γ(G=b_x)‖_F = {np.linalg.norm(A):.4e}")
print(f"\n  for reference: persisted ‖g0_mu(Γ)‖_F = {np.linalg.norm(g0_persisted):.4e}")
