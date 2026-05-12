"""Body-element cond-band convergence study at 60 Ry.

Tests whether χ_{G' G_pert}(q) for non-trivial G_pert converges more
slowly with N_cond than the head/wings (G_pert = 0).

Drives ``run_sternheimer`` directly with a custom ``V_pert_box``
(cell-periodic part of the perturbation) corresponding to a single
plane-wave  V_pert(r) = e^{i·2π·G_pert·r}.  Sweeps n_cond_bands and
records the diagonal body element  χ_{G_pert G_pert}(q)  for the
chosen G_pert plus a few off-diagonal (G' ≠ G_pert) wings.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import h5py

sys.path.insert(0, '/global/homes/j/jackm/software/lorrax_B/src')
sys.path.insert(0, '/global/homes/j/jackm/scratchperl/.isdf/isdf_venvs/isdf_site')
from runtime import set_default_env; set_default_env()

import jax
import jax.numpy as jnp
from psp.run_sternheimer import run_sternheimer, build_Gprime_list
from file_io import WFNReader

WFN  = 'WFN.h5'
PSEUDO_DIR = '.'
IQ        = 4                                # signed q ≈ (1/3, 1/3, 0)
TOL       = 1e-6
MAX_ITER  = 80
NG_OUT    = 256                              # output G'-list size — large enough
                                             # to include G_pert (|q+G_pert|²≈5)
                                             # in the output sphere
WORK = Path('body_sweep_60Ry'); WORK.mkdir(exist_ok=True)
N_LIST = [4, 8, 16, 32, 64, 128, 150]

# ── Pick a decently large G_pert from the q=4 G-sphere ──
wfn = WFNReader(WFN)
qcrys = np.asarray(wfn.kpoints[IQ], dtype=np.float64)
qsigned = qcrys - np.round(qcrys)
nx, ny, nz = (int(v) for v in wfn.fft_grid)
B = float(wfn.blat) * np.asarray(wfn.bvec, dtype=np.float64)

# Use the same |q+G'|-sorted output list and pick a moderate-G entry
G_list_all = build_Gprime_list(qsigned, wfn, NG_OUT * 4)
qG_cart = (qsigned[None, :] + G_list_all) @ B
qG_sq   = np.sum(qG_cart ** 2, axis=1)

# Pick G_pert with |q+G_pert|² ≈ 5 a.u.² (well above |q|² ≈ 0.49)
target = 5.0
i_pert = int(np.argmin(np.abs(qG_sq - target)))
G_pert = G_list_all[i_pert]
print(f"  q signed   = {qsigned}")
print(f"  picked G_pert = {tuple(G_pert)}  |q+G_pert|² = {qG_sq[i_pert]:.4f} a.u.")
print(f"     for ref:  |q|²        = {float(np.sum((qsigned @ B) ** 2)):.4f}")

# ── Build V_pert_box(r) = e^{i·2π·G_pert·r} on the FFT box ──
# Convention: f(r_j) = sum_G c_G e^{i·2π·j·G/N}.  For c_G = δ_{G,G_pert}:
#   c_G_box = scatter[G_pert] = 1
#   f_j     = sqrt(N) · ifft_ortho(c_G_box)
N_grid = nx * ny * nz
c_G_box = np.zeros((nx, ny, nz), dtype=np.complex128)
ix = int(G_pert[0]) % nx
iy = int(G_pert[1]) % ny
iz = int(G_pert[2]) % nz
c_G_box[ix, iy, iz] = 1.0
V_pert_box = np.sqrt(N_grid) * np.fft.ifftn(c_G_box, axes=(0, 1, 2), norm='ortho')
# Sanity: (V_pert_box * conj(V_pert_box)).mean()  should equal 1/N for δ-Fourier.
print(f"  V_pert_box: shape {V_pert_box.shape}, "
      f"|V|_∞ = {np.max(np.abs(V_pert_box)):.4f}  (expect 1)")

V_pert_box_j = jnp.asarray(V_pert_box, dtype=jnp.complex128)

# ── Run Sternheimer for varying N_cond with this V_pert ──
results = {}
for label, sos_only, n_cond in (
    [('full',     False, 150)] +                 # reference: full Sternheimer
    [(f'sos_n{N:03d}', True, N) for N in N_LIST]
):
    out = WORK / f"{label}.h5"
    print(f"\n  ── {label}  (sos_only={sos_only}, n_cond_bands={n_cond}) ──")
    t0 = time.perf_counter()
    run_sternheimer(
        wfn_path=WFN, pseudo_dir=PSEUDO_DIR,
        n_cond_bands=n_cond,
        iq_list=[IQ], ng_out=NG_OUT,
        tol=TOL, max_iter=MAX_ITER,
        truncation_2d=True,
        output_path=str(out),
        with_derivatives=False, with_s_tensor=False,
        sos_only=sos_only,
        V_pert_box=V_pert_box_j,
        verbose=False,
    )
    print(f"    done  {time.perf_counter()-t0:.1f}s")
    with h5py.File(out, 'r') as f:
        results[label] = (np.asarray(f['q_0/chi_col']),
                          np.asarray(f['q_0/G_int']))

# ── Tabulate ──
chi_full, Gint_out = results['full']
order = np.argsort(((qsigned[None, :] + Gint_out) @ B).__pow__(2).sum(axis=1))
Gint_out = Gint_out[order]; chi_full = chi_full[order]
qG2_out = np.sum(((qsigned[None, :] + Gint_out) @ B) ** 2, axis=1)

# Find the diagonal body element — G' = G_pert
diag_idx = -1
for i, g in enumerate(Gint_out):
    if (g % np.array([nx, ny, nz]) == np.array([ix, iy, iz])).all():
        diag_idx = i; break
print(f"\n  Diagonal body  G' = G_pert  found at output idx {diag_idx}, "
      f"|q+G'|² = {qG2_out[diag_idx]:.3f}")

# Find G'=0 component (head equiv)
g0 = -1
for i, g in enumerate(Gint_out):
    if (g == 0).all():
        g0 = i; break
print(f"  G'=0 component at output idx {g0}, |q|² = {qG2_out[g0]:.3f}")

# Save raw, plus tabulate the picked diagonal body convergence
N_arr = np.asarray(N_LIST)
chi_sos_diag = np.zeros(len(N_LIST), dtype=np.complex128)
chi_sos_g0   = np.zeros(len(N_LIST), dtype=np.complex128)
for i, N in enumerate(N_LIST):
    chi, _ = results[f'sos_n{N:03d}']
    chi = chi[order]
    if diag_idx >= 0: chi_sos_diag[i] = chi[diag_idx]
    if g0       >= 0: chi_sos_g0[i]   = chi[g0]

print("\n══ Body element χ_{G_pert G_pert}(q) — diagonal ══")
print(f"  Reference (full Sternheimer): {chi_full[diag_idx]:+.4e}")
for i, N in enumerate(N_LIST):
    rel = abs(chi_sos_diag[i] / chi_full[diag_idx]) if chi_full[diag_idx] != 0 else float('nan')
    print(f"    N={N:>4d}  chi = {chi_sos_diag[i]:+.4e}   |chi_N|/|chi_full| = {rel:.4f}")

print("\n══ For comparison: G'=0 (head, but with G_pert ≠ 0 perturbation) ══")
print(f"  Reference: {chi_full[g0]:+.4e}")
for i, N in enumerate(N_LIST):
    rel = abs(chi_sos_g0[i] / chi_full[g0]) if abs(chi_full[g0]) > 1e-30 else float('nan')
    print(f"    N={N:>4d}  chi = {chi_sos_g0[i]:+.4e}   |chi_N|/|chi_full| = {rel:.4f}")

np.savez(WORK / 'sweep.npz',
         N_list=N_arr, chi_full=chi_full, chi_sos=np.stack([results[f'sos_n{N:03d}'][0][order] for N in N_LIST]),
         Gint=Gint_out, qG_sq=qG2_out, q_signed=qsigned,
         G_pert=G_pert, qG_pert_sq=qG_sq[i_pert],
         diag_idx=diag_idx, g0_idx=g0, ecutwfc=60.0)
print(f"\n  Saved {WORK / 'sweep.npz'}")
