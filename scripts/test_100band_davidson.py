"""Fair 100-band comparison: LORRAX Davidson vs QE Davidson."""
import os
os.environ["JAX_ENABLE_X64"] = "1"
import sys, functools, re
sys.path.insert(0, "/global/homes/j/jackm/software")

import time
import numpy as np
import jax
import jax.numpy as jnp

from psp.qe_save_reader import CrystalData
from psp.get_DFT_mtxels import load_pseudopotentials
from psp.ionic_gspace import build_ionic_and_core
from psp.charge_density import build_G_cart
from psp.dft_operators import (
    compute_V_H_and_V_xc, build_V_scf, compute_ngkmax,
    setup_H_k_from_kvec, apply_H_k,
)
from psp.davidson import davidson_k, warmup_jit
import psp.vnl_ops as vnl_ops

ngpu = len(jax.devices())
print(f"JAX: {ngpu} GPU(s)")

qe_save = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf/silicon.save"
crystal = CrystalData.from_qe_save(qe_save)
pseudos = load_pseudopotentials(qe_save)
fft_grid = crystal.fft_grid
n_tgt = 100
nspinor = crystal.nspinor
_nx, _ny, _nz = int(fft_grid[0]), int(fft_grid[1]), int(fft_grid[2])

# ── Setup ──
t_setup_start = time.perf_counter()
V_loc_r, rho_core_r, rho_core_G = build_ionic_and_core(crystal, pseudos, fft_grid, truncation_2d=False)
rho_r, _ = crystal.load_charge_density()
rho_val = jnp.asarray(rho_r, dtype=jnp.float64)
B = float(crystal.blat) * np.asarray(crystal.bvec, dtype=float)
G_cart = build_G_cart(_nx, _ny, _nz, B)
V_H_r, V_xc_r = compute_V_H_and_V_xc(
    rho_val, rho_core_r, rho_core_G, G_cart,
    jnp.asarray(crystal.bdot, dtype=jnp.float64),
    jnp.asarray(crystal.bvec, dtype=jnp.float64), crystal.blat)
V_scf = build_V_scf(V_loc_r, V_H_r, V_xc_r)
jax.block_until_ready(V_scf)

vnl_setup = vnl_ops.build_vnl_setup(crystal, pseudos=pseudos, nspinor=nspinor,
                                     q_max=float(np.sqrt(float(crystal.ecutwfc))) * 1.01)
kpoints, _ = crystal.build_kgrid(nk=(4, 4, 4), nosym=True, noinv=True,
                                  no_t_rev=True, force_symmorphic=False)
kpoints = np.asarray(kpoints, dtype=np.float64)
nk = len(kpoints)
ngkmax = compute_ngkmax(kpoints, crystal.bdot, crystal.ecutwfc, crystal.fft_grid)
t_setup = time.perf_counter() - t_setup_start
print(f"Setup: {t_setup:.2f}s")
print(f"nk={nk}, ngkmax={ngkmax}, n_tgt={n_tgt}, nspinor={nspinor}")

# ── Single JIT'd apply_H: sparse-G → sparse-G ──
@functools.partial(jax.jit, static_argnames=('_nx', '_ny', '_nz'))
def _apply_H(psi_G, T, V, Gx, Gy, Gz, Z, E, mask, _nx, _ny, _nz):
    mask_f = mask[None, None, :].astype(psi_G.dtype)
    psi_masked = psi_G * mask_f
    psi_box = jnp.zeros((*psi_G.shape[:2], _nx, _ny, _nz), dtype=psi_G.dtype)
    psi_box = psi_box.at[:, :, Gx, Gy, Gz].add(psi_masked)
    return apply_H_k(psi_box, T, V, Gx, Gy, Gz, Z, E, mask)

# ── JIT warmup ──
t0 = time.perf_counter()
warmup_jit(ngkmax, nspinor, n_tgt)
# Also warmup _apply_H with dummy
H_k0 = setup_H_k_from_kvec(kpoints[0], V_scf, vnl_setup, crystal, None,
                             V_loc_r=V_loc_r, ngkmax=ngkmax)
dummy = jnp.zeros((1, nspinor, ngkmax), dtype=jnp.complex128)
_ = _apply_H(dummy, H_k0.T_diag, H_k0.V_scf,
              H_k0.Gx, H_k0.Gy, H_k0.Gz, H_k0.vnl_Z, H_k0.vnl_E, H_k0.mask,
              _nx, _ny, _nz)
t_warmup = time.perf_counter() - t0
print(f"JIT warmup: {t_warmup:.2f}s")

# ── Davidson at all k-points ──
all_evals = np.zeros((nk, n_tgt))
t_davidson_start = time.perf_counter()

for ik in range(nk):
    t0 = time.perf_counter()
    H_k = setup_H_k_from_kvec(kpoints[ik], V_scf, vnl_setup, crystal, None,
                                V_loc_r=V_loc_r, ngkmax=ngkmax)

    # Closure binds H_k data; _apply_H JIT is already compiled (same shapes)
    def apply_H(psi_G, _H=H_k):
        return _apply_H(psi_G, _H.T_diag, _H.V_scf,
                         _H.Gx, _H.Gy, _H.Gz, _H.vnl_Z, _H.vnl_E, _H.mask,
                         _nx, _ny, _nz)

    evals, _ = davidson_k(
        apply_H, h_diag=H_k.h_diag, nG=ngkmax, nspinor=nspinor,
        n_tgt=n_tgt, T_diag=H_k.T_diag, verbose=False,
    )
    all_evals[ik] = evals
    dt = time.perf_counter() - t0
    if ik < 5 or ik == nk - 1:
        print(f"  k={ik}: {dt:.3f}s")

t_davidson = time.perf_counter() - t_davidson_start
t_total = time.perf_counter() - t_setup_start
print(f"\nDavidson: {t_davidson:.2f}s ({t_davidson/nk:.3f}s/k)")
print(f"Total:    {t_total:.2f}s")

# ── Compare with QE ──
qe_out = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/Si/02_si_4x4x4_nosym/qe/nscf_200band/nscf_100_1gpu.out"
with open(qe_out) as f:
    txt = f.read()

# Parse QE eigenvalues (handle negative k-coords like "0.1768-0.1768")
blocks = txt.split("bands (ev):")
qe_evals_all = []
for b in blocks[1:]:
    lines = b.strip().split("\n")
    vals = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("k") or line.startswith("Writing"):
            break
        vals.extend([float(x) for x in line.split()])
    if vals:
        qe_evals_all.append(np.array(vals) / 13.605698)

print(f"\nParsed {len(qe_evals_all)} QE k-points")

# Compare k=0 (Gamma)
if qe_evals_all:
    n_occ = 4
    qe0 = qe_evals_all[0]
    lx0 = all_evals[0]
    n_cmp = min(len(qe0), n_tgt)
    diff = lx0[:n_cmp] - qe0[:n_cmp]
    off = np.mean(diff[:n_occ])
    mae_occ = np.mean(np.abs(diff[:n_occ] - off))
    mae_all = np.mean(np.abs(diff[:n_cmp] - off))
    max_err = np.max(np.abs(diff[:n_cmp] - off))

    print(f"\n=== k=0 accuracy ===")
    print(f"  Occupied ({n_occ} bands) MAE: {mae_occ*1000:.4f} mRy")
    print(f"  All {n_cmp} bands MAE:        {mae_all*1000:.4f} mRy")
    print(f"  All {n_cmp} bands max err:     {max_err*1000:.4f} mRy")
    print(f"  Offset: {off*1000:.2f} mRy")

m = re.search(r'PWSCF\s+:\s+(\S+)s CPU\s+(\S+)s WALL', txt)
if m:
    qe_wall = float(m.group(2))
    print(f"\n=== Timing: 100 bands × 64 k-points ===")
    print(f"  QE     (1 GPU):  {qe_wall:.1f}s")
    print(f"  LORRAX (1 GPU):  {t_total:.1f}s (setup {t_setup:.1f} + warmup {t_warmup:.1f} + davidson {t_davidson:.1f})")
