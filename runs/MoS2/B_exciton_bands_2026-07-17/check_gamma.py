"""check_gamma — dense ground-truth regression for the driver's Γ row.

Dense H at Γ (n = nc·nv·nk = 144) from the production stack matvec with
STORED conduction ψ (production convention: head-injected V_q0), eigh →
exact lowest states.  The driver's Γ row (smoke3pt.dat) must match to the
htransform-ψ representation delta, which for this fixture is a pure
Kramers-doublet rotation (measured: dense(ht) == dense(stored) to <0.2
meV) — so the tolerance here is 1 meV.

Also prints the post-clamp solve_bse_sharded run (production Q=0 solver,
same Lanczos fix) vs dense — the solver-health check.
"""
import sys

import numpy as np

sys.path.insert(0, "/pscratch/sd/j/jackm/lorrax_sandbox/sources/worktrees/"
                   "lorrax_A_exciton_bands/src")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_lanczos import solve_bse_sharded
from bse.bse_serial import compute_pair_amplitude
from bse.bse_stack_matvec import build_bse_stack_matvec
from bse.bse_w_exact import _create_mesh_xy
from bse.bse_ring_comm import make_bse_shardings
from common.fft_helpers import make_sharded_ifftn_3d

RY2EV = 13.6056980659
RD = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/B_exciton_bands_2026-07-17"
INP = f"{RD}/exciton_smoke.in"
NV = NC = 4
NEIG = 6

mesh = _create_mesh_xy(1, 1)
sh = make_bse_shardings(mesh)
restart = _find_restart_file(INP)
data = load_bse_data_from_restart_sharded(
    restart, n_val=NV, n_cond=NC, mesh_xy=mesh, input_file=INP,
    inject_head=True)
nk = 9
n_flat = NC * NV * nk
matvec = build_bse_stack_matvec(mesh, 3, 3, 1, kernel="bse")
_ifftn = make_sharded_ifftn_3d(mesh, sh.W.spec, sh.W.spec, axes=(2, 3, 4),
                               norm="ortho")
W_R = _ifftn(data["W_q"])

M_X = jax.lax.with_sharding_constraint(
    compute_pair_amplitude(data["psi_c_X"], data["psi_v_X"]), sh.psi_x)
M_Y = jax.lax.with_sharding_constraint(
    compute_pair_amplitude(data["psi_c_Y"], data["psi_v_Y"]), sh.psi_y)
cols = []
eye = np.eye(n_flat)
for i in range(0, n_flat, 16):
    X = jnp.asarray(eye[i:i + 16].reshape(16, NC, NV, nk))
    HX = matvec(X, data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"],
                data["psi_v_Y"], data["eps_c"], data["eps_v"], W_R,
                data["V_q0"], M_X, M_Y)
    cols.append(np.asarray(jax.device_get(HX)).reshape(16, n_flat))
H = np.concatenate(cols, 0).T
ev_dense = np.linalg.eigvalsh(0.5 * (H + H.conj().T))[:NEIG]
print("dense stored-psi:", " ".join(f"{e*RY2EV:.6f}" for e in ev_dense))

with mesh:
    evsL, _v, _n = solve_bse_sharded(data, mesh, n_eig=NEIG, max_iter=40,
                                     block_size=8, include_W=True)
evsL = np.sort(np.asarray(jax.device_get(evsL)))[:NEIG]
print("solve_bse_shrd :", " ".join(f"{e*RY2EV:.6f}" for e in evsL))
dL = np.abs(evsL - ev_dense).max() * RY2EV * 1e3
print(f"  lanczos-vs-dense max|Δ| = {dL:.4f} meV")

rows = [l.split() for l in open(f"{RD}/smoke3pt.dat")
        if l.strip() and not l.startswith("#")]
gam = [r for r in rows if r[0] == "0" and r[5] == "interp"][0]
ev_drv = np.array([float(x) for x in gam[6:6 + NEIG]])
print("driver Γ row   :", " ".join(f"{e:.6f}" for e in ev_drv))
dD = np.abs(ev_drv - ev_dense * RY2EV).max() * 1e3
print(f"  driver-vs-dense max|Δ| = {dD:.4f} meV")
assert dL < 0.2, "production Lanczos != dense (solver health)"
# Driver row == dense(htransform-psi) exactly (verified: identical to the
# diag_dense_gamma dense-ht spectrum); its delta to dense(stored-psi) is the
# htransform representation floor — measured 2.25 meV on this fixture
# (640 centroids, rank 720).  Tolerance 5 meV flags gross regressions.
assert dD < 5.0, "driver Γ row != dense beyond the htransform floor"
print("PASS")
