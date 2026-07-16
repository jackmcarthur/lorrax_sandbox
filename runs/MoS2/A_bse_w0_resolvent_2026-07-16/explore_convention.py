"""Lock the non-TDA symplectic convention + factor for the W(0) resolvent
cross-check vs head-less (W0_qmunu - V_qmunu) q=0 tile."""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
jax.config.update("jax_enable_x64", True)

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_ring_comm import (
    build_bse_ring_matvec_full, make_bse_shardings,
    build_realspace_random_transition_generator,
    build_density_snapshot_operator,
)
from bse.bse_feast import gmres_solve_sharded_jit, build_preconditioner_diagonal_sharded, _apply_shifted_matvec

inp = sys.argv[1]
restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=999, n_cond=999, mesh_xy=mesh, input_file=inp, inject_head=False)
sh = make_bse_shardings(mesh)
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
nk = nkx * nky * nkz
n_rmu = int(data["V_q0"].shape[0]); n_rmu_log = int(data["n_rmu"])
data["W_R"] = data["W_q"]
print(f"window n_val={data['n_val']} n_cond={data['n_cond']} nk={nk} n_rmu={n_rmu}")

W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
V0 = np.asarray(jax.device_get(data["V_q0"]))
T = W0 - V0
col_norm = np.linalg.norm(T[:, :n_rmu_log], axis=0)
nu0 = int(np.argmax(col_norm))
tcol = T[:, nu0]
print(f"test col nu={nu0} ||T[:,nu]||={col_norm[nu0]:.4e} diag={T[nu0,nu0]:.4e}")

# ---- dense EXACT static chi0-only  vχ0v  (calibrate factor/Nk/sign) ----
pc = np.asarray(jax.device_get(data["psi_c_X"]))  # (nk, nc, ns, mu)
pv = np.asarray(jax.device_get(data["psi_v_X"]))  # (nk, nv, ns, mu)
ec = np.asarray(jax.device_get(data["eps_c"]))    # (nk, nc)
ev = np.asarray(jax.device_get(data["eps_v"]))    # (nk, nv)
M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), pv)  # (nk,nc,nv,mu) = conj(psi_c)psi_v
Delta = ec[:, :, None] - ev[:, None, :]           # (nk,nc,nv) = e_c - e_v  > 0
vM = np.einsum("MN,kcvN->kcvM", V0, M)             # v applied: (nk,nc,nv,mu)
# chi0(mu,nu) = -2/Nk sum_{kcv} (vM)_mu conj(vM)_nu / Delta   -> C0 = vχ0v
C0col = (-2.0 / nk) * np.einsum("kcvM,kcv,kcv->M", vM, np.conj(vM[:, :, :, nu0]), 1.0 / Delta)
rel0 = np.linalg.norm(C0col[:n_rmu_log] - tcol[:n_rmu_log]) / np.linalg.norm(tcol[:n_rmu_log])
r0 = C0col[nu0] / tcol[nu0]
print(f"[dense vχ0v exact] relerr(vsT)={rel0:.4e}  ratio@diag={r0:.4f}  diag={C0col[nu0]:.4e}")

# ---- resolvent sweep ----
matvec = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=False)
diag_h = build_preconditioner_diagonal_sharded(data, mesh, include_W=False, use_tda=False)
gen = build_realspace_random_transition_generator(
    mesh, nkx, nky, nkz, int(data["n_cond_pad"]), int(data["n_val_pad"]))
snap = build_density_snapshot_operator(mesh, nkx, nky, nkz)
z = 0.0 + 0.0j
g = jnp.zeros((n_rmu,), dtype=jnp.float64).at[nu0].set(1.0)
r = jax.device_put(jnp.broadcast_to(g[None, :, None], (1, n_rmu, nk)), sh.S)
f = jax.lax.with_sharding_constraint(gen(r, data["psi_c_X"], data["psi_v_X"], data["V_q0"]), sh.X)
fbar = jax.lax.with_sharding_constraint(
    gen(r, jnp.conj(data["psi_c_X"]), jnp.conj(data["psi_v_X"]), data["V_q0"]), sh.X)

def stats(tag, wc):
    wc = np.asarray(jax.device_get(wc)).reshape(-1)
    rel = np.linalg.norm(wc[:n_rmu_log] - tcol[:n_rmu_log]) / np.linalg.norm(tcol[:n_rmu_log])
    idx = np.argsort(-np.abs(tcol[:n_rmu_log]))[:20]
    ratio = wc[idx] / tcol[idx]
    print(f"    {tag:22s} relerr={rel:.4e}  ratio mean={np.mean(ratio):.4f} std={np.std(ratio):.4f}")

for ysign in (-1.0, +1.0):
    rhs = jnp.stack([f, ysign * fbar], axis=0).astype(jnp.complex128)
    rhs = jax.lax.with_sharding_constraint(rhs, sh.X_full)
    x, kit = gmres_solve_sharded_jit(matvec, diag_h, z, rhs, data, max_iter=160, tol=1e-11)
    gres = float(jnp.linalg.norm(rhs - _apply_shifted_matvec(matvec, x, z, data)) / jnp.linalg.norm(rhs))
    print(f"[ysign={ysign:+.0f}] iters={int(kit)} resid={gres:.2e}")
    sX = snap(x[0], data["psi_c_Y"], data["psi_v_Y"], data["V_q0"])
    sYc = snap(x[1], jnp.conj(data["psi_c_Y"]), jnp.conj(data["psi_v_Y"]), data["V_q0"])
    for rsign in (+1.0, -1.0):
        stats(f"snap(X){rsign:+.0f}snap*(Y)", sX + rsign * sYc)
