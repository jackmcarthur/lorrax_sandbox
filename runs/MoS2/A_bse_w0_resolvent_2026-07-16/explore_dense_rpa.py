"""Dense static RPA W-v vs disk (W0_qmunu - V_qmunu): validate chi0 convention
and get the minimax-noise floor. Everything head-less, q=0 tile."""
import sys
import numpy as np
import jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)
from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded

inp = sys.argv[1]; restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=999, n_cond=999, mesh_xy=mesh, input_file=inp, inject_head=False)
nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"])
nlog = int(data["n_rmu"])
V0 = np.asarray(jax.device_get(data["V_q0"]))          # (mu,mu) head-less v
W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))
T = (W0 - V0)[:nlog, :nlog]
pc = np.asarray(jax.device_get(data["psi_c_X"]))       # (nk,nc,ns,mu)
pv = np.asarray(jax.device_get(data["psi_v_X"]))
ec = np.asarray(jax.device_get(data["eps_c"])); ev = np.asarray(jax.device_get(data["eps_v"]))
V0 = V0[:nlog, :nlog]
M = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), pv)[:, :, :, :nlog]  # conj(psi_c)psi_v
Delta = (ec[:, :, None] - ev[:, None, :])              # e_c-e_v

# chi0 density-basis: chi0(mu,nu) = -2/Nk sum_kcv M(mu) conj(M(nu)) / Delta   (exact 1/x)
Mf = M.reshape(-1, nlog)                                # (Ntrans, mu)
w = (-2.0 / nk) / Delta.reshape(-1)                     # (Ntrans,)
chi0 = np.einsum("tM,t,tN->MN", Mf, w, np.conj(Mf))     # (mu,mu)

I = np.eye(nlog)
# solve_w: W = (I - V chi0)^{-1} V   ->  W - V = (I - V chi0)^{-1} V chi0 V
Trpa = np.linalg.solve(I - V0 @ chi0, V0 @ chi0 @ V0)

def cmp(tag, A):
    rel = np.linalg.norm(A - T) / np.linalg.norm(T)
    d = np.diag(A); dt = np.diag(T)
    ratio = np.mean(d[np.argsort(-np.abs(dt))[:20]] / dt[np.argsort(-np.abs(dt))[:20]])
    print(f"{tag:26s} relerr_vs_disk={rel:.4e}  diag ratio(top20)={ratio:.5f}")

cmp("dense vχ0v (1st order)", V0 @ chi0 @ V0)
cmp("dense full RPA W-v", Trpa)
print(f"T-disk diag[0:3]={np.diag(T)[:3]}")
print(f"Trpa    diag[0:3]={np.diag(Trpa)[:3]}")
