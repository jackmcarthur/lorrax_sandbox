"""Localize the RPA-screening resolvent convention: folded static form,
symplectic with/without B-block, vs validated dense-RPA target T=W0-V (q=0)."""
import sys
import numpy as np
import jax
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)
from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded

inp = sys.argv[1]; restart = _find_restart_file(inp)
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
data = load_bse_data_from_restart_sharded(
    restart, n_val=999, n_cond=999, mesh_xy=mesh, input_file=inp, inject_head=False)
nk = int(data["nkx"]) * int(data["nky"]) * int(data["nkz"]); nlog = int(data["n_rmu"])
V0 = np.asarray(jax.device_get(data["V_q0"]))[:nlog, :nlog]
W0 = np.asarray(jax.device_get(data["W_q"][:, :, 0, 0, 0]))[:nlog, :nlog]
T = W0 - V0
pc = np.asarray(jax.device_get(data["psi_c_X"]))[:, :int(data["n_cond"]), :, :nlog]
pv = np.asarray(jax.device_get(data["psi_v_X"]))[:, :int(data["n_val"]), :, :nlog]
ec = np.asarray(jax.device_get(data["eps_c"]))[:, :int(data["n_cond"])]
ev = np.asarray(jax.device_get(data["eps_v"]))[:, :int(data["n_val"])]
Mmat = np.einsum("kcsm,kvsm->kcvm", np.conj(pc), pv).reshape(-1, nlog)   # (Nt,mu)
Delta = (ec[:, :, None] - ev[:, None, :]).reshape(-1)
Nt = Mmat.shape[0]
nu0 = int(np.argmax(np.linalg.norm(T, axis=0))); tcol = T[:, nu0]

def cmp(tag, col):
    rel = np.linalg.norm(col - tcol) / np.linalg.norm(tcol)
    idx = np.argsort(-np.abs(tcol))[:20]
    print(f"{tag:34s} relerr={rel:.4e} ratio(top20)={np.mean(col[idx]/tcol[idx]):.5f}")

KA = (1.0 / nk) * (np.conj(Mmat) @ V0 @ Mmat.T)
KB = (1.0 / nk) * (np.conj(Mmat) @ V0 @ np.conj(Mmat).T)

# (1) folded static:  chi = -(1/Nk) M^T (Delta/2 + KA)^{-1} conj(M);  T=V0 chi V0
chi_fold = -(1.0 / nk) * (Mmat.T @ np.linalg.solve(np.diag(Delta / 2) + KA, np.conj(Mmat)))
cmp("folded -(1/Nk)M(D/2+KA)^-1 Mc", (V0 @ chi_fold @ V0)[:, nu0])

# vertices (with the gen/snap 1/sqrt(Nk) each -> 1/Nk total)
e = np.zeros(nlog); e[nu0] = 1.0; Ve = V0 @ e
f = (np.conj(Mmat) @ Ve) / np.sqrt(nk)      # gen
fbar = (Mmat @ Ve) / np.sqrt(nk)            # conj-gen
def wc_of(H, ysign, rsign):
    x = np.linalg.solve(-H, np.concatenate([f, ysign * fbar]))
    X, Y = x[:Nt], x[Nt:]
    return (V0 @ (Mmat.T @ X) + rsign * (V0 @ (np.conj(Mmat).T @ Y))) / np.sqrt(nk)

A = np.diag(Delta) + KA
# SAME vertex in both blocks (RPA ring):  source=[f;-f], readout = V0 M^T (X + Y)
def wc_same(H, ysign, rsign):
    x = np.linalg.solve(-H, np.concatenate([f, ysign * f]))
    X, Y = x[:Nt], x[Nt:]
    return (V0 @ (Mmat.T @ (X + rsign * Y))) / np.sqrt(nk)
cmp("ring B=KA same-vtx (y-1,r+1)", wc_same(np.block([[A, KA], [-KA, -A]]), -1, +1))
cmp("ring B=KA same-vtx (y+1,r+1)", wc_same(np.block([[A, KA], [-KA, -A]]), +1, +1))
cmp("ring B=KB same-vtx (y-1,r+1)", wc_same(np.block([[A, KB], [-KB, -A]]), -1, +1))
