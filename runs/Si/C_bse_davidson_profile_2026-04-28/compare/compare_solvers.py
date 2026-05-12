"""Compare Lanczos vs Davidson eigenvalues / eigenvectors on the same BSE H.

Runs ``solve_bse_sharded`` twice on identical data + mesh, with
``solver_kind="lanczos"`` and ``solver_kind="davidson"``. Compares:

  * eigenvalue MAE (meV)
  * per-state |A^S(c,v,k)|^2 cosine similarity (gauge invariant)
  * cross-overlap matrix |<psi_lan_i | psi_dav_j>|^2 — diagonal close to
    1 means same eigenstates
"""
from __future__ import annotations
import sys, os
import numpy as np
import jax
import jax.numpy as jnp

# imports relative to the lorrax_C source tree on PYTHONPATH
from bse.bse_io import load_bse_data_from_restart_sharded, _find_restart_file
from bse.bse_lanczos import solve_bse_sharded
from bse.bse_ring_comm import create_mesh_2d


def _sigma_density(eigvec_flat):
    """|A^S(c,v,k)|^2 normalised — gauge-invariant fingerprint of state S."""
    # eigvec_flat shape: (n_eig, *trailing)  any layout
    rho = np.abs(eigvec_flat) ** 2
    rho = rho.reshape(rho.shape[0], -1)
    norm = rho.sum(axis=1, keepdims=True)
    return rho / np.maximum(norm, 1e-30)


def main():
    input_file = "cohsex_bse.in"
    n_val = 8; n_cond = 8; n_eig = 10

    restart_file = _find_restart_file(input_file)
    mesh_xy = create_mesh_2d()
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
        input_file=input_file, n_occ=None,
    )
    data["matvec_kind"] = "ring"

    print(f"=== solver: lanczos ===", flush=True)
    eig_l, vec_l, _ = solve_bse_sharded(
        data, mesh_xy, n_eig=n_eig, max_iter=200,
        n_reorth=200, include_W=True, block_size=1,
        rtol=0.0, atol=1e-10, solver_kind="lanczos",
    )
    eig_l = np.asarray(jax.device_get(eig_l))
    vec_l_np = np.asarray(jax.device_get(vec_l))   # (n_eig, 1, c, v, k)
    print(f"  eigvals (Ry): {eig_l[:n_eig]}", flush=True)

    # Reload data — block-Lanczos may have donated some buffers
    data = load_bse_data_from_restart_sharded(
        restart_file, n_val=n_val, n_cond=n_cond, mesh_xy=mesh_xy,
        input_file=input_file, n_occ=None,
    )
    data["matvec_kind"] = "ring"

    print(f"=== solver: davidson ===", flush=True)
    eig_d, vec_d, _ = solve_bse_sharded(
        data, mesh_xy, n_eig=n_eig, max_iter=80,
        include_W=True, block_size=1,
        atol=1e-10, solver_kind="davidson",
    )
    eig_d = np.asarray(jax.device_get(eig_d))
    vec_d_np = np.asarray(jax.device_get(vec_d))   # (n_eig, 1, c, v, k)
    print(f"  eigvals (Ry): {eig_d[:n_eig]}", flush=True)

    # === eigenvalue comparison ===
    ryd2mev = 13_605.6980659
    deig = (eig_d[:n_eig] - eig_l[:n_eig]) * ryd2mev
    print()
    print("=== eigenvalue agreement (Davidson - Lanczos) ===")
    for i in range(n_eig):
        print(f"  S={i:2d}  λ_lan={eig_l[i]*1e3*13.6057:9.3f} meV  "
              f"λ_dav={eig_d[i]*1e3*13.6057:9.3f} meV  Δ={deig[i]:+.3f} meV")
    print(f"  MAE = {np.mean(np.abs(deig)):.4f} meV   max = {np.max(np.abs(deig)):.4f} meV")

    # === per-state |A^S|^2 cosine similarity (gauge-invariant) ===
    rho_l = _sigma_density(vec_l_np)
    rho_d = _sigma_density(vec_d_np)
    cos = np.sum(np.sqrt(rho_l * rho_d), axis=1) ** 2  # squared Bhattacharyya / fidelity
    print()
    print("=== |A^S(c,v,k)|^2 fidelity per state (1=identical, gauge invariant) ===")
    for i in range(n_eig):
        print(f"  S={i:2d}  fidelity = {cos[i]:.4f}")
    print(f"  mean = {cos.mean():.4f}   min = {cos.min():.4f}")

    # === cross-overlap |<lan_i | dav_j>|^2 (matrix; diagonal = same state) ===
    # Flatten and treat (1, c, v, k) as a single vector axis.
    L = np.array(vec_l_np.reshape(n_eig, -1), copy=True)
    D = np.array(vec_d_np.reshape(n_eig, -1), copy=True)
    # normalise each row
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    M = np.abs(L.conj() @ D.T) ** 2  # (n_eig, n_eig) probabilities
    print()
    print("=== |<Lanczos S | Davidson S'>|^2 cross-overlap matrix ===")
    print("       " + "  ".join(f"D{j:1d}" for j in range(n_eig)))
    for i in range(n_eig):
        row = "  ".join(f"{M[i,j]:.2f}" for j in range(n_eig))
        print(f"  L{i:1d}  {row}")
    print(f"  diag mean = {np.mean(np.diag(M)):.4f}")
    print(f"  trace_sum = {np.trace(M):.4f}  (perfect = {n_eig})")


if __name__ == "__main__":
    main()
