"""DRAFT of src/bse/bse_nontda.py — tested standalone against the current tree
before placement.  Structure-preserving non-TDA (full BSE) eigensolver.

Physics (per-element, validated on the gnppm fixture, see PHASE2_LOG):
  The optical non-TDA operator is the para-Hermitian
      H = [[A, B], [-B*, -A*]]                       (* = complex conjugate)
  with A Hermitian (A = A^H) and B complex-symmetric (B = B^T).  Its eigenvalues
  are real and come in +-omega pairs.  The equivalent Hermitian-definite pencil is
      K z = omega Sigma z,   K = [[A, B], [B*, A*]] (Hermitian),  Sigma = diag(I, -I)
  and K is positive definite for a stable spectrum.  Lowest positive omega are
  the extreme eigenvalues of the Hermitian  Shat = K^{-1/2} Sigma K^{-1/2}
  (eigenvalue 1/omega); eigenvectors z=[X;Y] carry X^H X - Y^H Y = +1.

  (A +- B) actions compose from the full matvec VERBATIM (no new kernel):
      matvec([U;  U])[X-block] = A U + B U = (A + B) U
      matvec([U; -U])[X-block] = A U - B U = (A - B) U
  because H[U; sU] = [A U + s B U ; ...].  A and B individually:
      matvec([U; 0])[X-block] = A U ;   matvec([0; U])[X-block] = B U.
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)


def _materialize_A_B(matvec_full, mv_args, sh, nc, nv, nk, chunk=None):
    """Materialize the A and B blocks (N x N, N=nc*nv*nk) from the full matvec.

    A[:,i] = (matvec [e_i;0])[X-block];  B[:,i] = (matvec [0;e_i])[X-block].
    Batched over the identity basis; ``chunk`` limits the live batch (memory)."""
    N = nc * nv * nk
    eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk)
    zero = np.zeros_like(eye)

    def _apply_block(top):
        Xf = jnp.asarray(np.stack([top, (eye if top is zero else zero)], axis=0)) \
            if False else None
        return None

    def _cols(which):  # which='A' resonant, 'B' antiresonant
        if which == 'A':
            Xf = np.stack([eye, zero], axis=0)   # (2, N, nc,nv,nk)
        else:
            Xf = np.stack([zero, eye], axis=0)
        Xf = jax.lax.with_sharding_constraint(jnp.asarray(Xf), sh.X_full)
        out = matvec_full(Xf, *mv_args)          # (2, N, nc,nv,nk)
        top = np.asarray(out[0]).reshape(N, N)   # rows=col-index, flat=output
        return top.T                             # column i = matvec output for e_i
    A = _cols('A')
    B = _cols('B')
    return A, B


def _solve_definite_pencil(A, B, n_eig, *, pd_tol=1e-10):
    """Structure-preserving solve of K z = omega Sigma z (host, dense).

    Returns (omega_sorted[:n_eig], Z[:, :n_eig]) with Z Sigma-normalised so
    X^H X - Y^H Y = +1.  Raises on non-Hermitian / indefinite K (triplet
    instability => (A-B) or K not positive definite; physical, not hidden)."""
    N = A.shape[0]
    A = np.asarray(A); B = np.asarray(B)
    herm = np.linalg.norm(A - A.conj().T) / max(np.linalg.norm(A), 1e-300)
    if herm > 1e-6:
        raise ValueError(f"non-TDA A block is not Hermitian (rel {herm:.2e}); "
                         "the resonant block must be Hermitian for a real spectrum.")
    K = np.block([[A, B], [B.conj(), A.conj()]])
    K = 0.5 * (K + K.conj().T)
    wK = np.linalg.eigvalsh(K)
    if wK.min() <= pd_tol * wK.max():
        raise ValueError(
            f"non-TDA definite pencil K=[[A,B],[B*,A*]] is NOT positive definite "
            f"(min eig {wK.min():.3e} <= {pd_tol:.0e} * max {wK.max():.3e}). "
            "This is a triplet/charge instability: (A-B) is not positive definite, "
            "so the BSE has imaginary excitation energies. Not hidden — fix the "
            "input (screening/window) or use a stabilised kernel.")
    w, U = np.linalg.eigh(K)
    Kmh = (U * (1.0 / np.sqrt(w))) @ U.conj().T          # K^{-1/2}
    Sig = np.concatenate([np.ones(N), -np.ones(N)])
    Shat = Kmh @ (Sig[:, None] * Kmh)
    Shat = 0.5 * (Shat + Shat.conj().T)
    mu, Y = np.linalg.eigh(Shat)                         # mu = 1/omega (+/- pairs)
    pos = mu > 0
    idx = np.where(pos)[0]
    idx = idx[np.argsort(-mu[idx])][:n_eig]              # largest 1/omega = lowest omega
    omega = 1.0 / mu[idx]
    Z = Kmh @ Y[:, idx]                                  # z solves K^{-1}Sigma z = mu z
    # Sigma-normalise: X^H X - Y^H Y = +1
    Xh = Z[:N]; Yh = Z[N:]
    snorm = np.real(np.sum(np.conj(Xh) * Xh, axis=0) - np.sum(np.conj(Yh) * Yh, axis=0))
    Z = Z / np.sqrt(np.abs(snorm))[None, :]
    order = np.argsort(omega)
    return omega[order], Z[:, order]


# --- test harness against the fixture (current tree) ---
if __name__ == "__main__":
    import sys
    from jax.sharding import Mesh
    RESTART, INPUT = sys.argv[1], sys.argv[2]
    from bse.bse_io import load_bse_data_from_restart_sharded
    from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)
    data = load_bse_data_from_restart_sharded(RESTART, n_val=2, n_cond=2, mesh_xy=mesh,
                                              input_file=INPUT)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["n_cond_pad"]); nv = int(data["n_val_pad"])
    from common.fft_helpers import make_sharded_ifftn_3d
    W_ifft = make_sharded_ifftn_3d(mesh, sh.W.spec, sh.W.spec, axes=(2, 3, 4), norm='ortho')
    with mesh:
        W_R = W_ifft(data["W_q"])
        mv = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=True, screening=False)
        args = (data["psi_c_X"], data["psi_c_Y"], data["psi_v_X"], data["psi_v_Y"],
                data["eps_c"], data["eps_v"], W_R, data["V_q0"], data["M_X"], data["M_Y"])
        A, B = _materialize_A_B(mv, args, sh, nc, nv, nk)
    print(f"nc={nc} nv={nv} nk={nk} N={nc*nv*nk}")
    print(f"A Herm {np.linalg.norm(A-A.conj().T)/np.linalg.norm(A):.2e}  "
          f"B sym {np.linalg.norm(B-B.T)/np.linalg.norm(B):.2e}  "
          f"B Herm {np.linalg.norm(B-B.conj().T)/np.linalg.norm(B):.2e}")
    omega, Z = _solve_definite_pencil(A, B, n_eig=6)
    print("omega6 (solver):", omega)
    # reference: dense SHAO
    Href = np.block([[A, B], [-B.conj(), -A.conj()]])
    ev = np.linalg.eigvals(Href).real
    print("omega6 (dense) :", np.sort(ev[ev > 1e-9])[:6])
    N = nc * nv * nk
    for j in range(3):
        z = Z[:, j]; X = z[:N]; Y = z[N:]
        sn = np.real(np.conj(X)@X - np.conj(Y)@Y)
        res = np.linalg.norm(Href @ z - omega[j]*z)/np.linalg.norm(z)
        print(f"  omega={omega[j]:.6f} X^HX-Y^HY={sn:+.4f} residual={res:.2e}")
