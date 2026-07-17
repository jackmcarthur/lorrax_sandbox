"""FULL-BASIS W_q-v resolvent, all IBZ q, for the record (owner request 2026-07-17).

For each symmetry-reduced (IBZ) q on the MoS2 gnppm fixture, compute the ENTIRE
W_q - v tile via the resolvent engine `apply_screening_resolvent_block` seeded
with the IDENTITY probe block G_zeta = I(n_rmu) (the docstring's "full basis"
mode: every centroid column at once, not the 6/8-column spot check).  z = 0
(static W0).  Reuses the exact finite-q data path (`build_finite_q_data`) and the
compiled-ONCE block-GMRES engine (`_build_rpa_resolvent` built once, per-q
operands as runtime args), so only the first q pays the XLA compile.

Validation (stronger than the spot check): every logical column is compared to
the restart's own (W0_qmunu - V_qmunu)[q_flat] tile — max and median relative
column error per q, plus max GMRES residual over all columns.

usage:  full_basis_wq.py <input.in> [max_iter] [tol]
1 GPU (py=1); n_probe = n_rmu is trivially a multiple of py=1.
"""
import sys, time
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from bse.bse_io import _find_restart_file, load_bse_data_from_restart_sharded
from bse.bse_feast import ensure_W_R, build_preconditioner_diagonal_sharded
from bse.bse_w_exact import (
    _create_mesh_xy, _build_rpa_resolvent, build_finite_q_data,
    apply_screening_resolvent_block, _symmetry_reduced_q_list,
)

inp = sys.argv[1]
max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 300
tol = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
px, py = 1, 1
z = 0.0 + 0.0j

wall0 = time.perf_counter()

mesh_xy = _create_mesh_xy(px, py)
restart_file = _find_restart_file(inp)

# Full occ x cond chi0 window (matches GW compute_screening); load_v_full=True so
# build_finite_q_data can pull the V_qmunu[q] tiles.
data = load_bse_data_from_restart_sharded(
    restart_file, n_val=10**9, n_cond=10**9, mesh_xy=mesh_xy,
    input_file=inp, inject_head=False, load_v_full=True)

n_rmu = int(data["V_q0"].shape[0])
nlog = int(data["n_rmu"])
nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
nk = nkx * nky * nkz
n_cond_pad = int(data["n_cond_pad"]); n_val_pad = int(data["n_val_pad"])

# Build the q-INDEPENDENT resolvent engine ONCE (matvec/gen/snapshot depend only
# on mesh/kgrid/pad sizes, never on q).  Every q reuses this one compiled engine.
matvec, _, gen, snapshot, sh = _build_rpa_resolvent(mesh_xy, data)

# FULL-BASIS probe block: the identity on the padded centroid space.  Column i of
# the returned tile is W_q - v for probe e_i; n_probe = n_rmu (%py==1 -> ok).
G_full = np.eye(n_rmu, dtype=np.float64)

q_list = _symmetry_reduced_q_list(inp)

print(f"[cfg] px={px} py={py}  N_mu(logical)={nlog} (padded {n_rmu})  "
      f"nc={n_cond_pad} nv={n_val_pad}  kgrid={nkx}x{nky}x{nkz} (nk={nk})")
print(f"[cfg] restart={restart_file}")
print(f"[cfg] FULL-BASIS identity probe: n_probe={n_rmu}  z={z}  "
      f"gmres(max_iter={max_iter}, tol={tol:g})  head-less bodies")
print(f"[cfg] {len(q_list)} symmetry-reduced (IBZ) q-points, one at a time; each "
      f"vs its OWN (W0_qmunu-V_qmunu)[q_flat] tile\n")

hdr = (f"{'iq':>3} {'q (kgrid)':>11} {'q_flat':>6} {'n_cols':>6} "
       f"{'build[s]':>9} {'solve[s]':>9} {'total[s]':>9} "
       f"{'max_rel':>11} {'med_rel':>11} {'max_resid':>11} {'n_bad':>6}")
print(hdr)
print("-" * len(hdr))

rows = []
bad_total = 0
for iq, qv in enumerate(q_list):
    qx, qy, qz = int(qv[0]), int(qv[1]), int(qv[2])
    q_flat = qx * nky * nkz + qy * nkz + qz

    # ---- BUILD: q-shifted operands + preconditioner diagonal + disk target ----
    t_b0 = time.perf_counter()
    dq = build_finite_q_data(data, (qx, qy, qz), mesh_xy)
    ensure_W_R(dq, include_W=False)
    diag_hq = build_preconditioner_diagonal_sharded(
        dq, mesh_xy, include_W=False, use_tda=False)
    jax.block_until_ready(diag_hq)
    W0 = np.asarray(jax.device_get(data["W_q"][:, :, qx, qy, qz]))
    Vq = np.asarray(jax.device_get(data["V_q_full"][:, :, qx, qy, qz]))
    T = W0 - Vq                                   # (n_rmu, n_rmu) disk target
    t_build = time.perf_counter() - t_b0

    # ---- SOLVE: full identity-block scan-GMRES (seed+solve+project engine) ----
    # apply_screening_resolvent_block does SEED (zeta->pair), the per-column-scan
    # GMRES SOLVE, and the reduce-scatter PROJECT (pair->zeta) internally; we time
    # the whole compiled-once engine call.  First q carries the one-time compile.
    t_s0 = time.perf_counter()
    W_tile, resids = apply_screening_resolvent_block(
        G_full, z, dq, matvec, diag_hq, gen, snapshot, sh,
        max_iter=max_iter, tol=tol)
    jax.block_until_ready((W_tile, resids))
    t_solve = time.perf_counter() - t_s0

    # ---- COMPARE: EVERY logical column vs the disk (W0-V)[q_flat] tile ----
    wc = np.asarray(jax.device_get(W_tile))       # (n_rmu, n_probe)
    rr = np.asarray(jax.device_get(resids))       # (n_probe,)
    tnorm = np.linalg.norm(T[:nlog, :nlog], axis=0)      # per-column ||T||
    dnorm = np.linalg.norm(wc[:nlog, :nlog] - T[:nlog, :nlog], axis=0)
    nz = tnorm > 0.0
    rels = np.zeros(nlog, dtype=np.float64)
    rels[nz] = dnorm[nz] / tnorm[nz]
    max_rel = float(rels.max()); med_rel = float(np.median(rels))
    max_resid = float(rr[:nlog].max())
    # non-converged columns: GMRES residual above 10x the requested tol
    n_bad = int(np.count_nonzero(rr[:nlog] > 10.0 * tol))
    bad_total += n_bad

    total = t_build + t_solve
    print(f"{iq:3d} {str((qx, qy, qz)):>11} {q_flat:6d} {nlog:6d} "
          f"{t_build:9.3f} {t_solve:9.3f} {total:9.3f} "
          f"{max_rel:11.3e} {med_rel:11.3e} {max_resid:11.3e} {n_bad:6d}")
    rows.append(dict(iq=iq, q=(qx, qy, qz), q_flat=q_flat, n_cols=nlog,
                     build=t_build, solve=t_solve, total=total,
                     max_rel=max_rel, med_rel=med_rel, max_resid=max_resid,
                     n_bad=n_bad))

print("-" * len(hdr))
wall = time.perf_counter() - wall0
solve_sum = sum(r["solve"] for r in rows)
build_sum = sum(r["build"] for r in rows)
grand_max_rel = max(r["max_rel"] for r in rows)
grand_max_resid = max(r["max_resid"] for r in rows)
print(f"\nGRAND TOTAL wall = {wall:.3f}s  (build sum {build_sum:.3f}s, "
      f"solve sum {solve_sum:.3f}s incl. iq=0 one-time engine compile)")
print(f"worst-case over all q: max_rel={grand_max_rel:.3e}  "
      f"max_gmres_resid={grand_max_resid:.3e}  non-converged cols total={bad_total}")
print("note: iq=0 solve carries the one-time XLA compile of the block-GMRES "
      "engine; later q are warm dispatch (per-q recompile eliminated).")
print("note: solve[s] times the full apply_screening_resolvent_block engine "
      "(SEED + identity-block scan-GMRES + reduce-scatter PROJECT).")
if bad_total > 0:
    print(f"WARNING: {bad_total} column(s) did not reach tol={tol:g} at "
          f"max_iter={max_iter}; RAISE max_iter (do not loosen tol) and rerun.")
