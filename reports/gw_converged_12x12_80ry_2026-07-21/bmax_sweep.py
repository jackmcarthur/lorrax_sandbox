"""Stage 4(d): does the htransform b_max ceiling stay lifted at 80 Ry?

Background.  The htransform builds an interpolant fH from bands [0, b_max) of
the ISDF basis.  With the OLD occupied-rho-weighted centroids the QP
reconstruction blew up once b_max reached into the far conduction bands --
`reports/scissor_farband_htransform_2026-07-20` measured on-grid QP recon
errors of 325 038 meV at [0,40) and 348 818 meV at [0,50) on the 6x6 / 30 Ry
run, i.e. the interpolant was unusable and b_max was effectively capped near
the valence edge.  With BAND-RANGE centroids the same sweep
(`reports/gw_bandrange_centroids_2026-07-21/bmax/bmax_br200.log`) came back
448 meV at [0,40) and 1191 meV at [0,80): the ceiling was LIFTED, because the
far-conduction states finally have ISDF quadrature support.

That was measured at 30 Ry / 6x6 / n_mu=1481.  This script repeats it on the
converged 80 Ry / 12x12 / n_mu=2412 run so the recommendation is made on the
converged data rather than inherited.

Metrics, per window [0, b_max):
  * DFTrec / QPrec -- max on-grid reconstruction error (meV).  Interpolating
    at the coarse k-points themselves must return the input; whatever it
    misses by is pure interpolation error.  THIS is the ceiling diagnostic.
  * DFTd2 / QPd2 -- max |second difference| along the path (meV), a
    smoothness proxy.
  * a_Ry / shift_eV -- the f-transform parameters, reported for the record.

Lean by design: the expensive Galerkin build happens once, then each window
is four `compute_wfns_fi` calls.  Runs on 1 GPU.

env: BM_RUN (run dir with gwbands.in / WFN.h5 / eqp1.dat), BM_OUT,
     BM_NVAL, BM_NCOND, BM_NBAND, BM_WINDOWS ("2,12,26,40,48,56,64,80")
"""
import os

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
from gw.gw_config import read_lorrax_input                    # noqa: E402
from bandstructure import htransform as ht                    # noqa: E402
from bandstructure.bse_setup import compute_wfns_fi           # noqa: E402
from bse.bse_w_exact import _create_mesh_xy                   # noqa: E402

RY = 13.6056980659
RUN = os.environ["BM_RUN"]
OUT = os.environ.get("BM_OUT", ".")
NVAL = int(os.environ.get("BM_NVAL", "26"))
NCOND = int(os.environ.get("BM_NCOND", "54"))
NBAND = int(os.environ.get("BM_NBAND", "80"))
WINS = [int(x) for x in os.environ.get(
    "BM_WINDOWS", "2,12,26,40,48,56,64,80").split(",")]
os.makedirs(OUT, exist_ok=True)


def parse_eqp(path):
    data, ik = {}, -1
    for line in open(path):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = s.split()
        if len(p) == 4 and "." in p[0]:
            ik += 1
            data[ik] = {}
        elif len(p) == 4:
            data[ik][int(p[1])] = (float(p[2]), float(p[3]))
    return data


PX = int(os.environ.get("BM_PX", "1"))
PY = int(os.environ.get("BM_PY", "1"))
mesh_xy = _create_mesh_xy(PX, PY)
print(f"[mesh] {PX}x{PY}", flush=True)
INP = os.path.join(RUN, "gwbands.in")
params = read_lorrax_input(INP)
params["nval"], params["ncond"], params["nband"] = NVAL, NCOND, NBAND
(wfn, sym, meta, _m, _S, ctilde, B, enk_dft) = ht.initialize_wfns(
    INP, params, print, mesh_xy=mesh_xy)
kg = (int(meta.nkx), int(meta.nky), int(meta.nkz))
nk = kg[0] * kg[1] * kg[2]
rank = int(ctilde.shape[2])
nb_ret = min(int(ctilde.shape[1]), rank)
enk_dft = np.asarray(jax.device_get(enk_dft))
if enk_dft.shape[0] == nk:
    enk_dft = enk_dft.T
print(f"[cfg] kg={kg} nk={nk} nb_ret={nb_ret} rank={rank}", flush=True)

eqp1 = parse_eqp(os.path.join(RUN, "eqp1.dat"))
dS1 = np.zeros_like(enk_dft)
for k in range(nk):
    for b in range(min(nb_ret, NBAND)):
        ed, e1 = eqp1[k][b + 1]
        dS1[b, k] = (e1 - ed) / RY
enk_qp = enk_dft + dS1

wfn0, _s0 = ht.setup_wfn_and_sym(os.path.join(RUN, "WFN.h5"))
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params)
kpath = np.asarray(kpath_frac)
kgrid = np.stack(np.meshgrid(np.arange(kg[0]) / kg[0],
                             np.arange(kg[1]) / kg[1], [0.0],
                             indexing="ij"), axis=-1).reshape(-1, 3)
print(f"[path] nQ={kpath.shape[0]} nodes={node_idx}", flush=True)


def interp(enk, b0, b1, qlist):
    bnd = compute_wfns_fi(
        ctilde=ctilde[:, b0:b1, :], B_at_mu=B,
        enk_sigma=jax.numpy.asarray(enk[b0:b1, :]), kgrid_co=kg,
        band_window_fi=(0, b1 - b0), mesh_xy=mesh_xy,
        q_list=jax.numpy.asarray(qlist), a_band_index=None,
        log_fn=(lambda *a, **k: None))
    return np.asarray(jax.device_get(bnd.enk_full)) * RY


def d2(E):
    return np.abs(E[2:] - 2 * E[1:-1] + E[:-2]).max(axis=0) * 1e3


def recon(Eg, enk, b0, b1):
    inp = np.sort((enk[b0:b1, :].T) * RY, axis=1)
    return np.abs(np.sort(Eg, axis=1) - inp).max(axis=0) * 1e3


print()
print(f"{'window':>10} {'a_Ry':>8} {'shift_eV':>9} {'DFTd2':>10} "
      f"{'QPd2':>10} {'DFTrec':>10} {'QPrec':>11}   (meV)")
print("-" * 76)
rows = []
for b1 in WINS:
    if b1 > nb_ret:
        continue
    Ep_d, Ep_q = interp(enk_dft, 0, b1, kpath), interp(enk_qp, 0, b1, kpath)
    Eg_d, Eg_q = interp(enk_dft, 0, b1, kgrid), interp(enk_qp, 0, b1, kgrid)
    rc_d, rc_q = recon(Eg_d, enk_dft, 0, b1), recon(Eg_q, enk_qp, 0, b1)
    _, a_f, _n, sh = ht.f_transform_eigs(
        jax.numpy.asarray(enk_dft[0:b1, :]), None)
    print(f"[ 0,{b1:>3})  {float(a_f):>8.4f} {float(sh)*RY:>9.2f} "
          f"{d2(Ep_d).max():>10.1f} {d2(Ep_q).max():>10.1f} "
          f"{rc_d.max():>10.1f} {rc_q.max():>11.1f}", flush=True)
    rows.append((b1, float(a_f), float(sh) * RY, float(d2(Ep_d).max()),
                 float(d2(Ep_q).max()), float(rc_d.max()), float(rc_q.max())))

np.savez(os.path.join(OUT, "bmax_sweep.npz"), rows=np.array(rows),
         wins=np.array(WINS))
print()
print("Ceiling reading: QPrec is the ceiling diagnostic.  On the 30 Ry rho-")
print("weighted run it exploded to 3.3e5 meV by [0,40).  Anything staying")
print("O(1e2-1e3) meV means the ceiling is still LIFTED.")
