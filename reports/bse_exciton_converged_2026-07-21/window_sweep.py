"""htransform fH band-window sweep on the CONVERGED 12x12 / 80 Ry / n_mu=2412 data.

The question this answers
------------------------
``bandstructure/htransform.streaming_galerkin_solve`` builds the Galerkin basis
as an SVD of ``A = psi(centroids)`` reshaped ``(nk*nb, nspinor*n_mu)``.  Its
column rank can never exceed ``nspinor*n_mu``; its row count is ``nk*nb``.  When
the rows outnumber the columns the SVD cannot return an orthonormal row basis,
``ctilde`` stops being orthonormal, and ``fH = sum_n f(eps_n) c c^H`` no longer
recovers the input energies through its eigenvalues.  The capacity rule is
therefore

    nspinor * n_mu  >  nk * nb        <=>       nb  <  nspinor*n_mu / nk

and on this data (nspinor=2, n_mu=2412, nk=144) that is **nb < 33.5**.

The previous campaign (reports/gw_converged_12x12_80ry_2026-07-21 §4d/§5) held
nb fixed at 48/80 and concluded the interpolation needed n_mu ~ 5680.  That was
the requirement for a FULL 48-80 band window.  An 8v8c exciton needs only 16
bands plus guards, so this sweep drives nb DOWN instead of n_mu up.

What is measured, per window
----------------------------
* ``rank``           - SVD rank actually returned (= min(nk*nb, ns*n_mu) when healthy)
* ``ortho``          - ctilde[0] orthogonality error, printed by the Galerkin solve
* ``max|d eps_c|``   - the DRIVER's own gate metric, computed by importing
                       ``bse.exciton_bands.gate_htransform_vs_stored`` verbatim.
                       Limit 50 meV (assert is at 0.05 Ry; the tightened warn
                       is at 20 meV).
* ``min-sval``       - the same gate's conduction-subspace overlap.  Limit 0.5.
* ``recon``          - max |E_interp - E_stored| over the 8 BSE conduction bands
                       at the 144 ON-GRID k (pure interpolation error)
* ``d2``             - max |second difference| of those bands along Gamma-M-K-Gamma
                       (smoothness proxy; a broken interpolant rings)
* ``fH_R GiB/dev``   - nk_co * rank^2 * 16 B, the REPLICATED array in
                       ``bandstructure/bse_setup.compute_wfns_fi`` (line 156).
                       Independent of device count.

Everything is run twice: once on DFT energies, once on the CONVERGED G0W0 eqp1
energies (the ones the exciton run consumes).

Window convention
-----------------
``nval = ncond = nb/2``, so the fH window is bands ``[26-nb/2, 26+nb/2)`` -
CENTRED ON THE GAP, not anchored at band 0.  ``nband`` (= ``Meta.b_id_4_user``)
must be >= ``26+ncond`` or ``common/wfn_transforms.load_centroids_band_chunked``
ZEROES every band above it (``nb_user_in_range = max(0, b_id_4_user - b_start)``)
and the SVD returns rank 0.  That is what the previous campaign hit and recorded
as "a non-zero b_start is separately broken" - it is not broken, ``nband`` is an
ABSOLUTE band index and has to cover the window.

env: WS_RUN, WS_OUT, WS_WINDOWS, WS_PX, WS_PY, WS_NVEXC, WS_NCEXC
"""
import os

# exciton_bands owns the distributed bootstrap (set_default_env BEFORE jax is
# imported, init_jax_distributed BEFORE any device query).  Import it FIRST and
# take the driver's real gate from it - the metrics below are not a re-implementation.
from bse.exciton_bands import gate_htransform_vs_stored, _gather_host   # noqa: E402

import numpy as np                                               # noqa: E402
import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402
import h5py                                                      # noqa: E402

from gw.gw_config import read_lorrax_input                       # noqa: E402
from bandstructure import htransform as ht                       # noqa: E402
from bandstructure.bse_setup import compute_wfns_fi              # noqa: E402
from bse.bse_w_exact import _create_mesh_xy                      # noqa: E402
from bse.bse_io import read_bgw_eqp                              # noqa: E402

RY = 13.6056980659

RUN = os.environ["WS_RUN"]
OUT = os.environ.get("WS_OUT", ".")
WINS = [int(x) for x in os.environ.get("WS_WINDOWS", "16,20,24,28,32,36,40").split(",")]
PX = int(os.environ.get("WS_PX", "2"))
PY = int(os.environ.get("WS_PY", "2"))
NV_EXC = int(os.environ.get("WS_NVEXC", "8"))
NC_EXC = int(os.environ.get("WS_NCEXC", "8"))
BATCH = int(os.environ.get("WS_BATCH", "32"))
INP = os.path.join(RUN, os.environ.get("WS_INPUT", "gwbands.in"))
RESTART = os.environ.get("WS_RESTART",
                         os.path.join(RUN, "tmp", "isdf_tensors_2412.h5"))
EQP = os.path.join(RUN, "eqp1.dat")
os.makedirs(OUT, exist_ok=True)

_rank0 = jax.process_index() == 0


def log(*a, **k):
    if _rank0:
        print(*a, **k, flush=True)


mesh_xy = _create_mesh_xy(PX, PY)
log(f"[dist] devices={jax.device_count()} procs={jax.process_count()} mesh={PX}x{PY}")

params0 = read_lorrax_input(INP)

# ── stored reference: enk_full + psi_c straight off the GW restart ──────────
# Exactly the arrays load_bse_data_from_restart_sharded slices for eps_c /
# psi_c_X, read directly so the sweep does not have to pull the 134 GB
# V_qmunu / W0_qmunu tiles it has no use for.
NELEC = 26
with h5py.File(RESTART, "r") as f:
    enk_dft_full = np.asarray(f["enk_full"][:])              # (nk, nb_tot) Ry
    # the 8 BSE conduction bands, absolute [26, 26+NC_EXC) - the same slice
    # load_bse_data_from_restart_sharded takes for psi_c_X.
    psi_c_store = np.asarray(f["psi_full_y"][:, NELEC:NELEC + NC_EXC])
nk_tot = enk_dft_full.shape[0]

# ── QP energies from the converged eqp1.dat (shared BGW reader) ─────────────
kpts_e, e_dft_ev, e_qp_ev = read_bgw_eqp(EQP)
nk_e, nb_e = e_qp_ev.shape
assert nk_e == nk_tot, f"eqp1.dat nk={nk_e} != restart nk={nk_tot}"
# HARD CHECK that the eqp k-ordering matches the restart's full-BZ ordering:
# the file's own E_DFT column must reproduce the restart enk to round-off.
d_order = np.max(np.abs(e_dft_ev / RY - enk_dft_full[:, :nb_e]))
log(f"[eqp] {os.path.basename(EQP)}: nk={nk_e} nb={nb_e}; "
    f"max|E_DFT(file) - enk_full(restart)| = {d_order*RY*1e3:.4f} meV")
assert d_order * RY < 1e-4, "eqp1.dat k-ordering does not match the restart"
enk_qp_full = enk_dft_full.copy()
enk_qp_full[:, :nb_e] = e_qp_ev / RY

# ── k-grid and the Gamma-M-K-Gamma path (same K_POINTS block as the driver) ─
wfn0, _sym0 = ht.setup_wfn_and_sym(os.path.join(RUN, params0.get("wfn_file", "WFN.h5")))
kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(wfn0, params0)
kpath = np.asarray(kpath_frac, dtype=np.float64)
kg = (int(wfn0.kgrid[0]), int(wfn0.kgrid[1]), int(wfn0.kgrid[2]))
nk = kg[0] * kg[1] * kg[2]
k_frac = np.stack(np.meshgrid(np.arange(kg[0]) / kg[0], np.arange(kg[1]) / kg[1],
                              np.arange(kg[2]) / kg[2], indexing="ij"),
                  axis=-1).reshape(-1, 3)
log(f"[cfg] kgrid={kg} nk={nk} n_mu(from input) path_pts={kpath.shape[0]} "
    f"nodes={list(map(int, node_idx))}")
log(f"[cap] capacity rule nspinor*n_mu > nk*nb  ->  nb < {2*2412/nk:.2f} "
    f"(nspinor=2, n_mu=2412, nk={nk})")

import time                                                      # noqa: E402


def memstats(tag):
    s = jax.local_devices()[0].memory_stats() or {}
    log(f"   [mem:{tag}] in_use={s.get('bytes_in_use', 0)/2**30:7.2f} GiB  "
        f"peak={s.get('peak_bytes_in_use', 0)/2**30:7.2f} GiB  "
        f"reserved={s.get('bytes_reserved', 0)/2**30:7.2f} GiB  "
        f"limit={s.get('bytes_limit', 0)/2**30:7.2f} GiB  "
        f"largest_free={s.get('largest_free_block_bytes', 0)/2**30:7.2f} GiB")


rows = []
detail = {}
for nb in WINS:
    nval = nb // 2
    ncond = nb - nval
    b_start, b_end = NELEC - nval, NELEC + ncond
    if b_start < 0 or b_end > enk_dft_full.shape[1]:
        log(f"[skip] nb={nb}: window [{b_start},{b_end}) out of range")
        continue
    # nband == b_id_4_user must COVER the window (see module docstring).
    params = dict(params0)
    params["nval"], params["ncond"], params["nband"] = nval, ncond, b_end
    log(f"\n=== nb={nb}  window=[{b_start},{b_end})  nval={nval} ncond={ncond} "
        f"nband={b_end} ===")
    t0 = time.time()
    ortho = {}

    def _cap(msg, *a, **k):          # capture the Galerkin diagnostics verbatim
        s = " ".join(str(x) for x in (msg,) + a)
        if "orthogonality error" in s:
            ortho["val"] = float(s.split(":")[-1])
        if "SVD of" in s:
            ortho["svd"] = s.strip()
        log("   " + s.rstrip())

    try:
        (wfn, sym, meta, _m, _S, ctilde, B_at_mu, enk_sig_dft) = ht.initialize_wfns(
            INP, params, _cap, mesh_xy=mesh_xy)
    except Exception as exc:                                     # noqa: BLE001
        log(f"[FAIL] nb={nb}: Galerkin build raised {type(exc).__name__}: {exc}")
        rows.append(dict(nb=nb, rank=-1, status=f"build-error:{type(exc).__name__}"))
        continue
    t_gal = time.time() - t0
    rank = int(ctilde.shape[2])
    nb_ct = int(ctilde.shape[1])
    fH_R_bytes = nk * rank * rank * 16
    log(f"   rank={rank}  nb_ctilde={nb_ct}  Galerkin {t_gal:.1f}s  "
        f"fH_R(replicated) = {fH_R_bytes/1024**3:.2f} GiB/device")
    memstats("post-Galerkin")

    # BSE conduction sub-window, window-relative: exactly what exciton_bands does
    b_min, b_max = nval, nval + NC_EXC
    if b_max > nb_ct:
        log(f"[skip] nb={nb}: BSE window [{b_min},{b_max}) exceeds ctilde bands {nb_ct}")
        rows.append(dict(nb=nb, rank=rank, status="window-too-small"))
        continue
    n_guard = nb_ct - b_max

    enk_sig_dft = np.asarray(jax.device_get(enk_sig_dft))        # (nb, nk) Ry
    enk_sig_qp = enk_qp_full[:, b_start:b_end].T.copy()
    # consistency of the two energy sources for the same window
    d_src = float(np.max(np.abs(enk_sig_dft - enk_dft_full[:, b_start:b_end].T)))
    log(f"   |enk(WFN) - enk(restart)| over the window = {d_src*RY*1e3:.4f} meV")

    res = {}
    for tag, enk in (("DFT", enk_sig_dft), ("QP", enk_sig_qp)):
        # ONE compute_wfns_fi for the on-grid k AND the path: the replicated
        # fH_R is the peak, so build it once per energy set, not twice.
        t1 = time.time()
        q_all = np.concatenate([k_frac, kpath], axis=0)
        try:
            bnd = compute_wfns_fi(
                ctilde=ctilde, B_at_mu=B_at_mu, enk_sigma=jnp.asarray(enk),
                kgrid_co=kg, band_window_fi=(b_min, b_max), mesh_xy=mesh_xy,
                q_list=q_all, a_band_index=None, batch_size=BATCH,
                log_fn=(lambda *a, **k: None))
            eps_all = _gather_host(bnd.enk_full)                  # (nk+nQ, NC_EXC) Ry
            psi_grid = _gather_host(bnd.psi_rmu_Y)[:nk]           # (nk, NC_EXC, ns, n_mu)
            eps_grid, Ep = eps_all[:nk], eps_all[nk:] * RY
        except Exception as exc:                                  # noqa: BLE001
            log(f"   [{tag}] compute_wfns_fi raised {type(exc).__name__}: {exc}")
            memstats(f"{tag}-after-fail")
            res[tag] = dict(status=f"oom-or-error:{type(exc).__name__}")
            continue
        finally:
            bnd = None
            gc0 = __import__("gc"); gc0.collect()
        t_grid = time.time() - t1

        src = enk_dft_full if tag == "DFT" else enk_qp_full
        eps_st = src[:, b_start + b_min: b_start + b_max]          # (nk, NC_EXC) Ry
        recon = float(np.max(np.abs(np.sort(eps_grid, axis=1)
                                    - np.sort(eps_st, axis=1)))) * RY * 1e3

        # THE DRIVER'S OWN GATE, imported verbatim.  ``data`` carries exactly
        # the four keys it reads (eps_c, psi_c_X, n_cond, n_rmu).
        data = {"eps_c": jnp.asarray(eps_st), "psi_c_X": jnp.asarray(psi_c_store),
                "n_cond": NC_EXC, "n_rmu": int(B_at_mu.shape[2])}
        # The gate PRINTS its two numbers and then asserts.  A window that fails
        # is a legitimate sweep result, so capture the print and let the
        # AssertionError through as the FAIL verdict rather than killing the run.
        gate_lines = []

        def _glog(*a, **k):
            gate_lines.append(" ".join(str(x) for x in a))
            log(*a, **k)

        gate_refused = False
        try:
            gate_d, gate_s = gate_htransform_vs_stored(psi_grid, eps_grid, data,
                                                       log=_glog)
            gate_meV = gate_d * RY * 1e3
        except AssertionError as exc:                             # noqa: BLE001
            gate_refused = True
            txt = next((s for s in gate_lines if "[gate]" in s), "")
            gate_meV = float(txt.split("max|Δε_c| =")[1].split("meV")[0])
            gate_s = float(txt.split("min-sval =")[1])
            log(f"   [{tag}] gate REFUSED this configuration: "
                f"{str(exc).splitlines()[0]}")

        # path smoothness (Ep already computed in the same fH build above)
        d2 = float(np.max(np.abs(Ep[2:] - 2 * Ep[1:-1] + Ep[:-2]))) * 1e3
        t_path = 0.0

        # Diagnostic (NOT a second gate): where does min-sval come from?
        # fH_k = sum_n f(eps_n) c c^H has rank <= nb but lives on a (rank,rank)
        # face, so it carries rank-nb EXACT zero eigenvalues.  f(eps)->0 as eps
        # approaches the f-transform ``shift`` (= max energy of the window's TOP
        # band), so the highest bands of the window are numerically degenerate
        # with that null space and their eigenVECTORS are arbitrary.  If the
        # min-sval loss is concentrated in the last selected band, the cure is
        # guard bands, not centroids.
        A0 = psi_grid.reshape(nk, NC_EXC, -1)
        B0 = np.asarray(psi_c_store).reshape(nk, NC_EXC, -1)
        A0 = A0 / np.linalg.norm(A0, axis=2, keepdims=True)
        B0 = B0 / np.linalg.norm(B0, axis=2, keepdims=True)
        sv = np.stack([np.linalg.svd(A0[k].conj() @ B0[k].T, compute_uv=False)
                       for k in range(nk)])                        # (nk, NC_EXC)
        kworst = int(np.argmin(sv.min(axis=1)))
        # per-band |<psi_ht_b|psi_st_b>| (diagonal overlap), worst over k
        diag_ov = np.abs(np.einsum('kbm,kbm->kb', A0.conj(), B0))
        log(f"   [{tag}] sval spectrum @ worst k={kworst}: "
            + " ".join(f"{s:.3f}" for s in sv[kworst])
            + f" | per-band min_k |<ht|st>|: "
            + " ".join(f"{v:.3f}" for v in diag_ov.min(axis=0)))

        passed = (not gate_refused) and (gate_meV < 50.0) and (gate_s > 0.5)
        res[tag] = dict(gate_meV=gate_meV, min_sval=gate_s, recon_meV=recon,
                        d2_meV=d2, passed=bool(passed), refused=bool(gate_refused),
                        t_grid=t_grid, t_path=t_path, status="ok")
        log(f"   [{tag}] gate max|d eps_c| = {gate_meV:8.3f} meV   "
            f"min-sval = {gate_s:.4f}   recon = {recon:8.3f} meV   "
            f"d2 = {d2:8.2f} meV   ({'PASS' if passed else 'FAIL'})  "
            f"[{t_grid:.0f}s grid + {t_path:.0f}s path]")
        detail[f"{nb}_{tag}_path"] = Ep

    rows.append(dict(nb=nb, nval=nval, ncond=ncond, b_start=b_start, b_end=b_end,
                     rank=rank, n_guard=n_guard, ortho=ortho.get("val", np.nan),
                     fH_R_GiB=fH_R_bytes / 1024**3, t_galerkin=t_gal,
                     status="ok", **{f"{t}_{k}": v for t, d in res.items()
                                     for k, v in d.items()}))
    del ctilde, B_at_mu
    import gc
    gc.collect()

# ── table ──────────────────────────────────────────────────────────────────
log("\n" + "=" * 118)
log(f"{'nb':>4} {'window':>10} {'rank':>6} {'nk*nb':>7} {'ortho':>9} "
    f"{'fH_R GiB':>9} | {'DFT gate':>9} {'sval':>7} {'recon':>9} {'d2':>8} | "
    f"{'QP gate':>9} {'sval':>7} {'recon':>9} {'d2':>8} | verdict")
log("-" * 118)
for r in rows:
    if r.get("status") != "ok":
        log(f"{r['nb']:>4} {'':>10} {r.get('rank',-1):>6} {'':>7} "
            f"{'':>9} {'':>9} | {r.get('status','')}")
        continue
    v = ("PASS" if (r.get("QP_passed") and r.get("DFT_passed")) else "FAIL")
    log(f"{r['nb']:>4} [{r['b_start']:>2},{r['b_end']:>3})".ljust(16)
        + f"{r['rank']:>6} {nk*r['nb']:>7} {r['ortho']:>9.2e} {r['fH_R_GiB']:>9.2f} | "
        f"{r.get('DFT_gate_meV',float('nan')):>9.3f} {r.get('DFT_min_sval',float('nan')):>7.4f} "
        f"{r.get('DFT_recon_meV',float('nan')):>9.3f} {r.get('DFT_d2_meV',float('nan')):>8.2f} | "
        f"{r.get('QP_gate_meV',float('nan')):>9.3f} {r.get('QP_min_sval',float('nan')):>7.4f} "
        f"{r.get('QP_recon_meV',float('nan')):>9.3f} {r.get('QP_d2_meV',float('nan')):>8.2f} | {v}")
log("=" * 118)
log(f"capacity rule: nb < nspinor*n_mu/nk = {2*2412/nk:.2f}")

if _rank0:
    import json
    # One window per PROCESS (the shell loops): the replicated fH_R is 11-50
    # GiB and the BFC arena fragments badly when several windows share one
    # allocator, so each window gets a fresh process and appends its row here.
    jpath = os.path.join(OUT, "window_sweep.json")
    prev = []
    if os.path.exists(jpath):
        with open(jpath) as fh:
            prev = json.load(fh)
    keep = {int(r["nb"]) for r in rows}
    prev = [r for r in prev if int(r["nb"]) not in keep]
    with open(jpath, "w") as fh:
        json.dump(sorted(prev + rows, key=lambda r: int(r["nb"])), fh,
                  indent=1, default=float)
    npz = os.path.join(OUT, "window_sweep_paths.npz")
    old = dict(np.load(npz)) if os.path.exists(npz) else {}
    for _k in ("x_path", "node_idx", "kpath"):
        old.pop(_k, None)          # re-supplied below; duplicate kwarg = TypeError
    old.update(detail)
    np.savez(npz, x_path=np.asarray(x_path), node_idx=np.asarray(node_idx),
             kpath=kpath, **old)
    log(f"wrote {jpath}")
