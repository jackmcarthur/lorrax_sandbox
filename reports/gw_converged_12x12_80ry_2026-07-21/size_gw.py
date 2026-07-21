"""Stage-1b: size the downstream GW on the 80 Ry / 12x12x1 / 400-band MoS2
reference, using LORRAX's OWN production planner
(``gw.gflat_memory_model.plan_gflat_chunks``) — no hand arithmetic.

The planner is the same call ``gw.gw_init`` makes at the top of the ISDF
pipeline, with the same argument pattern (nb_total = (b3-b0)+(b4-b1),
n_q_disk = nk_tot, is_bispinor from cfg).  We feed it a ``common.meta.Meta``
built from the REAL WFN.h5 header, so n_rtot / nk_tot / nspinor / ngkmax are
measured, not assumed.

Must run with a real 4x4 device mesh (16 GPUs / 4 nodes) so the Stage-A/D FFT
box is XLA-QUERIED (true cuFFT plan scratch) rather than falling back to the
analytic factor.

    srun -N 4 -n 16 ... python3 -u size_gw.py <WFN.h5> [n_mu ...]
"""
import os
import sys

sys.path.insert(0, os.environ.get(
    "LORRAX_SRC", "/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src"))

from runtime import set_default_env
set_default_env()
import numpy as np
import jax
from jax.sharding import Mesh
from runtime import init_jax_distributed, fallback_to_cpu_if_no_gpu_backend

init_jax_distributed()
fallback_to_cpu_if_no_gpu_backend()

from file_io.mf_header import read_mf_header
from common.meta import Meta
from runtime.padding import padded_mu_extent
from gw.gflat_memory_model import plan_gflat_chunks

WFN = sys.argv[1]
MUS = [int(x) for x in sys.argv[2:]] or [1600, 2400, 3000, 4000]

# --- the GW the owner wants to run on this reference -----------------------
NVAL = int(os.environ.get("SIZE_NVAL", 26))     # Mo 14 + 2 x S 6
NBAND = int(os.environ.get("SIZE_NBAND", 326))  # sum-over-states -> b_id_4_user
# Sigma-window scenarios.  nb_total = (b3-b0) + (b4-b1) is what the planner
# sizes the four psi centroid copies with, and b3 = nelec + ncond, so the width
# of the Sigma QP window is a first-class memory lever independent of the
# screening sum.  SIZE_NCONDS overrides for control/validation runs.
_nconds = os.environ.get("SIZE_NCONDS")
if _nconds:
    SCENARIOS = [(f"ncond={c}", int(c)) for c in _nconds.split(",")]
else:
    SCENARIOS = [
        ("wide  Sigma (ncond=300, QP for bands 1..326)", 300),
        ("narrow Sigma (ncond=74,  QP for bands 1..100)", 74),
    ]
_budgets = os.environ.get("SIZE_BUDGETS", "28,60")
BUDGETS = tuple(float(b) for b in _budgets.split(","))

mf = read_mf_header(WFN)
fft_grid = tuple(int(x) for x in mf.fft_grid)
n_rtot = int(np.prod(fft_grid))
nk_tot = int(mf.nkpts)
nspinor = int(mf.nspinor)
ngkmax = int(mf.ngkmax)
nelec = NVAL       # occupied spinor bands (b_id_2)

# ngkmax the planner will ACTUALLY see.  ``gw_init`` does
#     _ngkmax = int(getattr(meta, 'ngkmax', 0)) or int(0.06 * meta.n_rtot)
# and ``common.meta.Meta`` has NO ``ngkmax`` field (verified: nothing in the
# tree ever assigns ``meta.ngkmax``), so the getattr branch is dead and the
# 0.06*n_rtot heuristic is what production always passes.  Default here = the
# production-faithful heuristic; SIZE_NGKMAX=true uses the WFN header value.
NGKMAX_TRUE = ngkmax
NGKMAX_PROD = int(0.06 * n_rtot)
_ngmode = os.environ.get("SIZE_NGKMAX", "prod")
ngkmax = NGKMAX_TRUE if _ngmode == "true" else NGKMAX_PROD
# band_chunk_size defaults to 16 in gw_config (NOT 0), so production always
# overrides the planner's band_chunk picker.  Mirror that unless told not to.
BAND_CHUNK_OVERRIDE = int(os.environ.get("SIZE_BAND_CHUNK", 16)) or None

mesh_xy = _mesh = None
total = jax.process_count() * jax.local_device_count()
gx = int(np.sqrt(total))
while gx > 1 and total % gx != 0:
    gx -= 1
mesh_xy = Mesh(np.array(jax.devices()).reshape(gx, total // gx), ['x', 'y'])
world = int(jax.device_count())

# band bookkeeping exactly as common.meta.Meta.from_system does
from runtime.padding import round_up

if jax.process_index() == 0:
    print("=" * 78)
    print("GW SIZING — MoS2 80 Ry / 12x12x1 reference")
    print("=" * 78)
    print(f"  WFN        : {WFN}")
    print(f"  mesh       : {gx} x {total // gx}  ({world} devices, "
          f"{jax.process_count()} processes, backend {jax.default_backend()})")
    print(f"  n_rtot     : {n_rtot}   FFT {fft_grid}")
    print(f"  nk = nq    : {nk_tot}")
    print(f"  nspinor    : {nspinor}")
    print(f"  ngkmax     : {ngkmax}   (mode '{_ngmode}': true={NGKMAX_TRUE}, "
          f"gw_init 0.06*n_rtot heuristic={NGKMAX_PROD})")
    print(f"  band_chunk : override {BAND_CHUNK_OVERRIDE} "
          f"(gw_config band_chunk_size default = 16)")
    print(f"  nband      : {NBAND} (screening sum-over-states)")
    print()

C = 16.0  # bytes / complex128
rows = []
for scen_name, NCOND in SCENARIOS:
  b0 = 0
  b1 = nelec - NVAL
  b2 = nelec
  b3 = nelec + NCOND
  b4 = round_up(NBAND, world)
  nb_total = (b3 - b0) + (b4 - b1)
  if jax.process_index() == 0:
      print("#" * 78)
      print(f"# SCENARIO: {scen_name}")
      print(f"#   band edges b0..b4 = {b0},{b1},{b2},{b3},{b4}   "
            f"nb_total = (b3-b0)+(b4-b1) = {nb_total}")
      print("#" * 78)
  for budget in BUDGETS:
    for mu in MUS:
        mu_pad = padded_mu_extent(mu, world)
        meta = Meta(
            rank=jax.process_index(), n_proc=jax.process_count(),
            b_id_0=b0, b_id_1=b1, b_id_2=b2, b_id_3=b3, b_id_4=b4,
            fft_grid=fft_grid, cell_volume=float(mf.cell_volume),
            n_rtot=n_rtot, n_rmu=mu, npol=1, nfreq=1,
            nspin=int(mf.nspin), nspinor=nspinor,
            nspinor_wfnfile=nspinor,
            nkx=int(mf.kgrid[0]), nky=int(mf.kgrid[1]), nkz=int(mf.kgrid[2]),
            nk_tot=nk_tot, n_rmu_padded=mu_pad, b_id_4_user=NBAND,
        )
        with mesh_xy:
            plan = plan_gflat_chunks(
                meta=meta, mesh_xy=mesh_xy,
                nb_total=nb_total, ngkmax=ngkmax,
                n_q_disk=nk_tot,
                budget_gb=budget,
                is_bispinor=False,
                max_chunks=64,
                band_chunk_override=BAND_CHUNK_OVERRIDE,
            )
        # raw quantities the owner asked for (aggregate, not per-rank)
        zeta_r_full = nk_tot * mu_pad * n_rtot * C          # zeta in r, all q
        zeta_r_chunk = nk_tot * mu_pad * plan.r_chunk * C    # one r-chunk, all q
        zeta_G_disk = nk_tot * mu_pad * ngkmax * C           # G-flat zeta on disk
        cct_stack = nk_tot * mu_pad * mu_pad * C             # L_q / CCT stack
        if jax.process_index() == 0:
            print("-" * 78)
            print(f"n_mu = {mu} (padded {mu_pad}), budget {budget:.0f} GB/dev, "
                  f"{world} devices")
            print(plan.format())
            print(f"    aggregate (all ranks, GB):")
            print(f"      CCT stack  n_q*n_mu^2 ......... {cct_stack/1e9:9.2f} "
                  f"({cct_stack/1e9/world:.2f}/dev)")
            print(f"      zeta(r) full n_q*n_mu*n_rtot .. {zeta_r_full/1e9:9.2f} "
                  f"(never resident; chunked over r)")
            print(f"      zeta(r) one r-chunk ........... {zeta_r_chunk/1e9:9.2f} "
                  f"({zeta_r_chunk/1e9/world:.2f}/dev)")
            print(f"      zeta(G) on disk n_q*n_mu*ngk .. {zeta_G_disk/1e9:9.2f}")
            rows.append(dict(
                scenario=scen_name, ncond=NCOND, nb_total=nb_total,
                budget=budget, mu=mu, mu_pad=mu_pad,
                hwm=plan.hwm_bytes / 1e9, persist=plan.persistent_bytes / 1e9,
                bottleneck=plan.bottleneck, p_min=plan.p_min,
                r_chunk=plan.r_chunk, n_r_chunks=plan.n_r_chunks,
                band_chunk=plan.band_chunk, q_chunk=plan.q_chunk,
                gflat_cs=plan.gflat_chunk_size,
                cct=cct_stack / 1e9, zr=zeta_r_full / 1e9,
                zg=zeta_G_disk / 1e9,
                feasible=(plan.hwm_bytes <= plan.budget_bytes
                          * plan.target_utilization
                          and plan.p_min <= world),
            ))

if jax.process_index() == 0:
    print()
    print("=" * 110)
    print("SUMMARY — 16 devices, MoS2 80 Ry 12x12 (n_rtot=%d, nq=144, nband=326)"
          % n_rtot)
    print("=" * 110)
    hdr = (f"{'ncond':>6} {'nb_tot':>7} {'budget':>7} {'n_mu':>6} {'HWM/dev':>9} "
           f"{'persist':>9} {'binder':>18} {'P_min':>6} {'r_chunk':>8} "
           f"{'#chunks':>8} {'CCT(GB)':>9} {'zetaG(GB)':>10} {'fits?':>6}")
    print(hdr)
    for r in rows:
        print(f"{r['ncond']:6d} {r['nb_total']:7d} {r['budget']:7.0f} "
              f"{r['mu']:6d} {r['hwm']:9.2f} "
              f"{r['persist']:9.2f} {r['bottleneck']:>18} {r['p_min']:6d} "
              f"{r['r_chunk']:8d} {r['n_r_chunks']:8d} {r['cct']:9.2f} "
              f"{r['zg']:10.2f} {'YES' if r['feasible'] else 'NO':>6}")
    import json
    with open(os.environ.get("SIZE_JSON", "gw_sizing.json"), "w") as fh:
        json.dump(dict(n_rtot=n_rtot, fft_grid=list(fft_grid), nk=nk_tot,
                       ngkmax=ngkmax, nspinor=nspinor, nband=NBAND,
                       nb_total=nb_total, world=world, rows=rows), fh, indent=2)
    print("\nwrote", os.environ.get("SIZE_JSON", "gw_sizing.json"))
