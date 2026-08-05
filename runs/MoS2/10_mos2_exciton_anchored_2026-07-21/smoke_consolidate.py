import importlib, traceback
mods = ["gw.gw_jax","gw.ppm_sigma","gw.ppm_pipeline","gw.ppm_accumulators","gw.cohsex_sigma",
        "gw.w_isdf","gw.head_correction","gw.sigma_dispatch","gw.wavefunction_bundle",
        "gw.ppm_tau_kernel","gw.ppm_windows","gw.minimax_screening",
        "bse.vq_interp","bse.exciton_bands","bse.bse_jax","bse.bse_io","bse.bse_lanczos",
        "bandstructure.htransform","bandstructure.bse_setup","isdf.core",
        "ffi.common.dispatch","runtime","psp.finite_q_head_interp"]
bad=[]
for m in mods:
    try: importlib.import_module(m)
    except Exception as e: bad.append((m, f"{type(e).__name__}: {e}")); 
print(f"IMPORT: {len(mods)-len(bad)}/{len(mods)} ok")
for m,e in bad: print("  FAIL", m, e)
# key functionality still wired?
from ffi.common.dispatch import dispatch_eigh
from isdf.core import _resolve_solver_kind_charge, _replicate_charge_ok
from bandstructure.htransform import _GALERKIN_CHUNK_MAX_BYTES
import bse.exciton_bands as eb, inspect
src = inspect.getsource(eb)
print("dispatch_eigh:", callable(dispatch_eigh))
print("galerkin chunk cap GiB:", _GALERKIN_CHUNK_MAX_BYTES/1024**3)
print("rank_truncate honoured below cap:", _replicate_charge_ok(74,2412))
print("--extra-q wired:", "--extra-q" in src or "extra_q" in src)
print("cusolvermg gone:", not any("cusolvermg" in m for m in [src]))
