import sys, importlib.util
spec = importlib.util.spec_from_file_location("old_gflat", "old_gflat.py")
old = importlib.util.module_from_spec(spec); sys.modules["old_gflat"]=old; spec.loader.exec_module(old)
from types import SimpleNamespace
meta = SimpleNamespace(nk_tot=9, nspinor=2, n_rmu=642, n_rmu_padded=642,
                       n_rtot=46080, ngkmax=1963, fft_grid=(24,24,80))
mesh = SimpleNamespace(shape={'x':1,'y':1})
pl = old.plan_gflat_chunks(meta=meta, mesh_xy=mesh, nb_total=160, ngkmax=1963,
        n_q_disk=9, budget_gb=28.0, target_utilization=0.80,
        fft_box_factor_A=4.0, is_bispinor=False, max_chunks=64)
print("OLD r_chunk =", pl.r_chunk, " band_chunk =", pl.band_chunk,
      " gflat_cs =", pl.gflat_chunk_size, " hwm =", pl.hwm_bytes/1e9)
