import sys; sys.path.insert(0,"/pscratch/sd/j/jackm/lorrax_sandbox/sources/lorrax_D/src")
import numpy as np, jax
from jax.sharding import Mesh
from types import SimpleNamespace
from gw.gflat_memory_model import plan_gflat_chunks, _stage_C_slope, _c128
devs = jax.devices()
print("ndev", len(devs))
m = Mesh(np.asarray(devs[:4]).reshape(2,2), axis_names=('x','y'))
print("real mesh.shape:", dict(m.shape), " shape['x']=", m.shape['x'], type(m.shape['x']))
meta = SimpleNamespace(nk_tot=9,nspinor=2,n_rmu=640,n_rmu_padded=640,
                       n_rtot=108000,ngkmax=5545,fft_grid=(30,30,120))
for tag,mesh in [("REAL_Mesh", m), ("SimpleNS", SimpleNamespace(shape={'x':2,'y':2}))]:
    pl = plan_gflat_chunks(meta=meta, mesh_xy=mesh, nb_total=40, ngkmax=5545,
            n_q_disk=9, budget_gb=10.0, is_bispinor=True, max_chunks=64)
    print(f"{tag}: r_chunk={pl.r_chunk} hwm={pl.hwm_bytes/1e9:.3f} "
          f"persist={pl.persistent_bytes/1e9:.3f} C={pl.peak_breakdown['C_fit_one_rchunk']/1e9:.3f}")
