import sys, numpy as np, jax, jax.numpy as jnp
from jax.sharding import Mesh
jax.config.update("jax_enable_x64", True)
from bse.bse_io import load_bse_data_from_restart_sharded
from bse.bse_lanczos import solve_bse_sharded
mesh = Mesh(np.array(jax.devices()[:1]).reshape(1,1), axis_names=("x","y"))
data = load_bse_data_from_restart_sharded(sys.argv[1], n_val=2, n_cond=2, mesh_xy=mesh, input_file=sys.argv[2])
print("dense TDA ref lowest4: [0.008189 0.008317 0.010682 0.010763]  (N=36)")
for bs, mi, nr in [(4,7,7),(4,7,2),(4,5,5),(4,4,4)]:
    ev,_,_ = solve_bse_sharded(data, mesh, n_eig=4, max_iter=mi, include_W=True, block_size=bs, n_reorth=nr, rtol=0.0)
    ev = np.sort(np.asarray(jax.device_get(ev)).real)
    print(f"bs={bs} max_iter={mi} (Krylov={bs*mi}) n_reorth={nr}: {ev}")
