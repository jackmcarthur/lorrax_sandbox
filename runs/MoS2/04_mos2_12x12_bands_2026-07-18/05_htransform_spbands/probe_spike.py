"""Attribute the isolated cross-window gate spikes (iQ 16, 20, 30) to a
window: fine Q-scan (13 points between path[iQ-1] and path[iQ+1]) of the
implicated bands in all three windows.  The spiky window shows a
non-smooth curve; the clean ones agree and vary smoothly."""
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from gw.gw_config import read_lorrax_input
from bandstructure import htransform as ht
from bandstructure.bse_setup import compute_wfns_fi
from bse.bse_w_exact import _create_mesh_xy

RY2EV = 13.6056980659
mesh_xy = _create_mesh_xy(1, 1)

wins = {}
for tag, inp, aband in [("ref2432", "hts_cond.in", None),
                        ("val2028", "hts_val.in", None),
                        ("cond2634", "hts_cond2.in", 2)]:
    params = read_lorrax_input(inp)
    (wfn, sym, meta, _m, _S, ctilde, B, enk) = ht.initialize_wfns(
        inp, params, print, mesh_xy=mesh_xy)
    wins[tag] = (ctilde, B, enk, aband, int(params["nband"]) - 8)
    if tag == "ref2432":
        kpath_frac, x_path, node_idx, node_labels, _ = ht.initialize_kpath(
            wfn, params)
kpath = np.asarray(kpath_frac)
kgrid_co = (12, 12, 1)

# band -> window index maps (window bottom absolute band stored above)
CASES = [(16, [26, 30, 31]), (20, [24, 28, 29]), (30, [25, 26])]
NS = 13

for iQ, bands in CASES:
    ts = np.linspace(0.0, 1.0, NS)
    scan = np.stack([kpath[iQ - 1] * (1 - t) + kpath[iQ + 1] * t for t in ts])
    E = {}
    for tag, (ctilde, B, enk, aband, b0) in wins.items():
        bnd = compute_wfns_fi(ctilde=ctilde, B_at_mu=B, enk_sigma=enk,
                              kgrid_co=kgrid_co, band_window_fi=(0, 8),
                              mesh_xy=mesh_xy, q_list=scan,
                              a_band_index=aband)
        E[tag] = np.asarray(jax.device_get(bnd.enk_full))  # (NS, 8)
    for b in bands:
        print(f"\n== iQ {iQ}, band {b} (eV) — scan path[{iQ-1}] -> path[{iQ+1}]")
        print("  t     " + "".join(f"{tag:>12s}" for tag in wins
                                   if b - wins[tag][4] in range(8)))
        for j, t in enumerate(ts):
            row = f"{t:5.2f}  "
            for tag, (_c, _B, _e, _a, b0) in wins.items():
                if b - b0 in range(8):
                    row += f"{E[tag][j, b - b0] * RY2EV:12.5f}"
            print(row)
