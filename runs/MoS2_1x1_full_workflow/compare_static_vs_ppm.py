import re, numpy as np

def parse_sigma_freq_debug(path):
    bands = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#') or s.startswith('k') or not s: continue
        p = s.split()
        if len(p) >= 13:
            try:
                n1 = int(p[1]) + 1
                # col indices: 5=sex_0, 6=coh_0, 7=x_bare, 8=sig_c(0), 12=sig_c(Edft)
                sex_0 = float(p[5])
                coh_0 = float(p[6])
                x_bare = float(p[7])
                sigc_0 = float(p[8])
                sigc_edft = float(p[12]) if p[12] != 'nan' else np.nan
                bands[n1] = {
                    'sex_0': sex_0, 'coh_0': coh_0, 'x_bare': x_bare,
                    'sigc_0': sigc_0, 'sigc_edft': sigc_edft,
                    'static_cor': (sex_0 - x_bare) + coh_0,
                }
            except (ValueError, IndexError):
                pass
    return bands

def parse_sigma_hp(path):
    bands = {}
    for line in open(path):
        p = line.strip().split()
        if len(p) >= 15 and p[0].isdigit():
            n = int(p[0])
            bands[n] = {'Corp': float(p[4]) + float(p[10]), 'SX_X': float(p[4]), 'CHp': float(p[10])}
    return bands

bgw_gn = parse_sigma_hp('00_bgw/sigma_hp.log')
bgw_coh = parse_sigma_hp('01_bgw_cohsex/sigma_hp.log')
gw = parse_sigma_freq_debug('00_lorrax/sigma_freq_debug.dat')

common = sorted(set(bgw_gn) & set(bgw_coh) & set(gw))

print("Static decomposition from PPM run vs BGW COHSEX and BGW GN-GPP:")
print(f"{'Band':>5} {'BGW GN':>9} {'BGW COH':>9} {'GW stat':>9} {'GW dyn':>9} | {'GW_s-BGW_c':>11} {'GW_d-BGW_g':>11}")
print("-" * 85)
ds, dd = [], []
for n in common:
    bg = bgw_gn[n]['Corp']
    bc = bgw_coh[n]['Corp']
    gs = gw[n]['static_cor']
    gd = gw[n]['sigc_edft'] if not np.isnan(gw[n]['sigc_edft']) else None
    d_static = gs - bc
    d_dyn = (gd - bg) if gd is not None else None
    ds.append(d_static)
    if d_dyn is not None: dd.append(d_dyn)
    gd_str = f"{gd:9.3f}" if gd is not None else "      nan"
    dd_str = f"{d_dyn:+11.3f}" if d_dyn is not None else "        nan"
    print(f"{n:5d} {bg:9.3f} {bc:9.3f} {gs:9.3f} {gd_str} | {d_static:+11.3f} {dd_str}")

ds = np.array(ds)
print(f"\nGWJAX static vs BGW COHSEX:  MAE = {np.mean(np.abs(ds)):.3f} eV")
dd = np.array(dd)
print(f"GWJAX dynamic vs BGW GN-GPP: MAE = {np.mean(np.abs(dd)):.3f} eV")
