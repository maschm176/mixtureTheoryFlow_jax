"""
Parity-panel data for the Fan (2005) INCLINED rows: short run (10 transits, before the
instability reaches the outlet window) vs. long run (150 transits), same solver path
(run_real_data_condition, N=150, default closure f_i_mult=k_wl=k_wg=1) as
_inclined_real_data_check.py. Only the 34 inclined rows that have a measured holdup.

Sharded so several processes can run at once:  python3 _fan_incl_instability_cache.py <shard> <n_shards>
Writes fan_incl_cache_shard<k>.pkl after every row (resumable).
Sign convention (positive angle = uphill) is ASSUMED, not verified against the Fan paper.
"""
import math, os, pickle, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver
from _fan_data import EXCLUDE_KH_USL

shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
TRANSITS = (10.0, 150.0)
OUT = f"fan_incl_cache_shard{shard}.pkl"
cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

d = pd.read_csv("fan2005_sw_subset.csv")
d['hl'] = pd.to_numeric(d['HL[-]'], errors='coerce')
rows = d[(d['Inclination angle[°]'] != 0) & d['hl'].notna()]
mine = [(i, r) for k, (i, r) in enumerate(rows.iterrows()) if k % n_shards == shard]
print(f"shard {shard}/{n_shards}: {len(mine)} rows", flush=True)

for idx, r in mine:
    if idx in cache and all(t in cache[idx]['runs'] for t in TRANSITS):
        continue
    U_sl, U_sg = float(r['vSL[m/s]']), float(r['vSG[m/s]'])
    D, L = float(r['Pipe Diameter[m]']), float(r['Pipe Length[m]'])
    rho_l, rho_g = float(r['rhoL[kg/m3]']), float(r['rhoG[kg/m3]'])
    mu_l, mu_g = float(r['viscosityL[Pa.s]']), float(r['viscosityG[Pa.s]'])
    angle = float(r['Inclination angle[°]']); beta = math.radians(angle)
    ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                                 1.0, 1.0, 1.0, beta=beta))
    A, A_l, *_ = solver.td_geometry(ht, D)
    rec = cache.setdefault(idx, dict(U_sl=U_sl, U_sg=U_sg, angle=angle, HL=float(r['hl']),
                                     cf_alpha_l=float(A_l / A),
                                     kh_flagged=any(abs(U_sl - x) < 1e-9 for x in EXCLUDE_KH_USL),
                                     runs={}))
    for nt in TRANSITS:
        t0 = time.time()
        res = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N=150,
                                             n_transit_times=nt, beta=beta)
        rec['runs'][nt] = dict(alpha_l_pred=res['alpha_l_pred'], kh_total=res['kh_total'],
                               any_nan=res['any_nan'], alpha_min=res['alpha_min'],
                               alpha_max=res['alpha_max'], n_steps=res['n_steps'],
                               seconds=time.time() - t0)
        print(f"  idx={idx} {angle:+.0f}deg U_sl={U_sl:.4f} T={nt:>5g}: a_pred={res['alpha_l_pred']:.5f} "
              f"cf={rec['cf_alpha_l']:.5f} HL={rec['HL']} kh={res['kh_total']} nan={res['any_nan']} "
              f"{rec['runs'][nt]['seconds']:.0f}s", flush=True)
    pickle.dump(cache, open(OUT, "wb"))
print("DONE", flush=True)
