"""
Dynamic-solver check of the two beta-aware ST calibrations on all 23 inclined (+2 deg) Vuong ST rows:
  flat_ST   : the beta-threaded dedicated fit (f_i_mult, k_wl, k_wg) = st_fit_beta_result.pkl  (3 scalars)
  pooled_NN : the beta-threaded pooled SW+ST fit, 'base' variant (pooled_sw_st_nn_fi_only_beta_base.eqx); the NN gives
              each row its OWN f_i_mult, k_wl/k_wg are the fitted flat scalars
Protocol = the production dynamic solver exactly as in _st_fit_stability_check.py: run_real_data_condition, N=150,
default dt, 150 nominal transits, beta=+2deg, L=85 m. Per row also records the closure's own closed-form equilibrium
holdup (same beta) so we can see whether the transient solver relaxes to it, plus the measured HL / dP/dL.
Sharded: python3 <this> <shard> <n_shards> -> st_closure_stability_shard<k>.pkl (resumable).
"""
import math, os, pickle, sys, time
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]      # keep shard args; the pooled module reads argv[1] as a tag
shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
sys.argv = [sys.argv[0], 'base']                         # tag for the pooled module's feature standardization
import equinox as eqx
import _pooled_sw_st_fit_nn_fi_only_beta as P
import mixtureTheoryFlow_jax_solver_water_gas as S

BETA, L_VUONG, N, N_TRANSIT = math.radians(2.0), 85.0, 150, 150.0
ROWS = [797, 805, 820, 821, 822, 826, 832, 840, 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,
        823, 836, 1523, 1524, 1543, 1550]
TRAIN, TEST = set(P.ST_TRAIN_IDX), set(P.ST_TEST_IDX)
OUT = f"st_closure_stability_shard{shard}.pkl"
cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

flat = pickle.load(open("st_fit_beta_result.pkl", "rb"))['A: SW production']['params']
nn = eqx.tree_deserialise_leaves("pooled_sw_st_nn_fi_only_beta_base.eqx", P.HybridParams(jax.random.PRNGKey(0)))
ff = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ff[(ff['Fluids'] == 'Nitrogen - Isopar L') & (ff['Flow Pattern'] == 'ST')]
mine = [i for k, i in enumerate(ROWS) if k % n_shards == shard]
print(f"shard {shard}/{n_shards}: rows {mine}; flat params {flat}", flush=True)

def eq_alpha(U_sl, U_sg, D, rl, rg, ml, mg, p):
    ht = S.solve_taitel_dukler_holdup(U_sl, U_sg, D, rl, rg, ml, mg, *p, beta=BETA)
    A, A_l, *_ = S.td_geometry(ht, D)
    return float(A_l / A)

for idx in mine:
    if idx in cache:
        continue
    r = st.loc[idx]
    U_sl, U_sg = float(r['vSL[m/s]']), float(r['vSG[m/s]']); D = float(r['Pipe Diameter[m]'])
    rl, rg = float(r['rhoL[kg/m3]']), float(r['rhoG[kg/m3]']); ml, mg = float(r['viscosityL[Pa.s]']), float(r['viscosityG[Pa.s]'])
    hl = float(pd.to_numeric(r['HL[-]'], errors='coerce')); dp = float(r['dP/dL[Pa/m]'])
    _, _, (f_nn, kl_nn, kg_nn) = P.predict_one(U_sl, U_sg, D, rl, rg, ml, mg, BETA, nn)
    closures = {'flat_ST': tuple(flat), 'pooled_NN': (float(f_nn), float(kl_nn), float(kg_nn))}
    rec = dict(U_sl=U_sl, U_sg=U_sg, HL=hl, dpdz=dp, split='train' if idx in TRAIN else 'test' if idx in TEST else 'no HL', runs={})
    for name, p in closures.items():
        t0 = time.time()
        res = S.run_real_data_condition(U_sl, U_sg, D, rl, rg, ml, mg, L_VUONG, N=N, n_transit_times=N_TRANSIT, beta=BETA,
                                        f_i_mult=p[0], k_wl=p[1], k_wg=p[2])
        rec['runs'][name] = dict(params=p, eq_alpha=eq_alpha(U_sl, U_sg, D, rl, rg, ml, mg, p),
                                 alpha_l_pred=res['alpha_l_pred'], dpdz_pred=abs(res['dPdz_pred_liquid']), kh_total=res['kh_total'],
                                 any_nan=res['any_nan'], alpha_min=res['alpha_min'], alpha_max=res['alpha_max'],
                                 n_steps=res['n_steps'], seconds=time.time() - t0)
        x = rec['runs'][name]
        print(f"  idx={idx} {name:>9} f={p[0]:.3f} nan={x['any_nan']} kh={x['kh_total']} a_pred={x['alpha_l_pred']:.4f} "
              f"eq={x['eq_alpha']:.4f} HL={hl:.3f} range=[{x['alpha_min']:.4f},{x['alpha_max']:.4f}] {x['seconds']:.0f}s", flush=True)
    cache[idx] = rec
    pickle.dump(cache, open(OUT, "wb"))
print("DONE", flush=True)
