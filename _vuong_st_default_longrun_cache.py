"""
Default-closure (f_i_mult = k_wl = k_wg = 1.0) long-run data for the 23 high-holdup inclined Vuong ST rows.

Question: does running longer help the DEFAULT closure here the way it helped Fan's inclined rows?
Each row is stepped at 0.25x CFL dt (the largest dt that avoided NaN for all 23 rows per the investigation
file -- at default dt, 8/23 go NaN within 10 transits, so a long run isn't possible there) to T_END nominal
transits, in one-transit chunks. After every transit we record: outlet-window median alpha_l (the quantity
run_real_data_condition reports as alpha_l_pred), pipe-mean alpha_l, alpha_min/max, and the number of
complex-eigenvalue cells. Full alpha_l(x) snapshots at SNAP_T. One run therefore gives both the "short"
(10 transits) and "long" (50) values at the same dt.

Sharded: python3 _vuong_st_default_longrun_cache.py <shard> <n_shards>  -> vuong_st_default_longrun_shard<k>.pkl
(resumable; saved after every row). Positive angle = uphill (+2 deg), L=85 m assumed as in the earlier checks.
"""
import math, os, pickle, sys, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
N, MULT, T_END, SNAP_T = 150, 0.25, 50, (10, 25, 50)
BETA, L_VUONG = math.radians(2.0), 85.0
OUT = f"vuong_st_default_longrun_shard{shard}.pkl"
cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

ff = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ff[(ff['Fluids'] == 'Nitrogen - Isopar L') & (ff['Flow Pattern'] == 'ST')]
ROWS = [797, 805, 820, 821, 822, 826, 832, 840, 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,
        823, 836, 1523, 1524, 1543, 1550]          # the 23 genuinely-high-holdup rows (cf_alpha_l > 0.1)
mine = [i for k, i in enumerate(ROWS) if k % n_shards == shard]
print(f"shard {shard}/{n_shards}: rows {mine}", flush=True)

for idx in mine:
    if idx in cache:
        continue
    r = st.loc[idx]
    U_sl, U_sg = float(r['vSL[m/s]']), float(r['vSG[m/s]'])
    D = float(r['Pipe Diameter[m]']); rho_l, rho_g = float(r['rhoL[kg/m3]']), float(r['rhoG[kg/m3]'])
    mu_l, mu_g = float(r['viscosityL[Pa.s]']), float(r['viscosityG[Pa.s]'])
    hl = float(pd.to_numeric(r['HL[-]'], errors='coerce'))
    ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=BETA))
    A, A_l, *_ = solver.td_geometry(ht, D); cf = float(A_l / A)

    Q = U_sl + U_sg
    nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, 0.0)
    usl, usg = U_sl / Q, U_sg / Q
    rl, rg, ml, mg, gs, Ds = (nd[k] for k in ('rho_l_star', 'rho_g_star', 'mu_l_star', 'mu_g_star', 'g_star', 'D_star'))
    Ls = L_VUONG / D; dx = Ls / N
    dt = MULT * float(solver.estimate_stable_dt(usl, usg, Ds, rl, rg, ml, mg, dx, 1.0, 1.0, 1.0, 0.25, g=gs, beta=BETA))
    chunk = int(round(Ls / dt))                       # steps per ONE nominal transit

    @jax.jit
    def run_chunk(s):
        def body(st_, _):
            ns, _k = solver.advance_state(st_, dt, dx, 1.0, Ds, rl, rg, ml, mg, usl, rl - rg, gs, beta=BETA, sigma=0.0)
            return ns, None
        f, _ = jax.lax.scan(body, s, None, length=chunk)
        return f

    @jax.jit
    def n_complex(s):
        _, _, c = solver.local_wave_speeds(s['alpha_l'], s['Delta'], 1.0, Ds, rl, rg, gs, BETA)
        return jnp.sum(c)

    rec = dict(U_sl=U_sl, U_sg=U_sg, HL=hl, cf_alpha_l=cf, dt=dt, chunk=chunk,
               transit=[], outlet=[], mean=[], a_min=[], a_max=[], n_complex=[], snaps={})
    state = solver.init_state_water_gas(N, usl, usg, rl, rg)
    t0 = time.time()
    for k in range(T_END + 1):
        a = np.asarray(state['alpha_l'])
        if np.isnan(a).any():
            rec['nan_at'] = k; break
        rec['transit'].append(k); rec['outlet'].append(float(np.median(a[-11:-1]))); rec['mean'].append(float(a.mean()))
        rec['a_min'].append(float(a.min())); rec['a_max'].append(float(a.max())); rec['n_complex'].append(int(n_complex(state)))
        if k in SNAP_T:
            rec['snaps'][k] = a.copy()
        if k == T_END:
            break
        state = run_chunk(state); jax.block_until_ready(state)
    rec['seconds'] = time.time() - t0
    cache[idx] = rec
    pickle.dump(cache, open(OUT, "wb"))
    last = rec['transit'][-1] if rec['transit'] else None
    print(f"  idx={idx} U_sl={U_sl:.4f} U_sg={U_sg:.2f} HL={hl:.3f} cf={cf:.3f} chunk={chunk} "
          f"nan_at={rec.get('nan_at')} last_transit={last} outlet@10={rec['outlet'][10] if len(rec['outlet'])>10 else None} "
          f"outlet@end={rec['outlet'][-1] if rec['outlet'] else None} {rec['seconds']:.0f}s", flush=True)
print("DONE", flush=True)
