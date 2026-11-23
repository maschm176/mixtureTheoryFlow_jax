"""
Profile/trace data for idx 797 under the two beta-aware ST closures (flat dedicated fit and pooled NN), stepped in
one-transit chunks with the production path (solver.advance_state, N=150, DEFAULT dt, beta=+2deg) to T_END transits.
Per transit: outlet-window median alpha_l, pipe-mean alpha_l, alpha_min/max, complex-eigenvalue cell count; snapshots of
alpha_l(x) at SNAP_T. Output: st_closure_row797_trace.pkl
"""
import math, pickle, sys, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
sys.argv = [sys.argv[0], 'base']
import equinox as eqx
import _pooled_sw_st_fit_nn_fi_only_beta as P
import mixtureTheoryFlow_jax_solver_water_gas as S

IDX, N, T_END, SNAP_T, BETA, L_VUONG = 797, 150, 150, (5, 25, 150), math.radians(2.0), 85.0
flat = pickle.load(open("st_fit_beta_result.pkl", "rb"))['A: SW production']['params']
nn = eqx.tree_deserialise_leaves("pooled_sw_st_nn_fi_only_beta_base.eqx", P.HybridParams(jax.random.PRNGKey(0)))
ff = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
r = ff[(ff['Fluids'] == 'Nitrogen - Isopar L') & (ff['Flow Pattern'] == 'ST')].loc[IDX]
U_sl, U_sg = float(r['vSL[m/s]']), float(r['vSG[m/s]']); D = float(r['Pipe Diameter[m]'])
rl, rg = float(r['rhoL[kg/m3]']), float(r['rhoG[kg/m3]']); ml, mg = float(r['viscosityL[Pa.s]']), float(r['viscosityG[Pa.s]'])
_, _, (f_nn, kl_nn, kg_nn) = P.predict_one(U_sl, U_sg, D, rl, rg, ml, mg, BETA, nn)
closures = {'flat_ST': tuple(flat), 'pooled_NN': (float(f_nn), float(kl_nn), float(kg_nn))}

Q = U_sl + U_sg
nd = S.compute_nondim_scales(U_sl, U_sg, D, rl, rg, ml, mg, 9.81, 0.0)
usl, usg = U_sl / Q, U_sg / Q
RL, RG, ML, MG, GS, DS = (nd[k] for k in ('rho_l_star', 'rho_g_star', 'mu_l_star', 'mu_g_star', 'g_star', 'D_star'))
Ls = L_VUONG / D; dx = Ls / N
out = {}
for name, (f, kl, kg) in closures.items():
    dt = float(S.estimate_stable_dt(usl, usg, DS, RL, RG, ML, MG, dx, f, kl, kg, 0.25, g=GS, beta=BETA))
    chunk = int(round(Ls / dt))
    ht = S.solve_taitel_dukler_holdup(U_sl, U_sg, D, rl, rg, ml, mg, f, kl, kg, beta=BETA)
    A, A_l, *_ = S.td_geometry(ht, D); eq = float(A_l / A)

    @jax.jit
    def run_chunk(s, f=f, kl=kl, kg=kg, dt=dt, chunk=chunk):
        def body(st_, _):
            ns, _k = S.advance_state(st_, dt, dx, 1.0, DS, RL, RG, ML, MG, usl, RL - RG, GS, f, kl, kg, beta=BETA, sigma=0.0)
            return ns, None
        return jax.lax.scan(body, s, None, length=chunk)[0]

    @jax.jit
    def n_complex(s):
        return jnp.sum(S.local_wave_speeds(s['alpha_l'], s['Delta'], 1.0, DS, RL, RG, GS, BETA)[2])

    rec = dict(params=(f, kl, kg), eq_alpha=eq, dt=dt, chunk=chunk, transit=[], outlet=[], mean=[], a_min=[], a_max=[], n_complex=[], snaps={})
    state = S.init_state_water_gas(N, usl, usg, RL, RG); t0 = time.time()
    for k in range(T_END + 1):
        a = np.asarray(state['alpha_l'])
        if np.isnan(a).any():
            rec['nan_at'] = k; break
        rec['transit'].append(k); rec['outlet'].append(float(np.median(a[-11:-1]))); rec['mean'].append(float(a.mean()))
        rec['a_min'].append(float(a.min())); rec['a_max'].append(float(a.max())); rec['n_complex'].append(int(n_complex(state)))
        if k in SNAP_T: rec['snaps'][k] = a.copy()
        if k == T_END: break
        state = run_chunk(state); jax.block_until_ready(state)
    out[name] = rec
    print(f"{name}: params={tuple(round(v,4) for v in (f,kl,kg))} eq={eq:.4f} final outlet={rec['outlet'][-1]:.4f} "
          f"mean={rec['mean'][-1]:.4f} n_complex={rec['n_complex'][-1]} nan_at={rec.get('nan_at')} {time.time()-t0:.0f}s", flush=True)
pickle.dump(out, open("st_closure_row797_trace.pkl", "wb"))
print("DONE", flush=True)
