"""
Reynolds-stress overlay data for the inclined Vuong ST row idx=797 (+2 deg).

Question: with the (bounded, implicit) Reynolds-stress closure switched on at several
magnitudes, are the instabilities (spike trains in alpha_l, complex-eigenvalue cells) still
present?  Uses advance_state_rs_imex from _reynolds_stress_implicit_test.py UNCHANGED, and
the same nu_eff formulas as run_real_data_condition_rs_imex (copied below), but steps in
chunks so we can save profiles and a complex-cell trace (the wrapper returns scalars only).

N=1800 on purpose: the RS scripts found RS has ~zero effect at N<=300, so N=150 would be an
unfair test. Horizon is SHORT (N_TRANSIT=2) to keep cost down -- at N=150 spikes were already
visible within ~1-2 nominal transit times. dt = default CFL (same estimate_stable_dt the RS
wrapper uses, cfl=0.25). Runs: no-RS (nu=0), then nu_t = 1x, 100x, 100000x NU_T_DEFAULT.
Resumable; writes vuong_st_rs_overlay_cache.pkl after each run.
"""
import math, os, pickle, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver
from _reynolds_stress_implicit_test import advance_state_rs_imex, NU_T_DEFAULT, C_EPS_DEFAULT

N, IDX, N_TRANSIT, CHUNK, SNAP_EVERY = 1800, 797, 2.0, 200, 10
MULTS = [0.0, 1.0, 100.0, 1e5]            # 0.0 == RS off (baseline at the SAME resolution)
BETA, L_VUONG, CFL = math.radians(2.0), 85.0, 0.25
OUT = "vuong_st_rs_overlay_cache.pkl"
cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}

st = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
row = st[(st['Fluids'] == 'Nitrogen - Isopar L') & (st['Flow Pattern'] == 'ST')].loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]']); rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81)
usl, usg = U_sl / Q, U_sg / Q
rl, rg, ml, mg, gs, Ds = (nd[k] for k in ('rho_l_star', 'rho_g_star', 'mu_l_star', 'mu_g_star', 'g_star', 'D_star'))
Ls = L_VUONG / D; dx = Ls / N
dt = solver.estimate_stable_dt(usl, usg, Ds, rl, rg, ml, mg, dx, 1.0, 1.0, 1.0, CFL, g=gs, beta=BETA)
n_steps = int(np.ceil(N_TRANSIT * Ls / float(dt)))
n_chunks = n_steps // CHUNK
print(f"idx={IDX} N={N} dt={float(dt):.4e} n_steps={n_steps} ({N_TRANSIT} nominal transits), chunks={n_chunks}", flush=True)

@jax.jit
def run_chunk(state, nu_l, nu_g):           # nu_* traced -> one compile shared by all runs
    def body(s, _):
        ns, _ = advance_state_rs_imex(s, dt, dx, 1.0, Ds, rl, rg, ml, mg, usl, rl - rg,
                                      nu_l, nu_g, gs, 1.0, 1.0, 1.0, 1.3, BETA)
        return ns, None
    f, _ = jax.lax.scan(body, state, None, length=CHUNK)
    return f

@jax.jit
def count_complex(s):
    _, _, c = solver.local_wave_speeds(s['alpha_l'], s['Delta'], 1.0, Ds, rl, rg, gs, BETA)
    return jnp.sum(c)

for mult in MULTS:
    if mult in cache:
        continue
    if mult == 0.0:
        nu_l = nu_g = 0.0
    else:                                     # same formulas as run_real_data_condition_rs_imex
        nu_t_star = (NU_T_DEFAULT * mult) / (Q * D)
        nu_l = C_EPS_DEFAULT * (ml + nu_t_star)
        nu_g = C_EPS_DEFAULT * (mg / rg + nu_t_star)
    state = solver.init_state_water_gas(N, usl, usg, rl, rg)
    tr = dict(step=[], n_complex=[], a_min=[], a_max=[], snaps={}, nu_l=float(nu_l), nu_g=float(nu_g))
    t0 = time.time()
    for i in range(n_chunks + 1):
        a = np.asarray(state['alpha_l'])
        if np.isnan(a).any():
            tr['nan_at'] = i * CHUNK; print(f"  mult={mult:g}: NaN at step {i * CHUNK}", flush=True); break
        tr['step'].append(i * CHUNK); tr['n_complex'].append(int(count_complex(state)))
        tr['a_min'].append(float(a.min())); tr['a_max'].append(float(a.max()))
        if i % SNAP_EVERY == 0 or i == n_chunks:
            tr['snaps'][i * CHUNK] = a.copy()
        if i == n_chunks:
            break
        state = run_chunk(state, nu_l, nu_g); jax.block_until_ready(state)
        if i % 10 == 0:
            print(f"  mult={mult:g} step {i * CHUNK:>6} a_max={tr['a_max'][-1]:.3f} "
                  f"n_cplx={tr['n_complex'][-1]:>4} elapsed={time.time() - t0:.0f}s", flush=True)
    tr['seconds'] = time.time() - t0
    cache[mult] = tr
    pickle.dump(cache, open(OUT, "wb"))
    print(f"mult={mult:g} done in {tr['seconds']:.0f}s  final a_max={tr['a_max'][-1]:.3f} "
          f"a_min={tr['a_min'][-1]:.4f} n_cplx={tr['n_complex'][-1]}", flush=True)
print("DONE", flush=True)
