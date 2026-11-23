"""
Does the dt-fixed (dt=0.25x CFL) Vuong ST solution actually reach its
true equilibrium given enough time, or does it plateau somewhere wrong?
At n_transit=150 (_vuong_st_dt_fix_accuracy_check.py), all 23 conditions
stayed NaN-free but alpha_l_pred (outlet) was still far below cf_alpha_l
in every case -- the same "hasn't propagated to the outlet yet" pattern
found earlier today for very-low-U_sl Fan rows, which needed up to 2000
transit times (not the usual 100-150) to fully settle. This script tests
one representative row from each of the three dt-threshold groups (fast
NaN, slow NaN, already-robust) at n_transit=2000, chunked (200k steps/
chunk) with a NaN check after every chunk so a mid-run blowup is caught
rather than silently running to NaN-at-the-end.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 2000.0
MULT = 0.25
CHUNK = 200_000
ROWS_TO_CHECK = [797, 1480, 1550]  # fast-NaN, slow-NaN, already-robust

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7} {'cf_a_l':>7} {'nan':>5} {'a_min':>8} {'a_max':>8} {'a_pred':>8} {'diff':>8}", flush=True)

for idx in ROWS_TO_CHECK:
    row = st.loc[idx]
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

    ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=beta))
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    cf_alpha_l = float(A_l / A)

    Q = U_sl + U_sg
    nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81)
    U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
    rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
    mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
    g_star, D_star = nd['g_star'], nd['D_star']
    L_star = L_VUONG / D
    dx_star = L_star / N
    dt_cfl = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                        mu_l_star, mu_g_star, dx_star, 1.0, 1.0, 1.0, 0.25,
                                        g=g_star, beta=beta)
    dt_star = dt_cfl * MULT
    alpha_l_inlet = U_sl_star
    Delta_inlet = rho_l_star - rho_g_star
    n_steps_total = int(np.ceil(N_TRANSIT * L_star / dt_star))
    n_chunks = int(np.ceil(n_steps_total / CHUNK))
    print(f"  [idx={idx}] n_steps_total={n_steps_total}  n_chunks={n_chunks}", flush=True)

    state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

    @jax.jit
    def run_chunk(state, dt_star=dt_star):
        def body(s, _):
            ns, nkh = solver.advance_state(s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
                                            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet,
                                            g_star, 1.0, 1.0, 1.0, 1.3, beta)
            return ns, nkh
        final, nkh = jax.lax.scan(body, state, None, length=CHUNK)
        return final, jnp.sum(nkh)

    any_nan = False
    for c in range(n_chunks):
        state, nkh = run_chunk(state)
        any_nan = bool(jnp.any(jnp.isnan(state['alpha_l'])) or jnp.any(jnp.isnan(state['Delta'])))
        if c % 5 == 0 or any_nan:
            print(f"    chunk {c+1}/{n_chunks}  any_nan={any_nan}", flush=True)
        if any_nan:
            break

    if any_nan:
        print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {cf_alpha_l:>7.4f}  True     nan      nan      nan      nan", flush=True)
        continue

    amin = float(jnp.min(state['alpha_l']))
    amax = float(jnp.max(state['alpha_l']))
    outlet_window = state['alpha_l'][-11:-1]
    alpha_l_pred = float(jnp.median(outlet_window))
    diff = abs(alpha_l_pred - cf_alpha_l)
    print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {cf_alpha_l:>7.4f} {str(any_nan):>5} "
          f"{amin:>8.4f} {amax:>8.4f} {alpha_l_pred:>8.4f} {diff:>8.4f}", flush=True)

print("\nDONE", flush=True)
