"""
Accuracy check on top of the dt-threshold finding: all 23
genuinely-high-holdup Vuong ST rows stop NaN-ing once dt drops to 0.25x
the CFL estimate (see _vuong_st_dt_threshold_check.py /
_vuong_st_dt_longhorizon_check.py). But NaN-free is not the same as
PHYSICALLY CORRECT -- a couple of the global alpha_max values seen
(0.91, 0.97) are suspiciously close to a fully-liquid-blocked state, not
the expected equilibrium (~0.4-0.5). This script checks whether the
OUTLET-WINDOW holdup (the actual diagnostic run_real_data_condition
reports, median of the last few interior cells) matches the closed-form
Taitel-Dukler equilibrium -- the same validation already done for the
low-holdup Fan/MDPI settling work earlier today -- for all 23 conditions
at dt=0.25x CFL, n_transit=150.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
MULT = 0.25
ROWS_TO_CHECK = [797, 805, 820, 821, 822, 826, 832, 840,            # fast-NaN group
                 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,  # slow-NaN group
                 823, 836, 1523, 1524, 1543, 1550]                  # already-robust group

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
    n_steps = int(np.ceil(N_TRANSIT * L_star / dt_star))

    state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

    @jax.jit
    def run(state, n_steps=n_steps, dt_star=dt_star):
        def body(s, _):
            ns, nkh = solver.advance_state(s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
                                            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet,
                                            g_star, 1.0, 1.0, 1.0, 1.3, beta)
            return ns, nkh
        final, nkh = jax.lax.scan(body, state, None, length=n_steps)
        return final, jnp.sum(nkh)

    final, nkh = run(state)
    any_nan = bool(jnp.any(jnp.isnan(final['alpha_l'])) or jnp.any(jnp.isnan(final['Delta'])))
    if any_nan:
        print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {cf_alpha_l:>7.4f}  True     nan      nan      nan      nan", flush=True)
        continue

    amin = float(jnp.min(final['alpha_l']))
    amax = float(jnp.max(final['alpha_l']))
    outlet_window = final['alpha_l'][-11:-1]
    alpha_l_pred = float(jnp.median(outlet_window))
    diff = abs(alpha_l_pred - cf_alpha_l)
    print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {cf_alpha_l:>7.4f} {str(any_nan):>5} "
          f"{amin:>8.4f} {amax:>8.4f} {alpha_l_pred:>8.4f} {diff:>8.4f}", flush=True)

print("\nDONE", flush=True)
