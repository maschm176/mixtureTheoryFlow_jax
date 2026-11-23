"""
Second follow-up: of the 15 Vuong ST rows that looked stable at mult=1.0
over a SHORT (n_transit=10) horizon, 9 turned out to NaN once given the
full n_transit=150 horizon at the SAME full dt -- a slow-developing
instability the short test missed entirely, independent of dt. Open
question: are these 9 the SAME dt-overshoot mechanism as the other 8
(just taking longer to grow large enough to interact badly with the
timestep), or a genuinely different, unboundedly-growing instability
that a smaller dt would only delay (not prevent)? This script re-tests
exactly those 9 rows at the SAME n_transit=150 horizon but mult=0.25
(already proven sufficient to rescue the "fast" 8 rows) to find out.
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
ROWS_TO_CHECK = [1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530]

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7} {'cf_a_l':>7} {'n_steps':>8} {'result':>20}", flush=True)

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
        result = "NaN"
    else:
        amin = float(jnp.min(final['alpha_l']))
        amax = float(jnp.max(final['alpha_l']))
        result = f"ok[{amin:.4f},{amax:.4f}]"

    print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {cf_alpha_l:>7.4f} {n_steps:>8} {result:>20}", flush=True)

print("\nDONE", flush=True)
