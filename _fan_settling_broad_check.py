"""
Broad test of the "run to settle, compare against closed-form equilibrium"
hypothesis across a spread of Fan (2005) conditions -- not just the two
known kh-unstable ones (Fan#1, Fan#2), which have already been confirmed
to settle to a grid-convergent value matching solve_taitel_dukler_holdup
to ~4e-7. This checks whether that pattern (dynamic scan settles to the
SAME value the independent, grid-free closed-form solver predicts,
regardless of whether kh_total ends up 0 or saturated) holds more broadly
across the dataset.

N=150 is used throughout -- already confirmed to reproduce the exact same
settled equilibrium as N=1800 for Fan#1/#2, so there's no need to pay for
large N just to get a trustworthy equilibrium reading. This makes the
whole sweep cheap (Fan#1 to 100 transit times at N=150 took 11s total).

"Settled" is declared once kh_this_chunk is IDENTICAL for 2 consecutive
chunks (a strong signal the state has stopped changing at all, not just
slowed down) -- same criterion directly validated against Fan#1/#2's
already-known settling behavior.
"""
import time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

df = pd.read_csv("fan2005_sw_subset.csv")
rows_to_test = [0, 7, 16, 19, 25, 30, 35, 40, 46, 56]

N = 150
CHUNK_TRANSIT = 5.0
MAX_CHUNKS = 20  # up to 100 transit times
g = 9.81

print(f"{'row':>4} {'U_sl':>10} {'U_sg':>7} {'settled@':>9} {'dynamic_alpha_l':>16} {'closed_form_alpha_l':>20} {'diff':>10} {'final_kh_rate':>14}")

for idx in rows_to_test:
    row = df.iloc[idx]
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D, Lp = float(row['Pipe Diameter[m]']), float(row['Pipe Length[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

    Q = U_sl + U_sg
    nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, g)
    U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
    rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
    mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
    g_star, D_star = nd['g_star'], nd['D_star']
    L_star = Lp / D
    dx_star = L_star / N
    dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                         mu_l_star, mu_g_star, dx_star, 1.0, 1.0, 1.0, 0.25,
                                         g=g_star, beta=0.0)
    alpha_l_inlet = U_sl_star
    Delta_inlet = rho_l_star - rho_g_star
    state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)
    steps_per_chunk = int(np.ceil(CHUNK_TRANSIT * L_star / dt_star))

    @jax.jit
    def run_chunk(state):
        def body(s, _):
            ns, nkh = solver.advance_state(s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
                                            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet,
                                            g_star, 1.0, 1.0, 1.0, 1.3, 0.0)
            return ns, nkh
        final, nkh = jax.lax.scan(body, state, None, length=steps_per_chunk)
        return final, jnp.sum(nkh)

    prev_kh = None
    settled_at = None
    for c in range(MAX_CHUNKS):
        state, nkh_chunk = run_chunk(state)
        nkh_chunk = int(nkh_chunk)
        if prev_kh is not None and nkh_chunk == prev_kh:
            settled_at = (c + 1) * CHUNK_TRANSIT
            break
        prev_kh = nkh_chunk

    al_final = state['alpha_l']
    dynamic_alpha_l = float(jnp.median(al_final))
    uniform = float(jnp.std(al_final)) < 1e-5

    ht = solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, g, 0.0)
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    closed_form_alpha_l = float(A_l / A)

    diff = abs(dynamic_alpha_l - closed_form_alpha_l)
    settled_str = f"{settled_at:.0f}" if settled_at else "NOT by 100"
    print(f"{idx:>4} {U_sl:>10.6f} {U_sg:>7.2f} {settled_str:>9} {dynamic_alpha_l:>16.6f} "
          f"{closed_form_alpha_l:>20.6f} {diff:>10.2e} {'uniform' if uniform else 'NOT uniform':>14}")

print("DONE")
