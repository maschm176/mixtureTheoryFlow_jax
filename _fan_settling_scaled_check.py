"""
Rerun of the 5 Fan conditions that hadn't moved from their inlet value by
100 (nominal, L/Q-based) transit times in _fan_settling_broad_check.py.
Hypothesis: these are low-U_sl conditions where the LIQUID phase's actual
residence time (set roughly by its own velocity, not the mixture velocity
Q) is much longer than the nominal L/Q transit time, so they simply need
much more simulated time before anything happens at all -- not evidence
against the "settles to the closed-form equilibrium" hypothesis.

Horizon raised to 2000 nominal transit times (vs. the previous 100) --
still cheap at N=150 (the earlier 100-transit Fan#1 run took 11s total).
"Settled" now requires 3 consecutive matching chunks, not 2, to avoid the
earlier false-positive ("hasn't started yet" trivially matches for 2
consecutive zero-kh chunks).
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

df = pd.read_csv("fan2005_sw_subset.csv")
rows_to_test = [25, 30, 40, 46, 56]

N = 150
CHUNK_TRANSIT = 50.0
MAX_CHUNKS = 40  # up to 2000 transit times
g = 9.81

print(f"{'row':>4} {'U_sl':>10} {'U_sg':>7} {'settled@':>9} {'dynamic_alpha_l':>16} {'closed_form_alpha_l':>20} {'diff':>10}")

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

    kh_history = []
    settled_at = None
    for c in range(MAX_CHUNKS):
        state, nkh_chunk = run_chunk(state)
        nkh_chunk = int(nkh_chunk)
        kh_history.append(nkh_chunk)
        if len(kh_history) >= 3 and kh_history[-1] == kh_history[-2] == kh_history[-3]:
            settled_at = (c + 1) * CHUNK_TRANSIT
            break

    al_final = state['alpha_l']
    dynamic_alpha_l = float(jnp.median(al_final))

    ht = solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, g, 0.0)
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    closed_form_alpha_l = float(A_l / A)

    diff = abs(dynamic_alpha_l - closed_form_alpha_l)
    settled_str = f"{settled_at:.0f}" if settled_at else f"NOT by {MAX_CHUNKS*CHUNK_TRANSIT:.0f}"
    print(f"{idx:>4} {U_sl:>10.6f} {U_sg:>7.2f} {settled_str:>9} {dynamic_alpha_l:>16.6f} "
          f"{closed_form_alpha_l:>20.6f} {diff:>10.2e}")

print("DONE")
