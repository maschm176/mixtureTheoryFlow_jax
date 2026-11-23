"""
MDPI validation, extending today's Fan finding two ways:
1. Does the same "run to settle, compare against solve_taitel_dukler_holdup"
   pattern hold for MDPI (air-water, atmospheric, very different Weber-
   number regime per the earlier Weber-number-check entry) -- not just Fan?
2. Does the settled PRESSURE GRADIENT (not just holdup) also behave
   sensibly? Two checks, both using predicted_pressure_gradient:
   (a) internal consistency: dPdz_liquid vs dPdz_gas from the dynamic
       scan's own settled state should agree (per that function's own
       docstring: a real disagreement means the profile hasn't actually
       settled, not that the formula split is wrong) -- a convergence
       check that needs no separate "ground truth" at all.
   (b) external consistency: a CLOSED-FORM equilibrium state (alpha_l_eq
       from solve_taitel_dukler_holdup, u_l_eq=J_L/alpha_l_eq,
       u_g_eq=J_G/alpha_g_eq via the uniform-flow relation already used
       inside taitel_dukler_residual, Delta_eq=rho_l*u_l_eq-rho_g*u_g_eq)
       fed into predicted_pressure_gradient, independent of the dynamic
       scan entirely, should match what the dynamic scan's own settled
       dPdz reads.

Constants (D, L, rho_l, rho_g, mu_l, mu_g) per the project's own verified
MDPI reset-globals convention (see "MDPI input verification" in
CLAUDE.md): D=0.060, L=24.0, rho_l=998.0, rho_g=1.204, mu_l=1.0e-3,
mu_g=1.81e-5. JL/JG columns in the CSV are the superficial velocities.
"""
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

df = pd.read_csv("mdpi_airwater_phase2g_dataset.csv")
rows_to_test = list(range(0, 32, 4))  # 8 conditions spread across the dataset

D, Lp = 0.060, 24.0
rho_l, rho_g = 998.0, 1.204
mu_l, mu_g = 1.0e-3, 1.81e-5
g = 9.81

N = 150
CHUNK_TRANSIT = 50.0
MAX_CHUNKS = 60  # up to 3000 transit times, generous safety margin

print(f"{'row':>4} {'J_L':>8} {'J_G':>8} {'settled@':>9} {'dyn_alpha_l':>12} {'cf_alpha_l':>12} {'hld_diff':>10} "
      f"{'dPdz_l':>9} {'dPdz_g':>9} {'dyn_lg_diff%':>12} {'cf_dPdz':>9} {'dyn_vs_cf%':>11}")

for idx in rows_to_test:
    row = df.iloc[idx]
    U_sl, U_sg = float(row['JL']), float(row['JG'])

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
    Delta_final = state['Delta']
    dyn_alpha_l = float(jnp.median(al_final))
    dyn_Delta = float(jnp.median(Delta_final))

    # closed-form holdup equilibrium
    ht = solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, g, 0.0)
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    cf_alpha_l = float(A_l / A)
    hld_diff = abs(dyn_alpha_l - cf_alpha_l)

    # dynamic scan's own settled dPdz (star units, outlet window)
    outlet_window = al_final[-11:-1]
    Delta_window = Delta_final[-11:-1]
    dPdz_l_star_arr, dPdz_g_star_arr = solver.predicted_pressure_gradient(
        outlet_window, Delta_window, 1.0, D_star, rho_l_star, rho_g_star,
        mu_l_star, mu_g_star, 1.0, 1.0, 1.0, g_star, 0.0
    )
    dPdz_l_star = float(jnp.median(dPdz_l_star_arr))
    dPdz_g_star = float(jnp.median(dPdz_g_star_arr))
    dyn_lg_diff_pct = 100.0 * abs(dPdz_l_star - dPdz_g_star) / (abs(dPdz_l_star) + 1e-12)

    # closed-form equilibrium dPdz, fully independent of the dynamic scan
    u_l_eq = U_sl_star / cf_alpha_l
    u_g_eq = U_sg_star / (1.0 - cf_alpha_l)
    Delta_eq_star = rho_l_star * u_l_eq - rho_g_star * u_g_eq
    cf_dPdz_l_star, cf_dPdz_g_star = solver.predicted_pressure_gradient(
        jnp.array([cf_alpha_l]), jnp.array([Delta_eq_star]), 1.0, D_star, rho_l_star, rho_g_star,
        mu_l_star, mu_g_star, 1.0, 1.0, 1.0, g_star, 0.0
    )
    cf_dPdz_l_star = float(cf_dPdz_l_star[0])
    dyn_vs_cf_pct = 100.0 * abs(dPdz_l_star - cf_dPdz_l_star) / (abs(cf_dPdz_l_star) + 1e-12)

    settled_str = f"{settled_at:.0f}" if settled_at else f"NOT by {MAX_CHUNKS*CHUNK_TRANSIT:.0f}"
    print(f"{idx:>4} {U_sl:>8.4f} {U_sg:>8.3f} {settled_str:>9} {dyn_alpha_l:>12.6f} {cf_alpha_l:>12.6f} "
          f"{hld_diff:>10.2e} {dPdz_l_star:>9.4f} {dPdz_g_star:>9.4f} {dyn_lg_diff_pct:>12.4f} "
          f"{cf_dPdz_l_star:>9.4f} {dyn_vs_cf_pct:>11.4f}")

print("DONE")
