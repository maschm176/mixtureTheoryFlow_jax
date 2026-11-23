"""
Follow-up to the disc-mechanism investigation: tau_i (interfacial
friction) is structurally absent from physical_flux, so it cannot change
whether local_wave_speeds' disc goes negative at a given (alpha_l,
Delta) state -- confirmed by reading _implicit_friction_half_step
directly (operates on Delta alone, via the Strang-split source term,
never touches the flux Jacobian local_wave_speeds differentiates).

But friction CAN still matter dynamically: it's applied as its own
half-step before/after the explicit transport step every timestep, so a
strong enough f_i_mult might relax slip back down fast enough, per step,
to keep the trajectory from ever reaching the disc<0 threshold region in
the first place -- a question about RATES relative to dt, not about
which term wins in a shared formula.

This sweeps f_i_mult (the interfacial friction multiplier) on the same
representative fast-NaN Vuong ST condition (idx=797) used in the earlier
sigma magnitude scan, to test directly whether stronger interfacial
friction can keep the run stable -- same methodology as
_vuong_st_sigma_magnitude_scan.py, swapping sigma for f_i_mult.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
IDX = 797
F_I_MULTS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0, 10000.0]

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

row = st.loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]'])
rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=beta))
A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
cf_alpha_l = float(A_l / A)
print(f"idx={IDX}  U_sl={U_sl}  U_sg={U_sg}  cf_alpha_l(f_i_mult=1)={cf_alpha_l:.4f}", flush=True)
print(f"{'f_i_mult':>10} {'nan':>6} {'a_pred':>8} {'cf_alpha_l':>10} {'diff':>8}", flush=True)

for f_i_mult in F_I_MULTS:
    # NOTE: the Taitel-Dukler equilibrium holdup itself also depends on
    # f_i_mult, so recompute cf_alpha_l at EACH f_i_mult -- a fair
    # comparison needs "did the dynamic solver reach ITS OWN closure's
    # correct equilibrium," not the f_i_mult=1 equilibrium.
    ht_i = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                                     f_i_mult, 1.0, 1.0, beta=beta))
    A_i, A_l_i, A_g_i, _, _, _ = solver.td_geometry(ht_i, D)
    cf_alpha_l_i = float(A_l_i / A_i)

    r = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                        N=N, n_transit_times=N_TRANSIT, beta=beta,
                                        f_i_mult=f_i_mult)
    a_pred = "nan" if r['any_nan'] else f"{r['alpha_l_pred']:.4f}"
    diff = "nan" if r['any_nan'] else f"{abs(r['alpha_l_pred'] - cf_alpha_l_i):.4f}"
    print(f"{f_i_mult:>10.1f} {str(r['any_nan']):>6} {a_pred:>8} {cf_alpha_l_i:>10.4f} {diff:>8}", flush=True)

print("\nDONE", flush=True)
