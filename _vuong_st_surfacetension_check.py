"""
The real payoff check for the new joint-implicit surface-tension term:
do the 23 genuinely-high-holdup Vuong ST conditions (see
_vuong_st_dt_threshold_check.py / _vuong_st_dt_longhorizon_check.py for
how this list was built: 8 fast-NaN + 9 slow-NaN + 6 already-robust, all
at the FULL, un-reduced CFL dt) now stay NaN-free at sigma=0.024 N/m
(real Isopar L - nitrogen surface tension), WITHOUT the earlier 0.25x
dt-reduction workaround?

Runs each of the 23 conditions twice -- sigma=0.0 (expected to reproduce
the already-known NaN pattern, a sanity check that this script's setup
matches the earlier investigation) and sigma=0.024 (the real test) --
at n_transit_times=150 (long enough to catch the SLOW-developing NaNs
the short-horizon test originally missed, per
_vuong_st_dt_longhorizon_check.py's finding) and the full, un-reduced
CFL dt (run_real_data_condition's own cfl=0.25 safety factor, no extra
MULT reduction on top of it).
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
SIGMA_REAL = 0.024
ROWS_TO_CHECK = [797, 805, 820, 821, 822, 826, 832, 840,                # fast-NaN group
                 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,  # slow-NaN group
                 823, 836, 1523, 1524, 1543, 1550]                      # already-robust group

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7} {'cf_a_l':>7}  "
      f"{'nan(s=0)':>8} {'a_pred(s=0)':>11}  "
      f"{'nan(s=.024)':>11} {'a_pred(s=.024)':>14} {'diff(s=.024)':>12}", flush=True)

n_rescued = 0
n_still_nan = 0
n_already_ok = 0

for idx in ROWS_TO_CHECK:
    row = st.loc[idx]
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

    ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=beta))
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    cf_alpha_l = float(A_l / A)

    r0 = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                         N=N, n_transit_times=N_TRANSIT, beta=beta, sigma=0.0)
    r1 = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                         N=N, n_transit_times=N_TRANSIT, beta=beta, sigma=SIGMA_REAL)

    a_pred0 = "nan" if r0['any_nan'] else f"{r0['alpha_l_pred']:.4f}"
    a_pred1 = "nan" if r1['any_nan'] else f"{r1['alpha_l_pred']:.4f}"
    diff1 = "nan" if r1['any_nan'] else f"{abs(r1['alpha_l_pred'] - cf_alpha_l):.4f}"

    if r0['any_nan'] and not r1['any_nan']:
        n_rescued += 1
    elif r1['any_nan']:
        n_still_nan += 1
    else:
        n_already_ok += 1

    print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {cf_alpha_l:>7.4f}  "
          f"{str(r0['any_nan']):>8} {a_pred0:>11}  "
          f"{str(r1['any_nan']):>11} {a_pred1:>14} {diff1:>12}", flush=True)

print(flush=True)
print(f"rescued by sigma (nan->stable): {n_rescued}", flush=True)
print(f"still nan at sigma=0.024:       {n_still_nan}", flush=True)
print(f"already stable at sigma=0:      {n_already_ok}", flush=True)
print("\nDONE", flush=True)
