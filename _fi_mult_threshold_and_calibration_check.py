"""
Follow-up to _fi_mult_magnitude_scan.py: that scan found idx=797 NaNs at
f_i_mult=2.0 but is clean (and lands exactly on its own equilibrium,
diff=0.0000) at f_i_mult=5.0 -- a real threshold somewhere in between.
Critically, the project's actual calibrated value from the Vuong SW fit
(f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413 -- see CLAUDE.md
"Nondimensionalization + equilibrium-path true-gradient training") sits
inside that exact window, and was never tested against these high-holdup
ST conditions (it was fit on the 36 low/moderate-holdup SW conditions).

Part A: fine-grained f_i_mult scan (k_wl=k_wg=1 fixed, isolating f_i_mult
alone) on idx=797, bracketing 2-5, to pin down the threshold and check
3.8893 specifically.

Part B: same question with the FULL production calibration (f_i_mult=
3.8893, k_wl=1.5419, k_wg=0.8413 together, not f_i_mult alone) on idx=797
-- does the actual calibration survive this condition?

Part C: run the full production calibration against ALL 23 Vuong ST
conditions (same list as _vuong_st_surfacetension_check.py) to see how
many of the 17 previously-NaN (at f_i_mult=k_wl=k_wg=1) conditions it
rescues -- the real generalization question.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
IDX = 797
F_I_MULT_PROD, K_WL_PROD, K_WG_PROD = 3.8893, 1.5419, 0.8413

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

row = st.loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]'])
rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

print("=== Part A: fine f_i_mult scan on idx=797 (k_wl=k_wg=1) ===", flush=True)
print(f"{'f_i_mult':>10} {'nan':>6} {'a_pred':>8} {'cf_alpha_l':>10} {'diff':>8}", flush=True)
for f_i_mult in [2.0, 2.5, 3.0, 3.5, 3.8893, 4.0, 4.5, 5.0]:
    ht_i = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                                     f_i_mult, 1.0, 1.0, beta=beta))
    A_i, A_l_i, A_g_i, _, _, _ = solver.td_geometry(ht_i, D)
    cf_alpha_l_i = float(A_l_i / A_i)
    r = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                        N=N, n_transit_times=N_TRANSIT, beta=beta,
                                        f_i_mult=f_i_mult)
    a_pred = "nan" if r['any_nan'] else f"{r['alpha_l_pred']:.4f}"
    diff = "nan" if r['any_nan'] else f"{abs(r['alpha_l_pred'] - cf_alpha_l_i):.4f}"
    print(f"{f_i_mult:>10.4f} {str(r['any_nan']):>6} {a_pred:>8} {cf_alpha_l_i:>10.4f} {diff:>8}", flush=True)

print(flush=True)
print("=== Part B: FULL production calibration on idx=797 ===", flush=True)
ht_prod = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                                    F_I_MULT_PROD, K_WL_PROD, K_WG_PROD, beta=beta))
A_p, A_l_p, A_g_p, _, _, _ = solver.td_geometry(ht_prod, D)
cf_alpha_l_prod = float(A_l_p / A_p)
r_prod = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                         N=N, n_transit_times=N_TRANSIT, beta=beta,
                                         f_i_mult=F_I_MULT_PROD, k_wl=K_WL_PROD, k_wg=K_WG_PROD)
a_pred_prod = "nan" if r_prod['any_nan'] else f"{r_prod['alpha_l_pred']:.4f}"
print(f"  f_i_mult={F_I_MULT_PROD} k_wl={K_WL_PROD} k_wg={K_WG_PROD}  "
      f"cf_alpha_l={cf_alpha_l_prod:.4f}  any_nan={r_prod['any_nan']}  a_pred={a_pred_prod}", flush=True)

print(flush=True)
print("=== Part C: full production calibration vs all 23 Vuong ST conditions ===", flush=True)
ROWS_TO_CHECK = [797, 805, 820, 821, 822, 826, 832, 840,
                 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,
                 823, 836, 1523, 1524, 1543, 1550]

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7}  "
      f"{'nan(base)':>9} {'nan(prod)':>9} {'a_pred(prod)':>12} {'cf_a_l(prod)':>12} {'diff(prod)':>10}", flush=True)
n_rescued = n_still_nan = n_already_ok = 0
for idx in ROWS_TO_CHECK:
    row = st.loc[idx]
    U_sl_i, U_sg_i = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D_i = float(row['Pipe Diameter[m]'])
    rho_l_i, rho_g_i = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l_i, mu_g_i = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

    r0 = solver.run_real_data_condition(U_sl_i, U_sg_i, D_i, rho_l_i, rho_g_i, mu_l_i, mu_g_i, L_VUONG,
                                         N=N, n_transit_times=N_TRANSIT, beta=beta)
    ht_i = float(solver.solve_taitel_dukler_holdup(U_sl_i, U_sg_i, D_i, rho_l_i, rho_g_i, mu_l_i, mu_g_i,
                                                     F_I_MULT_PROD, K_WL_PROD, K_WG_PROD, beta=beta))
    A_i, A_l_i, A_g_i, _, _, _ = solver.td_geometry(ht_i, D_i)
    cf_alpha_l_i = float(A_l_i / A_i)
    r1 = solver.run_real_data_condition(U_sl_i, U_sg_i, D_i, rho_l_i, rho_g_i, mu_l_i, mu_g_i, L_VUONG,
                                         N=N, n_transit_times=N_TRANSIT, beta=beta,
                                         f_i_mult=F_I_MULT_PROD, k_wl=K_WL_PROD, k_wg=K_WG_PROD)
    a_pred1 = "nan" if r1['any_nan'] else f"{r1['alpha_l_pred']:.4f}"
    diff1 = "nan" if r1['any_nan'] else f"{abs(r1['alpha_l_pred'] - cf_alpha_l_i):.4f}"

    if r0['any_nan'] and not r1['any_nan']:
        n_rescued += 1
    elif r1['any_nan']:
        n_still_nan += 1
    else:
        n_already_ok += 1

    print(f"{idx:>5} {U_sl_i:>8.5f} {U_sg_i:>7.3f}  "
          f"{str(r0['any_nan']):>9} {str(r1['any_nan']):>9} {a_pred1:>12} {cf_alpha_l_i:>12.4f} {diff1:>10}", flush=True)

print(flush=True)
print(f"rescued by production calibration (nan->stable): {n_rescued}", flush=True)
print(f"still nan at production calibration:              {n_still_nan}", flush=True)
print(f"already stable at baseline (f_i_mult=k_wl=k_wg=1): {n_already_ok}", flush=True)
print("\nDONE", flush=True)
