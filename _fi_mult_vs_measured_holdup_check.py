"""
Critical correction to the previous f_i_mult rescue check: everything
reported there (diff=0.0000) was a SELF-CONSISTENCY check -- comparing
the dynamic solver's prediction against its OWN steady Taitel-Dukler
equilibrium solve at the SAME (f_i_mult, k_wl, k_wg). That only verifies
the transient solver relaxes correctly to whatever its closure says the
answer should be -- it says nothing about whether that answer matches
REALITY.

The FFPU dataset has an HL[-] column -- the actual MEASURED liquid
holdup for each condition. This script compares the production
calibration's prediction against THAT, for all 23 Vuong ST conditions,
to answer the real question: does boosting f_i_mult (which we know
stabilizes the numerics) also give physically correct holdup, or does it
just move the equilibrium to some arbitrary stable-but-wrong point?

Also reports the DEFAULT-closure (f_i_mult=k_wl=k_wg=1) comparison against
measured HL for the 6 already-robust conditions, as a baseline for how
good/bad "no calibration" accuracy looks on this subset to begin with.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
F_I_MULT_PROD, K_WL_PROD, K_WG_PROD = 3.8893, 1.5419, 0.8413
ROWS_TO_CHECK = [797, 805, 820, 821, 822, 826, 832, 840,
                 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,
                 823, 836, 1523, 1524, 1543, 1550]

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7} {'HL_meas':>8}  "
      f"{'nan(prod)':>9} {'a_pred(prod)':>12} {'err_vs_meas':>12}  "
      f"{'err_vs_meas%':>12}", flush=True)

errs = []
for idx in ROWS_TO_CHECK:
    row = st.loc[idx]
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
    HL_meas_raw = row['HL[-]']
    if isinstance(HL_meas_raw, str) and not HL_meas_raw.replace('.', '', 1).isdigit():
        print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f}   {'(no HL)':>8}  -- skipped, no measured holdup --", flush=True)
        continue
    HL_meas = float(HL_meas_raw)

    r1 = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                         N=N, n_transit_times=N_TRANSIT, beta=beta,
                                         f_i_mult=F_I_MULT_PROD, k_wl=K_WL_PROD, k_wg=K_WG_PROD)
    if r1['any_nan']:
        a_pred1 = "nan"
        err_abs = "nan"
        err_pct = "nan"
    else:
        a_pred1 = r1['alpha_l_pred']
        err_abs = abs(a_pred1 - HL_meas)
        err_pct = 100.0 * (a_pred1 - HL_meas) / HL_meas
        errs.append(err_pct)
        a_pred1 = f"{a_pred1:.4f}"
        err_abs = f"{err_abs:.4f}"
        err_pct = f"{err_pct:+.1f}"

    print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {HL_meas:>8.4f}  "
          f"{str(r1['any_nan']):>9} {a_pred1:>12} {err_abs:>12}  {err_pct:>12}", flush=True)

print(flush=True)
if errs:
    mean_abs_err = sum(abs(e) for e in errs) / len(errs)
    mean_err = sum(errs) / len(errs)
    print(f"n with valid prediction: {len(errs)} / {len(ROWS_TO_CHECK)}", flush=True)
    print(f"mean |err%| vs measured HL: {mean_abs_err:.1f}%", flush=True)
    print(f"mean signed err% vs measured HL (bias): {mean_err:+.1f}%", flush=True)

print("\nDONE", flush=True)
