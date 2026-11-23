"""
Step (a), second half: does the dedicated ST equilibrium fit
(f_i_mult=10.6923, k_wl=2.6477, k_wg=0.8510 -- see
_st_fit_joint_equilibrium.py) keep the DYNAMIC/transient solver stable
across all 23 Vuong ST conditions, and does the dynamic solver's
prediction actually land close to the measured HL (confirming the
equilibrium-path fit's accuracy carries over to the real, used-in-
production transient solver, not just the algebraic shortcut)?
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
F_I_MULT, K_WL, K_WG = 10.6923, 2.6477, 0.8510
ROWS_TO_CHECK = [797, 805, 820, 821, 822, 826, 832, 840,
                 1457, 1464, 1479, 1480, 1487, 1496, 1508, 1522, 1530,
                 823, 836, 1523, 1524, 1543, 1550]

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7} {'HL_meas':>8}  {'nan':>6} {'a_pred':>8} {'err%':>8}", flush=True)
errs = []
n_nan = 0
for idx in ROWS_TO_CHECK:
    row = st.loc[idx]
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
    HL_raw = row['HL[-]']
    HL_meas = None if (isinstance(HL_raw, str) and not HL_raw.replace('.', '', 1).isdigit()) else float(HL_raw)

    r = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                        N=N, n_transit_times=N_TRANSIT, beta=beta,
                                        f_i_mult=F_I_MULT, k_wl=K_WL, k_wg=K_WG)
    if r['any_nan']:
        n_nan += 1
    a_pred_str = "nan" if r['any_nan'] else f"{r['alpha_l_pred']:.4f}"
    HL_str = "   --  " if HL_meas is None else f"{HL_meas:.4f}"
    err_str = "nan"
    if not r['any_nan'] and HL_meas is not None:
        err_pct = 100.0 * (r['alpha_l_pred'] - HL_meas) / HL_meas
        errs.append(err_pct)
        err_str = f"{err_pct:+.1f}"

    print(f"{idx:>5} {U_sl:>8.5f} {U_sg:>7.3f} {HL_str:>8}  {str(r['any_nan']):>6} {a_pred_str:>8} {err_str:>8}", flush=True)

print(flush=True)
print(f"conditions with NaN: {n_nan} / {len(ROWS_TO_CHECK)}", flush=True)
if errs:
    mean_abs = sum(abs(e) for e in errs) / len(errs)
    mean_signed = sum(errs) / len(errs)
    print(f"n with valid measured HL + stable run: {len(errs)}", flush=True)
    print(f"mean |err%| vs measured HL: {mean_abs:.1f}%", flush=True)
    print(f"mean signed err% (bias): {mean_signed:+.1f}%", flush=True)
print("\nDONE", flush=True)
