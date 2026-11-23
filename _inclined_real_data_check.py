"""
Run the CURRENT solver (no surface tension, beta-plumbed) against REAL
inclined data, rather than the synthetic U_sl x beta sweep the module
docstring's KNOWN LIMITATION section is based on.

Two sources, both already sitting in the repo unused:
  1. Fan (2005): fan2005_sw_subset.csv has 60 rows total; 47 of them have
     a nonzero 'Inclination angle[°]' (-2,-1,1,2 deg) -- every prior Fan
     analysis in this project filtered down to the 13 horizontal (0 deg)
     rows only.
  2. Vuong (2016): the full TUFFP compilation
     (FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx) has 28 rows
     tagged Fluids=='Nitrogen - Isopar L', Flow Pattern=='ST', all at
     Inclination angle[°]=2 -- these are the "28 inclined ST conditions"
     CLAUDE.md already flagged as found-but-never-pulled-into-a-usable-
     subset. Pipe length isn't in that raw compilation (unlike the
     horizontal SW subset's own CSV); using L=85.0 m, same facility as
     vuong2016_sw_subset.csv's documented horizontal pipe length.

Sign convention: positive beta = uphill (per the module's KNOWN
LIMITATION section: "uphill raises [holdup] via the fixed-U_sg/shrinking-
gas-area feedback"). Assuming the source compilations use the standard
literature convention (positive angle = uphill) -- not independently
verified against either paper's text, flagged here rather than silently
assumed to be checked.

For each condition: run run_real_data_condition (uncalibrated defaults,
f_i_mult=k_wl=k_wg=1.0, same as every other zero-refit check in this
project) and report any_nan, kh_total, alpha_min/max, alpha_l_pred vs.
the closed-form solve_taitel_dukler_holdup equilibrium (sanity check that
a "clean" result is actually trustworthy, not just non-NaN).
"""
import math
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0

def run_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, angle_deg):
    beta = math.radians(angle_deg)
    try:
        res = solver.run_real_data_condition(
            U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N=N,
            n_transit_times=N_TRANSIT, beta=beta
        )
    except Exception as e:
        return {'error': str(e)}
    # closed-form equilibrium cross-check
    try:
        ht = solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                                1.0, 1.0, 1.0, 9.81, beta)
        A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
        cf_alpha_l = float(A_l / A)
    except Exception:
        cf_alpha_l = float('nan')
    res['cf_alpha_l'] = cf_alpha_l
    return res

print("=" * 100)
print("FAN (2005) -- 47 inclined rows (nonzero Inclination angle[deg])")
print("=" * 100)
fan = pd.read_csv("fan2005_sw_subset.csv")
fan_incl = fan[fan['Inclination angle[°]'] != 0]
print(f"{len(fan_incl)} rows\n")
print(f"{'idx':>5} {'angle':>6} {'U_sl':>9} {'U_sg':>8} {'nan':>5} {'kh_total':>10} "
      f"{'a_min':>9} {'a_max':>9} {'a_pred':>9} {'cf_a_l':>9}")

n_nan = n_kh = n_clean = 0
for idx, row in fan_incl.iterrows():
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D, Lp = float(row['Pipe Diameter[m]']), float(row['Pipe Length[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
    angle = float(row['Inclination angle[°]'])
    res = run_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, Lp, angle)
    if 'error' in res:
        print(f"{idx:>5} {angle:>6.1f} {U_sl:>9.5f} {U_sg:>8.3f}  ERROR: {res['error']}")
        continue
    if res['any_nan']:
        n_nan += 1
    elif res['kh_total'] > 0:
        n_kh += 1
    else:
        n_clean += 1
    print(f"{idx:>5} {angle:>6.1f} {U_sl:>9.5f} {U_sg:>8.3f} {str(res['any_nan']):>5} "
          f"{res['kh_total']:>10} {res['alpha_min']:>9.5f} {res['alpha_max']:>9.5f} "
          f"{res['alpha_l_pred']:>9.5f} {res['cf_alpha_l']:>9.5f}")

print(f"\nFan inclined summary: clean={n_clean}, kh_total>0={n_kh}, NaN={n_nan}, total={len(fan_incl)}")

print()
print("=" * 100)
print("VUONG (2016) -- 28 inclined 'ST' rows (Inclination angle=2 deg)")
print("=" * 100)
ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
vuong_st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
L_VUONG = 85.0
print(f"{len(vuong_st)} rows, using L={L_VUONG} m (same facility as vuong2016_sw_subset.csv)\n")
print(f"{'idx':>5} {'angle':>6} {'U_sl':>9} {'U_sg':>8} {'nan':>5} {'kh_total':>10} "
      f"{'a_min':>9} {'a_max':>9} {'a_pred':>9} {'cf_a_l':>9}")

n_nan = n_kh = n_clean = 0
for idx, row in vuong_st.iterrows():
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
    angle = float(row['Inclination angle[°]'])
    res = run_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG, angle)
    if 'error' in res:
        print(f"{idx:>5} {angle:>6.1f} {U_sl:>9.5f} {U_sg:>8.3f}  ERROR: {res['error']}")
        continue
    if res['any_nan']:
        n_nan += 1
    elif res['kh_total'] > 0:
        n_kh += 1
    else:
        n_clean += 1
    print(f"{idx:>5} {angle:>6.1f} {U_sl:>9.5f} {U_sg:>8.3f} {str(res['any_nan']):>5} "
          f"{res['kh_total']:>10} {res['alpha_min']:>9.5f} {res['alpha_max']:>9.5f} "
          f"{res['alpha_l_pred']:>9.5f} {res['cf_alpha_l']:>9.5f}")

print(f"\nVuong ST summary: clean={n_clean}, kh_total>0={n_kh}, NaN={n_nan}, total={len(vuong_st)}")
print("\nDONE")
