"""
nu_t magnitude sweep on the BOUNDED (face-flux), IMPLICIT RS machinery --
the well-behaved variant from _reynolds_stress_implicit_test.py. Re-runs
the same question the original Stage 0 sensitivity sweep
(_rs_nu_t_sensitivity.py) asked, but this time with the numerical
pathologies (degenerate-diffusivity runaway, explicit amplification) fixed,
so the result isolates the PHYSICS question -- is the borrowed Thorpe
(1969) nu_t=1.3e-5 m^2/s simply the wrong order of magnitude for Fan's
high-shear gas-liquid conditions, as opposed to RS being structurally
unhelpful here regardless of magnitude.

Run at N=1800 (the only grid confirmed to show a real, non-negligible RS
effect at all -- N<=300 gave ~zero effect in earlier diagnostics) but with
a SHORTER n_transit_times=5 (vs. the "official" 15 used for the headline
comparisons) purely to keep per-run cost down for an exploratory sweep --
these numbers are for TREND detection across multipliers, not meant to be
read as directly comparable in absolute magnitude to the official N=1800/
15-transit-time table.
"""
import numpy as np
from _reynolds_stress_implicit_test import run_real_data_condition_rs_imex, NU_T_DEFAULT
import mixtureTheoryFlow_jax_solver_water_gas as solver

conditions = [
    ("Fan#1 (kh-unstable, U_sg=19.71)", dict(
        U_sl=0.014319654, U_sg=19.71, D=0.0508, L=19.0,
        rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5)),
    ("Fan#2 (kh-unstable, U_sg=19.88)", dict(
        U_sl=0.010300073, U_sg=19.88, D=0.0508, L=19.0,
        rho_l=1000.0, rho_g=2.66, mu_l=1.0e-3, mu_g=1.8e-5)),
]

multipliers = [0.01, 0.1, 1, 10, 100, 1_000, 10_000, 100_000]
N = 1800
N_TRANSIT = 5

for label, cond in conditions:
    base = solver.run_real_data_condition(**cond, N=N, n_transit_times=N_TRANSIT)
    print(f"=== {label}  (N={N}, n_transit_times={N_TRANSIT}) ===")
    print(f"  baseline (no RS)    : kh_total={base['kh_total']:>12,}  "
          f"alpha_range=[{base['alpha_min']:.5f},{base['alpha_max']:.5f}]  "
          f"alpha_l_pred={base['alpha_l_pred']:.5f}")
    for mult in multipliers:
        nu_t = NU_T_DEFAULT * mult
        r = run_real_data_condition_rs_imex(**cond, N=N, n_transit_times=N_TRANSIT,
                                             enable_rs=True, nu_t=nu_t)
        delta_pct = 100.0 * (r['kh_total'] - base['kh_total']) / base['kh_total']
        print(f"  nu_t={nu_t:10.3e} ({mult:>8,}x)  kh_total={r['kh_total']:>12,} ({delta_pct:+6.2f}%)  "
              f"any_nan={r['any_nan']!s:5}  alpha_l_pred={r['alpha_l_pred']:.5f}  "
              f"alpha_range=[{r['alpha_min']:.5f},{r['alpha_max']:.5f}]")
    print()

print("DONE")
