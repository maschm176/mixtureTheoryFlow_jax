"""
nu_t sensitivity sweep for the Stage 0 Reynolds-stress-only diagnostic
(_reynolds_stress_explicit_test.py). Checks whether the earlier
negligible-effect result was a magnitude problem (ν_t too small, borrowed
from a much gentler flume experiment) or a structural one (Reynolds stress
alone never crosses to true stability in this coupled system, regardless
of magnitude -- the linear dispersion analysis's prediction).
"""
import numpy as np
from _reynolds_stress_explicit_test import run_real_data_condition_rs, NU_T_DEFAULT

conditions = [
    ("Fan#1 (kh-unstable, U_sg=19.71)", dict(
        U_sl=0.014319654, U_sg=19.71, D=0.0508, L=19.0,
        rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5)),
    ("Fan#2 (kh-unstable, U_sg=19.88)", dict(
        U_sl=0.010300073, U_sg=19.88, D=0.0508, L=19.0,
        rho_l=1000.0, rho_g=2.66, mu_l=1.0e-3, mu_g=1.8e-5)),
    ("Synthetic Vuong (alpha_l0~0.55)", dict(
        U_sl=0.65, U_sg=2.8, D=0.1557, L=85.0,
        rho_l=760.0, rho_g=20.0, mu_l=1.35e-3, mu_g=1.8e-5)),
]

multipliers = [1, 10, 100, 1_000, 10_000, 100_000]

for label, cond in conditions:
    print(f"=== {label} ===")
    for mult in multipliers:
        nu_t = NU_T_DEFAULT * mult
        r = run_real_data_condition_rs(**cond, enable_rs=True, nu_t=nu_t)
        print(f"  nu_t={nu_t:10.3e} ({mult:>7,}x)  kh_total={r['kh_total']:>14,}  "
              f"any_nan={r['any_nan']!s:5}  alpha_l_pred={r['alpha_l_pred']:.5f}  "
              f"alpha_range=[{r['alpha_min']:.5f},{r['alpha_max']:.5f}]")
    print()
