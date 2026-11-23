"""
Convergence study: does alpha_l_pred settle down as N increases WITH
surface tension active, in contrast to the baseline's confirmed
non-convergence (26x swing between N=150 and N=1800 for Fan#1)?

This is the direct empirical test of whether surface tension genuinely
restores well-posedness (per the Fullmer framing: a real cutoff wavelength
lambda0 should make the model converge under refinement, unlike the
ill-posed baseline, where refinement only exposes shorter, worse
unstable wavelengths with no limit). N=1800 was only ever confirmed to
be "just barely sufficient" to reach lambda0 -- this sweep checks whether
alpha_l_pred is actually settling as N grows past that point, or still
drifting the way baseline does.

Runs both baseline (enable_st=False) and ST-active (enable_st=True) at
each N, for direct side-by-side comparison. n_transit_times=10 (vs. the
"official" 15) to keep the sweep's total cost down -- these are for
convergence TREND detection, not meant to exactly reproduce the official
headline numbers.
"""
import time
from _dispersive_explicit_test import run_real_data_condition_reg

cond = dict(U_sl=0.014319654, U_sg=19.71, D=0.0508, L=19.0,
            rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5, sigma=0.072)

N_values = [150, 300, 600, 1200, 1800]
N_TRANSIT = 10

print(f"=== Fan#1 convergence study (n_transit_times={N_TRANSIT}) ===")
print(f"{'N':>6} {'config':>10} {'alpha_l_pred':>14} {'kh_total':>14} {'alpha_min':>12} {'alpha_max':>10} {'time(s)':>8}")
for N in N_values:
    for label, enable_st in [("baseline", False), ("ST", True)]:
        t0 = time.time()
        r = run_real_data_condition_reg(**cond, N=N, n_transit_times=N_TRANSIT,
                                         enable_rs=False, enable_st=enable_st)
        dt = time.time() - t0
        print(f"{N:>6} {label:>10} {r['alpha_l_pred']:>14.6f} {r['kh_total']:>14,} "
              f"{r['alpha_min']:>12.6f} {r['alpha_max']:>10.5f} {dt:>8.1f}")

print()
print("DONE")
