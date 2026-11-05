"""
Diagnostic: run all 29 training conditions ONCE at (f_i_mult=1, k_wl=1, k_wg=1),
logging wall time AND process RSS after EVERY SINGLE condition, not just after
a full epoch. Goal: see in real time whether per-call cost climbs as the
process accumulates more distinct conditions (compile-cache growth theory),
or whether a specific condition is anomalously slow (straggler theory), rather
than waiting 30-80+ minutes to find out from an aggregate epoch time alone.
"""
import time
import resource
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

train, test = build_split()

print(f"{'i':>3} {'U_sl':>8} {'U_sg':>8} {'n_steps':>8} {'call_s':>8} {'cum_s':>9} {'RSS_MB':>8}")
t_start = time.time()
for i, c in enumerate(train):
    t0 = time.time()
    r = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                 c['mu_l'], c['mu_g'], c['L'], f_i_mult=1.0, k_wl=1.0, k_wg=1.0)
    call_s = time.time() - t0
    cum_s = time.time() - t_start
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    print(f"{i:3d} {c['U_sl']:8.4f} {c['U_sg']:8.3f} {r['n_steps']:8d} {call_s:8.1f} {cum_s:9.1f} {rss_mb:8.1f}")
