"""
Per-condition evaluation of ONE f_i_mult value (k_wl=k_wg=1, i.e. wall friction
uncalibrated) across all 36 Vuong conditions. Fresh process per call (see
_vuong_fit_joint.py docstring for why). Writes fionly_cache/f_<value>.pkl.
Usage: python3 _vuong_fionly_worker.py <f_i_mult>
"""
import sys, pickle
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

f = float(sys.argv[1])
train, test = build_split()
rows = []
for split, conds in (('train', train), ('test', test)):
    for c in conds:
        r = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                    c['mu_l'], c['mu_g'], c['L'],
                                    f_i_mult=f, k_wl=1.0, k_wg=1.0)
        rows.append(dict(split=split, U_sl=c['U_sl'], U_sg=c['U_sg'],
                         psi_meas=c['psi'], psi_pred=r['alpha_l_pred'],
                         dpdz_meas=c['dpdz_measured'], dpdz_pred=abs(r['dPdz_pred_liquid']),
                         kh_total=r['kh_total'], any_nan=r['any_nan']))
pickle.dump(rows, open(f'fionly_cache/f_{f:.4f}.pkl', 'wb'))
print(f"done f={f}")
