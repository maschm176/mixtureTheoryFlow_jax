"""
Per-condition evaluation of ONE k_wall value (f_i_mult=1, i.e. interfacial
friction uncalibrated), applied identically to both phases (k_wl=k_wg=k_wall
-- the single shared wall-friction scalar, before the k_wl/k_wg split), across
all 36 Vuong conditions. Mirrors _vuong_fionly_worker.py exactly, swapping
which knob is held fixed. Fresh process per call (see _vuong_fit_joint.py
docstring for why). Writes wallonly_cache/k_<value>.pkl.
Usage: python3 _vuong_wallonly_worker.py <k_wall>
"""
import sys, pickle
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

k = float(sys.argv[1])
train, test = build_split()
rows = []
for split, conds in (('train', train), ('test', test)):
    for c in conds:
        r = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                    c['mu_l'], c['mu_g'], c['L'],
                                    f_i_mult=1.0, k_wl=k, k_wg=k)
        rows.append(dict(split=split, U_sl=c['U_sl'], U_sg=c['U_sg'],
                         psi_meas=c['psi'], psi_pred=r['alpha_l_pred'],
                         dpdz_meas=c['dpdz_measured'], dpdz_pred=abs(r['dPdz_pred_liquid']),
                         kh_total=r['kh_total'], any_nan=r['any_nan']))
pickle.dump(rows, open(f'wallonly_cache/k_{k:.4f}.pkl', 'wb'))
print(f"done k={k}")
