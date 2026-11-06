"""
Per-condition evaluation of ONE (f_i_mult, k_wall) pair, k_wall applied as a
SINGLE shared scalar to both phases (k_wl=k_wg=k_wall) -- reconstructing the
original, pre-split combined-wall-friction scenario that was found to be
exactly degenerate with f_i_mult (see mixtureTheoryFlow_jax_solver_water_gas.py
's _wall_and_interfacial_shear docstring). Evaluated on a fixed 9-condition
subset (not all 36) to keep the 2D grid tractable -- picked deterministically
by sorting all 36 Vuong conditions (train+test combined) by U_sg and taking
every 4th, so the subset spans the full U_sg range rather than clustering.
This is a degeneracy-visualization scan, not an accuracy fit, so there's no
train/test split need here.
Writes combined_cache/f_<f>_k_<k>.pkl.
Usage: python3 _vuong_combined_worker.py <f_i_mult> <k_wall>
"""
import sys, pickle
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

f, k = float(sys.argv[1]), float(sys.argv[2])
train, test = build_split()
all_conds = sorted(train + test, key=lambda c: c['U_sg'])
subset = all_conds[::4]  # 9 of 36, spanning the full U_sg range

rows = []
for c in subset:
    r = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                c['mu_l'], c['mu_g'], c['L'],
                                f_i_mult=f, k_wl=k, k_wg=k)
    rows.append(dict(U_sl=c['U_sl'], U_sg=c['U_sg'],
                     psi_meas=c['psi'], psi_pred=r['alpha_l_pred'],
                     dpdz_meas=c['dpdz_measured'], dpdz_pred=abs(r['dPdz_pred_liquid']),
                     kh_total=r['kh_total'], any_nan=r['any_nan']))
pickle.dump(rows, open(f'combined_cache/f_{f:.4f}_k_{k:.4f}.pkl', 'wb'))
print(f"done f={f} k={k}")
