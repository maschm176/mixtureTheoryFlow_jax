"""
Validation: confirm the nondimensionalized run_real_data_condition
(mixtureTheoryFlow_jax_solver_water_gas.py, current working tree)
reproduces the pre-refactor dimensional version's output to near machine
precision, on real Vuong (2016) conditions at the current joint-fit
parameters. Same discipline as the oil-water solver's own
nondimensionalization validation (see CLAUDE.md).

Loads the pre-refactor module (git commit 73cd904) from a separate file
path under a distinct module name, so both versions can be imported and
compared side by side in one process.
"""
import importlib.util
import sys

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

old = load_module('/tmp/nondim_validate/mixtureTheoryFlow_jax_solver_water_gas_OLD.py', 'solver_old')
new = load_module('/Users/mschmitz/Documents/MixtureTheoryCode/mixtureTheoryFlow_jax/mixtureTheoryFlow_jax_solver_water_gas.py', 'solver_new')

from _vuong_fit_joint import build_split

F_I_MULT, K_WL, K_WG = 1.3265, 0.7532, 1.3173

train, test = build_split()
conditions = (train + test)[::4]  # a representative subset, not all 36 -- cheap sanity check

print(f"{'U_sl':>8} {'U_sg':>8} {'alpha_old':>10} {'alpha_new':>10} {'reldiff%':>10} "
      f"{'dPdz_old':>10} {'dPdz_new':>10} {'reldiff%':>10} {'Re_SL':>9} {'Re_SG':>9}")

max_alpha_diff, max_dpdz_diff = 0.0, 0.0
for c in conditions:
    res_old = old.run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                           c['mu_l'], c['mu_g'], c['L'],
                                           f_i_mult=F_I_MULT, k_wl=K_WL, k_wg=K_WG)
    res_new = new.run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                           c['mu_l'], c['mu_g'], c['L'],
                                           f_i_mult=F_I_MULT, k_wl=K_WL, k_wg=K_WG)
    a_old, a_new = res_old['alpha_l_pred'], res_new['alpha_l_pred']
    d_old, d_new = res_old['dPdz_pred_liquid'], res_new['dPdz_pred_liquid']
    rel_a = (a_new - a_old) / a_old * 100
    rel_d = (d_new - d_old) / d_old * 100
    max_alpha_diff = max(max_alpha_diff, abs(rel_a))
    max_dpdz_diff = max(max_dpdz_diff, abs(rel_d))
    print(f"{c['U_sl']:8.4f} {c['U_sg']:8.3f} {a_old:10.6f} {a_new:10.6f} {rel_a:10.2e} "
          f"{d_old:10.3f} {d_new:10.3f} {rel_d:10.2e} {res_new['Re_SL']:9.3f} {res_new['Re_SG']:9.1f}",
          flush=True)

print()
print(f"max |alpha_l rel diff| = {max_alpha_diff:.3e}%")
print(f"max |dPdz_l rel diff|  = {max_dpdz_diff:.3e}%")
