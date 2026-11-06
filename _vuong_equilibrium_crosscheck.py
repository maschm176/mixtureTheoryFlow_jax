"""
Cross-check: does the dynamic Strang-split scan (run_real_data_condition)
actually agree with the closed-form Taitel-Dukler equilibrium solver
(solve_taitel_dukler_holdup) across the real Vuong (2016) conditions, at
the current joint-fit parameters (f_i_mult=1.3265, k_wl=0.7532,
k_wg=1.3173)?

If so, this confirms the equilibrium path -- which has an EXACT
custom_jvp gradient and needs no dynamic scan at all -- is a legitimate,
much cheaper stand-in for calibration purposes (see the CLAUDE.md
discussion this script supports).

Single-shot, 36 calls -- same safe scale as _vuong_eval_epoch5_detail.py.
"""
import jax.numpy as jnp
from mixtureTheoryFlow_jax_solver_water_gas import (
    run_real_data_condition, solve_taitel_dukler_holdup, td_geometry,
    predicted_pressure_gradient,
)
from _vuong_fit_joint import build_split

F_I_MULT, K_WL, K_WG = 1.3265, 0.7532, 1.3173

train, test = build_split()
rows = [(c, 'train') for c in train] + [(c, 'test') for c in test]

print(f"{'split':>5} {'U_sl':>8} {'U_sg':>8} {'alpha_dyn':>10} {'alpha_eq':>10} {'reldiff%':>9} "
      f"{'dPdz_dyn':>10} {'dPdz_eq':>10} {'reldiff%':>9} {'kh_tot':>7}")

diffs_alpha, diffs_dpdz = [], []
for c, split in rows:
    # --- dynamic scan (the expensive, currently-used path) ---
    res = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                   c['mu_l'], c['mu_g'], c['L'],
                                   f_i_mult=F_I_MULT, k_wl=K_WL, k_wg=K_WG)
    alpha_dyn = res['alpha_l_pred']
    dPdz_dyn = res['dPdz_pred_liquid']
    kh = res['kh_total']

    # --- equilibrium (closed-form) path -- no scan at all ---
    Q = c['U_sl'] + c['U_sg']
    ht_eq = solve_taitel_dukler_holdup(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                        c['mu_l'], c['mu_g'], F_I_MULT, K_WL, K_WG)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht_eq, c['D'])
    alpha_l_eq = A_l / A
    alpha_g_eq = A_g / A
    u_l_eq = c['U_sl'] / jnp.maximum(alpha_l_eq, 1e-9)
    u_g_eq = c['U_sg'] / jnp.maximum(alpha_g_eq, 1e-9)
    Delta_eq = c['rho_l'] * u_l_eq - c['rho_g'] * u_g_eq

    dPdz_l_eq, dPdz_g_eq = predicted_pressure_gradient(
        alpha_l_eq, Delta_eq, Q, c['D'], c['rho_l'], c['rho_g'], c['mu_l'], c['mu_g'],
        F_I_MULT, K_WL, K_WG,
    )
    alpha_eq = float(alpha_l_eq)
    dPdz_eq = float(dPdz_l_eq)

    rel_alpha = (alpha_dyn - alpha_eq) / alpha_eq * 100
    rel_dpdz = (dPdz_dyn - dPdz_eq) / dPdz_eq * 100
    diffs_alpha.append(abs(rel_alpha))
    diffs_dpdz.append(abs(rel_dpdz))

    flag = " EXCL" if kh > 0 else ""
    print(f"{split:>5} {c['U_sl']:8.4f} {c['U_sg']:8.3f} {alpha_dyn:10.4f} {alpha_eq:10.4f} "
          f"{rel_alpha:9.3f} {dPdz_dyn:10.2f} {dPdz_eq:10.2f} {rel_dpdz:9.3f} {kh:7.0f}{flag}",
          flush=True)

print()
print(f"alpha_l: mean|rel diff|={sum(diffs_alpha)/len(diffs_alpha):.4f}%  max={max(diffs_alpha):.4f}%")
print(f"dPdz_l:  mean|rel diff|={sum(diffs_dpdz)/len(diffs_dpdz):.4f}%  max={max(diffs_dpdz):.4f}%")
