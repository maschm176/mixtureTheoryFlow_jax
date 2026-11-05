"""
One-shot, per-condition forward-solve evaluation at the epoch-5 parameters
(current best from the live joint fit: f_i_mult=1.3265, k_wl=0.7532,
k_wg=1.3173, loss=0.043427), across ALL 36 Vuong conditions (both train
and test), to see the actual fit quality condition-by-condition rather
than just the aggregate loss number.

Single-shot subprocess, not a long-lived process -- 36 calls is well
within the proven-clean ~230s/29-call scale (see _diag_percall_timing.py),
so this is safe to just run directly.
"""
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

F_I_MULT, K_WL, K_WG = 1.3265, 0.7532, 1.3173

train, test = build_split()
rows = [(c, 'train') for c in train] + [(c, 'test') for c in test]

print(f"{'split':>5} {'U_sl':>8} {'U_sg':>8} {'psi_meas':>9} {'psi_pred':>9} {'hld_err%':>9} "
      f"{'dpdz_meas':>10} {'dpdz_pred':>10} {'dpdz_err%':>10} {'kh_tot':>8}")

results = []
for c, split in rows:
    res = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                   c['mu_l'], c['mu_g'], c['L'],
                                   f_i_mult=F_I_MULT, k_wl=K_WL, k_wg=K_WG)
    psi_pred = res['alpha_l_pred']
    dpdz_pred = abs(res['dPdz_pred_liquid'])
    hld_err = (psi_pred - c['psi']) / c['psi'] * 100
    dpdz_err = (dpdz_pred - c['dpdz_measured']) / c['dpdz_measured'] * 100
    excluded = res['kh_total'] > 0
    results.append((split, c, psi_pred, dpdz_pred, hld_err, dpdz_err, excluded))
    flag = " EXCL" if excluded else ""
    print(f"{split:>5} {c['U_sl']:8.4f} {c['U_sg']:8.3f} {c['psi']:9.4f} {psi_pred:9.4f} "
          f"{hld_err:9.2f} {c['dpdz_measured']:10.2f} {dpdz_pred:10.2f} {dpdz_err:10.2f} "
          f"{res['kh_total']:8.1f}{flag}")

print()
for split in ('train', 'test'):
    rs = [r for r in results if r[0] == split and not r[6]]
    if not rs:
        continue
    hld_errs = [abs(r[4]) for r in rs]
    dpdz_errs = [abs(r[5]) for r in rs]
    print(f"{split}: n={len(rs)} (of {sum(1 for r in results if r[0]==split)})  "
          f"mean|hld_err%|={sum(hld_errs)/len(hld_errs):.2f}  "
          f"mean|dpdz_err%|={sum(dpdz_errs)/len(dpdz_errs):.2f}")
