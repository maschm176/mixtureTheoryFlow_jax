"""
Per-condition forward-solve evaluation at the NEW converged joint-fit
parameters (true-jax.grad fit through the equilibrium path,
_vuong_fit_joint_equilibrium.py, 3000 epochs, converged by epoch ~1475,
best epoch 2303): f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413, train
loss=0.033699. Supersedes the old finite-difference fit's
f_i_mult=1.3265/k_wl=0.7532/k_wg=1.3173.

Uses the DYNAMIC scan (run_real_data_condition), not the equilibrium
shortcut, specifically to re-verify kh_total (KH-instability stability)
at the new parameters across all 36 conditions -- stability was only
confirmed at the OLD parameters before, and friction/interfacial
multipliers changing this much could plausibly shift which conditions are
KH-stable.
"""
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

F_I_MULT, K_WL, K_WG = 3.8893, 1.5419, 0.8413

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
    results.append((split, c, psi_pred, res['dPdz_pred_liquid'], hld_err, dpdz_err, excluded, res['kh_total']))
    flag = " EXCL" if excluded else ""
    print(f"{split:>5} {c['U_sl']:8.4f} {c['U_sg']:8.3f} {c['psi']:9.4f} {psi_pred:9.4f} "
          f"{hld_err:9.2f} {c['dpdz_measured']:10.2f} {dpdz_pred:10.2f} {dpdz_err:10.2f} "
          f"{res['kh_total']:8.1f}{flag}", flush=True)

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

print()
print("# ROWS tuple format for plot_vuong_joint.py: "
      "(split, U_sl, U_sg, psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh_total)")
for split, c, psi_pred, dPdz_pred_liquid, hld_err, dpdz_err, excluded, kh in results:
    print(f'    ("{split}", {c["U_sl"]:.4f}, {c["U_sg"]:.3f}, {c["psi"]:.4f}, {psi_pred:.4f}, '
          f'{c["dpdz_measured"]:.2f}, {abs(dPdz_pred_liquid):.2f}, {int(kh)}),')
