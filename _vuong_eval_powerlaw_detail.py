"""
Per-condition forward-solve evaluation (dynamic scan, re-checking
kh_total) at the power-law f_i_mult fit's best-epoch parameters
(_vuong_fit_fi_powerlaw.py, 3000 epochs, converged by ~epoch 150, best
epoch 2725): f0=5.5650, alpha_fi=+0.6083, k_wl=2.1475, k_wg=0.5659 --
f_i_mult = f0 * exp(alpha_fi*(log(Re_SG) - MEAN_LOG_RE_SG)), MEAN_LOG_RE_SG
computed from the 29 training conditions' own Re_SG. k_wl/k_wg stay flat
scalars. Train loss=0.027961, test loss=0.026653 -- better than the flat
3-scalar fit (train 0.033699, test 0.042799) on both splits. See CLAUDE.md
"Power-law f_i_mult(Re_SG) extension beats the flat fit, with a sign
surprise" for the full writeup.
"""
import numpy as np
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
from _vuong_fit_joint import build_split

F0, ALPHA_FI, K_WL, K_WG = 5.5650, 0.6083, 2.1475, 0.5659

train, test = build_split()
rows = [(c, 'train') for c in train] + [(c, 'test') for c in test]

_log_re_sg_train = np.array([
    np.log(c['rho_g'] * c['U_sg'] * c['D'] / c['mu_g']) for c in train
])
MEAN_LOG_RE_SG = float(_log_re_sg_train.mean())
print(f"mean log(Re_SG) over train set = {MEAN_LOG_RE_SG:.4f}  (should match fit's 13.2389)")

print(f"{'split':>5} {'U_sl':>8} {'U_sg':>8} {'Re_SG':>10} {'f_i_mult':>9} {'psi_meas':>9} {'psi_pred':>9} "
      f"{'hld_err%':>9} {'dpdz_meas':>10} {'dpdz_pred':>10} {'dpdz_err%':>10} {'kh_tot':>8}")

results = []
for c, split in rows:
    Re_SG = c['rho_g'] * c['U_sg'] * c['D'] / c['mu_g']
    f_i_mult = float(F0 * np.exp(ALPHA_FI * (np.log(Re_SG) - MEAN_LOG_RE_SG)))
    res = run_real_data_condition(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                   c['mu_l'], c['mu_g'], c['L'],
                                   f_i_mult=f_i_mult, k_wl=K_WL, k_wg=K_WG)
    psi_pred = res['alpha_l_pred']
    dpdz_pred = abs(res['dPdz_pred_liquid'])
    hld_err = (psi_pred - c['psi']) / c['psi'] * 100
    dpdz_err = (dpdz_pred - c['dpdz_measured']) / c['dpdz_measured'] * 100
    excluded = res['kh_total'] > 0
    results.append((split, c, Re_SG, f_i_mult, psi_pred, res['dPdz_pred_liquid'], hld_err, dpdz_err, excluded, res['kh_total']))
    flag = " EXCL" if excluded else ""
    print(f"{split:>5} {c['U_sl']:8.4f} {c['U_sg']:8.3f} {Re_SG:10.1f} {f_i_mult:9.4f} {c['psi']:9.4f} {psi_pred:9.4f} "
          f"{hld_err:9.2f} {c['dpdz_measured']:10.2f} {dpdz_pred:10.2f} {dpdz_err:10.2f} "
          f"{res['kh_total']:8.1f}{flag}", flush=True)

print()
for split in ('train', 'test'):
    rs = [r for r in results if r[0] == split and not r[8]]
    if not rs:
        continue
    hld_errs = [abs(r[6]) for r in rs]
    dpdz_errs = [abs(r[7]) for r in rs]
    print(f"{split}: n={len(rs)} (of {sum(1 for r in results if r[0]==split)})  "
          f"mean|hld_err%|={sum(hld_errs)/len(hld_errs):.2f}  "
          f"mean|dpdz_err%|={sum(dpdz_errs)/len(dpdz_errs):.2f}")

usg_arr = np.array([r[1]['U_sg'] for r in results])
herr_arr = np.array([r[6] for r in results])
print(f"\ncorrelation(U_sg, holdup_err%) = {np.corrcoef(usg_arr, herr_arr)[0,1]:.3f}")

print()
print("# ROWS tuple format for plot_vuong_powerlaw.py: "
      "(split, U_sl, U_sg, Re_SG, f_i_mult, psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh_total)")
for split, c, Re_SG, f_i_mult, psi_pred, dPdz_pred_liquid, hld_err, dpdz_err, excluded, kh in results:
    print(f'    ("{split}", {c["U_sl"]:.4f}, {c["U_sg"]:.3f}, {Re_SG:.1f}, {f_i_mult:.4f}, {c["psi"]:.4f}, {psi_pred:.4f}, '
          f'{c["dpdz_measured"]:.2f}, {abs(dPdz_pred_liquid):.2f}, {int(kh)}),')
