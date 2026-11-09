"""
Retrains the 27-weight NN closure (_vuong_fit_nn_closure.py's exact
architecture/warm-start/hyperparameters, deterministic PRNGKey(0) -- same
network it already reported: train loss=0.005263, test loss=0.101338,
badly overfit), this time logging the full loss trajectory (every 50
epochs) for a convergence plot, and cross-checking every prediction
against the dynamic scan (kh_total) the way every other fit in this
project is checked -- never done for the NN closure before now.

Produces the same two tuple blocks plot_vuong_powerlaw.py consumes
(ROWS and the epoch-trajectory arrays), printed for plot_vuong_nn.py to
hardcode. See CLAUDE.md "NN closure overfits on 29 points...".
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from _vuong_fit_nn_closure import (
    ClosureNetwork, predict_one, train_c, test_c, loss_fn, W_DPDZ,
)
from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition

LR = 0.02
WEIGHT_DECAY = 1e-3
N_EPOCHS = 5000
LOG_EVERY = 50

net = ClosureNetwork(jax.random.PRNGKey(0))
grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
eval_fn = eqx.filter_jit(loss_fn)

optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)
opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

best_loss, best_net, best_epoch = float('inf'), net, None
epochs_log, loss_log, loss_h_log, loss_d_log = [], [], [], []

t0 = time.time()
for epoch in range(N_EPOCHS):
    (loss_val, (lh, ld)), grads = grad_fn(net, train_c)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(net, eqx.is_array))
    net = eqx.apply_updates(net, updates)

    loss_val = float(loss_val)
    if loss_val < best_loss:
        best_loss, best_net, best_epoch = loss_val, net, epoch

    if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
        epochs_log.append(epoch)
        loss_log.append(loss_val)
        loss_h_log.append(float(lh))
        loss_d_log.append(float(ld))

print(f"trained in {time.time()-t0:.1f}s. best_epoch={best_epoch}  best_loss={best_loss:.6f}")

train_loss, (train_lh, train_ld) = eval_fn(best_net, train_c)
test_loss, (test_lh, test_ld) = eval_fn(best_net, test_c)
print(f"train: loss={float(train_loss):.6f} (hld={float(train_lh):.6f} dpdz={float(train_ld):.6f})")
print(f"test:  loss={float(test_loss):.6f} (hld={float(test_lh):.6f} dpdz={float(test_ld):.6f})")

# per-condition dynamic-scan evaluation (kh_total check, never done for this fit before)
print(f"\n{'split':>5} {'U_sl':>8} {'U_sg':>8} {'f_i_mult':>9} {'k_wl':>7} {'k_wg':>7} "
      f"{'psi_meas':>9} {'psi_pred':>9} {'hld_err%':>9} {'dpdz_meas':>10} {'dpdz_pred':>10} "
      f"{'dpdz_err%':>10} {'kh_tot':>8}")

results = []
for split, cols in (('train', train_c), ('test', test_c)):
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred_eq, dpdz_pred_eq, (f, kl, kg) = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], best_net,
        )
        f, kl, kg = float(f), float(kl), float(kg)
        res = run_real_data_condition(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i],
                                       cols['rho_l'][i], cols['rho_g'][i], cols['mu_l'][i],
                                       cols['mu_g'][i], 85.0, f_i_mult=f, k_wl=kl, k_wg=kg)
        psi_pred = res['alpha_l_pred']
        dpdz_pred = abs(res['dPdz_pred_liquid'])
        psi_meas = cols['psi'][i]
        dpdz_meas = cols['dpdz_measured'][i]
        hld_err = (psi_pred - psi_meas) / psi_meas * 100
        dpdz_err = (dpdz_pred - dpdz_meas) / dpdz_meas * 100
        kh = res['kh_total']
        results.append((split, cols['U_sl'][i], cols['U_sg'][i], f, kl, kg,
                         psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh))
        flag = " EXCL" if kh > 0 else ""
        print(f"{split:>5} {cols['U_sl'][i]:8.4f} {cols['U_sg'][i]:8.3f} {f:9.4f} {kl:7.4f} {kg:7.4f} "
              f"{psi_meas:9.4f} {psi_pred:9.4f} {hld_err:9.2f} {dpdz_meas:10.2f} {dpdz_pred:10.2f} "
              f"{dpdz_err:10.2f} {kh:8.1f}{flag}", flush=True)

print()
for split in ('train', 'test'):
    rs = [r for r in results if r[0] == split and r[10] == 0]
    hld_errs = [abs((r[7] - r[6]) / r[6] * 100) for r in rs]
    dpdz_errs = [abs((r[9] - r[8]) / r[8] * 100) for r in rs]
    print(f"{split}: n={len(rs)} (of {sum(1 for r in results if r[0]==split)})  "
          f"mean|hld_err%|={sum(hld_errs)/len(hld_errs):.2f}  "
          f"mean|dpdz_err%|={sum(dpdz_errs)/len(dpdz_errs):.2f}")

usg_arr = np.array([r[2] for r in results])
herr_arr = np.array([(r[7] - r[6]) / r[6] * 100 for r in results])
print(f"\ncorrelation(U_sg, holdup_err%) = {np.corrcoef(usg_arr, herr_arr)[0,1]:.3f}")

eqx.tree_serialise_leaves('vuong_nn_closure_best.eqx', best_net)

print("\n# EPOCHS / LOSS / LOSS_H / LOSS_D for plot_vuong_nn.py")
print("EPOCHS =", epochs_log)
print("LOSS =", [round(v, 6) for v in loss_log])
print("LOSS_HLD =", [round(v, 6) for v in loss_h_log])
print("LOSS_DPDZ =", [round(v, 6) for v in loss_d_log])

print("\n# ROWS tuple format: (split, U_sl, U_sg, f_i_mult, k_wl, k_wg, psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh_total)")
for r in results:
    split, usl, usg, f, kl, kg, psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh = r
    print(f'    ("{split}", {usl:.4f}, {usg:.3f}, {f:.4f}, {kl:.4f}, {kg:.4f}, {psi_meas:.4f}, {psi_pred:.4f}, '
          f'{dpdz_meas:.2f}, {dpdz_pred:.2f}, {int(kh)}),')
