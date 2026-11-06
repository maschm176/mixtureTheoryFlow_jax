"""
Single-parameter f_i_mult optimization (k_wl=k_wg=1 fixed, wall friction
uncalibrated) against the SAME holdup+dPdz loss as the joint fit -- to get a
genuine epoch-by-epoch convergence trajectory for f_i_mult alone, unlike
_vuong_fionly_scan.py's grid scan (8 independent forward solves, no
gradient descent, no epoch dimension at all).

Reuses _vuong_fit_joint.py's build_split / eval_conditions_subprocess /
combined_loss_subprocess UNCHANGED (same loss definition, same train/test
split, same subprocess-per-batch architecture -- see that module's
docstring for why that architecture is necessary) so this run is directly
comparable to the joint fit, not a different methodology measuring a
different thing.

Central finite-difference gradient in log-space (z=log(f_i_mult)), Adam,
best-epoch tracking from the start. 3 forward-eval batches/epoch (center,
+h, -h) instead of the joint fit's 7, since there's only one free
parameter here -- roughly 3/7 the per-epoch wall-clock cost.

N_EPOCHS matches the joint fit's actual stopping point (20, not its
planned-but-unused 30) for a fair, equal-budget comparison. If this run is
STILL decreasing at epoch 19 the way the grid scan already was at its
f_i_mult=2.5 edge, that itself is further evidence there's no genuine
single-parameter optimum to find, not a sign the run was cut short
arbitrarily.
"""
import time
import pickle
import numpy as np
import optax
from _vuong_fit_joint import build_split, combined_loss_subprocess, eval_conditions_subprocess

H = 1e-3
LR = 0.06
N_EPOCHS = 20


def main():
    train, test = build_split()
    print(f"train: {len(train)} conditions, test: {len(test)} conditions")
    print("f_i_mult-only (k_wl=k_wg=1 fixed) gradient-descent run, "
          "same loss/split as the joint fit")

    z = np.array([np.log(1.0)])
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)

    best_loss = np.inf
    best_epoch = None
    best_f = None

    for epoch in range(N_EPOCHS):
        t0 = time.time()
        f_center = float(np.exp(z[0]))
        loss_c, lh_c, ld_c, n_exc, n_tot = combined_loss_subprocess(f_center, 1.0, 1.0, 'train')

        f_p = float(np.exp(z[0] + H))
        f_m = float(np.exp(z[0] - H))
        loss_p, _, _, _, _ = combined_loss_subprocess(f_p, 1.0, 1.0, 'train')
        loss_m, _, _, _, _ = combined_loss_subprocess(f_m, 1.0, 1.0, 'train')
        grad = np.array([(loss_p - loss_m) / (2 * H)])

        updates, opt_state = optimizer.update(grad, opt_state)
        z = np.asarray(z + updates)

        if loss_c < best_loss:
            best_loss = loss_c
            best_epoch = epoch
            best_f = f_center

        dt = time.time() - t0
        print(f"epoch {epoch:2d}  f_i_mult={f_center:.4f}  loss={loss_c:.6f} "
              f"(holdup={lh_c:.6f} dpdz={ld_c:.6f})  grad_z={grad[0]:+.4f}  "
              f"n_excluded={n_exc}/{n_tot}  ({dt:.1f}s)", flush=True)

        with open('vuong2016_fionly_fit_full.pkl', 'wb') as fo:
            pickle.dump({
                'epoch': epoch, 'z': z, 'f_i_mult': f_center,
                'loss': loss_c, 'loss_holdup': lh_c, 'loss_dpdz': ld_c,
                'best_epoch': best_epoch, 'best_f_i_mult': best_f, 'best_loss': best_loss,
            }, fo)

    print()
    print(f"Best epoch: {best_epoch}  f_i_mult={best_f:.4f}  loss={best_loss:.6f}")
    print()
    print("=== Test-set evaluation at best-epoch params (NOT used in fitting) ===")
    lh_test, ld_test, n_exc_test, n_tot_test = eval_conditions_subprocess(best_f, 1.0, 1.0, 'test')
    print(f"test holdup loss={lh_test:.6f}  test dPdz loss={ld_test:.6f}  n_excluded={n_exc_test}/{n_tot_test}")


if __name__ == '__main__':
    main()
