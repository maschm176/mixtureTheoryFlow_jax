"""
Quick power-law test: f_i_mult = exp(log_f0 + alpha_fi*(log(Re_SG) - mean)),
k_wl/k_wg stay flat scalars. 4 free parameters total vs. the flat fit's 3,
vs. the NN closure's 27 (which badly overfit 29 conditions -- see CLAUDE.md
"NN closure attempt..."). True jax.grad through the equilibrium path, same
machinery as _vuong_fit_joint_equilibrium.py.

Warm-started at alpha_fi=0, log_f0=log(3.8893), k_wl=1.5419, k_wg=0.8413 --
the already-converged flat fit -- so epoch 0's loss exactly reproduces the
known baseline (0.033699), and training only moves away from it if the
data rewards a Re_SG-dependence.

The actual question this answers: does a single extra parameter (alpha_fi)
close much of the gap the NN needed 27 (overfit) parameters to chase? And
does the fitted alpha_fi land anywhere near Carraretto et al.'s own
Re_SG^-0.53 (see References) -- fit on a completely different fluid pair
(water-air, not oil-nitrogen)?
"""
import time
import jax
import jax.numpy as jnp
import optax
import numpy as np
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _vuong_fit_joint import build_split

W_DPDZ = 1.0
LR = 0.02
N_EPOCHS = 3000

train, test = build_split()


def _cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'psi', 'dpdz_measured']
    return {k: [float(c[k]) for c in conds] for k in keys}


train_c = _cols(train)
test_c = _cols(test)

_log_re_sg_train = np.array([
    np.log(train_c['rho_g'][i] * train_c['U_sg'][i] * train_c['D'][i] / train_c['mu_g'][i])
    for i in range(len(train_c['U_sl']))
])
MEAN_LOG_RE_SG = float(_log_re_sg_train.mean())
print(f"mean log(Re_SG) over train set = {MEAN_LOG_RE_SG:.4f}")


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, log_f0, alpha_fi, log_kwl, log_kwg):
    Re_SG = rho_g * U_sg * D / mu_g
    f_i_mult = jnp.exp(log_f0 + alpha_fi * (jnp.log(Re_SG) - MEAN_LOG_RE_SG))
    k_wl = jnp.exp(log_kwl)
    k_wg = jnp.exp(log_kwg)

    Q = U_sl + U_sg
    ht_eq = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                        f_i_mult, k_wl, k_wg)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht_eq, D)
    alpha_l_eq = A_l / A
    alpha_g_eq = A_g / A
    u_l_eq = U_sl / jnp.maximum(alpha_l_eq, 1e-9)
    u_g_eq = U_sg / jnp.maximum(alpha_g_eq, 1e-9)
    Delta_eq = rho_l * u_l_eq - rho_g * u_g_eq
    dPdz_l_eq, _ = predicted_pressure_gradient(
        alpha_l_eq, Delta_eq, Q, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg,
    )
    return alpha_l_eq, dPdz_l_eq, (f_i_mult, k_wl, k_wg)


def loss_fn(z, cols):
    log_f0, alpha_fi, log_kwl, log_kwg = z[0], z[1], z[2], z[3]
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred, _ = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i],
            log_f0, alpha_fi, log_kwl, log_kwg,
        )
        alpha_err = (alpha_pred - cols['psi'][i]) / cols['psi'][i]
        dpdz_err = (jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
        sq_h.append(alpha_err ** 2)
        sq_d.append(dpdz_err ** 2)
    loss_h = jnp.mean(jnp.stack(sq_h))
    loss_d = jnp.mean(jnp.stack(sq_d))
    return loss_h + W_DPDZ * loss_d, (loss_h, loss_d)


grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
eval_fn = jax.jit(loss_fn)


def main():
    z = jnp.array([jnp.log(3.8893), 0.0, jnp.log(1.5419), jnp.log(0.8413)])
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)

    best_loss, best_z, best_epoch = float('inf'), z, None

    t_start = time.time()
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grad = grad_fn(z, train_c)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = z + updates

        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_z, best_epoch = loss_val, z, epoch

        if epoch % 150 == 0 or epoch == N_EPOCHS - 1:
            f0, alpha_fi = float(jnp.exp(z[0])), float(z[1])
            kwl, kwg = float(jnp.exp(z[2])), float(jnp.exp(z[3]))
            print(f"epoch {epoch:4d}  f0={f0:.4f}  alpha_fi={alpha_fi:+.4f}  k_wl={kwl:.4f}  "
                  f"k_wg={kwg:.4f}  loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})",
                  flush=True)

    dt = time.time() - t_start
    print(f"\nTotal wall-clock: {dt:.1f}s")
    f0, alpha_fi = float(jnp.exp(best_z[0])), float(best_z[1])
    kwl, kwg = float(jnp.exp(best_z[2])), float(jnp.exp(best_z[3]))
    print(f"Best epoch: {best_epoch}  f0={f0:.4f}  alpha_fi={alpha_fi:+.4f}  "
          f"k_wl={kwl:.4f}  k_wg={kwg:.4f}  loss={best_loss:.6f}")
    print(f"(Carraretto et al. 2020's own fit, water-air, different fluid pair: alpha=-0.53)")

    train_loss, (train_lh, train_ld) = eval_fn(best_z, train_c)
    test_loss, (test_lh, test_ld) = eval_fn(best_z, test_c)
    print(f"\ntrain: loss={float(train_loss):.6f}  holdup={float(train_lh):.6f}  dpdz={float(train_ld):.6f}")
    print(f"test:  loss={float(test_loss):.6f}  holdup={float(test_lh):.6f}  dpdz={float(test_ld):.6f}")
    print("\nReference (flat 3-scalar fit, fully converged): "
          "train loss=0.033699 (holdup=0.010319 dpdz=0.023380); "
          "test loss=0.042799 (holdup=0.022535 dpdz=0.020265)")
    print("Reference (27-weight NN closure, overfit): "
          "train loss=0.005263 (holdup=0.002971 dpdz=0.002292); "
          "test loss=0.101338 (holdup=0.005863 dpdz=0.095475)")

    all_rows = []
    for split, cols in (('train', train_c), ('test', test_c)):
        for i in range(len(cols['U_sl'])):
            alpha_pred, dpdz_pred, (f, kl, kg) = predict_one(
                cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i],
                best_z[0], best_z[1], best_z[2], best_z[3],
            )
            hld_err = float((alpha_pred - cols['psi'][i]) / cols['psi'][i] * 100)
            all_rows.append((split, cols['U_sg'][i], hld_err))

    usg_arr = np.array([r[1] for r in all_rows])
    err_arr = np.array([r[2] for r in all_rows])
    corr = np.corrcoef(usg_arr, err_arr)[0, 1]
    print(f"\ncorrelation(U_sg, holdup_err%) = {corr:.3f}  (flat fit: 0.610, NN: 0.281[overfit])")


if __name__ == '__main__':
    main()
