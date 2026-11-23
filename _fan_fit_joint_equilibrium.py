"""
Flat (f_i_mult, k_wl, k_wg) joint fit for Fan (2005) -- the Fan-specific
counterpart of _vuong_fit_joint_equilibrium.py, same true-jax.grad-through-
the-closed-form-Taitel-Dukler-equilibrium-path machinery (no scan, no
finite differences), existing purely as a warm-start reference point for
_fan_fit_nn_fi_only.py, mirroring how Vuong's flat fit preceded its power
law / Option B hybrid.

EXTENDED (2026-11-13) from the 11-condition horizontal-only subset to the
full 49-condition kh-clean set (11 horizontal + 38 inclined) -- see
_fan_data.py's module docstring for how the inclined rows were validated
(direct run_real_data_condition check, zero NaN, matching closed-form
equilibrium) and _fan_data.build_split for the train/test split (39/10,
grouped by angle x nominal U_sg level so both splits span the full
angle/U_sg range). beta is now threaded through predict_one (previously
implicitly 0.0 for every condition, which was fine when every condition
WAS horizontal but would have silently mispredicted every inclined one).

Loss: mean(holdup_err^2) over train conditions with measured psi, plus
W_DPDZ * mean(dpdz_err^2) over all train conditions. Test-set loss is
reported the same way, purely for monitoring -- this script is a
warm-start producer, not the final calibration (see
_fan_fit_nn_fi_only.py).
"""
import time
import jax
import jax.numpy as jnp
import optax
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _fan_data import load_all_conditions, build_split

W_DPDZ = 1.0
LR = 0.06
N_EPOCHS = 3000
PRINT_EVERY = 100

all_conditions = load_all_conditions()
train_conditions, test_conditions = build_split(all_conditions)
n_psi = sum(1 for c in all_conditions if c['psi'] is not None)
print(f"Fan (2005), kh-clean: n={len(all_conditions)}  (n with holdup={n_psi})")
print(f"train: n={len(train_conditions)}  test: n={len(test_conditions)}")


def _cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'dpdz_measured', 'beta']
    out = {k: [float(c[k]) for c in conds] for k in keys}
    out['psi'] = [c['psi'] for c in conds]  # may contain None
    return out


train_cols = _cols(train_conditions)
test_cols = _cols(test_conditions)


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, beta, f_i_mult, k_wl, k_wg):
    Q = U_sl + U_sg
    ht_eq = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                        f_i_mult, k_wl, k_wg, beta=beta)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht_eq, D)
    alpha_l_eq = A_l / A
    alpha_g_eq = A_g / A
    u_l_eq = U_sl / jnp.maximum(alpha_l_eq, 1e-9)
    u_g_eq = U_sg / jnp.maximum(alpha_g_eq, 1e-9)
    Delta_eq = rho_l * u_l_eq - rho_g * u_g_eq
    dPdz_l_eq, _ = predicted_pressure_gradient(
        alpha_l_eq, Delta_eq, Q, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg, beta=beta,
    )
    return alpha_l_eq, dPdz_l_eq


def loss_fn(z, cols):
    f_i_mult, k_wl, k_wg = jnp.exp(z[0]), jnp.exp(z[1]), jnp.exp(z[2])
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i],
            f_i_mult, k_wl, k_wg,
        )
        dpdz_err = (jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
        sq_d.append(dpdz_err ** 2)
        if cols['psi'][i] is not None:
            alpha_err = (alpha_pred - cols['psi'][i]) / cols['psi'][i]
            sq_h.append(alpha_err ** 2)
    loss_h = jnp.mean(jnp.stack(sq_h))
    loss_d = jnp.mean(jnp.stack(sq_d))
    return loss_h + W_DPDZ * loss_d, (loss_h, loss_d)


grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))
eval_fn = jax.jit(loss_fn)


def main():
    z = jnp.array([jnp.log(1.0), jnp.log(1.0), jnp.log(1.0)])
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)

    best_loss, best_epoch, best_z = float('inf'), None, z
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grad = grad_fn(z, train_cols)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = z + updates

        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_epoch, best_z = loss_val, epoch, z

        if epoch % PRINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            f, kl, kg = float(jnp.exp(z[0])), float(jnp.exp(z[1])), float(jnp.exp(z[2]))
            print(f"epoch {epoch:4d}  f_i_mult={f:.4f}  k_wl={kl:.4f}  k_wg={kg:.4f}  "
                  f"loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})", flush=True)

    dt = time.time() - t0
    f_best = (float(jnp.exp(best_z[0])), float(jnp.exp(best_z[1])), float(jnp.exp(best_z[2])))
    test_loss, (test_lh, test_ld) = eval_fn(best_z, test_cols)
    print(f"\nTotal wall-clock: {dt:.1f}s")
    print(f"Best epoch: {best_epoch}  f_i_mult={f_best[0]:.4f}  k_wl={f_best[1]:.4f}  "
          f"k_wg={f_best[2]:.4f}  train loss={best_loss:.6f}")
    print(f"Test loss={float(test_loss):.6f}  (holdup={float(test_lh):.6f} dpdz={float(test_ld):.6f})")


if __name__ == '__main__':
    main()
