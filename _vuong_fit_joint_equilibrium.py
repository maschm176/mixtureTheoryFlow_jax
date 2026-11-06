"""
Proof-of-concept: refit (f_i_mult, k_wl, k_wg) using TRUE jax.grad through
the closed-form Taitel-Dukler equilibrium path, instead of central finite
differences through the dynamic Strang-split scan (_vuong_fit_joint.py).

Confirms two things:
  1. True backprop works end-to-end through solve_taitel_dukler_holdup's
     custom_jvp -> geometry -> velocity recovery -> predicted_pressure_
     gradient -> loss, with ZERO scan and ZERO finite differences.
  2. It converges to (approximately) the SAME (f_i_mult, k_wl, k_wg) the
     finite-difference joint fit already found (1.3265, 0.7532, 1.3173) --
     real confirmation the new, much cheaper training path reproduces the
     old one's answer, not just that predictions agree at one parameter
     point (see _vuong_equilibrium_crosscheck.py for that separate check).

Loss definition matches _vuong_fit_joint.py's eval_conditions() exactly:
  alpha_err = (alpha_l_pred - psi_meas) / psi_meas
  dpdz_err  = (abs(dPdz_pred_liquid) - dpdz_measured) / dpdz_measured
  loss = mean(alpha_err^2) + W_DPDZ * mean(dpdz_err^2)
over the 29 TRAINING conditions (same build_split()). No kh exclusions
needed here: all 36 conditions were confirmed kh_total=0 at the fitted
point by the crosscheck script, and the equilibrium path has no kh_total
of its own anyway (that diagnostic is a property of the dynamic
TRANSIENT path, not the final equilibrium point).

Plain Python loop over conditions (not vmap) -- unrolled under jax.jit,
avoids any question of whether vmap batches cleanly through
solve_taitel_dukler_holdup's custom_jvp, and 29 unrolled iterations is a
trivial compile.
"""
import time
import jax
import jax.numpy as jnp
import optax
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _vuong_fit_joint import build_split

W_DPDZ = 1.0
LR = 0.06
N_EPOCHS = 3000
PRINT_EVERY = 25

train, test = build_split()

# Stack conditions into plain Python lists of floats (traced as constants,
# not arguments -- only z is differentiated).
def _cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'psi', 'dpdz_measured']
    return {k: [float(c[k]) for c in conds] for k in keys}

train_c = _cols(train)
test_c = _cols(test)


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg):
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
    return alpha_l_eq, dPdz_l_eq


def loss_fn(z, cols):
    f_i_mult, k_wl, k_wg = jnp.exp(z[0]), jnp.exp(z[1]), jnp.exp(z[2])
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], f_i_mult, k_wl, k_wg,
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
    z = jnp.array([jnp.log(1.0), jnp.log(1.0), jnp.log(1.0)])
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)

    best_loss = float('inf')
    best_epoch = None
    best_z = z

    t_start = time.time()
    for epoch in range(N_EPOCHS):
        t0 = time.time()
        (loss_val, (lh, ld)), grad = grad_fn(z, train_c)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = z + updates
        dt = time.time() - t0

        f_i_mult, k_wl, k_wg = (float(jnp.exp(z[0])), float(jnp.exp(z[1])), float(jnp.exp(z[2])))
        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_epoch, best_z = loss_val, epoch, z

        if epoch % PRINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:4d}  f_i_mult={f_i_mult:.4f}  k_wl={k_wl:.4f}  k_wg={k_wg:.4f}  "
                  f"loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})  "
                  f"grad={[round(float(g), 5) for g in grad]}  ({dt*1000:.2f}ms)", flush=True)

    total_dt = time.time() - t_start
    print()
    print(f"Total wall-clock for {N_EPOCHS} epochs: {total_dt:.2f}s "
          f"({total_dt/N_EPOCHS*1000:.1f}ms/epoch)")
    f_best = (float(jnp.exp(best_z[0])), float(jnp.exp(best_z[1])), float(jnp.exp(best_z[2])))
    print(f"Best epoch: {best_epoch}  f_i_mult={f_best[0]:.4f}  k_wl={f_best[1]:.4f}  "
          f"k_wg={f_best[2]:.4f}  loss={best_loss:.6f}")
    print()
    print("Reference (finite-difference joint fit, _vuong_fit_joint.py, epoch 5): "
          "f_i_mult=1.3265  k_wl=0.7532  k_wg=1.3173  loss=0.043427")

    print()
    print("=== Test-set evaluation at best-epoch params (NOT used in fitting) ===")
    (test_loss, (test_lh, test_ld)), _ = grad_fn(best_z, test_c)
    print(f"test loss={float(test_loss):.6f}  holdup={float(test_lh):.6f}  dpdz={float(test_ld):.6f}")


if __name__ == '__main__':
    main()
