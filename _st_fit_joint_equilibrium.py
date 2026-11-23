"""
Step (a) of the ST-dedicated-fit plan: refit (f_i_mult, k_wl, k_wg) via
TRUE jax.grad through the closed-form Taitel-Dukler equilibrium path
(same machinery as _vuong_fit_joint_equilibrium.py), targeting the 21
Vuong ST conditions that have BOTH measured holdup (HL[-]) and measured
dP/dL -- NOT the 36 SW conditions the production calibration was
originally fit on. Never touches the dynamic/transient solver during
fitting, so this is decoupled from the disc<0 instability question
entirely (the equilibrium solve is a plain algebraic root-find).

Train/test split: 2 of the 23 ST rows have no measured HL (idx 1480,
1530) and are excluded entirely. Of the remaining 21, one point is held
out per U_sl cluster that has more than one point (6 clusters: U_sl in
{0.10, 0.07, 0.05, 0.04, 0.02, 0.01}; the single-point U_sl=0.03 cluster,
idx 1508, stays entirely in train since holding it out would remove that
velocity from training altogether), picked as the median-U_sg point
within each cluster for a representative (not just-the-extremes) test
set. 15 train / 6 test.

Warm-started from the production SW-fit values (f_i_mult=3.8893,
k_wl=1.5419, k_wg=0.8413) rather than (1,1,1) -- already in a reasonable
ballpark, should converge faster.
"""
import math
import time
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import optax
import pandas as pd
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)

W_DPDZ = 1.0
LR = 0.06
N_EPOCHS = 3000
PRINT_EVERY = 100

TEST_IDX = [805, 836, 1457, 1487, 1523, 1543]
TRAIN_IDX = [797, 820, 821, 822, 826, 832, 840, 823,
             1464, 1479, 1496, 1508, 1522, 1524, 1550]

F_I_MULT_0, K_WL_0, K_WG_0 = 3.8893, 1.5419, 0.8413

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]


def _cols(idx_list):
    sub = st.loc[idx_list]
    return {
        'U_sl': [float(v) for v in sub['vSL[m/s]']],
        'U_sg': [float(v) for v in sub['vSG[m/s]']],
        'D': [float(v) for v in sub['Pipe Diameter[m]']],
        'rho_l': [float(v) for v in sub['rhoL[kg/m3]']],
        'rho_g': [float(v) for v in sub['rhoG[kg/m3]']],
        'mu_l': [float(v) for v in sub['viscosityL[Pa.s]']],
        'mu_g': [float(v) for v in sub['viscosityG[Pa.s]']],
        'psi': [float(v) for v in sub['HL[-]']],
        'dpdz_measured': [float(v) for v in sub['dP/dL[Pa/m]']],
    }


train_c = _cols(TRAIN_IDX)
test_c = _cols(TEST_IDX)


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


def main():
    z = jnp.array([jnp.log(F_I_MULT_0), jnp.log(K_WL_0), jnp.log(K_WG_0)])
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)

    best_loss = float('inf')
    best_epoch = None
    best_z = z

    t_start = time.time()
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grad = grad_fn(z, train_c)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = z + updates

        f_i_mult, k_wl, k_wg = (float(jnp.exp(z[0])), float(jnp.exp(z[1])), float(jnp.exp(z[2])))
        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_epoch, best_z = loss_val, epoch, z

        if epoch % PRINT_EVERY == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:4d}  f_i_mult={f_i_mult:.4f}  k_wl={k_wl:.4f}  k_wg={k_wg:.4f}  "
                  f"loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})", flush=True)

    total_dt = time.time() - t_start
    print()
    print(f"Total wall-clock for {N_EPOCHS} epochs: {total_dt:.2f}s ({total_dt/N_EPOCHS*1000:.2f}ms/epoch)")
    f_best = (float(jnp.exp(best_z[0])), float(jnp.exp(best_z[1])), float(jnp.exp(best_z[2])))
    print(f"Best epoch: {best_epoch}  f_i_mult={f_best[0]:.4f}  k_wl={f_best[1]:.4f}  "
          f"k_wg={f_best[2]:.4f}  loss={best_loss:.6f}")

    print()
    print("=== Test-set evaluation at best-epoch params (NOT used in fitting) ===")
    (test_loss, (test_lh, test_ld)), _ = grad_fn(best_z, test_c)
    print(f"test loss={float(test_loss):.6f}  holdup={float(test_lh):.6f}  dpdz={float(test_ld):.6f}")

    print()
    print("=== Per-condition (holdup, dpdz) at best-epoch params, train+test ===")
    for label, cols in [("TRAIN", train_c), ("TEST", test_c)]:
        for i in range(len(cols['U_sl'])):
            alpha_pred, dpdz_pred = predict_one(
                cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], *f_best
            )
            alpha_err_pct = 100.0 * (float(alpha_pred) - cols['psi'][i]) / cols['psi'][i]
            dpdz_err_pct = 100.0 * (abs(float(dpdz_pred)) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
            print(f"  [{label}] U_sl={cols['U_sl'][i]:.4f} U_sg={cols['U_sg'][i]:.3f}  "
                  f"holdup: pred={float(alpha_pred):.4f} meas={cols['psi'][i]:.4f} err={alpha_err_pct:+.1f}%  "
                  f"dpdz: pred={float(dpdz_pred):.2f} meas={cols['dpdz_measured'][i]:.2f} err={dpdz_err_pct:+.1f}%")


if __name__ == '__main__':
    main()
