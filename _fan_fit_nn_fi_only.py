"""
Fan (2005) counterpart of the Vuong "Option B" hybrid
(_vuong_fit_nn_fi_only.py): f_i_mult = NN(log Re_SL, log Re_SG), k_wl/k_wg
stay flat scalars, all parameters trained jointly through the true-
jax.grad closed-form Taitel-Dukler equilibrium path. Same 2->3(tanh)->
1(softplus) architecture (13 weights) + 2 flat scalars = 15 free params,
kept identical to Vuong's so the two fitted f_i_mult(Re_SG) trends are a
direct apples-to-apples comparison, not confounded by architecture
differences.

DOES NOT TOUCH, READ FROM AS A TARGET, OR OVERWRITE ANY _vuong_*.py FILE,
vuong_plots/ SCRIPT, OR figures/*vuong*/figures/nn_fi_only/ OUTPUT -- this
is a fully separate file/namespace (fan_plots/, figures/fan_nn_fi_only/,
*_fan_*.{py,eqx,pkl}), exactly as requested.

EXTENDED (2026-11-13) from the 11-condition horizontal-only subset (which
forced leave-one-out CV, too few points for a fixed split) to the full
49-condition kh-clean set (11 horizontal + 38 inclined, see _fan_data.py).
Now large enough for a fixed 39/10 train/test split (_fan_data.build_split,
grouped by angle x nominal U_sg level so both splits span the full range),
mirroring Vuong's methodology directly instead of falling back to LOO-CV.
beta is now threaded through predict_one -- previously implicitly 0.0,
which only happened to be correct because every condition WAS horizontal.
"""
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _fan_data import load_all_conditions, build_split

W_DPDZ = 1.0
LR = 0.02
WEIGHT_DECAY = 1e-3
N_EPOCHS = 3000

# Warm start -- from _fan_fit_joint_equilibrium.py's converged flat fit on
# the expanded 49-condition train split (2026-11-13): best epoch 680,
# train loss=0.192877 (holdup=0.138665 dpdz=0.054212), test loss=0.338893
# (holdup=0.287964 dpdz=0.050929) -- the flat fit itself already shows a
# real holdup generalization gap once inclination is in the mix, which is
# exactly the kind of residual structure the NN's extra flexibility (vs.
# flat scalars) is meant to help with.
WARM_FI = 0.8939
WARM_KWL = 0.5158
WARM_KWG = 1.3607

ALL_CONDITIONS = load_all_conditions()
TRAIN_CONDITIONS, TEST_CONDITIONS = build_split(ALL_CONDITIONS)


def _cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'dpdz_measured', 'sigma', 'beta']
    out = {k: [float(c[k]) for c in conds] for k in keys}
    out['psi'] = [c['psi'] for c in conds]
    return out


def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))


def _feat_stats(cols):
    re_pairs = []
    for i in range(len(cols['U_sl'])):
        Re_SL = cols['rho_l'][i] * cols['U_sl'][i] * cols['D'][i] / cols['mu_l'][i]
        Re_SG = cols['rho_g'][i] * cols['U_sg'][i] * cols['D'][i] / cols['mu_g'][i]
        re_pairs.append([np.log(Re_SL), np.log(Re_SG)])
    arr = np.array(re_pairs)
    return jnp.array(arr.mean(axis=0)), jnp.array(arr.std(axis=0))


class HybridParams(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    log_kwl: jax.Array
    log_kwg: jax.Array

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(2, 3, key=k1)
        self.layer2 = eqx.nn.Linear(3, 1, key=k2)
        self.layer2 = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            self.layer2,
            (jnp.zeros_like(self.layer2.weight), _inv_softplus(jnp.array([WARM_FI]))),
        )
        self.log_kwl = jnp.log(jnp.array(WARM_KWL))
        self.log_kwg = jnp.log(jnp.array(WARM_KWG))

    def f_i_mult(self, x):
        h = jnp.tanh(self.layer1(x))
        return jax.nn.softplus(self.layer2(h))[0]

    @property
    def k_wl(self):
        return jnp.exp(self.log_kwl)

    @property
    def k_wg(self):
        return jnp.exp(self.log_kwg)


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, beta, params, feat_mean, feat_std):
    Re_SL = rho_l * U_sl * D / mu_l
    Re_SG = rho_g * U_sg * D / mu_g
    x = (jnp.array([jnp.log(Re_SL), jnp.log(Re_SG)]) - feat_mean) / feat_std
    f_i_mult = params.f_i_mult(x)
    k_wl, k_wg = params.k_wl, params.k_wg

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
    return alpha_l_eq, dPdz_l_eq, (f_i_mult, k_wl, k_wg)


def make_loss_fn(feat_mean, feat_std):
    def loss_fn(params, cols):
        sq_h, sq_d = [], []
        n = len(cols['U_sl'])
        for i in range(n):
            alpha_pred, dpdz_pred, _ = predict_one(
                cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i], params,
                feat_mean, feat_std,
            )
            dpdz_err = (jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
            sq_d.append(dpdz_err ** 2)
            if cols['psi'][i] is not None:
                alpha_err = (alpha_pred - cols['psi'][i]) / cols['psi'][i]
                sq_h.append(alpha_err ** 2)
        loss_h = jnp.mean(jnp.stack(sq_h)) if sq_h else jnp.array(0.0)
        loss_d = jnp.mean(jnp.stack(sq_d))
        return loss_h + W_DPDZ * loss_d, (loss_h, loss_d)
    return loss_fn


def fit(train_cols, feat_mean, feat_std, seed=0, verbose=False):
    params = HybridParams(jax.random.PRNGKey(seed))
    loss_fn = make_loss_fn(feat_mean, feat_std)
    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))

    optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    best_loss, best_params, best_epoch = float('inf'), params, None
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grads = grad_fn(params, train_cols)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        params = eqx.apply_updates(params, updates)
        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_params, best_epoch = loss_val, params, epoch
        if verbose and (epoch % 500 == 0 or epoch == N_EPOCHS - 1):
            print(f"    epoch {epoch:4d}  loss={loss_val:.6f} "
                  f"(hld={float(lh):.6f} dpdz={float(ld):.6f})", flush=True)
    return best_params, best_loss, best_epoch, loss_fn


def _row_report(cols, conds, params, feat_mean, feat_std, label):
    n = len(cols['U_sl'])
    print(f"\n{label}: n={n}")
    print(f"{'U_sl':>9} {'U_sg':>8} {'angle':>6} {'Re_SG':>10} {'f_i_mult':>9} "
          f"{'psi_meas':>9} {'psi_pred':>9} {'dpdz_meas':>10} {'dpdz_pred':>10}")
    rows = []
    hld_errs, dpdz_errs = [], []
    for i in range(n):
        alpha_pred, dpdz_pred, (f, kl, kg) = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i], params,
            feat_mean, feat_std,
        )
        Re_SG = cols['rho_g'][i] * cols['U_sg'][i] * cols['D'][i] / cols['mu_g'][i]
        psi_meas = cols['psi'][i]
        psi_str = f"{psi_meas:9.4f}" if psi_meas is not None else f"{'--':>9}"
        angle = conds[i]['angle_deg']
        print(f"{cols['U_sl'][i]:9.5f} {cols['U_sg'][i]:8.3f} {angle:6.1f} {Re_SG:10.1f} "
              f"{float(f):9.4f} {psi_str} {float(alpha_pred):9.4f} "
              f"{cols['dpdz_measured'][i]:10.2f} {float(dpdz_pred):10.2f}")
        rows.append((cols['U_sl'][i], cols['U_sg'][i], Re_SG, float(f),
                     psi_meas, float(alpha_pred),
                     cols['dpdz_measured'][i], float(dpdz_pred), cols['sigma'][i]))
        dpdz_errs.append(abs(abs(float(dpdz_pred)) - cols['dpdz_measured'][i]) / abs(cols['dpdz_measured'][i]) * 100)
        if psi_meas is not None:
            hld_errs.append(abs(float(alpha_pred) - psi_meas) / psi_meas * 100)
    print(f"{label} mean|hld_err%|={sum(hld_errs)/len(hld_errs):.2f} (n={len(hld_errs)})  "
          f"mean|dpdz_err%|={sum(dpdz_errs)/len(dpdz_errs):.2f} (n={len(dpdz_errs)})")
    return rows


def main():
    train_cols = _cols(TRAIN_CONDITIONS)
    test_cols = _cols(TEST_CONDITIONS)
    n_train_psi = sum(1 for c in TRAIN_CONDITIONS if c['psi'] is not None)
    n_test_psi = sum(1 for c in TEST_CONDITIONS if c['psi'] is not None)
    print(f"Fan (2005), kh-clean, train/test split: "
          f"train n={len(TRAIN_CONDITIONS)} (holdup={n_train_psi})  "
          f"test n={len(TEST_CONDITIONS)} (holdup={n_test_psi})")
    print(f"warm start: f_i_mult={WARM_FI} k_wl={WARM_KWL} k_wg={WARM_KWG}\n")

    feat_mean, feat_std = _feat_stats(train_cols)  # stats from TRAIN only, no test leakage
    t0 = time.time()
    params, train_loss, best_epoch, loss_fn = fit(train_cols, feat_mean, feat_std, verbose=True)
    print(f"trained in {time.time()-t0:.1f}s. best_epoch={best_epoch} train_loss={train_loss:.6f}")
    print(f"k_wl={float(params.k_wl):.4f}  k_wg={float(params.k_wg):.4f}")

    test_loss, (test_lh, test_ld) = loss_fn(params, test_cols)
    print(f"test_loss={float(test_loss):.6f}  (holdup={float(test_lh):.6f} dpdz={float(test_ld):.6f})")

    train_rows = _row_report(train_cols, TRAIN_CONDITIONS, params, feat_mean, feat_std, "TRAIN")
    test_rows = _row_report(test_cols, TEST_CONDITIONS, params, feat_mean, feat_std, "TEST")

    eqx.tree_serialise_leaves('fan_nn_fi_only_best.eqx', params)
    print("\nsaved params to fan_nn_fi_only_best.eqx")

    print("\n# TRAIN_ROWS / TEST_ROWS tuples for fan_plots/plot_fan_nn_fi_only.py:")
    print("# (U_sl, U_sg, Re_SG, f_i_mult, psi_meas, psi_pred, dpdz_meas, dpdz_pred, sigma)")
    for name, rows in [("TRAIN_ROWS", train_rows), ("TEST_ROWS", test_rows)]:
        print(f"{name} = [")
        for r in rows:
            psi_m = f"{r[4]:.4f}" if r[4] is not None else "None"
            print(f"    ({r[0]:.6f}, {r[1]:.3f}, {r[2]:.1f}, {r[3]:.4f}, {psi_m}, {r[5]:.4f}, "
                  f"{r[6]:.2f}, {r[7]:.2f}, {r[8]:.3f}),")
        print("]")


if __name__ == '__main__':
    main()
