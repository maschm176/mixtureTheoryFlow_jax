"""
Step (b): pool the 36 Vuong SW conditions with the 21 Vuong ST conditions
(those with valid measured HL) and fit the "Option B" hybrid --
f_i_mult = NN(log Re_SL, log Re_SG), k_wl/k_wg as plain flat scalars --
exactly mirroring _vuong_fit_nn_fi_only.py's architecture/training
machinery, just on the combined dataset and a recentered warm start.

Train/test split: SW's own existing split (29 train / 7 test, from
build_split() in _vuong_fit_joint.py) combined with the ST split built in
_st_fit_joint_equilibrium.py (15 train / 6 test, one held-out point per
multi-point U_sl cluster) -> 44 train / 13 test total.

Warm start: geometric mean (natural in log-space, which is how these
params are parameterized/optimized) of the two independently-fit flat
values -- SW: (3.8893, 1.5419, 0.8413); ST:
(10.6923, 2.6477, 0.8510) -- giving f_i_mult_0=6.449, k_wl_0=2.021,
k_wg_0=0.846 (k_wg barely moves, consistent with it needing almost no
adjustment between the two regimes in either fit).

Entirely equilibrium-path (solve_taitel_dukler_holdup -> td_geometry ->
velocity recovery -> predicted_pressure_gradient), true jax.grad, no
scan, no finite differences, no dependence on dynamic-solver stability
during training -- same reasoning as every other equilibrium-path fit in
this project.
"""
import time
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import optax
import numpy as np
import pandas as pd
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _vuong_fit_joint import build_split

W_DPDZ = 1.0
LR = 0.02
WEIGHT_DECAY = 1e-3
N_EPOCHS = 15000

WARM_FI = float(np.sqrt(3.8893 * 10.6923))   # 6.449
WARM_KWL = float(np.sqrt(1.5419 * 2.6477))   # 2.021
WARM_KWG = float(np.sqrt(0.8413 * 0.8510))   # 0.846

ST_TEST_IDX = [805, 836, 1457, 1487, 1523, 1543]
ST_TRAIN_IDX = [797, 820, 821, 822, 826, 832, 840, 823,
                1464, 1479, 1496, 1508, 1522, 1524, 1550]


def _st_cols(idx_list):
    ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
    st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
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


def _sw_cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'psi', 'dpdz_measured']
    return {k: [float(c[k]) for c in conds] for k in keys}


def _concat(a, b):
    return {k: a[k] + b[k] for k in a}


sw_train, sw_test = build_split()
sw_train_c, sw_test_c = _sw_cols(sw_train), _sw_cols(sw_test)
st_train_c, st_test_c = _st_cols(ST_TRAIN_IDX), _st_cols(ST_TEST_IDX)

train_c = _concat(sw_train_c, st_train_c)
test_c = _concat(sw_test_c, st_test_c)
print(f"pooled train: {len(train_c['U_sl'])} ({len(sw_train_c['U_sl'])} SW + {len(st_train_c['U_sl'])} ST)")
print(f"pooled test:  {len(test_c['U_sl'])} ({len(sw_test_c['U_sl'])} SW + {len(st_test_c['U_sl'])} ST)")


def _re_pair(cols, i):
    Re_SL = cols['rho_l'][i] * cols['U_sl'][i] * cols['D'][i] / cols['mu_l'][i]
    Re_SG = cols['rho_g'][i] * cols['U_sg'][i] * cols['D'][i] / cols['mu_g'][i]
    return Re_SL, Re_SG


_log_re = np.array([[np.log(r) for r in _re_pair(train_c, i)] for i in range(len(train_c['U_sl']))])
FEAT_MEAN = jnp.array(_log_re.mean(axis=0))
FEAT_STD = jnp.array(_log_re.std(axis=0))
print(f"feature standardization (pooled train set): mean={FEAT_MEAN}  std={FEAT_STD}")
print(f"warm start: f_i_mult={WARM_FI:.4f}  k_wl={WARM_KWL:.4f}  k_wg={WARM_KWG:.4f}")


def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))


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


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, params):
    Re_SL = rho_l * U_sl * D / mu_l
    Re_SG = rho_g * U_sg * D / mu_g
    x = (jnp.array([jnp.log(Re_SL), jnp.log(Re_SG)]) - FEAT_MEAN) / FEAT_STD
    f_i_mult = params.f_i_mult(x)
    k_wl, k_wg = params.k_wl, params.k_wg

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


def loss_fn(params, cols):
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred, _ = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], params,
        )
        alpha_err = (alpha_pred - cols['psi'][i]) / cols['psi'][i]
        dpdz_err = (jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
        sq_h.append(alpha_err ** 2)
        sq_d.append(dpdz_err ** 2)
    loss_h = jnp.mean(jnp.stack(sq_h))
    loss_d = jnp.mean(jnp.stack(sq_d))
    return loss_h + W_DPDZ * loss_d, (loss_h, loss_d)


grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
eval_fn = eqx.filter_jit(loss_fn)


def main():
    params = HybridParams(jax.random.PRNGKey(0))

    optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(eqx.filter(params, eqx.is_array))

    best_loss, best_params, best_epoch = float('inf'), params, None
    t0 = time.time()
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grads = grad_fn(params, train_c)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(params, eqx.is_array))
        params = eqx.apply_updates(params, updates)

        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_params, best_epoch = loss_val, params, epoch

        if epoch % 150 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:5d}  k_wl={float(params.k_wl):.4f}  k_wg={float(params.k_wg):.4f}  "
                  f"loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})  "
                  f"grad_log_kwl={float(grads.log_kwl):.3e}  grad_log_kwg={float(grads.log_kwg):.3e}", flush=True)

    dt = time.time() - t0
    print(f"\nTotal wall-clock: {dt:.1f}s")
    print(f"Best epoch: {best_epoch}  loss={best_loss:.6f}  "
          f"k_wl={float(best_params.k_wl):.4f}  k_wg={float(best_params.k_wg):.4f}")

    train_loss, (train_lh, train_ld) = eval_fn(best_params, train_c)
    test_loss, (test_lh, test_ld) = eval_fn(best_params, test_c)
    print(f"\ntrain (pooled): loss={float(train_loss):.6f}  holdup={float(train_lh):.6f}  dpdz={float(train_ld):.6f}")
    print(f"test  (pooled): loss={float(test_loss):.6f}  holdup={float(test_lh):.6f}  dpdz={float(test_ld):.6f}")

    sw_train_loss, _ = eval_fn(best_params, sw_train_c)
    sw_test_loss, _ = eval_fn(best_params, sw_test_c)
    st_train_loss, _ = eval_fn(best_params, st_train_c)
    st_test_loss, _ = eval_fn(best_params, st_test_c)
    print(f"\n-- subset breakdown --")
    print(f"SW train: loss={float(sw_train_loss):.6f}   SW test: loss={float(sw_test_loss):.6f}")
    print(f"ST train: loss={float(st_train_loss):.6f}   ST test: loss={float(st_test_loss):.6f}")

    print("\n# per-condition detail (equilibrium path)")
    print(f"{'subset':>6} {'split':>5} {'U_sl':>8} {'U_sg':>8} {'f_i_mult':>9} "
          f"{'hld_err%':>9} {'dpdz_err%':>10}")
    all_rows = []
    for subset_name, (tr, te) in (('SW', (sw_train_c, sw_test_c)), ('ST', (st_train_c, st_test_c))):
        for split, cols in (('train', tr), ('test', te)):
            for i in range(len(cols['U_sl'])):
                alpha_pred, dpdz_pred, (f, kl, kg) = predict_one(
                    cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                    cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], best_params,
                )
                hld_err = float((alpha_pred - cols['psi'][i]) / cols['psi'][i] * 100)
                dpdz_err = float((jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i] * 100)
                all_rows.append((subset_name, cols['U_sg'][i], hld_err))
                print(f"{subset_name:>6} {split:>5} {cols['U_sl'][i]:8.4f} {cols['U_sg'][i]:8.3f} {float(f):9.4f} "
                      f"{hld_err:9.2f} {dpdz_err:10.2f}")

    usg_arr = np.array([r[1] for r in all_rows])
    err_arr = np.array([r[2] for r in all_rows])
    corr = np.corrcoef(usg_arr, err_arr)[0, 1]
    print(f"\ncorrelation(U_sg, holdup_err%), pooled SW+ST = {corr:.3f}")

    eqx.tree_serialise_leaves('pooled_sw_st_nn_fi_only_extended.eqx', best_params)
    print("saved best params to pooled_sw_st_nn_fi_only_extended.eqx")


if __name__ == '__main__':
    main()
