"""
Option B hybrid (per-session discussion, 2026-11-10): give f_i_mult a small
NN's flexibility while k_wl/k_wg stay ordinary flat scalars -- all three
trained JOINTLY, exactly mirroring _vuong_fit_fi_powerlaw.py's structure
(same warm start, same joint optimization), just swapping the power law's
2-parameter functional form for a 13-weight network on f_i_mult alone.

WHY THIS IS DIFFERENT FROM THE EARLIER NN CLOSURES (_vuong_fit_nn_closure.py,
_vuong_fit_nn_cv.py): those gave ONE shared network authority over all
three outputs (f_i_mult, k_wl, k_wg) -- which is exactly what let it exploit
the f_i_mult/k_wall ratio degeneracy (k_wg collapsing toward 0 per-condition
while f_i_mult/k_wl compensate; see CLAUDE.md "NN closure overfits on 29
points..."). Here k_wl/k_wg are plain scalar leaves of the parameter pytree
with NO dependence on (Re_SL, Re_SG) or condition identity -- structurally
incapable of varying per-condition, so that specific overfitting mechanism
is unavailable regardless of how much freedom the network has over
f_i_mult alone. Directly answers: does f_i_mult benefit from NN-level
flexibility (vs. the power law's fixed 2-parameter form) without
reintroducing the degeneracy exploit?

Architecture: 2 (log Re_SL, log Re_SG, standardized by the TRAIN set's own
mean/std) -> 3 (tanh) -> 1 (softplus) = 13 weights for f_i_mult, plus 2 free
scalars (log k_wl, log k_wg) = 15 total free parameters -- comparable in
count to the 15-weight leave-one-cluster-out CV network, but concentrated
entirely on f_i_mult instead of spread across all three outputs.

Warm-started to the converged flat fit (f_i_mult=3.8893 regardless of
input -- hidden->output weights zeroed, output bias = inverse_softplus of
the warm-start value; k_wl=1.5419, k_wg=0.8413) via the same trick every
NN closure in this family uses. True jax.grad through the closed-form
Taitel-Dukler equilibrium path -- no scan, no finite differences, same
machinery as _vuong_fit_joint_equilibrium.py / _vuong_fit_fi_powerlaw.py.
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
from _vuong_fit_joint import build_split

W_DPDZ = 1.0
LR = 0.02
WEIGHT_DECAY = 1e-3
N_EPOCHS = 3000
WARM_FI = 3.8893
WARM_KWL = 1.5419
WARM_KWG = 0.8413

train, test = build_split()


def _cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'psi', 'dpdz_measured']
    return {k: [float(c[k]) for c in conds] for k in keys}


train_c = _cols(train)
test_c = _cols(test)


def _re_pair(cols, i):
    Re_SL = cols['rho_l'][i] * cols['U_sl'][i] * cols['D'][i] / cols['mu_l'][i]
    Re_SG = cols['rho_g'][i] * cols['U_sg'][i] * cols['D'][i] / cols['mu_g'][i]
    return Re_SL, Re_SG


_log_re = np.array([[np.log(r) for r in _re_pair(train_c, i)] for i in range(len(train_c['U_sl']))])
FEAT_MEAN = jnp.array(_log_re.mean(axis=0))
FEAT_STD = jnp.array(_log_re.std(axis=0))
print(f"feature standardization (train set): mean={FEAT_MEAN}  std={FEAT_STD}")


def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))


class HybridParams(eqx.Module):
    """One pytree holding both the tiny f_i_mult network and the two flat
    wall-friction scalars, so a single optax optimizer updates all of them
    together -- but k_wl/k_wg are plain array leaves with no functional
    dependence on the input features, unlike f_i_mult."""
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

    x0 = (jnp.array([jnp.log(train_c['rho_l'][0] * train_c['U_sl'][0] * train_c['D'][0] / train_c['mu_l'][0]),
                      jnp.log(train_c['rho_g'][0] * train_c['U_sg'][0] * train_c['D'][0] / train_c['mu_g'][0])])
          - FEAT_MEAN) / FEAT_STD
    print(f"init check: f_i_mult(cond0)={float(params.f_i_mult(x0)):.4f} (should ~= {WARM_FI}), "
          f"k_wl={float(params.k_wl):.4f}, k_wg={float(params.k_wg):.4f}")

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
            print(f"epoch {epoch:4d}  k_wl={float(params.k_wl):.4f}  k_wg={float(params.k_wg):.4f}  "
                  f"loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})", flush=True)

    dt = time.time() - t0
    print(f"\nTotal wall-clock: {dt:.1f}s")
    print(f"Best epoch: {best_epoch}  loss={best_loss:.6f}  "
          f"k_wl={float(best_params.k_wl):.4f}  k_wg={float(best_params.k_wg):.4f}")

    train_loss, (train_lh, train_ld) = eval_fn(best_params, train_c)
    test_loss, (test_lh, test_ld) = eval_fn(best_params, test_c)
    print(f"\ntrain: loss={float(train_loss):.6f}  holdup={float(train_lh):.6f}  dpdz={float(train_ld):.6f}")
    print(f"test:  loss={float(test_loss):.6f}  holdup={float(test_lh):.6f}  dpdz={float(test_ld):.6f}")
    print("\nReference (flat 3-scalar, fully converged): "
          "train loss=0.033699 (hld=0.010319 dpdz=0.023380); "
          "test loss=0.042799 (hld=0.022535 dpdz=0.020265)")
    print("Reference (power-law f_i_mult(Re_SG), 4 params): "
          "train loss=0.027961 (hld=0.010173 dpdz=0.017788); "
          "test loss=0.026653 (hld=0.013547 dpdz=0.013106)")
    print("Reference (27-weight NN, all 3 outputs shared, overfit): "
          "train loss=0.005273 (hld=0.002973 dpdz=0.002300); "
          "test loss=0.104200 (hld=0.005847 dpdz=0.098352)")

    print("\n# per-condition detail (equilibrium path)")
    print(f"{'split':>5} {'U_sl':>8} {'U_sg':>8} {'f_i_mult':>9} "
          f"{'hld_err%':>9} {'dpdz_err%':>10}")
    all_rows = []
    for split, cols in (('train', train_c), ('test', test_c)):
        for i in range(len(cols['U_sl'])):
            alpha_pred, dpdz_pred, (f, kl, kg) = predict_one(
                cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], best_params,
            )
            hld_err = float((alpha_pred - cols['psi'][i]) / cols['psi'][i] * 100)
            dpdz_err = float((jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i] * 100)
            all_rows.append((split, cols['U_sg'][i], hld_err))
            print(f"{split:>5} {cols['U_sl'][i]:8.4f} {cols['U_sg'][i]:8.3f} {float(f):9.4f} "
                  f"{hld_err:9.2f} {dpdz_err:10.2f}")

    usg_arr = np.array([r[1] for r in all_rows])
    err_arr = np.array([r[2] for r in all_rows])
    corr = np.corrcoef(usg_arr, err_arr)[0, 1]
    print(f"\ncorrelation(U_sg, holdup_err%) = {corr:.3f}  "
          f"(flat: 0.610, power-law: 0.220, full NN: 0.283)")

    eqx.tree_serialise_leaves('vuong_nn_fi_only_best.eqx', best_params)
    print("saved best params to vuong_nn_fi_only_best.eqx")


if __name__ == '__main__':
    main()
