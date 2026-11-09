"""
First NN closure for the water-gas solver: a tiny network replacing the
flat scalars (f_i_mult, k_wl, k_wg) with functions of (Re_SL, Re_SG),
trained via true jax.grad through the closed-form equilibrium path (same
machinery as _vuong_fit_joint_equilibrium.py -- no scan, no finite
differences).

MOTIVATION: the fully-converged flat-3-scalar fit (f_i_mult=3.8893,
k_wl=1.5419, k_wg=0.8413, see CLAUDE.md "Nondimensionalization +
equilibrium-path true-gradient training...") still shows a real,
non-artifact residual: holdup error correlates with U_sg at
correlation=0.610 (mean -7.6% below U_sg=2.5 m/s, +4.7% above) -- this
survived full convergence, so it's a genuine functional-form limit of
"one flat number per phase", not an optimization shortfall. This script
tests whether letting (f_i_mult, k_wl, k_wg) vary with Re_SL/Re_SG closes
that gap.

DESIGN (structure-preserving, same philosophy as the old oil-water
solver's DragClosureNetwork -- see CLAUDE.md "Why Learn C_D First"):
  - Inputs: log(Re_SL), log(Re_SG), standardized by the TRAIN set's own
    mean/std. Log scale because Re spans orders of magnitude here, and
    because a power-law closure (Carraretto's own f_i=2.25*Re_SG^-0.53)
    becomes LINEAR in log-space -- easy for a tiny network to represent.
  - Architecture: 2 -> 4 (tanh) -> 3 (softplus), 27 weights total.
    Deliberately tiny -- only 29 training conditions, and this is
    explicitly a proof-of-concept, not a production closure.
  - Outputs: (f_i_mult, k_wl, k_wg) -- directly replaces the three flat
    scalars, multiplying into the EXACT SAME physics formulas unchanged
    (interfacial/wall shear closures in mixtureTheoryFlow_jax_solver_
    water_gas.py are not touched by this script at all).
  - Warm-start init: hidden->output weights zeroed, output bias set to
    inverse_softplus(WARM_START) -- the network starts EXACTLY at the
    already-converged flat-scalar fit regardless of input. optax.adamw's
    weight decay then acts as a principled regularizer, shrinking the
    network back toward that known-good constant unless the data
    genuinely rewards moving away from it.
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
N_EPOCHS = 5000
WARM_START = jnp.array([3.8893, 1.5419, 0.8413])  # fully-converged flat-scalar fit

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
FEAT_MEAN = _log_re.mean(axis=0)
FEAT_STD = _log_re.std(axis=0)
print(f"feature standardization (train set): mean={FEAT_MEAN}  std={FEAT_STD}")


def features(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g):
    Re_SL = rho_l * U_sl * D / mu_l
    Re_SG = rho_g * U_sg * D / mu_g
    x = jnp.array([jnp.log(Re_SL), jnp.log(Re_SG)])
    return (x - FEAT_MEAN) / FEAT_STD


def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))


class ClosureNetwork(eqx.Module):
    """2 (log Re_SL, log Re_SG, standardized) -> 4 (tanh) -> 3 (softplus:
    f_i_mult, k_wl, k_wg). See module docstring for the warm-start init."""
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(2, 4, key=k1)
        self.layer2 = eqx.nn.Linear(4, 3, key=k2)
        self.layer2 = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            self.layer2,
            (jnp.zeros_like(self.layer2.weight), _inv_softplus(WARM_START)),
        )

    def __call__(self, x):
        h = jnp.tanh(self.layer1(x))
        return jax.nn.softplus(self.layer2(h))


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, net):
    x = features(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g)
    f_i_mult, k_wl, k_wg = net(x)
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


def loss_fn(net, cols):
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred, _ = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], net,
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
    net = ClosureNetwork(jax.random.PRNGKey(0))

    x0 = features(train_c['U_sl'][0], train_c['U_sg'][0], train_c['D'][0],
                  train_c['rho_l'][0], train_c['rho_g'][0], train_c['mu_l'][0], train_c['mu_g'][0])
    print(f"init check: network(cond0) = {net(x0)}  (should ~= {WARM_START})")

    optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    best_loss, best_net, best_epoch = float('inf'), net, None

    t_start = time.time()
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grads = grad_fn(net, train_c)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)

        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_net, best_epoch = loss_val, net, epoch

        if epoch % 250 == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:4d}  loss={loss_val:.6f} (holdup={float(lh):.6f} dpdz={float(ld):.6f})",
                  flush=True)

    dt = time.time() - t_start
    print(f"\nTotal wall-clock: {dt:.1f}s  ({dt / N_EPOCHS * 1000:.2f}ms/epoch after compile)")
    print(f"Best epoch: {best_epoch}  loss={best_loss:.6f}")

    train_loss, (train_lh, train_ld) = eval_fn(best_net, train_c)
    test_loss, (test_lh, test_ld) = eval_fn(best_net, test_c)
    print(f"\ntrain: loss={float(train_loss):.6f}  holdup={float(train_lh):.6f}  dpdz={float(train_ld):.6f}")
    print(f"test:  loss={float(test_loss):.6f}  holdup={float(test_lh):.6f}  dpdz={float(test_ld):.6f}")
    print("\nReference (flat 3-scalar fit, fully converged): "
          "train loss=0.033699 (holdup=0.010319 dpdz=0.023380); "
          "test loss=0.042799 (holdup=0.022535 dpdz=0.020265)")

    print("\n# per-condition detail")
    print(f"{'split':>5} {'U_sl':>8} {'U_sg':>8} {'f_i_mult':>9} {'k_wl':>7} {'k_wg':>7} "
          f"{'hld_err%':>9} {'dpdz_err%':>10}")
    all_rows = []
    for split, cols in (('train', train_c), ('test', test_c)):
        for i in range(len(cols['U_sl'])):
            alpha_pred, dpdz_pred, (f, kl, kg) = predict_one(
                cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], best_net,
            )
            hld_err = float((alpha_pred - cols['psi'][i]) / cols['psi'][i] * 100)
            dpdz_err = float((jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i] * 100)
            all_rows.append((split, cols['U_sg'][i], hld_err))
            print(f"{split:>5} {cols['U_sl'][i]:8.4f} {cols['U_sg'][i]:8.3f} {float(f):9.4f} "
                  f"{float(kl):7.4f} {float(kg):7.4f} {hld_err:9.2f} {dpdz_err:10.2f}")

    usg_arr = np.array([r[1] for r in all_rows])
    err_arr = np.array([r[2] for r in all_rows])
    corr = np.corrcoef(usg_arr, err_arr)[0, 1]
    print(f"\ncorrelation(U_sg, holdup_err%) = {corr:.3f}  "
          f"(flat 3-scalar fit's was 0.610 -- did the NN reduce this?)")

    eqx.tree_serialise_leaves('vuong_nn_closure_best.eqx', best_net)
    print("saved best network to vuong_nn_closure_best.eqx")


if __name__ == '__main__':
    main()
