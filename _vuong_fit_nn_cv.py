"""
Leave-one-cluster-out cross-validation for a small NN closure --
addresses the two things explicitly deferred when the first (27-weight)
NN overfit badly on the held-out TEST set (see CLAUDE.md "NN closure
overfits on 29 points..."): a smaller/more-regularized architecture, and
a genuine validation procedure that never touches the TEST set.

Architecture: 2 (log Re_SL, log Re_SG, standardized) -> 2 (tanh) -> 3
(softplus) -- 15 weights, roughly half the first attempt's 27.
Regularization: weight_decay=1e-2, 10x the original (overfit) run's
1e-3. Same warm-start trick (hidden->output weights zeroed, output bias
= inverse_softplus(the converged flat-scalar fit)) and the same true-
jax.grad-through-the-equilibrium-path machinery as every other fit in
this family (no scan, no finite differences).

CV scheme: leave-one-cluster-out across the SAME 7 natural U_sg clusters
_vuong_fit_joint.py's build_split() uses (identical clustering logic:
sort by U_sg, start a new cluster whenever the gap exceeds 0.1 m/s),
restricted to the 29 TRAIN conditions only -- the 7-condition TEST set
is never touched by this script. For each fold: fit on the other 6
clusters, evaluate (no gradient) on the held-out cluster. Cluster sizes
are uneven (1 to 9 training points per cluster, since this is the real
experimental design, not rebalanced) -- the n=1 cluster's fold gives a
much noisier per-fold signal than the n=9 cluster's, which should be
kept in mind when reading the 7-fold average.

Usage: python3 _vuong_fit_nn_cv.py <fold_index 0-6>
Each fold is its own process -- run all 7 as separate background
processes so folds can run concurrently rather than accumulating XLA
compiles in one long-lived process (same discipline as this project's
other subprocess-per-batch designs, see CLAUDE.md "Execution
architecture").
"""
import sys
import time
import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import pandas as pd

from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)

W_DPDZ = 1.0
LR = 0.02
WEIGHT_DECAY = 1e-2
N_EPOCHS = 2000
WARM_START = jnp.array([3.8893, 1.5419, 0.8413])


def build_cv_folds():
    df = pd.read_csv('vuong2016_sw_subset.csv').reset_index(drop=True)
    df = df.sort_values('vSG[m/s]').reset_index(drop=True)
    gaps = df['vSG[m/s]'].diff().fillna(999)
    df['cluster'] = (gaps > 0.1).cumsum()

    test_idx = []
    for i, (cid, grp) in enumerate(df.groupby('cluster')):
        grp_sorted = grp.sort_values('vSL[m/s]')
        rank_to_hold = i % len(grp_sorted)
        test_idx.append(grp_sorted.index[rank_to_hold])

    df['split'] = 'train'
    df.loc[test_idx, 'split'] = 'test'

    renamed = df.rename(columns={
        'vSL[m/s]': 'U_sl', 'vSG[m/s]': 'U_sg', 'rhoL[kg/m3]': 'rho_l', 'rhoG[kg/m3]': 'rho_g',
        'viscosityL[Pa.s]': 'mu_l', 'viscosityG[Pa.s]': 'mu_g', 'Pipe Diameter[m]': 'D',
        'Pipe Length[m]': 'L', 'HL[-]': 'psi', 'dP/dL[Pa/m]': 'dpdz_measured',
    })
    cols = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'L', 'psi', 'dpdz_measured', 'cluster']
    train_df = renamed.loc[renamed.split == 'train', cols]

    folds = []
    for cid in sorted(train_df['cluster'].unique()):
        val = train_df[train_df.cluster == cid].drop(columns='cluster').to_dict('records')
        fit = train_df[train_df.cluster != cid].drop(columns='cluster').to_dict('records')
        folds.append((fit, val))
    assert len(folds) == 7
    return folds


def _cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'psi', 'dpdz_measured']
    return {k: [float(c[k]) for c in conds] for k in keys}


def _re_pair(cols, i):
    Re_SL = cols['rho_l'][i] * cols['U_sl'][i] * cols['D'][i] / cols['mu_l'][i]
    Re_SG = cols['rho_g'][i] * cols['U_sg'][i] * cols['D'][i] / cols['mu_g'][i]
    return Re_SL, Re_SG


def _inv_softplus(y):
    return jnp.log(jnp.expm1(y))


class ClosureNetworkSmall(eqx.Module):
    """2 (log Re_SL, log Re_SG, standardized) -> 2 (tanh) -> 3
    (softplus): 15 weights total, ~half the first attempt's 27."""
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(2, 2, key=k1)
        self.layer2 = eqx.nn.Linear(2, 3, key=k2)
        self.layer2 = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            self.layer2,
            (jnp.zeros_like(self.layer2.weight), _inv_softplus(WARM_START)),
        )

    def __call__(self, x):
        h = jnp.tanh(self.layer1(x))
        return jax.nn.softplus(self.layer2(h))


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, net, feat_mean, feat_std):
    Re_SL = rho_l * U_sl * D / mu_l
    Re_SG = rho_g * U_sg * D / mu_g
    x = (jnp.array([jnp.log(Re_SL), jnp.log(Re_SG)]) - feat_mean) / feat_std
    f_i_mult, k_wl, k_wg = net(x)
    Q = U_sl + U_sg
    ht_eq = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg)
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


def loss_fn(net, cols, feat_mean, feat_std):
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], net, feat_mean, feat_std,
        )
        alpha_err = (alpha_pred - cols['psi'][i]) / cols['psi'][i]
        dpdz_err = (jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
        sq_h.append(alpha_err ** 2)
        sq_d.append(dpdz_err ** 2)
    loss_h = jnp.mean(jnp.stack(sq_h))
    loss_d = jnp.mean(jnp.stack(sq_d))
    return loss_h + W_DPDZ * loss_d, (loss_h, loss_d)


def main():
    fold_idx = int(sys.argv[1])
    folds = build_cv_folds()
    fit_conds, val_conds = folds[fold_idx]
    fit_c = _cols(fit_conds)
    val_c = _cols(val_conds)
    print(f"fold {fold_idx}: fit n={len(fit_conds)}  val n={len(val_conds)}  "
          f"val U_sg range=[{min(val_c['U_sg']):.3f}, {max(val_c['U_sg']):.3f}]", flush=True)

    _log_re = np.array([[np.log(r) for r in _re_pair(fit_c, i)] for i in range(len(fit_c['U_sl']))])
    feat_mean = jnp.array(_log_re.mean(axis=0))
    feat_std = jnp.array(_log_re.std(axis=0))

    net = ClosureNetworkSmall(jax.random.PRNGKey(0))
    grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
    eval_fn = eqx.filter_jit(loss_fn)

    optimizer = optax.adamw(LR, weight_decay=WEIGHT_DECAY)
    opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

    best_val_loss, best_net, best_epoch = float('inf'), net, None
    t_start = time.time()
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grads = grad_fn(net, fit_c, feat_mean, feat_std)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(net, eqx.is_array))
        net = eqx.apply_updates(net, updates)

        if epoch % 100 == 0 or epoch == N_EPOCHS - 1:
            val_loss, (val_lh, val_ld) = eval_fn(net, val_c, feat_mean, feat_std)
            val_loss = float(val_loss)
            if val_loss < best_val_loss:
                best_val_loss, best_net, best_epoch = val_loss, net, epoch
            print(f"  fold {fold_idx} epoch {epoch:4d}  fit_loss={float(loss_val):.6f}  "
                  f"val_loss={val_loss:.6f} (hld={float(val_lh):.6f} dpdz={float(val_ld):.6f})", flush=True)

    dt = time.time() - t_start
    print(f"fold {fold_idx} done in {dt:.1f}s. best_epoch={best_epoch}  best_val_loss={best_val_loss:.6f}", flush=True)
    eqx.tree_serialise_leaves(f'vuong_nn_cv_fold{fold_idx}.eqx', best_net)


if __name__ == '__main__':
    main()
