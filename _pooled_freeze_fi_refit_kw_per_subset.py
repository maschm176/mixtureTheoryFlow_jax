"""
Second check requested alongside extending the pooled fit: freeze the
pooled fit's NN f_i_mult(log Re_SL, log Re_SG) surface exactly as fitted
(stop_gradient on its weights), then separately refit ONLY (k_wl, k_wg)
-- via true jax.grad through the same equilibrium path, no scan -- on
SW-only conditions and on ST-only conditions.

Purpose: distinguish two explanations for why the pooled fit's shared
k_wg drifted so far from where the two independent flat fits agreed
(SW-only: k_wg=0.8413, ST-only: k_wg=0.8510 -- nearly identical):
  (a) k_wg genuinely needs to differ between regimes once f_i_mult's
      shape is fixed to the pooled/shared NN surface (real
      condition-dependence) -- if so, SW-only and ST-only refits here
      should land on visibly different k_wl/k_wg.
  (b) the shared value is being dragged around mostly because the single
      shared f_i_mult NN surface doesn't fit either regime's individual
      preferences well, and k_wg was just the remaining free slack
      variable in that fit -- if so, even with f_i_mult FROZEN exactly
      as the pooled fit left it, refitting k_wl/k_wg on each subset
      alone might still land close to each other and close to the
      pooled fit's own value, since the frozen f_i_mult surface was
      already jointly optimized across both subsets.
Uses the EXTENDED (15000-epoch) checkpoint, not the original 3000-epoch
one, so the frozen f_i_mult surface reflects the fully-converged pooled
fit, not a mid-training snapshot.
"""
import sys
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import optax
import numpy as np
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _vuong_fit_joint import build_split
from _pooled_sw_st_fit_nn_fi_only import (
    HybridParams, FEAT_MEAN, FEAT_STD, _st_cols, _sw_cols, _concat,
    ST_TRAIN_IDX, ST_TEST_IDX,
)

CHECKPOINT = sys.argv[1] if len(sys.argv) > 1 else 'pooled_sw_st_nn_fi_only_extended.eqx'
LR = 0.06
N_EPOCHS = 2000

template = HybridParams(jax.random.PRNGKey(0))
loaded = eqx.tree_deserialise_leaves(CHECKPOINT, template)
frozen_net_params = (loaded.layer1, loaded.layer2)
print(f"loaded frozen f_i_mult NN from {CHECKPOINT}")
print(f"(pooled fit's own k_wl={float(loaded.k_wl):.4f}  k_wg={float(loaded.k_wg):.4f}, for reference)\n")


def frozen_f_i_mult(x):
    layer1, layer2 = frozen_net_params
    h = jnp.tanh(layer1(x))
    return jax.nn.softplus(layer2(h))[0]


def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, log_kwl, log_kwg):
    Re_SL = rho_l * U_sl * D / mu_l
    Re_SG = rho_g * U_sg * D / mu_g
    x = (jnp.array([jnp.log(Re_SL), jnp.log(Re_SG)]) - FEAT_MEAN) / FEAT_STD
    f_i_mult = jax.lax.stop_gradient(frozen_f_i_mult(x))
    k_wl, k_wg = jnp.exp(log_kwl), jnp.exp(log_kwg)

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
    return alpha_l_eq, dPdz_l_eq, f_i_mult


def loss_fn(z, cols):
    log_kwl, log_kwg = z[0], z[1]
    sq_h, sq_d = [], []
    n = len(cols['U_sl'])
    for i in range(n):
        alpha_pred, dpdz_pred, _ = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], log_kwl, log_kwg,
        )
        alpha_err = (alpha_pred - cols['psi'][i]) / cols['psi'][i]
        dpdz_err = (jnp.abs(dpdz_pred) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]
        sq_h.append(alpha_err ** 2)
        sq_d.append(dpdz_err ** 2)
    loss_h = jnp.mean(jnp.stack(sq_h))
    loss_d = jnp.mean(jnp.stack(sq_d))
    return loss_h + loss_d, (loss_h, loss_d)


grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))


def fit_subset(name, cols, kwl0=1.5, kwg0=0.85):
    z = jnp.array([jnp.log(kwl0), jnp.log(kwg0)])
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(z)
    best_loss, best_z, best_epoch = float('inf'), z, None
    for epoch in range(N_EPOCHS):
        (loss_val, (lh, ld)), grad = grad_fn(z, cols)
        updates, opt_state = optimizer.update(grad, opt_state)
        z = z + updates
        loss_val = float(loss_val)
        if loss_val < best_loss:
            best_loss, best_z, best_epoch = loss_val, z, epoch
    k_wl, k_wg = float(jnp.exp(best_z[0])), float(jnp.exp(best_z[1]))
    print(f"[{name}]  n={len(cols['U_sl']):3d}  best_epoch={best_epoch:5d}  loss={best_loss:.6f}  "
          f"k_wl={k_wl:.4f}  k_wg={k_wg:.4f}")
    return k_wl, k_wg


sw_train, sw_test = build_split()
sw_all = _concat(_sw_cols(sw_train), _sw_cols(sw_test))
st_all = _concat(_st_cols(ST_TRAIN_IDX), _st_cols(ST_TEST_IDX))

print("Refitting (k_wl, k_wg) per subset, f_i_mult NN frozen at the pooled fit's weights:\n")
sw_kwl, sw_kwg = fit_subset("SW-only", sw_all)
st_kwl, st_kwg = fit_subset("ST-only", st_all)

print(f"\nFor reference -- independent flat fits (f_i_mult NOT shared, each a free scalar too):")
print(f"  SW-only flat fit: f_i_mult=3.8893  k_wl=1.5419  k_wg=0.8413")
print(f"  ST-only flat fit: f_i_mult=10.6923 k_wl=2.6477  k_wg=0.8510")
print(f"\nPooled fit's own shared (k_wl, k_wg): k_wl={float(loaded.k_wl):.4f}  k_wg={float(loaded.k_wg):.4f}")
print(f"\nWith f_i_mult FROZEN to the pooled NN surface:")
print(f"  SW-only refit: k_wl={sw_kwl:.4f}  k_wg={sw_kwg:.4f}")
print(f"  ST-only refit: k_wl={st_kwl:.4f}  k_wg={st_kwg:.4f}")
