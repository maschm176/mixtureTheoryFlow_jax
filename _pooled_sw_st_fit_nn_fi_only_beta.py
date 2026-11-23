"""
Pooled SW + ST "Option B" refit WITH INCLINATION.

_pooled_sw_st_fit_nn_fi_only.py never passed beta to the equilibrium solve / pressure gradient, so its 21 ST rows
(+2 deg) were fit as horizontal. This copy threads a per-condition beta through (SW rows: 0, ST rows: +2 deg) and is
otherwise the same: same 44 train / 13 test split, same loss (relative holdup^2 + relative |dP/dz|^2, averaged over
rows), AdamW(lr=0.02, wd=1e-3), 15000 epochs, f_i_mult = softplus(NN(features)) with k_wl/k_wg flat scalars (NOT
network outputs), true jax.grad through the closed-form equilibrium path. Original script/checkpoints untouched.

Two variants (argv[1]):
  base   : features = (log Re_SL, log Re_SG)            -- the original architecture, 2->3->1 (13 weights)
  angle  : features = (log Re_SL, log Re_SG, beta)      -- adds inclination angle as a 3rd input (16 weights)
Warm start = geometric mean of the two INDIVIDUAL fits, using the new beta-aware ST fit:
  SW (3.8893, 1.5419, 0.8413) and ST (9.2809, 1.6477, 0.7736) -> f_i_mult0=6.008, k_wl0=1.594, k_wg0=0.807.
Writes pooled_beta_result_<tag>.pkl and pooled_sw_st_nn_fi_only_beta_<tag>.eqx.  Usage: python3 <this> base|angle [epochs]
"""
import math, pickle, sys, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import equinox as eqx
import optax
import numpy as np
import pandas as pd
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)
from _vuong_fit_joint import build_split

TAG = sys.argv[1] if len(sys.argv) > 1 else 'base'
N_EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 15000
USE_ANGLE = TAG == 'angle'
W_DPDZ, LR, WEIGHT_DECAY = 1.0, 0.02, 1e-3
BETA_ST = math.radians(2.0)
WARM_FI = float(np.sqrt(3.8893 * 9.2809)); WARM_KWL = float(np.sqrt(1.5419 * 1.6477)); WARM_KWG = float(np.sqrt(0.8413 * 0.7736))
ST_TEST_IDX = [805, 836, 1457, 1487, 1523, 1543]
ST_TRAIN_IDX = [797, 820, 821, 822, 826, 832, 840, 823, 1464, 1479, 1496, 1508, 1522, 1524, 1550]

def _st_cols(idx_list):
    ff = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
    s = ff[(ff['Fluids'] == 'Nitrogen - Isopar L') & (ff['Flow Pattern'] == 'ST')].loc[idx_list]
    c = {'U_sl': [float(v) for v in s['vSL[m/s]']], 'U_sg': [float(v) for v in s['vSG[m/s]']],
         'D': [float(v) for v in s['Pipe Diameter[m]']], 'rho_l': [float(v) for v in s['rhoL[kg/m3]']],
         'rho_g': [float(v) for v in s['rhoG[kg/m3]']], 'mu_l': [float(v) for v in s['viscosityL[Pa.s]']],
         'mu_g': [float(v) for v in s['viscosityG[Pa.s]']], 'psi': [float(v) for v in s['HL[-]']],
         'dpdz_measured': [float(v) for v in s['dP/dL[Pa/m]']]}
    c['beta'] = [BETA_ST] * len(idx_list)
    return c

def _sw_cols(conds):
    keys = ['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'psi', 'dpdz_measured']
    c = {k: [float(x[k]) for x in conds] for k in keys}
    c['beta'] = [0.0] * len(conds)                     # SW subset is horizontal by construction
    return c

def _concat(a, b): return {k: a[k] + b[k] for k in a}

sw_train, sw_test = build_split()
sw_train_c, sw_test_c = _sw_cols(sw_train), _sw_cols(sw_test)
st_train_c, st_test_c = _st_cols(ST_TRAIN_IDX), _st_cols(ST_TEST_IDX)
train_c, test_c = _concat(sw_train_c, st_train_c), _concat(sw_test_c, st_test_c)
print(f"[{TAG}] pooled train {len(train_c['U_sl'])} ({len(sw_train_c['U_sl'])} SW + {len(st_train_c['U_sl'])} ST), "
      f"test {len(test_c['U_sl'])} ({len(sw_test_c['U_sl'])} SW + {len(st_test_c['U_sl'])} ST)", flush=True)

def _feats(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, beta):
    f = [jnp.log(rho_l * U_sl * D / mu_l), jnp.log(rho_g * U_sg * D / mu_g)]
    if USE_ANGLE: f.append(beta)
    return jnp.array(f)

_F = np.array([[float(v) for v in _feats(train_c['U_sl'][i], train_c['U_sg'][i], train_c['D'][i], train_c['rho_l'][i],
                                         train_c['rho_g'][i], train_c['mu_l'][i], train_c['mu_g'][i], train_c['beta'][i])]
               for i in range(len(train_c['U_sl']))])
FEAT_MEAN, FEAT_STD = jnp.array(_F.mean(0)), jnp.array(_F.std(0))     # from TRAIN only
N_IN = _F.shape[1]
print(f"[{TAG}] warm start f_i_mult={WARM_FI:.4f} k_wl={WARM_KWL:.4f} k_wg={WARM_KWG:.4f}; {N_IN} NN inputs", flush=True)

def _inv_softplus(y): return jnp.log(jnp.expm1(y))

class HybridParams(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    log_kwl: jax.Array
    log_kwg: jax.Array
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(N_IN, 3, key=k1)
        l2 = eqx.nn.Linear(3, 1, key=k2)
        self.layer2 = eqx.tree_at(lambda l: (l.weight, l.bias), l2,
                                  (jnp.zeros_like(l2.weight), _inv_softplus(jnp.array([WARM_FI]))))
        self.log_kwl = jnp.log(jnp.array(WARM_KWL)); self.log_kwg = jnp.log(jnp.array(WARM_KWG))
    def f_i_mult(self, x): return jax.nn.softplus(self.layer2(jnp.tanh(self.layer1(x))))[0]
    @property
    def k_wl(self): return jnp.exp(self.log_kwl)
    @property
    def k_wg(self): return jnp.exp(self.log_kwg)

def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, beta, params):
    x = (_feats(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, beta) - FEAT_MEAN) / FEAT_STD
    f, kl, kg = params.f_i_mult(x), params.k_wl, params.k_wg
    ht = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f, kl, kg, beta=beta)
    A, A_l, A_g, *_ = td_geometry(ht, D)
    a_l, a_g = A_l / A, A_g / A
    Delta = rho_l * U_sl / jnp.maximum(a_l, 1e-9) - rho_g * U_sg / jnp.maximum(a_g, 1e-9)
    d, _ = predicted_pressure_gradient(a_l, Delta, U_sl + U_sg, D, rho_l, rho_g, mu_l, mu_g, f, kl, kg, beta=beta)
    return a_l, d, (f, kl, kg)

def loss_fn(params, cols):
    sh, sd = [], []
    for i in range(len(cols['U_sl'])):
        a, d, _ = predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i], cols['rho_g'][i],
                              cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i], params)
        sh.append(((a - cols['psi'][i]) / cols['psi'][i]) ** 2)
        sd.append(((jnp.abs(d) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]) ** 2)
    lh, ld = jnp.mean(jnp.stack(sh)), jnp.mean(jnp.stack(sd))
    return lh + W_DPDZ * ld, (lh, ld)

grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
eval_fn = eqx.filter_jit(loss_fn)

def main():
    params = HybridParams(jax.random.PRNGKey(0))
    opt = optax.adamw(LR, weight_decay=WEIGHT_DECAY); os_ = opt.init(eqx.filter(params, eqx.is_array))
    best = (float('inf'), params, None); t0 = time.time(); hist = []
    for ep in range(N_EPOCHS):
        (lv, (lh, ld)), g = grad_fn(params, train_c)
        up, os_ = opt.update(g, os_, eqx.filter(params, eqx.is_array)); params = eqx.apply_updates(params, up)
        lv = float(lv)
        if lv < best[0]: best = (lv, params, ep)
        if ep % 500 == 0 or ep == N_EPOCHS - 1:
            hist.append((ep, lv, float(lh), float(ld)))
            print(f"[{TAG}] epoch {ep:5d} loss={lv:.6f} (hld={float(lh):.6f} dpdz={float(ld):.6f}) "
                  f"k_wl={float(params.k_wl):.4f} k_wg={float(params.k_wg):.4f}  {time.time() - t0:.0f}s", flush=True)
    bp = best[1]
    print(f"\n[{TAG}] best epoch {best[2]}  k_wl={float(bp.k_wl):.4f} k_wg={float(bp.k_wg):.4f}", flush=True)
    res = dict(tag=TAG, best_epoch=best[2], k_wl=float(bp.k_wl), k_wg=float(bp.k_wg), history=hist)
    for nm, cols in (('train', train_c), ('test', test_c), ('SW train', sw_train_c), ('SW test', sw_test_c),
                     ('ST train', st_train_c), ('ST test', st_test_c)):
        l, (lh, ld) = eval_fn(bp, cols); res[nm] = (float(l), float(lh), float(ld))
        print(f"[{TAG}] {nm:>8}: loss={float(l):.6f} (hld={float(lh):.6f} dpdz={float(ld):.6f})", flush=True)
    rows = []
    for subset, (tr, te) in (('SW', (sw_train_c, sw_test_c)), ('ST', (st_train_c, st_test_c))):
        for split, cols in (('train', tr), ('test', te)):
            for i in range(len(cols['U_sl'])):
                a, d, (f, kl, kg) = predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                                                cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i], bp)
                rows.append(dict(subset=subset, split=split, U_sl=cols['U_sl'][i], U_sg=cols['U_sg'][i], beta=cols['beta'][i],
                                 f_i_mult=float(f), HL=cols['psi'][i], hl_pred=float(a),
                                 dpdz=cols['dpdz_measured'][i], dpdz_pred=abs(float(d))))
    res['rows'] = rows
    for sub in ('SW', 'ST'):
        r = [x for x in rows if x['subset'] == sub]
        e = np.array([(x['hl_pred'] - x['HL']) / x['HL'] * 100 for x in r]); ed = np.array([(x['dpdz_pred'] - x['dpdz']) / x['dpdz'] * 100 for x in r])
        fv = np.array([x['f_i_mult'] for x in r])
        print(f"[{TAG}] {sub} (n={len(r)}): holdup mean|err|={np.abs(e).mean():.1f}% bias={e.mean():+.1f}% range [{e.min():+.0f},{e.max():+.0f}]%  "
              f"dpdz mean|err|={np.abs(ed).mean():.1f}%  f_i_mult range {fv.min():.2f}-{fv.max():.2f} (median {np.median(fv):.2f})", flush=True)
    pickle.dump(res, open(f"pooled_beta_result_{TAG}.pkl", "wb"))
    eqx.tree_serialise_leaves(f"pooled_sw_st_nn_fi_only_beta_{TAG}.eqx", bp)
    print(f"[{TAG}] saved pooled_beta_result_{TAG}.pkl and pooled_sw_st_nn_fi_only_beta_{TAG}.eqx", flush=True)

if __name__ == '__main__':
    main()
