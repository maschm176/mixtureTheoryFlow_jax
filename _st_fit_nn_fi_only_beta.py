"""
ST-ONLY "Option B" NN fit -- the ST counterpart of the SW-only Option B (_vuong_fit_nn_fi_only.py):
  f_i_mult = softplus(NN(log Re_SL, log Re_SG))  (2 -> 3 tanh -> 1, 13 weights), k_wl/k_wg flat scalars (NOT NN outputs),
trained jointly (15 parameters) with true jax.grad through the closed-form equilibrium path, WITH beta=+2 deg threaded
through (the ST rows are inclined). Everything else mirrors the SW-only fit: AdamW lr=0.02, wd=1e-3, 3000 epochs,
best-epoch tracking, loss = mean(relative holdup err^2) + mean(relative |dP/dz| err^2), feature standardization from the
TRAIN rows only, warm start at the flat fit (here the beta-aware ST flat fit: 9.2809 / 1.6477 / 0.7736; SW used its own flat
3.8893 / 1.5419 / 0.8413). Same 15 train / 6 test ST rows as the flat ST fit.

With only 15 train rows (30 targets) for 15 parameters, ONE random init could mislead, so this runs several seeds and
reports all of them. Also re-evaluates the saved SW-only NN (vuong_nn_fi_only_best.eqx) with its own loss for a like-for-like
comparison. Usage: python3 _st_fit_nn_fi_only_beta.py   -> st_nn_fi_only_beta_result.pkl, st_nn_fi_only_beta_seed0.eqx
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

BETA = math.radians(2.0)
W_DPDZ, LR, WEIGHT_DECAY = 1.0, 0.02, 1e-3
EPOCH_RUNS = [(s, 3000) for s in range(5)] + [(0, 15000)]       # (seed, epochs); the 15000 run checks convergence
WARM_FI, WARM_KWL, WARM_KWG = 9.2809, 1.6477, 0.7736
TEST_IDX = [805, 836, 1457, 1487, 1523, 1543]
TRAIN_IDX = [797, 820, 821, 822, 826, 832, 840, 823, 1464, 1479, 1496, 1508, 1522, 1524, 1550]

ff = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ff[(ff['Fluids'] == 'Nitrogen - Isopar L') & (ff['Flow Pattern'] == 'ST')]

def _cols(idx):
    s = st.loc[idx]
    return {'U_sl': [float(v) for v in s['vSL[m/s]']], 'U_sg': [float(v) for v in s['vSG[m/s]']],
            'D': [float(v) for v in s['Pipe Diameter[m]']], 'rho_l': [float(v) for v in s['rhoL[kg/m3]']],
            'rho_g': [float(v) for v in s['rhoG[kg/m3]']], 'mu_l': [float(v) for v in s['viscosityL[Pa.s]']],
            'mu_g': [float(v) for v in s['viscosityG[Pa.s]']], 'psi': [float(v) for v in s['HL[-]']],
            'dpdz_measured': [float(v) for v in s['dP/dL[Pa/m]']]}

train_c, test_c = _cols(TRAIN_IDX), _cols(TEST_IDX)
_F = np.array([[math.log(train_c['rho_l'][i] * train_c['U_sl'][i] * train_c['D'][i] / train_c['mu_l'][i]),
                math.log(train_c['rho_g'][i] * train_c['U_sg'][i] * train_c['D'][i] / train_c['mu_g'][i])]
               for i in range(len(TRAIN_IDX))])
FEAT_MEAN, FEAT_STD = jnp.array(_F.mean(0)), jnp.array(_F.std(0))        # TRAIN only
print(f"ST-only NN fit: {len(TRAIN_IDX)} train / {len(TEST_IDX)} test rows; feature mean={FEAT_MEAN} std={FEAT_STD}", flush=True)

def _inv_softplus(y): return jnp.log(jnp.expm1(y))

class HybridParams(eqx.Module):
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    log_kwl: jax.Array
    log_kwg: jax.Array
    def __init__(self, key):
        k1, k2 = jax.random.split(key)
        self.layer1 = eqx.nn.Linear(2, 3, key=k1)
        l2 = eqx.nn.Linear(3, 1, key=k2)
        self.layer2 = eqx.tree_at(lambda l: (l.weight, l.bias), l2, (jnp.zeros_like(l2.weight), _inv_softplus(jnp.array([WARM_FI]))))
        self.log_kwl = jnp.log(jnp.array(WARM_KWL)); self.log_kwg = jnp.log(jnp.array(WARM_KWG))
    def f_i_mult(self, x): return jax.nn.softplus(self.layer2(jnp.tanh(self.layer1(x))))[0]
    @property
    def k_wl(self): return jnp.exp(self.log_kwl)
    @property
    def k_wg(self): return jnp.exp(self.log_kwg)

def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, params):
    x = (jnp.array([jnp.log(rho_l * U_sl * D / mu_l), jnp.log(rho_g * U_sg * D / mu_g)]) - FEAT_MEAN) / FEAT_STD
    f, kl, kg = params.f_i_mult(x), params.k_wl, params.k_wg
    ht = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f, kl, kg, beta=BETA)
    A, A_l, A_g, *_ = td_geometry(ht, D)
    a_l, a_g = A_l / A, A_g / A
    Delta = rho_l * U_sl / jnp.maximum(a_l, 1e-9) - rho_g * U_sg / jnp.maximum(a_g, 1e-9)
    d, _ = predicted_pressure_gradient(a_l, Delta, U_sl + U_sg, D, rho_l, rho_g, mu_l, mu_g, f, kl, kg, beta=BETA)
    return a_l, d, (f, kl, kg)

def loss_fn(params, cols):
    sh, sd = [], []
    for i in range(len(cols['U_sl'])):
        a, d, _ = predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i], cols['rho_g'][i],
                              cols['mu_l'][i], cols['mu_g'][i], params)
        sh.append(((a - cols['psi'][i]) / cols['psi'][i]) ** 2)
        sd.append(((jnp.abs(d) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]) ** 2)
    lh, ld = jnp.mean(jnp.stack(sh)), jnp.mean(jnp.stack(sd))
    return lh + W_DPDZ * ld, (lh, ld)

grad_fn = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn, has_aux=True))
eval_fn = eqx.filter_jit(loss_fn)

def rows_for(params, cols, split):
    out = []
    for i in range(len(cols['U_sl'])):
        a, d, (f, kl, kg) = predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i], cols['rho_g'][i],
                                        cols['mu_l'][i], cols['mu_g'][i], params)
        out.append(dict(split=split, U_sl=cols['U_sl'][i], U_sg=cols['U_sg'][i], f_i_mult=float(f), HL=cols['psi'][i], hl_pred=float(a),
                        dpdz=cols['dpdz_measured'][i], dpdz_pred=abs(float(d))))
    return out

results = {}
for seed, epochs in EPOCH_RUNS:
    params = HybridParams(jax.random.PRNGKey(seed))
    opt = optax.adamw(LR, weight_decay=WEIGHT_DECAY); os_ = opt.init(eqx.filter(params, eqx.is_array))
    best = (float('inf'), params, None); t0 = time.time()
    for ep in range(epochs):
        (lv, _), g = grad_fn(params, train_c)
        up, os_ = opt.update(g, os_, eqx.filter(params, eqx.is_array)); params = eqx.apply_updates(params, up)
        lv = float(lv)
        if lv < best[0]: best = (lv, params, ep)
    bp = best[1]
    trl, (trh, trd) = eval_fn(bp, train_c); tel, (teh, ted) = eval_fn(bp, test_c)
    rows = rows_for(bp, train_c, 'train') + rows_for(bp, test_c, 'test')
    def stats(split):
        r = [x for x in rows if x['split'] == split]
        eh = np.array([(x['hl_pred'] - x['HL']) / x['HL'] * 100 for x in r]); ed = np.array([(x['dpdz_pred'] - x['dpdz']) / x['dpdz'] * 100 for x in r])
        return np.abs(eh).mean(), eh.mean(), np.abs(ed).mean()
    key = f"seed{seed}_{epochs}"
    results[key] = dict(seed=seed, epochs=epochs, best_epoch=best[2], k_wl=float(bp.k_wl), k_wg=float(bp.k_wg),
                        train=(float(trl), float(trh), float(trd)), test=(float(tel), float(teh), float(ted)), rows=rows,
                        stats_train=stats('train'), stats_test=stats('test'),
                        f_range=(min(x['f_i_mult'] for x in rows), max(x['f_i_mult'] for x in rows)))
    if seed == 0 and epochs == 3000: eqx.tree_serialise_leaves("st_nn_fi_only_beta_seed0.eqx", bp)
    r = results[key]
    print(f"[{key}] {time.time() - t0:.0f}s best_epoch={best[2]} k_wl={r['k_wl']:.3f} k_wg={r['k_wg']:.3f} f_i_mult {r['f_range'][0]:.2f}-{r['f_range'][1]:.2f} | "
          f"train {r['train'][0]:.4f} (hld {r['train'][1]:.4f} dpdz {r['train'][2]:.4f})  test {r['test'][0]:.4f} (hld {r['test'][1]:.4f} dpdz {r['test'][2]:.4f})  "
          f"| mean|err| hld tr/te {r['stats_train'][0]:.1f}/{r['stats_test'][0]:.1f}%  dpdz tr/te {r['stats_train'][2]:.1f}/{r['stats_test'][2]:.1f}%", flush=True)

# --- like-for-like: the saved SW-only NN, with ITS OWN loss/data
sys.argv = [sys.argv[0]]
import _vuong_fit_nn_fi_only as SW
swp = eqx.tree_deserialise_leaves("vuong_nn_fi_only_best.eqx", SW.HybridParams(jax.random.PRNGKey(0)))
swtr, (swtrh, swtrd) = SW.loss_fn(swp, SW._cols(SW.train)); swte, (swteh, swted) = SW.loss_fn(swp, SW._cols(SW.test))
results['SW_only_NN'] = dict(train=(float(swtr), float(swtrh), float(swtrd)), test=(float(swte), float(swteh), float(swted)),
                             k_wl=float(swp.k_wl), k_wg=float(swp.k_wg), n_train=len(SW.train), n_test=len(SW.test))
print(f"\n[SW-only NN, re-evaluated] n={len(SW.train)}/{len(SW.test)} k_wl={float(swp.k_wl):.3f} k_wg={float(swp.k_wg):.3f} | "
      f"train {float(swtr):.4f} (hld {float(swtrh):.4f} dpdz {float(swtrd):.4f})  test {float(swte):.4f} (hld {float(swteh):.4f} dpdz {float(swted):.4f})", flush=True)
pickle.dump(results, open("st_nn_fi_only_beta_result.pkl", "wb"))
print("saved st_nn_fi_only_beta_result.pkl", flush=True)
