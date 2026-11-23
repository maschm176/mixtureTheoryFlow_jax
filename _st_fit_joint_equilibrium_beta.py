"""
ST-dedicated (f_i_mult, k_wl, k_wg) refit WITH INCLINATION.

_st_fit_joint_equilibrium.py's predict_one never passed beta to solve_taitel_dukler_holdup /
predicted_pressure_gradient, so the +2 deg Vuong ST rows were fit as if horizontal (its recorded losses
0.020856 / 0.066252 reproduce only at beta=0; at beta=+2deg they are 0.073 / 0.089). This is an otherwise
IDENTICAL copy of that script -- same 21 rows, same 15/6 train/test split, same loss (relative holdup error^2
+ relative |dP/dz| error^2), same Adam(lr=0.06, 3000 epochs) in log-space, same jax.grad through the closed-form
equilibrium path -- with beta = +2 deg threaded through. The original script is left untouched.

Run from two warm starts to see whether the answer depends on the basin:
  (A) the SW production values (as in the original), (B) the old beta=0 ST fit.
Writes st_fit_beta_result.pkl. Positive angle = uphill (assumed, as everywhere in this project).
"""
import math, pickle, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import optax
import pandas as pd
from mixtureTheoryFlow_jax_solver_water_gas import (
    solve_taitel_dukler_holdup, td_geometry, predicted_pressure_gradient,
)

BETA = math.radians(2.0)
W_DPDZ, LR, N_EPOCHS = 1.0, 0.06, 3000
TEST_IDX = [805, 836, 1457, 1487, 1523, 1543]
TRAIN_IDX = [797, 820, 821, 822, 826, 832, 840, 823, 1464, 1479, 1496, 1508, 1522, 1524, 1550]
STARTS = {'A: SW production': (3.8893, 1.5419, 0.8413), 'B: old beta=0 ST fit': (10.6923, 2.6477, 0.8510)}

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]

def _cols(idx):
    s = st.loc[idx]
    return {'idx': idx, 'U_sl': [float(v) for v in s['vSL[m/s]']], 'U_sg': [float(v) for v in s['vSG[m/s]']],
            'D': [float(v) for v in s['Pipe Diameter[m]']], 'rho_l': [float(v) for v in s['rhoL[kg/m3]']],
            'rho_g': [float(v) for v in s['rhoG[kg/m3]']], 'mu_l': [float(v) for v in s['viscosityL[Pa.s]']],
            'mu_g': [float(v) for v in s['viscosityG[Pa.s]']], 'psi': [float(v) for v in s['HL[-]']],
            'dpdz_measured': [float(v) for v in s['dP/dL[Pa/m]']]}

train_c, test_c = _cols(TRAIN_IDX), _cols(TEST_IDX)

def predict_one(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg, beta=BETA):
    Q = U_sl + U_sg
    ht = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg, beta=beta)
    A, A_l, A_g, *_ = td_geometry(ht, D)
    a_l, a_g = A_l / A, A_g / A
    Delta = rho_l * U_sl / jnp.maximum(a_l, 1e-9) - rho_g * U_sg / jnp.maximum(a_g, 1e-9)
    d, _ = predicted_pressure_gradient(a_l, Delta, Q, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg, beta=beta)
    return a_l, d

def loss_fn(z, cols):
    f, kl, kg = jnp.exp(z[0]), jnp.exp(z[1]), jnp.exp(z[2])
    sh, sd = [], []
    for i in range(len(cols['U_sl'])):
        a, d = predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i], cols['rho_g'][i],
                           cols['mu_l'][i], cols['mu_g'][i], f, kl, kg)
        sh.append(((a - cols['psi'][i]) / cols['psi'][i]) ** 2)
        sd.append(((jnp.abs(d) - cols['dpdz_measured'][i]) / cols['dpdz_measured'][i]) ** 2)
    lh, ld = jnp.mean(jnp.stack(sh)), jnp.mean(jnp.stack(sd))
    return lh + W_DPDZ * ld, (lh, ld)

grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

def per_condition(params, cols, label):
    out = []
    for i in range(len(cols['U_sl'])):
        a, d = predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i], cols['rho_g'][i],
                           cols['mu_l'][i], cols['mu_g'][i], *params)
        out.append(dict(split=label, idx=cols['idx'][i], U_sl=cols['U_sl'][i], U_sg=cols['U_sg'][i],
                        HL=cols['psi'][i], hl_pred=float(a), dpdz=cols['dpdz_measured'][i], dpdz_pred=abs(float(d))))
    return out

results = {}
for name, start in STARTS.items():
    z = jnp.log(jnp.array(start)); opt = optax.adam(LR); os_ = opt.init(z)
    best = (float('inf'), None, None); t0 = time.time()
    for ep in range(N_EPOCHS):
        (lv, (lh, ld)), g = grad_fn(z, train_c)
        u, os_ = opt.update(g, os_); z = z + u
        lv = float(lv)
        if lv < best[0]:
            best = (lv, ep, z)
    p = tuple(float(v) for v in jnp.exp(best[2]))
    (tl, (tlh, tld)), _ = grad_fn(best[2], test_c); (trl, (trh, trd)), _ = grad_fn(best[2], train_c)
    rows = per_condition(p, train_c, 'train') + per_condition(p, test_c, 'test')
    e = np.array([(r['hl_pred'] - r['HL']) / r['HL'] * 100 for r in rows])
    ed = np.array([(r['dpdz_pred'] - r['dpdz']) / r['dpdz'] * 100 for r in rows])
    results[name] = dict(params=p, best_epoch=best[1], train=(float(trl), float(trh), float(trd)),
                         test=(float(tl), float(tlh), float(tld)), rows=rows)
    print(f"\n[{name}] {time.time() - t0:.0f}s  best_epoch={best[1]}  f_i_mult={p[0]:.4f} k_wl={p[1]:.4f} k_wg={p[2]:.4f}")
    print(f"   train loss={float(trl):.6f} (hld={float(trh):.6f} dpdz={float(trd):.6f})   "
          f"test loss={float(tl):.6f} (hld={float(tlh):.6f} dpdz={float(tld):.6f})")
    print(f"   all 21 rows at beta=+2deg: holdup mean|err|={np.abs(e).mean():.1f}% bias={e.mean():+.1f}%   "
          f"dP/dz mean|err|={np.abs(ed).mean():.1f}% bias={ed.mean():+.1f}%")
pickle.dump(results, open("st_fit_beta_result.pkl", "wb"))
print("\nsaved st_fit_beta_result.pkl")
