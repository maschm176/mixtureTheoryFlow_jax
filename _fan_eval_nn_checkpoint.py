"""
Validate fan_plots/plot_fan_nn_fi_only.py's HARDCODED data table against the actual checkpoint.

The plot script's ROWS were transcribed from a /tmp log that no longer exists, so they can't be
checked against their source. This re-evaluates fan_nn_fi_only_best.eqx through the same
predict_one() path the fit used, regenerates every per-condition value, and compares them with the
ROWS table (parsed out of the plot script with ast -- importing it would run the plotting).
Also prints the train/test losses to compare with the ones recorded in that script's header
(train_loss=0.029812 hld=0.017348 dpdz=0.012464; test_loss=0.081156 holdup=0.070974 dpdz=0.010182).
Read-only: writes nothing.
"""
import ast
import jax, equinox as eqx
jax.config.update("jax_enable_x64", True)
import numpy as np
import _fan_fit_nn_fi_only as m

params = eqx.tree_deserialise_leaves('fan_nn_fi_only_best.eqx', m.HybridParams(jax.random.PRNGKey(0)))
train_cols, test_cols = m._cols(m.TRAIN_CONDITIONS), m._cols(m.TEST_CONDITIONS)
fm, fs = m._feat_stats(train_cols)                       # stats from TRAIN only, as in the fit
loss_fn = m.make_loss_fn(fm, fs)
tr, (trh, trd) = loss_fn(params, train_cols); te, (teh, ted) = loss_fn(params, test_cols)
print(f"k_wl={float(params.k_wl):.4f} k_wg={float(params.k_wg):.4f}   (recorded: 2.3110, 1.2617)")
print(f"train loss={float(tr):.6f} (hld={float(trh):.6f} dpdz={float(trd):.6f})   recorded: 0.029812 (0.017348, 0.012464)")
print(f"test  loss={float(te):.6f} (hld={float(teh):.6f} dpdz={float(ted):.6f})   recorded: 0.081156 (0.070974, 0.010182)")

src = open('fan_plots/plot_fan_nn_fi_only.py').read()
rows = next(ast.literal_eval(n.value) for n in ast.parse(src).body
            if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') == 'ROWS')

mx = dict(f=0.0, psi=0.0, dpdz=0.0, n=0)
for split, cols, conds in (('train', train_cols, m.TRAIN_CONDITIONS), ('test', test_cols, m.TEST_CONDITIONS)):
    exp = [r for r in rows if r[0] == split]
    assert len(exp) == len(cols['U_sl']), (split, len(exp), len(cols['U_sl']))
    for i, r in enumerate(exp):
        a, d, (f, kl, kg) = m.predict_one(cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
                                          cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i],
                                          params, fm, fs)
        assert abs(cols['U_sl'][i] - r[1]) < 1e-6 and abs(cols['U_sg'][i] - r[2]) < 1e-3, ("row order mismatch", split, i)
        mx['f'] = max(mx['f'], abs(float(f) - r[4])); mx['psi'] = max(mx['psi'], abs(float(a) - r[6]))
        mx['dpdz'] = max(mx['dpdz'], abs(float(d) - r[8])); mx['n'] += 1
print(f"\ncompared {mx['n']} rows against the plot script's ROWS (same order, same U_sl/U_sg):")
print(f"  max |f_i_mult diff|={mx['f']:.2e} (table has 4 d.p.)  max |psi_pred diff|={mx['psi']:.2e} (4 d.p.)  "
      f"max |dpdz_pred diff|={mx['dpdz']:.2e} (2 d.p.)")
