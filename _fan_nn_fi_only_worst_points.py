"""
Diagnose which Fan NN-fi-only test (and train) conditions carry the
largest holdup/dPdz error, using the ALREADY-SAVED checkpoint
(fan_nn_fi_only_best.eqx) -- no retraining. Sorted worst-first so we can
see whether the errors cluster (thin data region, specific angle, extreme
U_sg) or scatter randomly.
"""
import jax
import equinox as eqx
from _fan_data import load_all_conditions, build_split
from _fan_fit_nn_fi_only import HybridParams, predict_one, _cols, _feat_stats

ALL_CONDITIONS = load_all_conditions()
TRAIN_CONDITIONS, TEST_CONDITIONS = build_split(ALL_CONDITIONS)
train_cols = _cols(TRAIN_CONDITIONS)
test_cols = _cols(TEST_CONDITIONS)
feat_mean, feat_std = _feat_stats(train_cols)

template = HybridParams(jax.random.PRNGKey(0))
params = eqx.tree_deserialise_leaves('fan_nn_fi_only_best.eqx', template)
print(f"loaded checkpoint. k_wl={float(params.k_wl):.4f}  k_wg={float(params.k_wg):.4f}\n")


def report(name, cols, conds):
    rows = []
    for i in range(len(cols['U_sl'])):
        alpha_pred, dpdz_pred, (f, kl, kg) = predict_one(
            cols['U_sl'][i], cols['U_sg'][i], cols['D'][i], cols['rho_l'][i],
            cols['rho_g'][i], cols['mu_l'][i], cols['mu_g'][i], cols['beta'][i], params,
            feat_mean, feat_std,
        )
        Re_SG = cols['rho_g'][i] * cols['U_sg'][i] * cols['D'][i] / cols['mu_g'][i]
        Re_SL = cols['rho_l'][i] * cols['U_sl'][i] * cols['D'][i] / cols['mu_l'][i]
        dpdz_err = abs(abs(float(dpdz_pred)) - cols['dpdz_measured'][i]) / abs(cols['dpdz_measured'][i]) * 100
        psi_meas = cols['psi'][i]
        hld_err = None if psi_meas is None else abs(float(alpha_pred) - psi_meas) / psi_meas * 100
        angle = conds[i]['angle_deg']
        rows.append({
            'U_sl': cols['U_sl'][i], 'U_sg': cols['U_sg'][i], 'angle': angle,
            'Re_SL': Re_SL, 'Re_SG': Re_SG, 'f_i_mult': float(f),
            'psi_meas': psi_meas, 'psi_pred': float(alpha_pred), 'hld_err': hld_err,
            'dpdz_meas': cols['dpdz_measured'][i], 'dpdz_pred': float(dpdz_pred), 'dpdz_err': dpdz_err,
        })

    print(f"=== {name}: holdup errors, worst first ===")
    hld_rows = sorted([r for r in rows if r['hld_err'] is not None], key=lambda r: -r['hld_err'])
    print(f"{'U_sl':>9} {'U_sg':>8} {'angle':>6} {'Re_SL':>9} {'Re_SG':>10} {'f_i_mult':>9} "
          f"{'psi_meas':>9} {'psi_pred':>9} {'hld_err%':>9}")
    for r in hld_rows:
        print(f"{r['U_sl']:9.5f} {r['U_sg']:8.3f} {r['angle']:6.1f} {r['Re_SL']:9.1f} {r['Re_SG']:10.1f} "
              f"{r['f_i_mult']:9.4f} {r['psi_meas']:9.4f} {r['psi_pred']:9.4f} {r['hld_err']:9.2f}")

    print(f"\n=== {name}: dPdz errors, worst first ===")
    dpdz_rows = sorted(rows, key=lambda r: -r['dpdz_err'])
    print(f"{'U_sl':>9} {'U_sg':>8} {'angle':>6} {'Re_SL':>9} {'Re_SG':>10} {'f_i_mult':>9} "
          f"{'dpdz_meas':>10} {'dpdz_pred':>10} {'dpdz_err%':>9}")
    for r in dpdz_rows:
        print(f"{r['U_sl']:9.5f} {r['U_sg']:8.3f} {r['angle']:6.1f} {r['Re_SL']:9.1f} {r['Re_SG']:10.1f} "
              f"{r['f_i_mult']:9.4f} {r['dpdz_meas']:10.2f} {r['dpdz_pred']:10.2f} {r['dpdz_err']:9.2f}")
    print()
    return rows


train_rows = report("TRAIN", train_cols, TRAIN_CONDITIONS)
test_rows = report("TEST", test_cols, TEST_CONDITIONS)

print("=== Feature-space coverage check: for each TEST point, nearest TRAIN point in (log Re_SL, log Re_SG) ===")
import numpy as np
train_feat = np.array([[np.log(r['Re_SL']), np.log(r['Re_SG'])] for r in train_rows])
for r in sorted(test_rows, key=lambda r: -(r['hld_err'] or 0)):
    tf = np.array([np.log(r['Re_SL']), np.log(r['Re_SG'])])
    dists = np.linalg.norm((train_feat - tf) / np.array([float(feat_std[0]), float(feat_std[1])]), axis=1)
    nearest = dists.min()
    hld_str = f"{r['hld_err']:.1f}%" if r['hld_err'] is not None else "  -- "
    print(f"U_sl={r['U_sl']:.5f} U_sg={r['U_sg']:7.3f} angle={r['angle']:5.1f}  "
          f"hld_err={hld_str:>7}  dpdz_err={r['dpdz_err']:6.1f}%  nearest_train_dist(std units)={nearest:.3f}")
