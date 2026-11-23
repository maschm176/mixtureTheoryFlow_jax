from mixtureTheoryFlow_jax_solver_water_gas import run_real_data_condition
import pandas as pd, numpy as np, time

# --- MDPI (all 32, already confirmed all kh-stable with holdup meas) ---
mdpi_df = pd.read_csv('mdpi_airwater_phase2g_dataset.csv')
mdpi_clean = mdpi_df[['psi', 'JL', 'JG']].copy().rename(columns={'JL': 'U_sl', 'JG': 'U_sg'})
D_mdpi, L_mdpi = 0.060, 24.0
rho_l_mdpi, rho_g_mdpi = 998.0, 1.204
mu_l_mdpi, mu_g_mdpi = 1.0e-3, 1.81e-5

mdpi_conditions = []
for _, row in mdpi_clean.iterrows():
    mdpi_conditions.append(dict(
        U_sl=row['U_sl'], U_sg=row['U_sg'], D=D_mdpi, rho_l=rho_l_mdpi, rho_g=rho_g_mdpi,
        mu_l=mu_l_mdpi, mu_g=mu_g_mdpi, L=L_mdpi, alpha_l_meas=row['psi']
    ))

# --- Fan (2005), horizontal (0 deg) only, kh-stable, with HL measurement ---
fan_df = pd.read_csv('fan2005_sw_subset.csv')
fan_horiz = fan_df[fan_df['Inclination angle[°]'] == 0].copy()
fan_clean = fan_horiz.rename(columns={
    'vSL[m/s]': 'U_sl', 'vSG[m/s]': 'U_sg', 'rhoL[kg/m3]': 'rho_l', 'rhoG[kg/m3]': 'rho_g',
    'viscosityL[Pa.s]': 'mu_l', 'viscosityG[Pa.s]': 'mu_g', 'Pipe Diameter[m]': 'D', 'Pipe Length[m]': 'L',
})[['U_sl', 'U_sg', 'D', 'rho_l', 'rho_g', 'mu_l', 'mu_g', 'L']].copy()
fan_clean['psi'] = pd.to_numeric(fan_horiz['HL[-]'], errors='coerce')
fan_clean = fan_clean.dropna(subset=['psi']).reset_index(drop=True)

fan_conditions = []
for _, row in fan_clean.iterrows():
    # quick pre-check: skip known kh-unstable Fan conditions (identified earlier: the
    # two highest-U_sl members of the U_sg~19.7-19.9 group -- but re-verify via kh_total
    # at eval time below rather than hardcoding, since this is a fresh pooled fit.
    fan_conditions.append(dict(
        U_sl=row['U_sl'], U_sg=row['U_sg'], D=row['D'], rho_l=row['rho_l'], rho_g=row['rho_g'],
        mu_l=row['mu_l'], mu_g=row['mu_g'], L=row['L'], alpha_l_meas=row['psi']
    ))

all_conditions = mdpi_conditions + fan_conditions
print(f"Pooled: {len(mdpi_conditions)} MDPI + {len(fan_conditions)} Fan(0deg) = {len(all_conditions)} conditions", flush=True)


def pooled_loss(f_i_mult, verbose=False):
    errs = []
    n_kh_unstable = 0
    for i, c in enumerate(all_conditions):
        res = run_real_data_condition(
            c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'], c['mu_l'], c['mu_g'], c['L'],
            f_i_mult=f_i_mult
        )
        if res['kh_total'] > 0:
            n_kh_unstable += 1
            continue
        rel_err = (res['alpha_l_pred'] - c['alpha_l_meas']) / c['alpha_l_meas']
        errs.append(rel_err)
        if verbose:
            tag = 'MDPI' if i < len(mdpi_conditions) else 'Fan'
            print(f"  [{tag} {i:2d}] alpha_l_pred={res['alpha_l_pred']:.4f}  meas={c['alpha_l_meas']:.4f}  "
                  f"rel_err={rel_err:+.1%}  kh={res['kh_total']}", flush=True)
    errs = np.array(errs)
    loss = float(np.mean(errs**2))
    return loss, errs, n_kh_unstable


F_I_MULT_GUESS = 1.0
h = 1e-3

print("\n=== Center eval (f_i_mult=1.0, zero-refit) ===", flush=True)
t0 = time.time()
loss_center, errs_center, nkh_center = pooled_loss(F_I_MULT_GUESS, verbose=True)
print(f"took {time.time()-t0:.1f}s", flush=True)
print(f"zero-refit loss={loss_center:.6f}  mean|rel_err|={np.mean(np.abs(errs_center)):.1%}  "
      f"n_kh_unstable={nkh_center}/{len(all_conditions)}", flush=True)

print("\n=== f_i_mult + h ===", flush=True)
t0 = time.time()
loss_plus, _, nkh_plus = pooled_loss(F_I_MULT_GUESS + h)
print(f"took {time.time()-t0:.1f}s  loss={loss_plus:.6f}  n_kh_unstable={nkh_plus}", flush=True)

print("\n=== f_i_mult - h ===", flush=True)
t0 = time.time()
loss_minus, _, nkh_minus = pooled_loss(F_I_MULT_GUESS - h)
print(f"took {time.time()-t0:.1f}s  loss={loss_minus:.6f}  n_kh_unstable={nkh_minus}", flush=True)

fd_grad = (loss_plus - loss_minus) / (2 * h)
print(f"\ncentral finite-difference grad = {fd_grad:.6f}", flush=True)
print("(negative => increasing f_i_mult reduces pooled holdup error; positive => decreasing it does)", flush=True)
