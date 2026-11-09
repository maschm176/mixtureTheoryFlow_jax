"""
Gas-phase Weber number check: is the power-law fit's sign flip relative to
Carraretto et al. (2020)'s f_i=2.25*Re_SG^-0.53 (see "Power-law f_i_mult
(Re_SG) extension beats the flat fit, with a sign surprise" in CLAUDE.md)
explainable by Vuong (2016) and MDPI/Carraretto sitting in genuinely
different interfacial-wave regimes, not just "different fluids, different
numbers"?

We = rho_g * u_rel^2 * D / sigma -- ratio of gas-phase inertial/shear
force to the interfacial surface-tension restoring force. We<<1: surface
tension suppresses interfacial waves (smooth interface, friction
decreases with Re the way turbulent wall friction does -- Carraretto's
regime). We>>1: inertia overwhelms surface tension, waves grow and
roughen the interface, friction INCREASES with Re instead (Andritsos &
Hanratty 1987-type roughening, already connected to Rodriguez & Oliemans
2006's "interfacial wave roughness" fix cited elsewhere in this file).

u_rel (= u_g - u_l) comes from solving each condition to its Taitel-
Dukler equilibrium holdup (same solve_taitel_dukler_holdup/td_geometry
machinery as every other fit in this project) -- NOT from superficial
velocities alone, since the real interfacial shear velocity is set by the
actual phase velocities, which depend on holdup.
  - Vuong: evaluated at the fitted power-law closure
    (f0=5.5650, alpha_fi=+0.6083, k_wl=2.1475, k_wg=0.5659).
  - MDPI: no fitted closure exists for this dataset (Phase 2g is still
    blocked on the PHI_COLLAPSE issue -- see Known Issues). Evaluated at
    the UNCALIBRATED default closure (f_i_mult=k_wl=k_wg=1.0) purely for
    a representative holdup/velocity estimate -- the resulting We is an
    order-of-magnitude estimate, not a precise fitted value, since MDPI's
    own closure parameters are unknown.

sigma_vuong=0.024 N/m (oil Isopar L - nitrogen), sigma_waterair=0.072 N/m
-- both already in CLAUDE.md's cross-fluid-pair generalization table.

Result (see CLAUDE.md "Weber number check confirms..." for the full
writeup): Vuong median We=958 (range 336-1909), MDPI median We=3.7 (range
0.8-12.4) -- a ~259x gap. Decomposed: dominated by Vuong's gas density
being ~20x MDPI's (rho_g~24.2 vs 1.204 kg/m3 -- points to Vuong's
facility running at elevated pressure, ~20 atm by ideal-gas back-
calculation, not a nitrogen-vs-air chemistry effect), with pipe diameter
(2.6x) and relative velocity^2 (1.85x) as smaller contributors, and the
surface-tension difference itself only a 3.0x factor -- smaller than
originally guessed.
"""
import csv
import numpy as np
from mixtureTheoryFlow_jax_solver_water_gas import solve_taitel_dukler_holdup, td_geometry
from _vuong_fit_joint import build_split

SIGMA_VUONG = 0.024       # oil (Isopar L) - nitrogen, N/m
SIGMA_WATERAIR = 0.072    # water - air, N/m

F0, ALPHA_FI, K_WL, K_WG = 5.5650, 0.6083, 2.1475, 0.5659
MEAN_LOG_RE_SG = 13.2389

D_MDPI, RHO_L_MDPI, RHO_G_MDPI, MU_L_MDPI, MU_G_MDPI = 0.060, 998.0, 1.204, 1.0e-3, 1.81e-5
D_VUONG = 0.156


def solve_props(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg):
    ht_eq = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht_eq, D)
    alpha_l = float(A_l / A)
    alpha_g = float(A_g / A)
    u_l = U_sl / max(alpha_l, 1e-9)
    u_g = U_sg / max(alpha_g, 1e-9)
    return rho_g, u_g - u_l


def weber(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, sigma, f_i_mult=1.0, k_wl=1.0, k_wg=1.0):
    rho_g_used, u_rel = solve_props(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, k_wl, k_wg)
    We = rho_g_used * u_rel**2 * D / sigma
    return We, u_rel, rho_g_used


# --- Vuong, at the fitted power-law closure ---
train, test = build_split()
vuong_rows = []
for c, split in [(c, 'train') for c in train] + [(c, 'test') for c in test]:
    Re_SG = c['rho_g'] * c['U_sg'] * c['D'] / c['mu_g']
    f_i_mult = float(F0 * np.exp(ALPHA_FI * (np.log(Re_SG) - MEAN_LOG_RE_SG)))
    We, u_rel, rho_g_used = weber(c['U_sl'], c['U_sg'], c['D'], c['rho_l'], c['rho_g'],
                                   c['mu_l'], c['mu_g'], SIGMA_VUONG, f_i_mult, K_WL, K_WG)
    vuong_rows.append((split, c['U_sg'], Re_SG, We, u_rel, rho_g_used))

# --- MDPI, uncalibrated default closure (f_i_mult=k_wl=k_wg=1) ---
mdpi_rows = []
with open('mdpi_airwater_phase2g_dataset.csv') as f:
    for row in csv.DictReader(f):
        U_sl, U_sg = float(row['JL']), float(row['JG'])
        Re_SG = RHO_G_MDPI * U_sg * D_MDPI / MU_G_MDPI
        We, u_rel, rho_g_used = weber(U_sl, U_sg, D_MDPI, RHO_L_MDPI, RHO_G_MDPI,
                                       MU_L_MDPI, MU_G_MDPI, SIGMA_WATERAIR)
        mdpi_rows.append((row['flow_regime'], U_sg, Re_SG, We, u_rel, rho_g_used))

we_vuong = np.array([r[3] for r in vuong_rows])
we_mdpi = np.array([r[3] for r in mdpi_rows])

print(f"Vuong (oil-N2):   n={len(we_vuong)}  We median={np.median(we_vuong):.1f}  "
      f"range=[{we_vuong.min():.1f}, {we_vuong.max():.1f}]")
print(f"MDPI (water-air): n={len(we_mdpi)}   We median={np.median(we_mdpi):.1f}  "
      f"range=[{we_mdpi.min():.1f}, {we_mdpi.max():.1f}]")
print(f"ratio of medians: {np.median(we_vuong) / np.median(we_mdpi):.1f}x")

rg_v = np.array([r[5] for r in vuong_rows])
ur_v = np.array([r[4] for r in vuong_rows])
ur_m = np.array([r[4] for r in mdpi_rows])
rho_factor = np.median(rg_v) / RHO_G_MDPI
vel_factor = np.median(np.abs(ur_v))**2 / np.median(np.abs(ur_m))**2
D_factor = D_VUONG / D_MDPI
sigma_factor = SIGMA_WATERAIR / SIGMA_VUONG
print(f"\ndecomposition: rho_g={rho_factor:.1f}x  u_rel^2={vel_factor:.2f}x  "
      f"D={D_factor:.2f}x  1/sigma={sigma_factor:.2f}x  "
      f"product={rho_factor*vel_factor*D_factor*sigma_factor:.1f}x")

print(f"\n# VUONG_ROWS: (split, U_sg, Re_SG, We)")
for split, usg, re_sg, We, u_rel, rho_g_used in vuong_rows:
    print(f'    ("{split}", {usg:.3f}, {re_sg:.1f}, {We:.2f}),')

print(f"\n# MDPI_ROWS: (flow_regime, U_sg, Re_SG, We)")
for regime, usg, re_sg, We, u_rel, rho_g_used in mdpi_rows:
    print(f'    ("{regime}", {usg:.4f}, {re_sg:.1f}, {We:.2f}),')
