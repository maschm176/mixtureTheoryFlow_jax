"""
sigma=0.024 (real physical value) did not rescue ANY of the 17
previously-NaN Vuong ST conditions -- exactly reproducing the sigma=0
pattern. Before concluding anything about the physics or the
implementation, test the most likely grid-resolution explanation
directly: surface tension's stabilizing term grows as sigma*k^3, so it
only matters once k is large enough -- if the "crossover" wavenumber
where sigma*k^3 starts to compete with the destabilizing shear term
corresponds to a wavelength much shorter than this grid's own cell size
(dx ~ 85/150 = 0.57 m here), the grid simply cannot "see" sigma's
regularizing effect at all, no matter how correctly it's implemented.

Scans sigma across several orders of magnitude above the real value, on
ONE representative fast-NaN condition (idx=797), to check whether the
mechanism rescues the condition at SOME large-enough sigma -- confirming
whether this is a scale-separation issue (real sigma too weak at this
grid's resolvable wavenumbers) rather than an implementation problem.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 150.0
IDX = 797
SIGMAS = [0.0, 0.024, 0.24, 2.4, 24.0, 240.0, 2400.0, 24000.0]

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

row = st.loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]'])
rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=beta))
A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
cf_alpha_l = float(A_l / A)
print(f"idx={IDX}  U_sl={U_sl}  U_sg={U_sg}  cf_alpha_l={cf_alpha_l:.4f}", flush=True)
print(f"{'sigma':>10} {'nan':>6} {'a_pred':>8} {'diff':>8}", flush=True)

for sigma in SIGMAS:
    r = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L_VUONG,
                                        N=N, n_transit_times=N_TRANSIT, beta=beta, sigma=sigma)
    a_pred = "nan" if r['any_nan'] else f"{r['alpha_l_pred']:.4f}"
    diff = "nan" if r['any_nan'] else f"{abs(r['alpha_l_pred'] - cf_alpha_l):.4f}"
    print(f"{sigma:>10.3f} {str(r['any_nan']):>6} {a_pred:>8} {diff:>8}", flush=True)

print("\nDONE", flush=True)
