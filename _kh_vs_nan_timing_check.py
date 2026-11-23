"""
Option (3) from the surface-tension review: before assuming a grid-
resolution fix (finer N) or a numerics fix (tighter Newton/GMRES) is the
right next step, check whether this is even the right CLASS of fix at
all.

local_wave_speeds' is_complex flag is computed PURELY from the local
per-cell state (alpha_l, Delta) and the fixed problem parameters (Q, D,
rho_l, rho_g, g, beta) -- no spatial derivative, no wavenumber, nothing
grid-dependent. It is the condition for the (alpha_l, Delta) system's
flux Jacobian to have genuinely complex eigenvalues RIGHT THERE, in the
k->0 (locally uniform) limit. Surface tension's regularization enters as
a dispersive term that scales as sigma*k^3 -- it VANISHES as k->0. If
is_complex is already True (hyperbolicity already locally lost) before
the state ever goes NaN, no amount of a k^3-type term, at any magnitude,
could be structurally expected to fix it -- it only damps short
wavelengths, not the local breakdown itself.

This script replicates run_real_data_condition's exact setup for one
representative fast-NaN Vuong ST condition (idx=797) at sigma=0, but
steps through advance_state in small chunks (instead of one big
jax.lax.scan) so we can inspect, between chunks: (1) whether any NaN has
appeared yet, (2) how many cells currently have is_complex=True (via
local_wave_speeds on the CURRENT state, before the next chunk runs).
This tells us whether local hyperbolicity loss PRECEDES the NaN
(supporting the "wrong class of fix" hypothesis) or whether the state
goes straight from "clean, real eigenvalues" to NaN with no complex-
eigenvalue warning at all (which would point elsewhere -- e.g. a pure
numerical/CFL violation, not a hyperbolicity-loss mechanism).
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
IDX = 797
CHUNK = 20          # steps between diagnostic checks
N_CHUNKS = 150      # 150*20 = 3000 steps -- enough to see the fast-NaN condition blow up

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

row = st.loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]'])
rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, 0.0)
U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
g_star, D_star = nd['g_star'], nd['D_star']
L_star = L_VUONG / D
dx_star = L_star / N

dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                     mu_l_star, mu_g_star, dx_star, g=g_star, beta=beta)

alpha_l_inlet = U_sl_star
Delta_inlet_star = rho_l_star - rho_g_star
state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

@jax.jit
def run_chunk(state):
    def body(s, _):
        ns, nkh = solver.advance_state(
            s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet_star, g_star,
            beta=beta, sigma=0.0
        )
        return ns, nkh
    final, nkh = jax.lax.scan(body, state, None, length=CHUNK)
    return final, jnp.sum(nkh)

@jax.jit
def count_complex(state):
    _, _, is_complex = solver.local_wave_speeds(
        state['alpha_l'], state['Delta'], 1.0, D_star, rho_l_star, rho_g_star,
        g_star, beta
    )
    return jnp.sum(is_complex)

print(f"idx={IDX}  U_sl={U_sl}  U_sg={U_sg}  dt_star={dt_star:.6e}  dx_star={dx_star:.6e}", flush=True)
print(f"{'step':>8} {'any_nan':>8} {'n_complex':>10} {'alpha_min':>10} {'alpha_max':>10}", flush=True)

first_complex_step = None
first_nan_step = None

for i in range(N_CHUNKS):
    n_complex_before = int(count_complex(state))
    if first_complex_step is None and n_complex_before > 0:
        first_complex_step = i * CHUNK

    state, nkh_chunk = run_chunk(state)
    jax.block_until_ready(state)

    any_nan = bool(jnp.any(jnp.isnan(state['alpha_l'])) or jnp.any(jnp.isnan(state['Delta'])))
    a_min = float(jnp.nanmin(state['alpha_l'])) if not any_nan else float('nan')
    a_max = float(jnp.nanmax(state['alpha_l'])) if not any_nan else float('nan')

    if first_nan_step is None and any_nan:
        first_nan_step = (i + 1) * CHUNK

    if i % 5 == 0 or any_nan or n_complex_before > 0:
        print(f"{(i+1)*CHUNK:>8} {str(any_nan):>8} {n_complex_before:>10} {a_min:>10.5f} {a_max:>10.5f}", flush=True)

    if any_nan:
        break

print(flush=True)
print(f"first step with is_complex>0 (BEFORE that chunk ran): {first_complex_step}", flush=True)
print(f"first step with any_nan (AFTER that chunk ran):       {first_nan_step}", flush=True)
print("\nDONE", flush=True)
