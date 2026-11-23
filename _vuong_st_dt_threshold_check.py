"""
dt-threshold check against the REAL, now-precisely-identified
genuinely-high-holdup Vuong (2016) "ST" (inclined, beta=+2deg) rows --
not a synthetic proxy condition (see the earlier, inconclusive check at
U_sl=0.50/U_sg=5.0/beta=0 on MDPI-like fluid properties).

Context: of the 28 Vuong ST rows, the solver's _bisect_ht multi-root fix
(2026-11-14) showed 5 (U_sl~0.005-0.01) are actually genuinely LOW-holdup
-- the dynamic solver had already settled them correctly; the old
"cf_alpha_l" reference was wrong, not the dynamics. The other 23 are
unaffected by the fix: their equilibrium genuinely has only one root, at
high holdup (0.33-0.56) -- these are the real test of whether the
documented high-holdup/no-surface-tension instability is a genuine
ill-posedness (needs surface tension) or an explicit-dt-overshoot
artifact (same blind spot that caused the RS degenerate-diffusivity
issue -- estimate_stable_dt is pure-CFL-based, has zero awareness of
local disc<0 severity).

Method: for each of the 23 conditions, sweep dt multipliers applied to
estimate_stable_dt's CFL estimate, holding total simulated physical time
fixed. SCOPE CUT (2026-11-14): the first version of this script used
n_transit=50 and 6 multipliers down to 0.01x -- at the smallest
multiplier, n_steps is ~100x the full-dt value, and jax.lax.scan does NOT
early-exit on NaN, so several of the smallest-dt runs alone could take
many minutes; 23 conditions x 6 multipliers made the total run
open-endedly long (killed after 51 min with zero output, due to a SEPARATE
problem -- stdout was fully block-buffered through `tee`, not just slow).
Cut to n_transit=10 (just detecting NaN onset, not reaching a settled
state) and 4 multipliers (1.0, 0.25, 0.05, 0.01) instead of 6. Output is
now unpiped/line-flushed so progress is visible live, not buffered until
exit.
"""
import sys
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 10.0
MULTS = [1.0, 0.25, 0.05, 0.01]

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)

# Classify using the FIXED solver's closed-form equilibrium -- keep only
# the 23 genuinely-high-holdup rows (cf_alpha_l > 0.1; the 5 corrected
# low-holdup rows all sit under 0.03).
rows = []
for idx, row in st.iterrows():
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
    ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=beta))
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    cf_alpha_l = float(A_l / A)
    if cf_alpha_l > 0.1:
        rows.append(dict(idx=idx, U_sl=U_sl, U_sg=U_sg, D=D, rho_l=rho_l, rho_g=rho_g,
                          mu_l=mu_l, mu_g=mu_g, cf_alpha_l=cf_alpha_l))

print(f"{len(rows)} genuinely-high-holdup Vuong ST conditions to test (cf_alpha_l > 0.1)", flush=True)
L_VUONG = 85.0

print(f"{'idx':>5} {'U_sl':>8} {'U_sg':>7} {'cf_a_l':>7} " + " ".join(f"mult={m:<5.2f}" for m in MULTS), flush=True)

for r in rows:
    U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g = r['U_sl'], r['U_sg'], r['D'], r['rho_l'], r['rho_g'], r['mu_l'], r['mu_g']
    Q = U_sl + U_sg
    nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81)
    U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
    rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
    mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
    g_star, D_star = nd['g_star'], nd['D_star']
    L_star = L_VUONG / D
    dx_star = L_star / N
    dt_cfl = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                        mu_l_star, mu_g_star, dx_star, 1.0, 1.0, 1.0, 0.25,
                                        g=g_star, beta=beta)
    alpha_l_inlet = U_sl_star
    Delta_inlet = rho_l_star - rho_g_star

    results = []
    for mult in MULTS:
        dt_star = dt_cfl * mult
        n_steps = int(np.ceil(N_TRANSIT * L_star / dt_star))
        print(f"  ... row idx={r['idx']} mult={mult:.2f} n_steps={n_steps}", flush=True)
        state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

        @jax.jit
        def run(state, n_steps=n_steps, dt_star=dt_star):
            def body(s, _):
                ns, nkh = solver.advance_state(s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
                                                mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet,
                                                g_star, 1.0, 1.0, 1.0, 1.3, beta)
                return ns, nkh
            final, nkh = jax.lax.scan(body, state, None, length=n_steps)
            return final, jnp.sum(nkh)

        final, nkh = run(state)
        any_nan = bool(jnp.any(jnp.isnan(final['alpha_l'])) or jnp.any(jnp.isnan(final['Delta'])))
        if any_nan:
            results.append("NaN")
        else:
            amin = float(jnp.min(final['alpha_l']))
            amax = float(jnp.max(final['alpha_l']))
            results.append(f"ok[{amin:.3f},{amax:.3f}]")
        print(f"      -> {results[-1]}", flush=True)

    print(f"{r['idx']:>5} {U_sl:>8.5f} {U_sg:>7.3f} {r['cf_alpha_l']:>7.4f} " + " ".join(f"{s:<10}" for s in results), flush=True)

print("\nDONE", flush=True)
