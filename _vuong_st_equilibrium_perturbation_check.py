"""
Is the closed-form equilibrium for idx=797 (inclined Vuong ST row) a state the dynamic
solver can HOLD?  Start from the uniform closed-form state plus a tiny random perturbation
and watch whether the perturbation decays (stable) or grows into waves (unstable).

Two variants, because the real inlet BC pins alpha_l to the no-slip value U_sl/Q (~0.018),
NOT the equilibrium value (~0.47), so it injects an entrance layer by itself:
  A: real inlet BC     -- what the production solver does
  B: inlet pinned to the equilibrium state -- isolates stability of the uniform state itself

Runs at 0.25x CFL dt (the largest dt that avoided NaN), same horizon as the earlier checks
(10 nominal transit times, L/Q -- NOT settled for the slow liquid; see the printed liquid
residence time). Prints; saves vuong_st_eq_perturbation_check.pkl for plotting.
"""
import math, pickle
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N, IDX, MULT, N_TRANSIT, CHUNK = 150, 797, 0.25, 10.0, 200
AMP = 1e-4                       # perturbation amplitude on alpha_l (absolute)
BETA, L_VUONG = math.radians(2.0), 85.0

st = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
row = st[(st['Fluids'] == 'Nitrogen - Isopar L') & (st['Flow Pattern'] == 'ST')].loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]']); rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, 0.0)
usl, usg = U_sl / Q, U_sg / Q
rl, rg, ml, mg, gs, Ds = (nd[k] for k in ('rho_l_star', 'rho_g_star', 'mu_l_star', 'mu_g_star', 'g_star', 'D_star'))
Ls = L_VUONG / D; dx = Ls / N
dt = MULT * solver.estimate_stable_dt(usl, usg, Ds, rl, rg, ml, mg, dx, g=gs, beta=BETA)
n_steps = int(np.ceil(N_TRANSIT * Ls / dt))

# closed-form equilibrium (default closure), converted to the reduced state (alpha_l, Delta)
ht = float(solver.solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 1.0, 1.0, 1.0, beta=BETA))
A, A_l, *_ = solver.td_geometry(ht, D)
a_eq = float(A_l / A)
u_l_eq, u_g_eq = usl / a_eq, usg / (1 - a_eq)           # superficial = alpha * actual velocity
Delta_eq = rl * u_l_eq - rg * u_g_eq
print(f"idx={IDX} closed-form alpha_l={a_eq:.4f}  Delta_eq={Delta_eq:.4f}  steps={n_steps}", flush=True)
print(f"liquid residence time ~ {Ls / u_l_eq:.0f} nondim vs. run length {N_TRANSIT * Ls:.0f}  "
      f"(run covers {N_TRANSIT * Ls / (Ls / u_l_eq):.2f} liquid residence times)", flush=True)

@jax.jit
def count_complex(s):
    _, _, c = solver.local_wave_speeds(s['alpha_l'], s['Delta'], 1.0, Ds, rl, rg, gs, BETA)
    return jnp.sum(c)

rng = np.random.default_rng(0)
noise = rng.standard_normal(N)
out = {"a_eq": a_eq}
for name, a_in, D_in in (("A_real_inlet", usl, rl - rg),                 # same inlet values the production run uses
                         ("B_inlet_at_equilibrium", a_eq, Delta_eq)):
    # NB: production inlet Delta is (rho_l - rho_g)*Q with Q=1 in nondim units
    @jax.jit
    def run_chunk(s, a_in=a_in, D_in=D_in):
        def body(st_, _):
            ns, nkh = solver.advance_state(st_, dt, dx, 1.0, Ds, rl, rg, ml, mg, a_in, D_in, gs,
                                           beta=BETA, sigma=0.0)
            return ns, nkh
        f, _ = jax.lax.scan(body, s, None, length=CHUNK)
        return f
    state = {'alpha_l': jnp.asarray(a_eq + AMP * noise), 'Delta': jnp.full((N,), Delta_eq)}
    print(f"\n{name}: uniform-state complex cells at t=0: {int(count_complex(state))}/{N}", flush=True)
    print(f"{'step':>7} {'max|a-a_eq|':>12} {'mean a':>8} {'a_min':>7} {'a_max':>7} {'n_cplx':>7}", flush=True)
    tr = []
    for i in range(n_steps // CHUNK + 1):
        a = np.asarray(state['alpha_l'])
        if np.isnan(a).any():
            print(f"{i * CHUNK:>7} NaN", flush=True); tr.append((i * CHUNK, *(np.nan,) * 5)); break
        rec = (i * CHUNK, np.max(np.abs(a - a_eq)), a.mean(), a.min(), a.max(), int(count_complex(state)))
        tr.append(rec)
        if i % 10 == 0:
            print(f"{rec[0]:>7} {rec[1]:12.2e} {rec[2]:8.4f} {rec[3]:7.4f} {rec[4]:7.4f} {rec[5]:7d}", flush=True)
        state = run_chunk(state); jax.block_until_ready(state)
    out[name] = dict(trace=np.array(tr), final_alpha=np.asarray(state['alpha_l']))
pickle.dump(out, open("vuong_st_eq_perturbation_check.pkl", "wb"))
print("DONE", flush=True)
