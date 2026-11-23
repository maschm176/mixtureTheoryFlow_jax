"""
Profile + trace data for Fan (2005) inclined row 0 (U_sl=0.0241, U_sg=19.97, -1 deg), one of the
9 kh_total-flagged rows. Steps the production path (solver.advance_state, N=150, default closure,
default CFL) in chunks to 150 nominal transits and saves, per chunk: transit time, outlet-window
alpha_l (median of the last 10 interior cells, the same quantity run_real_data_condition reports),
complex-eigenvalue cell count, kh faces in the chunk, alpha_min/max; plus full alpha_l(x) snapshots
at chosen transit times. Regression check: the final outlet alpha_l should reproduce the wrapper's
0.02898 at 150 transits (printed). Output: fan_incl_row0_trace.pkl
"""
import math, pickle, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N, ROW, N_TRANSIT, CHUNK = 150, 0, 150.0, 500
SNAP_T = [0.5, 2, 5, 10, 25, 50, 100, 150]
r = pd.read_csv("fan2005_sw_subset.csv").iloc[ROW]
U_sl, U_sg = float(r['vSL[m/s]']), float(r['vSG[m/s]'])
D, L = float(r['Pipe Diameter[m]']), float(r['Pipe Length[m]'])
rho_l, rho_g = float(r['rhoL[kg/m3]']), float(r['rhoG[kg/m3]'])
mu_l, mu_g = float(r['viscosityL[Pa.s]']), float(r['viscosityG[Pa.s]'])
beta = math.radians(float(r['Inclination angle[°]']))

Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, 0.0)
usl, usg = U_sl / Q, U_sg / Q
rl, rg, ml, mg, gs, Ds = (nd[k] for k in ('rho_l_star', 'rho_g_star', 'mu_l_star', 'mu_g_star', 'g_star', 'D_star'))
Ls = L / D; dx = Ls / N
dt = float(solver.estimate_stable_dt(usl, usg, Ds, rl, rg, ml, mg, dx, 1.0, 1.0, 1.0, 0.25, g=gs, beta=beta))
n_steps = int(np.ceil(N_TRANSIT * Ls / dt))
print(f"row {ROW}: dt={dt:.4e} n_steps={n_steps} (wrapper reported 90232 for 150 transits)", flush=True)

@jax.jit
def run_chunk(s):
    def body(st, _):
        ns, nkh = solver.advance_state(st, dt, dx, 1.0, Ds, rl, rg, ml, mg, usl, rl - rg, gs,
                                       beta=beta, sigma=0.0)
        return ns, nkh
    f, nkh = jax.lax.scan(body, s, None, length=CHUNK)
    return f, jnp.sum(nkh)

@jax.jit
def count_complex(s):
    _, _, c = solver.local_wave_speeds(s['alpha_l'], s['Delta'], 1.0, Ds, rl, rg, gs, beta)
    return jnp.sum(c)

snap_steps = sorted({int(round(t * Ls / dt / CHUNK)) * CHUNK for t in SNAP_T})
state = solver.init_state_water_gas(N, usl, usg, rl, rg)
tr = dict(transit=[], outlet=[], n_complex=[], kh_chunk=[], a_min=[], a_max=[], snaps={},
          dt=dt, Ls=Ls, n_steps=n_steps)
t0 = time.time()
for i in range(n_steps // CHUNK + 1):
    a = np.asarray(state['alpha_l'])
    step = i * CHUNK
    tr['transit'].append(step * dt / Ls)
    tr['outlet'].append(float(np.median(a[-11:-1])))
    tr['n_complex'].append(int(count_complex(state)))
    tr['a_min'].append(float(a.min())); tr['a_max'].append(float(a.max()))
    if step in snap_steps:
        tr['snaps'][step * dt / Ls] = a.copy()
    if i == n_steps // CHUNK:
        break
    state, nkh = run_chunk(state); jax.block_until_ready(state)
    tr['kh_chunk'].append(int(nkh))
    if i % 20 == 0:
        print(f"  step {step:>6} transit={tr['transit'][-1]:6.1f} outlet={tr['outlet'][-1]:.5f} "
              f"n_cplx={tr['n_complex'][-1]:>3} {time.time() - t0:.0f}s", flush=True)
print(f"final outlet alpha_l = {tr['outlet'][-1]:.5f} at transit {tr['transit'][-1]:.1f} "
      f"(wrapper at 150 transits: 0.02898)", flush=True)
pickle.dump(tr, open("fan_incl_row0_trace.pkl", "wb"))
print("DONE", flush=True)
