"""
Settling diagnostic: runs a condition out to a long time horizon in chunks,
tracking the outlet-window median, global alpha range, and PER-CHUNK (not
just cumulative) kh count, to see (a) whether/when the system settles to a
uniform equilibrium, and (b) whether kh accumulation saturates once settled
-- i.e. whether the equilibrium itself is permanently disc<0, independent
of any regularizer (since surface tension / Reynolds stress both vanish
identically at a spatially uniform state).

Usage: python3 _settling_diagnostic.py <fan1|fan2> <N> [n_chunks] [chunk_transit]
"""
import sys
import time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import mixtureTheoryFlow_jax_solver_water_gas as solver

CONDITIONS = {
    "fan1": dict(U_sl=0.014319654, U_sg=19.71, D=0.0508, L=19.0,
                 rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5),
    "fan2": dict(U_sl=0.010300073, U_sg=19.88, D=0.0508, L=19.0,
                 rho_l=1000.0, rho_g=2.66, mu_l=1.0e-3, mu_g=1.8e-5),
}

label = sys.argv[1]
N = int(sys.argv[2])
n_chunks = int(sys.argv[3]) if len(sys.argv) > 3 else 16
chunk_transit = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0

cond = CONDITIONS[label]
U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L = (cond['U_sl'], cond['U_sg'], cond['D'],
                                               cond['rho_l'], cond['rho_g'], cond['mu_l'],
                                               cond['mu_g'], cond['L'])
g = 9.81
Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, g)
U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
g_star, D_star = nd['g_star'], nd['D_star']
L_star = L / D
dx_star = L_star / N
dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                     mu_l_star, mu_g_star, dx_star, 1.0, 1.0, 1.0, 0.25,
                                     g=g_star, beta=0.0)
alpha_l_inlet = U_sl_star
Delta_inlet = rho_l_star - rho_g_star
state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

steps_per_chunk = int(np.ceil(chunk_transit * L_star / dt_star))

@jax.jit
def run_chunk(state):
    def body(s, _):
        ns, nkh = solver.advance_state(s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
                                        mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet,
                                        g_star, 1.0, 1.0, 1.0, 1.3, 0.0)
        return ns, nkh
    final, nkh = jax.lax.scan(body, state, None, length=steps_per_chunk)
    return final, jnp.sum(nkh)

print(f"=== {label}  N={N}  steps_per_chunk={steps_per_chunk}  alpha_l_inlet={float(alpha_l_inlet):.6f} ===")
cum_transit = 0.0
cum_kh = 0
t0 = time.time()
for c in range(n_chunks):
    state, nkh_chunk = run_chunk(state)
    nkh_chunk = int(nkh_chunk)
    cum_kh += nkh_chunk
    cum_transit += chunk_transit
    al = state['alpha_l']
    outlet_window = al[-11:-1]
    outlet_median = float(jnp.median(outlet_window))
    amax = float(jnp.max(al))
    amin = float(jnp.min(al))
    print(f"transit={cum_transit:6.1f}  outlet_median={outlet_median:.6f}  "
          f"alpha_range=[{amin:.6f},{amax:.5f}]  kh_this_chunk={nkh_chunk:>12,}  "
          f"kh_cumulative={cum_kh:>14,}  t={time.time()-t0:.1f}s")

print("DONE")
