"""
Sanity + timing check for the new joint-implicit surface-tension
half-step with sigma > 0 (not just the sigma=0 no-op case already
regression-tested separately):
  1. Confirm a short run with real Vuong sigma (0.024 N/m) doesn't NaN
     or raise (singular Jacobian, etc.) on an ordinary low-holdup
     condition -- a basic smoke test before touching any of the actual
     high-holdup Vuong ST rows that motivated this work.
  2. Measure the actual per-step wall-clock cost increase from the new
     half-steps (dense jacfwd + linalg.solve, 3 Newton iterations, x2
     half-steps per full timestep) at a production-ish grid size
     (N=150), against the pre-existing baseline -- this is the number
     the runtime discussion in CLAUDE.md needs before deciding whether
     the sparsity-exploiting (banded/colored) Jacobian optimization is
     actually necessary.
"""
import time

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import mixtureTheoryFlow_jax_solver_water_gas as solver

# Vuong-like condition (oil Isopar L - nitrogen, sigma=0.024 N/m), modest
# holdup so the basic smoke test isn't also fighting a known-unstable case.
U_sl, U_sg = 0.10030, 5.503
D, rho_l, rho_g = 0.156, 760.0, 24.2
mu_l, mu_g = 1.35e-3, 1.5e-5
L = 85.0
SIGMA = 0.024

print("=== smoke test: sigma=0.024, N=60, short run ===", flush=True)
r = solver.run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L,
                                    N=60, n_transit_times=5.0, sigma=SIGMA)
for k, v in r.items():
    print(f"  {k}: {v}")

print(flush=True)
print("=== timing: advance_state per-step cost, N=150 ===", flush=True)
N = 150
Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, SIGMA)
U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
g_star, D_star, sigma_star = nd['g_star'], nd['D_star'], nd['sigma_star']
L_star = L / D
dx_star = L_star / N
dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                     mu_l_star, mu_g_star, dx_star, g=g_star)
alpha_l_inlet = U_sl_star
Delta_inlet = rho_l_star - rho_g_star
state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

N_STEPS = 2000

def make_runner(sigma_val):
    @jax.jit
    def run(state):
        def body(s, _):
            ns, nkh = solver.advance_state(
                s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
                mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet, g_star,
                sigma=sigma_val
            )
            return ns, nkh
        final, nkh = jax.lax.scan(body, state, None, length=N_STEPS)
        return final, jnp.sum(nkh)
    return run

for label, sigma_val in [("sigma=0 (baseline)", 0.0), ("sigma=sigma_star (new)", sigma_star)]:
    run = make_runner(sigma_val)
    final, nkh = run(state)  # warm up / compile
    jax.block_until_ready(final)
    t0 = time.time()
    final, nkh = run(state)
    jax.block_until_ready(final)
    t1 = time.time()
    per_step_us = (t1 - t0) / N_STEPS * 1e6
    print(f"  {label:28s}: {N_STEPS} steps in {t1-t0:.4f}s  -> {per_step_us:.2f} us/step, "
          f"any_nan={bool(jnp.any(jnp.isnan(final['alpha_l'])))}", flush=True)
