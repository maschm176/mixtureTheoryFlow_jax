"""
Digging into what in the closure actually drives disc = trace^2/4 - det
negative (local_wave_speeds' is_complex trigger).

Step 1: hand-derive the flux Jacobian entries analytically from
physical_flux/recover_velocities/h_l_of_alpha, and verify them against
jax.jacfwd at an arbitrary state -- a correctness check before trusting
the decomposition below (per this project's "verify before concluding"
norm).

Hand derivation (du_l/dalpha, du_g/dalpha, du_l/dDelta, du_g/dDelta from
differentiating recover_velocities' defining linear system directly,
Delta=rho_l*u_l-rho_g*u_g and Q=alpha*u_l+(1-alpha)*u_g, w.r.t. alpha and
Delta at fixed Q):

    denom = rho_l*(1-alpha) + rho_g*alpha
    du = u_g - u_l   (slip, gas minus liquid)

    du_l/dalpha  = rho_g*du/denom
    du_g/dalpha  = rho_l*du/denom
    du_l/dDelta  = (1-alpha)/denom
    du_g/dDelta  = -alpha/denom

Flux Jacobian entries (flux_alpha=alpha*u_l,
flux_Delta=0.5*rho_l*u_l^2-0.5*rho_g*u_g^2+(rho_l-rho_g)*g*cos(beta)*h_l(alpha)):

    dFa/dalpha = u_l + alpha*rho_g*du/denom
    dFa/dDelta = alpha*(1-alpha)/denom
    dFd/dalpha = -rho_l*rho_g*du**2/denom + (rho_l-rho_g)*g*cos(beta)*(A/S_i)
    dFd/dDelta = (rho_l*(1-alpha)*u_l + rho_g*alpha*u_g)/denom

dFd/dalpha is the interesting one: a NEGATIVE "Bernoulli-suction" piece
that grows with the SQUARE of the slip velocity, competing directly
against a POSITIVE "gravity-restoring" piece from the h_l(alpha) term
(dh_l/dalpha = A/S_i, from h_l_of_alpha's own docstring). This is the
closed-form version of the physical-mechanism discussion already in
CLAUDE.md/the module docstring ("Bernoulli-suction destabilizing term
vs. gravity-restoring stabilizing term").

Step 2: pull an actual unstable state out of a real run (idx=797, the
same fast-NaN Vuong ST condition used in the is_complex-vs-NaN timing
check) and decompose disc cell-by-cell for the flagged (is_complex=True)
cells, to see numerically whether it really is the Bernoulli term
overtaking the gravity term that flips the sign, and by how much.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

# -----------------------------------------------------------------------
# Step 1: verify the hand-derived Jacobian against jax.jacfwd
# -----------------------------------------------------------------------
rng = np.random.default_rng(0)
Q, D, rho_l, rho_g, g, beta = 1.0, 0.156, 760.0, 24.2, 9.81, math.radians(2.0)

def hand_jacobian(alpha, Delta):
    alpha_g = 1.0 - alpha
    denom = rho_l * alpha_g + rho_g * alpha
    u_l, u_g = solver.recover_velocities(alpha, Delta, Q, rho_l, rho_g)
    du = u_g - u_l

    dul_dalpha = rho_g * du / denom
    dug_dalpha = rho_l * du / denom
    dul_dDelta = alpha_g / denom
    dug_dDelta = -alpha / denom

    ht = solver._newton_ht_from_alpha(jnp.array([alpha]), D)[0]
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D)
    dh_dalpha = A / S_i

    dFa_dalpha = u_l + alpha * dul_dalpha
    dFa_dDelta = alpha * dul_dDelta
    dFd_dalpha = (-rho_l * rho_g * du**2 / denom
                  + (rho_l - rho_g) * g * math.cos(beta) * dh_dalpha)
    dFd_dDelta = (rho_l * alpha_g * u_l + rho_g * alpha * u_g) / denom
    return dFa_dalpha, dFa_dDelta, dFd_dalpha, dFd_dDelta

print("=== Step 1: hand-derived vs jax.jacfwd Jacobian, random states ===", flush=True)
max_rel_err = 0.0
for _ in range(10):
    alpha = float(rng.uniform(0.05, 0.9))
    Delta = float(rng.uniform(-500.0, 500.0))

    jac_fn = jax.jacfwd(solver.physical_flux, argnums=(0, 1))
    (dFa_da, dFa_dD), (dFd_da, dFd_dD) = jac_fn(
        jnp.array(alpha), jnp.array(Delta), Q, D, rho_l, rho_g, g, beta
    )
    hand = hand_jacobian(alpha, Delta)
    auto = (float(dFa_da), float(dFa_dD), float(dFd_da), float(dFd_dD))
    rel_errs = [abs(h - a) / (abs(a) + 1e-12) for h, a in zip(hand, auto)]
    max_rel_err = max(max_rel_err, max(rel_errs))
    print(f"  alpha={alpha:.3f} Delta={Delta:8.2f}  max_rel_err={max(rel_errs):.2e}", flush=True)

print(f"\n  overall max_rel_err = {max_rel_err:.2e}  ({'PASS' if max_rel_err < 1e-8 else 'FAIL'})\n", flush=True)

# -----------------------------------------------------------------------
# Step 2: decompose disc at an actual unstable state from idx=797
# -----------------------------------------------------------------------
print("=== Step 2: decomposition at idx=797's actual unstable state ===", flush=True)

N = 150
IDX = 797
STEP_TARGET = 400  # well inside the persistently-is_complex regime per the earlier timing check

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]

row = st.loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D_c = float(row['Pipe Diameter[m]'])
rho_l_c, rho_g_c = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l_c, mu_g_c = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
L_VUONG = 85.0

Q_c = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D_c, rho_l_c, rho_g_c, mu_l_c, mu_g_c, 9.81, 0.0)
U_sl_star, U_sg_star = U_sl / Q_c, U_sg / Q_c
rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
g_star, D_star = nd['g_star'], nd['D_star']
L_star = L_VUONG / D_c
dx_star = L_star / N

dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                     mu_l_star, mu_g_star, dx_star, g=g_star, beta=beta)

alpha_l_inlet = U_sl_star
Delta_inlet_star = rho_l_star - rho_g_star
state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

from functools import partial

@partial(jax.jit, static_argnums=(1,))
def run_chunk(state, n):
    def body(s, _):
        ns, nkh = solver.advance_state(
            s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet_star, g_star,
            beta=beta, sigma=0.0
        )
        return ns, nkh
    final, nkh = jax.lax.scan(body, state, None, length=n)
    return final, jnp.sum(nkh)

state, _ = run_chunk(state, STEP_TARGET)
jax.block_until_ready(state)
any_nan = bool(jnp.any(jnp.isnan(state['alpha_l'])) or jnp.any(jnp.isnan(state['Delta'])))
print(f"  after {STEP_TARGET} steps: any_nan={any_nan}", flush=True)

alpha_arr = np.array(state['alpha_l'])
Delta_arr = np.array(state['Delta'])

lam1, lam2, is_complex = solver.local_wave_speeds(
    state['alpha_l'], state['Delta'], 1.0, D_star, rho_l_star, rho_g_star, g_star, beta
)
is_complex = np.array(is_complex)
n_complex = int(is_complex.sum())
print(f"  cells with is_complex=True: {n_complex} / {N}", flush=True)

flagged_idx = np.where(is_complex)[0]
print(f"\n  {'i':>4} {'alpha_l':>8} {'Delta':>9} {'u_l':>9} {'u_g':>9} {'du=ug-ul':>9} "
      f"{'bernoulli':>11} {'gravity':>11} {'dFd/dalpha':>11} {'trace':>9} {'det':>10} {'disc':>10}", flush=True)

for i in flagged_idx[:15]:  # first 15 flagged cells is plenty to see the pattern
    alpha = float(alpha_arr[i])
    Delta = float(Delta_arr[i])
    u_l, u_g = solver.recover_velocities(jnp.array(alpha), jnp.array(Delta), 1.0, rho_l_star, rho_g_star)
    u_l, u_g = float(u_l), float(u_g)
    du = u_g - u_l
    alpha_g = 1.0 - alpha
    denom = rho_l_star * alpha_g + rho_g_star * alpha

    ht = float(solver._newton_ht_from_alpha(jnp.array([alpha]), D_star)[0])
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, D_star)
    dh_dalpha = float(A / S_i)

    bernoulli = -rho_l_star * rho_g_star * du**2 / denom
    gravity = (rho_l_star - rho_g_star) * g_star * math.cos(beta) * dh_dalpha
    dFd_dalpha = bernoulli + gravity

    jac_fn = jax.jacfwd(solver.physical_flux, argnums=(0, 1))
    (dFa_da, dFa_dD), (dFd_da, dFd_dD) = jac_fn(
        jnp.array(alpha), jnp.array(Delta), 1.0, D_star, rho_l_star, rho_g_star, g_star, beta
    )
    trace = float(dFa_da + dFd_dD)
    det = float(dFa_da * dFd_dD - dFa_dD * dFd_da)
    disc = trace**2 / 4.0 - det

    print(f"  {i:>4} {alpha:>8.4f} {Delta:>9.3f} {u_l:>9.4f} {u_g:>9.4f} {du:>9.4f} "
          f"{bernoulli:>11.4f} {gravity:>11.4f} {dFd_dalpha:>11.4f} {trace:>9.4f} {det:>10.4f} {disc:>10.4f}",
          flush=True)

print("\nDONE", flush=True)
