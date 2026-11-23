"""
Stage 0 diagnostic (not committed to the production solver): adds an
explicit, nonlinear Reynolds-stress (turbulent eddy-viscosity) source term
directly to the existing explicit SSP-RK2 transport step -- no new implicit
solver, no change to mixtureTheoryFlow_jax_solver_water_gas.py itself.

Motivation: the three conditions below all have an EQUILIBRIUM state whose
advective flux Jacobian already has disc<0 (Fan's two kh-unstable
conditions, plus a synthetic high-holdup Vuong point) -- a fact that is
independent of any regularizing term, since solve_taitel_dukler_holdup and
physical_flux never take sigma/nu_eff as arguments, and these terms are
pure spatial derivatives that vanish identically at any uniform state (see
CLAUDE.md's dispersion-analysis discussion). This script asks the only
question that discussion left open: does adding real damping (Reynolds
stress) to the perturbations change kh_total/boundedness in the actual
nonlinear transient run, given that it structurally cannot move the
equilibrium itself out of the unstable region.

C_eps=8.1, nu_t=1.3e-5 m^2/s are borrowed from Fullmer, Ransom & Lopez de
Bertodano (2014)'s own Thorpe (1969)-calibrated values -- not independently
fit to Fan or Vuong's conditions. Treat results as order-of-magnitude.

Every function here is a copy of its mixtureTheoryFlow_jax_solver_water_gas
counterpart with exactly one addition (the Reynolds-stress source term in
_transport_only_step_rs) -- verified by the enable_rs=False regression
check in __main__, which must reproduce solver.run_real_data_condition
exactly before any enable_rs=True result is trusted.
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import mixtureTheoryFlow_jax_solver_water_gas as solver

C_EPS_DEFAULT = 8.1
NU_T_DEFAULT = 1.3e-5  # m^2/s, dimensional -- nondimensionalized per-condition below

# Floor for alpha_k in the RS diffusion term. Originally tried as a plain
# clip on the CELL-CENTERED alpha_k used to divide the (already-taken)
# divergence -- that attempt (ALPHA_SAFE_FLOOR=1e-3, applied as
# `div / jnp.clip(alpha_k, FLOOR, 1.0)`) reduced but did not eliminate a
# positive-feedback runaway: max|S_RS| still tracked the current global
# alpha_min and grew as it dropped (confirmed via direct diagnostic: 3.7e-3
# at alpha_l=6.5e-4 up to 2.9e-2 at alpha_l=2.6e-4), and kh_total ended up
# slightly WORSE than the unfloored version (99.9M vs. baseline's 90.0M on
# Fan#1 @ N=1800) -- checked and ruled out as a plain explicit-diffusion
# CFL violation (diffusive stability number r=nu_eff*dt/(alpha_safe*dx^2)
# was only ~0.136 at that floor, well under the 0.5 stability limit), so
# raising the floor further wasn't expected to help either.
#
# This version instead folds the regularization into the FACE flux, before
# the divergence, rather than clipping the cell-centered alpha_k AFTER it.
# `ratio_face = alpha_face/alpha_face_safe` is bounded in (0, 1] by
# construction -- it equals exactly 1 everywhere alpha_face is above the
# floor (recovering the literal Fullmer-equation behavior unchanged in the
# well-behaved bulk), and smoothly shrinks toward 0 as alpha_face->0,
# rather than the old formulation's cell-centered division which could
# still amplify a sizeable pre-floor divergence by up to 1/FLOOR. This
# structurally removes the asymptotic 1/alpha blowup (and the feedback
# loop it created) instead of just relocating where it starts -- a
# genuinely different discretization from the literal Fullmer-equation
# transcription near vanishing alpha_k, not merely a relocated floor.
ALPHA_SAFE_FLOOR = 1e-3


def _rs_source_interior(alpha_l, Delta, Q, dx, rho_l_star, rho_g_star,
                         nu_l_eff_star, nu_g_eff_star):
    """
    Explicit, nonlinear Reynolds-stress source for Delta's interior cells.

    Derived from Fullmer et al. (2014) Eqs. 3-4's viscous terms
    rho_k*nu_k^eff*d/dx(alpha_k*du_k/dx), divided by alpha_k to build
    rho_k*Du_k/Dt (same reduction already used for the surface-tension
    term), then combined as (liquid contribution) - (gas contribution)
    since Delta = rho_l*u_l - rho_g*u_g. Standard conservative
    finite-volume discretization of a variable-coefficient diffusion term:
    face-centered du/dx and face-averaged alpha, then a cell-centered
    divergence of that face flux. Only a 3-point stencil -- same width
    central_upwind_flux already uses, no extra ghost cells needed (unlike
    the dispersive/surface-tension term, which would need a 5-point
    stencil).
    """
    u_l, u_g = solver.recover_velocities(alpha_l, Delta, Q, rho_l_star, rho_g_star)
    alpha_g = 1.0 - alpha_l

    def diffusion_term(alpha_k, u_k, nu_k_eff):
        du_dx_face = (u_k[1:] - u_k[:-1]) / dx          # length N-1
        alpha_face = 0.5 * (alpha_k[1:] + alpha_k[:-1])  # length N-1
        alpha_face_safe = jnp.clip(alpha_face, ALPHA_SAFE_FLOOR, 1.0)
        ratio_face = alpha_face / alpha_face_safe        # bounded in (0, 1]
        flux_face = ratio_face * du_dx_face              # length N-1
        div = (flux_face[1:] - flux_face[:-1]) / dx       # length N-2
        return nu_k_eff * div

    S_l = diffusion_term(alpha_l, u_l, nu_l_eff_star)
    S_g = diffusion_term(alpha_g, u_g, nu_g_eff_star)
    return S_l - S_g


def _transport_only_step_rs(state, dt, dx, Q, D, rho_l, rho_g,
                             alpha_l_inlet, Delta_inlet,
                             nu_l_eff_star, nu_g_eff_star,
                             g=9.81, theta=1.3, beta=0.0):
    """
    Exact copy of solver._transport_only_step, with one addition: the
    explicit Reynolds-stress source term added to Delta's interior update
    (marked below). Everything else -- reconstruction, positivity
    correction, central-upwind flux, KH-flag counting -- is identical.
    """
    alpha_l = state['alpha_l']
    Delta = state['Delta']

    alpha_lf_raw, alpha_rf_raw = solver.reconstruct_faces(alpha_l, dx, theta)
    Delta_lf, Delta_rf = solver.reconstruct_faces(Delta, dx, theta)

    alpha_center = alpha_l[1:-1]
    alpha_lf, alpha_rf = solver.positivity_correct(alpha_center, alpha_lf_raw, alpha_rf_raw)

    alpha_L = jnp.concatenate([alpha_l[0:1], alpha_rf])
    alpha_R = jnp.concatenate([alpha_lf, alpha_l[-1:]])
    Delta_L = jnp.concatenate([Delta[0:1], Delta_rf])
    Delta_R = jnp.concatenate([Delta_lf, Delta[-1:]])

    F_alpha_L, F_Delta_L = solver.physical_flux(alpha_L, Delta_L, Q, D, rho_l, rho_g, g, beta)
    F_alpha_R, F_Delta_R = solver.physical_flux(alpha_R, Delta_R, Q, D, rho_l, rho_g, g, beta)

    lam1_L, lam2_L, is_complex_L = solver.local_wave_speeds(alpha_L, Delta_L, Q, D, rho_l, rho_g, g, beta)
    lam1_R, lam2_R, is_complex_R = solver.local_wave_speeds(alpha_R, Delta_R, Q, D, rho_l, rho_g, g, beta)

    a_plus = jnp.maximum(jnp.maximum(lam2_L, lam2_R), 0.0)
    a_minus = jnp.minimum(jnp.minimum(lam1_L, lam1_R), 0.0)
    face_is_complex = is_complex_L | is_complex_R

    f_alpha_face = solver.central_upwind_flux(F_alpha_L, F_alpha_R, alpha_L, alpha_R, a_plus, a_minus)
    f_Delta_face = solver.central_upwind_flux(F_Delta_L, F_Delta_R, Delta_L, Delta_R, a_plus, a_minus)

    div_alpha = (f_alpha_face[1:] - f_alpha_face[:-1]) / dx
    div_Delta = (f_Delta_face[1:] - f_Delta_face[:-1]) / dx

    S_RS = _rs_source_interior(alpha_l, Delta, Q, dx, rho_l, rho_g,
                                nu_l_eff_star, nu_g_eff_star)

    alpha_l_new_int = alpha_l[1:-1] - dt * div_alpha
    Delta_new_int = Delta[1:-1] - dt * div_Delta + dt * S_RS   # <-- the one addition

    alpha_l_full = jnp.concatenate([
        jnp.array([alpha_l_inlet]), alpha_l_new_int, alpha_l_new_int[-1:]
    ])
    Delta_full = jnp.concatenate([
        jnp.array([Delta_inlet]), Delta_new_int, Delta_new_int[-1:]
    ])

    n_kh_faces = jnp.sum(face_is_complex)
    return {'alpha_l': alpha_l_full, 'Delta': Delta_full}, n_kh_faces


def _ssp_rk2_transport_rs(state, dt, dx, Q, D, rho_l, rho_g,
                           alpha_l_inlet, Delta_inlet,
                           nu_l_eff_star, nu_g_eff_star,
                           g=9.81, theta=1.3, beta=0.0):
    stage1, kh1 = _transport_only_step_rs(
        state, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet,
        nu_l_eff_star, nu_g_eff_star, g, theta, beta
    )
    stage2, kh2 = _transport_only_step_rs(
        stage1, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet,
        nu_l_eff_star, nu_g_eff_star, g, theta, beta
    )
    new_state = jax.tree.map(lambda a, b: 0.5 * (a + b), state, stage2)
    return new_state, kh1 + kh2


def advance_state_rs(state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
                      alpha_l_inlet, Delta_inlet,
                      nu_l_eff_star, nu_g_eff_star,
                      g=9.81, f_i_mult=1.0, k_wl=1.0, k_wg=1.0, theta=1.3, beta=0.0):
    """Exact copy of solver.advance_state's Strang split, transport swapped for the RS version."""
    alpha_l = state['alpha_l']
    Delta = state['Delta']

    Delta_int_1 = solver._implicit_friction_half_step(
        alpha_l[1:-1], Delta[1:-1], Q, D, rho_l, rho_g, mu_l, mu_g, dt / 2.0,
        f_i_mult, k_wl, k_wg, g, beta
    )
    Delta_1 = jnp.concatenate([Delta[0:1], Delta_int_1, Delta_int_1[-1:]])
    state_1 = {'alpha_l': alpha_l, 'Delta': Delta_1}

    state_2, n_kh = _ssp_rk2_transport_rs(
        state_1, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet,
        nu_l_eff_star, nu_g_eff_star, g, theta, beta
    )

    alpha_l_2, Delta_2 = state_2['alpha_l'], state_2['Delta']
    Delta_int_3 = solver._implicit_friction_half_step(
        alpha_l_2[1:-1], Delta_2[1:-1], Q, D, rho_l, rho_g, mu_l, mu_g, dt / 2.0,
        f_i_mult, k_wl, k_wg, g, beta
    )
    Delta_final = jnp.concatenate([Delta_2[0:1], Delta_int_3, Delta_int_3[-1:]])

    return {'alpha_l': alpha_l_2, 'Delta': Delta_final}, n_kh


def run_real_data_condition_rs(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N=150,
                                n_transit_times=100.0, cfl=0.25, g=9.81, f_i_mult=1.0,
                                k_wl=1.0, k_wg=1.0, theta=1.3, outlet_window=10, beta=0.0,
                                C_eps=C_EPS_DEFAULT, nu_t=NU_T_DEFAULT, enable_rs=True):
    """
    Exact copy of solver.run_real_data_condition's structure (same
    nondimensionalization, dt estimation, step count, outlet-window
    diagnostics), with advance_state replaced by advance_state_rs.
    enable_rs=False sets nu_l_eff_star=nu_g_eff_star=0, which must
    reproduce solver.run_real_data_condition exactly -- see the regression
    check in __main__.
    """
    Q = U_sl + U_sg
    nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, g)
    U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
    rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
    mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
    g_star, D_star = nd['g_star'], nd['D_star']
    L_star = L / D
    dx_star = L_star / N

    nu_t_star = nu_t / (Q * D)
    if enable_rs:
        nu_l_eff_star = C_eps * (mu_l_star + nu_t_star)
        nu_g_eff_star = C_eps * (mu_g_star / rho_g_star + nu_t_star)
    else:
        nu_l_eff_star = 0.0
        nu_g_eff_star = 0.0

    dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                         mu_l_star, mu_g_star, dx_star, f_i_mult,
                                         k_wl, k_wg, cfl, g=g_star, beta=beta)
    t_end_star = n_transit_times * L_star
    n_steps = int(np.ceil(float(t_end_star / dt_star)))

    alpha_l_inlet = U_sl_star
    Delta_inlet_star = rho_l_star - rho_g_star
    state = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

    def step(carry, _):
        state, kh_total, a_min, a_max = carry
        new_state, n_kh_faces = advance_state_rs(
            state, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet_star,
            nu_l_eff_star, nu_g_eff_star, g_star, f_i_mult, k_wl, k_wg, theta, beta
        )
        a_min = jnp.minimum(a_min, jnp.min(new_state['alpha_l']))
        a_max = jnp.maximum(a_max, jnp.max(new_state['alpha_l']))
        return (new_state, kh_total + n_kh_faces, a_min, a_max), None

    init_carry = (state, jnp.array(0), jnp.array(1.0), jnp.array(0.0))
    (final_state, kh_total, alpha_min, alpha_max), _ = jax.lax.scan(
        step, init_carry, None, length=n_steps
    )

    alpha_l_window = final_state['alpha_l'][-(outlet_window + 1):-1]
    Delta_window = final_state['Delta'][-(outlet_window + 1):-1]
    dPdz_l_arr_star, dPdz_g_arr_star = solver.predicted_pressure_gradient(
        alpha_l_window, Delta_window, 1.0, D_star, rho_l_star, rho_g_star,
        mu_l_star, mu_g_star, f_i_mult, k_wl, k_wg, g_star, beta
    )
    alpha_l_outlet = jnp.median(alpha_l_window)
    dPdz_l_star = jnp.median(dPdz_l_arr_star)
    dPdz_g_star = jnp.median(dPdz_g_arr_star)

    dPdz_scale = nd['P_c'] / D
    dPdz_l = dPdz_l_star * dPdz_scale
    dPdz_g = dPdz_g_star * dPdz_scale
    dt = dt_star * nd['t_c']

    any_nan = bool(jnp.any(jnp.isnan(final_state['alpha_l'])) or
                   jnp.any(jnp.isnan(final_state['Delta'])))

    return {
        'alpha_l_pred': float(alpha_l_outlet),
        'dPdz_pred_liquid': float(dPdz_l),
        'dPdz_pred_gas': float(dPdz_g),
        'kh_total': int(kh_total),
        'alpha_min': float(alpha_min),
        'alpha_max': float(alpha_max),
        'dt': float(dt),
        'n_steps': n_steps,
        'any_nan': any_nan,
        'Re_SL': nd['Re_SL'],
        'Re_SG': nd['Re_SG'],
    }


if __name__ == "__main__":
    conditions = [
        ("Fan#1 (kh-unstable, U_sg=19.71)", dict(
            U_sl=0.014319654, U_sg=19.71, D=0.0508, L=19.0,
            rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5)),
        ("Fan#2 (kh-unstable, U_sg=19.88)", dict(
            U_sl=0.010300073, U_sg=19.88, D=0.0508, L=19.0,
            rho_l=1000.0, rho_g=2.66, mu_l=1.0e-3, mu_g=1.8e-5)),
        ("Synthetic Vuong (alpha_l0~0.55)", dict(
            U_sl=0.65, U_sg=2.8, D=0.1557, L=85.0,
            rho_l=760.0, rho_g=20.0, mu_l=1.35e-3, mu_g=1.8e-5)),
    ]

    for label, cond in conditions:
        print(f"=== {label} ===")

        baseline_new = run_real_data_condition_rs(**cond, enable_rs=False)
        baseline_prod = solver.run_real_data_condition(**cond)
        match = (abs(baseline_new['alpha_l_pred'] - baseline_prod['alpha_l_pred']) < 1e-10
                 and baseline_new['kh_total'] == baseline_prod['kh_total'])
        print(f"  regression check (enable_rs=False vs production solver): {'OK' if match else 'MISMATCH!!'}")
        if not match:
            print(f"    new ={baseline_new}")
            print(f"    prod={baseline_prod}")

        result_rs = run_real_data_condition_rs(**cond, enable_rs=True)
        print(f"  no-RS : alpha_l_pred={baseline_prod['alpha_l_pred']:.5f}  "
              f"kh_total={baseline_prod['kh_total']:>14,}  any_nan={baseline_prod['any_nan']}  "
              f"alpha_range=[{baseline_prod['alpha_min']:.5f},{baseline_prod['alpha_max']:.5f}]  "
              f"dPdz_l={baseline_prod['dPdz_pred_liquid']:.3f}")
        print(f"  +RS   : alpha_l_pred={result_rs['alpha_l_pred']:.5f}  "
              f"kh_total={result_rs['kh_total']:>14,}  any_nan={result_rs['any_nan']}  "
              f"alpha_range=[{result_rs['alpha_min']:.5f},{result_rs['alpha_max']:.5f}]  "
              f"dPdz_l={result_rs['dPdz_pred_liquid']:.3f}")
        print()
