"""
Stage 1 diagnostic (not committed to the production solver): adds the
explicit, nonlinear surface-tension (dispersive) source term directly to
the existing explicit SSP-RK2 transport step, on top of Stage 0's
Reynolds-stress term (both independently toggleable) -- still no implicit
solve, still no change to mixtureTheoryFlow_jax_solver_water_gas.py.

Per CLAUDE.md's dispersion-analysis discussion: the earlier claim that
explicit treatment of this term is "unconditionally unstable under SSP-RK2"
was a walk-back candidate -- that conclusion came from analyzing the
dispersive term in ISOLATION (a bare, purely-imaginary eigenvalue), but the
actual coupled (alpha_l, Delta) system's large-k eigenvalue develops a
large, genuinely REAL part (confirmed numerically in the "ST only" column
of the dispersion decomposition), which sits well inside SSP-RK2's
ordinary stable region -- not on the imaginary axis at all, except possibly
in a narrow band right at the growth/decay crossing wavenumber itself. This
script is the empirical test of whether that band is actually a problem in
practice, rather than resolving it analytically.

Surface tension needs a 5-point stencil (2 ghost cells each side) for
d^3(alpha_l)/dx^3 -- wider than anything else in this file. The boundary
treatment used here (2 ghost cells extending the existing single-ghost
convention: inlet value repeated, outlet value repeated) is the simplest
defensible choice for a Stage-1 explicit test, not a final design.

Surface tension coefficient: G = sigma * (dh_l/dalpha_l), with
dh_l/dalpha_l = A/S_i evaluated LOCALLY at each cell's own current alpha_l
(not frozen at a base state -- unlike the earlier linear stability
analysis, this is the real nonlinear forward run). Nondimensionalized as
sigma_star = sigma / (rho_l * Q^2 * D) -- the same P_c*L_c pressure-length
scale compute_nondim_scales already uses elsewhere, confirmed dimensionally
(sigma has units N/m = Pa*m = kg/s^2, matching P_c*L_c exactly); A_star/
S_i_star (from td_geometry at D_star=1.0) are already correctly
nondimensionalized with no further scaling needed.
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import mixtureTheoryFlow_jax_solver_water_gas as solver
from _reynolds_stress_explicit_test import _rs_source_interior, NU_T_DEFAULT, C_EPS_DEFAULT


def _st_source_interior(alpha_l, dx, D, sigma_star):
    """
    Explicit, nonlinear surface-tension source for Delta's interior cells:
    S_ST = sigma_star * (A_star/S_i_star) * d^3(alpha_l)/dx_star^3.
    See module docstring for the stencil/boundary/nondimensionalization
    choices.
    """
    N = alpha_l.shape[0]
    alpha_ext = jnp.concatenate([
        alpha_l[0:1], alpha_l[0:1], alpha_l, alpha_l[-1:], alpha_l[-1:]
    ])  # length N+4, 2 ghost cells each side

    d3 = (-alpha_ext[1:N - 1] + 2.0 * alpha_ext[2:N]
          - 2.0 * alpha_ext[4:N + 2] + alpha_ext[5:N + 3]) / (2.0 * dx ** 3)

    alpha_int = alpha_l[1:-1]
    ht_int = solver._newton_ht_from_alpha(alpha_int, D)
    A_int, A_l_int, A_g_int, S_l_int, S_g_int, S_i_int = solver.td_geometry(ht_int, D)
    G_int = sigma_star * A_int / (S_i_int + 1e-9)

    return G_int * d3


def _transport_only_step_reg(state, dt, dx, Q, D, rho_l, rho_g,
                              alpha_l_inlet, Delta_inlet,
                              nu_l_eff_star, nu_g_eff_star, sigma_star,
                              g=9.81, theta=1.3, beta=0.0):
    """
    Exact copy of solver._transport_only_step, with two additions: the
    explicit Reynolds-stress and surface-tension source terms added to
    Delta's interior update (marked below). sigma_star=0 / nu_*_eff_star=0
    disables each independently.
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

    S_RS = _rs_source_interior(alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff_star, nu_g_eff_star)
    S_ST = _st_source_interior(alpha_l, dx, D, sigma_star)

    alpha_l_new_int = alpha_l[1:-1] - dt * div_alpha
    Delta_new_int = Delta[1:-1] - dt * div_Delta + dt * S_RS + dt * S_ST   # <-- the two additions

    alpha_l_full = jnp.concatenate([
        jnp.array([alpha_l_inlet]), alpha_l_new_int, alpha_l_new_int[-1:]
    ])
    Delta_full = jnp.concatenate([
        jnp.array([Delta_inlet]), Delta_new_int, Delta_new_int[-1:]
    ])

    n_kh_faces = jnp.sum(face_is_complex)
    return {'alpha_l': alpha_l_full, 'Delta': Delta_full}, n_kh_faces


def _ssp_rk2_transport_reg(state, dt, dx, Q, D, rho_l, rho_g,
                            alpha_l_inlet, Delta_inlet,
                            nu_l_eff_star, nu_g_eff_star, sigma_star,
                            g=9.81, theta=1.3, beta=0.0):
    stage1, kh1 = _transport_only_step_reg(
        state, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet,
        nu_l_eff_star, nu_g_eff_star, sigma_star, g, theta, beta
    )
    stage2, kh2 = _transport_only_step_reg(
        stage1, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet,
        nu_l_eff_star, nu_g_eff_star, sigma_star, g, theta, beta
    )
    new_state = jax.tree.map(lambda a, b: 0.5 * (a + b), state, stage2)
    return new_state, kh1 + kh2


def advance_state_reg(state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
                       alpha_l_inlet, Delta_inlet,
                       nu_l_eff_star, nu_g_eff_star, sigma_star,
                       g=9.81, f_i_mult=1.0, k_wl=1.0, k_wg=1.0, theta=1.3, beta=0.0):
    """Exact copy of solver.advance_state's Strang split, transport swapped for the regularized version."""
    alpha_l = state['alpha_l']
    Delta = state['Delta']

    Delta_int_1 = solver._implicit_friction_half_step(
        alpha_l[1:-1], Delta[1:-1], Q, D, rho_l, rho_g, mu_l, mu_g, dt / 2.0,
        f_i_mult, k_wl, k_wg, g, beta
    )
    Delta_1 = jnp.concatenate([Delta[0:1], Delta_int_1, Delta_int_1[-1:]])
    state_1 = {'alpha_l': alpha_l, 'Delta': Delta_1}

    state_2, n_kh = _ssp_rk2_transport_reg(
        state_1, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet,
        nu_l_eff_star, nu_g_eff_star, sigma_star, g, theta, beta
    )

    alpha_l_2, Delta_2 = state_2['alpha_l'], state_2['Delta']
    Delta_int_3 = solver._implicit_friction_half_step(
        alpha_l_2[1:-1], Delta_2[1:-1], Q, D, rho_l, rho_g, mu_l, mu_g, dt / 2.0,
        f_i_mult, k_wl, k_wg, g, beta
    )
    Delta_final = jnp.concatenate([Delta_2[0:1], Delta_int_3, Delta_int_3[-1:]])

    return {'alpha_l': alpha_l_2, 'Delta': Delta_final}, n_kh


def run_real_data_condition_reg(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, sigma, N=150,
                                 n_transit_times=100.0, cfl=0.25, g=9.81, f_i_mult=1.0,
                                 k_wl=1.0, k_wg=1.0, theta=1.3, outlet_window=10, beta=0.0,
                                 C_eps=C_EPS_DEFAULT, nu_t=NU_T_DEFAULT,
                                 enable_rs=False, enable_st=True):
    """
    Exact copy of solver.run_real_data_condition's structure, with
    advance_state replaced by advance_state_reg. enable_rs=False,
    enable_st=False together must reproduce solver.run_real_data_condition
    exactly -- see the regression check in __main__.
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

    sigma_star = (sigma / (rho_l * Q ** 2 * D)) if enable_st else 0.0

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
        new_state, n_kh_faces = advance_state_reg(
            state, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
            mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet_star,
            nu_l_eff_star, nu_g_eff_star, sigma_star, g_star, f_i_mult, k_wl, k_wg, theta, beta
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
            rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5, sigma=0.072)),
        ("Fan#2 (kh-unstable, U_sg=19.88)", dict(
            U_sl=0.010300073, U_sg=19.88, D=0.0508, L=19.0,
            rho_l=1000.0, rho_g=2.66, mu_l=1.0e-3, mu_g=1.8e-5, sigma=0.072)),
        ("Synthetic Vuong (alpha_l0~0.55)", dict(
            U_sl=0.65, U_sg=2.8, D=0.1557, L=85.0,
            rho_l=760.0, rho_g=20.0, mu_l=1.35e-3, mu_g=1.8e-5, sigma=0.024)),
    ]

    for label, cond in conditions:
        print(f"=== {label} ===")

        # regression check: both disabled must match the production solver exactly
        prod_cond = {k: v for k, v in cond.items() if k != 'sigma'}
        baseline_new = run_real_data_condition_reg(**cond, enable_rs=False, enable_st=False)
        baseline_prod = solver.run_real_data_condition(**prod_cond)
        match = (baseline_new['kh_total'] == baseline_prod['kh_total'] and
                 (abs(baseline_new['alpha_l_pred'] - baseline_prod['alpha_l_pred']) < 1e-10
                  or (np.isnan(baseline_new['alpha_l_pred']) and np.isnan(baseline_prod['alpha_l_pred']))))
        print(f"  regression check (both disabled vs production solver): {'OK' if match else 'MISMATCH!!'}")

        r_base = run_real_data_condition_reg(**cond, enable_rs=False, enable_st=False)
        r_st = run_real_data_condition_reg(**cond, enable_rs=False, enable_st=True)
        r_rs = run_real_data_condition_reg(**cond, enable_rs=True, enable_st=False)
        r_both = run_real_data_condition_reg(**cond, enable_rs=True, enable_st=True)

        for name, r in [("baseline (neither)", r_base), ("ST only", r_st),
                        ("RS only", r_rs), ("ST+RS", r_both)]:
            print(f"  {name:20s}: alpha_l_pred={r['alpha_l_pred']:.5f}  "
                  f"kh_total={r['kh_total']:>14,}  any_nan={r['any_nan']!s:5}  "
                  f"alpha_range=[{r['alpha_min']:.5f},{r['alpha_max']:.5f}]")
        print()
