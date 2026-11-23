"""
Stage 2: implicit/IMEX treatment of the Reynolds-stress diffusion term.

Motivation (see the explicit-RS investigation in _reynolds_stress_explicit_
test.py and the CLAUDE.md-adjacent discussion this session): explicit
(forward-Euler-style) integration of the RS diffusion term has a hard
amplification threshold -- the single-mode amplification factor is
g_explicit = 1 - dt*lambda, which flips from damping to literal unbounded
growth once dt*lambda > 2, where lambda ~ 4*nu_eff/(alpha_k*dx^2). Because
nu_eff/alpha_k is solution-dependent and swings over orders of magnitude as
alpha_l dips toward the floor, no single dt chosen for the ADVECTIVE CFL
(all estimate_stable_dt knows about) is safe everywhere. Implicit (backward-
Euler) treatment has amplification factor g_implicit = 1/(1+dt*lambda),
which is in (0,1] for ANY lambda>=0 and ANY dt>0 -- no threshold, provably,
via a discrete negative-semi-definiteness argument on the conservative
flux-divergence operator (positive diagonal rescaling by 1/alpha_k preserves
real, non-positive eigenvalues through a similarity transform).

Key structural simplification that makes this tractable without a nonlinear
solve: recover_velocities gives u_l, u_g as EXACTLY AFFINE functions of
Delta when alpha_l is held fixed (frozen, same convention as the existing
_implicit_friction_half_step). That makes the RS source S_RS(Delta; alpha_l
frozen) an affine (linear + constant) function of the interior Delta
vector, with a TRIDIAGONAL Jacobian (each cell's S_RS only depends on its
own and its immediate neighbors' velocities). So the implicit backward-
Euler update is a single, exact tridiagonal linear solve -- no Newton
iteration needed, unlike friction's per-cell nonlinear solve.

The tridiagonal coefficients below were derived by hand from
recover_velocities' closed form and the face-flux-regularized diffusion
operator in _reynolds_stress_explicit_test.py's _rs_source_interior (same
ALPHA_SAFE_FLOOR regularization reused, imported not reimplemented). Given
the real risk of an algebra/indexing slip in a derivation this fiddly, the
__main__ validation block checks the hand-derived bands against a brute-
force jax.jacfwd of the EXISTING, already-trusted _rs_source_interior at
small N before anything here is trusted at the real problem's scale (where
a dense N x N Jacobian every step would be O(N^2) per step -- far too
expensive to use directly in the scanned simulation loop; it's a
validation tool only, not part of the fast path).
"""
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import mixtureTheoryFlow_jax_solver_water_gas as solver
from _reynolds_stress_explicit_test import (
    C_EPS_DEFAULT, NU_T_DEFAULT, ALPHA_SAFE_FLOOR, _rs_source_interior,
)


# =============================================================================
# THOMAS ALGORITHM -- O(N) tridiagonal solver, jax.lax.scan based
# =============================================================================
def thomas_solve(sub, main, super_, rhs):
    """
    Solves A x = rhs for a tridiagonal A with sub-diagonal `sub` (length
    n-1; sub[i] is the entry at row i+1, column i), main diagonal `main`
    (length n), and super-diagonal `super_` (length n-1; super_[i] is the
    entry at row i, column i+1). Standard forward-elimination / back-
    substitution Thomas algorithm.
    """
    n = main.shape[0]
    sub_padded = jnp.concatenate([jnp.zeros(1), sub])        # length n, [0] unused
    super_padded = jnp.concatenate([super_, jnp.zeros(1)])   # length n, [-1] unused

    cp0 = super_padded[0] / main[0]
    dp0 = rhs[0] / main[0]

    def fwd(carry, x):
        cp_prev, dp_prev = carry
        sub_i, main_i, super_i, rhs_i = x
        m = main_i - sub_i * cp_prev
        cp_i = super_i / m
        dp_i = (rhs_i - sub_i * dp_prev) / m
        return (cp_i, dp_i), (cp_i, dp_i)

    (_, _), (cp_rest, dp_rest) = jax.lax.scan(
        fwd, (cp0, dp0),
        (sub_padded[1:], main[1:], super_padded[1:], rhs[1:])
    )
    cp = jnp.concatenate([cp0[None], cp_rest])
    dp = jnp.concatenate([dp0[None], dp_rest])

    def back(x_next, x):
        cp_i, dp_i = x
        x_i = dp_i - cp_i * x_next
        return x_i, x_i

    x_last = dp[-1]
    _, x_rest_rev = jax.lax.scan(back, x_last, (cp[:-1], dp[:-1]), reverse=True)
    x = jnp.concatenate([x_rest_rev, x_last[None]])
    return x


# =============================================================================
# RS TRIDIAGONAL OPERATOR ASSEMBLY (hand-derived, validated in __main__)
# =============================================================================
def _velocity_affine_coeffs(alpha_l, Q, rho_l, rho_g):
    """u_l = a_l + b_l*Delta, u_g = a_g + b_g*Delta (exact, given frozen alpha_l)."""
    alpha_g = 1.0 - alpha_l
    denom = rho_l * alpha_g + rho_g * alpha_l + 1e-12
    a_g = rho_l * Q / denom
    b_g = -alpha_l / denom
    a_l = (rho_g / rho_l) * a_g
    b_l = 1.0 / rho_l + (rho_g / rho_l) * b_g
    return a_l, b_l, a_g, b_g


def _face_ratios(alpha_k, floor):
    """ratio_face[f] for the face between cells f, f+1 -- same regularization as
    _reynolds_stress_explicit_test.py's _rs_source_interior."""
    alpha_face = 0.5 * (alpha_k[1:] + alpha_k[:-1])
    alpha_face_safe = jnp.clip(alpha_face, floor, 1.0)
    return alpha_face / alpha_face_safe


def _rs_tridiagonal_system(alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff,
                            dt_half, floor=ALPHA_SAFE_FLOOR):
    """
    Assembles the backward-Euler system (I - dt_half*L) x = rhs for the RS
    operator acting on interior unknowns x = Delta_new[1:-1], given alpha_l
    frozen at its current (full, length-N) values. Boundary Delta[0]
    (inlet) and Delta[-1] (outlet, held at its pre-half-step value, same
    convention as the explicit version) are treated as known inputs, folded
    into the RHS. Returns (sub, main, super, rhs) ready for thomas_solve.
    """
    N = alpha_l.shape[0]
    alpha_g = 1.0 - alpha_l

    a_l, b_l, a_g, b_g = _velocity_affine_coeffs(alpha_l, Q, rho_l, rho_g)
    r_l = _face_ratios(alpha_l, floor)   # length N-1, r_l[f] = ratio at face (f, f+1)
    r_g = _face_ratios(alpha_g, floor)

    inv_dx2 = 1.0 / dx**2
    I = jnp.arange(1, N - 1)  # interior full-array indices, length N-2

    rL_l, rR_l = r_l[I - 1], r_l[I]
    rL_g, rR_g = r_g[I - 1], r_g[I]

    sub_full = ((nu_l_eff * inv_dx2) * rL_l * b_l[I - 1]
                - (nu_g_eff * inv_dx2) * rL_g * b_g[I - 1])
    super_full = ((nu_l_eff * inv_dx2) * rR_l * b_l[I + 1]
                  - (nu_g_eff * inv_dx2) * rR_g * b_g[I + 1])
    main_full = (-(nu_l_eff * inv_dx2) * (rR_l + rL_l) * b_l[I]
                 + (nu_g_eff * inv_dx2) * (rR_g + rL_g) * b_g[I])
    const_full = ((nu_l_eff * inv_dx2) * (rL_l * a_l[I - 1] - (rR_l + rL_l) * a_l[I] + rR_l * a_l[I + 1])
                  - (nu_g_eff * inv_dx2) * (rL_g * a_g[I - 1] - (rR_g + rL_g) * a_g[I] + rR_g * a_g[I + 1]))

    sub_sys = -dt_half * sub_full[1:]      # length N-3, rows I=2..N-2
    super_sys = -dt_half * super_full[:-1]  # length N-3, rows I=1..N-3
    main_sys = 1.0 - dt_half * main_full    # length N-2

    Delta_old_int = Delta[1:-1]
    rhs = Delta_old_int + dt_half * const_full
    # fold known boundary values (moved from LHS to RHS, sign flips) into
    # the first/last rows
    rhs = rhs.at[0].add(dt_half * sub_full[0] * Delta[0])
    rhs = rhs.at[-1].add(dt_half * super_full[-1] * Delta[-1])

    return sub_sys, main_sys, super_sys, rhs


def _implicit_rs_half_step(alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half):
    """One implicit RS half-step, alpha_l frozen. Returns the new interior
    Delta (length N-2) -- caller reassembles with the boundary cells, same
    convention as solver._implicit_friction_half_step."""
    sub, main, super_, rhs = _rs_tridiagonal_system(
        alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half
    )
    return thomas_solve(sub, main, super_, rhs)


# =============================================================================
# CELL-CENTERED-DIVISION VARIANT -- the ORIGINAL (pre-face-flux) formula,
# S_RS(I) = nu_eff * div(I) / alpha_safe(I), with alpha_safe clipped at the
# OUTPUT cell I rather than folded into the face flux. Kept as a fully
# separate, independently-validated parallel path (not a flag on the
# face-flux version above) specifically to test whether implicit treatment
# needs the face-flux reformulation for stability, or whether -- per the
# general negative-semi-definiteness-preserved-under-positive-diagonal-
# rescaling argument -- it's unconditionally stable regardless of which of
# the two regularizations is used. Ground-truth explicit function included
# here (not reusing _reynolds_stress_explicit_test's _rs_source_interior,
# which has already been overwritten to the face-flux version).
# =============================================================================
def _rs_source_interior_cellcentered(alpha_l, Delta, Q, dx, rho_l_star, rho_g_star,
                                      nu_l_eff_star, nu_g_eff_star, floor=ALPHA_SAFE_FLOOR):
    u_l, u_g = solver.recover_velocities(alpha_l, Delta, Q, rho_l_star, rho_g_star)
    alpha_g = 1.0 - alpha_l

    def diffusion_term(alpha_k, u_k, nu_k_eff):
        du_dx_face = (u_k[1:] - u_k[:-1]) / dx
        alpha_face = 0.5 * (alpha_k[1:] + alpha_k[:-1])   # RAW, not ratio-regularized
        flux_face = alpha_face * du_dx_face
        div = (flux_face[1:] - flux_face[:-1]) / dx
        alpha_k_safe = jnp.clip(alpha_k[1:-1], floor, 1.0)  # cell-centered, AFTER divergence
        return nu_k_eff * div / alpha_k_safe

    S_l = diffusion_term(alpha_l, u_l, nu_l_eff_star)
    S_g = diffusion_term(alpha_g, u_g, nu_g_eff_star)
    return S_l - S_g


def _rs_tridiagonal_system_cellcentered(alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff,
                                         dt_half, floor=ALPHA_SAFE_FLOOR):
    """Same derivation pattern as _rs_tridiagonal_system, but for the
    cell-centered-division formula: the 1/alpha_safe(I) prefactor is
    evaluated at the OUTPUT cell I and multiplies the WHOLE bracket
    (raw, un-regularized face-averaged flux), not folded per-face."""
    N = alpha_l.shape[0]
    alpha_g = 1.0 - alpha_l

    a_l, b_l, a_g, b_g = _velocity_affine_coeffs(alpha_l, Q, rho_l, rho_g)

    alpha_face_l = 0.5 * (alpha_l[1:] + alpha_l[:-1])   # length N-1, RAW
    alpha_face_g = 0.5 * (alpha_g[1:] + alpha_g[:-1])

    alpha_l_safe = jnp.clip(alpha_l, floor, 1.0)   # length N, cell-centered
    alpha_g_safe = jnp.clip(alpha_g, floor, 1.0)

    inv_dx2 = 1.0 / dx**2
    I = jnp.arange(1, N - 1)  # interior full-array indices, length N-2

    aFL_l, aFR_l = alpha_face_l[I - 1], alpha_face_l[I]   # left/right RAW face values for cell I
    aFL_g, aFR_g = alpha_face_g[I - 1], alpha_face_g[I]

    K_l = nu_l_eff * inv_dx2 / alpha_l_safe[I]
    K_g = nu_g_eff * inv_dx2 / alpha_g_safe[I]

    sub_full = K_l * aFL_l * b_l[I - 1] - K_g * aFL_g * b_g[I - 1]
    super_full = K_l * aFR_l * b_l[I + 1] - K_g * aFR_g * b_g[I + 1]
    main_full = -K_l * (aFR_l + aFL_l) * b_l[I] + K_g * (aFR_g + aFL_g) * b_g[I]
    const_full = (K_l * (aFL_l * a_l[I - 1] - (aFR_l + aFL_l) * a_l[I] + aFR_l * a_l[I + 1])
                  - K_g * (aFL_g * a_g[I - 1] - (aFR_g + aFL_g) * a_g[I] + aFR_g * a_g[I + 1]))

    sub_sys = -dt_half * sub_full[1:]
    super_sys = -dt_half * super_full[:-1]
    main_sys = 1.0 - dt_half * main_full

    Delta_old_int = Delta[1:-1]
    rhs = Delta_old_int + dt_half * const_full
    rhs = rhs.at[0].add(dt_half * sub_full[0] * Delta[0])
    rhs = rhs.at[-1].add(dt_half * super_full[-1] * Delta[-1])

    return sub_sys, main_sys, super_sys, rhs


def _implicit_rs_half_step_cellcentered(alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half):
    sub, main, super_, rhs = _rs_tridiagonal_system_cellcentered(
        alpha_l, Delta, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half
    )
    return thomas_solve(sub, main, super_, rhs)


def advance_state_rs_imex_cellcentered(state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
                                        alpha_l_inlet, Delta_inlet, nu_l_eff, nu_g_eff,
                                        g=9.81, f_i_mult=1.0, k_wl=1.0, k_wg=1.0, theta=1.3, beta=0.0):
    alpha_l = state['alpha_l']
    Delta = state['Delta']

    Delta_int_a = _implicit_rs_half_step_cellcentered(alpha_l, Delta, Q, dx, rho_l, rho_g,
                                                        nu_l_eff, nu_g_eff, dt / 2.0)
    Delta_a = jnp.concatenate([Delta[0:1], Delta_int_a, Delta_int_a[-1:]])
    state_a = {'alpha_l': alpha_l, 'Delta': Delta_a}

    state_b, n_kh = solver.advance_state(
        state_a, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g, alpha_l_inlet, Delta_inlet,
        g, f_i_mult, k_wl, k_wg, theta, beta
    )

    alpha_l_b, Delta_b = state_b['alpha_l'], state_b['Delta']
    Delta_int_c = _implicit_rs_half_step_cellcentered(alpha_l_b, Delta_b, Q, dx, rho_l, rho_g,
                                                        nu_l_eff, nu_g_eff, dt / 2.0)
    Delta_final = jnp.concatenate([Delta_b[0:1], Delta_int_c, Delta_int_c[-1:]])

    return {'alpha_l': alpha_l_b, 'Delta': Delta_final}, n_kh


def run_real_data_condition_rs_imex_cellcentered(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N=150,
                                                  n_transit_times=100.0, cfl=0.25, g=9.81, f_i_mult=1.0,
                                                  k_wl=1.0, k_wg=1.0, theta=1.3, outlet_window=10, beta=0.0,
                                                  C_eps=C_EPS_DEFAULT, nu_t=NU_T_DEFAULT, enable_rs=True):
    """Identical structure to run_real_data_condition_rs_imex, swapping in
    the cell-centered-division implicit half-step."""
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
        new_state, n_kh_faces = advance_state_rs_imex_cellcentered(
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


# =============================================================================
# IMEX STRANG-SPLIT WRAPPER -- RS implicit half-steps bracket the existing,
# untouched friction+transport block (solver.advance_state)
# =============================================================================
def advance_state_rs_imex(state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
                           alpha_l_inlet, Delta_inlet, nu_l_eff, nu_g_eff,
                           g=9.81, f_i_mult=1.0, k_wl=1.0, k_wg=1.0, theta=1.3, beta=0.0):
    alpha_l = state['alpha_l']
    Delta = state['Delta']

    Delta_int_a = _implicit_rs_half_step(alpha_l, Delta, Q, dx, rho_l, rho_g,
                                          nu_l_eff, nu_g_eff, dt / 2.0)
    Delta_a = jnp.concatenate([Delta[0:1], Delta_int_a, Delta_int_a[-1:]])
    state_a = {'alpha_l': alpha_l, 'Delta': Delta_a}

    state_b, n_kh = solver.advance_state(
        state_a, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g, alpha_l_inlet, Delta_inlet,
        g, f_i_mult, k_wl, k_wg, theta, beta
    )

    alpha_l_b, Delta_b = state_b['alpha_l'], state_b['Delta']
    Delta_int_c = _implicit_rs_half_step(alpha_l_b, Delta_b, Q, dx, rho_l, rho_g,
                                          nu_l_eff, nu_g_eff, dt / 2.0)
    Delta_final = jnp.concatenate([Delta_b[0:1], Delta_int_c, Delta_int_c[-1:]])

    return {'alpha_l': alpha_l_b, 'Delta': Delta_final}, n_kh


def run_real_data_condition_rs_imex(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N=150,
                                     n_transit_times=100.0, cfl=0.25, g=9.81, f_i_mult=1.0,
                                     k_wl=1.0, k_wg=1.0, theta=1.3, outlet_window=10, beta=0.0,
                                     C_eps=C_EPS_DEFAULT, nu_t=NU_T_DEFAULT, enable_rs=True):
    """Mirrors solver.run_real_data_condition's structure exactly (same
    nondim/dt/step-count/outlet-window pattern), swapping in the IMEX
    advance_state_rs_imex. enable_rs=False sets nu_eff=0, which must
    reproduce solver.run_real_data_condition exactly -- regression check in
    __main__."""
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
        new_state, n_kh_faces = advance_state_rs_imex(
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
    rng = np.random.default_rng(0)

    # ---- Validation 1: Thomas algorithm vs. dense jnp.linalg.solve ----
    print("=== Validation 1: Thomas algorithm vs. dense solve ===")
    n = 37
    sub = jnp.array(rng.normal(size=n - 1))
    main = jnp.array(rng.normal(size=n) * 5.0 + 20.0)  # diagonally dominant-ish
    super_ = jnp.array(rng.normal(size=n - 1))
    rhs = jnp.array(rng.normal(size=n))
    A = jnp.diag(main) + jnp.diag(sub, -1) + jnp.diag(super_, 1)
    x_dense = jnp.linalg.solve(A, rhs)
    x_thomas = thomas_solve(sub, main, super_, rhs)
    err = float(jnp.max(jnp.abs(x_dense - x_thomas)))
    print(f"  max abs error: {err:.3e}  {'OK' if err < 1e-9 else 'MISMATCH!!'}")
    print()

    # ---- Validation 2: hand-derived tridiagonal bands vs. jax.jacfwd of
    #      the existing, trusted _rs_source_interior ----
    print("=== Validation 2: hand-derived RS bands vs. jax.jacfwd ground truth ===")
    N_small = 24
    Q, dx = 1.0, 0.15
    rho_l, rho_g = 1.0, 0.04
    nu_l_eff, nu_g_eff = 3.7e-4, 2.1e-3
    dt_half = 0.6  # deliberately large -- irrelevant to this check, bands are dt-independent

    alpha_l_test = jnp.array(0.05 + 0.5 * (1 + np.sin(np.linspace(0, 3 * np.pi, N_small))) *
                              np.exp(-((np.arange(N_small) - N_small / 2) / 4.0) ** 2) + 0.02)
    Delta_test = jnp.array((rho_l - rho_g) * (0.7 + 0.3 * rng.normal(size=N_small)))

    def f_interior(x):
        Delta_full = jnp.concatenate([Delta_test[0:1], x, Delta_test[-1:]])
        return _rs_source_interior(alpha_l_test, Delta_full, Q, dx, rho_l, rho_g,
                                    nu_l_eff, nu_g_eff)

    x0 = Delta_test[1:-1]
    J_autodiff = jax.jacfwd(f_interior)(x0)  # dense (N-2, N-2)
    c_autodiff = f_interior(x0) - J_autodiff @ x0

    sub_hand, main_hand, super_hand, _ = _rs_tridiagonal_system(
        alpha_l_test, Delta_test, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half=1.0
    )
    # _rs_tridiagonal_system returns the BACKWARD-EULER (I - dt*L) system bands;
    # recover the raw operator L's bands by undoing the dt_half=1.0, (1-dt*main) transform.
    sub_L = -sub_hand          # since sub_sys = -dt_half*sub_full, dt_half=1.0 here
    super_L = -super_hand
    main_L = 1.0 - main_hand   # since main_sys = 1 - dt_half*main_full, dt_half=1.0

    n_int = N_small - 2
    J_hand = jnp.diag(main_L) + jnp.diag(sub_L, -1) + jnp.diag(super_L, 1)
    err_J = float(jnp.max(jnp.abs(J_hand - J_autodiff)))
    print(f"  max abs error in Jacobian (hand vs. autodiff): {err_J:.3e}  "
          f"{'OK' if err_J < 1e-8 else 'MISMATCH!!'}")

    # constant term: evaluate hand-derived const via the same _rs_tridiagonal_system
    # (rhs = x_old + dt*const + boundary folds -> back out const at dt=1, x_old=0 convention)
    _, _, _, rhs_check = _rs_tridiagonal_system(
        alpha_l_test, Delta_test, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half=1.0
    )
    c_hand = rhs_check - x0  # rhs = x_old + dt*const + boundary folds, dt=1 -> c_hand = const + boundary folds
    # c_autodiff already includes boundary contributions (since f_interior holds
    # Delta_test[0], Delta_test[-1] fixed at their TRUE values) -- same quantity.
    err_c = float(jnp.max(jnp.abs(c_hand - c_autodiff)))
    print(f"  max abs error in constant term: {err_c:.3e}  "
          f"{'OK' if err_c < 1e-8 else 'MISMATCH!!'}")
    print()

    # ---- Validation 3: known-answer heat-equation decay, dt that would be
    #      unconditionally unstable for explicit ----
    print("=== Validation 3: known-answer decay test (implicit, large dt) ===")
    N_decay = 60
    Q, dx = 1.0, 0.05
    rho_l, rho_g = 1.0, 0.04
    nu_const = 0.02  # uniform effective diffusivity for a clean analytic comparison
    alpha_l_uniform = jnp.full(N_decay, 0.3)  # uniform -- isolates pure diffusion behavior
    kmode = 2  # low wavenumber, periodic-ish sinusoidal perturbation
    x_grid = jnp.arange(N_decay) * dx
    L_domain = N_decay * dx
    k_phys = 2 * jnp.pi * kmode / L_domain
    Delta0 = rho_l - rho_g
    perturb_amp = 0.01 * Delta0
    Delta_decay = Delta0 + perturb_amp * jnp.cos(k_phys * x_grid)
    # Boundary cells held at the UNPERTURBED background value (not the
    # perturbed cosine value) -- a Dirichlet BC pinned at a perturbed value
    # would continuously re-inject the perturbation every step regardless
    # of how much the implicit solve damps the interior, which is not what
    # this test is checking.
    Delta_decay = Delta_decay.at[0].set(Delta0)
    Delta_decay = Delta_decay.at[-1].set(Delta0)

    lam = nu_const * k_phys ** 2
    dt_big = 50.0  # huge relative to 1/lam -- would be wildly unstable explicit
    print(f"  lambda={float(lam):.4e}, dt_big*lambda={float(dt_big * lam):.2f} "
          f"(>>2 -> explicit would blow up; implicit should still just damp)")

    n_steps_decay = 20
    state_decay = Delta_decay
    for _ in range(n_steps_decay):
        sub_, main_, super_, rhs_ = _rs_tridiagonal_system(
            alpha_l_uniform, state_decay,
            Q, dx, rho_l, rho_g, nu_const, 0.0, dt_half=dt_big, floor=1e-9
        )
        new_int = thomas_solve(sub_, main_, super_, rhs_)
        state_decay = jnp.concatenate([state_decay[0:1], new_int, state_decay[-1:]])

    amp_numeric = float(jnp.max(jnp.abs(state_decay - Delta0)))
    amp_exact = perturb_amp * float(jnp.exp(-lam * dt_big * n_steps_decay))
    print(f"  amplitude after {n_steps_decay} implicit steps: numeric={amp_numeric:.4e}  "
          f"exact(continuous)={amp_exact:.4e}")
    print(f"  bounded & decaying: {'OK' if amp_numeric <= perturb_amp else 'FAILED -- grew!!'}")
    print(f"  any_nan: {bool(jnp.any(jnp.isnan(state_decay)))}")
    print()

    # ---- Validation 4: regression check (enable_rs=False) vs. production solver ----
    print("=== Validation 4: regression check (enable_rs=False) vs. production solver ===")
    conditions = [
        ("Fan#1 (kh-unstable, U_sg=19.71)", dict(
            U_sl=0.014319654, U_sg=19.71, D=0.0508, L=19.0,
            rho_l=1000.0, rho_g=2.67, mu_l=1.0e-3, mu_g=1.8e-5)),
        ("Fan#2 (kh-unstable, U_sg=19.88)", dict(
            U_sl=0.010300073, U_sg=19.88, D=0.0508, L=19.0,
            rho_l=1000.0, rho_g=2.66, mu_l=1.0e-3, mu_g=1.8e-5)),
    ]
    for label, cond in conditions:
        base_new = run_real_data_condition_rs_imex(**cond, N=150, n_transit_times=20, enable_rs=False)
        base_prod = solver.run_real_data_condition(**cond, N=150, n_transit_times=20)
        match = (abs(base_new['alpha_l_pred'] - base_prod['alpha_l_pred']) < 1e-10
                 and base_new['kh_total'] == base_prod['kh_total'])
        print(f"  {label}: {'OK' if match else 'MISMATCH!!'}")
        if not match:
            print(f"    new ={base_new}")
            print(f"    prod={base_prod}")

    # ---- Validation 5: cell-centered variant -- bands vs. jax.jacfwd, and
    #      regression check ----
    print("=== Validation 5: cell-centered variant -- bands vs. jax.jacfwd ===")

    def f_interior_cc(x):
        Delta_full = jnp.concatenate([Delta_test[0:1], x, Delta_test[-1:]])
        return _rs_source_interior_cellcentered(alpha_l_test, Delta_full, Q, dx, rho_l, rho_g,
                                                 nu_l_eff, nu_g_eff)

    J_autodiff_cc = jax.jacfwd(f_interior_cc)(x0)
    c_autodiff_cc = f_interior_cc(x0) - J_autodiff_cc @ x0

    sub_hand_cc, main_hand_cc, super_hand_cc, rhs_check_cc = _rs_tridiagonal_system_cellcentered(
        alpha_l_test, Delta_test, Q, dx, rho_l, rho_g, nu_l_eff, nu_g_eff, dt_half=1.0
    )
    sub_L_cc, super_L_cc, main_L_cc = -sub_hand_cc, -super_hand_cc, 1.0 - main_hand_cc
    J_hand_cc = jnp.diag(main_L_cc) + jnp.diag(sub_L_cc, -1) + jnp.diag(super_L_cc, 1)
    err_J_cc = float(jnp.max(jnp.abs(J_hand_cc - J_autodiff_cc)))
    print(f"  max abs error in Jacobian (hand vs. autodiff): {err_J_cc:.3e}  "
          f"{'OK' if err_J_cc < 1e-8 else 'MISMATCH!!'}")
    c_hand_cc = rhs_check_cc - x0
    err_c_cc = float(jnp.max(jnp.abs(c_hand_cc - c_autodiff_cc)))
    print(f"  max abs error in constant term: {err_c_cc:.3e}  "
          f"{'OK' if err_c_cc < 1e-8 else 'MISMATCH!!'}")
    print()

    print("=== Validation 6: cell-centered variant -- regression check vs. production ===")
    for label, cond in conditions:
        base_new = run_real_data_condition_rs_imex_cellcentered(**cond, N=150, n_transit_times=20, enable_rs=False)
        base_prod = solver.run_real_data_condition(**cond, N=150, n_transit_times=20)
        match = (abs(base_new['alpha_l_pred'] - base_prod['alpha_l_pred']) < 1e-10
                 and base_new['kh_total'] == base_prod['kh_total'])
        print(f"  {label}: {'OK' if match else 'MISMATCH!!'}")

    print()
    print("DONE")
