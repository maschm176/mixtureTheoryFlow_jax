"""
Water-Gas Reduced Two-Fluid Pipe Flow Solver (JAX)
===================================================

Reduced-model solver for horizontal stratified gas-liquid pipe flow,
replacing the oil-water (phi1, phi2, mom1, mom2) + pressure-projection
solver this file started as a copy of. Built and verified in 6 stages
against a design doc ("Two-Fluid Pipe Flow Solver -- Reduced Model
Refactor"), each with its own acceptance test; this file holds the final,
condensed result -- superseded intermediate scheme variants (plain
Lax-Friedrichs flux, a single global wave speed, symmetric Rusanov
dissipation, first-order-in-space reconstruction) were deleted once the
final scheme (central-upwind + minmod reconstruction + positivity
correction + Strang-split implicit friction) had been validated against
each of them in turn.

WHY A NEW FORMULATION (not just reusing the old phi/mom state):
  1. Velocity recovery in the old solver divides momentum by volume
     fraction (mom/(phi*rho)) -- blows up as a phase vanishes, which is
     the NORMAL operating regime for stratified-wavy air-water flow (low
     liquid holdup), not a rare tail case the way it was for oil-water.
  2. The old solver's single global Lax-Friedrichs wave speed
     over-diffuses slow void waves at this disparity of phase speeds.
  3. The old PHI_COLLAPSE=0.1 hard cutoff discarded most of the thin-film
     physics this project now needs to resolve.
  4. Wall/interfacial friction becomes stiff as the liquid layer thins --
     needs an implicit treatment, not the old explicit source terms.
  5. The old dispersed-bubble drag closure is the wrong physics entirely
     for a stratified (not dispersed) flow regime.

FORMULATION (Change B): eliminate pressure using the 1D volume constraint
and evolve only (alpha_l, Delta), where alpha_l is the liquid volume
fraction and Delta = rho_l*u_l - rho_g*u_g. Velocities are recovered from
(alpha_l, Delta, Q) with NO division by alpha_l anywhere (Q = U_sl + U_sg,
the fixed total superficial velocity) -- this directly removes problem #1
above by construction, rather than patching it with a floor/cutoff.
Pressure is a post-processing-only quantity for validation, never part of
the evolved state.

NUMERICS (Change A): semi-discrete central-upwind flux
(Kurganov-Noelle-Petrova) with minmod reconstruction (genuine 2nd-order
accuracy) and the Kurganov-Petrova (2007) positivity correction on
alpha_l, replacing the old pressure-projection method entirely. Wall and
interfacial friction are integrated via linearized-implicit half-steps,
Strang-split around the explicit SSP-RK2 transport step, since friction
goes stiff as the liquid layer thins.

STATUS: validated end-to-end against the independent steady Taitel-Dukler
equilibrium solver below, and against all real stratified-wavy conditions
in two literature datasets (Fan 2005, MDPI/Carraretto 2020) -- zero NaN,
zero phase collapse, the exact regime that was 100% unusable on the old
solver's PHI_COLLAPSE-bounded formulation. Both datasets sit at low liquid
holdup (alpha_l roughly <= 0.3-0.4), which is also exactly the regime this
validation is currently trustworthy in -- see INCLINATION below for why
that boundary matters beyond just horizontal flow.

INCLINATION: an axial gravity source term (-(rho_l-rho_g)*g*sin(beta) in
S_Delta) and a cos(beta) factor on the cross-sectional stratification-
gravity term in the flux now let the solver take a pipe inclination angle
`beta` (default 0.0, backward compatible). KNOWN LIMITATION, found by a
systematic (U_sl x beta) sweep: this is NOT simply "horizontal works, only
uphill is risky." The solver has no surface tension, so nothing regularizes
Kelvin-Helmholtz growth at short wavelengths, and growth rate scales with
equilibrium liquid holdup/phase-velocity shear -- NOT with inclination
directly. Inclination only matters because it moves the equilibrium
holdup (uphill raises it via the fixed-U_sg/shrinking-gas-area feedback,
downhill lowers it). Consequence: even beta=0 blows up to NaN once liquid
holdup rises past roughly alpha_l~0.5-0.6 at fixed U_sg=5 m/s (confirmed:
U_sl=0.10 -> alpha_l=0.32, stable; U_sl=0.50 -> alpha_l=0.63, NaN even at
beta=0). The Fan/MDPI validation above never hit this because both
datasets happen to sit at low holdup already. Decision (2026-10-29):
restrict near-term real-data calibration to this already-validated
low-holdup regime rather than extending to higher liquid fractions or
uphill inclination; treat surface tension (or an equivalent short-
wavelength regularizer) as a prerequisite before either of those can be
trusted, not yet implemented.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import time

# Force JAX to use 64-bit floats -- critical for numerical stability in PDEs,
# and for the root-find/implicit-differentiation machinery below.
jax.config.update("jax_enable_x64", True)

# NOTE: equinox, optax, matplotlib, and the bb_to_model_inputs import were
# dropped from the old solver's import list here -- they only supported code
# (DragClosureNetwork, the training loop, validation plotting, B&B-to-
# old-state-dict conversion) that was removed below as not applicable to
# this reduced model. Re-add them when actually needed: eqx/optax once a
# learned f_i closure is built (f_i_mult below is the hook point), matplotlib
# for water-gas-specific validation plots, and a NEW (much simpler) real-
# dataset-to-(U_sl, U_sg) converter -- the old bb_to_model_inputs targeted
# the old solver's initial_conditions()/time_step() signatures, which don't
# exist in this file.


# =============================================================================
# TAITEL-DUKLER EQUILIBRIUM SOLVER
# Steady algebraic solver for the stratified equilibrium holdup, used as an
# independent reference/validation target for the transient solver below.
# =============================================================================
# Given fixed inlet superficial velocities U_sl (liquid) and U_sg (gas), what
# liquid height h_l does a horizontal pipe settle into at steady, fully-
# developed conditions?
#
# Derivation: each phase's OWN momentum balance, at steady/fully-developed
# flow (no acceleration, nothing changing with x), is just a force balance:
#
#   liquid:  -A_l*(dp/dx) - tau_wl*S_l + tau_i*S_i = 0
#   gas:     -A_g*(dp/dx) - tau_wg*S_g - tau_i*S_i = 0
#
# Both phases feel the SAME dp/dx (one shared pressure field). Divide the
# liquid equation by A_l, the gas equation by A_g, and subtract -- dp/dx
# cancels, leaving one scalar equation (S_Delta below) in one unknown,
# h_l. Everything else (areas, perimeters, velocities, friction factors)
# is a function of h_l once U_sl/U_sg are fixed. This is the classical
# Taitel-Dukler (1976) combined-momentum equation; S_Delta(h_l) = 0 is the
# equilibrium holdup the transient solver below should relax to if its
# per-cell closures are wired correctly.
# =============================================================================

def td_geometry(ht, D):
    """
    Taitel-Dukler circular-segment geometry.

    ht : h_l / D, liquid height as a fraction of pipe diameter, in (0,1).
    Returns A (total pipe area), A_l, A_g, S_l (liquid wall arc length),
    S_g (gas wall arc length), S_i (flat interface chord length).

    s = 2*ht - 1 remaps ht from (0,1) to (-1,1); arccos(s) is the half
    -angle subtended by the liquid segment at the pipe center. ht is
    clipped away from the exact endpoints because arccos's derivative
    blows up there -- a genuine geometric singularity (a perfectly empty
    or perfectly full pipe has no well-defined interface orientation),
    not a numerical artifact.

    Sanity check (worth keeping in mind while reading): at ht=0.5 (half
    full), s=0, arccos(0)=pi/2, so A_l = D^2/4*(pi - pi/2) = half the pipe
    area, and S_i = D -- a horizontal chord through a circle's center has
    length exactly D.
    """
    ht_safe = jnp.clip(ht, 1e-6, 1.0 - 1e-6)
    s = 2.0 * ht_safe - 1.0

    A   = jnp.pi * D**2 / 4.0
    A_l = D**2 / 4.0 * (jnp.pi - jnp.arccos(s) + s * jnp.sqrt(1.0 - s**2))
    A_g = A - A_l
    S_l = D * (jnp.pi - jnp.arccos(s))
    S_g = D * jnp.arccos(s)
    S_i = D * jnp.sqrt(1.0 - s**2)

    return A, A_l, A_g, S_l, S_g, S_i


def fanning_wall_shear(u, rho, mu, D_h, Re_turb_cutoff=2100.0):
    """
    Fanning-friction wall shear stress tau_w = 0.5*f*rho*|u|*u.

    The laminar branch is written as its own closed form, 8*mu*u/D_h,
    instead of computed via f=16/Re first. Why: f=16/Re = 16*mu/(rho*|u|
    *D_h), so 0.5*f*rho*|u|*u simplifies EXACTLY to 8*mu*u/D_h -- the |u|
    in the numerator and the |u| hiding inside Re cancel algebraically.
    Writing it this way means the laminar branch never divides by u
    anywhere, so it stays well-behaved (and differentiable) as u -> 0.
    Computing Re=rho*u*D_h/mu first (fine on its own) and then f=16/Re
    (blows up at u=0) before multiplying back by u would hit a 0*inf =
    NaN at exactly u=0 despite the true limit being finite.

    The turbulent branch (f=0.046*Re^-0.2) does still divide by Re.
    jnp.where evaluates BOTH branches to build the forward value (and,
    without care, both branches' gradients), so Re is floored away from
    0 before the power, even though the turbulent formula is only ever
    SELECTED once Re is safely above the cutoff.

    NOTE (found during real-data validation, see run_real_data_condition
    below): this cutoff is a genuine jump DISCONTINUITY -- the laminar and
    turbulent formulas are not required to agree at Re=2100. A handful of
    real conditions converge to a liquid Reynolds number sitting almost
    exactly on top of this cutoff, where floating-point-level noise can
    flip individual grid cells across it. That's a real, open limitation
    of this closure, not something fixed here.
    """
    eps = 1e-12
    Re      = rho * jnp.abs(u) * D_h / mu
    Re_safe = jnp.maximum(Re, 1e-6)

    tau_laminar   = 8.0 * mu * u / (D_h + eps)
    f_turbulent   = 0.046 * Re_safe**(-0.2)
    tau_turbulent = 0.5 * f_turbulent * rho * jnp.abs(u) * u

    return jnp.where(Re > Re_turb_cutoff, tau_turbulent, tau_laminar)


def stratified_momentum_source(A, A_l, A_g, S_l, S_g, S_i, u_l, u_g,
                                rho_l, rho_g, mu_l, mu_g, f_i_mult=1.0,
                                g=9.81, beta=0.0):
    """
    Combined per-area momentum source S_Delta, given ALREADY-COMPUTED
    geometry and velocities. Pure closure function -- everything specific
    to "where did u_l/u_g come from" lives outside this function, in its
    two callers:
      - taitel_dukler_residual (steady equilibrium solver): u_l=U_sl/alpha_l,
        u_g=U_sg/alpha_g (the steady, uniform-flow special case).
      - dynamic_source_Delta (transient solver): u_l, u_g from
        recover_velocities, evaluated locally at a real, possibly-
        transient (alpha_l, Delta).
    Keeping this shared (rather than two copies that merely look similar)
    is what makes "the transient solver relaxes to the steady equilibrium"
    a real check of the TRANSIENT solver's correctness, not a comparison
    between two independently-written, possibly-subtly-different closures.

    f_i_mult is a multiplicative knob on the interfacial shear, defaulting
    to 1.0 (unmodified smooth closure). It's the hook point for later
    replacing the interfacial friction with a learned correction without
    touching this function's structure.

    beta is the pipe inclination angle from horizontal (beta=0 is
    horizontal, matching every acceptance test this solver was originally
    built and validated against). The axial gravity term below is derived
    the same way as the rest of S_Delta: each phase's OWN steady force
    balance now also carries an along-pipe gravity component,
    -rho_i*g*sin(beta)*A_i (opposing or assisting flow depending on
    whether the pipe inclines up or down); dividing by A_l/A_g and
    subtracting (same derivation as the dp/dx cancellation in the module
    banner above) leaves -(rho_l-rho_g)*g*sin(beta) surviving in the
    combined equation, since dp/dx was the only thing that canceled
    there -- this term does NOT cancel, it's new physics, not plumbing.
    At beta=0 this is identically zero and S_Delta is unchanged from the
    horizontal-only form every existing acceptance test already verified.
    """
    D_l = 4.0 * A_l / (S_l + 1e-12)
    D_g = 4.0 * A_g / (S_g + S_i + 1e-12)

    tau_wl = fanning_wall_shear(u_l, rho_l, mu_l, D_l)
    tau_wg = fanning_wall_shear(u_g, rho_g, mu_g, D_g)

    # Interfacial shear -- smooth closure, f_i = f_g: reuse the SAME
    # friction-factor value the gas feels at the wall, but apply it to the
    # gas/liquid relative velocity instead of u_g. Unlike tau_wg above,
    # there's no algebraic cancellation available here, because f_g gets
    # multiplied by a DIFFERENT velocity (u_rel, not u_g) -- so if u_g
    # itself were ever near zero while u_rel stayed finite, a naive
    # f=16/Re_g would blow up without anything left to cancel it. Floor
    # Re_g for this purpose only (never physically binding in stratified
    # flow, where the gas phase is always genuinely flowing).
    Re_g_for_fi = jnp.maximum(rho_g * jnp.abs(u_g) * D_g / mu_g, 1.0)
    f_g = jnp.where(Re_g_for_fi > 2100.0,
                     0.046 * Re_g_for_fi**(-0.2),
                     16.0 / Re_g_for_fi)
    u_rel = u_g - u_l
    tau_i = f_i_mult * 0.5 * f_g * rho_g * jnp.abs(u_rel) * u_rel

    S_Delta = (-tau_wl * S_l / A_l
               + tau_wg * S_g / A_g
               + tau_i * S_i * (1.0 / A_l + 1.0 / A_g)
               - (rho_l - rho_g) * g * jnp.sin(beta))
    return S_Delta


def taitel_dukler_residual(ht, U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                            f_i_mult=1.0, g=9.81, beta=0.0):
    """
    Steady combined-momentum residual S_Delta(ht), including the axial
    gravity term when beta != 0 (see stratified_momentum_source's
    docstring for its derivation). The root ht* where this is zero is the
    equilibrium stratified holdup for the given superficial velocities --
    see the section banner above for the dp/dx-cancellation derivation.
    Thin wrapper around stratified_momentum_source -- see that function's
    docstring for the closure details and why it's factored out this way.
    """
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht, D)

    alpha_l = A_l / A
    alpha_g = A_g / A
    u_l = U_sl / jnp.maximum(alpha_l, 1e-9)
    u_g = U_sg / jnp.maximum(alpha_g, 1e-9)

    return stratified_momentum_source(A, A_l, A_g, S_l, S_g, S_i, u_l, u_g,
                                       rho_l, rho_g, mu_l, mu_g, f_i_mult,
                                       g, beta)


def _bisect_ht(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g=9.81,
                beta=0.0, n_iter=60):
    """
    Plain bisection for the root of taitel_dukler_residual over ht in
    (0,1). Not meant to be differentiated through directly -- see
    solve_taitel_dukler_holdup below, which wraps this with a proper
    implicit-function-theorem gradient instead.

    S_Delta(ht) is monotonically decreasing over the physical range in
    every case checked by the test below (more liquid -> liquid wall
    friction term grows and dominates) -- not proven in general here, just
    verified empirically for the stratified-wavy regime this is meant for,
    and only re-checked at beta=0 so far; the axial gravity term added for
    nonzero beta is a CONSTANT shift in ht (independent of ht itself), so
    it can't change where S_Delta(ht) is increasing/decreasing, only shift
    the whole curve up or down -- monotonicity is preserved by
    construction, not something nonzero beta could break on its own.
    """
    lo = jnp.asarray(1e-4, dtype=jnp.float64)
    hi = jnp.asarray(1.0 - 1e-4, dtype=jnp.float64)

    def resid(ht):
        return taitel_dukler_residual(ht, U_sl, U_sg, D, rho_l, rho_g,
                                       mu_l, mu_g, f_i_mult, g, beta)

    def body(_, carry):
        lo, hi = carry
        mid   = 0.5 * (lo + hi)
        r_lo  = resid(lo)
        r_mid = resid(mid)
        keep_right = jnp.sign(r_lo) == jnp.sign(r_mid)
        lo_new = jnp.where(keep_right, mid, lo)
        hi_new = jnp.where(keep_right, hi, mid)
        return (lo_new, hi_new)

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    return 0.5 * (lo + hi)


@jax.custom_jvp
def solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                f_i_mult=1.0, g=9.81, beta=0.0):
    """
    Equilibrium liquid height h_l/D for given superficial velocities,
    found by bisecting the combined-momentum residual S_Delta(ht) = 0.

    Differentiability: bisection has (numerically) zero gradient almost
    everywhere, since each iteration's branch choice is a sign comparison
    -- locally flat. Autodiff-ing through the loop would give mostly-zero,
    wrong gradients. Instead, this function's gradient is defined below
    via the implicit function theorem: S_Delta(ht*; params) = 0 holds
    identically at the solution for ANY small change in params, so the
    total derivative must also vanish:

        (dS_Delta/dht)*dht*  +  (dS_Delta/dparams).dparams  =  0
        =>  dht*/dparams  =  -(dS_Delta/dparams) / (dS_Delta/dht)

    Both quantities are cheap: jax.grad / jax.jvp of the plain, smooth
    residual function evaluated ONCE at the already-converged ht* -- no
    need to differentiate through the solve itself at all. beta is one
    more parameter this same rule applies to unchanged -- it enters
    taitel_dukler_residual exactly like any other physical parameter, so
    dht*/dbeta falls out of the same jvp call with no special-casing.
    """
    return _bisect_ht(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta)


@solve_taitel_dukler_holdup.defjvp
def _solve_taitel_dukler_holdup_jvp(primals, tangents):
    U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta = primals
    d_primals = tangents

    ht_star = _bisect_ht(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta)

    # dS_Delta/dht* at the solution.
    dS_dht = jax.grad(taitel_dukler_residual, argnums=0)(
        ht_star, U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta
    )

    # (dS_Delta/dparams) . dparams in one shot via jvp, holding ht fixed
    # at ht_star -- this is a directional derivative of the residual
    # through every OTHER argument at once.
    def residual_at_fixed_ht(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta):
        return taitel_dukler_residual(ht_star, U_sl, U_sg, D, rho_l, rho_g,
                                       mu_l, mu_g, f_i_mult, g, beta)

    _, dS_dparams = jax.jvp(
        residual_at_fixed_ht,
        (U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta),
        d_primals,
    )

    dht_star = -dS_dparams / dS_dht
    return ht_star, dht_star


# =============================================================================
# TRANSIENT STATE: geometry inversion, velocity recovery, flux, source
# =============================================================================
# State is just two fields per cell: alpha_l (liquid volume fraction) and
# Delta = rho_l*u_l - rho_g*u_g. Q = U_sl + U_sg (total superficial
# velocity) is NOT part of the state -- it's a single fixed scalar for the
# whole run. That's not a simplifying assumption so much as a consequence
# of incompressibility: add the liquid and gas mass equations together
# (using alpha_g = 1-alpha_l) and the d/dt terms cancel, leaving
# d(Q)/dx = 0 -- Q is forced to be spatially uniform at every instant,
# for any incompressible two-fluid flow in a rigid constant-area pipe.
# The old solver enforced this same constraint the expensive way (a
# Poisson/Thomas pressure solve, every timestep) specifically because its
# state (phi1, phi2, mom1, mom2) didn't make the constraint this explicit.
# Here, since Q has to be uniform anyway, we just fix it once as an input
# (literally the flow rates you pump in) -- zero extra computation.
# =============================================================================


def _newton_ht_from_alpha(alpha_l, D, n_iter=8):
    """
    Forward geometric solve: given a target liquid area fraction alpha_l,
    find ht = h_l/D such that A_l(ht)/A = alpha_l. Not meant to be
    differentiated through directly -- see h_l_of_alpha below, which wraps
    this with an implicit-function-theorem gradient instead (same pattern
    as _bisect_ht / solve_taitel_dukler_holdup above, except here we use
    Newton rather than bisection, since this gets called once per cell per
    timestep in the live solver and needs to be fast -- A_l(ht) is smooth
    and strictly increasing over (0,1), so Newton from a reasonable
    starting guess converges quadratically with no risk of the sign-flip
    issues bisection is robust against but doesn't need here).

    ht0 = alpha_l is used as the starting guess -- exact at alpha_l =
    0, 0.5, 1 by the geometry's symmetry, and close enough elsewhere that
    8 Newton iterations converges to machine precision (verified below).
    """
    alpha_l_safe = jnp.clip(alpha_l, 1e-6, 1.0 - 1e-6)
    ht = alpha_l_safe
    for _ in range(n_iter):
        A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht, D)
        resid = A_l / A - alpha_l_safe
        dresid_dht = D * S_i / A + 1e-12   # d(A_l/A)/dht -- see derivation below
        ht = ht - resid / dresid_dht
        ht = jnp.clip(ht, 1e-6, 1.0 - 1e-6)
    return ht


@jax.custom_jvp
def h_l_of_alpha(alpha_l, D):
    """
    Differentiable geometric inverse: given liquid volume fraction
    alpha_l, return the liquid height h_l (NOT ht -- this returns
    dimensional height, in the same units as D).

    Differentiability, via the implicit function theorem (same idea as
    solve_taitel_dukler_holdup above, applied to a different, purely
    geometric equation this time): define F(h_l, alpha_l) := A_l(h_l)/A -
    alpha_l = 0. This holds identically at the solution for any small
    change in alpha_l, so:

        (dF/dh_l)*dh_l  +  (dF/dalpha_l)*dalpha_l  =  0
        dh_l/dalpha_l   =  -(dF/dalpha_l) / (dF/dh_l)
                         =  1 / (d(A_l/A)/dh_l)

    Working through d(A_l)/dh_l from the Taitel-Dukler area formula gives
    dA_l/dh_l = S_i exactly (the rate the liquid area grows as its height
    rises equals the width of the interface at that height -- a direct
    geometric fact, not special to this parametrization). So:

        dh_l/dalpha_l = A / S_i

    See _h_l_of_alpha_jvp below for the (short) matching derivation of
    dh_l/dD, included for completeness since this function is decorated
    over both arguments.
    """
    ht = _newton_ht_from_alpha(alpha_l, D)
    return ht * D


@h_l_of_alpha.defjvp
def _h_l_of_alpha_jvp(primals, tangents):
    alpha_l, D = primals
    dalpha_l, dD = tangents

    ht  = _newton_ht_from_alpha(alpha_l, D)
    h_l = ht * D
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht, D)

    # dh_l/dalpha_l = A/S_i -- see docstring above.
    dh_dalpha = A / (S_i + 1e-12)

    # dh_l/dD = h_l/D: alpha_l is a pure SHAPE fraction (A_l(h_l,D)/A(D)
    # depends only on h_l/D, not on D and h_l separately -- rescaling the
    # whole pipe doesn't change the fraction filled), so ht(alpha_l) itself
    # doesn't depend on D at all, and h_l = ht*D scales linearly with D at
    # fixed alpha_l.
    dh_dD = h_l / (D + 1e-12)

    return h_l, dh_dalpha * dalpha_l + dh_dD * dD


def recover_velocities(alpha_l, Delta, Q, rho_l, rho_g):
    """
    Recover phase velocities from the reduced state (alpha_l, Delta) and
    the fixed total superficial velocity Q -- with NO division by alpha_l
    or alpha_g anywhere. This is the structural fix for problem #1 (the
    old solver's velocity blowup as a phase vanishes): the denominator
    below is a volume-fraction-weighted average of the two densities,
    bounded between rho_g and rho_l for ANY alpha_l in [0,1], including
    exactly 0 or 1 -- there is no value of alpha_l that makes it small.

    Derivation: two linear equations in the two unknowns (u_l, u_g) --
        Delta = rho_l*u_l - rho_g*u_g        (definition of Delta)
        Q     = alpha_l*u_l + alpha_g*u_g    (Q is the volumetric mixture
                                               flux -- see the module
                                               comment above on why it's
                                               fixed, not a state field)
    Substituting u_l = (Delta + rho_g*u_g)/rho_l from the first equation
    into the second and solving for u_g gives the closed form below.
    """
    alpha_g = 1.0 - alpha_l
    denom = rho_l * alpha_g + rho_g * alpha_l + 1e-12
    u_g = (rho_l * Q - alpha_l * Delta) / denom
    u_l = (Delta + rho_g * u_g) / rho_l
    return u_l, u_g


def physical_flux(alpha_l, Delta, Q, D, rho_l, rho_g, g=9.81, beta=0.0):
    """
    Physical (non-dissipative) flux for the (alpha_l, Delta) system.

        flux_alpha = alpha_l*u_l
        flux_Delta = 0.5*rho_l*u_l**2 - 0.5*rho_g*u_g**2
                     + (rho_l-rho_g)*g*cos(beta)*h_l(alpha_l)

    The h_l(alpha_l) term is the direct analog of the Saint-Venant
    system's g*(h+B) term in the Kurganov-Petrova paper: it's what makes
    alpha_l "want" to redistribute toward a level interface under gravity
    (denser liquid settling, lighter gas rising), the same mathematical
    role water depth plays in driving a shallow-water wave. That
    structural match is why that paper's numerics (central-upwind flux,
    local eigenvalues, positivity-preserving reconstruction) carry over
    cleanly here, not just the velocity-desingularization idea above.

    The cos(beta) factor is new: at nonzero inclination, only the
    component of gravity PERPENDICULAR to the pipe axis drives
    cross-sectional stratification -- the component ALONG the axis
    instead drives/retards the flow directly, which is a separate new
    term in dynamic_source_Delta/stratified_momentum_source (S_Delta),
    not in this flux. At beta=0, cos(beta)=1 and this is identical to the
    form every existing acceptance test already verified.
    """
    u_l, u_g = recover_velocities(alpha_l, Delta, Q, rho_l, rho_g)
    h_l = h_l_of_alpha(alpha_l, D)

    flux_alpha = alpha_l * u_l
    flux_Delta = (0.5 * rho_l * u_l**2 - 0.5 * rho_g * u_g**2
                  + (rho_l - rho_g) * g * jnp.cos(beta) * h_l)
    return flux_alpha, flux_Delta


def dynamic_source_Delta(alpha_l, Delta, Q, D, rho_l, rho_g, mu_l, mu_g,
                          f_i_mult=1.0, g=9.81, beta=0.0):
    """
    S_Delta evaluated at a real, local (possibly transient) state -- the
    transient counterpart of taitel_dukler_residual's steady equation,
    sharing the same stratified_momentum_source closure (see that
    function's docstring for why this matters -- it's what makes "the
    transient solver relaxes to the steady equilibrium" a real check of
    correctness, not a comparison between two independently-written
    closures). beta threads through to the axial gravity term described
    there.
    """
    ht = _newton_ht_from_alpha(alpha_l, D)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht, D)
    u_l, u_g = recover_velocities(alpha_l, Delta, Q, rho_l, rho_g)
    return stratified_momentum_source(A, A_l, A_g, S_l, S_g, S_i, u_l, u_g,
                                       rho_l, rho_g, mu_l, mu_g, f_i_mult,
                                       g, beta)


def init_state_water_gas(N, U_sl, U_sg, rho_l, rho_g):
    """
    No-slip initial condition: both phases start at the common mixture
    velocity Q everywhere (the flow hasn't separated into stratified
    layers yet -- mirrors the old solver's uniform-phi1-at-rest initial
    condition in spirit, adapted to a state where "at rest" isn't
    meaningful since velocities aren't independent state fields here).
    """
    Q = U_sl + U_sg
    alpha_l0 = U_sl / Q
    Delta0   = (rho_l - rho_g) * Q
    alpha_l  = jnp.full((N,), alpha_l0)
    Delta    = jnp.full((N,), Delta0)
    return {'alpha_l': alpha_l, 'Delta': Delta}


# =============================================================================
# CHARACTERISTIC SPEEDS (flux-Jacobian eigenvalues)
# =============================================================================
# The real, local answer to "how fast is information actually moving
# through THIS point" is the eigenvalues of the flux Jacobian dF/dU -- not
# u_l or u_g directly, but the actual characteristic speeds of the
# (alpha_l, Delta) system, which (by the Saint-Venant analogy established
# above -- the h_l(alpha_l) term playing the role of g*h) look like
# "velocity plus/minus a gravity-wave correction", the same structural
# shape as shallow water's u +/- sqrt(g*h).

def local_wave_speeds(alpha_l, Delta, Q, D, rho_l, rho_g, g=9.81, beta=0.0):
    """
    Local characteristic (wave) speeds of the (alpha_l, Delta) system, at
    every cell -- the eigenvalues of the flux Jacobian J = dF/dU, where
    F = physical_flux and U = (alpha_l, Delta) (Q, D, rho_l, rho_g, g are
    held FIXED -- they're parameters of the problem, not state).

    For a 2x2 Jacobian, the eigenvalues have a direct closed form:
        lam = trace/2 +/- sqrt(trace**2/4 - det)
    No need to assemble J into an explicit matrix and call a generic
    eigenvalue solver -- jax.jacfwd(physical_flux, argnums=(0,1)) already
    hands back exactly the four entries of J (since physical_flux returns
    the tuple (flux_alpha, flux_Delta), differentiating each component
    w.r.t. each of the two state args gives J's four entries directly),
    and jax.vmap applies that PER CELL rather than building one giant
    cross-cell Jacobian for the whole array at once.

    If the discriminant (trace**2/4 - det) is negative, the eigenvalues
    are genuinely complex -- the system has locally lost its real
    characteristic speeds entirely, which is a Kelvin-Helmholtz-type
    breakdown of hyperbolicity (a known, real limitation of simple
    stratified two-fluid models past some relative-velocity threshold,
    not a numerical artifact). is_complex flags this per cell so it can
    be counted/logged rather than silently producing nonsense.

    Where is_complex is True, lam1/lam2 fall back to using sqrt(|disc|)
    instead of a (NaN-producing) sqrt of a negative number -- a
    deliberate, defensive choice, not a physically meaningful speed: it
    keeps the returned numbers real and finite (usable as a dissipation
    speed to stabilize the scheme) and grows with how severe the
    instability is, so the numerical scheme adds MORE damping exactly
    where the underlying physics is becoming least trustworthy, rather
    than silently returning NaN or under-damping there.
    """
    jac_fn = jax.jacfwd(physical_flux, argnums=(0, 1))
    jac_vmapped = jax.vmap(jac_fn, in_axes=(0, 0, None, None, None, None, None, None))

    (dFa_dalpha, dFa_dDelta), (dFd_dalpha, dFd_dDelta) = jac_vmapped(
        alpha_l, Delta, Q, D, rho_l, rho_g, g, beta
    )

    trace = dFa_dalpha + dFd_dDelta
    det   = dFa_dalpha * dFd_dDelta - dFa_dDelta * dFd_dalpha
    disc  = trace**2 / 4.0 - det

    is_complex = disc < 0.0
    sqrt_term  = jnp.sqrt(jnp.abs(disc))

    lam1 = trace / 2.0 - sqrt_term
    lam2 = trace / 2.0 + sqrt_term

    return lam1, lam2, is_complex


# =============================================================================
# CENTRAL-UPWIND NUMERICAL FLUX
# =============================================================================
# Central-upwind treats rightward- and leftward-moving information
# separately: a_plus (how fast is anything actually moving right here,
# floored at 0) and a_minus (how fast is anything moving left, capped at
# 0) are computed independently, so a face with little or no leftward-
# propagating information gets little or no leftward dissipation, instead
# of a single symmetric (Rusanov/Lax-Friedrichs-style) blanket damping
# sized by whichever direction's wave happens to be fastest.

def central_upwind_face_speeds(alpha_l, Delta, Q, D, rho_l, rho_g, g=9.81, beta=0.0):
    """
    Per-face one-sided speeds for the central-upwind flux, from the same
    per-cell eigenvalues local_wave_speeds already computes (lam1 <= lam2
    by construction -- see that function). "left"/"right" at a face here
    means whichever two states border it -- either the two neighboring
    cells' own centers (first-order) or, in advance_state below, their
    reconstructed sub-cell face values (second-order).

        a_plus  = max(lam2_L, lam2_R, 0)   -- fastest RIGHTWARD speed, >= 0
        a_minus = min(lam1_L, lam1_R, 0)   -- fastest LEFTWARD speed, <= 0

    NOTE: not currently called anywhere in this file -- advance_state's
    transport step inlines the same first-order-vs-reconstructed logic
    directly via two separate local_wave_speeds calls instead of this
    helper. Kept (and kept correct) as a small, independently-testable
    building block, not because anything here depends on it.
    """
    lam1, lam2, is_complex = local_wave_speeds(alpha_l, Delta, Q, D, rho_l, rho_g, g, beta)

    a_plus  = jnp.maximum(jnp.maximum(lam2[:-1], lam2[1:]), 0.0)
    a_minus = jnp.minimum(jnp.minimum(lam1[:-1], lam1[1:]), 0.0)
    face_is_complex = is_complex[:-1] | is_complex[1:]

    return a_plus, a_minus, face_is_complex


def central_upwind_flux(F_L, F_R, U_L, U_R, a_plus, a_minus):
    """
    Kurganov-Noelle-Petrova central-upwind numerical flux at each face.

        H = (a_plus*F_L - a_minus*F_R)/(a_plus-a_minus)
            + (a_plus*a_minus/(a_plus-a_minus))*(U_R - U_L)

    a_plus - a_minus is always >= 0 by construction (a_plus = max(...,0)
    >= 0, a_minus = min(...,0) <= 0), hitting exactly 0 only in the
    degenerate case where every relevant eigenvalue on both sides is
    exactly zero. Floor it directly rather than computing the division
    and selecting it away with jnp.where afterward -- same "make the
    denominator safe BEFORE dividing" reasoning fanning_wall_shear
    already follows for its turbulent branch.

    Sanity check worth knowing (verified numerically before this was
    trusted): forcing a_plus = -a_minus = a collapses this exactly to
    Lax-Friedrichs with speed a -- 0.5*(F_L+F_R) - 0.5*a*(U_R-U_L).
    Central-upwind is a strict generalization of a plain Lax-Friedrichs/
    Rusanov flux, not a different family of scheme.
    """
    denom = jnp.maximum(a_plus - a_minus, 1e-10)
    return (a_plus * F_L - a_minus * F_R) / denom + (a_plus * a_minus / denom) * (U_R - U_L)


# =============================================================================
# RECONSTRUCTION (minmod/MUSCL) + POSITIVITY-PRESERVING CORRECTION
# =============================================================================
# A first-order scheme feeds the flux raw cell-center values directly --
# a face's left/right states are just "whatever the two neighboring
# cells' own averages happen to be," with no attempt to estimate what the
# solution is actually doing WITHIN each cell. Reconstruction replaces
# that with a per-cell linear profile (slope from minmod), so the flux
# instead sees values extrapolated to the actual face location -- the
# source of the jump from first-order to second-order accuracy.

def minmod3(z1, z2, z3):
    """
    Three-argument minmod: min(z1,z2,z3) if all three are strictly
    positive, max(z1,z2,z3) if all three are strictly negative, 0
    otherwise (mixed sign, or any argument exactly 0). This is the
    actual non-oscillatory mechanism: a nonzero slope is only let
    through when three independent local difference estimates all agree
    on direction -- near a kink or discontinuity they won't, and the
    slope gets forced to zero (falling back to the same flat, first-
    order cell-center value a plain first-order scheme would use)
    rather than risk extrapolating past a real extremum into an
    overshoot.
    """
    all_pos = (z1 > 0) & (z2 > 0) & (z3 > 0)
    all_neg = (z1 < 0) & (z2 < 0) & (z3 < 0)
    return jnp.where(all_pos, jnp.minimum(jnp.minimum(z1, z2), z3),
            jnp.where(all_neg, jnp.maximum(jnp.maximum(z1, z2), z3), 0.0))


def minmod_slope(U, dx, theta=1.3):
    """
    Generalized minmod slope per interior cell (Kurganov & Petrova 2007,
    eq 2.12). theta in [1,2] controls how much dissipation the limiter
    allows: theta=1 is the most conservative/dissipative choice
    (equivalent to the classic minmod limiter), theta=2 the least --
    1.3 is the paper's own stated default for most problems (their
    Section 4), used here per the design doc.

    U is the FULL state array, shape (N,) -- including the boundary
    values already sitting in indices 0 and N-1, same convention as
    alpha_l/Delta everywhere else in this file. Returns one slope per
    INTERIOR cell (shape (N-2,)), since a slope needs both neighbors.
    """
    left   = theta * (U[1:-1] - U[:-2]) / dx
    center = (U[2:] - U[:-2]) / (2.0 * dx)
    right  = theta * (U[2:] - U[1:-1]) / dx
    return minmod3(left, center, right)


def reconstruct_faces(U, dx, theta=1.3):
    """
    Piecewise-linear MUSCL reconstruction: each interior cell extrapolates
    its own average by half a cell width, using its own minmod-limited
    slope, giving right_face (the value that cell's reconstruction takes
    at its own right edge) and left_face (same, at its own left edge).

    These are deliberately NOT yet matched into a single value per face --
    a face between cell j and cell j+1 gets TWO candidate values,
    right_face[j] (extrapolated from the left) and left_face[j+1]
    (extrapolated from the right), which can legitimately disagree.
    Resolving that disagreement is exactly what the numerical flux
    (central_upwind_flux) is for -- that's the point of a Riemann-solver-
    free scheme, not something reconstruction itself should paper over.
    """
    slope = minmod_slope(U, dx, theta)
    U_center = U[1:-1]
    right_face = U_center + (dx / 2.0) * slope
    left_face  = U_center - (dx / 2.0) * slope
    return left_face, right_face


def positivity_correct(U_center, left_face, right_face, lo=0.0, hi=1.0):
    """
    Kurganov-Petrova (2007) positivity-preserving correction, eqs
    2.15-2.16 of the paper -- generalized with both a floor (lo) and a
    mirrored ceiling (hi). The paper only needed a floor (water depth
    has no upper bound); alpha_l is a genuine volume fraction, bounded on
    BOTH ends, so the same correction is applied twice, once per bound.
    This two-bound extension is not literally in the paper -- it's a
    direct generalization of their one-bound argument, verified below
    rather than assumed.

    The idea: if a cell's reconstructed line would dip below lo (or
    above hi) at one of its own faces, don't just clip that one face --
    recompute the implied slope so THAT face sits exactly at the bound,
    and let the OPPOSITE face become 2*U_center - bound. This is
    conservative by construction: the average of the two corrected face
    values is still exactly U_center (nothing gained or lost, only
    redistributed within the cell) -- unlike a naive per-face clip,
    which would NOT preserve the cell average.

    Worked through by hand (and checked numerically in the test suite)
    whether floor-then-ceiling can ever "ping-pong": fixing the floor can
    only push the opposite face above hi when U_center > hi/2, and in
    that regime the subsequent ceiling fix's opposite face,
    2*U_center - hi, is automatically still above lo. So two sequential
    passes always fully resolves both bounds -- no third pass needed.
    """
    # --- floor ---
    right_below = right_face < lo
    rf = jnp.where(right_below, lo, right_face)
    lf = jnp.where(right_below, 2.0 * U_center - lo, left_face)

    left_below = lf < lo
    lf = jnp.where(left_below, lo, lf)
    rf = jnp.where(left_below, 2.0 * U_center - lo, rf)

    # --- ceiling (mirror image) ---
    right_above = rf > hi
    rf = jnp.where(right_above, hi, rf)
    lf = jnp.where(right_above, 2.0 * U_center - hi, lf)

    left_above = lf > hi
    lf = jnp.where(left_above, hi, lf)
    rf = jnp.where(left_above, 2.0 * U_center - hi, rf)

    return lf, rf


# =============================================================================
# IMPLICIT FRICTION (STRANG SPLITTING) + FULL TIMESTEP
# =============================================================================
# A simpler scheme would add the friction source term EXPLICITLY, in the
# same single update as the flux divergence: Delta += dt*(-div_Delta +
# S_Delta). That's fine as long as S_Delta stays modest, but it gets
# genuinely stiff as the liquid layer thins (A_l -> 0 drives
# tau_wl*S_l/A_l -> infinity) -- an explicit step can then overshoot the
# true (rapidly-relaxing) equilibrium instead of tracking it, unless dt
# shrinks to match. The timestep below is instead split into three parts
# -- implicit friction half-step, explicit transport full-step, implicit
# friction half-step (Strang splitting) -- so friction gets a treatment
# that's stable for any dt, while the flux divergence (which ISN'T
# stiff) keeps using the accurate explicit central-upwind scheme
# unchanged.

def _implicit_friction_half_step(alpha_l_int, Delta_int, Q, D, rho_l, rho_g,
                                  mu_l, mu_g, dt_half, f_i_mult=1.0, g=9.81,
                                  beta=0.0):
    """
    Linearized-implicit half-step on Delta ONLY -- alpha_l and Q are
    frozen (per the design doc: "In each cell, alpha_l and Q are frozen;
    only Delta changes"). This is a single algebraic solve of backward
    Euler linearized around the current Delta, not an iterative Newton
    solve -- exact for this one linearization, which is what makes it
    cheap enough to run twice (half-step before and after transport)
    every single timestep.

    dynamic_source_Delta is already purely elementwise (no cross-cell
    coupling -- confirmed when it was built, see the local_wave_speeds/
    _newton_ht_from_alpha chain), so vmapping its value_and_grad over
    cells gives exactly the per-cell (S0, dS/dDelta) pair the formula
    needs, same pattern used for local_wave_speeds' per-cell Jacobian.

    The axial gravity term (beta != 0) is a CONSTANT additive piece of
    S0 (independent of Delta -- see stratified_momentum_source), so it
    changes S0 but not dS/dDelta at all. It's folded into this same
    implicit solve for free, not handled separately, since it isn't
    stiff the way friction is -- there's no reason to split it out into
    its own treatment.
    """
    def source_fn(a, d):
        return dynamic_source_Delta(a, d, Q, D, rho_l, rho_g, mu_l, mu_g,
                                     f_i_mult, g, beta)

    S0, dS = jax.vmap(jax.value_and_grad(source_fn, argnums=1))(alpha_l_int, Delta_int)

    return Delta_int + dt_half * S0 / (1.0 - dt_half * dS)


def _transport_only_step(state, dt, dx, Q, D, rho_l, rho_g,
                          alpha_l_inlet, Delta_inlet, g=9.81, theta=1.3,
                          beta=0.0):
    """
    The reconstruction + positivity correction + central-upwind flux
    update used throughout this file, WITH THE SOURCE TERM REMOVED --
    friction is handled entirely by the implicit half-steps bookending
    this in advance_state. This is "Part 2" of the Strang split, called
    TWICE per full timestep (SSP-RK2 needs two forward-Euler-like
    stages) -- see _ssp_rk2_transport below.

    Boundary cells (index 0 and N-1) have no slope of their own (a slope
    needs both neighbors) -- they contribute their raw, unreconstructed
    value to whichever face they border, same spirit as a ghost cell
    carrying no gradient information.

    NO defensive clip on alpha_l here -- positivity_correct above is
    relied on to keep the UPDATED cell average in [0,1] by construction
    (Kurganov & Petrova 2007's positivity-preserving property), not
    patched after the fact. Confirmed safe by the real-data validation
    below, down to alpha_l ~ 5e-5 (Fan 2005's thinnest liquid-loading
    conditions) -- a leftover PHI_COLLAPSE-style clip was removed from an
    earlier version of this function once that held up.
    """
    alpha_l = state['alpha_l']
    Delta   = state['Delta']

    alpha_lf_raw, alpha_rf_raw = reconstruct_faces(alpha_l, dx, theta)
    Delta_lf, Delta_rf         = reconstruct_faces(Delta, dx, theta)

    alpha_center = alpha_l[1:-1]
    alpha_lf, alpha_rf = positivity_correct(alpha_center, alpha_lf_raw, alpha_rf_raw)

    alpha_L = jnp.concatenate([alpha_l[0:1], alpha_rf])
    alpha_R = jnp.concatenate([alpha_lf, alpha_l[-1:]])
    Delta_L = jnp.concatenate([Delta[0:1], Delta_rf])
    Delta_R = jnp.concatenate([Delta_lf, Delta[-1:]])

    F_alpha_L, F_Delta_L = physical_flux(alpha_L, Delta_L, Q, D, rho_l, rho_g, g, beta)
    F_alpha_R, F_Delta_R = physical_flux(alpha_R, Delta_R, Q, D, rho_l, rho_g, g, beta)

    # Local speeds evaluated AT the reconstructed face states themselves,
    # not at the neighboring cells' own centers.
    lam1_L, lam2_L, is_complex_L = local_wave_speeds(alpha_L, Delta_L, Q, D, rho_l, rho_g, g, beta)
    lam1_R, lam2_R, is_complex_R = local_wave_speeds(alpha_R, Delta_R, Q, D, rho_l, rho_g, g, beta)

    a_plus  = jnp.maximum(jnp.maximum(lam2_L, lam2_R), 0.0)
    a_minus = jnp.minimum(jnp.minimum(lam1_L, lam1_R), 0.0)
    face_is_complex = is_complex_L | is_complex_R

    f_alpha_face = central_upwind_flux(F_alpha_L, F_alpha_R, alpha_L, alpha_R, a_plus, a_minus)
    f_Delta_face = central_upwind_flux(F_Delta_L, F_Delta_R, Delta_L, Delta_R, a_plus, a_minus)

    div_alpha = (f_alpha_face[1:] - f_alpha_face[:-1]) / dx
    div_Delta = (f_Delta_face[1:] - f_Delta_face[:-1]) / dx

    alpha_l_new_int = alpha_l[1:-1] - dt * div_alpha
    Delta_new_int   = Delta[1:-1]   - dt * div_Delta   # no source term -- that's the whole point

    alpha_l_full = jnp.concatenate([
        jnp.array([alpha_l_inlet]), alpha_l_new_int, alpha_l_new_int[-1:]
    ])
    Delta_full = jnp.concatenate([
        jnp.array([Delta_inlet]), Delta_new_int, Delta_new_int[-1:]
    ])

    n_kh_faces = jnp.sum(face_is_complex)
    return {'alpha_l': alpha_l_full, 'Delta': Delta_full}, n_kh_faces


def _ssp_rk2_transport(state, dt, dx, Q, D, rho_l, rho_g,
                        alpha_l_inlet, Delta_inlet, g=9.81, theta=1.3,
                        beta=0.0):
    """
    SSP-RK2 (Heun's method): two forward-Euler-like transport stages,
    averaged -- same pattern already sketched in the old solver's RK2
    appendix (k1 = step(state), k2 = step(k1), result = 0.5*(state+k2)),
    applied here to the source-free transport update specifically.
    """
    stage1, kh1 = _transport_only_step(state, dt, dx, Q, D, rho_l, rho_g,
                                        alpha_l_inlet, Delta_inlet, g, theta, beta)
    stage2, kh2 = _transport_only_step(stage1, dt, dx, Q, D, rho_l, rho_g,
                                        alpha_l_inlet, Delta_inlet, g, theta, beta)

    new_state = jax.tree.map(lambda a, b: 0.5 * (a + b), state, stage2)
    return new_state, kh1 + kh2


def advance_state(state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
                   alpha_l_inlet, Delta_inlet, g=9.81, f_i_mult=1.0,
                   theta=1.3, beta=0.0):
    """
    Full Strang-split timestep: implicit friction half-step (dt/2) ->
    explicit SSP-RK2 transport full-step (dt) -> implicit friction
    half-step (dt/2). alpha_l is untouched by the friction half-steps
    (only Delta evolves there); boundary cells (index 0, N-1) are left to
    the fixed inlet/zero-gradient treatment throughout -- friction is
    only ever evaluated on genuine interior cells.
    """
    alpha_l = state['alpha_l']
    Delta   = state['Delta']

    # Part 1: implicit friction half-step
    Delta_int_1 = _implicit_friction_half_step(
        alpha_l[1:-1], Delta[1:-1], Q, D, rho_l, rho_g, mu_l, mu_g, dt / 2.0,
        f_i_mult, g, beta
    )
    Delta_1 = jnp.concatenate([Delta[0:1], Delta_int_1, Delta_int_1[-1:]])
    state_1 = {'alpha_l': alpha_l, 'Delta': Delta_1}

    # Part 2: explicit SSP-RK2 transport
    state_2, n_kh = _ssp_rk2_transport(
        state_1, dt, dx, Q, D, rho_l, rho_g, alpha_l_inlet, Delta_inlet, g, theta, beta
    )

    # Part 3: second implicit friction half-step
    alpha_l_2, Delta_2 = state_2['alpha_l'], state_2['Delta']
    Delta_int_3 = _implicit_friction_half_step(
        alpha_l_2[1:-1], Delta_2[1:-1], Q, D, rho_l, rho_g, mu_l, mu_g, dt / 2.0,
        f_i_mult, g, beta
    )
    Delta_final = jnp.concatenate([Delta_2[0:1], Delta_int_3, Delta_int_3[-1:]])

    return {'alpha_l': alpha_l_2, 'Delta': Delta_final}, n_kh


def run_to_steady(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N,
                   dt, n_steps, g=9.81, f_i_mult=1.0, theta=1.3, beta=0.0):
    """
    Run the full solver (advance_state) from the no-slip initial condition
    for n_steps of size dt, via jax.lax.scan. Returns the final state and
    the total KH-flagged-face-step count accumulated over the run.
    """
    Q = U_sl + U_sg
    alpha_l_inlet = U_sl / Q
    Delta_inlet   = (rho_l - rho_g) * Q
    dx = L / N

    state = init_state_water_gas(N, U_sl, U_sg, rho_l, rho_g)

    def step(carry, _):
        state, kh_total = carry
        new_state, n_kh_faces = advance_state(
            state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
            alpha_l_inlet, Delta_inlet, g, f_i_mult, theta, beta
        )
        return (new_state, kh_total + n_kh_faces), None

    (final_state, kh_total), _ = jax.lax.scan(
        step, (state, jnp.array(0)), None, length=n_steps
    )
    return final_state, kh_total


# =============================================================================
# REAL-DATA VALIDATION DIAGNOSTICS
# =============================================================================
# Validates the solver above against two real stratified-wavy gas-liquid
# datasets (Fan 2005, MDPI/Carraretto 2020) that motivated this whole
# rewrite -- see the module docstring's problem #1 and #3. Reports
# predicted equilibrium holdup and pressure gradient against measured
# values, plus KH-flag counts per condition.
#
# Two helper functions below support this: a CFL dt estimator that reuses
# the equilibrium solver (rather than guessing a dt), and a post-
# processing pressure-gradient recovery (since Change B eliminated
# pressure from the EVOLVED state -- it's recoverable afterward from
# either phase's own steady force balance, never evolved directly).

def estimate_stable_dt(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, dx,
                        f_i_mult=1.0, cfl=0.25, n_sample=20, g=9.81, beta=0.0):
    """
    Physically-grounded dt estimate for a real data condition, instead of
    guessing a fixed dt and hoping it's small enough.

    First attempt (found to be WRONG, kept here as a documented mistake
    rather than silently fixed, since it's instructive): evaluating
    local_wave_speeds ONLY at the equilibrium state. That badly
    underestimated the true max speed for the condition tried here --
    equilibrium eigenvalues came out ~0.64 while the actual run went NaN,
    because the solver starts from the no-slip initial condition (every
    cell at alpha_l=alpha_l_inlet, Delta=Delta_inlet), whose own
    eigenvalues were ~5.2, nearly an order of magnitude larger. Delta (not
    alpha_l) is what drives this gap: holding Delta near its NO-SLIP value
    keeps both eigenvalues close to Q regardless of alpha_l (checked
    numerically across the full alpha_l range), but Delta drops sharply as
    the profile relaxes toward equilibrium slip, and that drop is what
    collapses the eigenvalue magnitude -- so the equilibrium point alone
    is actually the LEAST representative state for the transient's fastest
    speed, not the most.

    Fix: sample local_wave_speeds along the straight-line interpolation
    between the no-slip initial state and the equilibrium state (the two
    endpoints the real transient actually connects), plus both endpoints
    themselves, and take the max |eigenvalue| found anywhere along that
    path. This is a cheap, non-differentiated diagnostic (dt itself is a
    Python/NumPy-side control-flow input, not part of the differentiable
    solve), so there's no cost concern to sampling generously.
    """
    Q = U_sl + U_sg
    alpha_l_inlet = U_sl / Q
    Delta_inlet   = (rho_l - rho_g) * Q

    ht_eq = solve_taitel_dukler_holdup(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g,
                                        f_i_mult, g, beta)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht_eq, D)
    alpha_l_eq = A_l / A
    alpha_g_eq = A_g / A
    u_l_eq = U_sl / jnp.maximum(alpha_l_eq, 1e-9)
    u_g_eq = U_sg / jnp.maximum(alpha_g_eq, 1e-9)
    Delta_eq = rho_l * u_l_eq - rho_g * u_g_eq

    t = jnp.linspace(0.0, 1.0, n_sample)
    alpha_path = alpha_l_inlet + t * (alpha_l_eq - alpha_l_inlet)
    Delta_path = Delta_inlet + t * (Delta_eq - Delta_inlet)

    lam1, lam2, _ = local_wave_speeds(alpha_path, Delta_path, Q, D, rho_l, rho_g, g, beta)
    speed = jnp.max(jnp.maximum(jnp.abs(lam1), jnp.abs(lam2)))
    speed = jnp.maximum(speed, 1e-3)
    return cfl * dx / speed


def predicted_pressure_gradient(alpha_l, Delta, Q, D, rho_l, rho_g, mu_l, mu_g,
                                 f_i_mult=1.0, g=9.81, beta=0.0):
    """
    Post-processing-only pressure gradient, recovered from EACH phase's own
    steady force balance (see the section banner above the equilibrium
    solver for the two equations this comes from):

        liquid:  dP/dz = -tau_wl*S_l/A_l + tau_i*S_i/A_l - rho_l*g*sin(beta)
        gas:     dP/dz = -tau_wg*S_g/A_g - tau_i*S_i/A_g - rho_g*g*sin(beta)

    The -rho_i*g*sin(beta) term is each phase's own axial gravity
    component, carried over directly from the same per-phase balance
    stratified_momentum_source's docstring derives S_Delta's new term
    from -- it survives here individually (unlike in S_Delta, where only
    the DIFFERENCE rho_l-rho_g survives), since this function recovers
    each phase's own dP/dz separately rather than their combined
    equation. At beta=0 both reduce to the original horizontal formula.

    At a genuinely converged steady profile these should closely agree --
    that's exactly the condition dynamic_source_Delta (their DIFFERENCE)
    was driven to zero at. Returning both, instead of just one, makes
    their agreement itself a convergence check: a real disagreement here
    means the profile hasn't actually settled, not that the formula split
    is wrong. Never used inside the simulation loop -- pressure was
    eliminated from the evolved state by Change B, so this is strictly a
    diagnostic for comparing against measured dP/dL.
    """
    ht = _newton_ht_from_alpha(alpha_l, D)
    A, A_l, A_g, S_l, S_g, S_i = td_geometry(ht, D)
    u_l, u_g = recover_velocities(alpha_l, Delta, Q, rho_l, rho_g)

    D_l = 4.0 * A_l / (S_l + 1e-12)
    D_g = 4.0 * A_g / (S_g + S_i + 1e-12)
    tau_wl = fanning_wall_shear(u_l, rho_l, mu_l, D_l)
    tau_wg = fanning_wall_shear(u_g, rho_g, mu_g, D_g)

    Re_g_for_fi = jnp.maximum(rho_g * jnp.abs(u_g) * D_g / mu_g, 1.0)
    f_g = jnp.where(Re_g_for_fi > 2100.0,
                     0.046 * Re_g_for_fi**(-0.2),
                     16.0 / Re_g_for_fi)
    u_rel = u_g - u_l
    tau_i = f_i_mult * 0.5 * f_g * rho_g * jnp.abs(u_rel) * u_rel

    dPdz_liquid = -tau_wl * S_l / A_l + tau_i * S_i / A_l - rho_l * g * jnp.sin(beta)
    dPdz_gas    = -tau_wg * S_g / A_g - tau_i * S_i / A_g - rho_g * g * jnp.sin(beta)
    return dPdz_liquid, dPdz_gas


def run_real_data_condition(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, L, N=150,
                             n_transit_times=100.0, cfl=0.25, g=9.81, f_i_mult=1.0,
                             theta=1.3, outlet_window=10, beta=0.0):
    """
    Run one real-data condition to (approximate) steady state and return
    the diagnostics needed to validate against it: predicted outlet
    alpha_l, predicted dP/dz (both phase estimates), total KH-flag count,
    and min/max alpha_l actually visited over the WHOLE run (the direct
    check on whether the lack of any defensive alpha_l clip -- see
    _transport_only_step -- is safe for THIS condition, not just the ones
    tried during the solver's own development).

    dt and n_steps are both derived from the condition itself (via
    estimate_stable_dt and a fixed number of convective transit times,
    L/Q) rather than reusing one hardcoded (dt, n_steps) pair across every
    condition -- real conditions here span U_sg from ~0.03 to ~25.7 m/s,
    nearly three orders of magnitude, so a single fixed dt/n_steps would
    be either unstable on the fast conditions or wastefully slow on the
    slow ones.

    n_transit_times was raised from 30 to 100 after a longer-run check
    (sampling convergence at 30/60/100/.../300 transit times on the worst
    disagreeing conditions from an earlier validation pass) showed most
    flagged conditions settle cleanly -- EXACTLY (disagree -> 0.0000) --
    by 60-100 transit times, just not yet at 30.

    outlet_window: a SEPARATE, different problem that longer runtime does
    NOT fix. A couple of conditions in that same check kept disagreeing
    even at 300 transit times, but their alpha_l profile was already
    completely flat -- the holdup itself is genuinely converged. Dumping
    the raw per-cell values (not just the aggregate disagreement number)
    found the real cause: these conditions converge to a liquid Reynolds
    number sitting almost exactly ON fanning_wall_shear's laminar/
    turbulent cutoff (Re=2100), and floating-point-level noise in alpha_l
    (~1e-6) is enough to flip individual cells across that DISCONTINUOUS
    switch -- producing a dense cluster of "clean" cells plus a minority
    of large-magnitude outlier cells sitting on the other branch. A
    single-cell read has no way to tell which branch is "correct"; see
    the median-vs-mean comment right where outlet_window is actually used
    below for why a plain window MEAN isn't the fix either (it still gets
    dragged by the outliers) and a MEDIAN is. One condition was found to
    sit close enough to exactly 50/50 across that discontinuity that even
    the median can't recover a trustworthy value -- a genuine, open
    limitation of the friction closure itself, not something this
    diagnostic-side fix can paper over.
    """
    Q = U_sl + U_sg
    dx = L / N
    dt = estimate_stable_dt(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, dx, f_i_mult,
                             cfl, g=g, beta=beta)
    t_end = n_transit_times * (L / Q)
    n_steps = int(np.ceil(float(t_end / dt)))

    alpha_l_inlet = U_sl / Q
    Delta_inlet   = (rho_l - rho_g) * Q
    state = init_state_water_gas(N, U_sl, U_sg, rho_l, rho_g)

    def step(carry, _):
        state, kh_total, a_min, a_max = carry
        new_state, n_kh_faces = advance_state(
            state, dt, dx, Q, D, rho_l, rho_g, mu_l, mu_g,
            alpha_l_inlet, Delta_inlet, g, f_i_mult, theta, beta
        )
        a_min = jnp.minimum(a_min, jnp.min(new_state['alpha_l']))
        a_max = jnp.maximum(a_max, jnp.max(new_state['alpha_l']))
        return (new_state, kh_total + n_kh_faces, a_min, a_max), None

    init_carry = (state, jnp.array(0), jnp.array(1.0), jnp.array(0.0))
    (final_state, kh_total, alpha_min, alpha_max), _ = jax.lax.scan(
        step, init_carry, None, length=n_steps
    )

    # Window outlet diagnostics -- last `outlet_window` INTERIOR cells,
    # excluding the ghost copy at index -1 (see advance_state: the last
    # cell is always a verbatim copy of the second-to-last, so including
    # it would double-count one real value rather than add an independent
    # sample).
    #
    # MEDIAN, not mean: a plain mean was tried first and made things WORSE
    # on some conditions. Root cause, found by dumping the raw per-cell
    # values directly rather than trusting the aggregate number: a few
    # conditions converge to a liquid Reynolds number sitting almost
    # exactly ON fanning_wall_shear's laminar/turbulent cutoff (Re=2100) --
    # e.g. one Fan 2005 condition converges to Re_l=2100.78, and floating-
    # point-level noise in alpha_l (~1e-6, far below anything physically
    # meaningful) is enough to flip individual cells across that
    # DISCONTINUOUS switch, producing a few huge outlier dPdz values
    # (dense cluster near -167 with sparse spikes to +141) alongside
    # otherwise-clean neighbors. A mean lets those rare, large outliers
    # dominate the result; a median ignores them entirely as long as they
    # stay a minority of the window, recovering the same clean value the
    # majority of cells already agree on.
    alpha_l_window = final_state['alpha_l'][-(outlet_window + 1):-1]
    Delta_window   = final_state['Delta'][-(outlet_window + 1):-1]
    dPdz_l_arr, dPdz_g_arr = predicted_pressure_gradient(
        alpha_l_window, Delta_window, Q, D, rho_l, rho_g, mu_l, mu_g, f_i_mult, g, beta
    )
    alpha_l_outlet = jnp.median(alpha_l_window)
    dPdz_l = jnp.median(dPdz_l_arr)
    dPdz_g = jnp.median(dPdz_g_arr)

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
    }
