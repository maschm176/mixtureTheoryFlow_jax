"""
1D Two-Phase Mixture Flow Solver in JAX
========================================

Solves the two-phase mixture theory equations for flow in a 1D pipe using
explicit time integration with a pressure projection method to enforce
volume conservation.

GOVERNING EQUATIONS (per phase α = 1, 2):

  Mass:
    ∂(φ_α ρ_α)/∂t + ∂(φ_α ρ_α u_α)/∂x = 0

  Momentum:
    ∂(φ_α ρ_α u_α)/∂t + ∂(φ_α ρ_α u_α² + φ_α p)/∂x
        = p ∂φ_α/∂x + M_α

  Volume constraint:
    φ₁ + φ₂ = 1   (enforced via pressure projection)

  Inter-phase drag (Stokes):
    M₁ =  C_d φ₁ φ₂ (u₂ - u₁)
    M₂ = -M₁

SOLUTION SEQUENCE PER TIME STEP:
  1. Advance mass equations    → new φ₁, φ₂
  2. Advance momentum eqs      → intermediate velocities u₁*, u₂*
  3. Pressure Poisson solve    → pressure correction dp
  4. Velocity projection       → volume-conserving u₁, u₂
  5. Algebraic cleanup         → enforce φ₂ = 1 - φ₁

NUMERICAL METHODS:
  - Spatial:  Lax-Friedrichs flux (upwind-biased, stable for convection)
  - Time:     Forward Euler (RK2 commented out at bottom for easy upgrade)
  - Pressure: Thomas algorithm (tridiagonal solve — exact in 1D, O(N))
  - BCs:      Fixed pressure inlet, zero-gradient outlet
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import numpy as np

import importlib
import bb_to_model_inputs as _bb_mod
importlib.reload(_bb_mod)
from bb_to_model_inputs import bb_to_model_inputs



# Force JAX to use 64-bit floats — critical for numerical stability in PDEs.
# By default JAX uses 32-bit, which accumulates floating point error quickly
# in time-marching solvers.
jax.config.update("jax_enable_x64", True)


# =============================================================================
# SECTION 1: GRID SETUP
# =============================================================================
# We use a finite volume method on a uniform 1D grid.
# Each "cell" is a control volume of width dx.
# Variables are stored at cell centers.
# Fluxes are computed at cell faces (between centers).
#
#   |--dx--|--dx--|--dx--|
#   |  0   |  1   |  2   |  ...  |  N-1  |   <- cell indices
#   ^      ^      ^                        <- face indices (N+1 faces total)
#   0      1      2                N
#
# Cell centers: x[i] = (i + 0.5) * dx
# Faces:        x_face[i] = i * dx

def make_grid(L, N):
    """
    Create a uniform 1D finite volume grid.
    
    L : float — pipe length [m]
    N : int   — number of cells
    
    Returns dx (cell width) and x (cell center coordinates).
    """
    dx = L / N
    x  = (jnp.arange(N) + 0.5) * dx   # cell centers, shape (N,)
    return dx, x


# =============================================================================
# SECTION 2: INITIAL CONDITIONS
# =============================================================================
# At t=0 we set the state of every cell in the pipe.
# Everything is at rest initially (u=0), with uniform volume fractions
# and a linear pressure gradient from inlet to outlet.

def initial_conditions(x, L, phi1_0, rho1, rho2, p_inlet, p_outlet):
    """
    Set initial conditions across all cells.

    x       : (N,) cell center coordinates
    L       : pipe length
    phi1_0  : initial volume fraction of phase 1 (uniform)
    rho1    : density of phase 1 [kg/m³]
    rho2    : density of phase 2 [kg/m³]
    p_inlet : inlet pressure [Pa]
    p_outlet: outlet pressure [Pa]

    Returns a dict of state arrays, each shape (N,).
    """
    N = len(x)

    phi1 = jnp.full(N, phi1_0)
    phi2 = 1.0 - phi1                          # volume constraint at t=0

    u1   = jnp.zeros(N)                        # both phases at rest
    u2   = jnp.zeros(N)

    # Linear pressure profile: high at inlet, low at outlet
    phi2_min = 1e-4
    #phi2_min = 7.5e-4
    phi1     = jnp.full(N, phi1_0)
    phi2     = jnp.maximum(1.0 - phi1, phi2_min)
    phi1     = 1.0 - phi2    # re-normalize so phi1 + phi2 = 1 exactly

    u1 = jnp.zeros(N)
    u2 = jnp.zeros(N)

    # This is the driving force for the flow
    p    = p_inlet + (p_outlet - p_inlet) * (x / L)

    return dict(phi1=phi1, phi2=phi2, u1=u1, u2=u2, p=p,
                rho1=jnp.full(N, rho1), rho2=jnp.full(N, rho2))


def initial_conditions_slug(x, L, rho1, rho2, p_inlet, p_outlet):
    N = len(x)
    
    # Step discontinuity at x = L/2
    # Left: 80% liquid (phase 1), Right: 20% liquid
    phi1 = jnp.where(x < L / 2, 0.8, 0.2)
    phi2 = 1.0 - phi1

    # Both phases initially at rest
    u1 = jnp.zeros(N)
    u2 = jnp.zeros(N)

    # Linear pressure gradient to drive flow left to right
    p = p_inlet + (p_outlet - p_inlet) * (x / L)

    return dict(phi1=phi1, phi2=phi2, u1=u1, u2=u2, p=p,
                rho1=jnp.full(N, rho1), rho2=jnp.full(N, rho2))



# =============================================================================
# SECTION 3: CFL TIME STEP
# =============================================================================
# The CFL (Courant-Friedrichs-Lewy) condition is a stability requirement for
# explicit time integration. It says: information cannot travel more than one
# cell per time step. If dt is too large, the numerical scheme sees a wave
# "jump over" a cell and the solution blows up.
#
# dt < CFL * dx / max_wavespeed
#
# We use CFL=0.4 (conservative — 0.5 is the theoretical limit for
# Lax-Friedrichs but we stay below it for safety).

def compute_dt(state, dx, mu1, mu2, rho1_val, rho2_val, cfl=0.4, dt_max=1e-4):
            
    """
    Compute the maximum stable time step from the CFL condition.
    Wave speeds are the phase velocities plus pressure wave speed.
    For incompressible phases we use max(|u1|, |u2|) as a proxy.
    """
    eps_phi = 1e-6
    
    max_speed = jnp.maximum(
        jnp.max(jnp.abs(state['u1'])),
        jnp.max(jnp.abs(state['u2'])))
    
    max_speed = jnp.maximum(max_speed, 1e-6)
    dt_cfl = cfl * dx / max_speed
    
    # checking viscous stability conditions as well, where nu = mu/rho_val
    nu_max  = max(mu1/rho1_val, mu2/rho2_val)
    dt_visc = 0.5 * dx**2 / nu_max
    
    dt_cfl = jnp.minimum(dt_cfl,  dt_visc)
    
    return jnp.minimum(dt_cfl, dt_max)   # never exceed dt_max


# =============================================================================
# SECTION 4: LAX-FRIEDRICHS FLUX
# =============================================================================
# The Lax-Friedrichs scheme is the simplest stable numerical flux for
# hyperbolic conservation laws like our mass and momentum equations.
#
# For a conservation law  ∂q/∂t + ∂F(q)/∂x = 0,
# the Lax-Friedrichs numerical flux at face i+1/2 is:
#
#   F_{i+1/2} = 0.5*(F(q_i) + F(q_{i+1})) - 0.5*α*(q_{i+1} - q_i)
#
# where α = max wave speed (numerical dissipation coefficient).
#
# The first term is the average of left and right fluxes (central scheme).
# The second term adds dissipation proportional to the jump in q — this
# is what makes the scheme stable by damping oscillations at faces.
#
# Indexing: if q has shape (N,), the output has shape (N-1,) — one flux
# value per interior face.

def lax_friedrichs_flux(F, q, alpha):
    """
    Compute Lax-Friedrichs numerical flux at all interior faces.

    F     : (N,) physical flux at cell centers
    q     : (N,) conserved variable at cell centers
    alpha : scalar max wave speed for dissipation

    Returns flux at N-1 interior faces, shape (N-1,).
    
    F[:-1] = left cell of each face (cell i)
    F[1:]  = right cell of each face (cell i+1)
    """
    return 0.5 * (F[:-1] + F[1:]) - 0.5 * alpha * (q[1:] - q[:-1])


# =============================================================================
# SECTION 5: STEP 1 — ADVANCE MASS EQUATIONS
# =============================================================================
# Mass equation for phase α:
#   ∂(φ_α ρ_α)/∂t = -∂(φ_α ρ_α u_α)/∂x
#
# We time-integrate the conserved variable m_α = φ_α ρ_α.
# The flux is F_mα = φ_α ρ_α u_α (mass flux).
#
# After advancing m_α, we recover φ_α = m_α / ρ_α (since ρ_α is constant).

def advance_mass(state, dt, dx):
    """
    Advance mass equations for both phases by one time step.
    Returns updated phi1, phi2 (interior cells only, shape N-2).
    
    """
    phi1 = state['phi1']
    phi2 = state['phi2']
    rho1 = state['rho1']
    rho2 = state['rho2']
    u1   = state['u1']
    u2   = state['u2']
    p    = state['p']

    eps_phi = 1e-6

    # --- Conserved mass variables ---
    m1 = phi1 * rho1    # shape (N,)
    m2 = phi2 * rho2

    # --- Physical mass fluxes at cell centers ---
    F_m1 = phi1 * rho1 * u1
    F_m2 = jnp.where(phi2 > eps_phi, phi2 * rho2 * u2, 0.0)

    # --- Max wave speed for Lax-Friedrichs dissipation ---
    alpha = jnp.maximum(jnp.max(jnp.abs(u1)), jnp.max(jnp.abs(u2)))
    alpha = jnp.maximum(alpha, 1e-6)

    # --- Numerical fluxes at faces, shape (N-1,) ---
    f_m1 = lax_friedrichs_flux(F_m1, m1, alpha)
    f_m2 = lax_friedrichs_flux(F_m2, m2, alpha)

    # --- Flux divergence at interior cells, shape (N-2,) ---
    div_m1 = (f_m1[1:] - f_m1[:-1]) / dx
    div_m2 = (f_m2[1:] - f_m2[:-1]) / dx

    # --- Forward Euler advance of conserved mass ---
    m1_new = m1[1:-1] - dt * div_m1   # shape (N-2,)
    m2_new = m2[1:-1] - dt * div_m2

    # --- Recover volume fractions from updated mass ---
    phi1_new = m1_new / rho1[1:-1]
    phi2_new = m2_new / rho2[1:-1]

    # ── Drift flux correction ──────────────────────────────────────────────────
    # The drift flux of phase 1 relative to the mixture is:
    #     j_drift = phi1 * phi2 * (u1 - u2)
    #
    # Its divergence acts as a redistribution source in the phi1 equation:
    #     d(phi1)/dt += -d(j_drift)/dx
    #
    # Sign check:
    #   oil faster than water → u2 > u1 → (u1 - u2) < 0
    #   → j_drift < 0
    #   → -d(j_drift)/dx > 0 in the interior (flux converging)
    #   → phi1 increases → holdup rises above water cut ✓
    #
    # Note: j_drift is purely a volumetric redistribution — no rho needed
    # because it acts on volume fractions directly, not mass.
    # ──────────────────────────────────────────────────────────────────────────

    # Drift flux at all cell centers, shape (N,)
    j_drift_centers = phi1 * phi2 * (u1 - u2)

    # Drift flux at faces via simple averaging, shape (N-1,)
    # Simple averaging is consistent with the Lax-Friedrichs approach
    # used for the main mass fluxes above
    j_drift_faces = 0.5 * (j_drift_centers[:-1] + j_drift_centers[1:])

    # Divergence of drift flux at interior cells, shape (N-2,)
    div_drift = (j_drift_faces[1:] - j_drift_faces[:-1]) / dx
    
    # --- CFL stability limiter for drift flux ---
    # Drift flux introduces effective advection at speed phi2 * |u2 - u1|
    # Fixed dt in scan means no adaptive safety net — clip manually
    max_drift_speed = jnp.max(jnp.abs(phi2 * (u2 - u1)))
    max_drift_speed = jnp.maximum(max_drift_speed, 1e-10)
    dt_cfl_drift    = 0.5 * dx / max_drift_speed
    cfl_scale       = jnp.minimum(1.0, dt_cfl_drift / (dt + 1e-10))

    # Apply SCALED drift flux
    # need when dt is too low and using jax.lax.scan instead of while loop for the time loop
    phi1_new = phi1_new - dt * cfl_scale * div_drift
    phi2_new = 1.0 - phi1_new

    # Apply drift flux correction to volume fractions directly
    # This is separate from the mass conservation step above because
    # drift flux acts on volume fractions, not conserved mass variables
    #phi1_new = phi1_new - dt * div_drift
    #phi2_new = 1.0 - phi1_new   # enforce phi1 + phi2 = 1 exactly

    # ── Clip to physical bounds ────────────────────────────────────────────────
    # Small overshoots from numerics can push fractions slightly outside
    # [0,1]. Clip first, then re-enforce the sum constraint.
    phi1_new = jnp.clip(phi1_new, 0.0, 1.0)
    phi2_new = jnp.clip(phi2_new, 0.0, 1.0)

    # Re-normalize to guarantee phi1 + phi2 = 1 after clipping
    # Clipping both independently can break the sum constraint
    total    = phi1_new + phi2_new + 1e-10
    phi1_new = phi1_new / total
    phi2_new = phi2_new / total

    return phi1_new, phi2_new


# =============================================================================
# SECTION 6: STEP 2 — ADVANCE MOMENTUM EQUATIONS (INTERMEDIATE VELOCITIES)
# =============================================================================
# Momentum equation for phase α:
#   ∂(φ_α ρ_α u_α)/∂t = -∂(φ_α ρ_α u_α² + φ_α p)/∂x
#                        + p ∂φ_α/∂x
#                        + M_α
#
# The non-conservative term p ∂φ_α/∂x appears because φ_α multiplies p
# in the momentum flux — when you expand the divergence you get this
# extra term. It ensures the sum of all phase momentum equations gives
# the correct mixture momentum equation.
#
#      ∇·τ_α  : viscous stress — φ_α μ_α ∂²u_α/∂x²
#     φ_α ρ_α g : gravity body force along pipe axis
#
# M_α is the inter-phase drag: M₁ = C_d φ₁ φ₂ (u₂ - u₁), M₂ = -M₁
# The drag is proportional to the volume fractions of both phases
# (more drag when both phases are present) and the velocity difference.
#
# These are INTERMEDIATE velocities u* — not yet volume-conserving.
# The pressure projection in Step 3 will correct them.

def advance_momentum(state, dt, dx, drag_coeff, d_b, D, mu1, mu2, theta):
    """
    Advance momentum equations for both phases (intermediate step).
    Returns intermediate velocities u1_star, u2_star (interior, shape N-2).
    """
    phi1 = state['phi1']
    phi2 = state['phi2']
    rho1 = state['rho1']
    rho2 = state['rho2']
    u1   = state['u1']
    u2   = state['u2']
    p    = state['p']
    
    eps_phi = 1e-6

    # Conserved momentum variables
    mom1 = phi1 * rho1 * u1
    u2_safe = jnp.where(phi2 > eps_phi, u2, 0.0)
    mom2 = phi2 * rho2 * u2_safe

    # Physical momentum fluxes (convection + pressure)
    F_mom1 = phi1 * rho1 * u1**2 + phi1 * p
    #F_mom2 = phi2 * rho2 * u2**2 + phi2 * p
    F_mom2 = jnp.where(phi2 > eps_phi,
                    phi2 * rho2 * u2**2 + phi2 * p,
                    0.0)

    # Max wave speed
    alpha = jnp.maximum(jnp.max(jnp.abs(u1)), jnp.max(jnp.abs(u2)))
    alpha = jnp.maximum(alpha, 1e-6)
    
    mom2_safe = jnp.where(phi2 > eps_phi, mom2, 0.0)

    # Numerical fluxes at faces, shape (N-1,)
    f_mom1 = lax_friedrichs_flux(F_mom1, mom1, alpha)
    f_mom2 = lax_friedrichs_flux(F_mom2, mom2_safe, alpha)

    # Flux divergence at interior cells, shape (N-2,)
    div_mom1 = (f_mom1[1:] - f_mom1[:-1]) / dx
    div_mom2 = (f_mom2[1:] - f_mom2[:-1]) / dx

    # Non-conservative pressure term: p * d(phi)/dx
    dphi1_dx = (phi1[2:] - phi1[:-2]) / (2.0 * dx)   # shape (N-2,)
    dphi2_dx = (phi2[2:] - phi2[:-2]) / (2.0 * dx)
    p_int    = p[1:-1]                                  # shape (N-2,)

    # Interior values
    u1_int   = u1[1:-1]
    u2_int   = u2[1:-1]
    phi1_int = phi1[1:-1]
    phi2_int = phi2[1:-1]
    rho1_int = rho1[1:-1]
    rho2_int = rho2[1:-1]
    
    
    # --- Calculating the viscous stress ---
    # calculate second derivative of velocity for each phase using central difference
    d2u1_dx2 = (u1[2:] - 2.0*u1[1:-1] + u1[:-2]) / dx**2
    d2u2_dx2 = (u2[2:] - 2.0*u2[1:-1] + u2[:-2]) / dx**2

    # using second derivative of velocity to compute viscous stress for each phase
    visc1 = phi1_int * mu1 * d2u1_dx2
    visc2 = phi2_int * mu2 * d2u2_dx2
    
    # --- Gravity Term ---
    g     = 9.81    # [m/s²]
    # Gravity component along pipe axis
    # Positive θ means flow going uphill — gravity opposes flow
    # Negative θ means flow going downhill — gravity assists flow
    g_x = -g * jnp.sin(theta)   # negative because gravity opposes upward flow

    grav1 = phi1_int * rho1[1:-1] * g_x
    grav2 = phi2_int * rho2[1:-1] * g_x
    
    # --- Inter-phase drag ---
    delta_u    = u2_int - u1_int
    #drag_coeff = 0.44
    #d_b        = 1e-3

    #jax.debug.print("drag_coeff: {dc}, d_b: {db}", dc=drag_coeff, db=d_b)
    '''
    ##### - adding a drift flux term to try to cause plugging - ######
    # Drift velocity — buoyancy-driven relative motion
    u_drift = (rho1_int - rho2_int) * g * d_b**2 / (18 * mu1)
    # Add to the effective slip in the drag term
    delta_u_effective = (u2_int - u1_int) - u_drift
    M1 = drag_coeff * (3.0/4.0) * (phi2_int * phi1_int * rho1_int / d_b) \
        * jnp.abs(delta_u_effective) * delta_u_effective
    '''
    #### drag for modeling gas and liquid annular flow #####
    #M1, M2 = compute_drag(phi1_int, phi2_int, rho1_int, rho2_int, u1_int, u2_int, mu1, d_b, D)

    ###### original drag term without drift flux ######
    M1 = drag_coeff * (3.0/4.0) * (phi2_int * phi1_int * rho1[1:-1] / d_b) * jnp.abs(delta_u) * delta_u
    M2 = -M1

    #'''
    # --- Wall friction (Darcy-Weisbach) ---
    # F_wall = -(f / 2D) * rho_mix * u_mix * |u_mix|
    # The negative sign is critical — friction always opposes flow direction.
    f_darcy = 0.02    # Darcy friction factor (dimensionless)
                      # 0.01-0.02 typical for turbulent pipe flow
                      # 64/Re for laminar flow (Re = rho*u*D/mu)
    #D       = 0.0381     # pipe diameter [m]r
    # Momentum-weighted mixture velocity at interior cells
    rho1_int = rho1[1:-1]
    rho2_int = rho2[1:-1]

    rho_mix = phi1_int * rho1_int + phi2_int * rho2_int
    u_mix   = (phi1_int * rho1_int * u1_int + phi2_int * rho2_int * u2_int) \
              / (rho_mix + 1e-10)

    # Friction force per unit volume on the mixture
    F_friction = -(f_darcy / (2.0 * D)) * rho_mix * u_mix * jnp.abs(u_mix)

    # Distribute friction to each phase by volume fraction
    # Each phase feels friction in proportion to how much of the
    # pipe cross-section it occupies
    friction1 = phi1_int * F_friction
    friction2 = phi2_int * F_friction
    #'''

    '''
    # --- Wall friction (Darcy-Weisbach with Blasius friction factor) ---
    # Friction factor is now computed per-phase from the local Reynolds number
    # rather than using a single hardcoded constant.
    #
    # Blasius correlation (smooth pipe, turbulent):
    #     f_D = 0.316 * Re^(-0.25)    valid for 4000 < Re < 100,000
    #
    # Laminar flow:
    #     f_D = 64 / Re               exact analytical solution
    #
    # We switch between the two using jnp.where so JAX can trace through
    # both branches without data-dependent Python control flow.
    #
    # Each phase gets its own Reynolds number using its own velocity,
    # density, viscosity, and hydraulic diameter — consistent with
    # how Ibarra's two-fluid model treats wall friction per phase.

    eps = 1e-10

    rho1_int = rho1[1:-1]
    rho2_int = rho2[1:-1]

    # Hydraulic diameter per phase — each phase only occupies a fraction
    # of the pipe cross-section, so its effective diameter is scaled
    # by its volume fraction. This is the standard two-fluid model
    # approximation for stratified flow.
    D_h1 = D * phi1_int           # water hydraulic diameter [m]
    D_h2 = D * phi2_int           # oil hydraulic diameter [m]

    # Reynolds number per phase
    # Using absolute velocity to keep Re positive regardless of flow direction
    Re1 = rho1_int * jnp.abs(u1_int) * D_h1 / (mu1 + eps)
    Re2 = rho2_int * jnp.abs(u2_int) * D_h2 / (mu2 + eps)

    # Blasius friction factor per phase
    # jnp.where traces both branches — guards against Re=0 division
    f_D1 = jnp.where(
        Re1 > 2100,
        0.316 * Re1**(-0.25),      # turbulent — Blasius
        64.0 / (Re1 + eps)         # laminar — exact solution
    )

    f_D2 = jnp.where(
        Re2 > 2100,
        0.316 * Re2**(-0.25),      # turbulent — Blasius
        64.0 / (Re2 + eps)         # laminar — exact solution
    )

    # Friction force per unit volume on each phase individually
    # Each phase feels friction from its own wall contact only —
    # not a mixture average. The negative sign ensures friction
    # always opposes the direction of flow.
    #
    # F = -(f_D / (2*D_h)) * rho * u * |u|
    #
    # u * |u| gives u² with the correct sign for direction.
    friction1 = -(f_D1 / (2.0 * D_h1 + eps)) \
                * rho1_int * u1_int * jnp.abs(u1_int) * phi1_int

    friction2 = -(f_D2 / (2.0 * D_h2 + eps)) \
                * rho2_int * u2_int * jnp.abs(u2_int) * phi2_int
    '''

    # --- Advance conserved momentum ---
    # All source terms combined: drag + wall friction
    mom1_new = (mom1[1:-1]
                - dt * div_mom1     
                + dt * p_int * dphi1_dx # pressure gradient term
                + dt * visc1            # viscous stress
                + dt * grav1            # gravitational force
                + dt * M1               # interphase drag term
                + dt * friction1)       # wall friction term         

    mom2_new = (mom2[1:-1]
                - dt * div_mom2
                + dt * p_int * dphi2_dx # pressure gradient term
                + dt * visc2            # viscous stress
                + dt * grav2            # gravitational force
                + dt * M2               # interphase drag term
                + dt * friction2)       # wall friction term
    
    
    '''
    # before were recovering intermediate velcoities here, but instead we will just return the updated momentum, 
    # and recover velocities after the pressure projection step, 
    # --- Recover intermediate velocities ---
    eps     = 1e-10
    eps_phi = 1e-6

    u1_star = jnp.where(
        phi1_int > eps_phi,
        mom1_new / (phi1_int * rho1_int + eps),
        0.0
    )

    u2_star = jnp.where(
        phi2_int > eps_phi,
        mom2_new / (phi2_int * rho2_int + eps),
        0.0
    )
    '''
    
    # Zero out momentum where phase 2 is absent
    mom2_new = jnp.where(phi2_int > eps_phi, mom2_new, 0.0)

    #return u1_star, u2_star
    return mom1_new, mom2_new


def compute_drag(phi1, phi2, rho1, rho2, u1, u2, mu1, d_b, D):
    delta_u = u2 - u1
    
    # --- Bubbly flow drag (phi_G < 0.25) ---
    Re_b = rho1 * jnp.abs(delta_u) * d_b / (mu1 + 1e-10)
    C_D_bubble = 24/(Re_b + 1e-6) * (1 + 0.15 * Re_b**0.687)
    M_bubbly = (3/4) * (C_D_bubble / d_b) * rho1 * phi2 * phi1 \
               * jnp.abs(delta_u) * delta_u

    # --- Annular flow drag (phi_G > 0.75) ---
    f_i = 0.005                    # Wallis interfacial friction factor
    a_i = 4 * phi2 / (D + 1e-10)  # interfacial area per unit volume
    M_annular = 0.5 * f_i * a_i * rho2 \
                * jnp.abs(delta_u) * delta_u

    # --- Slug flow drag (0.25 < phi_G < 0.75) ---
    C_D_slug = 0.44
    M_slug = (3/4) * (C_D_slug / D) * rho1 * phi2 * phi1 \
             * jnp.abs(delta_u) * delta_u

    # --- Smooth regime blending ---
    # Use phi2 (gas fraction) as the regime indicator
    w_bubbly  = jnp.clip((0.25 - phi2) / 0.25, 0.0, 1.0)
    w_annular = jnp.clip((phi2 - 0.75) / 0.25, 0.0, 1.0)
    w_slug    = 1.0 - w_bubbly - w_annular

    M1 = w_bubbly * M_bubbly + w_slug * M_slug + w_annular * M_annular
    M2 = -M1
    return M1, M2


# =============================================================================
# SECTION 7: STEP 3 — PRESSURE POISSON SOLVE (THOMAS ALGORITHM)
# =============================================================================
# After advancing mass and momentum, the intermediate velocities u1*, u2*
# generally do NOT satisfy volume conservation:
#
#   ∂/∂x (φ₁u₁ + φ₂u₂) ≠ 0
#
# We need to find a pressure correction dp such that after applying it,
# the corrected velocities DO satisfy volume conservation.
#
# The pressure correction equation (derived by substituting the velocity
# correction u_α = u_α* - (dt/ρ_α) * dp/dx into the divergence-free
# condition) is:
#
#   ∂/∂x [ (φ₁/ρ₁ + φ₂/ρ₂) * dp/dx ] = (1/dt) * ∂/∂x(φ₁u₁* + φ₂u₂*)
#
# This is a 1D Poisson equation for dp. In 1D with uniform grid it becomes
# a tridiagonal linear system, solved exactly in O(N) by the Thomas algorithm.
#
# The Thomas algorithm (tridiagonal matrix algorithm) is just Gaussian
# elimination specialized for tridiagonal matrices. It has two passes:
#   Forward sweep: eliminate the lower diagonal
#   Back substitution: solve from bottom to top

def solve_tridiagonal(a, b, c, d):
    """
    Solve a tridiagonal system A·x = d using the Thomas algorithm.
    
    a : (N,) lower diagonal  (a[0] unused)
    b : (N,) main diagonal
    c : (N,) upper diagonal  (c[-1] unused)
    d : (N,) right-hand side
    
    Returns x, shape (N,).
    
    NOTE: JAX does not support in-place mutation, so we use lax.scan
    to perform the forward and backward sweeps functionally.
    This is the idiomatic JAX way to express sequential recurrences.
    """
    N = len(b)

    # --- Forward sweep ---
    # Eliminate lower diagonal by modifying b and d in place (functionally)
    # c'[0] = c[0]/b[0],  d'[0] = d[0]/b[0]
    # For i > 0:
    #   w = a[i] / b'[i-1]
    #   b'[i] = b[i] - w * c[i-1]
    #   d'[i] = d[i] - w * d'[i-1]

    def forward_step(carry, i):
        b_prev, d_prev = carry
        w      = a[i] / b_prev
        b_curr = b[i] - w * c[i - 1]
        d_curr = d[i] - w * d_prev
        return (b_curr, d_curr), (b_curr, d_curr)

    # Initial values for first cell
    b0 = b[0]
    d0 = d[0]

    _, (b_mod, d_mod) = jax.lax.scan(
        forward_step,
        (b0, d0),
        jnp.arange(1, N)
    )

    # Prepend the first cell values
    b_mod = jnp.concatenate([jnp.array([b0]), b_mod])
    d_mod = jnp.concatenate([jnp.array([d0]), d_mod])

    # --- Back substitution ---
    # x[-1] = d'[-1] / b'[-1]
    # x[i]  = (d'[i] - c[i] * x[i+1]) / b'[i]

    def backward_step(x_next, i):
        x_curr = (d_mod[i] - c[i] * x_next) / b_mod[i]
        return x_curr, x_curr

    x_last = d_mod[-1] / b_mod[-1]

    _, x_interior = jax.lax.scan(
        backward_step,
        x_last,
        jnp.arange(N - 2, -1, -1)   # iterate N-2 down to 0
    )

    # Reverse because scan went backwards
    x = jnp.concatenate([x_interior[::-1], jnp.array([x_last])])
    return x


def pressure_poisson_solve(phi1, phi2, rho1, rho2, u1_star, u2_star, dx, dt):
    """
    Solve for pressure correction dp that makes the corrected velocities
    satisfy volume conservation.

    Interior cells only — phi, rho, u arrays are all shape (N-2,) here,
    representing the N-2 interior cells (excluding boundary cells).

    Returns dp, shape (N-2,), the pressure correction at interior cells.
    """
    N = len(phi1)   # number of interior cells

    # Mobility coefficient: how easily pressure drives volume flux
    # Higher mobility = pressure more effective at redistributing volume
    mob = phi1 / rho1 + phi2 / rho2    # shape (N-2,), units [m³·s/kg]

    # Mobility at faces (average of neighboring cells), shape (N-3,)
    mob_face = 0.5 * (mob[:-1] + mob[1:])

    # Right-hand side: divergence of mixture volumetric flux
    # div(φ₁u₁* + φ₂u₂*) at interior cells, shape (N-2,)
    # We use a simple central-ish difference on the interior cells
    mix_flux = phi1 * u1_star + phi2 * u2_star   # shape (N-2,)

    # Divergence of mix_flux using one-sided differences
    # For the interior-of-interior cells we have neighbors; for edge cells
    # we use one-sided. Here we use a simple centered difference where possible.
    rhs = jnp.zeros(N)
    # Interior of interior: shape (N-4,)
    rhs = rhs.at[1:-1].set((mix_flux[2:] - mix_flux[:-2]) / (2.0 * dx))
    # Edge cells: one-sided
    rhs = rhs.at[0].set((mix_flux[1] - mix_flux[0]) / dx)
    rhs = rhs.at[-1].set((mix_flux[-1] - mix_flux[-2]) / dx)

    rhs = rhs / dt   # scale by 1/dt

    # Build tridiagonal system for the pressure correction
    # Discretization of ∂/∂x[mob * ∂dp/∂x] = rhs
    # At cell i (0-indexed, N interior cells):
    #   main diagonal:  -(mob_face[i] + mob_face[i-1]) / dx²
    #   upper diagonal: mob_face[i] / dx²
    #   lower diagonal: mob_face[i-1] / dx²
    # Boundary conditions: dp = 0 at both ends (Dirichlet)

    dx2 = dx * dx

    # Main diagonal, shape (N,)
    b_diag = jnp.zeros(N)
    b_diag = b_diag.at[0].set(-mob_face[0] / dx2)               # left BC
    b_diag = b_diag.at[-1].set(-mob_face[-1] / dx2)             # right BC
    b_diag = b_diag.at[1:-1].set(
        -(mob_face[1:] + mob_face[:-1]) / dx2
    )
    # Ensure no zeros on main diagonal
    b_diag = jnp.where(jnp.abs(b_diag) < 1e-14,
                        jnp.full_like(b_diag, -1.0), b_diag)

    # Upper diagonal, shape (N,) — c[-1] unused
    c_diag = jnp.zeros(N)
    c_diag = c_diag.at[:-1].set(mob_face / dx2)

    # Lower diagonal, shape (N,) — a[0] unused
    a_diag = jnp.zeros(N)
    a_diag = a_diag.at[1:].set(mob_face / dx2)

    # Set Dirichlet BC: dp = 0 at boundaries means row 0 and row N-1
    # become trivial: 1*dp[0] = 0, 1*dp[-1] = 0
    a_diag = a_diag.at[0].set(0.0)
    b_diag = b_diag.at[0].set(1.0)
    c_diag = c_diag.at[0].set(0.0)
    rhs    = rhs.at[0].set(0.0)

    a_diag = a_diag.at[-1].set(0.0)
    b_diag = b_diag.at[-1].set(1.0)
    c_diag = c_diag.at[-1].set(0.0)
    rhs    = rhs.at[-1].set(0.0)

    dp = solve_tridiagonal(a_diag, b_diag, c_diag, rhs)
    return dp


# =============================================================================
# SECTION 8: STEP 4 — VELOCITY PROJECTION
# =============================================================================
# Correct the intermediate velocities using the pressure gradient:
#
#   u_α = u_α* - (dt / ρ_α) * dp/dx
#
# This subtracts the pressure-driven acceleration that restores volume
# conservation. After this correction, ∂/∂x(φ₁u₁ + φ₂u₂) ≈ 0.

def project_velocities(u1_star, u2_star, dp, phi1, phi2, rho1, rho2, dx, dt):
    """
    Project intermediate velocities onto the volume-conserving subspace.
    
    u1_star, u2_star : intermediate velocities, shape (N-2,)
    dp               : pressure correction, shape (N-2,)
    Returns corrected u1, u2, shape (N-2,).
    """
    # Pressure gradient at interior cells via central difference
    # dp has shape (N-2,); we difference within it
    dp_dx = jnp.zeros_like(dp)
    dp_dx = dp_dx.at[1:-1].set((dp[2:] - dp[:-2]) / (2.0 * dx))
    dp_dx = dp_dx.at[0].set((dp[1] - dp[0]) / dx)
    dp_dx = dp_dx.at[-1].set((dp[-1] - dp[-2]) / dx)

    eps = 1e-10
    u1_new = u1_star - (dt / (rho1 + eps)) * dp_dx
    u2_new = u2_star - (dt / (rho2 + eps)) * dp_dx
    
    # again zeroing out the velocity when below eps_phi
    eps_phi = 1e-6    # threshold below which we consider phase absent
    u1_new = jnp.where(
        phi1 > eps_phi,
        u1_star - (dt / (rho1 + eps)) * dp_dx,
        0.0
    )

    u2_new = jnp.where(
        phi2 > eps_phi,
        u2_star - (dt / (rho2 + eps)) * dp_dx,
        0.0
    )

    return u1_new, u2_new

# function to only recover velocity where phase is presentt, otherwise return zero without division
# prevents divide by zero issues when phi is really small
def safe_velocity(mom, phi, rho, eps_phi=1e-6):
    """
    Only recover velocity where phase is present.
    Everywhere else return zero without division.
    """
    return jnp.where(
        phi > eps_phi,
        mom / (phi * rho + 1e-10),
        0.0
    )


# =============================================================================
# SECTION 9: BOUNDARY CONDITIONS
# =============================================================================
# Boundary cells (index 0 and N-1) are not part of the interior solve.
# We set them here to enforce:
#   Inlet (left, x=0):  fixed pressure p_inlet, zero-gradient velocity
#   Outlet (right, x=L): fixed pressure p_outlet, zero-gradient velocity
#
# Zero-gradient means: copy the value from the nearest interior cell.
# This is a common "outflow" BC — it lets the flow exit without reflection.
#
# JAX arrays are immutable, so .at[].set() returns a NEW array.
# This is not in-place mutation — JAX traces these as functional updates.
    
def apply_boundary_conditions(state, phi1_int, phi2_int,
                               u1_int, u2_int, p_inlet, p_outlet,
                               phi1_inlet=0.95, phi2_min=1e-4):
    
    # Interior volume fractions
    phi1_interior = phi1_int
    phi2_interior = phi2_int
    
    # Inlet boundary: fix to initial composition
    # This represents fresh sludge entering at constant composition
    phi1_inlet_val = jnp.array([phi1_inlet])
    phi2_inlet_val = jnp.array([1.0 - phi1_inlet])
    
    # Outlet boundary: zero-gradient (let whatever is there exit freely)
    phi1_outlet_val = phi1_int[-1:]
    phi2_outlet_val = phi2_int[-1:]
    
    # Assemble full arrays
    phi1 = jnp.concatenate([phi1_inlet_val, phi1_interior, phi1_outlet_val])
    phi2 = jnp.concatenate([phi2_inlet_val, phi2_interior, phi2_outlet_val])
    
    # Apply floor and re-normalize
    phi2 = jnp.maximum(phi2, phi2_min)
    phi1 = 1.0 - phi2
    
    # Velocities — inlet: fix to zero-gradient but also
    # clamp u2 at inlet to prevent boundary blowup
    u1 = jnp.concatenate([u1_int[:1], u1_int, u1_int[-1:]])
    u2 = jnp.concatenate([u2_int[:1], u2_int, u2_int[-1:]])
    
    # Pressure
    p_int = state['p'][1:-1]
    p = jnp.concatenate([jnp.array([p_inlet]),
                          p_int,
                          jnp.array([p_outlet])])
    
    return dict(phi1=phi1, phi2=phi2, u1=u1, u2=u2, p=p,
                rho1=state['rho1'], rho2=state['rho2'])


# =============================================================================
# SECTION 10: FULL TIME STEP
# =============================================================================
# Combines all steps into one function that advances the state by dt.
# This is the function JAX will JIT-compile into a single XLA kernel.

#def time_step(state, dt, dx, drag_coeff, p_inlet, p_outlet):
def time_step(state, dt, dx, drag_coeff, D, p_inlet, p_outlet, d_b, mu1, mu2, theta, phi1_inlet):
    """
    Advance the simulation state by one time step dt.

    Sequence:
      1. Advance mass equations         → phi1_int, phi2_int  (N-2,)
      2. Advance momentum equations     → u1_star, u2_star    (N-2,)
      3. Pressure Poisson solve         → dp                  (N-2,)
      4. Project velocities             → u1_int, u2_int      (N-2,)
      5. Apply BCs + volume cleanup     → full state          (N,)
    """
    rho1_int = state['rho1'][1:-1]
    rho2_int = state['rho2'][1:-1]

    # Step 1: advance mass
    phi1_int, phi2_int = advance_mass(state, dt, dx)

    # Step 2: advance momentum (intermediate velocities)
    mom1_new, mom2_new = advance_momentum(state, dt, dx, drag_coeff, d_b, D, mu1, mu2, theta)
    
    # calculate intermediate velocities from momentum, avoiding division by zero when phase is absent
    eps_phi = 1e-6
    eps     = 1e-10

    u1_star = jnp.where(
        phi1_int > eps_phi,
        mom1_new / (phi1_int * rho1_int + eps),
        0.0)

    u2_star = jnp.where(
        phi2_int > eps_phi,
        mom2_new / (phi2_int * rho2_int + eps),
        0.0)

    # Step 3: pressure Poisson solve for volume conservation
    dp = pressure_poisson_solve(
        phi1_int, phi2_int,
        rho1_int, rho2_int,
        u1_star, u2_star,
        dx, dt
    )

    # Step 4: project velocities to be volume-conserving
    u1_int, u2_int = project_velocities(
        u1_star, u2_star, dp,
        phi1_int, phi2_int,
        rho1_int, rho2_int,
        dx, dt
    )

    # Step 5: apply BCs and volume fraction cleanup
    #new_state = apply_boundary_conditions(
    #    state, phi1_int, phi2_int, u1_int, u2_int,
    #    p_inlet, p_outlet
    #) 
    new_state = apply_boundary_conditions(
        state, phi1_int, phi2_int, u1_int, u2_int,
        p_inlet, p_outlet, phi1_inlet=phi1_inlet
    )

    return new_state


# =============================================================================
# SECTION 11: DIAGNOSTICS
# =============================================================================
# Functions to check physical conservation laws during the run.
# These are your sanity checks — if they drift, something is wrong.

def compute_diagnostics(state, dx):
    """
    Compute key diagnostic quantities for monitoring the simulation.
    
    Returns a dict of scalar values.
    """
    phi1 = state['phi1']
    phi2 = state['phi2']
    rho1 = state['rho1']
    rho2 = state['rho2']
    u1   = state['u1']
    u2   = state['u2']

    # Total mass of each phase (should be conserved, modulo BCs)
    total_mass1 = jnp.sum(phi1 * rho1) * dx
    total_mass2 = jnp.sum(phi2 * rho2) * dx

    # Volume fraction constraint violation — should be ~0 everywhere
    vol_error = jnp.max(jnp.abs(phi1 + phi2 - 1.0))

    # Mixture volumetric flux divergence — should be ~0 (volume conservation)
    mix_flux = phi1 * u1 + phi2 * u2
    div_mix_flux = jnp.max(jnp.abs(jnp.diff(mix_flux) / dx))

    return dict(
        total_mass1=total_mass1,
        total_mass2=total_mass2,
        vol_error=vol_error,
        div_mix_flux=div_mix_flux,
        max_u1=jnp.max(jnp.abs(u1)),
        max_u2=jnp.max(jnp.abs(u2)),
    )


# =============================================================================
# SECTION 12: Neural Network Definition for Drag Closure
# =============================================================================

# ── Network architecture ───────────────────────────────────────────────────────
class DragClosureNetwork(eqx.Module):
    """
    Phase 1: learns effective drag coefficient C_D_eff
    from 4 scaled dimensional flow inputs.

    Input:  [phi1, u1/1.5, u2/1.5, delta_u/0.1]  shape (4,)
    Output: C_D_eff > 0                            scalar
    """
    layer1: eqx.nn.Linear
    layer2: eqx.nn.Linear
    layer3: eqx.nn.Linear
    layer4: eqx.nn.Linear

    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.layer1 = eqx.nn.Linear(4,  32, key=k1)
        self.layer2 = eqx.nn.Linear(32, 32, key=k2)
        self.layer3 = eqx.nn.Linear(32, 16, key=k3)
        self.layer4 = eqx.nn.Linear(16,  1, key=k4)

    def __call__(self, x):
        x = jax.nn.tanh(self.layer1(x))
        x = jax.nn.tanh(self.layer2(x))
        x = jax.nn.tanh(self.layer3(x))
        x = self.layer4(x)
        return jax.nn.softplus(x[0])


# ── Input feature builder ──────────────────────────────────────────────────────
def build_network_inputs(phi1, u1, u2):
    """
    Build 4-dimensional scaled input vector.

    Scaling brings all inputs to O(1):
        phi1:    already O(1)
        u1, u2:  O(0.3-1.5) m/s  → divide by 1.5
        delta_u: O(0.01-0.1) m/s → divide by 0.1
    """
    delta_u = u2 - u1
    return jnp.array([
        phi1,
        u1      / 1.5,
        u2      / 1.5,
        delta_u / 0.1,
    ])


# ── time_step wrapper that takes drag_c as explicit argument ───────────────────
# Your existing time_step captures drag_coeff from outer scope.
# This wrapper makes drag_c an explicit argument so jax.grad can
# differentiate through it into the network weights.
def time_step_learned(state, dt, dx, drag_c, D,
                       p_inlet, p_outlet, d_b, mu1, mu2,
                       theta, phi1_inlet):
    """
    Identical physics to your existing time_step.
    Only difference: drag_c is an explicit argument, not captured
    from outer scope — this is what makes it differentiable.
    """
    rho1_int = state['rho1'][1:-1]
    rho2_int = state['rho2'][1:-1]

    phi1_int, phi2_int = advance_mass(state, dt, dx)

    mom1_new, mom2_new = advance_momentum(
        state, dt, dx, drag_c, d_b, D, mu1, mu2, theta
    )

    eps_phi = 1e-6
    eps     = 1e-10

    u1_star = jnp.where(
        phi1_int > eps_phi,
        mom1_new / (phi1_int * rho1_int + eps),
        0.0
    )
    u2_star = jnp.where(
        phi2_int > eps_phi,
        mom2_new / (phi2_int * rho2_int + eps),
        0.0
    )

    dp = pressure_poisson_solve(
        phi1_int, phi2_int,
        rho1_int, rho2_int,
        u1_star, u2_star,
        dx, dt
    )

    u1_int, u2_int = project_velocities(
        u1_star, u2_star, dp,
        phi1_int, phi2_int,
        rho1_int, rho2_int,
        dx, dt
    )

    new_state = apply_boundary_conditions(
        state, phi1_int, phi2_int,
        u1_int, u2_int,
        p_inlet, p_outlet,
        phi1_inlet=phi1_inlet
    )

    return new_state


# ── Loss function ──────────────────────────────────────────────────────────────
def loss_fn(network, condition, n_window, dt_fixed,
             dx, D, d_b, mu1, mu2, theta, N):
    """
    Phase 1 loss function using slip velocity as the training signal.
    
    Why slip instead of phi1/Um:
        Drag directly controls the velocity difference between phases.
        When drag increases, phases are pulled together → slip decreases.
        When drag decreases, phases move independently → slip increases.
        This direct physical connection means slip has a nonzero gradient
        w.r.t. drag_coeff, unlike phi1 and Um which are dominated by
        wall friction and pressure gradient terms.

    The network outputs C_D_eff → simulation runs with that drag →
    simulation produces a slip velocity → we compare against the slip
    produced by the known true drag_coeff → loss drives network toward
    outputting the true C_D_eff.
    """
    # ── Step 1: get C_D_eff from network ──────────────────────────────────────
    # Build input features from plateau-averaged conditions
    # These are the same quantities we'd have access to in Phase 2
    # from experimental measurements
    x = build_network_inputs(
        jnp.array(condition['WC']),
        jnp.array(condition['u1_mean']),
        jnp.array(condition['u2_mean']),
    )
    C_D_eff = network(x)

    # ── Step 2: run differentiable simulation window with learned C_D ─────────
    # This is the core of the physics-informed approach —
    # the network's output C_D_eff flows into the simulation,
    # and JAX traces gradients all the way back through every
    # time step into the network weights
    def learned_step(state, dt):
        new_state = time_step_learned(
            state, dt, dx, C_D_eff, D,
            jnp.array(condition['p_inlet']),
            jnp.array(condition['p_outlet']),
            d_b, mu1, mu2, theta,
            jnp.array(condition['phi1_inlet'])
        )
        return new_state, None

    final, _ = jax.lax.scan(
        learned_step,
        condition['spinup_state'],
        jnp.full(n_window, dt_fixed),
    )

    # ── Step 3: extract plateau quantities from simulation output ──────────────
    i_start = N // 4
    i_end   = 3 * N // 4

    phi1_p = final['phi1'][i_start:i_end]
    phi2_p = final['phi2'][i_start:i_end]
    u1_p   = final['u1'][i_start:i_end]
    u2_p   = final['u2'][i_start:i_end]

    # Slip velocity — what the simulation produces with C_D_eff
    # This is what we compare against the known target
    slip_pred = jnp.mean(u2_p - u1_p)

    # Also compute phi1 and Um — not used in loss but useful for monitoring
    phi1_pred = jnp.mean(phi1_p)
    Um_pred   = jnp.mean(phi1_p * u1_p + phi2_p * u2_p)

    # ── Step 4: compute loss ───────────────────────────────────────────────────
    slip_target = jnp.array(condition['slip_target'])

    # Normalized squared error on slip velocity
    # Normalizing by |slip_target| makes the loss dimensionless and
    # comparable across conditions with different slip magnitudes —
    # a 10% error at WC=0.2 contributes the same as a 10% error at WC=0.8
    loss_slip = ((slip_pred - slip_target)
                 / (jnp.abs(slip_target) + 1e-6))**2

    # Return loss and auxiliary quantities for monitoring during training
    # has_aux=True in the training loop expects this tuple structure
    return loss_slip, (phi1_pred, Um_pred, slip_pred, C_D_eff)


# =============================================================================
# SECTION 12: MAIN — RUN THE SIMULATION
# =============================================================================

if __name__ == "__main__":
    '''
    # --- Physical parameters --- 
    L        = 1.0      # pipe length [m]
    N        = 100      # number of cells
    rho1_val = 1000.0   # density of phase 1 (e.g. water) [kg/m³]
    rho2_val = 1.0      # density of phase 2 (e.g. air)   [kg/m³]
    phi1_0   = 1.0   # initial volume fraction of phase 1
    p_inlet  = 1.01e5   # inlet pressure [Pa]  (slightly above atmospheric)
    p_outlet = 1.00e5   # outlet pressure [Pa] (atmospheric)
    drag_coeff = 50.0  # inter-phase drag coefficient [kg/(m³·s)]
    t_end    = 2.0      # simulation end time [s]
    
    # --- Sludge parameters ---
    L          = 1.0      # pipe length [m]
    N          = 100      # number of cells
    rho1_val   = 1000.0   # water [kg/m³]
    rho2_val   = 1050.0   # suspended solids [kg/m³]
    phi1_0     = 0.95     # 95% water
    p_inlet    = 1.01e5    # [Pa]
    p_outlet   = 1.0000e5    # [Pa]
    #drag_coeff = 50000.0   # [kg/(m³·s)]
    drag_coeff = 0.44     # [kg/(m³·s)] — use with the new drag model
    d_b       = 1e-3     # effective particle diameter for drag [m]
    t_end      = 6.0      # [s]
    #dt_max     = 1e-4
    '''
    
    
    # --- Oil (EXXSOL D140) parameters, from Ibarra Paper ---
    L          = 6.7      # pipe length [m]
    D          = 0.032    # pipe diameter [m] — matches their 32mm test section
    N          = 500      # number of cells
    theta      = 0.0     # pipe inclination angle from horizontal [radians], # 0 = horizontal, π/2 = vertical
    rho1_val   = 998.0   # water [kg/m³]
    rho2_val   = 825.0   # oil [kg/m³]
    
    phi1_0     = 0.3156   # 30% water
    # defining phase composition of inlet BC when have plugging
    phi1_inlet_bc = 0.3156    # ≈ 0.05
    
    dpdz_pa     = 91.411 # taken/calculated from figure 10
    #p_inlet    = 1.0001e5    # [Pa]
    p_outlet   = 1.0000e5    # [Pa]
    
    mu1        = 5.4E-3    # oil [Pa·s]
    mu2        = 0.9E-3     # water [Pa·s]
    #drag_coeff = 50000.0   # [kg/(m³·s)]
    #drag_coeff = 0.001     # [kg/(m³·s)] — use with the new drag model
    drag_coeff = 1E-04
    d_b       = 1e-3     # effective particle diameter for drag [m]
    t_end      = 150.00     # [s]
    #t_end      = 200.00     # [s]
    #dt_max     = 1e-4
    
    delta_p = dpdz_pa * L
    p_inlet = p_outlet + delta_p
    

    '''
    # modeling gas-liquid annular flow in a 1.5-inch pipe, with the drag model that includes bubbly, slug, and annular regimes
    p = bb_to_model_inputs(
    GL          = 33.95,    # lbm/(ft²·s)
    GG          = 1.76,     # lbm/(ft²·s)
    P_psia      = 88.88,
    T_F         = 69.0,
    theta_rad   = 0.0,
    D_m         = 0.0381,   # 1.5-inch pipe
    dpdz_psi_ft = 0.0041,   # directly from DP/DZ MEAS. column
)
    
    # Unpack into the local names the rest of the code expects
    L          = p['L']
    N          = p['N']
    D          = p['D_m']
    theta      = p['theta']
    rho1_val   = p['rho1_val']
    rho2_val   = p['rho2_val']
    mu1        = p['mu1']
    mu2        = p['mu2']
    p_inlet    = p['p_inlet']
    p_outlet   = p['p_outlet']
    drag_coeff = p['drag_coeff']
    d_b        = p['d_b']
    t_end      = p['t_end']
    D          = p['D_m'] # pipe diameter [m]
    N          = 100      # number of cells
    '''
    # --- Grid ---
    dx, x = make_grid(L, N)

    # Slug flow with oil-water properties
    #phi1_0 = jnp.where(x < L/2, 0.3, 0.9)   # oil-rich left, water-rich right
    #phi2_0 = 1.0 - phi1_0

    #phi1_0 = p['phi1_0']   # uniform initial condition from f_L

    # --- Initial conditions ---
    state = initial_conditions(
        x, L, phi1_0,
        rho1_val, rho2_val,
        p_inlet, p_outlet
    )
    
    '''
    # --- Initial conditions for slug verification problem ---
    state = initial_conditions_slug(
        x, L, rho1_val, rho2_val,
        p_inlet, p_outlet
    )
    '''    
    '''
    print("=" * 60)
    print("1D Two-Phase Mixture Flow Solver")
    print("=" * 60)
    print(f"  Grid:       N={N} cells, dx={dx:.4f} m")
    print(f"  Phase 1:    ρ={rho1_val} kg/m³, φ₀={phi1_0}")
    print(f"  Phase 2:    ρ={rho2_val} kg/m³, φ₀={1-phi1_0:.2f}")
    print(f"  ΔP drive:   {p_inlet - p_outlet:.0f} Pa")
    print(f"  Drag coeff: {drag_coeff}")
    print(f"  t_end:      {t_end} s")
    
    print()
    '''

    print("=" * 60)
    print("1D Two-Phase Mixture Flow Solver")
    print("=" * 60)
    print(f"  Grid:       N={N} cells, dx={dx:.4f} m")
    print(f"  Phase 1:    ρ={rho1_val} kg/m³")
    print(f"  Phase 2:    ρ={rho2_val} kg/m³")

    # Handle phi1_0 being either a scalar or an array
    if hasattr(phi1_0, 'shape') and phi1_0.ndim > 0:
        # It's an array — print summary statistics instead
        print(f"  phi1_0:     min={float(phi1_0.min()):.3f}  "
            f"max={float(phi1_0.max()):.3f}  "
            f"(slug flow initial condition)")
        print(f"  phi2_0:     min={float(phi2_0.min()):.3f}  "
            f"max={float(phi2_0.max()):.3f}")
    else:
        # It's a scalar — format normally
        print(f"  phi1_0:     {phi1_0:.3f}")
        print(f"  phi2_0:     {1-phi1_0:.3f}")

    print(f"  ΔP drive:   {p_inlet - p_outlet:.0f} Pa")
    print(f"  Drag coeff: {drag_coeff}")
    print(f"  t_end:      {t_end} s")
    print()

    # JIT-compile the time step function.
    # JAX traces time_step once on the first call, compiles it to an XLA
    # kernel, and then every subsequent call runs the compiled version.
    # This is why JAX is fast — after the first step, there's no Python
    # overhead in the time loop at all.
    #step_jit = jax.jit(
    #    lambda s, dt: time_step(s, dt, dx, drag_coeff, p_inlet, p_outlet)
    #)

    # use the below for modeling air and water in annular flow
    # Your existing lambda — keep this for reference but won't use in scan
    step_jit = jax.jit(
        lambda s, dt: time_step(s, dt, dx, drag_coeff, D,
                                p_inlet, p_outlet, d_b, mu1, mu2, theta, phi1_inlet_bc)
    )

    # New scan step — same closure pattern, scan-compatible signature
    def scan_step(state, dt):
        """
        Wraps time_step for jax.lax.scan.
        All args except state and dt captured from outer scope via closure.
        Returns (new_state, new_state) — carry and stacked output.
        """
        new_state = time_step(
            state, dt, dx, drag_coeff, D,
            p_inlet, p_outlet, d_b, mu1, mu2,
            theta, phi1_inlet_bc
        )
        return new_state, new_state

    # JIT compile the scan step once
    scan_step_jit = jax.jit(scan_step)
    
    def scan_step(state, dt):
        new_state = time_step(
            state, dt, dx, drag_coeff, D,
            p_inlet, p_outlet, d_b, mu1, mu2,
            theta, phi1_inlet_bc
        )
        return new_state, None   # None saves memory vs returning full state
    
    # --- Storage for output ---
    # use when use while loop with save_every
    #save_every  = 5E4   # save state every N steps
    saved_times = []
    saved_phi1  = []
    saved_u1    = []
    saved_u2    = []
    
    saved_phi2 = []
    

    # --- Time loop ---
    t      = 0.0
    step_n = 0

    # ── Parameters ─────────────────────────────────────────────────────────────────
    dt_fixed = 1e-4      # [s] — your existing dt_max, safe fixed value

    t_spinup =   150.0     # [s] — from your experiments, converges by ~140s
    #t_spinup =   5.0     # [s] — from your experiments, converges by ~140s
    t_window =   1.0     # [s] — short differentiable window
    
    # Replace your step counts with these:
    #t_spinup = 1.0      # just 1 second instead of 150
    #t_window = 0.1      # just 0.1 seconds


    n_spinup = int(t_spinup / dt_fixed)   # 1,500,000
    n_window = int(t_window / dt_fixed)   # 10,000

    print(f"Spinup:  {n_spinup:,} steps  ({t_spinup:.0f}s)")
    print(f"Window:  {n_window:,} steps  ({t_window:.0f}s)")

    # use the below when use the differentiable window approach
    save_every_window = 1000    # save state every this many steps
    n_chunks          = n_window // save_every_window


    # ── Phase 1: Spinup (no gradient tracking) ─────────────────────────────────────
    print("\nRunning spinup...")
    
    # ── Define scan step (no jit here — scan handles it) ──────────────────────────
    def scan_step(state, dt):
        new_state = time_step(
            state, dt, dx, drag_coeff, D,
            p_inlet, p_outlet, d_b, mu1, mu2,
            theta, phi1_inlet_bc
        )
        return new_state, None   # None saves memory

    # ── Define jitted spinup function ─────────────────────────────────────────────
    @jax.jit
    def run_spinup_full(init_state):
        final, _ = jax.lax.scan(
            scan_step,
            init_state,
            jnp.full(n_spinup, dt_fixed),
        )
        return final

    #### define once here then will use at almost every step to run chunks #######
    @jax.jit
    def run_chunk(init_state):
        final, _ = jax.lax.scan(
            scan_step,
            init_state,
            jnp.full(save_every_window, dt_fixed),
        )
        return final


    # ── 3. Run spinup ──────────────────────────────────────────────────────────────
    import time
    t0 = time.time()

    spinup_state = run_spinup_full(state)
    spinup_state['phi1'].block_until_ready()

    print(f"Spinup complete in {time.time()-t0:.1f}s")
    print(f"  phi1: [{float(spinup_state['phi1'].min()):.6f}, "
        f"{float(spinup_state['phi1'].max()):.6f}]")
    print(f"  u1={float(jnp.max(jnp.abs(spinup_state['u1']))):.4f}  "
        f"u2={float(jnp.max(jnp.abs(spinup_state['u2']))):.4f}")

    spinup_state = jax.lax.stop_gradient(spinup_state)

    
    # ── Phase 2: Differentiable window ────────────────────────────────────────────
    # Run in chunks so we can save snapshots for plotting
    # Each chunk = save_every_window steps

    saved_times = []
    saved_phi1  = []
    saved_phi2  = []
    saved_u1    = []
    saved_u2    = []

    current_state = spinup_state
    t_current     = t_spinup

    print("\nRunning differentiable window...")

    for chunk_idx in range(n_chunks):
        t0 = time.time()
        
        # ← replace the old bare jax.lax.scan call with run_chunk
        current_state = run_chunk(current_state)
        current_state['phi1'].block_until_ready()

        t_current += save_every_window * dt_fixed

        saved_times.append(t_current)
        saved_phi1.append(np.array(current_state['phi1']))
        saved_phi2.append(np.array(current_state['phi2']))
        saved_u1.append(np.array(current_state['u1']))
        saved_u2.append(np.array(current_state['u2']))

        print(f"  t={t_current:.3f}s  "
            f"phi1=[{float(current_state['phi1'].min()):.4f}, "
            f"{float(current_state['phi1'].max()):.4f}]  "
            f"u1={float(jnp.max(jnp.abs(current_state['u1']))):.4f}  "
            f"u2={float(jnp.max(jnp.abs(current_state['u2']))):.4f}")

    final_state = current_state
    
    
    
    # ══════════════════════════════════════════════════════════════════════════════
    # Section 7c — Validation of spinup and plateau extraction (to confirm grads work before adding NN)
    # ══════════════════════════════════════════════════════════════════════════════
    '''
    i_start = N // 4
    i_end   = 3 * N // 4

    phi1_p  = final_state['phi1'][i_start:i_end]
    phi2_p  = final_state['phi2'][i_start:i_end]
    u1_p    = final_state['u1'][i_start:i_end]
    u2_p    = final_state['u2'][i_start:i_end]
    rho1_p  = final_state['rho1'][i_start:i_end]
    rho2_p  = final_state['rho2'][i_start:i_end]

    phi1_plateau = float(jnp.mean(phi1_p))
    u1_mean      = float(jnp.mean(u1_p))
    u2_mean      = float(jnp.mean(u2_p))
    Um_vol       = float(jnp.mean(phi1_p * u1_p + phi2_p * u2_p))
    rho_mix_p    = phi1_p * rho1_p + phi2_p * rho2_p
    Um_mom       = float(jnp.mean(
        (phi1_p * rho1_p * u1_p + phi2_p * rho2_p * u2_p)
        / (rho_mix_p + 1e-10)
    ))

    Um_target = 0.50   # update per test condition

    print(f"\n{'='*45}")
    print(f"  phi1 plateau:   {phi1_plateau:.4f}  (input was {phi1_inlet_bc:.4f})")
    print(f"  u1 (water):     {u1_mean:.4f} m/s")
    print(f"  u2 (oil):       {u2_mean:.4f} m/s")
    print(f"  slip (u2-u1):   {u2_mean - u1_mean:+.4f} m/s")
    print(f"  U_m vol:        {Um_vol:.4f} m/s")
    print(f"  U_m momentum:   {Um_mom:.4f} m/s")
    print(f"  U_m target:     {Um_target:.4f} m/s")
    print(f"  Error (vol):    {(Um_vol  - Um_target)/Um_target*100:+.2f}%")
    print(f"  Error (mom):    {(Um_mom  - Um_target)/Um_target*100:+.2f}%")
    print(f"{'='*45}")

    # ── Gradient flow sanity check ─────────────────────────────────────────────────
    # Verify jax.grad can differentiate through the window before adding NN.
    # Run this once after the loop to confirm infrastructure is ready.

    def gradient_check(spinup_state, drag_coeff_val):
        """
        Check that d(phi1_plateau)/d(drag_coeff) is nonzero.
        If zero: gradients are not flowing — fix before adding network.
        If nonzero: infrastructure is ready for neural network.
        """
        def loss_from_drag(drag_c):
            # Tiny 500-step scan with drag_c as differentiable input
            def step(s, dt):
                new_s = time_step(
                    s, dt, dx, drag_c, D,
                    p_inlet, p_outlet, d_b,
                    mu1, mu2, theta, phi1_inlet_bc
                )
                return new_s, None

            final, _ = jax.lax.scan(
                step,
                spinup_state,
                jnp.full(500, dt_fixed),
            )
            return jnp.mean(final['phi1'][N//4:3*N//4])

        grad = jax.grad(loss_from_drag)(jnp.array(drag_coeff_val))
        print(f"\nGradient check:")
        print(f"  d(phi1)/d(drag_coeff) = {float(grad):.6e}")
        if abs(float(grad)) > 1e-12:
            print(f"  ✓ Gradients flowing — ready for neural network")
        else:
            print(f"  ✗ Zero gradient — investigate before adding network")
        return grad

    gradient_check(spinup_state, drag_coeff)
    '''
    
    '''
    while t < t_end:
        # Compute stable dt from CFL condition (adaptive time stepping)
        dt = float(compute_dt(state, dx, mu1, mu2, rho1_val, rho2_val, cfl=0.4, dt_max=1e-4))
        dt = min(dt, t_end - t)   # don't overshoot t_end
        
        

        # Advance one time stepr
        state = step_jit(state, dt)
        t     += dt
        step_n += 1

        # Diagnostics every save_every steps
        if step_n % save_every == 0:
            diag = compute_diagnostics(state, dx)
            print(f"  t={t:.4f}s  step={step_n:5d}  "
                f"dt={dt:.2e}  "
                f"vol_err={diag['vol_error']:.2e}  "
                f"max|u1|={diag['max_u1']:.4f}  "
                f"max|u2|={diag['max_u2']:.4f}")
        
            saved_times.append(t)
            saved_phi1.append(np.array(state['phi1']))
            saved_u1.append(np.array(state['u1']))
            saved_u2.append(np.array(state['u2']))
            
            saved_phi2.append(np.array(state['phi2']))
            
            # Add to time loop
            if step_n % save_every == 0:
                print(f"phi1 min={float(state['phi1'].min()):.6f} "
                    f"phi1 max={float(state['phi1'].max()):.6f} "
                    f"phi2 min={float(state['phi2'].min()):.6f} "
                    f"phi2 max={float(state['phi2'].max()):.6f}")
                
    '''
        

    print(f"\nDone. {step_n} steps completed.")
    

    
    ############################################
        
    # ══════════════════════════════════════════════════════════════════════════════
    # Section 8 — Visualization of results
    # ══════════════════════════════════════════════════════════════════════════════
    '''
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    fig.patch.set_facecolor('#0f0f1a')
    x_np = np.array(x)

    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(saved_times)))

    titles  = ['Volume Fraction φ₁', 'Velocity u₁ [m/s]', 'Velocity u₂ [m/s]']
    data    = [saved_phi1, saved_u1, saved_u2]
    ylabels = ['φ₁', 'u₁ [m/s]', 'u₂ [m/s]']

    for ax, title, series, ylabel in zip(axes, titles, data, ylabels):
        ax.set_facecolor('#0f0f1a')
        for i, (arr, color) in enumerate(zip(series, colors)):
            label = f't={saved_times[i]:.3f}s' if i % 3 == 0 else None
            ax.plot(x_np, arr, color=color, linewidth=1.5,
                    alpha=0.8, label=label)
        ax.set_xlabel('x [m]', color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.set_title(title, color='white')
        ax.tick_params(colors='white')
        ax.spines[:].set_color('#333355')
        if any(l is not None for l in [ylabel]):
            ax.legend(facecolor='#1a1a2e', labelcolor='white',
                      fontsize=8, loc='best')

    plt.tight_layout(pad=2.0)
    plt.savefig("mixture_flow_result.png", dpi=150,
                bbox_inches='tight', facecolor='#0f0f1a')
    plt.show()
    print("Plot saved to mixture_flow_result.png")

############## for validating data with the Ibarra Paper ########
##### input pressure gradient into model, water cut (vol fraction), 
##### and extract u1, u2, to compare with experimental data from the paper.

# Check that mixture velocity is actually constant along pipe
# (confirms your simulation has reached steady state)
Um_profile = state['phi1'] * state['u1'] + state['phi2'] * state['u2']

print(f"U_m at inlet:   {Um_profile[5]:.4f} m/s")
print(f"U_m at midpipe: {Um_profile[N//2]:.4f} m/s")
print(f"U_m at outlet:  {Um_profile[-5]:.4f} m/s")
print(f"Variation:      {jnp.std(Um_profile):.4f} m/s")


# After simulation reaches steady state, extract plateau region
# Use middle 50% of pipe to avoid inlet/outlet boundary effects
i_start = N // 4      # 25% along pipe
i_end   = 3 * N // 4  # 75% along pipe

phi1_plateau = jnp.mean(state['phi1'][i_start:i_end])
phi2_plateau = jnp.mean(state['phi2'][i_start:i_end])
u1_plateau   = jnp.mean(state['u1'][i_start:i_end])
u2_plateau   = jnp.mean(state['u2'][i_start:i_end])

# Mixture velocity — volume-fraction weighted average of phase velocities
U_m_predicted = phi1_plateau * u1_plateau + phi2_plateau * u2_plateau

print(f"phi1 plateau:    {phi1_plateau:.4f}  (input was {phi1_0})")
print(f"phi2 plateau:    {phi2_plateau:.4f}")
print(f"u1 (water):      {u1_plateau:.4f} m/s")
print(f"u2 (oil):        {u2_plateau:.4f} m/s")
print(f"U_m predicted:   {U_m_predicted:.4f} m/s")
print(f"U_m target:      0.5000 m/s")
print(f"Error:           {(U_m_predicted - 0.5)/0.5 * 100:.2f}%")
'''
# =============================================================================
# APPENDIX: UPGRADING TO RK2 TIME INTEGRATION
# =============================================================================
# When you're ready to improve accuracy, replace the euler_step calls with
# this Heun's method (RK2). The structure is identical — you just evaluate
# the RHS twice and average the result.
#
# def rk2_step(state, dt, dx, drag_coeff, p_inlet, p_outlet):
#     """Heun's method (explicit RK2) — second-order accurate in time."""
#     # Stage 1: full Euler step to get intermediate state k1
#     k1 = time_step(state, dt, dx, drag_coeff, p_inlet, p_outlet)
#
#     # Stage 2: Euler step from k1
#     k2 = time_step(k1, dt, dx, drag_coeff, p_inlet, p_outlet)
#
#     # Average the two stages — this is the RK2 correction
#     # In practice: average the state arrays from state and k2
#     def avg(a, b): return 0.5 * (a + b)
#
#     return jax.tree.map(avg, state, k2)
#
# The time loop then becomes:
#   state = rk2_step(state, dt, dx, drag_coeff, p_inlet, p_outlet)
#
# RK2 is second-order accurate (error ~ dt²) vs Euler's first-order (error ~ dt).
# This means you can use larger dt for the same accuracy, or get much better
# accuracy at the same dt. The cost is two RHS evaluations per step instead of one.


"""
validation_plot.py
------------------
Generates two validation plots comparing your two-fluid model predictions
against Ibarra et al. (2015) experimental data.

    Plot 1 — WC vs Mixture Velocity (one subplot per Um)
             Shows predicted Um against the experimental target line
             for each water cut tested.

    Plot 2 — Parity plot (Um measured vs Um predicted)
             All conditions on one plot, colored by mixture velocity,
             with 1:1 line and ±10% error bands.

CSV FORMAT EXPECTED
-------------------
One file per mixture velocity, e.g. ibarra_Um_0p50.csv
Columns:
    WC               — water cut (input volume fraction)
    dpdz_measured    — pressure gradient Pa/m from Figure 10
    Um_target        — target mixture velocity m/s (constant per file)
    flow_regime      — flow regime string from Figure 6 (SW, SWD, DC, etc.)
    phi1_predicted   — model steady-state water volume fraction
    u1_predicted     — model steady-state water velocity m/s
    u2_predicted     — model steady-state oil velocity m/s
    Um_predicted     — model mixture velocity = phi1*u1 + phi2*u2

USAGE
-----
1. Fill in phi1_predicted, u1_predicted, u2_predicted, Um_predicted
   columns in each CSV after running your simulation for each row.
2. Run:  python validation_plot.py
3. Plots saved as:
       validation_wc_vs_Um.png
       validation_parity.png
"""
###### Data Generation - Generating Synthetic Dataset for Training ###############
# ══════════════════════════════════════════════════════════════════════════════
# Section 7a — Synthetic dataset generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(drag_coeff_true, conditions,
                                n_spinup, dt_fixed,
                                dx, D, d_b, mu1, mu2,
                                theta, N,
                                rho1_val, rho2_val):
    """
    Run simulation at known drag_coeff_true for each condition.
    Saves phi1_plateau, Um, u1_mean, u2_mean as training targets
    and the converged spinup state for use during training.
    """
    dataset = []

    for idx, cond in enumerate(conditions):
        WC   = cond['WC']
        dpdz = cond['dpdz']
        Um_t = cond['Um_target']

        p_out = 1.0000e5
        p_in  = p_out + dpdz * L

        print(f"\n  [{idx+1}/{len(conditions)}] "
            f"WC={WC:.1f}  dpdz={dpdz:.1f} Pa/m  "
            f"Um_target={Um_t:.2f} m/s")

        # Initial state for this condition
        dx_c, x_c = make_grid(L, N)
        state_c = initial_conditions(
            x_c, L, WC,
            rho1_val, rho2_val,
            p_in, p_out
        )

        # Scan step with known true drag
        def make_scan_step(p_in_c, p_out_c, WC_c):
            def scan_step_c(state, dt):
                new_state = time_step_learned(
                    state, dt, dx_c, drag_coeff_true, D,
                    p_in_c, p_out_c, d_b, mu1, mu2,
                    theta, WC_c
                )
                return new_state, None
            return scan_step_c

        scan_fn = make_scan_step(p_in, p_out, WC)

        run_spinup_c = jax.jit(lambda s: jax.lax.scan(
            scan_fn, s, jnp.full(n_spinup, dt_fixed)
        )[0])

        t0 = time.time()
        spinup = run_spinup_c(state_c)
        spinup['phi1'].block_until_ready()
        spinup = jax.lax.stop_gradient(spinup)
        print(f"    Spinup done in {time.time()-t0:.1f}s")

        # Extract plateau quantities
        i_start = N // 4
        i_end   = 3 * N // 4

        phi1_p = spinup['phi1'][i_start:i_end]
        phi2_p = spinup['phi2'][i_start:i_end]
        u1_p   = spinup['u1'][i_start:i_end]
        u2_p   = spinup['u2'][i_start:i_end]

        phi1_plateau = float(jnp.mean(phi1_p))
        u1_mean      = float(jnp.mean(u1_p))
        u2_mean      = float(jnp.mean(u2_p))
        Um_sim       = float(jnp.mean(phi1_p * u1_p + phi2_p * u2_p))
        slip_target  = float(jnp.mean(u2_p - u1_p))  # calculating slip velocity

        print(f"    phi1={phi1_plateau:.4f}  "
            f"u1={u1_mean:.4f} m/s  "
            f"u2={u2_mean:.4f} m/s  "
            f"Um={Um_sim:.4f} m/s")

        dataset.append({
            'WC':           WC,
            'dpdz':         dpdz,
            'Um_target':    Um_t,
            'p_inlet':      p_in,
            'p_outlet':     p_out,
            'phi1_inlet':   WC,
            'phi1_target':  phi1_plateau,
            'Um_sim':       Um_sim,
            'u1_mean':      u1_mean,
            'u2_mean':      u2_mean,
            'slip_target':  slip_target,  
            'spinup_state': spinup,
        })

    print(f"\nDataset complete: {len(dataset)} conditions generated.")
    return dataset


# ── Training conditions ────────────────────────────────────────────────────────
# 8 conditions: 4 WC values × 2 mixture velocities
# Pressure gradients from Ibarra Figure 10

conditions = [
    {'WC': 0.2, 'dpdz': 100.0, 'Um_target': 0.50},
    {'WC': 0.3, 'dpdz': 105.0, 'Um_target': 0.50},
    {'WC': 0.7, 'dpdz': 106.0, 'Um_target': 0.50},
    {'WC': 0.8, 'dpdz': 105.0, 'Um_target': 0.50},
    {'WC': 0.2, 'dpdz': 220.0, 'Um_target': 0.75},
    {'WC': 0.3, 'dpdz': 228.0, 'Um_target': 0.75},
    {'WC': 0.7, 'dpdz': 228.0, 'Um_target': 0.75},
    {'WC': 0.8, 'dpdz': 225.0, 'Um_target': 0.75},
]

# Generate dataset
print("=" * 60)
print("Generating synthetic training dataset")
print(f"  Known drag_coeff = {drag_coeff:.2e}  (network must recover this)")
print("=" * 60)

dataset = generate_synthetic_dataset(
    drag_coeff_true = drag_coeff,
    conditions      = conditions,
    n_spinup        = n_spinup,
    dt_fixed        = dt_fixed,
    dx              = dx,
    D               = D,
    d_b             = d_b,
    mu1             = mu1,
    mu2             = mu2,
    theta           = theta,
    N               = N,
    rho1_val        = rho1_val,
    rho2_val        = rho2_val,
)

# Save dataset metadata (without spinup states — too large)
import pickle
with open('synthetic_dataset_phase1.pkl', 'wb') as f:
    pickle.dump([
        {k: v for k, v in cond.items() if k != 'spinup_state'}
        for cond in dataset
    ], f)
print("Dataset metadata saved to synthetic_dataset_phase1.pkl")

###### adding slip to synthetic dataset after it's been created ######
# Add slip_target to every condition in your existing dataset
# u2_mean is oil velocity, u1_mean is water velocity
# positive value means oil faster than water — correct for your case
for cond in dataset:
    cond['slip_target'] = cond['u2_mean'] - cond['u1_mean']

###### ML Training Loop (Phase 1) — learn drag coefficient from slip velocity

# ══════════════════════════════════════════════════════════════════════════════
# Section 7b — Training loop
# ══════════════════════════════════════════════════════════════════════════════

# ── Hyperparameters ────────────────────────────────────────────────────────────
n_window    = 1000    # differentiable window steps (0.1s at dt=1e-4)
n_epochs    = 300
lr          = 1e-3
print_every = 25

# ── Initialize network and optimizer ──────────────────────────────────────────
key       = jax.random.PRNGKey(42)
network   = DragClosureNetwork(key)
optimizer = optax.adam(learning_rate=lr)
opt_state = optimizer.init(eqx.filter(network, eqx.is_array))

print(f"\n{'='*60}")
print(f"Phase 1 Training — recover drag_coeff = {drag_coeff:.2e}")
print(f"  Conditions:  {len(dataset)}")
print(f"  Epochs:      {n_epochs}")
print(f"  Window:      {n_window} steps ({n_window*dt_fixed:.2f}s)")
print(f"  LR:          {lr}")
print(f"{'='*60}\n")

# ── Training ───────────────────────────────────────────────────────────────────
for epoch in range(n_epochs):

    # Reset accumulators at the start of each epoch
    epoch_loss     = 0.0
    epoch_C_D      = 0.0
    epoch_slip_err = 0.0
    n              = len(dataset)

    # ── Inner loop: one pass through all training conditions ───────────────────
    for cond in dataset:

        # Compute loss and gradients
        (loss_val, aux), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True
        )(
            network, cond, n_window, dt_fixed,
            dx, D, d_b, mu1, mu2, theta, N
        )

        # Unpack 4 auxiliary values from loss_fn
        phi1_pred, Um_pred, slip_pred, C_D_pred = aux

        # Update network weightss
        updates, opt_state = optimizer.update(
            eqx.filter(grads,   eqx.is_array),
            opt_state,
            eqx.filter(network, eqx.is_array),
        )
        network = eqx.apply_updates(network, updates)

        # Accumulate metrics across all conditions this epoch
        epoch_loss     += float(loss_val)
        epoch_C_D      += float(C_D_pred)
        epoch_slip_err += abs(float(slip_pred) - cond['slip_target']) \
                          / (abs(cond['slip_target']) + 1e-6) * 100

    # ── After inner loop: full epoch complete ──────────────────────────────────
    # Everything below runs once per epoch, not once per condition

    # Compute epoch averages
    avg_loss     = epoch_loss     / n
    avg_C_D      = epoch_C_D      / n
    avg_slip_err = epoch_slip_err / n

    # Print progress at regular intervals and on final epoch
    if epoch % print_every == 0 or epoch == n_epochs - 1:
        print(f"Epoch {epoch:4d}  "
              f"loss={avg_loss:.6f}  "
              f"C_D={avg_C_D:.4e}  "
              f"true={drag_coeff:.4e}  "
              f"ratio={avg_C_D/drag_coeff:.3f}  "
              f"slip_err={avg_slip_err:.2f}%")

    # Save checkpoint every 50 epochs
    if epoch % 50 == 0:
        eqx.tree_serialise_leaves(
            f"checkpoint_epoch{epoch}.eqx", network
        )
        with open(f"checkpoint_opt_epoch{epoch}.pkl", "wb") as f:
            pickle.dump(opt_state, f)

    # Check stop flag — only at epoch boundary so state is consistent
    if controller.stop:
        print(f"\nTraining stopped at epoch {epoch}.")
        print(f"Saving final state...")
        eqx.tree_serialise_leaves("drag_network_phase1.eqx", network)
        with open("optimizer_state_phase1.pkl", "wb") as f:
            pickle.dump(opt_state, f)
        print(f"Saved. Resume from epoch {epoch}.")
        break   # exits the outer epoch loop only

# ── Final evaluation ───────────────────────────────────────────────────────────
# This block is OUTSIDE the training loop entirely
# It runs exactly once after training completes (or is stopped)
print(f"\n{'='*60}")
print(f"Final evaluation")
print(f"Target C_D = {drag_coeff:.6e}")
print(f"{'='*60}")

# Header printed BEFORE data rows
print(f"{'WC':>5}  {'Um':>5}  {'C_D_learned':>14}  "
      f"{'ratio':>7}  {'slip_pred':>10}  {'slip_tgt':>10}  {'slip_err%':>9}")
print("-" * 65)

for cond in dataset:
    x = build_network_inputs(
        jnp.array(cond['WC']),
        jnp.array(cond['u1_mean']),
        jnp.array(cond['u2_mean']),
    )
    C_D_final = float(network(x))
    ratio     = C_D_final / drag_coeff

    (_, (phi1_pred, Um_pred, slip_pred, _)), _ = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(network, cond, n_window, dt_fixed, dx, D, d_b, mu1, mu2, theta, N)

    slip_err = abs(float(slip_pred) - cond['slip_target']) \
               / (abs(cond['slip_target']) + 1e-6) * 100

    print(f"{cond['WC']:>5.1f}  "
          f"{cond['Um_target']:>5.2f}  "
          f"{C_D_final:>14.6e}  "
          f"{ratio:>7.3f}  "
          f"{float(slip_pred):>10.6f}  "
          f"{cond['slip_target']:>10.6f}  "
          f"{slip_err:>7.2f}%")

# Save final network
eqx.tree_serialise_leaves("drag_network_phase1.eqx", network)
print(f"\nTrained network saved to drag_network_phase1.eqx")