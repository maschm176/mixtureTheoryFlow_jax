# CLAUDE.md — Project Context: Two-Phase Pipe Flow Simulation with Neural Network Drag Closure

## Working Preferences

- This project is written in Python (JAX, Equinox, Optax, NumPy).
- Before modifying any existing code, explain what you plan to change
  and why. Wait for confirmation before implementing anything.
- Walk through the conceptual logic of any new code before showing
  the implementation — I want to understand what the code is doing
  and why, not just have something that works.
- Provide explanations alongside code, not just after. Comment
  non-obvious lines so the reasoning is clear inline.
- If a change could break existing functionality, flag it explicitly
  before proceeding.
- I am a researcher, not primarily a software engineer — prioritize
  physical intuition and clarity over code brevity.
- After every time you make changes to mixtureTheoryFlow_jax.ipynb, commit
  those changes to git using the /push-to-git command, with an insightful
  commit message describing what changed and why.

---

## Current Status

Phase 1 (scalar C_D optimization) is validated. The wall friction closure has
been swapped from per-phase Blasius (D_h_i = D*phi_i) to Taitel-Dukler
stratified segment geometry, after the Blasius version was found to badly
over-penalize whichever phase is the minority. The vanishing-phase NaN (from
the original Darcy-Weisbach → Blasius swap) was fixed via an explicit
single-phase collapse, and the traveling-wave instability that swap had
revealed near/past inversion (WC≈0.7) is no longer observed since switching
to Taitel-Dukler — confirmed via grid convergence (N=250/500/1000) on a
previously-affected condition. An interfacial term (S_i) with tunable weight
w was subsequently added to the Taitel-Dukler closure to fix a separate
sign-crossing-location problem, and w has been fit against real data
(w=0.4920, holding out two near-inversion conditions). Full details under
"Completed Phases" below.

**Next steps (Phase 2a real-data training, blocked on this work):**

1. Sweep all real training conditions (not just WC=0.3156 and WC=0.7) for
   oscillation — run the chunked diagnostic across every WC/Um pair actually
   in ibarra_phase2a_dataset.csv / real_dataset, especially the higher-WC
   ones (WC=0.6-0.8) most likely to be in the disadvantaged-oil regime.
2. For each condition, quantify quiet-time, not just presence/absence of a
   quiet moment — report what fraction of the run is quiet vs. actively
   oscillating, and how long typical quiet stretches last. This distinguishes
   a condition that's mostly stable with rare events from one that's mostly
   unstable with a narrow calm sliver.
3. Define one consistent, quantitative "quiet enough" criterion (e.g. phi2
   hasn't moved more than some threshold over the last several chunks, or
   peak velocity below a bound) and apply it uniformly across all
   conditions — not eyeballed per condition, which risks inconsistent
   selection quality even where absolute timing legitimately differs.
4. Flag any conditions that are mostly unstable (little genuine quiet time
   available) as a judgment call rather than silently picking a
   technically-quiet snapshot — decide case by case whether to exclude them,
   handle them specially, or accept the representativeness risk.
   **Decision made**: conditions with no adequate quiet streak (per the step 3
   criterion) are excluded from the Phase 2a training set entirely, rather
   than pulling out whatever moment happens to be locally quietest within an
   otherwise-unstable run — a technically-quiet snapshot cherry-picked from a
   mostly-oscillating condition isn't representative of how that condition
   actually behaves. Flagged for possible later attention: revisit this by
   instead including the quietest available region (even if short/marginal)
   for currently-excluded conditions, to see whether/how it changes training
   results — not done now since it reintroduces the representativeness risk
   this exclusion was meant to avoid.
   **Extended**: conditions whose assigned window is phase-collapsed (a
   minority phase's local volume fraction dips below PHI_COLLAPSE, forcing
   its velocity to match the dominant phase regardless of C_D) are excluded
   on the same grounds — found via WC=0.1@Um=0.75, which had excellent Um
   agreement but permanently zero slip from ~t=6s onward. See "Current
   Limitation: Vanishing-Phase Singularity" under Known Issues for why this
   can't be fixed by a richer M1 closure either.
5. Update generate_real_dataset (Section 7a) to select each condition's
   spinup/training-window starting point per-condition based on the
   criterion above, instead of a fixed spinup duration for every condition.
6. Deferred, not immediate: if future work needs trustworthy transient
   dynamics right at/past inversion specifically (not just steady plateau
   values), revisit the numerical scheme — a shock-capturing/flux-limited
   scheme would resolve the traveling wave more cleanly than the current
   Lax-Friedrichs setup. Not needed for the current goal.
   **Update**: the traveling wave that motivated this item is no longer
   observed after switching wall friction to Taitel-Dukler (see "Wall
   friction upgrade" under Completed Phases) — confirmed at N=250/500/1000
   on a previously-affected condition. Downgrading from "known open
   numerical issue" to "watch for recurrence once the full condition sweep
   is re-run"; not deleting this item since it's only been checked on one
   condition so far, not the full real-data set.
7. Phase 2a single-scalar C_D training (Section 7c) plateaued at a clear,
   informative pattern rather than a noisy one: pre-inversion (positive
   target slip) conditions fit well, post-inversion (negative target slip)
   conditions all land at the *correct sign* but ~4-5x too small a
   magnitude. That's consistent with the two regimes needing genuinely
   different effective drag — something one shared constant C_D structurally
   cannot represent, no matter how well-optimized. Added a lightweight
   two-parameter test (Section 7d) as a cheap intermediate check before
   committing to a full Phase 2b: two independent scalars, `C_D_pos` and
   `C_D_neg`, split by the *sign of each condition's own experimentally-
   measured target slip* (not learned — the data already tells us which
   physical regime a condition is in). Kept fully separate from the
   1-parameter baseline (own variables, own checkpoint files) so both
   results stay directly comparable. If both regimes fit well with their
   own dedicated scalar, that's direct evidence the fix needs a regime-
   dependent closure, strengthening the case for Phase 2b; if one regime
   still struggles even with its own parameter, the remaining gap isn't
   simply "pre- vs. post-inversion." **Not yet run/evaluated as of this
   note** — see Section 7d output for results once available.
   **Superseded**: this sign-mismatch question ended up being answered by a
   separate, deeper investigation instead of by running this test — see
   "Interfacial term (S_i) added to Taitel-Dukler wall friction" under
   Completed Phases. That work established that M1/drag is *structurally*
   incapable of moving *where* slip crosses zero — it's built to vanish at
   slip=0 (M1 = C_D·|Δu|·Δu), so it can only shrink slip magnitude, never
   relocate the zero-crossing. No number of regime-specific C_D scalars
   (one, two, or a future NN) could ever fix a mismatch caused by the
   zero-crossing itself sitting at the wrong WC — only non-drag physics
   (wall friction, pressure) sets that location. The real root cause was the
   wall-friction closure's geometry, not drag being insufficiently
   regime-aware. Section 7d's `C_D_pos`/`C_D_neg` split was never run — it's
   superseded rather than completed, since the premise motivating it turned
   out to be incomplete.
8. Current active step: refit the single scalar C_D against real Ibarra
   slip targets, holding w fixed at its now-converged value (0.4920 — see
   Completed Phases), reusing the original Phase 1/2a jax.grad-based scalar
   optimization pattern (short spinup + short differentiable window near the
   plateau) — not the finite-difference approach used for w, which was
   specifically needed because w's effect on slip only becomes visible after
   the full ~37s transient settles, unlike C_D's, which is visible locally
   near the plateau within a short window.
   **Redirected**: attempting this refit surfaced a deeper problem before it
   could even be evaluated meaningfully — see "C_D refit surfaces a
   fundamental ceiling..." under Completed Phases. C_D ran away (1.0 → 18.6
   over 30 epochs) toward a degenerate "predict zero slip everywhere"
   solution instead of converging to a real fit, traced to most real
   conditions being structurally unreachable by C_D alone regardless of its
   value — the achievable |slip| ceiling (at C_D→0) is set entirely by
   non-drag physics (wall friction, pressure), which C_D can only ever
   shrink below, never exceed. This refit is paused, not abandoned — worth
   revisiting once wall friction magnitude (item 9) is better calibrated and
   the ceiling itself has moved.
9. New active step: wall friction MAGNITUDE (distinct from the stratified
   distribution S_i/w already addresses) is now understood to be the actual
   binding constraint on both Um accuracy and achievable slip range — see
   "C_D refit surfaces a fundamental ceiling..." under Completed Phases,
   including a literature check confirming this is a known, actively-studied
   limitation of stratified two-fluid models generally (Rodriguez & Oliemans
   2006, Chakrabarti et al. 2005, Ng et al. 2004), not an artifact specific
   to this project. Planned: fit phase-specific friction multipliers k1
   (water) and k2 (oil) — applied unconditionally in every condition, not
   regime-switched, mirroring the phase split D_h1/D_h2 already has — via
   the same jax.grad scalar-optimization pattern already validated for w and
   C_D, as a cheap test before considering a richer NN-learned friction
   correction. Not yet implemented; user is reviewing the referenced
   literature first before deciding on the exact functional form.
   **Result**: fit succeeded — k1=0.6523, k2=0.9458 (see "Phase-specific
   wall friction magnitude multipliers..." under Completed Phases). Average
   Um error across the 9 trained conditions dropped to ~3.2% (from ~15.0% at
   w alone), every one under 10%, all signs unchanged from the w fit.
   Confirms wall friction magnitude, not drag closure choice, was the
   dominant remaining error source.
10. New active step: (1) bake k1=0.6523/k2=0.9458 in as new defaults in
    advance_momentum/time_step_learned (same treatment already given to
    w=0.4920), (2) re-run the zero-drag-ceiling sweep (see "C_D refit
    surfaces a fundamental ceiling..." under Completed Phases) under this
    improved friction closure to check whether the previously-unreachable
    conditions are now reachable, before (3) deciding whether to resume the
    paused C_D refit (item 8) now that its binding ceiling has likely
    moved. Not yet done.

---

## Completed Phases

**Simulation development**
- Built 1D two-fluid model with Lax-Friedrichs finite volume fluxes,
  Thomas algorithm pressure Poisson solver, Darcy-Weisbach wall friction,
  viscous stress, gravity terms, and interphase drag.
- Implemented advance_mass (continuity) and advance_momentum (momentum)
  for both phases, with pressure projection to enforce volume conservation.
- Applied boundary conditions: fixed phi1_inlet_bc at inlet, zero-gradient
  at outlet for both phi1 and velocity.

**Validation against Ibarra et al. (2015)**
- Validated momentum physics against Figure 10 (pressure gradient data):
  7.2% mean absolute error on mixture velocity across 31 data points,
  +1.5% mean bias (near zero — no systematic over or underprediction).
- Generated validation plots: WC vs Um subplots (one per mixture velocity)
  and parity plot with ±10% bands, colored by Um and shaped by flow regime.
- Computed both volumetric Um = phi1*u1 + phi2*u2 and momentum-weighted
  Um = (phi1*rho1*u1 + phi2*rho2*u2) / (phi1*rho1 + phi2*rho2) for comparison.

**Drift flux addition to advance_mass**
- Added drift flux correction j_drift = phi1*phi2*(u1-u2) to drive phi1
  redistribution when oil moves faster than water.
- Sign convention: oil faster (u2 > u1) → j_drift < 0 → -div(j_drift) > 0
  → phi1 increases above inlet water cut (physically correct accumulation).
- Found phi1 is nearly insensitive to drag_coeff: 0.001% change across
  10,000x drag variation. Wall friction dominates by factor of ~54,000x.

**Time loop conversion to jax.lax.scan**
- Converted Python while loop to jax.lax.scan with fixed dt = 1e-4 s.
- Two-phase structure: spinup (stop_gradient, ~18s for 150s simulation)
  and differentiable window (short scan for gradient tracking).
- Confirmed throughput: ~80,000 steps/second at N=500 cells.
- Verified scan produces identical results to original while loop.

**Gradient chain verification**
- Confirmed JAX can differentiate through simulation back to drag_coeff:
  d(slip)/d(drag_coeff) = -0.070 (nonzero, negative — physically correct).
- phi1 and Um gradients are essentially zero — not usable as training signals
  with current dispersed bubble drag formulation.
- Slip velocity is the only simulation output with meaningful gradient signal.

**Phase 1 neural network attempt (abandoned)**
- Implemented DragClosureNetwork in Equinox: 4 scaled inputs → 32 → 32 → 16
  → 1 scalar output with softplus activation enforcing C_D_eff > 0.
- Trained with phi1 and Um in loss function — failed, near-zero gradients.
- Switched to slip velocity in loss function — confirmed nonzero gradient signal.
- Training was converging (ratio ~17x → ~16x over hundreds of epochs) but
  extremely slowly — estimated 6,700+ epochs to reach target.
- Decision: C_D is a single scalar — a neural network is unnecessary complexity.
  Switching to direct scalar gradient-based optimization for Phase 1 validation.

**Phase 1 direct scalar C_D optimization (validated)**
- Replaced the NN training loop (Section 7b) with direct gradient-based
  optimization of a single scalar C_D in log-space (C_D = exp(z)), using Adam
  with the same per-condition update loop as the NN version — deliberately
  kept identical so the comparison isolates whether the network architecture
  itself was the bottleneck.
- Initial learning rate (1e-3, carried over from the NN version) converged
  far too slowly: Adam's step size in log-space is ~lr per update, so closing
  the ~4-decade gap between the initial guess (C_D=1.0) and the true value
  (1e-4) would have taken ~1,150 epochs. Raised lr to ~0.1 to close that gap
  in tens of epochs instead.
- Final result: C_D converged to 1.625e-3 (ratio = 16.25x from true 1e-4),
  with slip velocity predictions matching targets to within 0.5-2.9% across
  all 8 conditions.
- Key finding: this ~16x ratio is nearly identical to where the abandoned NN
  attempt also plateaued (~16-17x), despite a completely different
  optimization mechanism (1,745 weights vs. one scalar). This strongly
  suggests the plateau reflects a real physics limitation, not a
  training-method artifact — it confirms the earlier finding that slip
  velocity is only weakly sensitive to drag_coeff (3.2% change across 5
  orders of magnitude) under the current dispersed-bubble drag formulation,
  making precise C_D recovery from bulk slip alone difficult regardless of
  optimizer.
- Decision: treat Phase 1 as validated — the gradient-based optimization
  chain works end-to-end and converges efficiently — but the precision
  ceiling reinforces the need to move to the stratified interfacial drag
  formulation (already planned) before Phase 2a.

**Phase inversion physics investigation (wall friction fix + traveling wave
characterization)**
- Trigger: training the NN on real Ibarra data (Phase 2a) exposed a sign
  mismatch the model could never fit — 6 of 13 real conditions back-calculate
  to negative slip (water faster than oil), but the simulation always
  predicted oil faster, at every condition, regardless of C_D. Root cause:
  the drag closure (M1 = C_D * |delta_u| * delta_u) is a damping force — it
  can only shrink slip magnitude, never reverse its sign. The sign is set
  entirely by non-drag physics (wall friction, pressure), so no amount of
  C_D/M1 flexibility could ever fix this.
- Traced the fixed bias to the old Darcy-Weisbach wall friction: mixture-
  averaged and split by volume fraction, its per-unit-mass deceleration
  reduces to F_friction/rho_i — depends only on density, never on water cut.
  Oil (less dense) was *always* relatively disadvantaged, at every WC, with
  no mechanism to flip near the physical inversion point (WC≈0.5).
- Fix: swapped to the per-phase Blasius wall friction (previously present
  but disabled in advance_momentum). Its hydraulic diameter D_h_i = D*phi_i
  makes per-unit-mass deceleration scale as ~1/phi_i — whichever phase is
  the *minority* gets disproportionately penalized. This bias flips exactly
  where minority/majority status swaps (WC≈0.5), giving a real physical
  mechanism for phase inversion instead of a fixed bias.
- The swap introduced a new failure mode: NaN, traced to the classic
  "vanishing phase" pathology in two-fluid models — recovering velocity
  requires dividing by volume fraction (u_i = mom_i/(phi_i*rho_i)), and as
  phi_i shrinks toward the numerical floor this division amplifies noise
  into exploding, unstable velocities. Confirmed via chunked/instrumented
  diagnostics: phi2 got squeezed toward the floor by drift flux, then
  diverged once pinned there.
- Fix: implemented an explicit single-phase collapse (PHI_COLLAPSE = 1e-3
  threshold) in advance_momentum, time_step, and time_step_learned. Below
  threshold, a phase's velocity is set to the *other* phase's velocity
  (carried along with the dominant phase) instead of the old zero-velocity
  behavior, and it contributes no independent wall friction. Implemented
  with a "safe substitution" pattern (jnp.where substitutes a safe dummy
  value into the division's *input* before the risky branch is computed) so
  it stays NaN-safe through jax.grad, not just the forward pass — a naive
  jnp.where can still leak NaN through its gradient from a branch whose
  *output* is discarded but whose *computation* was still dangerous.
- Validated at WC=0.3156: phi2 now hits the same 1e-4 floor that previously
  caused NaN at t=30s, but survives the full 150s run without diverging.
- At WC=0.7 (past the inversion point), no NaN, but a new phenomenon
  appeared: a recurring traveling wave — phi2 spikes to ~0.9 then crashes to
  near-zero, the whole front migrating downstream over time, recurring every
  ~20-30s. Confirmed via spatial (not just domain-min/max) diagnostics.
- Characterized the wave via two further checks:
  (1) Grid convergence (N=250/500/1000): the wave persists at a consistent
      location/timing across a 4x resolution range — real physics, not a
      grid artifact — but gets *noisier* (not cleanly convergent) at higher
      N, consistent with an under-resolved shock interacting with Lax-
      Friedrichs' limited dissipation, not a fully grid-converged solution.
  (2) Propagation speed: measured wave speed (~0.31 m/s, clean linear fit)
      is close to bulk advection speed (u1≈0.247, u_mix≈0.240 m/s, within
      25-30%) and nowhere near the drift-only kinematic wave speed
      ((1-2*phi1)*(u1-u2) ≈ -0.007 m/s — wrong sign, ~43x too small).
      Conclusion: the wave is carried downstream by bulk advection: the
      phi-dependent friction nonlinearity drives the *steepening* into a
      shock-like spike, not the propagation speed itself.
- Decision: treat this as a known, understood limitation rather than fully
  resolving the numerics now — the phase-inversion mechanism itself is
  physically motivated and validated; the transient wave behavior near/past
  the inversion point is real but not yet cleanly grid-resolved by the
  current Lax-Friedrichs scheme. Practical mitigation: avoid extracting
  training plateau values from windows where this oscillation is actively
  occurring, rather than upgrading to a shock-capturing scheme right now.
- **Superseded**: this traveling-wave/collapse behavior is no longer
  observed after switching wall friction to the Taitel-Dukler stratified
  geometry — see "Wall friction upgrade: Taitel-Dukler stratified geometry"
  below.

**Wall friction upgrade: Taitel-Dukler stratified geometry (replacing
per-phase Blasius's naive D*phi_i scaling)**
- Trigger: a single scalar C_D (and even a two-parameter C_D_pos/C_D_neg
  split by pre-/post-inversion sign) plateaued on real Ibarra conditions
  with a clean, informative pattern rather than noise — low-WC conditions
  fit to 2-7% error, high-WC conditions (WC=0.7@Um=0.50, WC=0.7@Um=0.75,
  WC=0.8@Um=0.75) all landed at the *correct sign* but 60-82% too small in
  magnitude. A zero-drag-ceiling diagnostic (forcing C_D→0 on the struggling
  conditions) showed even zero drag only reached 22-40% of the target |slip|
  — proof no drag closure of the "positive coefficient × |Δu|·Δu" form
  (scalar, per-regime, or a future NN-learned M1) could ever close that
  gap, since such a closure can only shrink slip, never create it. The
  bottleneck had to be non-drag physics.
- Cross-checked the struggling conditions against Ibarra's own flow pattern
  map (Figure 6, from the source PDF): none are dual-continuous (ruling out
  a fundamental 2D/3D representation limit this 1D model genuinely
  couldn't capture); all are SWD (stratified wavy with droplets) — the
  *same* nominal regime as a well-fitting low-WC condition (WC=0.3@Um=0.75).
  So the failure tracked water cut continuously within one regime, not a
  qualitative regime boundary — pointing specifically at the wall friction
  closure's geometry assumption, not at flow-pattern-dependent physics
  the model was missing entirely.
- Root cause: the per-phase Blasius friction (previous entry) used
  D_h_i = D*phi_i — implicitly modeling each phase as flowing through its
  own smaller circular sub-pipe scaled by volume fraction. This geometry
  has no relationship to the real stratified picture (two circular
  segments split by a flat interface, water pooled at the bottom, oil
  layered on top) and badly underestimates the true hydraulic diameter of
  whichever phase is the minority — verified numerically to be off by up
  to ~3.9x at a 90/10 split — because a real thin segment's wall-contact
  arc length shrinks much more slowly than its area does, something the
  naive linear phi_i scaling has no way to represent.
- Fix: derived the actual Taitel-Dukler circular-segment geometry. The
  water segment's half-angle phi_angle relates to its area fraction via
  the transcendental equation
  `phi1 = (phi_angle - sin(phi_angle)*cos(phi_angle)) / pi`,
  solved with 6 unrolled Newton-Raphson iterations (verified to converge
  to machine precision, residuals ~1e-16, across the whole
  PHI_COLLAPSE-bounded range) to give closed-form hydraulic diameters
  `D_h1 = pi*D*phi1/phi_angle`, `D_h2 = pi*D*phi2/(pi-phi_angle)`. Confirmed
  both algebraically and numerically that the result is independent of
  which phase is arbitrarily assigned the "bottom" role in the derivation
  (a circle's mirror symmetry guarantees this) — so the fix only assumes
  the flow is genuinely stratified into two segments (true for the SWD
  conditions in question, per the Figure 6 check above), not any
  particular phase ordering.
- Everything downstream of D_h1/D_h2 (Re1/Re2, the Blasius/laminar f_D
  switch, both friction formulas, the PHI_COLLAPSE collapse override) was
  left unchanged — the existing friction formula's functional form was
  already dimensionally correct in general, it just needed the right
  D_h_i plugged in.
- Verified in stages before touching the live simulation: (1) isolated
  geometry solve — symmetric-point check (D_h1=D_h2=D_pipe exactly at
  phi1=0.5) and Newton convergence both passed; (2) isolated gradient
  check — jax.grad through the geometry solve stays finite everywhere,
  correct sign (dD_h1/dphi1>0, dD_h2/dphi1<0) in the interior; (3) wired
  into advance_momentum and re-ran both the forward pass and the gradient
  chain (d(loss)/d(drag_coeff), d(loss)/d(phi1)) directly — all finite,
  no NaN.
- Result: after the switch, the traveling-wave oscillation-to-crash
  behavior from the previous entry (recurring phi2 spike-and-crash near
  WC=0.7) is no longer observed. Confirmed via the grid convergence check
  (N=250/500/1000) re-run on a previously-affected (collapsing) condition
  — all three resolutions are free of oscillation and structurally
  consistent with each other.
- Why this isn't a coincidental side effect: near a vanishing phase, the
  naive formula's D_h_i ~ phi_i (linear) makes friction (~1/D_h_i) blow up
  as a first-order pole in 1/phi_i. The Taitel-Dukler geometry's
  D_h_i ~ phi_i^(2/3) near the same limit (from the small-angle expansion
  of the segment-area equation) is a much gentler falloff, so friction
  only blows up as phi_i^(-2/3) — a weaker singularity. That directly
  reduces both the runaway minority-phase deceleration that was driving
  phase collapse, and the friction-nonlinearity-driven steepening
  implicated in the traveling wave's shock-like spikes — the same
  underlying fix plausibly explains both improvements at once.
- Not yet done: confirmed on one previously-problematic condition across
  three resolutions, not yet the full sweep across all conditions in
  ibarra_phase2a_dataset.csv. Phase 2a next-steps item 1 (sweep every real
  condition for oscillation) should be re-run under this new friction
  model before assuming the existing collapse/oscillation exclusion gates
  are no longer needed for any condition — expected to trigger far less
  often (if at all), but to be confirmed rather than assumed. Also expect
  the resulting quiet-window timing to now correlate with Um (bulk transit
  time L/Um) rather than being erratic per-condition, which would
  meaningfully simplify generate_real_dataset's per-condition window
  selection (Current Status item 5) if confirmed.

**Interfacial term (S_i) added to Taitel-Dukler wall friction, weight w fit
to real data**
- Trigger: even after the Taitel-Dukler switch, several real conditions
  still showed a slip *sign* mismatch. Traced this to the simulated
  inversion point (the WC where slip crosses zero) sitting at a different
  location than the experimental one — e.g. simulated ≈0.72-0.75 vs.
  experimental ≈0.47 at Um=0.75 — not to residual drag insufficiency.
- Established a structural fact that reframed the whole problem: M1/drag is
  built to vanish at zero slip (M1 = C_D·|Δu|·Δu), so it can only shrink
  slip magnitude, never move *where* it crosses zero. Only non-drag physics
  (wall friction, pressure) sets that location. This means Current Status
  item 7's planned `C_D_pos`/`C_D_neg` two-parameter test could never have
  fixed a zero-crossing-location mismatch regardless of its outcome, since
  it only varies drag by regime, not the crossing location itself — see the
  "Superseded" note on that item.
- Fix: added the interfacial chord term S_i = D·sin(phi_angle) — the chord
  length of the flat interface between the two circular segments — to oil's
  wetted perimeter in the Taitel-Dukler hydraulic diameter, weighted by a
  tunable scalar w ∈ [0,1]:
  `D_h2 = pi*D*phi2/(pi - phi_angle + w*sin(phi_angle) + eps)`.
  w=0 recovers the original unweighted Taitel-Dukler geometry; w=1 is the
  literature convention of treating the full interfacial chord as if it
  were solid wall contact for the lighter/faster phase.
- Manually hand-tested w=1.0, 0.7, 0.4 against real data: higher w fixed
  more sign mismatches (w=1.0 fixed all 11) but increasingly inflated a
  pre-existing Um deficit — a monotonic tradeoff, not independent knobs.
- Built a rigorous gradient-based fitting procedure for w, reusing Phase 1's
  scalar-optimization scaffolding: Adam over an unconstrained variable z,
  with w = sigmoid(z) keeping it bounded in (0,1) by construction.
- Kernel crash, then fix: unlike C_D (whose effect on slip is visible in a
  short differentiable window near the plateau), w's effect only becomes
  visible after the full ~37s transient has settled, so directly
  differentiating (jax.grad) through the required ~370,000-step scan needed
  memory proportional to the full step count — confirmed via direct memory
  profiling (50,000 differentiated steps alone used ~8GB; nested
  jax.checkpoint didn't meaningfully reduce this). Fixed by switching to
  central finite-difference gradients for this one scalar: three
  forward-only simulations per gradient step (center, w+h, w-h), which need
  no memory proportional to step count (confirmed: the full 370,000-step
  forward pass uses ~300MB). This is scoped specifically to w — C_D
  refitting still uses jax.grad through the original short window and was
  never affected by this issue.
- First fit, all 11 real conditions: converged to w=0.9811 — essentially
  independently rediscovering the literature convention (w≈1) via gradient
  descent rather than by hand. Fixed all 11 sign mismatches, but pushed
  average |Um error| to ~20.2% (up from ~8.5% at w=0), and *every single*
  condition's Um error measurably worsened as w increased from 0 to 0.98 —
  confirming the additive S_i term inflates total friction broadly, not
  just the relative balance between phases.
- Diagnostic sweep (w = 0, 0.3, 0.5, 0.7, 0.9811) showed 9 of 11 conditions
  already have the correct sign by w=0.5, with no further improvement
  between w=0.5 and w=0.7 — only two conditions (WC=0.40 and WC=0.50 @
  Um=0.50) remain mismatched that whole range, only flipping once w is
  pushed to ~0.98. These two also have by far the smallest-magnitude slip
  targets in the dataset (-0.0608, -0.0148 m/s), and WC=0.50 is exactly the
  phase-inversion point this document already flags elsewhere as a regime
  "no simple two-fluid model handles well."
- Decision: excluded these two near-inversion conditions from the w fit
  entirely (`NEAR_INVERSION_EXCLUDE`) — same exclusion logic already used
  for phase-collapsed conditions. Chasing an exact sign on a near-zero
  target in a regime the model is known not to resolve isn't worth the Um
  cost it imposes on every other condition.
- Refit against the remaining 9 conditions: converged to w=0.4920. All 9
  trained conditions now have the correct sign; average |Um error| = 15.0%
  (down from 20.2% at w=0.9811), range -3.7% to -22.1%.
- Not resolved by this work, flagged as open: a meaningful share of the
  residual ~15% Um error predates S_i entirely — several of the same
  conditions already showed comparable errors (up to -15%) at w=0, pointing
  to a Taitel-Dukler base friction magnitude issue independent of the
  interfacial term. C_D refitting won't address this either, since wall
  friction dominates Um by ~54,000x (see "Current Limitation: Drag
  Insensitivity" under Known Issues) — a separate, later item, not blocking
  the current C_D refit.

**C_D refit surfaces a fundamental ceiling: wall friction magnitude, not
drag closure choice, is the binding constraint**
- Trigger: attempted the Phase 2a C_D refit (Section 7c) against real data,
  holding w fixed at 0.4920. Instead of converging, C_D ran away (1.0 →
  18.6 over 30 epochs) while every condition's slip_pred shrank toward zero
  and the average loss crept toward ~1.0 — the "predict slip=0 everywhere"
  asymptote — rather than actually improving. Final state: mean slip error
  100.32% ± 3.22% across all 11 conditions — essentially uniform failure
  disguised as "loss decreasing."
- Diagnosed the mechanism: the loss is normalized by |slip_target|, so a
  badly sign-mismatched condition's error falls toward the 100% floor as
  its prediction shrinks toward zero (looks like improvement), while every
  correctly-signed condition's error *rises* toward that same floor from
  below as its already-too-small prediction shrinks further away from
  target. One early sign-mismatched outlier dominated the average loss at
  first, masking that the whole population was collectively converging to
  the same degenerate zero-slip solution, not to a real fit.
- Root structural cause: M1 is a pure damping force
  (M1 = C_D·|Δu|·Δu, coefficient ≥ 0 by construction) — it can only ever
  shrink |slip| below whatever the non-drag physics (wall friction +
  pressure) produces at C_D=0; it can never grow slip beyond that
  "zero-drag ceiling." C_D started at 1.0 (10,000x the physically-motivated
  reference value, drag_coeff=1e-4) — already deep into a regime where drag
  rivals or exceeds wall friction in magnitude — so there was no direction
  left to move except toward the degenerate collapse.
- Built a diagnostic sweeping slip_pred vs. C_D (1e-6 to 1e2, log-spaced)
  across all 11 real conditions, reusing each condition's cached
  real_dataset spinup_state — a direct extension of the earlier "neg-group"
  zero-drag-ceiling check (git commit 57a611c) to the full dataset. Result:
  most conditions are structurally unreachable by C_D alone, specifically
  WC=0.6/0.7/0.8 @ Um=0.50 and the post-inversion WC points @ Um=0.75 —
  exactly the same conditions carrying the largest Um error in the w-fit
  evaluation table. Not a coincidence: both symptoms trace to the same root
  cause.
- Key structural realization, broader than just this formula: at C_D→0,
  M1→0 regardless of which formula computed the coefficient — dispersed
  bubble, stratified interfacial, a regime-split scalar, or a full
  NN-learned coefficient (as long as it's architecturally constrained
  non-negative, which the existing DragClosureNetwork's softplus output
  enforces on purpose). All of them collapse to the identical zero-drag
  simulation at that limit. This means the achievable slip ceiling is set
  ENTIRELY by non-drag physics and is invariant to which drag closure is
  chosen — so neither switching to the stratified interfacial drag formula
  originally planned for Phase 2b, nor a full NN-learned M1, would resolve
  the unreachable conditions. Both would hit the identical ceiling for the
  identical reason.
- Decision: redirect focus toward wall friction MAGNITUDE (distinct from
  the geometric/distributional S_i/w work already done) as the actual
  binding constraint, rather than continuing to refine the drag closure.
  Planned next step: fit phase-specific friction multipliers k1 (water) and
  k2 (oil) — `friction1_learned = k1*friction1`, `friction2_learned =
  k2*friction2` — applied unconditionally in every condition (a structural
  per-phase distinction that already exists in the code, same spirit as
  D_h1/D_h2, not a regime-switched branch), fit jointly via the same
  jax.grad scalar-optimization pattern already validated for w and C_D. A
  cheap scalar test before considering a richer NN-learned friction
  correction, same "cheap test before full commitment" philosophy already
  used for w before S_i, and originally intended for C_D_pos/C_D_neg before
  Phase 2b.
- Literature check: searched for prior work on this exact issue before
  committing further effort, given how confidently the physics pointed at
  wall/interfacial friction. Found strong, direct confirmation this is a
  known, actively-studied limitation of stratified two-fluid models
  generally, not an artifact of this project's implementation:
  - Rodriguez & Oliemans (2006): two-fluid model errors for stratified
    oil-water flow average 20-40%, exceeding 100% in some cases — closely
    matches this project's own error magnitudes.
  - Chakrabarti et al. (2005): pressure gradient predictions vary 40-200%
    depending on which closure relations are chosen.
  - Ng et al. (2004): oil-water wall/interfacial equivalent friction
    factors are "quite inconsistent" and "difficult to formulate" — the
    same conclusion reached independently here via the S_i/w investigation.
  - Taitel & Dukler's original 1976 formulation assumed a *constant* ratio
    between interfacial and wall friction factor — a simplification the
    literature has identified ever since as the model's primary weakness,
    causing regime-dependent over/under-prediction of holdup. This is
    effectively the exact gap the S_i/w term patches, independently
    rediscovered here rather than sourced from the literature.
  - Rodriguez & Oliemans improved accuracy to ~35% error specifically by
    incorporating interfacial wave roughness — physically consistent with
    the struggling conditions here being independently classified as SWD
    (stratified wavy *with droplets*) via the Figure 6 cross-check done
    earlier (see "Wall friction upgrade" entry above).
  - A 2023 Petroleum Science paper ("Prediction of pressure gradient and
    hold-up in horizontal liquid-liquid pipe flow") reached much better
    accuracy (mean error -7.06%, std 20.72%) by modifying friction
    correlations for oil viscosity, interfacial curvature, and phase
    fraction jointly, rather than a single flat multiplier — a useful
    reference point for how far a well-designed closure can go, and a
    caution that k1/k2 (two flat scalars) may not be the final answer if it
    doesn't close enough of the gap.
- Not yet done: the k1/k2 friction-magnitude fit itself (planned next step,
  not yet implemented). User is reading the referenced literature first
  (Rodriguez & Oliemans 2006, the 2023 Petroleum Science paper, and
  re-checking Ibarra et al.'s own discussion section) before deciding
  whether a flat multiplier or a richer functional correction is the right
  starting point.

**Phase-specific wall friction magnitude multipliers (k1/k2) fit to real
data — closes most of the residual Um gap**
- Trigger: both the C_D-refit ceiling diagnostic and the literature search
  (previous entry) pointed to wall friction MAGNITUDE — distinct from the
  distributional/geometric fix S_i/w already made — as the actual binding
  constraint on Um accuracy and achievable slip range.
- Fix: added two independent multiplicative scalars, k1 (water) and k2
  (oil), applied unconditionally to each phase's finalized friction term
  (after the PHI_COLLAPSE collapse override, before it enters the momentum
  update): `friction1_learned = k1*friction1`, `friction2_learned =
  k2*friction2`. Defaults of 1.0 leave physics unchanged unless overridden
  — a structural per-phase distinction that already existed in the code
  (mirroring D_h1/D_h2), not a regime-switched branch.
- Fitting procedure: reused w's exact scaffolding (fresh initial_conditions
  each trial, full N_TOTAL_STEPS=370,000-step forward-only pass,
  finite-difference gradients — the same OOM reasoning as w applies, since
  k1/k2 are wall-friction parameters whose effect likewise only fully
  develops after the full ~37s transient settles) generalized to two
  parameters: a 2D central finite difference needs 5 forward evaluations
  per gradient step (center, k1±h, k2±h) instead of w's 3, ~1.7x the
  per-epoch cost (~11 min/epoch at 11 conditions). Reparameterized in
  log-space (k=exp(z)) like C_D, since k must stay positive with no natural
  upper bound (unlike w's bounded-(0,1) sigmoid). Loss identical in
  structure to w's: Um relative error squared + sign-preservation penalty,
  no explicit slip-magnitude term. Reused w_fit_conditions directly (same
  collapsed + near-inversion exclusions); deliberately did NOT exclude the
  conditions found unreachable by the C_D ceiling sweep, since raising
  their ceiling was the whole point of this fit.
- First attempt (lr=0.3, 5 epochs, starting k1=k2=1.0 — "no correction"):
  clearly unstable. k1 and k2 oscillated in opposite directions each epoch
  (k1: 1.25→0.96→0.56→0.88→1.07; k2: 0.57→0.78→1.36→0.92→0.72), loss was
  non-monotonic (epoch 1 got worse before partially recovering), mismatches
  bounced 3→5→2→2→2. Diagnosed as the learning rate being too large for two
  COUPLED parameters that both influence the same Um — a per-parameter step
  reasonable in isolation can combine into an overshoot in the joint
  (k1, k2) space. lr=0.3 had worked fine for w and C_D, both single,
  less-coupled parameters.
- Fix: lowered the learning rate to 0.08 (~4x reduction) and added
  best-epoch tracking (report/use the lowest-loss epoch seen across the
  run, not whichever epoch happens to be last) — necessary because the
  noisy per-condition-update trajectory meant the final epoch wasn't
  reliably the best one (confirmed directly: in the first unstable run,
  epoch 3's loss was roughly half of the final epoch 4's).
- Result: converged to k1=0.6523, k2=0.9458. Evaluated against all 12
  conditions: average Um error across the 9 trained conditions ~3.2% (down
  from ~15.0% at w alone, and down from the ~20.2%/degenerate-collapse of
  the earlier raw C_D-refit attempt), every trained condition under 10%
  error, most under 5%, all 9 signs still correct — the k1/k2 fit didn't
  undo w's sign fix. The one remaining outlier (WC=0.10@Um=0.75, 17.57%
  error) is the already-understood, separately-flagged phase-collapse
  condition, unaffected by any friction/drag closure by construction (see
  "Current Limitation: Vanishing-Phase Singularity" under Known Issues).
- Side note, held loosely (not a training target, so not to be
  over-interpreted): WC=0.50@Um=0.50 — one of the two excluded
  near-inversion conditions — also picked up the correct sign under the
  fitted k1/k2, even though it was never part of the loss. Possibly
  coincidental.
- Physical interpretation: k1=0.65 (water's friction cut by ~35%) vs.
  k2=0.95 (oil's barely changed) is a different story than "friction
  magnitude was uniformly overpredicted" — it indicates the residual
  magnitude error was concentrated specifically on WATER's friction, a
  phase the S_i/w work never touched (w only ever modified oil's D_h2, via
  the interfacial chord term). This is a coherent complement to, not a
  duplicate of, the S_i/w fix: w corrected oil's distributional/geometric
  friction, k1/k2 (mostly k1) corrects water's magnitude — a lever w never
  had.
- Not yet done: baking k1/k2 in as new defaults (planned, not yet applied);
  re-running the zero-drag-ceiling sweep to check whether the conditions
  found unreachable before are now reachable under this improved friction
  closure, before deciding whether to resume the paused C_D refit.

---

## Project Overview

This project develops a 1D two-fluid model for oil-water pipe flow in Python/JAX,
validated against experimental data, with the eventual goal of replacing the
analytical interphase drag closure with a physics-informed neural network (PINN).

The work sits at the intersection of:
- Computational fluid mechanics (two-fluid model, finite volume method)
- Scientific machine learning (physics-informed neural networks)
- Experimental validation (Ibarra et al. 2015 oil-water pipe flow data)

---

## Simulation Overview

### Governing Equations

The simulation solves the two-fluid model — separate continuity and momentum
equations for each phase (water = phase 1, oil = phase 2):

```
Continuity:  d(phi_s)/dt + d(phi_s * u_s)/dx = 0
Momentum:    d(phi_s * rho_s * u_s)/dt + ... = pressure + drag + friction + gravity
```

The interphase drag term M_alpha is the unknown closure the neural network
will eventually learn. Currently a dispersed bubble formulation is used:

```python
M1 = drag_coeff * (3/4) * (phi2*phi1*rho1/d_b) * |delta_u| * delta_u
M2 = -M1
```

### Numerical Methods

- **Mass/momentum equations**: Finite volume with Lax-Friedrichs flux
- **Pressure solve**: Finite difference Thomas algorithm (tridiagonal solver)
- **Viscous stress**: Central finite difference second derivative
- **Time integration**: Forward Euler with fixed dt = 1e-4 s
- **Time loop**: `jax.lax.scan` (converted from Python while loop for differentiability)

### State Dictionary

```python
state = {
    'p':    shape (N,),   # pressure [Pa]
    'phi1': shape (N,),   # water volume fraction
    'phi2': shape (N,),   # oil volume fraction
    'rho1': shape (N,),   # water density [kg/m³]
    'rho2': shape (N,),   # oil density [kg/m³]
    'u1':   shape (N,),   # water velocity [m/s]
    'u2':   shape (N,),   # oil velocity [m/s]
}
```

### Key Physics Terms in advance_momentum

```python
mom1_new = mom1[1:-1]
         - dt * div_mom1        # convection  (Lax-Friedrichs flux)
         + dt * p_int * dphi1_dx # pressure gradient (non-conservative term)
         + dt * visc1            # viscous stress (phi*mu*d²u/dx²)
         + dt * grav1            # gravity (phi*rho*g*sin(theta))
         + dt * M1               # interphase drag
         + dt * friction1        # wall friction (Darcy-Weisbach)
```

### Drift Flux Correction in advance_mass

Added to capture phase redistribution due to slip between phases:

```python
j_drift_centers = phi1 * phi2 * (u1 - u2)   # positive when water faster
j_drift_faces   = 0.5 * (j_drift_centers[:-1] + j_drift_centers[1:])
div_drift       = (j_drift_faces[1:] - j_drift_faces[:-1]) / dx
phi1_new        = phi1_new - dt * div_drift
phi2_new        = 1.0 - phi1_new
```

Sign convention: when oil faster (u2 > u1), j_drift < 0, so -div_drift > 0,
so phi1 increases — water accumulates as expected physically.

---

## Physical Parameters (Ibarra et al. 2015)

```python
# Fluid properties — Exxsol D140 oil and water at 25°C
rho1_val = 998.0     # water density [kg/m³]
rho2_val = 825.0     # oil density [kg/m³]
mu1      = 0.9e-3    # water viscosity [Pa·s]
mu2      = 5.4e-3    # oil viscosity [Pa·s]
sigma    = 40e-3     # interfacial tension [N/m]

# Geometry
D        = 0.032     # pipe diameter [m] — 32mm ID
L        = 6.7       # pipe length [m] — matches Ibarra L/D = 209
N        = 500       # number of cells
theta    = 0.0       # horizontal [rad]

# Current drag parameters
drag_coeff = 1e-4    # effective drag coefficient
d_b        = 1e-3    # bubble/particle diameter [m]
```

**Important**: mu1 = water (0.9e-3), mu2 = oil (5.4e-3).
Phase 1 = water (denser, bottom), Phase 2 = oil (lighter, top).

---

## Simulation Time Loop Structure

Converted from Python while loop to `jax.lax.scan` for JAX differentiability.
Two-phase approach:

```python
# Phase 1: spinup — run to steady state without gradient tracking
@jax.jit
def run_spinup_full(init_state):
    final, _ = jax.lax.scan(scan_step, init_state, jnp.full(n_spinup, dt_fixed))
    return final

spinup_state = run_spinup_full(state)
spinup_state = jax.lax.stop_gradient(spinup_state)   # no gradients through spinup

# Phase 2: differentiable window — short run with gradient tracking
@jax.jit
def run_chunk(init_state):
    final, _ = jax.lax.scan(scan_step, init_state, jnp.full(save_every_window, dt_fixed))
    return final
```

Timing at N=500, dt=1e-4:
- Steps per second: ~80,000
- Full 150s spinup: ~18 seconds
- Each 1000-step chunk: ~0.012 seconds

---

## Validation

### Experimental Dataset: Ibarra et al. (2015)

**Citation**: Ibarra R, Matar OK, Markides CN, Zadrazil I. "An experimental study
of oil-water flows in horizontal pipes." BHR Group, 2015.

**What the paper provides**:
- Figure 7/10: pressure gradient dP/dL vs water cut at 6 mixture velocities
- Figure 11: in-situ water fraction ψ vs water cut (stratified flow only)
- Figure 6: flow pattern map

**Key distinction**:
```
Water cut (WC):        what you pump IN  = phi1_inlet_bc  (you control this)
In-situ holdup (psi):  what's IN the pipe = phi1_plateau  (physics determines this)
```

### Validation Approach

Primary validation (completed, ~7.2% MAE):
- Input: dP/dL from Figure 10 → set as p_inlet - p_outlet / L
- Output: mixture velocity Um = phi1*u1 + phi2*u2 at plateau
- Compare against: Um_target from figure caption

Secondary validation (holdup):
- Input: WC → phi1_inlet_bc, dP/dL from Figure 7
- Output: phi1_plateau (in-situ water fraction)
- Compare against: ψ from Figure 11

### Mixture Velocity Definitions

```python
# Volumetric (matches Ibarra's Q_T/A definition)
Um_vol = phi1*u1 + phi2*u2

# Momentum-weighted (from mixture theory)
rho_mix = phi1*rho1 + phi2*rho2
Um_mom  = (phi1*rho1*u1 + phi2*rho2*u2) / rho_mix
```

Use volumetric definition when comparing against Ibarra — they set pump
flow rates (volumetric), so Um = Q_T/A is their natural definition.

### CSV Validation Structure

One CSV per mixture velocity:
```
WC, dpdz_measured, Um_target, flow_regime, phi1_predicted,
u1_predicted, u2_predicted, Um_predicted_vol, Um_predicted_momentum
```

---

## Key Physical Concepts

### Holdup vs No-Slip

The gap between water cut (input) and holdup (measured) is the slip effect:
```
phi1_inlet = WC = 0.30    ← what you pump in (no-slip assumption)
phi1_plateau = ψ = 0.35   ← what actually exists in pipe (slip present)
```

Oil moves faster → exits sooner → water accumulates → holdup > water cut.
Your model must capture this. Currently phi1 is nearly insensitive to drag_coeff
(0.001% change across 10,000x drag variation) — a known limitation.

### Slip Velocity Sensitivity

Sensitivity check across 5 orders of magnitude of drag_coeff:
```
phi1:  changes 0.001%  ← not sensitive to drag (dominated by wall friction)
Um:    changes 0.01%   ← not sensitive to drag
slip:  changes 3.2%    ← weakly sensitive — best available training signal
```

Wall friction (~1,515 N/m³) dominates drag (~0.028 N/m³) by 54,000x,
which is why bulk quantities are insensitive to drag changes.

### Flow Regimes (Ibarra classification)

```
SS:    stratified smooth         (low Um, any WC)
SW:    stratified wavy           (moderate Um)
SWD:   stratified wavy + drops   (moderate Um)
DC:    dual continuous           (Um ≥ 0.85 m/s)
DOW:   dispersion oil-in-water   (high WC, high Um)
D:     fully dispersed
```

Avoid WC ≈ 0.5 (phase inversion point) — pressure spikes, model breaks down.

### Inversion Point

At WC ≈ 0.5, oil and water compete for continuity. Neither is clearly
continuous. Effective mixture viscosity spikes, pressure gradient peaks,
flow morphology is chaotic. No simple two-fluid model handles this well.

---

## Neural Network: Drag Closure Learning

### Research Goal

Replace the analytical drag closure M1 with a neural network that learns
the true interphase momentum exchange from experimental data.

### Three-Phase Plan

```
Phase 1:   synthetic data    → learn C_D scalar    → loss: slip velocity
           goal: verify gradient chain works end-to-end
           success: network recovers drag_coeff = 1e-4 from slip observations

Phase 2a:  Ibarra data       → learn C_D scalar    → loss: slip (back-calculated)
           back-calculate slip from Um, WC, psi:
           u1 = (WC * Um) / psi
           u2 = ((1-WC) * Um) / (1-psi)
           slip = u2 - u1

Phase 2b:  Ibarra data       → learn full M1       → loss: phi1 + u1
           ~24 data points, proof-of-concept only
           requires stratified drag formulation (current dispersed bubble wrong)

Phase 2c:  larger dataset    → learn full M1       → loss: phi1 + u1
           Arirachakaran, Lovick-Angeli, Elseth, etc.
```

Note: richer M1 (Phase 2b/2c) does not by itself resolve the vanishing-phase
singularity that bounds usable WC range (e.g. WC=0.1) -- see "Current
Limitation: Vanishing-Phase Singularity" under Known Issues.

Note: before committing to full Phase 2b, a much lighter intermediate test
was added (Section 7d) — two independent C_D scalars (`C_D_pos`/`C_D_neg`)
instead of one, split by the sign of each condition's target slip
(pre-/post-inversion), rather than a full NN-learned M1. This directly tests
whether the single-scalar plateau (see Current Status, item 7) is a
regime-dependence problem specifically, before paying the cost of the full
Phase 2b effort. **Superseded** — see the note on Current Status item 7.

**Redirect**: a zero-drag-ceiling sweep (see "C_D refit surfaces a
fundamental ceiling..." under Completed Phases) established that *any*
drag closure of this damping-force form — dispersed bubble, stratified
interfacial, or a full NN-learned M1 — collapses to the identical
zero-drag simulation as its coefficient → 0, so the achievable slip range
is set entirely by non-drag physics (wall friction) and is invariant to
which drag closure is chosen. This means richer M1 formulations (Phase
2b/2c as originally scoped) would not resolve the currently-unreachable
conditions. Current priority has shifted to characterizing and correcting
wall friction *magnitude* first (Current Status, item 9) — with the
expectation that "the NN" (if still needed once a simple scalar correction
is tested) would eventually target a wall-friction correction rather than
M1.

### Why Learn C_D First (Not Full M1)

Learning C_D_eff instead of M1 directly:
- Network only needs to learn HOW drag varies, not the full functional form
- Known physics structure is guaranteed (quadratic in delta_u, zero at no-slip)
- More interpretable — can compare C_D_eff against Blasius, Wallis correlations
- Better extrapolation — physics structure maintained outside training range
- Much smaller learning problem for limited data (~30 points)

### Network Architecture (Phase 1)

```python
class DragClosureNetwork(eqx.Module):
    layer1: eqx.nn.Linear   # 4 → 32, tanh
    layer2: eqx.nn.Linear   # 32 → 32, tanh
    layer3: eqx.nn.Linear   # 32 → 16, tanh
    layer4: eqx.nn.Linear   # 16 → 1, softplus (enforces C_D > 0)
```

Total parameters: 1,745
Input features (4, dimensional but scaled to O(1)):
```python
def build_network_inputs(phi1, u1, u2):
    delta_u = u2 - u1
    return jnp.array([
        phi1,           # water volume fraction, already O(1)
        u1   / 1.5,    # water velocity, normalized
        u2   / 1.5,    # oil velocity, normalized
        delta_u / 0.1, # slip velocity, normalized
    ])
```

Phase 2 will switch to dimensionless inputs (Re, We, phi ratios).

### Hard Physics Constraints in Architecture

```python
# C_D_eff > 0 always — via softplus output activation
C_D_eff = jax.nn.softplus(x[0])

# M2 = -M1 always — Newton's third law, computed outside network
M2 = -M1

# M1 = 0 when delta_u = 0 — via formula structure
M1 = C_D_eff * (3/4) * (phi2*phi1*rho1/d_b) * |delta_u| * delta_u
```

### Key Design Decisions

**Why Equinox over PyTorch/Flax**:
Simulation is in JAX. Gradients must flow from loss → simulation → network.
Equinox networks are JAX PyTrees — compatible with jax.grad, jax.jit, jax.vmap
without boilerplate. PyTorch is incompatible. Flax works but adds params dict overhead.

**Why tanh activation**:
Inputs are centered around 0. tanh is symmetric, smooth everywhere, bounded (-1,1).
Better gradient flow than sigmoid, handles negative inputs unlike ReLU.

**Why softplus output**:
Ensures C_D_eff > 0 always. Random init gives C_D_eff ≈ 0.693 (log(2)).
Prevents negative drag during early training which would blow up simulation.

### Training Infrastructure

```python
# Gradient check — verify chain works before training
def loss_from_drag(drag_c):
    def step(s, dt):
        return time_step_learned(s, dt, dx, drag_c, ...), None
    final, _ = jax.lax.scan(step, spinup_state, jnp.full(500, dt_fixed))
    return jnp.mean(final['u2'][N//4:3*N//4] - final['u1'][N//4:3*N//4])

grad = jax.grad(loss_from_drag)(jnp.array(drag_coeff))
# Result: d(slip)/d(drag_coeff) = -0.070  ← nonzero, chain works ✓
# Negative: stronger drag → less slip (physically correct)
```

### Loss Function (Phase 1)

```python
def loss_fn(network, condition, n_window, dt_fixed, ...):
    # Get C_D_eff from network
    x       = build_network_inputs(WC, u1_mean, u2_mean)
    C_D_eff = network(x)

    # Run differentiable simulation window with learned C_D
    def learned_step(state, dt):
        return time_step_learned(state, dt, dx, C_D_eff, ...), None

    final, _ = jax.lax.scan(learned_step, spinup_state, jnp.full(n_window, dt_fixed))

    # Extract slip from plateau region
    slip_pred   = jnp.mean(final['u2'][N//4:3*N//4] - final['u1'][N//4:3*N//4])
    slip_target = jnp.array(condition['slip_target'])

    # Normalized squared error
    loss_slip = ((slip_pred - slip_target) / (jnp.abs(slip_target) + 1e-6))**2

    return loss_slip, (phi1_pred, Um_pred, slip_pred, C_D_eff)
```

Why slip not phi1/Um: drag directly controls velocity difference between phases.
phi1 and Um are dominated by wall friction (54,000x larger than drag force).

### Training Loop Pattern

```python
for epoch in range(n_epochs):
    epoch_loss = epoch_C_D = epoch_slip_err = 0.0
    n = len(dataset)

    for cond in dataset:                          # inner loop: all conditions
        (loss_val, aux), grads = eqx.filter_value_and_grad(
            loss_fn, has_aux=True)(network, cond, ...)
        phi1_pred, Um_pred, slip_pred, C_D_pred = aux
        updates, opt_state = optimizer.update(...)
        network = eqx.apply_updates(network, updates)
        epoch_loss += float(loss_val)
        epoch_C_D  += float(C_D_pred)
        epoch_slip_err += abs(float(slip_pred) - cond['slip_target']) / ... * 100

    # After inner loop — epoch-level operations
    avg_loss = epoch_loss / n
    if epoch % print_every == 0:
        print(f"Epoch {epoch}  loss={avg_loss:.6f}  ratio={avg_C_D/drag_coeff:.3f}")
    if epoch % 50 == 0:
        eqx.tree_serialise_leaves(f"checkpoint_epoch{epoch}.eqx", network)
    if controller.stop:
        break   # exits outer loop only
```

### Saving and Resuming

```python
# Save both network and optimizer state
eqx.tree_serialise_leaves("drag_network_phase1.eqx", network)
with open("optimizer_state_phase1.pkl", "wb") as f:
    pickle.dump(opt_state, f)

# Resume (safe to rerun — checks for existing files)
if os.path.exists("drag_network_phase1.eqx"):
    network = eqx.tree_deserialise_leaves(
        "drag_network_phase1.eqx",
        DragClosureNetwork(jax.random.PRNGKey(42))
    )
    with open("optimizer_state_phase1.pkl", "rb") as f:
        opt_state = pickle.load(f)
else:
    network   = DragClosureNetwork(jax.random.PRNGKey(42))
    opt_state = optimizer.init(eqx.filter(network, eqx.is_array))
```

### Stopping Training Gracefully (Jupyter)

```python
# Run BEFORE training loop
class TrainingController:
    def __init__(self): self.stop = False
    def request_stop(self):
        self.stop = True
        print("Stop requested.")
controller = TrainingController()

# Run in a NEW cell while training is running to stop it
controller.request_stop()
```

---

## Known Issues and Open Questions

### Current Limitation: Drag Insensitivity

The dispersed bubble drag formulation with d_b=1e-3 produces a drag force
~54,000x smaller than wall friction. phi1 and Um are essentially insensitive
to drag_coeff across 5 orders of magnitude. This means:
- phi1 cannot be a training signal for drag learning with current formulation
- Must use slip velocity as training signal instead
- Stratified interfacial drag formulation would be more physically appropriate:

```python
# Stratified interfacial drag (TODO: implement and test)
a_i = 4.0 * phi1_int * phi2_int / (D + eps)   # interfacial area per volume
M1  = 0.5 * f_i * a_i * rho_mix * |delta_u| * delta_u
# f_i = interfacial friction factor (what network would learn)
```

### Current Limitation: Vanishing-Phase Singularity Bounds Usable WC Range
(independent of drag closure richness)

Discovered while investigating a Phase 2a real condition (WC=0.1@Um=0.75)
that showed excellent bulk-flow agreement (Um within -4.7% of target, clearly
converged) but a back-calculated slip sign mismatch. Root cause:
phi1_inlet = WC = 0.1 sits right at PHI_COLLAPSE, and water (the minority
phase there) collapses below threshold within ~3-6s and never recovers for
the rest of the 150s run. Once collapsed, that phase's velocity is set equal
to the dominant phase's directly in the momentum→velocity recovery step
(u_i = mom_i/(phi_i*rho_i) diverges as phi_i→0) -- **downstream of every
force term, including M1.**

Consequence for future phases: richer M1 formulations (Phase 2b's full
NN-learned M1, Phase 2c's larger dataset) will **not** resolve this on their
own. The collapse override is a property of the two-fluid (Eulerian-
Eulerian) formulation itself -- the 1/phi_i singularity as a phase vanishes
exists regardless of what closure computes the momentum source terms, and
the override sits after that closure has already acted. A richer M1 could
only help indirectly, by changing slip enough to alter how drift flux
redistributes phi1 before it ever reaches the threshold -- a weak, unproven
lever, especially for conditions (like WC=0.1) that already start at the
threshold from the inlet BC itself. Actually escaping this limitation would
require reconsidering the formulation near phase vanishing/inversion (e.g. a
drift-flux/mixture treatment, or a stratified/segregated approach that
doesn't track two independent phase velocities all the way to a vanishing
fraction), not just improving the closure term.

Practical mitigation (same spirit as the oscillation-exclusion decision under
Current Status): conditions whose assigned quiet+developed training window is
phase-collapsed are excluded from the Phase 2a training set entirely, same as
WC=0.8@Um=0.5 was excluded from Phase 1's synthetic set for the same reason
-- a collapsed condition contributes ~zero gradient signal (C_D cannot move
slip_pred away from 0) while permanently dragging the average loss up, since
it can never be fit no matter what C_D is tried.

### Drift Flux Sign

The drift flux currently uses `(u1 - u2)` which gives negative j_drift when
oil is faster — causing phi1 to increase (correct). Earlier experiments with
sign reversal caused phi1 to decrease (wrong direction). Current sign is correct.

### Phase Labeling

Phase 1 = water (rho1=998, mu1=0.9e-3)
Phase 2 = oil   (rho2=825, mu2=5.4e-3)

Note: mu1 and mu2 labels were swapped in earlier code versions. Current
values are correct — verify before any new simulation runs.

---

## Code Structure

```
Section 1:  imports
Section 2:  helper functions
                make_grid, initial_conditions
                lax_friedrichs_flux
                advance_mass          ← includes drift flux correction
                advance_momentum      ← includes drag, friction, gravity, viscosity
                pressure_poisson_solve
                project_velocities
                apply_boundary_conditions
                compute_dt, compute_diagnostics
                time_step             ← existing, captures drag_coeff from scope

Section 3:  neural network definitions
                DragClosureNetwork    ← eqx.Module, 4→32→32→16→1, softplus output
                build_network_inputs  ← 4 scaled dimensional inputs
                time_step_learned     ← wrapper with explicit drag_c argument
                loss_fn               ← slip-based loss, returns 4 aux values

Section 4:  physical parameters
Section 5:  initial conditions
Section 6:  scan infrastructure
                scan_step, run_spinup_full, run_chunk

Section 7a: synthetic dataset generation
                generate_synthetic_dataset()
                conditions list (8 points: 4 WC × 2 Um)
                slip_target = u2_mean - u1_mean (computed from existing dataset)

Section 7b: training loop
                DragClosureNetwork initialization (or load from checkpoint)
                optax.adam optimizer
                epoch loop with slip_err metric
                TrainingController stop flag
                final evaluation table

Section 8:  plotting
```

---

## References

- Ibarra R, Matar OK, Markides CN, Zadrazil I (2015). "An experimental study
  of oil-water flows in horizontal pipes." BHR Group.
  → Primary validation dataset. 32mm horizontal pipe, Exxsol D140/water.
  → Figure 10: pressure gradient. Figure 11: in-situ holdup (stratified only).

- Beggs HD (1972). "An Experimental Study of Two-Phase Flow in Inclined Pipes."
  PhD Dissertation, University of Tulsa.
  → Air-water data. Used for initial validation of momentum physics.
  → 584 tests, 1" and 1.5" pipe, 0° to ±90° inclination.

- Nagoo AS. "Pipe Fractional Flow Theory: Principles and Applications."
  PhD Dissertation, University of Texas.
  → Theoretical framework. ANNA Global Pipe Flow Database.
  → Not a two-fluid model — fractional flow / drift flux approach.

- Arirachakaran et al. (1989). SPE-18836-MS.
  → Oil-water horizontal pipe data. 1,200 data points. Future dataset for Phase 2c.

- Trallero JL (1995). "Oil-Water Flow Pattern in Horizontal Pipes."
  PhD Dissertation, University of Tulsa.
  → Most comprehensive raw oil-water holdup dataset. Target for Phase 2c.

- Taitel Y, Dukler AE (1976). Original stratified flow model paper — assumed
  a *constant* ratio between interfacial and wall friction factor, since
  identified in follow-up literature as the model's primary weakness.
  → Base geometry this project's wall friction closure is derived from; see
  "Wall friction upgrade: Taitel-Dukler stratified geometry" under
  Completed Phases.

- Rodriguez OMH, Oliemans RVA (2006). Two-fluid model errors for stratified
  oil-water flow: 20-40% average, exceeding 100% in some cases; improved to
  ~35% by incorporating interfacial wave roughness.
  → Found via literature search (2026-08-18) confirming this project's own
  Um/slip error magnitudes are typical for this model class, not a bug.
  See "C_D refit surfaces a fundamental ceiling..." under Completed Phases.

- Chakrabarti DP et al. (2005). Pressure gradient predictions for stratified
  flow vary 40-200% depending on closure relations chosen (adapting
  Brauner et al. 1998).
  → Same literature search as above.

- Ng TS, Lawrence CJ, Hewitt GF (2004). Oil-water equivalent wall/interfacial
  friction factors described as "quite inconsistent" and "difficult to
  formulate."
  → Same literature search as above.

- "Prediction of pressure gradient and hold-up in horizontal liquid-liquid
  pipe flow" (2023), Petroleum Science.
  → Reached -7.06% mean error (20.72% std) by modifying friction
  correlations for oil viscosity, interfacial curvature, and phase fraction
  jointly, rather than a single flat multiplier — reference point for how
  far a well-designed closure can go beyond a simple k1/k2 correction.
  https://www.sciencedirect.com/science/article/pii/S1995822623001681
