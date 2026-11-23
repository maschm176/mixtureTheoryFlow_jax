# Investigation summary: inclined-flow validation, Reynolds stress, dt-threshold, and surface-tension planning

**Scope of this document**: everything done in this investigation arc, starting from the
decision to fix the Reynolds-stress (RS) implementation and extending through pulling real
inclined data into the water-gas solver's validation/calibration pipeline, discovering and
fixing a genuine solver bug, and planning (not yet implementing) surface tension. Written for
a reviewer with no prior context beyond the project's `CLAUDE.md` — file paths and concrete
numbers are included throughout so claims can be checked directly rather than taken on faith.

**Solver under discussion**: `mixtureTheoryFlow_jax_solver_water_gas.py` — the reduced
`(alpha_l, Delta)` two-fluid stratified gas-liquid pipe solver (see that file's own module
docstring and `CLAUDE.md`'s "Part 2: Reduced-Model Water-Gas Solver" section for background).
Entering this investigation, the solver was validated at `beta=0` (horizontal) and low liquid
holdup (`alpha_l` roughly <= 0.3-0.4) against Fan (2005) and MDPI/Carraretto (2020) data. The
documented `KNOWN LIMITATION` in that file states: no surface tension exists in the model, so
nothing regularizes Kelvin-Helmholtz (KH) growth at short wavelengths; a synthetic
`U_sl x beta` sweep at `U_sg=5 m/s` found only 7/42 conditions stable, with holdup
(`alpha_l_eq > ~0.4-0.5`) — not inclination itself — identified as the real trigger.

---

## 1. Reynolds-stress investigation: does RS provide genuine KH stabilization?

**Motivation**: before investing in surface tension, test whether a Reynolds-stress-type
closure (cited in the literature as stabilizing for similar two-fluid models) could regularize
the KH instability more cheaply.

**What was built and tested** (`_reynolds_stress_explicit_test.py`, `_reynolds_stress_implicit_test.py`,
`_rs_nu_t_sensitivity.py`, `_rs_nu_t_sensitivity_implicit.py` — all untracked, exploratory):
- Reviewed the original (explicit) RS implementation: a cell-centered formula dividing the
  already-taken flux divergence by `alpha_k_safe` *after* differencing — unbounded when
  `alpha_k` is small.
- Tried two fixes: (a) raising the floor constant (`ALPHA_SAFE_FLOOR`, made results *worse*,
  not better); (b) a face-flux reformulation — folding the `alpha_face/alpha_face_safe` ratio
  (bounded in `(0,1]`) into the flux *before* taking the divergence, instead of dividing the
  divergence after the fact.
- Explained (and the user asked for this explicitly, in depth) **why explicit RS has the
  opposite of the intended effect, and why implicit is structurally suited to the intended
  effect**: von Neumann amplification factor for explicit treatment is `g = 1 - dt*lambda`,
  which grows unboundedly past `dt*lambda > 2` — i.e. past a threshold the dt-selection logic
  (`estimate_stable_dt`) has zero awareness of, since it's purely advective-CFL-based. Implicit
  (backward Euler) treatment gives `g = 1/(1+dt*lambda)`, which stays in `(0,1]` for *any*
  `dt>=0`, `lambda>=0` — unconditionally stable regardless of step size.
- Answered a follow-up question (why RS can't just be folded into friction's existing implicit
  half-step): friction is purely local/elementwise per cell, while RS needs neighbor-cell
  information (a 3-point stencil), so it needs its own split layer.
- Built a full implicit/IMEX RS solver: `thomas_solve` (O(N) tridiagonal solver via
  `jax.lax.scan`), exploiting the exact affine structure of `recover_velocities`
  (`u_l = a_l(alpha_l) + b_l(alpha_l)*Delta`) to make the backward-Euler RS half-step a single
  exact tridiagonal solve (no Newton iteration). Hand-derived tridiagonal bands were validated
  against `jax.jacfwd` ground truth to ~1e-16 error, and a known-answer heat-equation decay
  test confirmed genuine 5-order-of-magnitude damping (after fixing a test-setup bug where
  boundary cells were pinned at perturbed rather than background values).
- Also built and tested a **cell-centered variant** under implicit treatment (what happens
  without the face-flux reformulation, just implicit time-stepping) — this matched the
  explicit cell-centered formula's own distortion (`alpha_min` ~5.7e-4, matching the explicit
  cell-centered result, not the face-flux result), confirming implicit treatment alone doesn't
  fix an unbounded-operator formula — both bounded-operator AND implicit-treatment are needed.
- Ran a `0.01x`-`100,000x` sensitivity sweep on the bounded+implicit RS machinery across two
  real Fan conditions.

**Conclusion**: Reynolds stress, in every formula (face-flux, cell-centered) and numerical
treatment (explicit, implicit) and magnitude tested, **does not provide genuine KH
stabilization**. The one apparent "improvement" at extreme magnitude was shown to be an
artifact of crushing the real KH wave amplitude to near-flatness, not real physical
stabilization. This result motivated moving on from RS as a candidate fix.

---

## 2. The low-holdup settling/kh_total discovery (not this document's main focus, but load-bearing context)

While investigating RS, a **major methodological error** was caught and corrected: the
project's `kh_total > 0` exclusion criterion for Fan/MDPI calibration data (documented in
`CLAUDE.md`) was built from runs capped at 10-15 "nominal" transit times. Direct inspection of
full spatial profiles showed the KH instability (seeded by numerical noise near the inlet)
takes substantial simulated time to propagate to the outlet sampling window — so
`alpha_l_pred` in every short run was literally just the raw inlet boundary value, not a
converged reading.

Longer, chunked settling runs (`_settling_diagnostic.py`, `_fan_settling_broad_check.py`,
`_fan_settling_scaled_check.py`, `_mdpi_settling_pdz_check.py`) established that, given enough
time (up to ~2000 nominal transit times for very-low-`U_sl` conditions), the system **does**
settle to a spatially uniform, numerically stable equilibrium that matches the independent,
grid-free closed-form `solve_taitel_dukler_holdup` solver to `1e-7`-`1e-12` across 10 Fan rows
and 8 MDPI rows (holdup and both internal/external pressure-gradient cross-checks). Critically,
this settled equilibrium is **permanently `kh_total`-accumulating** (every face, every step,
`disc<0` forever) — a genuine, inherent property of the equilibrium itself, not a transient
artifact — yet it is numerically bounded and physically correct. This is explained via a
turbulence/Reynolds-averaging analogy: pointwise/trajectory non-convergence (real; `kh_total`
scales with grid resolution `N`) coexists with genuine statistical/bulk convergence (the
equilibrium holdup value), because `alpha_l`'s governing equation is exactly conservative.

**Practical implication**: the existing `kh_total==0` exclusion criterion has been
misclassifying usable low-holdup conditions as broken, purely because evaluation runs were
never long enough. This directly unblocks the project's own planned "Stage 4" pooled
cross-fluid-pair closure work (see `CLAUDE.md`). **This does not extend to the high-holdup
problem** — see below.

---

## 3. Pulling on the high-holdup/inclination thread: real data, not synthetic sweeps

Rather than continue testing synthetic `U_sl x beta` conditions, real inclined data already
sitting unused in the repo was located and run through the solver as-is (no surface tension,
no changes):

- **Fan (2005)**: `fan2005_sw_subset.csv` has 60 rows total; 47 have nonzero
  `Inclination angle[deg]` (-2,-1,1,2). Every prior analysis had filtered to the 13 horizontal
  (0 deg) rows only.
- **Vuong (2016)**: the current `vuong2016_sw_subset.csv` (36 rows, used for the adopted
  calibration — see `CLAUDE.md`) is horizontal-only by construction. The full source
  compilation, `FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx` (424 Vuong rows), also
  has **28 rows tagged `Flow Pattern=='ST'`, all at `Inclination angle[deg]=2`** — never pulled
  into a usable subset CSV before this investigation.
- Sign convention assumed (not independently verified against either paper's text): positive
  angle = uphill, matching the solver's own documented `beta` convention.

**Result** (`_inclined_real_data_check.py`, `run_real_data_condition`, `N=150`,
`n_transit_times=150`, default closure params `f_i_mult=k_wl=k_wg=1.0`):

- **Fan, 47 inclined rows: 0/47 NaN.** 38/47 fully clean (`kh_total=0`), matching the closed-form
  equilibrium to 4-5 significant figures. 9/47 show large `kh_total` (same shear-driven
  high-`U_sg` trigger already documented for 2 of the 13 *horizontal* Fan rows in `CLAUDE.md`)
  but still settle to the exact correct equilibrium. All 47 rows have `cf_alpha_l < ~0.04` —
  deep in the already-validated low-holdup regime regardless of tilt.
- **Vuong, 28 "ST" rows: effectively 0/28 safely usable as-is.** 17/28 NaN; 7/28 show huge
  `kh_total` with wild spatial spread (`alpha_max` up to 0.93-0.97) and predictions far from the
  closed-form equilibrium; the remaining 4 looked "clean" but hadn't actually propagated from
  the inlet yet within 150 transit times. All 28 rows have `cf_alpha_l` in 0.32-0.56 —
  squarely in the documented unstable zone.

**Initial conclusion** (later partially revised, see Section 4): inclination itself isn't the
discriminant — holdup level is. Fan's inclined rows are safe because they're low-holdup
regardless of tilt; Vuong's are blocked because they're intrinsically high-holdup.

---

## 4. Fan NN recalibration + discovery and fix of a genuine solver bug

A pre-existing (undocumented, from an earlier session — `_fan_data.py`, `_fan_fit_joint_equilibrium.py`,
`_fan_fit_nn_fi_only.py`, dated 2026-11-12) Fan calibration pipeline was found, mirroring
Vuong's "Option B" hybrid closure (`f_i_mult = NN(log Re_SL, log Re_SG)`, `k_wl`/`k_wg` flat
scalars, trained via true `jax.grad` through the closed-form Taitel-Dukler equilibrium path —
see `CLAUDE.md`'s Vuong calibration history for the architecture). It was scoped to the 11
kh-clean *horizontal-only* conditions (8 with measured holdup) — too few for a fixed
train/test split, forced into leave-one-out CV.

**Extension**: `_fan_data.py` rewritten to load all 60 rows (49 kh-clean: 11 horizontal + 38
inclined, using the 9-row inclined exclusion list from Section 3), with `beta` (radians)
threaded through every function call, and a new `build_split()` grouping by
`(angle, nominal U_sg level)` producing a 39/10 train/test split spanning the full
angle/U_sg range.

**First fit attempt (on the then-unfixed solver) produced a catastrophic result**: one held-out
test condition (`U_sl=0.00308, U_sg=20.04, angle=+2deg`) blew up to **1610% holdup error**
(predicted `0.171` vs. measured `0.0100`), and two unmeasured training conditions predicted
nonsensical holdup (`0.0954`, `0.1956`) versus neighbors' `~0.005-0.01`.

**Root cause, confirmed directly, not assumed**: `solve_taitel_dukler_holdup`'s bisection
(`_bisect_ht`) silently assumes the equilibrium force balance `S_Delta(h_l)` has exactly one
root — a claim its own docstring admitted was "only re-checked at beta=0." At low `f_i_mult`
(weak interfacial friction) combined with uphill `beta` and very low `U_sl`, this is false: a
direct scan of the residual at the failing condition found **three roots** at `f_i_mult=0.25`
(`alpha_l = 0.030, 0.089, 0.113`) collapsing to **one** at `f_i_mult=0.234` (a genuine
saddle-node/fold bifurcation — the low and unstable-middle roots collide and annihilate,
confirmed via a fine `f_i_mult` sweep: `0.236 -> 3 roots (0.0834, 0.0954, 0.1943)`,
`0.234 -> 1 root (0.1973)`). Plain bisection has no way to know this — it tracks whichever root
its halving happens to land near, and jumps discontinuously onto a different (physically
different!) branch when the tracked root vanishes.

**Fix** (`mixtureTheoryFlow_jax_solver_water_gas.py`, `_bisect_ht`): added a coarse 256-point
pre-scan over `(1e-4, 1-1e-4)` to find the FIRST (lowest-`h_l`) sign crossing and restrict the
`n_iter`-step bisection to that narrow bracket — the lowest root is always the thin-film branch,
consistent with every holdup measurement validated against in this project (Fan/MDPI/Vuong are
all thin-film data). Falls back to the full bracket if the scan finds no crossing. When the
curve is monotonic (the normal case), this is a no-op by construction (only one crossing to
find).

**Regression test** (compared old vs. new `solve_taitel_dukler_holdup` directly, old version
preserved via `git stash`): bit-identical (`~1e-15` diffs, pure floating-point noise) for Fan
horizontal, MDPI, Vuong horizontal, and Vuong's own adopted calibration parameters
(`f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413`) — a verified safe no-op everywhere previously
trusted. The fix changed results ONLY in the newly-discovered corner: 6 Fan conditions at the
Fan-fit's low-`f_i_mult` parameters, and — importantly — 5 of the 28 Vuong "ST" rows even at
*completely default* parameters (`f_i_mult=k_wl=k_wg=1.0`).

**Refit Fan NN with the fixed solver**: the catastrophic outlier disappeared entirely.

| | TRAIN (n=39) | TEST (n=10) |
|---|---|---|
| holdup mean\|err%\| (max) | before fix: 8.98% (37%) / after: 8.86% (32.5%) | before: 223% (1610% outlier) / after: 21.19% (52% max, no outlier) |
| dPdz mean\|err%\| (max) | before: 11.47% (76%) / after: 8.97% (25.1%) | before: 9.68% (19.6%) / after: 8.84% (16.3%) |

This was not just a diagnostic nicety — the optimizer converged to a *different, better* basin
once gradients near the bifurcation stopped being corrupted by the old solver's inconsistent
branch selection (`f_i_mult` landed at `~0.40` at the problem condition instead of `~0.20`,
comfortably clear of the `~0.235` bifurcation threshold). Final fitted parameters:
`k_wl=2.3110, k_wg=1.2617`. Saved to `fan_nn_fi_only_best.eqx`.

**Correction to Section 3's Vuong conclusion**: the same fix revealed that 5 of the 28 Vuong
"ST" rows (`U_sl~0.005-0.01`: idx 843, 846, 854, 856, 1544) were **mischaracterized** in the
original inclined-data check. Their old `cf_alpha_l` reference (e.g. `0.4176` for idx 843) was
the OLD bisection locking onto the thick-film branch; the corrected value (`0.0140`) **exactly
matches** the dynamic solver's own already-settled prediction (`alpha_l_pred=0.01400`, to 5
significant figures) — the dynamics had been correct all along; only the diagnostic reference
was wrong. **The other 23 of 28 Vuong ST rows are unaffected by the fix** — confirmed via a
direct full-range residual scan at default parameters: exactly one root each, matching the old
`cf_alpha_l` (0.33-0.56) exactly. So the real split is **23 genuinely high-holdup / 5 actually
fine**, not 28/0.

`fan_plots/plot_fan_nn_fi_only.py` was rewritten with the corrected post-fix train/test data
(mirroring `vuong_plots/plot_vuong_nn_fi_only.py`'s structure/conventions) but **deliberately
not run yet** — saved so figures can be generated on request. `_fan_data.py`'s
`EXCLUDE_KH_USL` set and `build_split()` are the canonical source for the 49-condition Fan
kh-clean set and its train/test split going forward.

**UPDATE (2026-11-23)**: the plot script HAS been run since this was written. The figures are in
`figures/fan_nn_fi_only/` (the older horizontal-only set was moved to
`_prev_2026-11-12_horizontal_only/`), and its hardcoded table was checked against the
checkpoint with `_fan_eval_nn_checkpoint.py` (all 49 rows match to printed precision; train loss
0.029808 vs 0.029812 recorded, test 0.081156 exact). Also: the `kh_total>0` exclusion list used
here (`EXCLUDE_KH_USL`, 9 inclined rows) is now known to be unnecessary — those rows settle onto
the closed-form equilibrium at ~150 transit times (see `CLAUDE.md`, "Inclined-flow investigation,
as it stands now...") — so a refit including them would add 9 rows that all have measured holdup.

---

## 5. dt-threshold investigation on the real, corrected 23 high-holdup Vuong rows

**Motivation**: `estimate_stable_dt` is purely advective-CFL-based, with zero awareness of
local KH growth severity — the exact blind spot that caused the RS degenerate-diffusivity
problem (Section 1). Before assuming surface tension is strictly necessary, test whether the
documented high-holdup NaN is an explicit-timestep-overshoot artifact instead.

**Round 1** (`_vuong_st_dt_threshold_check.py`, 23 conditions x 4 `dt` multipliers
`[1.0, 0.25, 0.05, 0.01]`, `n_transit=10` — shortened from an initial, far-too-expensive
`n_transit=50`/6-multiplier design that was killed after 51 minutes with zero output due to
`jax.lax.scan` not early-exiting on NaN plus unrelated stdout block-buffering): **8 of 23**
conditions (797, 805, 820, 821, 822, 826, 832, 840) NaN'd at the full default `dt` but were
**fully rescued by `dt<=0.25x`**, staying stable all the way to `0.01x`. The other **15**
appeared stable even at full `dt` over this short horizon.

**Round 2** (`_vuong_st_dt_longhorizon_check.py`, same 15 rows, full `dt`, `n_transit=150` —
matching the original check's horizon): **9 of the 15** (1457, 1464, 1479, 1480, 1487, 1496,
1508, 1522, 1530) **did eventually NaN** given enough time — the short-horizon "stability" was
the same "hasn't propagated yet" artifact from Section 2, not genuine robustness, independent
of `dt`. **6 remained genuinely stable** (823, 836, 1523, 1524, 1543, 1550) at both horizons.

**Round 3** (same script, the 9 "slow" rows, `dt=0.25x`, `n_transit=150`): **all 9 stabilized.**

**Combined result: all 23 genuinely-high-holdup Vuong ST conditions avoid NaN once `dt<=0.25x`
the CFL estimate, regardless of simulated time horizon** — a clean, repeatable, same-grid/
same-total-simulated-time comparison, which is the textbook signature of a numerical
scheme-stability issue (not grid-dependent ill-posedness, which would need refinement, not a
smaller timestep, to reveal).

**But accuracy, not just stability, was then checked** (`_vuong_st_dt_fix_accuracy_check.py`,
all 23 conditions at `dt=0.25x`, `n_transit=150`): **none of the 23 conditions reach the
correct closed-form equilibrium holdup.** `alpha_l_pred` at the outlet stayed tiny (0.005-0.22)
versus `cf_alpha_l` (0.34-0.56) in every single case, with large persistent spatial spread
(`alpha_max` 0.42-0.97) elsewhere in the domain.

**Extended to `n_transit=2000`** (13x longer; `_vuong_st_longhorizon2000_check.py`, 3
representative rows, one per sub-group: 797, 1480, 1550) to rule out "just needs more time":
the outlet value was **essentially unchanged** from the 150-transit result (e.g. row 1550:
`diff=0.3803` at both horizons, literally identical) — ruling out a settling-time explanation.

**Checked for a hidden lower root** the dynamics might correctly be finding instead (the same
mechanism as Section 4's fix): confirmed **no** — a full-range residual scan at these three
conditions' default parameters finds exactly one root, matching `cf_alpha_l` exactly. The
equilibrium is genuinely unique; the dynamics just isn't reaching it.

**Conclusion**: dt-reduction eliminates the *numerical* blowup (robustly confirmed) but does
NOT make the flow reach a physically sensible, calibration-usable steady state. The uniform
equilibrium is not a dynamically stable attractor at these conditions — the flow stays
persistently non-uniform/wavy regardless of how long it runs, consistent with genuine,
sustained KH instability. **"Keep the small timestep" is not a viable substitute for surface
tension** for this dataset's actual use case (real-data calibration against measured steady
holdup) — it only prevents the crash, not the underlying physical problem.

---

## 6. Surface tension: physical derivation and the numerical subtlety (planned, NOT yet implemented)

**A factual check was done first**: grepped the solver file and the full git history for
"surface tension"/"tension" — found nothing. No prior surface-tension implementation exists
anywhere in this codebase; only the deferred TODO note in the `KNOWN LIMITATION` docstring. The
"tried explicit, had to switch to implicit" story recalled by the user appears to match the
Reynolds-stress journey (Section 1), not a prior surface-tension attempt — flagged directly
rather than assumed.

**Physical derivation** (re-deriving the model's own pressure-elimination procedure — the same
one that currently produces `S_Delta`'s friction/gravity terms and `physical_flux`'s
`h_l(alpha_l)` hydrostatic term — with a capillary pressure jump included): a curved interface
has `p_l - p_g = sigma*kappa` (Young-Laplace), where the axial curvature `kappa ~= d^2 h_l/dx^2`
in the small-slope approximation. Differentiating along `x` (to match the model's `dP/dz`
convention) and redoing the liquid-minus-gas subtraction that currently cancels the shared
pressure gradient, the capillary term does NOT cancel — it contributes a new term:

```
S_Delta_new = S_Delta_old  -+  sigma * d^3(h_l(alpha_l))/dx^3
```

Confirmed dimensionally consistent with `S_Delta`'s existing terms (Pa/m). This is a
**dispersive** (odd-order, third-derivative) correction, matching the standard literature form
for surface-tension-corrected shallow-water/stratified models — structurally different from
Reynolds stress's diffusive (second-derivative) term. Sign not yet pinned down (needs
verification once built, same discipline as every other term in this solver).

**Key numerical subtlety identified, not yet resolved**: Reynolds stress's implicit treatment
worked by freezing `alpha_l` and solving a tridiagonal linear system *in Delta*, because RS fed
back into itself through Delta (via `recover_velocities`' affine structure). Surface tension's
new term depends on `alpha_l` ONLY, not `Delta`. Freezing `alpha_l` and attempting the same
trick collapses to a trivial, already-unconditionally-exact explicit update for Delta (nothing
to solve — the right-hand side doesn't contain `Delta_new` at all). The genuine feedback loop
that could cause instability runs **across** time steps (`alpha_l -> Delta kick (via the
curvature term) -> u_l (via recover_velocities) -> alpha_l's own flux on the NEXT transport
step -> repeat`), not within a single half-step. This means real stabilization likely requires
treating `alpha_l` and `Delta` **jointly**, implicitly — a materially bigger numerical-methods
task than RS's single-tridiagonal-solve pattern, not a direct reuse of RS's machinery.

**Two paths proposed, neither started**:
1. Build the full coupled (alpha_l, Delta) implicit treatment directly (the numerically correct
   fix, bigger lift).
2. Cheaply build the simpler frozen-`alpha_l`/explicit-Delta version first and empirically test
   whether it actually blows up at normal `dt` over many repeated steps, before committing to
   the bigger joint solve (consistent with this project's repeated "cheap test before big
   investment" pattern — e.g. Section 1's RS investigation, Section 5's dt-threshold check).

**This is the open, next thread of work** — not yet decided which path to take, and no code
has been written for surface tension as of this document.

---

## 7. Status of in-progress / uncommitted artifacts

**UPDATE (2026-11-23)**: the `_bisect_ht` fix described in Section 4 is NOT uncommitted any more —
it is in commit `fe9d0d8` (and surface tension in `653d4eb`; see Section 15). The remaining files
listed below are still untracked unless stated otherwise in Section 15.

All of the following are untracked, exploratory files (not yet committed) as of this document:
- `_reynolds_stress_explicit_test.py`, `_reynolds_stress_implicit_test.py`,
  `_rs_nu_t_sensitivity.py`, `_rs_nu_t_sensitivity_implicit.py` — RS investigation (Section 1).
- `_settling_diagnostic.py`, `_fan_settling_broad_check.py`, `_fan_settling_scaled_check.py`,
  `_mdpi_settling_pdz_check.py` — low-holdup settling discovery (Section 2).
- `_inclined_real_data_check.py` — real inclined-data zero-refit check (Section 3).
- `_fan_data.py`, `_fan_fit_joint_equilibrium.py`, `_fan_fit_nn_fi_only.py`,
  `fan_plots/plot_fan_nn_fi_only.py` — Fan NN recalibration (Section 4); the plot script is
  updated with corrected data but has not been run (no PNGs generated yet).
- `_vuong_st_dt_threshold_check.py`, `_vuong_st_dt_longhorizon_check.py`,
  `_vuong_st_dt_fix_accuracy_check.py`, `_vuong_st_longhorizon2000_check.py` — dt-threshold
  investigation (Section 5).
- `fan2005_oscillation_sweep_results.pkl`, `fan2005_real_data_results.pkl`, `fan_nn_fi_only_best.eqx`
  — cached results from the above.

**Committed, production change**: `mixtureTheoryFlow_jax_solver_water_gas.py`'s `_bisect_ht`
multi-root fix (Section 4) is a real, regression-tested change to the production solver file —
this is the one piece of this investigation that has actually modified shipped code. It has
**not yet been committed to git** as of this document (check `git status` / `git diff` before
assuming it's part of any existing commit).

---

## 8. Open questions as of the end of the surface-tension-planning stage (superseded below)

1. Which surface-tension implementation path (Section 6, full joint-implicit vs. cheap-test-first)?
2. Should the Fan/MDPI/Vuong training datasets be rebuilt using a genuine settling-based
   criterion (replacing the stale `kh_total==0` cutoff) now that Section 2's finding is
   established? Explicitly deferred by the user in favor of the inclination thread.
3. Should the original synthetic `U_sl x beta` sweep (42 conditions, 7/42 survived, documented
   in the `KNOWN LIMITATION` section) be rerun once surface tension exists, to see if the
   stability boundary actually moves?
4. The Vuong "23 genuinely high-holdup / 5 actually fine" split (Section 4) should be folded
   into any future Vuong dataset rebuild that incorporates inclined data.
5. `fan_plots/plot_fan_nn_fi_only.py` is ready to run whenever figures are wanted.
6. Should `_bisect_ht`'s fix be committed to git on its own, given it's a genuine, isolated,
   regression-tested bug fix independent of the surface-tension work?

**Resolution of these as of Section 9+ below**: (1) decided via the "cheap test first" path
(#2 from Section 6) — built, then found moot once the mechanism was understood (Section 10).
(6) done — committed standalone as `fe9d0d8` before the surface-tension work. (2) and (3)
remain open/deferred (listed again in Section 15). (4) is exactly what Section 13's dedicated
ST fit uses. (5) still not run (unrelated to the ST/inclination thread, lower priority).

---

## 9. Surface tension built: joint-implicit (alpha_l, Delta) half-step

**Why not just reuse Reynolds stress's implicit machinery (Section 1)**: RS's tridiagonal
solve worked because RS fed back into itself *through `Delta`* via `recover_velocities`'s
affine structure — freezing `alpha_l` and solving for `Delta` alone was a legitimate
simplification. Surface tension's new term (`sigma * d^3(h_l(alpha_l))/dx^3`, Section 6)
depends on `alpha_l` ONLY. Freezing `alpha_l` the same way collapses the update to a trivial,
already-explicit statement for `Delta` — there is nothing implicit left to solve, because the
right-hand side never contains `Delta_new`. The genuine feedback loop that could destabilize
runs *across* steps (`alpha_l -> Delta kick (curvature term) -> u_l (recover_velocities) ->
alpha_l's own flux next step -> repeat`), so a real fix needs `alpha_l` and `Delta` solved
**jointly**, implicitly, in the same half-step — the "bigger lift" path from Section 6 was the
one actually built.

**Implementation** (`mixtureTheoryFlow_jax_solver_water_gas.py` — committed in `653d4eb`):
- New `_st_joint_residual`/`_implicit_surface_tension_half_step`: a joint Newton solve over
  `(alpha_l, Delta)` as coupled unknowns, Strang-split around the existing transport step
  exactly the way the friction half-step already is.
- **Why joint, not just an implicit update to `Delta` alone**: backward-Euler on a 2-variable
  *coupled* dispersive feedback loop gives amplification factor `|g| = 1/sqrt(1+(dt*omega)^2)
  <= 1` for *any* `dt` — unconditionally stable. Treating only one leg of the loop implicitly
  (e.g. `Delta` implicit, `alpha_l` still explicit) reverts to the explicit case's unbounded
  growth, defeating the purpose.
- **A double-counting bug caught before writing any code**: `alpha_l`'s own continuity
  equation has no existing source/flux split for Strang splitting to exploit (unlike `Delta`'s,
  which already has one for friction). So the joint Newton solve computes `alpha_l_new` only
  to make `Delta_new` self-consistent, and **discards it** — only `Delta_new` is written back;
  `alpha_l`'s real evolution stays 100% owned by the existing `_ssp_rk2_transport` step. Getting
  this backwards would have silently double-applied `alpha_l`'s own transport physics.
- Third derivative via the standard centered 4-point/5-cell stencil, 2 ghost cells per side
  (one layer deeper than the existing single-ghost-cell boundary-repetition convention).
- `sigma_star = sigma/(rho_l*Q**2*D)` added to `compute_nondim_scales` — derived by dimensional
  analysis, since (unlike most of this solver's formulas — see the Vuong nondimensionalization
  entry in `CLAUDE.md`) this term does NOT come "for free" under the existing scaling.
- **Performance**: dense `jax.jacfwd` + `jnp.linalg.solve` Newton, profiled at `N=150`, cost
  ~83x the pre-change per-step time — isolated profiling showed the dense *linear solve*
  (1277 us/call), not the Jacobian build (155 us/call), was the dominant cost, overturning the
  original plan to optimize via a sparse/colored Jacobian build. Replaced with **Jacobian-free
  Newton-Krylov** (matrix-free matvec via `jax.jvp`, solved with
  `jax.scipy.sparse.linalg.gmres`) — cut the solve cost ~200x, bringing total overhead to ~4x
  baseline.
- **Validated**: a no-op regression (`sigma=0` against the pre-change solver loaded via
  `git show HEAD:...`, `_st_noop_regression_check.py`) stayed bit-identical both before and
  after the JFNK swap; a `sigma=0.024` smoke test (`_st_sigma_on_sanity_check.py`) ran clean
  with no NaN/solve errors on an ordinary condition.

**Default**: `sigma=0.0` (a verified no-op) — the new machinery is wired through but inert
unless explicitly turned on, so nothing else in the project changed behavior as a side effect
of this work.

---

## 10. Surface tension directly tested against the actual high-holdup blocker — found to be the wrong CLASS of fix

**The payoff check** (`_vuong_st_surfacetension_check.py`, full un-reduced CFL `dt`,
`n_transit_times=150`, against the actual 23 high-holdup Vuong ST conditions this was built
for): **real-magnitude `sigma` (0.024 N/m, the real Isopar L-nitrogen value) rescued 0 of the
17 previously-NaN conditions** — identical NaN pattern to `sigma=0`, no exceptions. The 6
already-robust conditions stayed stable at both.

**Magnitude scan** (`_vuong_st_sigma_magnitude_scan.py`, one representative fast-NaN condition,
idx=797, `sigma` swept 0.024 to 24000 — i.e. up to ~1,000,000x the real value): NaN through
`sigma=240` (10,000x real); at `sigma=2400` (100,000x) the run goes *stable* but settles at
`alpha_l_pred=0.0237` against a true equilibrium of `0.4731` — badly wrong, just near-total
brute-force suppression of interfacial variation, not a real fix; NaN again at `sigma=24000`.
Non-monotonic, not a clean "just needs more sigma" story.

**The decisive diagnostic** (`_kh_vs_nan_timing_check.py`): stepped idx=797 forward in 20-step
chunks at `sigma=0`, checking at each boundary whether `local_wave_speeds`'s `is_complex` flag
(`disc = trace^2/4 - det < 0` in the flux Jacobian — computed from the LOCAL per-cell state and
fixed parameters only, **no spatial derivative, no wavenumber at all**) was active, against
whether the state had gone NaN yet. Result: `is_complex` turns on by step 160 and **stays on
continuously through step 3000** (run stopped there, still no NaN) — 10 to 28 of 150 cells have
genuinely complex eigenvalues at every single checkpoint, `alpha_max` climbing steadily
(0.68 -> 0.93). Local hyperbolicity loss is the system's **persistent operating condition**,
not a brief pre-blowup excursion.

**Conclusion**: this directly explains the null/non-monotonic result above. `is_complex` is a
`k -> 0` (local, wavenumber-independent) condition; surface tension's regularization is a
**dispersive `sigma*k^3` term that vanishes as `k -> 0` by construction**. It has no
mathematical leverage over an instability already active at the local level — cranking `sigma`
up just buys brute-force suppression of all variation (hence the wrong-equilibrium result at
`sigma=2400`), not a genuine fix, and at even higher `sigma` a presumably separate
numerical-stiffness issue (Newton/GMRES convergence, not isolated) NaNs again. **Surface
tension is the wrong CLASS of fix for this mechanism, not merely under-scaled or
under-resolved** — this directly contradicted the module docstring's prior "KNOWN LIMITATION"
framing ("expected to be the actual fix, not a refinement"), which was corrected to reflect
this finding (and this correction, plus the implementation above, is what commit `653d4eb`
captures).

**Decision**: do not pursue further sigma-magnitude tuning or grid refinement for this
instability — both would just chase the same brute-force-suppression mechanism already
observed. The new surface-tension machinery (Section 9) is left in place, wired through via
the verified `sigma=0.0` no-op default — correct machinery for a dispersive regularizer, just
not the fix for *this* mechanism; it may still be useful once short-wavelength noise needs
damping on top of a real fix. The open question left behind: why does `disc` go negative in
the first place? That's Section 11.

---

## 11. Disc-negative mechanism identified: Bernoulli-suction vs. the gravity/buoyancy restoring term

**Analytical derivation, verified against `jax.jacfwd`** (`_disc_mechanism_check.py`,
max relative error ~3.6e-11): hand-differentiated `recover_velocities`'s defining linear system
and pushed the result through `physical_flux`. With `denom = rho_l*alpha_g + rho_g*alpha_l` and
slip `du = u_g - u_l`:

```
dF_alpha/dalpha  = u_l + alpha*rho_g*du/denom
dF_alpha/dDelta  = alpha*alpha_g/denom
dF_Delta/dDelta  = (rho_l*alpha_g*u_l + rho_g*alpha*u_g)/denom
dF_Delta/dalpha  = -rho_l*rho_g*du**2/denom + (rho_l-rho_g)*g*cos(beta)*(A/S_i)
```

The interesting entry is the last one: a **negative "Bernoulli" term that grows with the
SQUARE of the slip velocity**, competing against a **positive "gravity/buoyancy" term**
depending only on density difference and local geometry (`A/S_i`, i.e. `dh_l/dalpha` — see
`h_l_of_alpha`'s own docstring) and NOT on slip at all.

**Physical mechanism** (classical two-layer shallow-water Kelvin-Helmholtz): a local interface
bulge narrows the gas channel there -> gas speeds up (Venturi/continuity) -> gas pressure drops
there (Bernoulli, `p + 1/2 rho u^2 ~= const`) -> the pressure drop sucks the interface further
up instead of pushing back -> self-reinforcing. It scales as `du^2` because both the Venturi
speed-up and Bernoulli's pressure-velocity tradeoff are themselves quadratic in velocity. The
restoring gravity term has a **fixed magnitude**, independent of slip — unlike the Bernoulli
term, which is unbounded as slip grows, so it inevitably wins at high enough slip.

**Confirmed numerically** on idx=797 (the fast-NaN Vuong ST condition, step 400): every
`is_complex`-flagged cell shows the Bernoulli term beating the gravity term by 3-5x, with the
clearest case (cell 13) showing a nearly-stagnant liquid layer (`u_l~0.011`) under fast gas
(`u_g~1.74`) — the textbook KH setup.

**Structural finding: interfacial friction (`tau_i`) cannot touch `disc` at all.** Reading
`_wall_and_interfacial_shear`/`stratified_momentum_source`/`_implicit_friction_half_step`
directly: `tau_i` only ever enters through `dynamic_source_Delta`, consumed by the Strang-split
friction half-step, which solves for `Delta` alone with `alpha_l` FROZEN — entirely separate
from `physical_flux` (the only thing `local_wave_speeds`/`is_complex` differentiates). So
`f_i_mult`/`k_wl`/`k_wg` cannot change whether `disc` goes negative at a given snapshot, at any
magnitude — this is architectural, not a question of strength.

**But friction CAN change what holdup the system is trying to reach in the first place**:
stronger interfacial friction (physically, the fast gas dragging the liquid forward harder)
lets the liquid carry the same fixed `U_sl` at a higher velocity through a correspondingly
thinner layer, pulling the *equilibrium* holdup down. If that pulls the target below the
instability threshold, the trajectory never needs to visit the unstable region to get there —
an indirect but real lever. This reframes the whole investigation from "fix the instability
directly" to "find a closure whose equilibrium naturally lands outside the unstable region" —
which is exactly what Sections 12-13 pursue.

---

## 12. Testing whether friction can route around the instability — stability succeeds, but a critical correction shows accuracy does not

**Empirical stability test** (`_fi_mult_magnitude_scan.py`, `_fi_mult_threshold_and_calibration_check.py`):
confirmed the mechanism above directly. A fine `f_i_mult` scan on idx=797 (`k_wl=k_wg=1`) shows
equilibrium holdup dropping monotonically with `f_i_mult` (0.47 at 1.0 down to 0.02 at 10000),
and a NaN/stable threshold between `f_i_mult=2.0` (NaN) and `~3.5` (clean).

The project's **already-adopted** Vuong production (SW) calibration (`f_i_mult=3.8893,
k_wl=1.5419, k_wg=0.8413` — fit entirely on the 36 low/moderate-holdup horizontal SW
conditions, for unrelated reasons, never tested against this ST subset) sits inside that
window. Applied with **zero refitting** to all 23 high-holdup ST conditions, it rescues **16 of
17** previously-NaN conditions, converging cleanly to self-consistency (matches its own steady
Taitel-Dukler equilibrium) in 15 of those 16. Two stragglers: idx=820 (still NaN — its own
target equilibrium, 0.28, sits closer to the danger threshold than the others) and idx=832
(stable but does not converge to its own equilibrium, diff=0.143 — neither investigated
further).

**CRITICAL CORRECTION — self-consistency is not accuracy, and conflating the two was a real
mistake made and caught in this investigation.** "Converges to its own equilibrium" only
confirms the transient solver correctly relaxes to whatever ITS OWN closure's steady-state
equation says the answer should be — it says nothing about whether that answer matches reality.
Checked properly against the FFPU dataset's actual measured `HL[-]` column
(`_fi_mult_vs_measured_holdup_check.py`), across the 20 ST conditions with a valid measured
value (2 rows have no measured `HL`; 1 is the still-NaN idx=820): the production SW calibration
**over-predicts holdup in every single case**, from +3.3% (lowest-flow condition) up to
+149.3% (highest-flow conditions) — **mean |err%|=79.9%, mean signed bias=+79.9%** — a large,
one-directional, systematic error, growing with flow rate/holdup rather than scattered noise.

**Conclusion**: the friction-moves-the-equilibrium mechanism (Section 11) is real and correctly
understood — stronger interfacial friction genuinely does pull the trajectory out of the
Bernoulli-dominated unstable region. But the *specific* production-calibrated values (fit on a
different dataset/regime — the horizontal SW set) overshoot well past the real answer on the
way down: numerically stable, but systematically wrong — the same "large, same-signed bias"
pattern this project has flagged before as a real cross-condition generalization gap (e.g. the
Russell/Angeli out-of-sample checks under Part 1 of `CLAUDE.md`), not a bug. Numerical
stability and physical accuracy are separate questions; passing the first does not imply
passing the second.

**Decision**: do not use the SW-fit production calibration for ST as-is. A dedicated fit
against the ST conditions specifically was needed — Section 13.

---

## 13. Dedicated ST-specific equilibrium-path fit — stability AND accuracy both achieved

> **CORRECTION (2026-11-23) — read before quoting any number in this section.**
> `_st_fit_joint_equilibrium.py` never passed `beta` to `solve_taitel_dukler_holdup` /
> `predicted_pressure_gradient`, so the +2° ST rows were fit as if horizontal. The recorded
> losses below (0.020856 train / 0.066252 test) reproduce only at `beta=0`; at `beta=+2°` the same
> parameters give 0.073373 / 0.088835, and the "15.7% mean error, +8.9% bias" below is the old
> parameters evaluated under physics they were not fit to (at `beta=0` they give 12.7%, −1.7%).
> The corrected refit (`_st_fit_joint_equilibrium_beta.py`, `st_fit_beta_result.pkl`, both warm
> starts agree) is **`f_i_mult=9.2809, k_wl=1.6477, k_wg=0.7736`**, train 0.023027 / test
> 0.062817, holdup 13.1% (bias −3.1%), dP/dz 6.0%. The stability result (0/23 NaN) was rechecked
> and holds for the corrected fit in the dynamic solver (0/23 NaN, `kh_total=0` throughout). Full
> details in `CLAUDE.md`.

**This is the resolution of the project's inclination-modeling goal** (the "23 genuinely
high-holdup" Vuong rows, Section 4, are exactly the real inclined — `beta=2deg` — ST
conditions this whole investigation arc has been circling).

**Fit** (`_st_fit_joint_equilibrium.py`): same architecture as the Vuong SW production fit
(true `jax.grad` through the closed-form Taitel-Dukler equilibrium path — no scan, no finite
differences, immune to the dynamic solver's NaN issues entirely during training, since the
equilibrium solve is a plain algebraic root-find that never touches `physical_flux` or
`local_wave_speeds`), warm-started from the SW production values. Target: the 21 ST conditions
with valid measured `HL`, split 15 train / 6 test (one held out per multi-point `U_sl` cluster;
the single-point `U_sl=0.03` cluster stays entirely in train).

Converged cleanly (best epoch 2175, ~16s wall-clock) to **`f_i_mult=10.6923, k_wl=2.6477,
k_wg=0.8510`** — train loss=0.020856, test loss=0.066252.

**Stability check on the real, production dynamic solver** (`_st_fit_stability_check.py`, all
23 conditions, including idx=820/832 which the borrowed SW calibration couldn't fully handle):
**0/23 NaN** — this dedicated fit resolves the two stragglers the borrowed calibration left
behind.

**Accuracy check against real measured holdup** (same script): mean |err%|=15.7%, **mean
signed bias=+8.9%** — mixed-sign, not one-directional — a large improvement over the borrowed
calibration's +79.9% systematic bias, and roughly comparable in magnitude to this project's
other validated closures (e.g. the SW production fit's own ~8-15% errors).

**Bottom line for the inclination goal**: a closure now exists (`f_i_mult=10.6923,
k_wl=2.6477, k_wg=0.8510`) that runs the real, production transient solver stably across all 23
real inclined (`beta=2deg`), high-holdup Vuong conditions AND matches their measured holdup to
within a mean 15.7% (mixed-sign bias) — the originally-stated goal (model data with inclination
not exclusively zero) is met for this dataset, via a dedicated-fit route rather than surface
tension or a shared SW/ST closure.

---

## 14. Extending to a single closure across BOTH low/moderate and high holdup (pooled SW+ST fit)

> **CORRECTION (2026-11-23) — the numbers in this section are superseded.**
> `_pooled_sw_st_fit_nn_fi_only.py` has the same `beta` omission for its ST rows. Its odd
> `k_wl=2.86, k_wg=0.343` (and the explanation below that a shared `f_i_mult` curve pushed the
> mismatch onto `k_wg`) came from fitting the ST rows as horizontal. Refit with `beta` threaded
> through (`_pooled_sw_st_fit_nn_fi_only_beta.py`, same architecture/split, 15000 epochs):
> **`k_wl=1.5357, k_wg=0.8567`**, train 0.019228 / test 0.045510 (old: 0.0272 / 0.0590); `f_i_mult`
> from the NN has median 4.00 for SW rows and 9.22 for ST rows; subset test/train ratios SW 1.0x,
> ST 5.0x. Mean errors over all 57 rows: holdup 8.1%, dP/dz 8.5% (test 11.4% / 8.7%). Both
> beta-aware closures are stable in the dynamic solver on all 23 ST rows. Full details in
> `CLAUDE.md`.

**Goal, restated**: having a dataset-specific ST fit (Section 13) answers "can we model high-
holdup/inclined data at all" — but the project's broader aim is a closure that spans the full
holdup range (low/moderate holdup, horizontal SW conditions, AND high holdup, inclined ST
conditions) under one function, not two disconnected fits. This section covers that attempt.

**Architecture reused unchanged**: the "Option B" hybrid already validated for SW alone
(`_vuong_fit_nn_fi_only.py`, documented in `CLAUDE.md`) — `f_i_mult = NN(log Re_SL, log Re_SG)`
(13-weight: 2 -> 3 tanh -> 1 softplus), with `k_wl`/`k_wg` as plain flat scalar leaves (NOT NN
outputs) in an `eqx.Module` (`HybridParams`). This design was chosen specifically because it
was the one architecture that had already been shown (on SW alone) to avoid the
ratio-degeneracy overfitting exploit that wrecked an earlier full-3-output NN (where `k_wg`
could collapse per-condition while `f_i_mult`/`k_wl` compensated).

**Build** (`_pooled_sw_st_fit_nn_fi_only.py`): pooled SW's existing 29-train/7-test split
(`build_split()` from `_vuong_fit_joint.py`) with ST's 15-train/6-test split (Section 13) ->
44 train / 13 test. Warm-started at the **geometric mean** (natural in log-space, how these
parameters are optimized) of the two independently-fit flat values: `f_i_mult_0=6.449,
k_wl_0=2.021, k_wg_0=0.846`.

**First result (3000 epochs)**: train loss=0.0273 (holdup=0.0160, dpdz=0.0113), test
loss=0.0590 (holdup=0.0339, dpdz=0.0251). Subset breakdown: SW train/test loss 0.0261/0.0377
(~1.4x ratio, healthy); **ST train/test loss 0.0297/0.0840 (~2.8x ratio)** — ST generalizes
distinctly worse than SW within the pooled fit. `k_wg` had drifted sharply from its 0.846 warm
start down to 0.335, with `k_wl` rising from 2.021 to 2.877, and was **still visibly moving** at
the final epoch — an open question at this point, not yet resolved.

**Convergence check (extended to 15,000 epochs,** `N_EPOCHS` **bumped, gradient on
`log_k_wl`/`log_k_wg` logged directly)**: resolved cleanly. `grad_log_kwg` was already down to
8.4e-5 by epoch 900 and sits at ~1e-6 to 1e-7 from epoch 5000 onward (brief noise spikes to
~1e-4 aside); loss moved only `0.027327 -> 0.027175` across the extra 12,000 epochs (test loss
unchanged within noise: `0.059031 -> 0.059143`). **`k_wg` has genuinely converged — to
`k_wl=2.86, k_wg=0.343` — it just does so slowly (a long, flat loss valley), not because it was
heading toward something materially different.** This directly ruled out "it hasn't finished
training yet" as an explanation for `k_wg`'s large departure from either individual fit's
~0.84-0.85 value.

**Second check — does `k_wl`/`k_wg` actually want to differ by subset, once `f_i_mult`'s shared
NN shape is frozen?** (`_pooled_freeze_fi_refit_kw_per_subset.py`, using the converged 15,000-
epoch checkpoint): froze the pooled fit's `f_i_mult` NN exactly as trained, then independently
refit only `(k_wl, k_wg)` on SW-alone and ST-alone data.

```
                         k_wl      k_wg      loss
SW-only refit          2.6056    0.3326    0.0250
ST-only refit          3.2568    0.4466    0.0371
Pooled shared value    2.8607    0.3432    (sits between the two, as expected)
```

For reference, the two fully-independent flat fits (where `f_i_mult` was ALSO free per
subset, not shared) had `k_wg=0.8413` (SW) vs. `k_wg=0.8510` (ST) — essentially identical.
**Once `f_i_mult` is forced into one shared shape, `k_wg`'s subset-optimal values split apart
(0.33 vs. 0.45, ~34%) where they previously showed almost none; `k_wl`'s split actually
*narrows*  (2.61 vs. 3.26, ~25%, versus 1.54 vs. 2.65, ~72%, when `f_i_mult` was free).**
Interpretation: this isn't evidence of a real, previously-hidden physical dependence of `k_wg`
on regime — it's evidence that constraining `f_i_mult` to one shared curve leaves residual
regime-mismatch that has to land *somewhere* among the remaining free parameters, and some of
it now lands on `k_wg` instead of (as before) entirely on `k_wl`/`f_i_mult`. Quantified cost of
keeping `k_wl`/`k_wg` shared rather than subset-specific: ~9% higher loss for SW, ~20% higher
for ST — real, but modest, nowhere near `f_i_mult`'s own ~3x variation across the fitted
`Re_SG` range.

**Decision (discussed with the user, not yet acted on further)**: do NOT give `k_wg` its own
NN. The observed subset-specific gap is comparable in size to the gap `k_wl` already has as a
plain flat scalar (never flagged as needing per-condition richness), not the kind of large
mismatch that would indicate a genuinely missing functional dependence — and letting *both*
`f_i_mult` and `k_wg` vary per-condition, with only 15-21 ST points to anchor it, is close to
the exact setup (two friction-type multipliers both resisting gas-phase momentum, both free
per-condition) that caused the earlier full-3-output NN's ratio-degeneracy overfitting. A
lighter, lower-risk alternative floated but not yet pursued: a 1-2 parameter correction on
`k_wg` alone (mirroring the successful `f_i_mult(Re_SG)` power-law extension already used for
SW-only — see `CLAUDE.md`), rather than a full network.

**Bottom line for the volume-fraction-range goal**: a single pooled closure (NN `f_i_mult`,
flat `k_wl=2.86`, `k_wg=0.343`) now exists spanning both regimes, with modest-but-real
generalization cost specifically on the high-holdup (ST) subset relative to its own dedicated
fit (Section 13: 15.7% mean holdup error / 8.9% bias vs. the pooled fit's larger, mixed-sign
per-condition ST errors, roughly -28% to +34%). Whether that cost is acceptable, or whether the
lightweight `k_wg` correction above is worth building, is an open decision — not yet made.

---

## 15. Status of in-progress / uncommitted artifacts (updated)

**Committed** (through `20c658f`, confirmed via `git status`): the `_bisect_ht` multi-root fix
(`fe9d0d8`), the surface-tension implementation (Section 9) and its "wrong class of fix"
finding (Section 10), both bundled with their `CLAUDE.md` write-ups in `653d4eb`; the
disc-negative mechanism and friction-routing findings (Sections 11-12) bundled with their
`CLAUDE.md` write-up in `20c658f`.

**Still untracked / not yet committed** (confirmed via `git status --short` as of this
document):
- `_disc_mechanism_check.py` — Jacobian derivation/verification (Section 11).
- `_fi_mult_magnitude_scan.py`, `_fi_mult_threshold_and_calibration_check.py`,
  `_fi_mult_vs_measured_holdup_check.py` — friction-routing stability/accuracy checks
  (Section 12).
- `_st_fit_joint_equilibrium.py`, `_st_fit_stability_check.py` — the dedicated ST fit and its
  stability/accuracy verification (Section 13).
- `_pooled_sw_st_fit_nn_fi_only.py`, `_pooled_freeze_fi_refit_kw_per_subset.py` — the pooled
  SW+ST Option B fit, its extended-convergence run, and the per-subset frozen-`f_i_mult`
  refit check (Section 14).
- `_vuong_st_dt_threshold_check.py`, `_vuong_st_dt_longhorizon_check.py`,
  `_vuong_st_dt_fix_accuracy_check.py`, `_vuong_st_longhorizon2000_check.py` — dt-threshold
  investigation (Section 5, carried over, still untracked).
- `_vuong_st_surfacetension_check.py`, `_vuong_st_sigma_magnitude_scan.py`,
  `_kh_vs_nan_timing_check.py` — the surface-tension payoff check and decisive diagnostic
  (Section 10).
- Checkpoints: `pooled_sw_st_nn_fi_only_best.eqx` (3000-epoch), `pooled_sw_st_nn_fi_only_extended.eqx`
  (15000-epoch, converged — the one to use going forward), `fan_nn_fi_only_best.eqx` (Section 4,
  carried over).

---

## 16. Open questions for review / next steps (current)

1. **Pooled SW+ST closure, Section 14**: accept the current pooled fit (`k_wl=2.86,
   k_wg=0.343`, shared `f_i_mult` NN) as-is, or build the lightweight `k_wg(Re_SG)`-style
   correction floated in Section 14 to recover some of the ~9-20% subset-specific accuracy
   gap? Not yet decided.
2. Should the dedicated ST fit (Section 13) be adopted as the project's "official" high-holdup
   calibration (mirroring how the SW fit is already documented as the official SW calibration
   in `CLAUDE.md`), independent of whether the pooled closure is pursued further?
3. Fold this document's Sections 9-14 findings into `CLAUDE.md`'s Part 2 narrative in the same
   level of detail already given to the Vuong SW calibration history (partially done — the
   surface-tension and disc-mechanism findings are already in `CLAUDE.md`; the ST-dedicated-fit
   and pooled-fit results from Sections 13-14 are not yet written up there).
4. Carried over from Section 8, still open: should Fan/MDPI/Vuong training datasets be rebuilt
   using a genuine settling-based criterion (Section 2) instead of the stale `kh_total==0`
   cutoff? Still deferred.
5. Carried over from Section 8, still open: rerun the original synthetic `U_sl x beta` sweep
   (42 conditions, 7/42 survived) now that the real mechanism (Section 11) is understood, to
   see whether the characterized stability boundary matches the analytical Bernoulli-vs-gravity
   picture quantitatively, not just qualitatively.
6. `fan_plots/plot_fan_nn_fi_only.py` is still ready to run whenever figures are wanted
   (unrelated to the inclination/holdup-range thread, lower priority).
   **UPDATE (2026-11-23)**: done — it has been run and the figures regenerated (see Section 4's
   update). Items 1-2 above are also overtaken by the `beta` correction in Sections 13-14.
7. None of the Section 15 untracked files have been committed yet — worth deciding which (if
   any) should be committed now that the investigation has reached a real, documented
   conclusion (Section 13's dedicated ST fit in particular is a genuine, reusable result).
