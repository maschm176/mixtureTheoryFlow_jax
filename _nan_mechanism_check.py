"""
Follow-up to _kh_vs_nan_timing_check.py: that script confirmed is_complex
(local hyperbolicity loss, the Bernoulli-suction mechanism) PRECEDES any
NaN by thousands of steps, but didn't identify what actually produces
the NaN once it happens. This script finds the exact single step where
NaN first appears for the same condition (idx=797, sigma=0, default
params) and inspects, right at that transition:
  (1) whether alpha_l's cell-average (not the reconstructed/positivity-
      corrected FACE values -- those are provably bounded in [0,1] by
      Kurganov-Petrova; the cell average after a flux-divergence update
      has NO defensive clip, per the module's own comment in
      _transport_only_step) has already overshot outside [0,1] before
      the step that produces NaN,
  (2) whether recover_velocities' denom (rho_l*alpha_g + rho_g*alpha_l)
      -- provably bounded away from zero ONLY for alpha_l genuinely in
      [0,1] -- goes non-positive at that same point.

Method: Phase A advances in JIT'd chunks of 200 steps (fast) until
alpha_max climbs above 0.97 or NaN appears in a chunk. Phase B then
re-runs JUST that one chunk's 200 steps ONE AT A TIME (same starting
state, fully deterministic) to pinpoint the exact step and print
per-step diagnostics around the transition.
"""
import math
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
IDX = 797
CHUNK_A = 200
MAX_CHUNKS_A = 2000  # up to 400,000 steps

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]
beta = math.radians(2.0)
L_VUONG = 85.0

row = st.loc[IDX]
U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
D = float(row['Pipe Diameter[m]'])
rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])

Q = U_sl + U_sg
nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, 0.0)
U_sl_star, U_sg_star = U_sl / Q, U_sg / Q
rho_l_star, rho_g_star = nd['rho_l_star'], nd['rho_g_star']
mu_l_star, mu_g_star = nd['mu_l_star'], nd['mu_g_star']
g_star, D_star = nd['g_star'], nd['D_star']
L_star = L_VUONG / D
dx_star = L_star / N

dt_star = solver.estimate_stable_dt(U_sl_star, U_sg_star, D_star, rho_l_star, rho_g_star,
                                     mu_l_star, mu_g_star, dx_star, g=g_star, beta=beta)

alpha_l_inlet = U_sl_star
Delta_inlet_star = rho_l_star - rho_g_star
state0 = solver.init_state_water_gas(N, U_sl_star, U_sg_star, rho_l_star, rho_g_star)

print(f"idx={IDX}  U_sl={U_sl}  U_sg={U_sg}  rho_l_star={rho_l_star:.4f}  rho_g_star={rho_g_star:.6f}  "
      f"dt_star={dt_star:.6e}  dx_star={dx_star:.6e}", flush=True)


def denom_minmax(alpha_l):
    alpha_g = 1.0 - alpha_l
    denom = rho_l_star * alpha_g + rho_g_star * alpha_l + 1e-12
    return float(jnp.min(denom)), float(jnp.max(denom))


@jax.jit
def one_step(s):
    ns, nkh = solver.advance_state(
        s, dt_star, dx_star, 1.0, D_star, rho_l_star, rho_g_star,
        mu_l_star, mu_g_star, alpha_l_inlet, Delta_inlet_star, g_star,
        beta=beta, sigma=0.0
    )
    return ns, nkh


@jax.jit
def chunk_A(s):
    def body(carry, _):
        ns, _ = one_step(carry)
        return ns, None
    final, _ = jax.lax.scan(body, s, None, length=CHUNK_A)
    return final


# ---------------- Phase A: coarse search ----------------
state = state0
chunk_start_state = state0
chunk_start_step = 0
found_chunk = False

for ci in range(MAX_CHUNKS_A):
    chunk_start_state = state
    chunk_start_step = ci * CHUNK_A
    state = chunk_A(state)
    jax.block_until_ready(state)

    any_nan = bool(jnp.any(jnp.isnan(state['alpha_l'])) or jnp.any(jnp.isnan(state['Delta'])))
    a_max = float(jnp.nanmax(state['alpha_l'])) if not any_nan else float('nan')
    a_min = float(jnp.nanmin(state['alpha_l'])) if not any_nan else float('nan')

    if ci % 10 == 0:
        print(f"[A] step {( ci+1)*CHUNK_A:>7}  any_nan={any_nan}  alpha_min={a_min:.5f}  alpha_max={a_max:.5f}", flush=True)

    if any_nan or (not math.isnan(a_max) and a_max > 0.97):
        print(f"\n[A] transition chunk found: steps {chunk_start_step} -> {(ci+1)*CHUNK_A} "
              f"(any_nan={any_nan}, alpha_max={a_max})", flush=True)
        found_chunk = True
        break

if not found_chunk:
    print("Phase A never found a transition chunk within MAX_CHUNKS_A -- stopping.", flush=True)
    raise SystemExit(0)

# ---------------- Phase B: fine, step-by-step, within that one chunk ----------------
print(f"\n[B] stepping one-at-a-time from step {chunk_start_step}:", flush=True)
print(f"{'step':>7} {'any_nan':>8} {'a_min':>10} {'a_max':>10} {'out_of_[0,1]':>12} "
      f"{'denom_min':>12} {'denom_max':>12}", flush=True)

s = chunk_start_state
first_nan_step = None
pre_nan_state = None
for k in range(CHUNK_A):
    step_num = chunk_start_step + k

    a_min_pre = float(jnp.min(s['alpha_l']))
    a_max_pre = float(jnp.max(s['alpha_l']))
    out_of_bounds_pre = a_min_pre < 0.0 or a_max_pre > 1.0
    dmin_pre, dmax_pre = denom_minmax(s['alpha_l'])

    s_new, _ = one_step(s)
    jax.block_until_ready(s_new)
    any_nan = bool(jnp.any(jnp.isnan(s_new['alpha_l'])) or jnp.any(jnp.isnan(s_new['Delta'])))

    if out_of_bounds_pre or any_nan or k < 3 or k % 20 == 0:
        print(f"{step_num:>7} {str(any_nan):>8} {a_min_pre:>10.5f} {a_max_pre:>10.5f} "
              f"{str(out_of_bounds_pre):>12} {dmin_pre:>12.6e} {dmax_pre:>12.6e}", flush=True)

    if any_nan and first_nan_step is None:
        first_nan_step = step_num + 1
        pre_nan_state = s
        print(f"\n[B] FIRST NaN appears going INTO step {first_nan_step} "
              f"(i.e. produced by advance_state call number {step_num}).", flush=True)
        print(f"[B] pre-NaN state (step {step_num}, the input to the NaN-producing call):")
        print(f"    alpha_l min={a_min_pre:.6f}  max={a_max_pre:.6f}  out_of_[0,1]={out_of_bounds_pre}")
        print(f"    recover_velocities denom min={dmin_pre:.6e}  max={dmax_pre:.6e}")
        # Where exactly is the min/max alpha_l, and what does recover_velocities give there?
        idx_min = int(jnp.argmin(s['alpha_l']))
        idx_max = int(jnp.argmax(s['alpha_l']))
        u_l, u_g = solver.recover_velocities(s['alpha_l'], s['Delta'], 1.0, rho_l_star, rho_g_star)
        print(f"    cell with min alpha_l: idx={idx_min}  alpha_l={float(s['alpha_l'][idx_min]):.6f}  "
              f"u_l={float(u_l[idx_min]):.4f}  u_g={float(u_g[idx_min]):.4f}")
        print(f"    cell with max alpha_l: idx={idx_max}  alpha_l={float(s['alpha_l'][idx_max]):.6f}  "
              f"u_l={float(u_l[idx_max]):.4f}  u_g={float(u_g[idx_max]):.4f}")
        print(f"    any |u_l| or |u_g| > 1e3 (blown up)?  "
              f"{bool(jnp.any(jnp.abs(u_l) > 1e3) or jnp.any(jnp.abs(u_g) > 1e3))}")
        break

    s = s_new

if first_nan_step is None:
    print("\n[B] No NaN found within this chunk at single-step resolution -- unexpected, investigate.", flush=True)

print("\nDONE", flush=True)
