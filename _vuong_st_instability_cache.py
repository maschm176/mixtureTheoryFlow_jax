"""
Stage 1 cache for the "instability in inclined Vuong ST data" figures.

Nothing here changes the solver. It reruns two existing diagnostics and SAVES
their results (the original scripts only printed to the terminal):

  Part A (parity panel): for each of the 23 genuinely-high-holdup Vuong ST rows
         (closed-form cf_alpha_l > 0.1), run the dynamic solver at the default
         CFL dt (mult=1.0) and at 0.25x CFL for N_TRANSIT transit times, and
         save the FULL final alpha_l(x) profile, NaN flag, closed-form holdup,
         and measured holdup HL[-]. Mirrors _vuong_st_dt_threshold_check.py.

  Part B (profile + hyperbolicity-trace panels): for idx=797 at the default dt,
         step in 20-step chunks to 3000 steps, saving per chunk the number of
         cells with complex eigenvalues (is_complex), alpha_min/max, and a full
         alpha_l(x) snapshot every SNAP_EVERY chunks. Snapshots matter because
         this row goes NaN at the default dt, so its FINAL profile is useless --
         we need the profile before blow-up. Mirrors _kh_vs_nan_timing_check.py.

Output: vuong_st_instability_cache.pkl (rewritten after every row, resumable).
Run from the repo root:  python3 _vuong_st_instability_cache.py
"""
import math, os, pickle, time
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import pandas as pd
import mixtureTheoryFlow_jax_solver_water_gas as solver

N = 150
N_TRANSIT = 10.0          # same short horizon as the original threshold check
MULTS = [1.0, 0.25]       # default CFL dt, and 0.25x (the largest dt that avoided NaN)
L_VUONG = 85.0
BETA = math.radians(2.0)
IDX_TRACE = 797           # representative fast-NaN row for Part B
CHUNK, N_CHUNKS, SNAP_EVERY = 20, 150, 10
OUT = "vuong_st_instability_cache.pkl"

cache = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {"rows": {}, "trace": None}

def save():
    pickle.dump(cache, open(OUT, "wb"))

def setup(row):
    """Dimensional row -> nondimensional solver inputs (same as the existing scripts)."""
    U_sl, U_sg = float(row['vSL[m/s]']), float(row['vSG[m/s]'])
    D = float(row['Pipe Diameter[m]'])
    rho_l, rho_g = float(row['rhoL[kg/m3]']), float(row['rhoG[kg/m3]'])
    mu_l, mu_g = float(row['viscosityL[Pa.s]']), float(row['viscosityG[Pa.s]'])
    Q = U_sl + U_sg
    nd = solver.compute_nondim_scales(U_sl, U_sg, D, rho_l, rho_g, mu_l, mu_g, 9.81, 0.0)
    p = dict(U_sl=U_sl, U_sg=U_sg, D=D, rho_l=rho_l, rho_g=rho_g, mu_l=mu_l, mu_g=mu_g,
             U_sl_star=U_sl / Q, U_sg_star=U_sg / Q,
             rho_l_star=nd['rho_l_star'], rho_g_star=nd['rho_g_star'],
             mu_l_star=nd['mu_l_star'], mu_g_star=nd['mu_g_star'],
             g_star=nd['g_star'], D_star=nd['D_star'])
    p['L_star'] = L_VUONG / D
    p['dx_star'] = p['L_star'] / N
    p['dt_cfl'] = solver.estimate_stable_dt(
        p['U_sl_star'], p['U_sg_star'], p['D_star'], p['rho_l_star'], p['rho_g_star'],
        p['mu_l_star'], p['mu_g_star'], p['dx_star'], g=p['g_star'], beta=BETA)
    p['alpha_l_inlet'] = p['U_sl_star']
    p['Delta_inlet'] = p['rho_l_star'] - p['rho_g_star']
    return p

def make_runner(p, dt_star, length):
    @jax.jit
    def run(state):
        def body(s, _):
            ns, nkh = solver.advance_state(
                s, dt_star, p['dx_star'], 1.0, p['D_star'], p['rho_l_star'], p['rho_g_star'],
                p['mu_l_star'], p['mu_g_star'], p['alpha_l_inlet'], p['Delta_inlet'],
                p['g_star'], beta=BETA, sigma=0.0)
            return ns, nkh
        final, nkh = jax.lax.scan(body, state, None, length=length)
        return final, jnp.sum(nkh)
    return run

ffpu = pd.read_excel("FFPU-FilteredEntrainmentDatabase_Mod_updated2026.xlsx", sheet_name="DataBase (-)")
st = ffpu[(ffpu['Fluids'] == 'Nitrogen - Isopar L') & (ffpu['Flow Pattern'] == 'ST')]

# ---------------- Part A ----------------
print("Part A: selecting the genuinely-high-holdup rows via the closed-form solver", flush=True)
sel = []
for idx, row in st.iterrows():
    p = setup(row)
    ht = float(solver.solve_taitel_dukler_holdup(p['U_sl'], p['U_sg'], p['D'], p['rho_l'], p['rho_g'],
                                                 p['mu_l'], p['mu_g'], 1.0, 1.0, 1.0, beta=BETA))
    A, A_l, A_g, S_l, S_g, S_i = solver.td_geometry(ht, p['D'])
    cf = float(A_l / A)
    if cf > 0.1:
        # The spreadsheet uses '-' for missing values; coerce those to NaN, don't guess a number
        hl = pd.to_numeric(row['HL[-]'], errors='coerce')
        sel.append((idx, cf, float(hl)))
print(f"{len(sel)} rows selected; {sum(np.isnan(s[2]) for s in sel)} have no measured HL", flush=True)

for idx, cf, hl in sel:
    if idx in cache["rows"] and all(m in cache["rows"][idx]["runs"] for m in MULTS):
        continue
    p = setup(st.loc[idx])
    rec = cache["rows"].setdefault(idx, dict(U_sl=p['U_sl'], U_sg=p['U_sg'], cf_alpha_l=cf,
                                             HL_measured=hl, runs={}))
    for mult in MULTS:
        dt_star = p['dt_cfl'] * mult
        n_steps = int(np.ceil(N_TRANSIT * p['L_star'] / dt_star))
        t0 = time.time()
        state = solver.init_state_water_gas(N, p['U_sl_star'], p['U_sg_star'],
                                            p['rho_l_star'], p['rho_g_star'])
        final, nkh = make_runner(p, dt_star, n_steps)(state)
        jax.block_until_ready(final)
        a = np.asarray(final['alpha_l'])
        rec["runs"][mult] = dict(n_steps=n_steps, any_nan=bool(np.isnan(a).any()),
                                 alpha_l=a, kh_total=int(nkh), seconds=time.time() - t0)
        print(f"  idx={idx} mult={mult:<4} n_steps={n_steps:>8} nan={rec['runs'][mult]['any_nan']} "
              f"{rec['runs'][mult]['seconds']:.1f}s", flush=True)
    save()

# ---------------- Part B ----------------
if cache["trace"] is None:
    print(f"Part B: hyperbolicity trace + snapshots for idx={IDX_TRACE}", flush=True)
    p = setup(st.loc[IDX_TRACE])
    run_chunk = make_runner(p, p['dt_cfl'], CHUNK)

    @jax.jit
    def count_complex(s):
        _, _, is_c = solver.local_wave_speeds(s['alpha_l'], s['Delta'], 1.0, p['D_star'],
                                              p['rho_l_star'], p['rho_g_star'], p['g_star'], BETA)
        return jnp.sum(is_c)

    state = solver.init_state_water_gas(N, p['U_sl_star'], p['U_sg_star'],
                                        p['rho_l_star'], p['rho_g_star'])
    tr = dict(step=[], n_complex=[], a_min=[], a_max=[], any_nan=[], snaps={})
    for i in range(N_CHUNKS):
        tr["n_complex"].append(int(count_complex(state)))   # state BEFORE this chunk runs
        tr["step"].append(i * CHUNK)
        a_now = np.asarray(state['alpha_l'])
        tr["a_min"].append(float(np.nanmin(a_now))); tr["a_max"].append(float(np.nanmax(a_now)))
        tr["any_nan"].append(bool(np.isnan(a_now).any()))
        if i % SNAP_EVERY == 0:
            tr["snaps"][i * CHUNK] = a_now.copy()
        if tr["any_nan"][-1]:
            break
        state, _ = run_chunk(state)
        jax.block_until_ready(state)
    tr["dt_star"], tr["dx_star"], tr["L_star"] = p['dt_cfl'], p['dx_star'], p['L_star']
    cache["trace"] = tr
    save()
    print(f"  trace done: {len(tr['step'])} points, first NaN at step "
          f"{next((s for s, n in zip(tr['step'], tr['any_nan']) if n), None)}", flush=True)

print("DONE", flush=True)
