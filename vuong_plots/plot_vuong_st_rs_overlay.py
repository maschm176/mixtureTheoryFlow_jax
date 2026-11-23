"""
Figures: Reynolds stress (RS) did not remove the instability on the inclined Vuong ST row
idx=797 (+2 deg). Reads vuong_st_rs_overlay_cache.pkl (from _vuong_st_rs_overlay_cache.py,
N=1800, default CFL dt, 2 nominal transits) and vuong_st_instability_cache.pkl (for the
closed-form and measured holdup of this row). No solver runs here.

  1. vuong_st_rs_profile.png    alpha_l(x) at a common step (6000, before ANY run goes NaN)
  2. vuong_st_rs_trace.png      complex-eigenvalue cells + alpha_max vs. step, all 4 runs
  3. vuong_st_rs_magnitude.png  at step 6000: alpha spread, complex cells; and step of NaN
Usage (repo root): python3 vuong_plots/plot_vuong_st_rs_overlay.py
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S2 = '#eb6834'                                    # baseline (no RS)
RAMP = {1.0: '#9fc3ee', 100.0: '#2a78d6', 1e5: '#0d2f5e'}   # light -> dark = weak -> strong RS
OUTDIR = 'figures/st_instability'; os.makedirs(OUTDIR, exist_ok=True)
IDX, STEP_CMP, N_CELLS = 797, 6000, 1800

rs = pickle.load(open('vuong_st_rs_overlay_cache.pkl', 'rb'))
ref = pickle.load(open('vuong_st_instability_cache.pkl', 'rb'))['rows'][IDX]
cf, hl = ref['cf_alpha_l'], ref['HL_measured']
MULTS = [0.0, 1.0, 100.0, 1e5]
LABEL = {0.0: 'no RS', 1.0: 'RS, 1x nu_t', 100.0: 'RS, 100x nu_t', 1e5: 'RS, 100,000x nu_t'}
COL = {0.0: S2, **RAMP}

def style(a):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

def legend(a, **kw):
    l = a.legend(fontsize=8.5, **kw); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)

# ---------------------------------------------------------------- 1. profile at a common step
fig, a = plt.subplots(figsize=(7.4, 4.8), facecolor='white'); style(a)
x = (np.arange(N_CELLS) + 0.5) / N_CELLS
for m in MULTS:
    a.plot(x, rs[m]['snaps'][STEP_CMP], color=COL[m], lw=1.3 if m else 1.6, label=LABEL[m])
a.axhline(cf, color=INK2, lw=1.4, ls='--', label='closed-form equilibrium')
a.axhline(hl, color=INK, lw=2.6, zorder=6, label='measured HL (experiment)')
a.set_xlabel('Position along pipe,  x / L', color=INK2); a.set_ylabel('Liquid fraction  alpha_l', color=INK2)
a.set_xlim(0, 1); a.set_ylim(0, 1)
a.set_title(f'idx {IDX}, step {STEP_CMP} (before any run NaNs), N=1800: RS leaves the spikes in place;\n'
            'only the extreme magnitude flattens the profile (see trace figure for what follows)', color=INK, loc='left', fontsize=10.5)
legend(a, loc='upper left', ncol=2)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_rs_profile.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 2. traces
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.0), sharex=True, facecolor='white')
for a in (a1, a2): style(a)
for m in MULTS:
    t = rs[m]; s = np.array(t['step'])
    a1.plot(s, t['n_complex'], color=COL[m], lw=1.8, label=LABEL[m])
    a2.plot(s, t['a_max'], color=COL[m], lw=1.8)
    if 'nan_at' in t:                                # mark where a run died
        for a, y in ((a1, t['n_complex'][-1]), (a2, t['a_max'][-1])):
            a.scatter([s[-1]], [y], marker='x', s=70, color=COL[m], lw=2, zorder=5)
a1.set_ylabel(f'cells with complex\neigenvalues (of {N_CELLS})', color=INK2)
a2.set_ylabel('alpha_max', color=INK2); a2.set_ylim(0, 1)
a2.set_xlabel('Step  (x = run went NaN at the next chunk)', color=INK2)
a1.set_title(f'idx {IDX}: complex-eigenvalue cells appear at every RS magnitude\n(strongest RS only delays them; they then grow)',
             color=INK, loc='left', fontsize=10.5)
legend(a1, loc='upper left')
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_rs_trace.png', dpi=200); plt.close(fig)

# ---------------------------------------------------------------- 3. magnitude summary
def at(m, key):
    t = rs[m]; i = t['step'].index(STEP_CMP); return t[key][i]
xs = np.arange(len(MULTS)); names = ['no RS', '1x', '100x', '100,000x']
spread = [at(m, 'a_max') - at(m, 'a_min') for m in MULTS]
ncx = [at(m, 'n_complex') for m in MULTS]
nan_step = [rs[m].get('nan_at') for m in MULTS]
fig, axs = plt.subplots(1, 3, figsize=(11.2, 4.0), facecolor='white')
for a in axs: style(a); a.set_xticks(xs); a.set_xticklabels(names, color=INK2); a.set_xlabel('RS magnitude (nu_t multiplier)', color=INK2)
cols = [COL[m] for m in MULTS]
axs[0].bar(xs, spread, color=cols, width=0.6); axs[0].set_ylabel('alpha_max - alpha_min', color=INK2)
axs[0].set_title(f'profile variation at step {STEP_CMP}', color=INK, loc='left', fontsize=10)
last_n = [rs[m]['n_complex'][-1] for m in MULTS]; last_s = [rs[m]['step'][-1] for m in MULTS]
axs[1].bar(xs, last_n, color=cols, width=0.6); axs[1].set_ylabel(f'cells with complex eigenvalues (of {N_CELLS})', color=INK2)
axs[1].set_title("complex cells at each run's last step", color=INK, loc='left', fontsize=10)
axs[1].set_ylim(0, max(last_n) * 1.15)
for i, (v, st_) in enumerate(zip(last_n, last_s)): axs[1].text(i, v + 6, f'{v}\n(step {st_})', ha='center', color=INK2, fontsize=8.5)
top = max(nan_step[i] or 0 for i in range(4))
ns = [(v if v else rs[m]['step'][-1]) for v, m in zip(nan_step, MULTS)]
axs[2].bar(xs, ns, color=cols, width=0.6); axs[2].set_ylabel('step at which run went NaN\n(or end of run)', color=INK2)
axs[2].set_title('time to NaN', color=INK, loc='left', fontsize=10)
for i, (v, m) in enumerate(zip(nan_step, MULTS)):
    axs[2].text(i, ns[i] + 150, f'NaN @ {v}' if v else 'no NaN', ha='center', color=INK2, fontsize=9)
axs[2].set_ylim(0, max(ns) * 1.15)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_rs_magnitude.png', dpi=200); plt.close(fig)
print('wrote', sorted(f for f in os.listdir(OUTDIR) if 'rs_' in f))
