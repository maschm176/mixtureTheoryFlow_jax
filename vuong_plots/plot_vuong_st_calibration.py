"""
Figures: the beta-aware ST calibrations on the inclined (+2 deg) Vuong ST rows, checked with the DYNAMIC solver.
Closures: flat_ST = dedicated 3-scalar fit (st_fit_beta_result.pkl); pooled_NN = pooled SW+ST NN fit, 'base' variant
(pooled_sw_st_nn_fi_only_beta_base.eqx). Reads st_closure_stability_shard*.pkl (_st_closure_stability_check.py: production
solver, default dt, 150 transits, N=150, beta=+2deg), st_closure_row797_trace.pkl (_st_closure_row797_trace.py) and
pooled_beta_result_base.pkl (for the SW rows' f_i_mult). No solver runs here.

  1. st_cal_parity.png            measured vs dynamic-solver holdup, one panel per closure, train/test marked
  2. st_cal_error_vs_holdup.png   signed holdup error vs measured holdup, both closures
  3. st_cal_self_consistency.png  |dynamic - own closed-form equilibrium| per row, both closures (+ kh_total / NaN counts)
  4. st_cal_profile.png           idx 797 alpha_l(x) at several transit times, one panel per closure
  5. st_cal_trace.png             idx 797 outlet/pipe-mean alpha_l and complex-eigenvalue cells vs transit time
  6. st_cal_fi_mult.png           f_i_mult each closure assigns, vs gas rate (pooled NN across SW and ST rows)
Positive angle = uphill is ASSUMED (unverified against the source paper). Usage (repo root): python3 vuong_plots/plot_vuong_st_calibration.py
"""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2 = '#2a78d6', '#eb6834'
OUTDIR = 'figures/st_calibration'; os.makedirs(OUTDIR, exist_ok=True)
NAMES = {'flat_ST': 'flat 3-scalar ST fit', 'pooled_NN': 'pooled SW+ST NN fit'}
COL = {'flat_ST': S1, 'pooled_NN': S2}

cache = {}
for p in sorted(glob.glob('st_closure_stability_shard*.pkl')):
    cache.update(pickle.load(open(p, 'rb')))
tr = pickle.load(open('st_closure_row797_trace.pkl', 'rb'))
pooled = pickle.load(open('pooled_beta_result_base.pkl', 'rb'))
rows = [dict(idx=i, **r) for i, r in sorted(cache.items())]
print(f"{len(rows)} ST rows loaded")
for c in NAMES:
    nn_ = sum(r['runs'][c]['any_nan'] for r in rows); kh = sum(r['runs'][c]['kh_total'] > 0 for r in rows)
    print(f"{c}: NaN {nn_}/{len(rows)}  kh_total>0 {kh}/{len(rows)}")

def style(a):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

def legend(a, **kw):
    l = a.legend(fontsize=8.5, **kw); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)

meas = [r for r in rows if r['HL'] == r['HL']]
hl = np.array([r['HL'] for r in meas]); split = np.array([r['split'] for r in meas])
def pred(c, key='alpha_l_pred'): return np.array([r['runs'][c][key] for r in meas])
def err(c): return (pred(c) - hl) / hl * 100
u = ((np.arange(len(hl)) * 0.6180339887) % 1.0) * 2 - 1; xj = hl * (1 + 0.04 * u)   # HL reported coarsely -> jitter

# ------------------------------------------------------------------ 1. parity, one panel per closure
fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.0), facecolor='white', sharex=True, sharey=True)
lo, hi = 0.006, 0.14
for a, c in zip(axs, NAMES):
    style(a)
    x = np.array([lo, hi])
    a.plot(x, x, color=INK2, lw=1.2, zorder=1)
    for f, ls in ((0.8, '--'), (1.2, '--')): a.plot(x, x * f, color=INK2, lw=0.9, ls=ls, zorder=1)
    for sp, mk, nm in (('train', 'o', 'train'), ('test', 's', 'test (held out)')):
        m = split == sp
        a.scatter(xj[m], pred(c)[m], s=60, marker=mk, color=COL[c], edgecolors='white', lw=0.9, zorder=4, label=nm)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel('Measured liquid holdup  HL  (jittered)', color=INK2)
    e = err(c)
    a.set_title(f'{NAMES[c]}\nmean |error| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%  (n={len(e)})', color=INK, loc='left', fontsize=10.5)
    legend(a, loc='upper left')
axs[0].set_ylabel('Dynamic-solver predicted holdup  (outlet window, 150 transits)', color=INK2)
fig.suptitle('Inclined (+2 deg) Vuong ST rows: calibrated closures, production solver at default dt   (dashed = +-20%)',
             color=INK, x=0.01, ha='left', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f'{OUTDIR}/st_cal_parity.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 2. signed error vs measured holdup
fig, a = plt.subplots(figsize=(7.2, 4.8), facecolor='white'); style(a)
a.axhline(0, color=INK, lw=1.2)
for c in NAMES:
    for sp, mk in (('train', 'o'), ('test', 's')):
        m = split == sp
        a.scatter(hl[m] * (1 + (0.03 if c == 'pooled_NN' else -0.03)), err(c)[m], s=55, marker=mk, color=COL[c], edgecolors='white', lw=0.8, zorder=4)
a.set_xscale('log'); a.set_ylim(-35, 75); a.set_xlabel('Measured liquid holdup  HL', color=INK2); a.set_ylabel('holdup error  (pred - meas)/meas, %', color=INK2)
a.set_title('Holdup error vs. measured holdup, ST rows: no growth with holdup\n(compare the SW-production closure: +3% to +284%, rising with holdup)',
            color=INK, loc='left', fontsize=10.5)
legend(a, loc='upper left', handles=[
    Line2D([], [], marker='o', ls='', color=S1, markersize=8, label=NAMES['flat_ST']),
    Line2D([], [], marker='o', ls='', color=S2, markersize=8, label=NAMES['pooled_NN']),
    Line2D([], [], marker='o', ls='', color=INK2, markersize=7, label='train'),
    Line2D([], [], marker='s', ls='', color=INK2, markersize=7, label='test (held out)')])
fig.tight_layout(); fig.savefig(f'{OUTDIR}/st_cal_error_vs_holdup.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 3. dynamic solver vs the closure's own equilibrium
fig, a = plt.subplots(figsize=(8.2, 4.8), facecolor='white'); style(a)
order = sorted(range(len(rows)), key=lambda i: rows[i]['U_sl'])
xs = np.arange(len(rows))
for c, dx_ in (('flat_ST', -0.12), ('pooled_NN', 0.12)):
    d = np.array([abs(rows[i]['runs'][c]['alpha_l_pred'] - rows[i]['runs'][c]['eq_alpha']) / rows[i]['runs'][c]['eq_alpha'] for i in order])
    d = np.where(np.array([rows[i]['runs'][c]['any_nan'] for i in order]), np.nan, d)
    a.scatter(xs + dx_, np.maximum(d, 1e-9), s=50, color=COL[c], edgecolors='white', lw=0.8, zorder=4, label=NAMES[c])
a.set_yscale('log'); a.set_xticks(xs); a.set_xticklabels([f"{rows[i]['U_sl']:.3f}" for i in order], rotation=60, fontsize=8, color=INK2)
a.set_xlabel('U_sl  [m/s]  (one point per ST row)', color=INK2); a.set_ylabel('|dynamic - closure equilibrium| / equilibrium\n(exact matches plotted at 1e-9)', color=INK2)
txt = '   '.join(f"{NAMES[c]}: NaN {sum(r['runs'][c]['any_nan'] for r in rows)}/{len(rows)}, kh_total>0 {sum(r['runs'][c]['kh_total'] > 0 for r in rows)}/{len(rows)}" for c in NAMES)
a.set_title('The transient solver relaxes onto each closure\'s own equilibrium\n' + txt, color=INK, loc='left', fontsize=9.5)
legend(a, loc='best')
fig.tight_layout(); fig.savefig(f'{OUTDIR}/st_cal_self_consistency.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 4. profile idx 797
fig, axs = plt.subplots(1, 2, figsize=(11.4, 4.6), facecolor='white', sharey=True)
r797 = cache[797]
for a, c in zip(axs, NAMES):
    style(a); t = tr[c]; ts = sorted(t['snaps']); x = (np.arange(len(t['snaps'][ts[0]])) + 0.5) / len(t['snaps'][ts[0]])
    for col, k in zip(cm.Blues(np.linspace(0.4, 1.0, len(ts))), ts): a.plot(x, t['snaps'][k], color=col, lw=1.7, label=f'{k} transits')
    a.axhline(t['eq_alpha'], color=INK2, lw=1.4, ls='--', label='closure equilibrium')
    a.axhline(r797['HL'], color=INK, lw=2.4, zorder=6, label='measured HL (experiment)')
    a.set_xlabel('Position along pipe,  x / L', color=INK2); a.set_xlim(0, 1)
    a.set_title(f'{NAMES[c]}   (f_i_mult={t["params"][0]:.2f})', color=INK, loc='left', fontsize=10.5)
    legend(a, loc='lower right')
axs[0].set_ylabel('Liquid fraction  alpha_l', color=INK2)
fig.suptitle('idx 797 (U_sl=0.100, U_sg=5.50 m/s, +2 deg): calibrated closures give a smooth profile that settles, unlike the default closure',
             color=INK, x=0.01, ha='left', fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f'{OUTDIR}/st_cal_profile.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 5. trace idx 797
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True, facecolor='white')
for a in (a1, a2): style(a)
for c in NAMES:
    t = tr[c]; T = np.array(t['transit'])
    a1.plot(T, t['outlet'], color=COL[c], lw=1.8, label=f'{NAMES[c]}')
    a2.plot(T, t['n_complex'], color=COL[c], lw=1.8)
a1.axhline(r797['HL'], color=INK, lw=2.4, label='measured HL (experiment)')
a1.set_ylabel('outlet-window alpha_l', color=INK2); legend(a1, loc='center right')
a2.set_ylabel('cells with complex\neigenvalues (of 150)', color=INK2); a2.set_ylim(-3, 150)
a2.text(75, 12, 'both closures: 0 complex cells at every transit (lines overlap)', ha='center', fontsize=9, color=INK2)
a2.set_xlabel('Nominal transit times  (default dt)', color=INK2)
a1.set_title('idx 797: holdup settles and the calibrated closures never lose hyperbolicity\n(default closure: complex cells 25-112 of 150, holdup still fluctuating)',
             color=INK, loc='left', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/st_cal_trace.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 6. f_i_mult assigned
fig, a = plt.subplots(figsize=(7.2, 4.8), facecolor='white'); style(a)
prow = pooled['rows']
for sub, mk, nm in (('SW', 'o', 'SW rows (horizontal)'), ('ST', 's', 'ST rows (+2 deg)')):
    r = [x for x in prow if x['subset'] == sub]
    a.scatter([x['U_sg'] for x in r], [x['f_i_mult'] for x in r], s=45, marker=mk, color=S2, edgecolors='white', lw=0.8, alpha=0.9, zorder=4, label=f'pooled NN: {nm}')
a.axhline(cache[797]['runs']['flat_ST']['params'][0], color=S1, lw=1.8, label=f'flat ST fit: {cache[797]["runs"]["flat_ST"]["params"][0]:.2f}')
a.axhline(3.8893, color=INK2, lw=1.4, ls='--', label='flat SW production fit: 3.89')
a.set_xlabel('Superficial gas velocity  U_sg  [m/s]', color=INK2); a.set_ylabel('interfacial friction multiplier  f_i_mult', color=INK2)
a.set_title('What the pooled NN assigns: ~4 for SW rows, ~9 for ST rows\n(many ST rows share the same value, i.e. close to a two-level switch)', color=INK, loc='left', fontsize=10.5)
legend(a, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUTDIR}/st_cal_fi_mult.png', dpi=200); plt.close(fig)
print('wrote', sorted(os.listdir(OUTDIR)))
