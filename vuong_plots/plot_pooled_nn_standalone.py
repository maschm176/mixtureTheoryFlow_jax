"""
Standalone figures for the pooled SW + ST NN fit (beta-aware, 2-input 'base' variant) -- NOT paired with the flat ST scalars.

  pooled_nn_holdup_st.png      holdup parity, ST rows only (the right-hand panel of st_cal_parity.png, on its own);
                               DYNAMIC production solver, default dt, 150 transits (st_closure_stability_shard*.pkl)
  pooled_nn_dpdz_st.png        pressure-gradient parity, ST rows only (right-hand panel of st_cal_dpdz.png, on its own); dynamic
  pooled_nn_holdup_sw_st.png   holdup parity for the pooled fit on ALL 57 rows, one panel per subset (SW | ST) -- the holdup
                               counterpart of pooled_nn_dpdz_sw_st.png; EQUILIBRIUM path (pooled_beta_result_base.pkl)
Circles = train, squares = held-out test. Positive angle = uphill is ASSUMED. Output dir: figures/pooled_nn/.
Usage (repo root): python3 vuong_plots/plot_pooled_nn_standalone.py
"""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID, S2 = '#0b0b0b', '#52514e', '#e4e3df', '#eb6834'      # S2 = the pooled NN's colour in the earlier figures
OUTDIR = 'figures/pooled_nn'; os.makedirs(OUTDIR, exist_ok=True)

def style(a):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

def legend(a, handles):
    l = a.legend(handles=handles, loc='upper left', fontsize=8.5); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)

def bands(a, lo, hi, pm=(0.8, 1.2), pm2=None):
    x = np.array([lo, hi]); a.plot(x, x, color=INK, lw=1.2, zorder=1)
    for f in pm: a.plot(x, x * f, color=INK2, lw=0.9, ls='--', zorder=1)
    if pm2:
        for f in pm2: a.plot(x, x * f, color=INK2, lw=0.9, ls=':', zorder=1)

def jitter(x, n, frac=0.04): return x * (1 + frac * (((np.arange(n) * 0.6180339887) % 1.0) * 2 - 1))

# ---------------- ST rows, dynamic solver (pooled NN closure only)
cache = {}
for p in sorted(glob.glob('st_closure_stability_shard*.pkl')): cache.update(pickle.load(open(p, 'rb')))
rows = [dict(idx=i, **r) for i, r in sorted(cache.items())]
for r in rows: r['run'] = r['runs']['pooled_NN']

# 1. holdup, ST rows (21 with measured HL)
m = [r for r in rows if r['HL'] == r['HL']]
hl = np.array([r['HL'] for r in m]); pr = np.array([r['run']['alpha_l_pred'] for r in m]); sp = np.array([r['split'] for r in m])
e = (pr - hl) / hl * 100; xj = jitter(hl, len(hl)); lo, hi = 0.006, 0.14
fig, a = plt.subplots(figsize=(6.4, 5.4), facecolor='white'); style(a); bands(a, lo, hi)
for k, mk in (('train', 'o'), ('test', 's')):
    mm = sp == k; a.scatter(xj[mm], pr[mm], s=60, marker=mk, color=S2, edgecolors='white', lw=0.9, zorder=4)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('Measured liquid holdup  HL  (jittered)', color=INK2); a.set_ylabel('Dynamic-solver predicted holdup  (outlet window, 150 transits)', color=INK2)
a.set_title(f'Pooled SW+ST NN fit, ST rows (+2 deg)\nmean |error| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%  (n={len(e)}; dashed = +-20%)', color=INK, loc='left', fontsize=10.5)
legend(a, [Line2D([], [], marker='o', ls='', color=S2, markersize=8, label=f'train (n={int((sp == "train").sum())})'),
           Line2D([], [], marker='s', ls='', color=S2, markersize=8, label=f'test, held out (n={int((sp == "test").sum())})')])
fig.tight_layout(); fig.savefig(f'{OUTDIR}/pooled_nn_holdup_st.png', dpi=200); plt.close(fig)
print(f"ST holdup: mean|err| {np.abs(e).mean():.1f}% bias {e.mean():+.1f}% max {np.abs(e).max():.0f}% within20 {(np.abs(e) <= 20).sum()}/{len(e)}")

# 2. pressure gradient, ST rows (all 23; the 2 without measured holdup were in neither train nor test)
meas = np.array([r['dpdz'] for r in rows]); pd_ = np.array([r['run']['dpdz_pred'] for r in rows]); usg = np.array([r['U_sg'] for r in rows])
sp = np.array([r['split'] for r in rows]); e = (pd_ - meas) / meas * 100
norm = plt.Normalize(usg.min(), usg.max())
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues_trunc', plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
lo, hi = 45, 125
fig, a = plt.subplots(figsize=(6.9, 5.4), facecolor='white', layout='constrained'); style(a); bands(a, lo, hi, pm=(0.9, 1.1), pm2=(0.8, 1.2))
MK = {'train': 'o', 'test': 's', 'no HL': '^'}; LAB = {'train': 'train', 'test': 'test, held out', 'no HL': 'not in fit (no measured holdup)'}
for k in MK:
    mm = sp == k
    if mm.any(): a.scatter(meas[mm], pd_[mm], c=usg[mm], cmap=cmap, norm=norm, marker=MK[k], s=60, edgecolors='white', lw=0.9, zorder=4)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('measured |dP/dz|  [Pa/m]', color=INK2); a.set_ylabel('predicted |dP/dz|  [Pa/m]  (dynamic solver, 150 transits)', color=INK2)
a.set_title(f'Pooled SW+ST NN fit, ST rows (+2 deg)\nmean |error| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%  (n={len(e)}; max {np.abs(e).max():.0f}%)\n(dashed = +-10%, dotted = +-20%)',
            color=INK, loc='left', fontsize=10.5)
legend(a, [Line2D([], [], marker=MK[k], ls='', color=INK2, markersize=7, label=f'{LAB[k]} (n={int((sp == k).sum())})') for k in MK])
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=a, fraction=0.05, pad=0.02); cb.set_label('U_sg  [m/s]', color=INK2); cb.ax.tick_params(colors=INK2)
fig.savefig(f'{OUTDIR}/pooled_nn_dpdz_st.png', dpi=200); plt.close(fig)
print(f"ST dP/dz: mean|err| {np.abs(e).mean():.1f}% bias {e.mean():+.1f}% max {np.abs(e).max():.0f}% within10 {(np.abs(e) <= 10).sum()}/{len(e)} within20 {(np.abs(e) <= 20).sum()}/{len(e)}")

# 3. holdup for the pooled fit on ALL 57 rows, SW | ST (equilibrium path)
prow = pickle.load(open('pooled_beta_result_base.pkl', 'rb'))['rows']
SUB = {'SW': 'SW rows (horizontal)', 'ST': 'ST rows (+2 deg)'}
fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.3), facecolor='white', layout='constrained')
for a, sub in zip(axs, SUB):
    r = [x for x in prow if x['subset'] == sub]
    h = np.array([x['HL'] for x in r]); p = np.array([x['hl_pred'] for x in r]); s_ = np.array([x['split'] for x in r]); e = (p - h) / h * 100
    lo, hi = min(h.min(), p.min()) * 0.8, max(h.max(), p.max()) * 1.25
    style(a); bands(a, lo, hi); xj = jitter(h, len(h), 0.03)
    for k, mk in (('train', 'o'), ('test', 's')):
        mm = s_ == k; a.scatter(xj[mm], p[mm], s=60, marker=mk, color=S2, edgecolors='white', lw=0.9, zorder=4)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel('Measured liquid holdup  HL  (jittered)', color=INK2); a.set_ylabel('predicted liquid holdup  (equilibrium path)', color=INK2)
    a.set_title(f'{SUB[sub]}\nmean |error| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%  (n={len(e)}; within 20%: {(np.abs(e) <= 20).sum()}/{len(e)})', color=INK, loc='left', fontsize=10.5)
    legend(a, [Line2D([], [], marker='o', ls='', color=S2, markersize=8, label=f'train (n={int((s_ == "train").sum())})'),
               Line2D([], [], marker='s', ls='', color=S2, markersize=8, label=f'test, held out (n={int((s_ == "test").sum())})')])
    et = e[s_ == 'test']
    print(f"{sub} holdup: mean|err| {np.abs(e).mean():.1f}% bias {e.mean():+.1f}% max {np.abs(e).max():.0f}% within20 {(np.abs(e) <= 20).sum()}/{len(e)} | test-only mean|err| {np.abs(et).mean():.1f}%")
fig.suptitle('Pooled SW + ST NN fit: holdup   (dashed = +-20%)', color=INK, x=0.01, ha='left', fontsize=11)
fig.savefig(f'{OUTDIR}/pooled_nn_holdup_sw_st.png', dpi=200); plt.close(fig)
print('wrote', sorted(os.listdir(OUTDIR)))
