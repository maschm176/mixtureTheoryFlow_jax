"""
Pressure-gradient parity for the pooled SW + ST NN fit (beta-aware, 2-input 'base' variant) on ALL 57 pooled rows:
36 horizontal SW rows (29 train / 7 test) and 21 inclined +2 deg ST rows (15 train / 6 test). One panel per subset on shared
axes; colour = U_sg, circles = train, squares = held-out test; dashed = +-10%, dotted = +-20%.
Source: pooled_beta_result_base.pkl (_pooled_sw_st_fit_nn_fi_only_beta.py), EQUILIBRIUM-path predictions (the dynamic solver
matched the equilibrium path to <1e-8 on the ST rows). Output: figures/pooled_nn/pooled_nn_dpdz_sw_st.png.
Usage (repo root): python3 vuong_plots/plot_pooled_nn_dpdz.py
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
OUTDIR = 'figures/pooled_nn'; os.makedirs(OUTDIR, exist_ok=True)
rows = pickle.load(open('pooled_beta_result_base.pkl', 'rb'))['rows']
allm = np.array([r['dpdz'] for r in rows]); allp = np.array([r['dpdz_pred'] for r in rows]); usg = np.array([r['U_sg'] for r in rows])
lo, hi = min(allm.min(), allp.min()) * 0.8, max(allm.max(), allp.max()) * 1.25
norm = plt.Normalize(usg.min(), usg.max())
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues_trunc', plt.cm.Blues(np.linspace(0.3, 1.0, 256)))   # keep low-U_sg points visible
SUB = {'SW': 'SW rows (horizontal)', 'ST': 'ST rows (+2 deg)'}

fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.3), facecolor='white', layout='constrained')   # independent axes: SW is 3-22 Pa/m, ST is 57-95
for a, sub in zip(axs, SUB):
    r = [x for x in rows if x['subset'] == sub]
    m = np.array([x['dpdz'] for x in r]); p = np.array([x['dpdz_pred'] for x in r]); u = np.array([x['U_sg'] for x in r])
    sp = np.array([x['split'] for x in r]); e = (p - m) / m * 100
    lo, hi = min(m.min(), p.min()) * 0.8, max(m.max(), p.max()) * 1.25      # per-panel range
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    x = np.array([lo, hi]); a.plot(x, x, color=INK, lw=1.2, zorder=1)
    for f in (0.9, 1.1): a.plot(x, x * f, color=INK2, lw=0.9, ls='--', zorder=1)
    for f in (0.8, 1.2): a.plot(x, x * f, color=INK2, lw=0.9, ls=':', zorder=1)
    for k, mk in (('train', 'o'), ('test', 's')):
        mm = sp == k
        a.scatter(m[mm], p[mm], c=u[mm], cmap=cmap, norm=norm, marker=mk, s=60, edgecolors='white', lw=0.9, zorder=4)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel('measured |dP/dz|  [Pa/m]', color=INK2)
    a.set_title(f'{SUB[sub]}\nmean |error| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%  (n={len(e)}; within 20%: {(np.abs(e) <= 20).sum()}/{len(e)})',
                color=INK, loc='left', fontsize=10.5)
    l = a.legend(handles=[Line2D([], [], marker='o', ls='', color=INK2, markersize=7, label=f'train (n={int((sp == "train").sum())})'),
                          Line2D([], [], marker='s', ls='', color=INK2, markersize=7, label=f'test, held out (n={int((sp == "test").sum())})')],
                 loc='upper left', fontsize=8.5); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)
    et = e[sp == 'test']
    print(f"{sub}: mean|err| {np.abs(e).mean():.1f}% bias {e.mean():+.1f}% max {np.abs(e).max():.0f}% within10 {(np.abs(e) <= 10).sum()}/{len(e)} within20 {(np.abs(e) <= 20).sum()}/{len(e)} | test-only mean|err| {np.abs(et).mean():.1f}%")
for a_ in axs: a_.set_ylabel('predicted |dP/dz|  [Pa/m]', color=INK2)
fig.suptitle('Pooled SW + ST NN fit: pressure gradient   (dashed = +-10%, dotted = +-20%)', color=INK, x=0.01, ha='left', fontsize=11)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axs), fraction=0.025, pad=0.02)
cb.set_label('U_sg  [m/s]', color=INK2); cb.ax.tick_params(colors=INK2)
fig.savefig(f'{OUTDIR}/pooled_nn_dpdz_sw_st.png', dpi=200); plt.close(fig)
print('wrote', f'{OUTDIR}/pooled_nn_dpdz_sw_st.png', '| dP/dz range', allm.min(), allm.max())
