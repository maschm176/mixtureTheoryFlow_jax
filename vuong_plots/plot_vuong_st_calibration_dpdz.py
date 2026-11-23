"""
Pressure-gradient parity for the two beta-aware ST calibrations on the 23 inclined (+2 deg) Vuong ST rows, in the same
style as the Fan NN dP/dz figure (log-log, dashed +-10% / dotted +-20% bands, colour = U_sg, squares = held-out test).
Predicted |dP/dz| is the DYNAMIC production solver's value at 150 transits (st_closure_stability_shard*.pkl, from
_st_closure_stability_check.py); measured = FFPU dP/dL[Pa/m]. The 2 rows without a measured holdup (idx 1480, 1530) were in
neither train nor test of the fits; shown as triangles. Closures: flat 3-scalar ST fit, pooled SW+ST NN fit (2-input variant).
Output: figures/st_calibration/st_cal_dpdz.png.  Usage (repo root): python3 vuong_plots/plot_vuong_st_calibration_dpdz.py
"""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
OUTDIR = 'figures/st_calibration'; os.makedirs(OUTDIR, exist_ok=True)
NAMES = {'flat_ST': 'flat 3-scalar ST fit', 'pooled_NN': 'pooled SW+ST NN fit'}

cache = {}
for p in sorted(glob.glob('st_closure_stability_shard*.pkl')):
    cache.update(pickle.load(open(p, 'rb')))
rows = [dict(idx=i, **r) for i, r in sorted(cache.items())]
meas = np.array([r['dpdz'] for r in rows]); usg = np.array([r['U_sg'] for r in rows]); split = np.array([r['split'] for r in rows])
MK = {'train': 'o', 'test': 's', 'no HL': '^'}
LAB = {'train': 'train', 'test': 'test (held out)', 'no HL': 'not in fit (no measured holdup)'}

fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.3), facecolor='white', sharex=True, sharey=True, layout='constrained')
lo, hi = 45, 125          # ST dP/dz is only ~57-95 Pa/m (the Fan axis range would squash it)
norm = plt.Normalize(usg.min(), usg.max())
cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues_trunc', plt.cm.Blues(np.linspace(0.3, 1.0, 256)))  # drop the near-white end so low-U_sg points stay visible
for a, c in zip(axs, NAMES):
    pred = np.array([r['runs'][c]['dpdz_pred'] for r in rows]); e = (pred - meas) / meas * 100
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    x = np.array([lo, hi]); a.plot(x, x, color=INK, lw=1.2, zorder=1)
    for f in (0.9, 1.1): a.plot(x, x * f, color=INK2, lw=0.9, ls='--', zorder=1)
    for f in (0.8, 1.2): a.plot(x, x * f, color=INK2, lw=0.9, ls=':', zorder=1)
    for sp in ('train', 'test', 'no HL'):
        m = split == sp
        if m.any():
            sc = a.scatter(meas[m], pred[m], c=usg[m], cmap=cmap, norm=norm, marker=MK[sp], s=60, edgecolors='white', lw=0.9, zorder=4)
    a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel('measured |dP/dz|  [Pa/m]', color=INK2)
    a.set_title(f'{NAMES[c]}\nmean |error| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%   (n={len(e)}; max |error| {np.abs(e).max():.0f}%)',
                color=INK, loc='left', fontsize=10.5)
    l = a.legend(handles=[Line2D([], [], marker=MK[k], ls='', color=INK2, markersize=7, label=f'{LAB[k]} (n={int((split == k).sum())})') for k in MK],
                 loc='upper left', fontsize=8.5); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)
    print(f"{c}: mean|err| {np.abs(e).mean():.1f}%  bias {e.mean():+.1f}%  max {np.abs(e).max():.0f}%  within 10%: {(np.abs(e) <= 10).sum()}/{len(e)}  within 20%: {(np.abs(e) <= 20).sum()}/{len(e)}")
axs[0].set_ylabel('predicted |dP/dz|  [Pa/m]  (dynamic solver, 150 transits)', color=INK2)
fig.suptitle('Pressure gradient, inclined (+2 deg) Vuong ST rows   (dashed = +-10%, dotted = +-20%)', color=INK, x=0.01, ha='left', fontsize=11)
cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=list(axs), fraction=0.025, pad=0.02)
cb.set_label('U_sg  [m/s]', color=INK2); cb.ax.tick_params(colors=INK2)
fig.savefig(f'{OUTDIR}/st_cal_dpdz.png', dpi=200); plt.close(fig)
print('wrote', f'{OUTDIR}/st_cal_dpdz.png', '| measured dP/dz range', meas.min(), meas.max())
