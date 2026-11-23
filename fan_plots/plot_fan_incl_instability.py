"""
Figures: Fan (2005) INCLINED rows -- run long enough for the instability to propagate, the data become usable.
Reads fan_incl_cache_shard*.pkl (from _fan_incl_instability_cache.py) and fan_incl_row0_trace.pkl
(from _fan_incl_row0_trace.py). No solver runs here.

  1. fan_incl_parity.png    predicted vs MEASURED holdup, 10 vs 150 transit times (34 inclined rows with HL)
  2. fan_incl_profile.png   alpha_l(x) of row 0 at increasing transit times
  3. fan_incl_trace.png     row 0: outlet alpha_l and complex-eigenvalue cells vs transit time
Sign convention (positive angle = uphill) is ASSUMED, not verified against the Fan paper.
Usage (repo root): python3 fan_plots/plot_fan_incl_instability.py
"""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2 = '#2a78d6', '#eb6834'
OUTDIR = 'figures/fan_incl_instability'; os.makedirs(OUTDIR, exist_ok=True)

cache = {}
for p in sorted(glob.glob('fan_incl_cache_shard*.pkl')):
    cache.update(pickle.load(open(p, 'rb')))
rows = [dict(idx=i, **r) for i, r in sorted(cache.items()) if all(t in r['runs'] for t in (10.0, 150.0))]
tr = pickle.load(open('fan_incl_row0_trace.pkl', 'rb'))

def style(a):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

def legend(a, **kw):
    l = a.legend(fontsize=8.5, **kw); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)

# ------------------------------------------------------------------ 1. parity
hl = np.array([r['HL'] for r in rows]); flag = np.array([r['kh_flagged'] for r in rows])
cf = np.array([r['cf_alpha_l'] for r in rows])
p10 = np.array([r['runs'][10.0]['alpha_l_pred'] for r in rows])
p150 = np.array([r['runs'][150.0]['alpha_l_pred'] for r in rows])
nan150 = np.array([r['runs'][150.0]['any_nan'] for r in rows])
print(f"{len(rows)} rows; flagged={flag.sum()}; any NaN at 150: {nan150.sum()}; "
      f"max |pred150-cf|/cf = {np.nanmax(np.abs(p150 - cf) / cf):.2e}; flagged max = "
      f"{np.nanmax((np.abs(p150 - cf) / cf)[flag]):.2e}")

# HL is quantized (0.01 / 0.02), so identical x values stack and hide points: spread them with a small
# DETERMINISTIC horizontal jitter (same x for a row in every series), and say so on the axis.
u = ((np.arange(len(hl)) * 0.6180339887) % 1.0) * 2 - 1
xj = hl * (1 + 0.16 * u)
lo = min(p10.min(), p150.min(), hl.min()) * 0.6; hi = max(hl.max(), p150.max(), cf.max()) * 1.6
fig, a = plt.subplots(figsize=(6.6, 5.4), facecolor='white'); style(a)
a.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls='--', zorder=1)
for mk, m in (('o', ~flag), ('D', flag)):
    a.scatter(xj[m], p10[m], s=50, marker=mk, color=S2, edgecolors='white', lw=0.8, zorder=3)
    a.scatter(xj[m], cf[m], s=95, marker=mk, facecolors='none', edgecolors=INK2, lw=1.2, zorder=4)
    a.scatter(xj[m], p150[m], s=50, marker=mk, color=S1, edgecolors='white', lw=0.8, zorder=5)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
a.set_xlabel('Measured liquid holdup  HL  (reported as 0.01 or 0.02; points jittered horizontally)', color=INK2); a.set_ylabel('Predicted liquid holdup  (outlet window)', color=INK2)
a.set_title(f'Inclined Fan rows (n={len(rows)} with measured HL): 10 vs. 150 transits\n'
            f'default closure, uncalibrated; diamonds = {flag.sum()} kh_total>0 rows',
            color=INK, loc='left', fontsize=10.5)
legend(a, loc='upper left', handles=[
    Line2D([], [], marker='o', ls='', color=S2, markersize=8, label='10 transit times'),
    Line2D([], [], marker='o', ls='', color=S1, markersize=8, label='150 transit times'),
    Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, mew=1.2, markersize=10, label='closed-form equilibrium'),
    Line2D([], [], marker='D', ls='', color=INK2, markersize=7, label='diamond: kh_total>0 row'),
    Line2D([], [], ls='--', color=INK2, label='1:1')])
fig.tight_layout(); fig.savefig(f'{OUTDIR}/fan_incl_parity.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 2. profile
cf0 = cache[0]['cf_alpha_l']; hl0 = cache[0]['HL']
ts = sorted(tr['snaps']); x = (np.arange(len(tr['snaps'][ts[0]])) + 0.5) / len(tr['snaps'][ts[0]])
fig, a = plt.subplots(figsize=(7.4, 4.8), facecolor='white'); style(a)
cols = cm.Blues(np.linspace(0.3, 1.0, len(ts)))
for c, t in zip(cols, ts):
    a.plot(x, tr['snaps'][t], color=c, lw=1.7, label=f'{t:.0f} transits')
a.axhline(cf0, color=INK2, lw=1.4, ls='--', label='closed-form equilibrium')
a.axhline(hl0, color=INK, lw=2.6, zorder=6, label='measured HL (experiment)')
a.set_xlabel('Position along pipe,  x / L', color=INK2); a.set_ylabel('Liquid fraction  alpha_l', color=INK2)
a.set_xlim(0, 1)
a.set_title(f'Fan row 0 (U_sl={cache[0]["U_sl"]:.4f}, U_sg={cache[0]["U_sg"]:.2f} m/s, {cache[0]["angle"]:+.0f} deg): '
            'profile fills in downstream\nand settles onto the closed-form value', color=INK, loc='left', fontsize=10.5)
legend(a, loc='center right', bbox_to_anchor=(0.99, 0.33), ncol=2)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/fan_incl_profile.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 3. trace
T = np.array(tr['transit'])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, facecolor='white')
for a in (a1, a2): style(a)
a1.plot(T, tr['outlet'], color=S1, lw=2, label='outlet-window alpha_l')
a1.axhline(cf0, color=INK2, lw=1.4, ls='--', label='closed-form equilibrium')
a1.axhline(hl0, color=INK, lw=2.4, label='measured HL (experiment)')
a1.set_ylabel('outlet alpha_l', color=INK2); legend(a1, loc='center right')
a2.plot(T, tr['n_complex'], color=S1, lw=2)
a2.set_ylabel('cells with complex\neigenvalues (of 150)', color=INK2); a2.set_ylim(0, 150)
a2.set_xlabel('Nominal transit times', color=INK2)
a1.set_title('Fan row 0: complex-eigenvalue cells persist, yet the outlet holdup converges\n'
             '(statistical equilibrium: instability present, mean value usable)', color=INK, loc='left', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/fan_incl_trace.png', dpi=200); plt.close(fig)
print('wrote', sorted(os.listdir(OUTDIR)))
