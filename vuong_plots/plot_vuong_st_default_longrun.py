"""
Figures: Vuong ST (inclined +2 deg) rows with the DEFAULT closure (f_i_mult = k_wl = k_wg = 1.0) --
does a longer run help, the way it did for Fan's inclined rows?
Reads vuong_st_default_longrun_shard*.pkl (from _vuong_st_default_longrun_cache.py; 0.25x CFL dt, 50 nominal
transits, recorded every transit). No solver runs here.

  1. vuong_st_default_parity.png        predicted vs MEASURED holdup at 10 vs 50 transits (+ closed-form)
  2. vuong_st_default_profile.png       alpha_l(x) of idx 797 at 10 / 25 / 50 transits
  3. vuong_st_default_trace.png         idx 797: outlet + pipe-mean alpha_l and complex-eigenvalue cells vs transit
  4. vuong_st_default_error_vs_time.png mean |error vs measured| over the rows, vs transit time
"Predicted" = outlet-window median alpha_l (what run_real_data_condition reports as alpha_l_pred) unless stated.
Usage (repo root): python3 vuong_plots/plot_vuong_st_default_longrun.py
"""
import glob, os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2 = '#2a78d6', '#eb6834'
OUTDIR = 'figures/vuong_st_default_longrun'; os.makedirs(OUTDIR, exist_ok=True)
IDX = 797

cache = {}
for p in sorted(glob.glob('vuong_st_default_longrun_shard*.pkl')):
    cache.update(pickle.load(open(p, 'rb')))
rows = [dict(idx=i, **r) for i, r in sorted(cache.items())]
nan_rows = [r['idx'] for r in rows if 'nan_at' in r]
print(f"{len(rows)} rows loaded; rows that went NaN: {nan_rows}")

def style(a):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

def legend(a, **kw):
    l = a.legend(fontsize=8.5, **kw); l.get_frame().set_edgecolor(GRID)
    for t in l.get_texts(): t.set_color(INK2)

def at(r, key, k):                     # value at transit k (None if the run didn't get that far)
    return r[key][k] if len(r[key]) > k else np.nan

# ------------------------------------------------------------------ 1. parity
meas = [r for r in rows if r['HL'] == r['HL']]
hl = np.array([r['HL'] for r in meas]); cf = np.array([r['cf_alpha_l'] for r in meas])
p10 = np.array([at(r, 'outlet', 10) for r in meas]); p50 = np.array([at(r, 'outlet', 50) for r in meas])
u = ((np.arange(len(hl)) * 0.6180339887) % 1.0) * 2 - 1; xj = hl * (1 + 0.05 * u)   # jitter: HL is coarsely reported
lo, hi = 0.005, 1.0
fig, a = plt.subplots(figsize=(6.6, 5.4), facecolor='white'); style(a)
a.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls='--', zorder=1)
a.scatter(xj, cf, s=95, facecolors='none', edgecolors=INK2, lw=1.2, zorder=4)
a.scatter(xj, p10, s=50, color=S2, edgecolors='white', lw=0.8, zorder=3)
a.scatter(xj, p50, s=50, color=S1, edgecolors='white', lw=0.8, zorder=5)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, 0.2); a.set_ylim(0.001, 3.0)   # headroom so the legend sits in empty space
a.set_xlabel('Measured liquid holdup  HL  (points jittered horizontally)', color=INK2)
a.set_ylabel('Predicted liquid holdup  (outlet window)', color=INK2)
a.set_title(f'Vuong ST rows, default closure (all multipliers 1.0), n={len(meas)}:\n'
            'longer runs do not reach the measurements or the equilibrium', color=INK, loc='left', fontsize=10.5)
legend(a, loc='upper left', handles=[
    Line2D([], [], marker='o', ls='', color=S2, markersize=8, label='10 transit times'),
    Line2D([], [], marker='o', ls='', color=S1, markersize=8, label='50 transit times'),
    Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, mew=1.2, markersize=10, label='closed-form equilibrium'),
    Line2D([], [], ls='--', color=INK2, label='1:1')])
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_default_parity.png', dpi=200); plt.close(fig)

# ---- 1b. same parity using the PIPE-MEAN alpha_l. The outlet window is a ~10-cell slice of a wavy field and
# fluctuates a lot (see the trace figure), so it can land near the data by chance; the pipe mean is the steadier
# holdup measure and shows the systematic over-prediction.
m10 = np.array([at(r, 'mean', 10) for r in meas]); m50 = np.array([at(r, 'mean', 50) for r in meas])
fig, a = plt.subplots(figsize=(6.6, 5.4), facecolor='white'); style(a)
a.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls='--', zorder=1)
a.scatter(xj, cf, s=95, facecolors='none', edgecolors=INK2, lw=1.2, zorder=4)
a.scatter(xj, m10, s=50, color=S2, edgecolors='white', lw=0.8, zorder=3)
a.scatter(xj, m50, s=50, color=S1, edgecolors='white', lw=0.8, zorder=5)
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(lo, 0.2); a.set_ylim(0.001, 3.0)
a.set_xlabel('Measured liquid holdup  HL  (points jittered horizontally)', color=INK2)
a.set_ylabel('Predicted liquid holdup  (pipe mean)', color=INK2)
a.set_title(f'Vuong ST rows, default closure, n={len(meas)}: pipe-mean holdup\n'
            f'over-predicted in {(m50 > hl).sum()} of {len(meas)} rows at 50 transits', color=INK, loc='left', fontsize=10.5)
legend(a, loc='upper left', handles=[
    Line2D([], [], marker='o', ls='', color=S2, markersize=8, label='10 transit times'),
    Line2D([], [], marker='o', ls='', color=S1, markersize=8, label='50 transit times'),
    Line2D([], [], marker='o', ls='', mfc='none', mec=INK2, mew=1.2, markersize=10, label='closed-form equilibrium'),
    Line2D([], [], ls='--', color=INK2, label='1:1')])
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_default_parity_pipemean.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 2. profile (idx 797)
r0 = next(r for r in rows if r['idx'] == IDX)
ts = sorted(r0['snaps']); x = (np.arange(len(r0['snaps'][ts[0]])) + 0.5) / len(r0['snaps'][ts[0]])
fig, a = plt.subplots(figsize=(7.4, 4.8), facecolor='white'); style(a)
for c, t in zip(cm.Blues(np.linspace(0.4, 1.0, len(ts))), ts):
    a.plot(x, r0['snaps'][t], color=c, lw=1.5, label=f'{t} transits')
a.axhline(r0['cf_alpha_l'], color=INK2, lw=1.4, ls='--', label='closed-form equilibrium')
a.axhline(r0['HL'], color=INK, lw=2.6, zorder=6, label='measured HL (experiment)')
a.set_xlabel('Position along pipe,  x / L', color=INK2); a.set_ylabel('Liquid fraction  alpha_l', color=INK2)
a.set_xlim(0, 1); a.set_ylim(0, 1)
a.set_title(f'idx {IDX} (U_sl={r0["U_sl"]:.3f}, U_sg={r0["U_sg"]:.2f} m/s, +2 deg), default closure:\n'
            'the profile stays wavy; it does not settle onto either reference line', color=INK, loc='left', fontsize=10.5)
legend(a, loc='upper left', ncol=2)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_default_profile.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 3. trace (idx 797)
T = np.array(r0['transit'])
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True, facecolor='white')
for a in (a1, a2): style(a)
a1.plot(T, r0['outlet'], color=S1, lw=1.8, label='outlet-window alpha_l')
a1.plot(T, r0['mean'], color=S2, lw=1.8, label='pipe-mean alpha_l')
a1.axhline(r0['cf_alpha_l'], color=INK2, lw=1.4, ls='--', label='closed-form equilibrium')
a1.axhline(r0['HL'], color=INK, lw=2.4, label='measured HL (experiment)')
a1.set_ylabel('alpha_l', color=INK2); a1.set_ylim(0, 0.6); legend(a1, loc='upper left', ncol=2)
a2.plot(T, r0['n_complex'], color=S1, lw=2)
a2.set_ylabel('cells with complex\neigenvalues (of 150)', color=INK2); a2.set_ylim(0, 150)
a2.set_xlabel('Nominal transit times  (0.25x CFL dt)', color=INK2)
a1.set_title(f'idx {IDX}, default closure: holdup keeps fluctuating and complex cells persist\n'
             '(compare Fan row 0, which converged)', color=INK, loc='left', fontsize=10.5)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_default_trace.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 4. error vs time over all rows with HL
kmax = min(len(r['outlet']) for r in meas) - 1
ks = np.arange(1, kmax + 1)
def err(key): return np.array([[abs(r[key][k] - r['HL']) / r['HL'] * 100 for k in ks] for r in meas])
e_out, e_mean = err('outlet'), err('mean')
cf_err = np.mean(np.abs(cf - hl) / hl * 100)
fig, a = plt.subplots(figsize=(7.0, 4.8), facecolor='white'); style(a)
a.plot(ks, e_out.mean(axis=0), color=S1, lw=2, label='outlet-window alpha_l')
a.plot(ks, e_mean.mean(axis=0), color=S2, lw=2, label='pipe-mean alpha_l')
a.axhline(cf_err, color=INK2, lw=1.4, ls='--', label=f'closed-form equilibrium ({cf_err:.0f}%)')
a.set_yscale('log'); a.set_xlabel('Nominal transit times  (0.25x CFL dt)', color=INK2)
a.set_ylabel(f'mean |error| vs measured holdup, % (n={len(meas)} rows)', color=INK2)
a.set_title('Default closure on the Vuong ST rows: error vs. run time\n'
            'outlet-window error fluctuates; pipe-mean error grows. Neither trends toward zero', color=INK, loc='left', fontsize=10.5)
legend(a, loc='best')
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_default_error_vs_time.png', dpi=200); plt.close(fig)
print(f"mean|err|% at transit 10/25/50: outlet {e_out.mean(0)[9]:.0f}/{e_out.mean(0)[24]:.0f}/{e_out.mean(0)[min(49,len(ks)-1)]:.0f}; "
      f"pipe-mean {e_mean.mean(0)[9]:.0f}/{e_mean.mean(0)[24]:.0f}/{e_mean.mean(0)[min(49,len(ks)-1)]:.0f}; closed-form {cf_err:.0f}")
print('wrote', sorted(os.listdir(OUTDIR)))
