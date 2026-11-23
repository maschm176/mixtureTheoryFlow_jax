"""
Figures: the KH-type instability in the inclined (beta=+2deg) Vuong (2016) "ST" data,
the 23 genuinely-high-holdup rows. Reads vuong_st_instability_cache.pkl (written by
_vuong_st_instability_cache.py) -- no solver runs here.

  1. vuong_st_parity.png    pipe-mean predicted holdup vs. MEASURED holdup (HL), with the
                            closed-form equilibrium and the NaN rows shown explicitly.
  2. vuong_st_profile.png   alpha_l(x) along the pipe for idx=797: wavy, not flat.
  3. vuong_st_hyperbolicity.png  cells with complex eigenvalues + alpha_max vs. step (idx=797).

Horizon caveat (belongs in any caption): the Part A runs are only N_TRANSIT=10 transit
times, so they are NOT settled equilibria -- "after" figures should reuse this layout.
Usage (from repo root): python3 vuong_plots/plot_vuong_st_instability.py
"""
import os, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2 = '#2a78d6', '#eb6834'
RAMP = ['#9fc3ee', '#2a78d6', '#0d2f5e']          # light -> dark blue = early -> late in time
OUTDIR = 'figures/st_instability'
os.makedirs(OUTDIR, exist_ok=True)

c = pickle.load(open('vuong_st_instability_cache.pkl', 'rb'))
rows = [dict(idx=i, **r) for i, r in sorted(c['rows'].items())]
tr = c['trace']
IDX = 797

def new_ax(figsize=(6.4, 5.2)):
    fig, a = plt.subplots(figsize=figsize, facecolor='white')
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    return fig, a

def style_legend(leg):
    leg.get_frame().set_edgecolor(GRID)
    for t in leg.get_texts(): t.set_color(INK2)

# ------------------------------------------------------------------ 1. parity
hl   = np.array([r['HL_measured'] for r in rows])
cf   = np.array([r['cf_alpha_l'] for r in rows])
run1 = [r['runs'][1.0] for r in rows]
run4 = [r['runs'][0.25] for r in rows]
nan1 = np.array([x['any_nan'] for x in run1])
mean4 = np.array([x['alpha_l'].mean() for x in run4])
have = ~np.isnan(hl)                                   # 2 rows have no measured HL

fig, a = new_ax()
lo, hi, strip = 0.006, 0.9, 0.0085                     # 'strip' = where NaN rows are drawn
a.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls='--', zorder=1)
a.scatter(hl[have], cf[have], s=55, facecolors='none', edgecolors=S2, lw=1.8, zorder=3)
a.scatter(hl[have], mean4[have], s=55, color=S1, edgecolors='white', lw=1.0, zorder=4)
m = have & nan1
a.scatter(hl[m], np.full(m.sum(), strip), s=60, marker='x', color=INK, lw=1.8, zorder=5)
a.axhspan(lo, 0.0105, color=GRID, alpha=0.55, zorder=0)
a.text(0.0615, 0.0070, 'default-dt run went NaN (no prediction)', fontsize=8.5, color=INK2, ha='right')
# y-axis runs above the data (to 4) so the legend sits in empty space instead of over the markers
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(0.006, 0.12); a.set_ylim(lo, 4.0)
a.set_xlabel('Measured liquid holdup  HL', color=INK2)
a.set_ylabel('Predicted liquid holdup  (pipe mean)', color=INK2)
a.set_title('Inclined Vuong ST rows: holdup is not reproduced\n'
            f'(n={have.sum()} of {len(rows)} rows have measured HL; 10 transit times, not settled)',
            color=INK, loc='left', fontsize=11)
style_legend(a.legend(handles=[
    Line2D([], [], marker='o', ls='', color=S1, markersize=8, label='dynamic solver, 0.25x CFL dt'),
    Line2D([], [], marker='o', ls='', mfc='none', mec=S2, mew=1.8, markersize=8,
           label='closed-form equilibrium (default closure)'),
    Line2D([], [], marker='x', ls='', color=INK, mew=1.8, markersize=8, label='default dt: NaN'),
    Line2D([], [], ls='--', color=INK2, label='1:1')], loc='upper left', fontsize=8.5))
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_parity.png', dpi=200); plt.close(fig)

# ---- 1b. same parity, but ONLY the default-CFL-dt runs (no 0.25x dt results).
# Rows that stayed stable at the default dt are plotted at their default-dt value;
# rows that went NaN get the x marker in the strip, as above.
mean1 = np.array([np.nan if x['any_nan'] else x['alpha_l'].mean() for x in run1])
stable = have & ~nan1
fig, a = new_ax()
a.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, ls='--', zorder=1)
a.scatter(hl[have], cf[have], s=55, facecolors='none', edgecolors=S2, lw=1.8, zorder=3)
a.scatter(hl[stable], mean1[stable], s=55, color=S1, edgecolors='white', lw=1.0, zorder=4)
a.scatter(hl[m], np.full(m.sum(), strip), s=60, marker='x', color=INK, lw=1.8, zorder=5)
a.axhspan(lo, 0.0105, color=GRID, alpha=0.55, zorder=0)
a.text(0.0615, 0.0070, 'default-dt run went NaN (no prediction)', fontsize=8.5, color=INK2, ha='right')
a.set_xscale('log'); a.set_yscale('log'); a.set_xlim(0.006, 0.12); a.set_ylim(lo, 4.0)
a.set_xlabel('Measured liquid holdup  HL', color=INK2)
a.set_ylabel('Predicted liquid holdup  (pipe mean)', color=INK2)
a.set_title('Inclined Vuong ST rows, default CFL dt only\n'
            f'({stable.sum()} stable, {m.sum()} NaN, of {have.sum()} rows with measured HL; 10 transit times)',
            color=INK, loc='left', fontsize=11)
style_legend(a.legend(handles=[
    Line2D([], [], marker='o', ls='', color=S1, markersize=8, label='dynamic solver, default CFL dt'),
    Line2D([], [], marker='o', ls='', mfc='none', mec=S2, mew=1.8, markersize=8,
           label='closed-form equilibrium (default closure)'),
    Line2D([], [], marker='x', ls='', color=INK, mew=1.8, markersize=8, label='default dt: NaN'),
    Line2D([], [], ls='--', color=INK2, label='1:1')], loc='upper left', fontsize=8.5))
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_parity_default_dt.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 2. profile
r797 = next(r for r in rows if r['idx'] == IDX)
x = (np.arange(len(r797['runs'][0.25]['alpha_l'])) + 0.5) / len(r797['runs'][0.25]['alpha_l'])
snap_steps = [400, 1400, 2800]                         # default dt, before it ever goes NaN
fig, a = new_ax(figsize=(7.4, 4.8))
for col, s in zip(RAMP, snap_steps):
    a.plot(x, tr['snaps'][s], color=col, lw=1.6, label=f'default dt, step {s}')
a.plot(x, r797['runs'][0.25]['alpha_l'], color=S2, lw=1.6, label='0.25x dt, 10 transit times')
a.axhline(r797['cf_alpha_l'], color=INK2, lw=1.4, ls='--', label='closed-form equilibrium')
# Measured HL is a single pipe-average number per condition (no spatial data), so it is a
# horizontal line. Drawn thick and solid in near-black so it reads as "the data".
a.axhline(r797['HL_measured'], color=INK, lw=2.6, ls='-', zorder=6, label='measured HL (experiment)')
a.text(0.995, r797['cf_alpha_l'] + 0.012, f'closed-form {r797["cf_alpha_l"]:.3f}',
       ha='right', va='bottom', fontsize=9, color=INK2)
a.text(0.995, r797['HL_measured'] + 0.012, f'measured {r797["HL_measured"]:.3f}',
       ha='right', va='bottom', fontsize=9, color=INK, fontweight='bold', zorder=7)
a.set_xlabel('Position along pipe,  x / L', color=INK2)
a.set_ylabel('Liquid fraction  alpha_l', color=INK2)
a.set_xlim(0, 1); a.set_ylim(0, 1)
a.set_title(f'idx {IDX}  (U_sl={r797["U_sl"]:.3f}, U_sg={r797["U_sg"]:.2f} m/s, +2 deg): '
            'profile is wavy, not uniform', color=INK, loc='left', fontsize=11)
style_legend(a.legend(loc='upper left', fontsize=8.5, ncol=2))
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_profile.png', dpi=200); plt.close(fig)

# ------------------------------------------------------------------ 3. hyperbolicity trace
step = np.array(tr['step']); nc = np.array(tr['n_complex']); amax = np.array(tr['a_max'])
ntr = step[-1] * tr['dt_star'] / tr['L_star']          # transit times covered by the trace
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.0, 5.6), sharex=True, facecolor='white')
for a in (a1, a2):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
a1.step(step, nc, where='post', color=S1, lw=2)
a1.set_ylabel('cells with complex\neigenvalues (of 150)', color=INK2); a1.set_ylim(0, 150)
first = step[np.argmax(nc > 0)]
a1.axvline(first, color=INK2, lw=1, ls=':'); a1.text(first + 30, 100, f'first complex cell: step {first}',
                                                    fontsize=8.5, color=INK2)
a2.plot(step, amax, color=S1, lw=2)
a2.set_ylabel('alpha_max  (max liquid fraction)', color=INK2); a2.set_ylim(0, 1)
a2.set_xlabel(f'Step at default CFL dt  (end of trace = {ntr:.1f} transit times)', color=INK2)
a1.set_title(f'idx {IDX}: hyperbolicity is lost locally and stays lost\n(default dt, no NaN through step {step[-1]})',
             color=INK, loc='left', fontsize=11)
fig.tight_layout(); fig.savefig(f'{OUTDIR}/vuong_st_hyperbolicity.png', dpi=200); plt.close(fig)
print('wrote', sorted(os.listdir(OUTDIR)))
