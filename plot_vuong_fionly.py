"""
Figure: Vuong (2016) fit quality when ONLY f_i_mult is calibrated
(k_wl = k_wg = 1, wall friction uncalibrated), vs. the cached scan in fionly_cache/.
Best f_i_mult = lowest TRAIN-split mean holdup squared relative error (holdup-only loss,
as in the original single-parameter fit). kh_total>0 conditions are excluded.
Usage: python3 plot_vuong_fionly.py
"""
import glob, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2 = '#2a78d6', '#eb6834'
seq = LinearSegmentedColormap.from_list('seq', ['#cfe1f7', '#2a78d6', '#0d2f5e'])

def _log_tick_fmt(val, pos=None):
    """6 -> '6' (exponent 0, no clutter); 60 -> '6E1' (compact, no x10^n)."""
    if val <= 0:
        return ''
    exp = int(np.floor(np.log10(val) + 1e-9))
    m = int(round(val / 10.0**exp))
    if m >= 10:
        m, exp = 1, exp + 1
    return f"{m}" if exp == 0 else f"{m}E{exp}"

def apply_log_ticks(ax):
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 6.0)))
        axis.set_major_formatter(FuncFormatter(_log_tick_fmt))
        axis.set_minor_formatter(NullFormatter())

data = {}
for p in sorted(glob.glob('fionly_cache/f_*.pkl')):
    data[float(p.split('f_')[1][:-4])] = pickle.load(open(p, 'rb'))
fs = sorted(data)

def arr(rows, k): return np.array([r[k] for r in rows], float)
def ok(rows): return np.array([r['kh_total'] == 0 and not r['any_nan'] for r in rows])
def herr(rows): return (arr(rows,'psi_pred')-arr(rows,'psi_meas'))/arr(rows,'psi_meas')*100
def derr(rows): return (arr(rows,'dpdz_pred')-arr(rows,'dpdz_meas'))/arr(rows,'dpdz_meas')*100

def tr_loss(f):
    r = [x for x in data[f] if x['split']=='train']; m = ok(r)
    return np.mean(herr(r)[m]**2)
best = min(fs, key=tr_loss)
rows = data[best]; m = ok(rows)
split = np.array([r['split'] for r in rows])

def new_ax(figsize=(6.2, 5.6)):
    fig, a = plt.subplots(figsize=figsize, facecolor='white')
    fig.subplots_adjust(top=0.78)
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top','right'): a.spines[s].set_visible(False)
    for s in ('left','bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    return fig, a

usg = arr(rows,'U_sg'); norm = plt.Normalize(usg.min(), usg.max())
def parity(a, meas, pred, name):
    lo, hi = min(meas.min(), pred.min())*0.8, max(meas.max(), pred.max())*1.2
    xs = np.array([lo, hi])
    a.plot(xs, xs, color=INK, lw=1.2)
    for pct, ls in ((10, '--'), (20, ':')):
        a.plot(xs, xs*(1+pct/100), color=INK2, lw=1, ls=ls)
        a.plot(xs, xs*(1-pct/100), color=INK2, lw=1, ls=ls)
    for sp, mk in (('train','o'), ('test','s')):
        k = (split==sp) & m
        sc = a.scatter(meas[k], pred[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=46,
                       edgecolor='white', linewidth=1.2, label=f'{sp} (n={k.sum()})', zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
    a.set_xlabel(f'measured {name}', color=INK); a.set_ylabel(f'predicted {name}', color=INK)
    return sc

trm = (split=='train') & m; tem = (split=='test') & m
title = (f'Vuong (2016), f_i_mult-only calibration (f_i_mult={best:g}, wall friction not calibrated)\n'
         f'train holdup {np.mean(np.abs(herr(rows)[trm])):.1f}% / dP/dz {np.mean(np.abs(derr(rows)[trm])):.1f}%   |   '
         f'test holdup {np.mean(np.abs(herr(rows)[tem])):.1f}% / dP/dz {np.mean(np.abs(derr(rows)[tem])):.1f}%   '
         f'(mean |error|; {(~m).sum()} kh-excluded)')

# --- figure 1: holdup parity ---
fig1, a = new_ax()
sc = parity(a, arr(rows,'psi_meas'), arr(rows,'psi_pred'), 'liquid holdup')
a.set_title('Holdup (dashed ±10%, dotted ±20%)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, loc='upper left')
cb = fig1.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig1.suptitle(title, color=INK, fontsize=10.5, x=0.02, ha='left', wrap=True)
fig1.savefig('figures/f_i_mult_only/vuong_fi_mult_only_holdup.png', dpi=160, bbox_inches='tight')

# --- figure 2: pressure gradient parity ---
fig2, a = new_ax()
sc = parity(a, arr(rows,'dpdz_meas'), arr(rows,'dpdz_pred'), '|dP/dz| [Pa/m]')
a.set_title('Pressure gradient', color=INK, loc='left', fontsize=11)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.legend(frameon=False, loc='upper left')
cb = fig2.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig2.suptitle(title, color=INK, fontsize=10.5, x=0.02, ha='left', wrap=True)
fig2.savefig('figures/f_i_mult_only/vuong_fi_mult_only_dpdz.png', dpi=160, bbox_inches='tight')

# --- figure 3: error vs f_i_mult (mean |err|, train+test pooled, kh-clean) ---
fig3, a = new_ax()
mh = [np.mean(np.abs(herr(data[f])[ok(data[f])])) for f in fs]
md = [np.mean(np.abs(derr(data[f])[ok(data[f])])) for f in fs]
a.plot(fs, mh, '-o', color=S1, lw=2, ms=6, label='holdup')
a.plot(fs, md, '-o', color=S2, lw=2, ms=6, label='dP/dz')
a.axvline(best, color=INK2, lw=1, ls='--')
a.text(best, a.get_ylim()[1], f' best (holdup fit)\n f_i_mult={best:g}', color=INK2, va='top', fontsize=9)
a.set_xlabel('f_i_mult  (k_wl = k_wg = 1)', color=INK); a.set_ylabel('mean |error| [%]', color=INK)
a.set_title('Error vs. f_i_mult', color=INK, loc='left', fontsize=11)
a.legend(frameon=False)
fig3.suptitle(title, color=INK, fontsize=10.5, x=0.02, ha='left', wrap=True)
fig3.savefig('figures/f_i_mult_only/vuong_fi_mult_only_vs_param.png', dpi=160, bbox_inches='tight')

# --- figure 4: holdup error vs U_sg at best ---
fig4, a = new_ax()
a.axhline(0, color=INK, lw=1)
for sp, mk in (('train','o'), ('test','s')):
    k = (split==sp) & m
    a.scatter(usg[k], herr(rows)[k], marker=mk, s=40, color=S1, edgecolor='white', linewidth=1.2, label=sp, zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate', color=INK, loc='left', fontsize=11); a.legend(frameon=False)
fig4.suptitle(title, color=INK, fontsize=10.5, x=0.02, ha='left', wrap=True)
fig4.savefig('figures/f_i_mult_only/vuong_fi_mult_only_residual.png', dpi=160, bbox_inches='tight')

print('best', best, '-- saved: f_i_mult_only/vuong_fi_mult_only_{holdup,dpdz,vs_param,residual}.png')
