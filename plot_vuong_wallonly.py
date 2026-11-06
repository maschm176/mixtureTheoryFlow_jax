"""
Figure: Vuong (2016) fit quality when ONLY k_wall is calibrated (f_i_mult=1,
interfacial friction uncalibrated, k_wall applied identically to both phases
-- the single shared wall-friction scalar), vs. the cached scan in
wallonly_cache/. Mirrors plot_vuong_fionly.py exactly, swapping the scanned
variable, for direct visual comparison between the two single-knob fits.
Best k_wall = lowest TRAIN-split mean holdup squared relative error
(holdup-only loss, as in the original single-parameter fit). kh_total>0
conditions are excluded.
Usage: python3 plot_vuong_wallonly.py
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
for p in sorted(glob.glob('wallonly_cache/k_*.pkl')):
    data[float(p.split('k_')[1][:-4])] = pickle.load(open(p, 'rb'))
ks = sorted(data)

def arr(rows, k): return np.array([r[k] for r in rows], float)
def ok(rows): return np.array([r['kh_total'] == 0 and not r['any_nan'] for r in rows])
def herr(rows): return (arr(rows,'psi_pred')-arr(rows,'psi_meas'))/arr(rows,'psi_meas')*100
def derr(rows): return (arr(rows,'dpdz_pred')-arr(rows,'dpdz_meas'))/arr(rows,'dpdz_meas')*100

def tr_loss(k):
    r = [x for x in data[k] if x['split']=='train']; m = ok(r)
    return np.mean(herr(r)[m]**2)
best = min(ks, key=tr_loss)
rows = data[best]; m = ok(rows)
split = np.array([r['split'] for r in rows])

fig, ax = plt.subplots(2, 2, figsize=(11, 9), facecolor='white')
for a in ax.flat:
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top','right'): a.spines[s].set_visible(False)
    for s in ('left','bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

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

sc = parity(ax[0,0], arr(rows,'psi_meas'), arr(rows,'psi_pred'), 'liquid holdup')
ax[0,0].set_title('Holdup (dashed ±10%, dotted ±20%)', color=INK, loc='left', fontsize=11)
ax[0,0].legend(frameon=False, loc='upper left')
parity(ax[0,1], arr(rows,'dpdz_meas'), arr(rows,'dpdz_pred'), '|dP/dz| [Pa/m]')
ax[0,1].set_title('Pressure gradient', color=INK, loc='left', fontsize=11)
ax[0,1].set_xscale('log'); ax[0,1].set_yscale('log')
apply_log_ticks(ax[0,1])
cb = fig.colorbar(sc, ax=ax[0,:], shrink=0.7, pad=0.02); cb.set_label('U_sg [m/s]', color=INK2)

# error vs k_wall (mean |err|, train+test pooled, kh-clean)
a = ax[1,0]
mh = [np.mean(np.abs(herr(data[k])[ok(data[k])])) for k in ks]
md = [np.mean(np.abs(derr(data[k])[ok(data[k])])) for k in ks]
a.plot(ks, mh, '-o', color=S1, lw=2, ms=6); a.plot(ks, md, '-o', color=S2, lw=2, ms=6)
a.text(ks[-1], mh[-1], '  holdup', color=INK, va='center'); a.text(ks[-1], md[-1], '  dP/dz', color=INK, va='center')
a.axvline(best, color=INK2, lw=1, ls='--'); a.text(best, a.get_ylim()[1], f' best (holdup fit)\n k_wall={best:g}', color=INK2, va='top', fontsize=9)
a.set_xlabel('k_wall  (f_i_mult = 1)', color=INK); a.set_ylabel('mean |error| [%]', color=INK)
a.set_title('Error vs. k_wall', color=INK, loc='left', fontsize=11)

# holdup error vs U_sg at best
a = ax[1,1]; a.axhline(0, color=INK, lw=1)
for sp, mk in (('train','o'), ('test','s')):
    k = (split==sp) & m
    a.scatter(usg[k], herr(rows)[k], marker=mk, s=40, color=S1, edgecolor='white', linewidth=1.2, label=sp, zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate', color=INK, loc='left', fontsize=11); a.legend(frameon=False)

trm = (split=='train') & m; tem = (split=='test') & m
fig.suptitle(f'Vuong (2016), k_wall-only calibration (k_wall={best:g}, interfacial friction not calibrated)\n'
             f'train holdup {np.mean(np.abs(herr(rows)[trm])):.1f}% / dP/dz {np.mean(np.abs(derr(rows)[trm])):.1f}%   |   '
             f'test holdup {np.mean(np.abs(herr(rows)[tem])):.1f}% / dP/dz {np.mean(np.abs(derr(rows)[tem])):.1f}%   '
             f'(mean |error|; {(~m).sum()} kh-excluded)', color=INK, fontsize=12, x=0.02, ha='left')
fig.savefig('figures/k_wall_only/vuong_k_wall_only_fit.png', dpi=160, bbox_inches='tight')
print('best', best, 'saved')
