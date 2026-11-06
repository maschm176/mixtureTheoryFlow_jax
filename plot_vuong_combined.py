"""
Figure: the f_i_mult / k_wall degeneracy itself -- the reason a single shared
wall-friction scalar (k_wl=k_wg=k_wall) had to be split into independent
k_wl/k_wg (see mixtureTheoryFlow_jax_solver_water_gas.py's
_wall_and_interfacial_shear docstring). Built from the 5x5 (f_i_mult, k_wall)
grid in combined_cache/, evaluated on a fixed 9-condition Vuong subset
(see _vuong_combined_worker.py for how it was picked).

Two things shown, now as four separate standalone figures:
  (a) heatmap: holdup-only pooled loss over the 2D grid -- if the degeneracy
      claim is right, this should show an elongated low-loss valley rather
      than a single well-defined minimum, since many (f_i_mult, k_wall)
      combinations fit comparably well.
  (b)-(d): predicted holdup vs. the ratio f_i_mult/k_wall for three
      conditions spanning the subset's U_sg range, with all 25 grid points
      plotted (colored by k_wall). If only the ratio is identifiable, points
      with different absolute (f_i_mult, k_wall) but the same ratio should
      collapse onto the same curve -- this is the direct, per-condition
      version of the "scaling both by the same factor leaves alpha_l_pred
      unchanged" check already described in CLAUDE.md, shown here across the
      whole grid rather than for one isolated scale factor.

Usage: python3 plot_vuong_combined.py
"""
import glob, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

INK, INK2, GRID_C = '#0b0b0b', '#52514e', '#e4e3df'
seq = LinearSegmentedColormap.from_list('seq', ['#cfe1f7', '#2a78d6', '#0d2f5e'])

data = {}
for p in sorted(glob.glob('combined_cache/f_*_k_*.pkl')):
    tag = p.split('/')[-1][:-4]  # f_<f>_k_<k>
    fs, ks = tag.split('_k_')
    f = float(fs[2:]); k = float(ks)
    data[(f, k)] = pickle.load(open(p, 'rb'))

fs = sorted({f for f, k in data})
ks = sorted({k for f, k in data})
n_cond = len(next(iter(data.values())))

def ok(rows): return np.array([r['kh_total'] == 0 and not r['any_nan'] for r in rows])
def herr(rows):
    m, p = np.array([r['psi_meas'] for r in rows]), np.array([r['psi_pred'] for r in rows])
    return (p - m) / m

loss = np.full((len(fs), len(ks)), np.nan)
for i, f in enumerate(fs):
    for j, k in enumerate(ks):
        rows = data[(f, k)]; m = ok(rows)
        loss[i, j] = np.mean(herr(rows)[m]**2) if m.any() else np.nan
i_best, j_best = np.unravel_index(np.nanargmin(loss), loss.shape)
f_best, k_best = fs[i_best], ks[j_best]

def new_ax(figsize=(6.2, 5.6)):
    fig, a = plt.subplots(figsize=figsize, facecolor='white')
    fig.subplots_adjust(top=0.78)
    a.set_facecolor('white'); a.grid(color=GRID_C, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    return fig, a

suptitle = ('Vuong (2016): the f_i_mult / k_wall degeneracy (single shared wall-friction scalar)\n'
            'Holdup alone identifies only the ratio f_i_mult/k_wall -- motivates splitting into k_wl, k_wg')

# --- figure 1: 2D loss heatmap -- the degenerate valley ---
fig1, a = new_ax()
F, K = np.meshgrid(fs, ks, indexing='ij')
pc = a.pcolormesh(F, K, np.log10(loss), cmap=seq, shading='nearest')
a.set_xscale('log'); a.set_yscale('log')
a.scatter([f_best], [k_best], marker='*', s=220, color='#d64550', edgecolor='white',
          linewidth=1, zorder=5, label=f'min loss (f={f_best:g}, k={k_best:g})')
cb = fig1.colorbar(pc, ax=a, shrink=0.85, pad=0.03); cb.set_label('log10(holdup loss)', color=INK2)
a.set_xlabel('f_i_mult', color=INK); a.set_ylabel('k_wall (shared, k_wl=k_wg)', color=INK)
a.set_title(f'Holdup loss over (f_i_mult, k_wall), n={n_cond} conditions', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, loc='upper left', fontsize=8)
fig1.suptitle(suptitle, color=INK, fontsize=10.5, x=0.02, ha='left', wrap=True)
fig1.savefig('figures/degeneracy/vuong_fi_kwall_degeneracy_heatmap.png', dpi=160, bbox_inches='tight')

# --- figure 2: ratio-collapse check, three conditions spanning the subset's U_sg range, one row ---
all_Usg = {idx: data[(fs[0], ks[0])][idx]['U_sg'] for idx in range(n_cond)}
order = sorted(all_Usg, key=all_Usg.get)
cond_idxs = [order[0], order[len(order)//2], order[-1]]

fig2, axes = plt.subplots(1, 3, figsize=(16, 5.2), facecolor='white')
for a in axes:
    a.set_facecolor('white'); a.grid(color=GRID_C, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)

sc = None
for panel_i, (a, idx) in enumerate(zip(axes, cond_idxs)):
    is_leftmost = (panel_i == 0)
    is_rightmost = (panel_i == len(axes) - 1)
    ratios, preds, kvals, stable = [], [], [], []
    for f in fs:
        for k in ks:
            rows = data[(f, k)]
            r = rows[idx]
            ratios.append(f / k); preds.append(r['psi_pred']); kvals.append(k)
            stable.append(r['kh_total'] == 0 and not r['any_nan'])
    ratios, preds, kvals, stable = map(np.array, (ratios, preds, kvals, stable))
    sc = a.scatter(ratios[stable], preds[stable], c=kvals[stable], cmap=seq, s=55,
                    edgecolor='white', linewidth=1, zorder=3)
    if (~stable).any():
        a.scatter(ratios[~stable], preds[~stable], facecolors='none', edgecolors='#d64550',
                   s=90, linewidth=1.6, zorder=4,
                   label='kh-unstable' if is_leftmost else None)
    a.set_xscale('log')
    meas = data[(fs[0], ks[0])][idx]['psi_meas']
    a.axhline(meas, color=INK, lw=1, ls='--', label='measured' if is_leftmost else None)
    if is_leftmost:
        a.legend(frameon=False, fontsize=8, loc='upper right')
    a.set_xlabel('f_i_mult / k_wall  (ratio)', color=INK)
    if is_leftmost:
        a.set_ylabel('alpha_l_pred', color=INK)
    a.set_title(f"U_sg={all_Usg[idx]:.2f} m/s", color=INK, loc='left', fontsize=10)

cb = fig2.colorbar(sc, ax=axes[-1], shrink=0.85, pad=0.03)
cb.set_label('k_wall', color=INK2, fontsize=9)
fig2.savefig('figures/degeneracy/vuong_fi_kwall_degeneracy_ratio.png', dpi=160, bbox_inches='tight')

print('saved 2 figures, min loss at f=%g k=%g' % (f_best, k_best))
