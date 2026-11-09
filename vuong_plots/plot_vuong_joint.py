"""
Figure: Vuong (2016) fit quality for the FULL joint fit -- f_i_mult, k_wl,
k_wg all calibrated together (holdup + dPdz loss), the fix for the
f_i_mult/k_wall degeneracy (see plot_vuong_combined.py / CLAUDE.md). Mirrors
plot_vuong_fionly.py / plot_vuong_wallonly.py's layout for direct visual
comparison across all three calibration stages (interfacial-only,
wall-only, joint).

Data is not re-run here -- it's the already-verified real output from the
NEW true-jax.grad fit through the closed-form equilibrium path
(_vuong_fit_joint_equilibrium.py), which superseded the original
finite-difference fit (f_i_mult=1.3265, k_wl=0.7532, k_wg=1.3173) once
the equilibrium path was validated to reproduce the dynamic scan to
~1e-10% (see CLAUDE.md "Nondimensionalization + equilibrium-path true-
gradient training supersedes the finite-difference joint fit"):
  - Per-condition values: fresh run of _vuong_eval_converged_detail.py
    (the DYNAMIC scan, re-checking kh_total at the new parameters) at
    f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413, all 36 conditions,
    kh_total=0 throughout -- stability was never in question, but it was
    only previously confirmed at the OLD parameters.
  - Per-epoch trajectory: the 3000-epoch true-gradient run's own stdout,
    thinned (dense through epoch 500, sparse afterward -- gradient is
    essentially exactly zero from ~epoch 1475 onward, a real converged
    optimum, not a plateau artifact). Best epoch 2303.

Bottom-left panel replaces the other two figures' "error vs. scanned
parameter" slot with loss vs. training epoch, since the joint fit has no
single scanned knob -- this is its own natural x-axis of progress.

Usage: python3 plot_vuong_joint.py
"""
import numpy as np
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

# fields: split, U_sl, U_sg, psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh_total
ROWS = [
    ("train", 0.0500, 1.620, 0.1501, 0.1460, 6.69, 5.93, 0),
    ("train", 0.0200, 1.623, 0.0833, 0.0688, 5.20, 4.42, 0),
    ("train", 0.0401, 1.625, 0.1360, 0.1224, 6.21, 5.46, 0),
    ("train", 0.0301, 1.626, 0.1101, 0.0969, 5.48, 4.96, 0),
    ("train", 0.0402, 2.379, 0.0882, 0.0892, 9.38, 9.74, 0),
    ("train", 0.0303, 2.382, 0.0672, 0.0700, 8.20, 9.04, 0),
    ("train", 0.0501, 2.396, 0.1045, 0.1065, 10.57, 10.49, 0),
    ("train", 0.0202, 2.400, 0.0552, 0.0491, 7.28, 8.34, 0),
    ("train", 0.0100, 2.417, 0.0314, 0.0259, 7.14, 7.50, 0),
    ("train", 0.0200, 2.476, 0.0500, 0.0434, 12.38, 10.42, 0),
    ("train", 0.0403, 2.479, 0.0791, 0.0799, 16.45, 12.12, 0),
    ("train", 0.0302, 2.481, 0.0657, 0.0623, 14.13, 11.33, 0),
    ("train", 0.0499, 2.482, 0.0907, 0.0955, 17.87, 12.89, 0),
    ("train", 0.0401, 2.823, 0.0805, 0.0861, 8.10, 10.10, 0),
    ("train", 0.0199, 2.828, 0.0503, 0.0471, 6.62, 8.55, 0),
    ("train", 0.0099, 2.829, 0.0294, 0.0251, 6.31, 7.66, 0),
    ("train", 0.0505, 2.844, 0.0960, 0.1037, 8.98, 10.92, 0),
    ("train", 0.0501, 3.131, 0.0728, 0.0790, 21.60, 18.30, 0),
    ("train", 0.0200, 3.132, 0.0361, 0.0353, 17.30, 15.19, 0),
    ("train", 0.0300, 3.136, 0.0497, 0.0506, 18.66, 16.33, 0),
    ("train", 0.0100, 3.138, 0.0217, 0.0188, 15.88, 13.93, 0),
    ("train", 0.0201, 3.262, 0.0330, 0.0374, 14.40, 13.52, 0),
    ("train", 0.0302, 3.264, 0.0458, 0.0536, 16.17, 14.60, 0),
    ("train", 0.0401, 3.273, 0.0603, 0.0685, 17.23, 15.60, 0),
    ("train", 0.0102, 3.288, 0.0195, 0.0201, 13.02, 12.42, 0),
    ("train", 0.0100, 4.111, 0.0169, 0.0160, 19.84, 18.10, 0),
    ("train", 0.0400, 4.387, 0.0509, 0.0600, 17.57, 19.28, 0),
    ("train", 0.0200, 4.419, 0.0294, 0.0324, 15.35, 16.99, 0),
    ("train", 0.0100, 4.439, 0.0175, 0.0171, 14.11, 15.59, 0),
    ("test", 0.0100, 1.628, 0.0559, 0.0373, 3.52, 3.87, 0),
    ("test", 0.0104, 2.457, 0.0256, 0.0242, 10.35, 9.38, 0),
    ("test", 0.0300, 2.834, 0.0654, 0.0670, 7.35, 9.39, 0),
    ("test", 0.0401, 3.134, 0.0616, 0.0652, 20.11, 17.35, 0),
    ("test", 0.0501, 3.277, 0.0740, 0.0827, 18.29, 16.52, 0),
    ("test", 0.0200, 4.114, 0.0275, 0.0303, 21.48, 19.74, 0),
    ("test", 0.0300, 4.406, 0.0412, 0.0465, 16.43, 18.16, 0),
]
split = np.array([r[0] for r in ROWS])
U_sl = np.array([r[1] for r in ROWS])
usg = np.array([r[2] for r in ROWS])
psi_meas = np.array([r[3] for r in ROWS])
psi_pred = np.array([r[4] for r in ROWS])
dpdz_meas = np.array([r[5] for r in ROWS])
dpdz_pred = np.array([r[6] for r in ROWS])
herr = (psi_pred - psi_meas) / psi_meas * 100
derr = (dpdz_pred - dpdz_meas) / dpdz_meas * 100

# true-jax.grad fit through the closed-form equilibrium path (3000 epochs,
# thinned: dense through epoch 500, sparse afterward -- flat/converged by
# ~epoch 500, confirmed still exactly flat all the way to 2999)
EPOCHS = np.array([
    0, 25, 50, 75, 100, 125, 150, 175,
    200, 225, 250, 275, 300, 325, 350, 375,
    400, 425, 450, 475, 500, 600, 800, 1000,
    1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600,
    2800, 2999,
])
F_I_MULT = np.array([
    1.0618, 1.4637, 1.5916, 1.6913, 1.8436, 2.0373, 2.2629, 2.5161,
    2.7841, 3.0482, 3.2869, 3.4837, 3.6317, 3.7343, 3.8005, 3.8407,
    3.8638, 3.8765, 3.8831, 3.8864, 3.888, 3.8892, 3.8893, 3.8893,
    3.8893, 3.8893, 3.8893, 3.8893, 3.8893, 3.8893, 3.8893, 3.8893,
    3.8893, 3.8893,
])
LOSS = np.array([
    0.174014, 0.038609, 0.037732, 0.037014, 0.036517, 0.035995, 0.035462, 0.034953,
    0.034511, 0.03417, 0.033941, 0.033809, 0.033743, 0.033715, 0.033704, 0.0337,
    0.033699, 0.033699, 0.033699, 0.033699, 0.033699, 0.033699, 0.033699, 0.033699,
    0.033699, 0.033699, 0.033699, 0.033699, 0.033699, 0.033699, 0.033699, 0.033699,
    0.033699, 0.033699,
])
LOSS_HLD = np.array([
    0.122244, 0.010037, 0.010173, 0.010006, 0.010032, 0.010052, 0.010082, 0.010119,
    0.01016, 0.010201, 0.010236, 0.010264, 0.010285, 0.010299, 0.010307, 0.010313,
    0.010316, 0.010317, 0.010318, 0.010319, 0.010319, 0.010319, 0.010319, 0.010319,
    0.010319, 0.010319, 0.010319, 0.010319, 0.010319, 0.010319, 0.010319, 0.010319,
    0.010319, 0.010319,
])
LOSS_DPDZ = np.array([
    0.05177, 0.028572, 0.02756, 0.027008, 0.026485, 0.025944, 0.02538, 0.024834,
    0.024351, 0.02397, 0.023705, 0.023545, 0.023459, 0.023416, 0.023397, 0.023388,
    0.023383, 0.023381, 0.02338, 0.02338, 0.02338, 0.02338, 0.02338, 0.02338,
    0.02338, 0.02338, 0.02338, 0.02338, 0.02338, 0.02338, 0.02338, 0.02338,
    0.02338, 0.02338,
])
BEST_EPOCH_IDX = len(EPOCHS) - 1   # 2999, last thinned point -- true best (2303) is indistinguishable (flat)
BEST_EPOCH = 2303

def new_ax(figsize=(6.2, 5.6)):
    fig, a = plt.subplots(figsize=figsize, facecolor='white')
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    return fig, a

norm = plt.Normalize(usg.min(), usg.max())
def parity(a, meas, pred, name):
    lo, hi = min(meas.min(), pred.min()) * 0.8, max(meas.max(), pred.max()) * 1.2
    xs = np.array([lo, hi])
    a.plot(xs, xs, color=INK, lw=1.2)
    for pct, ls in ((10, '--'), (20, ':')):
        a.plot(xs, xs * (1 + pct / 100), color=INK2, lw=1, ls=ls)
        a.plot(xs, xs * (1 - pct / 100), color=INK2, lw=1, ls=ls)
    for sp, mk in (('train', 'o'), ('test', 's')):
        k = split == sp
        sc = a.scatter(meas[k], pred[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=46,
                        edgecolor='white', linewidth=1.2, label=f'{sp} (n={k.sum()})', zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
    a.set_xlabel(f'measured {name}', color=INK); a.set_ylabel(f'predicted {name}', color=INK)
    return sc

trm, tem = split == 'train', split == 'test'

# --- figure 1: holdup parity ---
fig1, a = new_ax()
sc = parity(a, psi_meas, psi_pred, 'liquid holdup')
a.set_title('Holdup (dashed ±10%, dotted ±20%)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, loc='upper left')
cb = fig1.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig1.savefig('figures/joint_fit/vuong_joint_fit_holdup.png', dpi=160, bbox_inches='tight')

# --- figure 2: pressure gradient parity (custom log tick formatting) ---
fig2, a = new_ax()
sc = parity(a, dpdz_meas, dpdz_pred, '|dP/dz| [Pa/m]')
a.set_title('Pressure gradient', color=INK, loc='left', fontsize=11)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.legend(frameon=False, loc='upper left')
cb = fig2.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig2.savefig('figures/joint_fit/vuong_joint_fit_dpdz.png', dpi=160, bbox_inches='tight')

# --- figure 3: joint fit convergence ---
fig3, a = new_ax()
a.plot(EPOCHS, LOSS, '-o', color=INK, lw=2, ms=5, label='total (holdup + dPdz)')
a.plot(EPOCHS, LOSS_HLD, '-o', color=S1, lw=1.5, ms=4, label='holdup component')
a.plot(EPOCHS, LOSS_DPDZ, '-o', color=S2, lw=1.5, ms=4, label='dPdz component')
a.axvline(BEST_EPOCH, color=INK2, lw=1, ls='--')
a.text(BEST_EPOCH, a.get_ylim()[1], f' best epoch {BEST_EPOCH}\n f_i_mult={F_I_MULT[BEST_EPOCH_IDX]:.3f}',
       color=INK2, va='top', fontsize=9)
a.set_xlabel('epoch', color=INK); a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('Joint fit convergence (f_i_mult, k_wl, k_wg)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8)
fig3.savefig('figures/joint_fit/vuong_joint_fit_convergence.png', dpi=160, bbox_inches='tight')

# --- figure 4: holdup residual vs U_sg ---
fig4, a = new_ax()
a.axhline(0, color=INK, lw=1)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    a.scatter(usg[k], herr[k], marker=mk, s=40, color=S1, edgecolor='white', linewidth=1.2, label=sp, zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate', color=INK, loc='left', fontsize=11); a.legend(frameon=False)
fig4.savefig('figures/joint_fit/vuong_joint_fit_residual.png', dpi=160, bbox_inches='tight')

# --- figure 5: terminal-style "Final evaluation" summary panel, matching
# stage2_synthetic_recovery.py's styling, standing in for the suptitle that
# used to be repeated atop each of the four plots above ---
figS = plt.figure(figsize=(6.4, 4.6))
ax_text = figS.add_axes([0.06, 0.06, 0.88, 0.88])
ax_text.set_facecolor('black')
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    'Final evaluation -- Vuong (2016) joint fit',
    '',
    'Fitted: f_i_mult=3.8893  k_wl=1.5419  k_wg=0.8413',
    f'  (true jax.grad through the equilibrium path -- no finite',
    f'   differences, no dynamic scan. Converged by ~epoch 1475,',
    f'   gradient exactly 0 from there on; best epoch {BEST_EPOCH})',
    '',
    f'Dataset: 36 conditions ({trm.sum()} train / {tem.sum()} test)',
    '         0 kh-excluded, 36/36 stable',
    '',
    '                 holdup      dP/dz',
    f'  train ({trm.sum():>2})     {np.mean(np.abs(herr[trm])):5.2f}%     {np.mean(np.abs(derr[trm])):5.2f}%',
    f'  test  ({tem.sum():>2})     {np.mean(np.abs(herr[tem])):5.2f}%     {np.mean(np.abs(derr[tem])):5.2f}%',
    '',
    'mean |relative error|; test = 7 held-out U_sg clusters,',
    'never used in fitting',
]
ax_text.text(0.04, 0.96, '\n'.join(lines), transform=ax_text.transAxes,
             color='white', fontsize=10, fontfamily='monospace',
             va='top', ha='left', linespacing=1.55, clip_on=False)
figS.savefig('figures/joint_fit/vuong_joint_fit_summary.png', dpi=160, bbox_inches='tight')

print('saved: vuong_joint_fit_{holdup,dpdz,convergence,residual,summary}.png')
