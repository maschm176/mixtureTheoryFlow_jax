"""
Figure: Vuong (2016) fit quality for the FULL joint fit -- f_i_mult, k_wl,
k_wg all calibrated together (holdup + dPdz loss), the fix for the
f_i_mult/k_wall degeneracy (see plot_vuong_combined.py / CLAUDE.md). Mirrors
plot_vuong_fionly.py / plot_vuong_wallonly.py's layout for direct visual
comparison across all three calibration stages (interfacial-only,
wall-only, joint).

Data is not re-run here -- it's the already-verified real output already
used to add Vuong to figures/stage1_forward_stability.py:
  - Per-condition values: fresh run of _vuong_eval_epoch5_detail.py at the
    converged params (f_i_mult=1.3265, k_wl=0.7532, k_wg=1.3173, epoch 5),
    all 36 conditions, kh_total=0 throughout. Matches CLAUDE.md's reported
    9.90%/14.41% train and 15.42%/12.38% test holdup/dPdz error exactly.
  - Per-epoch trajectory: the live _vuong_fit_joint.py run's own stdout
    (epochs 0-19 before being stopped manually -- best epoch 5, loss
    plateaued/noisy afterward; see CLAUDE.md "Vuong (2016) joint ...
    calibration" for the full writeup of why it was stopped there).

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
    ("train", 0.0500, 1.620, 0.1501, 0.1468, 6.69, 5.44, 0),
    ("train", 0.0200, 1.623, 0.0833, 0.0722, 5.20, 4.47, 0),
    ("train", 0.0401, 1.625, 0.1360, 0.1244, 6.21, 5.14, 0),
    ("train", 0.0301, 1.626, 0.1101, 0.0998, 5.48, 4.81, 0),
    ("train", 0.0402, 2.379, 0.0882, 0.0922, 9.38, 9.56, 0),
    ("train", 0.0303, 2.382, 0.0672, 0.0734, 8.20, 9.14, 0),
    ("train", 0.0501, 2.396, 0.1045, 0.1090, 10.57, 10.07, 0),
    ("train", 0.0202, 2.400, 0.0552, 0.0524, 7.28, 8.76, 0),
    ("train", 0.0100, 2.417, 0.0314, 0.0285, 7.14, 8.36, 0),
    ("train", 0.0200, 2.476, 0.0500, 0.0464, 12.38, 11.19, 0),
    ("train", 0.0403, 2.479, 0.0791, 0.0829, 16.45, 12.20, 0),
    ("train", 0.0302, 2.481, 0.0657, 0.0655, 14.13, 11.72, 0),
    ("train", 0.0499, 2.482, 0.0907, 0.0980, 17.87, 12.69, 0),
    ("train", 0.0401, 2.823, 0.0805, 0.0897, 8.10, 9.82, 0),
    ("train", 0.0199, 2.828, 0.0503, 0.0506, 6.62, 8.90, 0),
    ("train", 0.0099, 2.829, 0.0294, 0.0277, 6.31, 8.47, 0),
    ("train", 0.0505, 2.844, 0.0960, 0.1069, 8.98, 10.37, 0),
    ("train", 0.0501, 3.131, 0.0728, 0.0820, 21.60, 18.44, 0),
    ("train", 0.0200, 3.132, 0.0361, 0.0381, 17.30, 16.63, 0),
    ("train", 0.0300, 3.136, 0.0497, 0.0538, 18.66, 17.27, 0),
    ("train", 0.0100, 3.138, 0.0217, 0.0208, 15.88, 16.10, 0),
    ("train", 0.0201, 3.262, 0.0330, 0.0405, 14.40, 14.57, 0),
    ("train", 0.0302, 3.264, 0.0458, 0.0570, 16.17, 15.17, 0),
    ("train", 0.0401, 3.273, 0.0603, 0.0719, 17.23, 15.78, 0),
    ("train", 0.0102, 3.288, 0.0195, 0.0223, 13.02, 14.14, 0),
    ("train", 0.0100, 4.111, 0.0169, 0.0179, 19.84, 21.00, 0),
    ("train", 0.0400, 4.387, 0.0509, 0.0637, 17.57, 19.50, 0),
    ("train", 0.0200, 4.419, 0.0294, 0.0354, 15.35, 18.30, 0),
    ("train", 0.0100, 4.439, 0.0175, 0.0191, 14.11, 17.78, 0),
    ("test", 0.0100, 1.628, 0.0559, 0.0404, 3.52, 4.17, 0),
    ("test", 0.0104, 2.457, 0.0256, 0.0265, 10.35, 10.62, 0),
    ("test", 0.0300, 2.834, 0.0654, 0.0708, 7.35, 9.40, 0),
    ("test", 0.0401, 3.134, 0.0616, 0.0684, 20.11, 17.86, 0),
    ("test", 0.0501, 3.277, 0.0740, 0.0859, 18.29, 16.36, 0),
    ("test", 0.0200, 4.114, 0.0275, 0.0330, 21.48, 21.68, 0),
    ("test", 0.0300, 4.406, 0.0412, 0.0500, 16.43, 18.87, 0),
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

# live _vuong_fit_joint.py run, epochs 0-19 (stopped manually; best=epoch 5)
EPOCHS = np.arange(20)
F_I_MULT = np.array([1.0000, 1.0618, 1.1265, 1.1931, 1.2604, 1.3265, 1.3895, 1.4473, 1.4980,
                      1.5405, 1.5743, 1.5992, 1.6159, 1.6249, 1.6272, 1.6239, 1.6161, 1.6046,
                      1.5904, 1.5745])
LOSS = np.array([0.174014, 0.124628, 0.087816, 0.062882, 0.048727, 0.043427, 0.043909, 0.046593,
                  0.048888, 0.049851, 0.049639, 0.048804, 0.047880, 0.047217, 0.046938, 0.046978,
                  0.047148, 0.047225, 0.047021, 0.046428])
LOSS_HLD = np.array([0.122244, 0.084082, 0.055481, 0.035213, 0.021947, 0.014280, 0.010774, 0.010058,
                      0.010964, 0.012623, 0.014443, 0.016054, 0.017246, 0.017932, 0.018108, 0.017827,
                      0.017177, 0.016264, 0.015195, 0.014075])
LOSS_DPDZ = np.array([0.051770, 0.040547, 0.032335, 0.027670, 0.026781, 0.029147, 0.033135, 0.036535,
                       0.037924, 0.037228, 0.035196, 0.032750, 0.030635, 0.029285, 0.028831, 0.029151,
                       0.029971, 0.030962, 0.031826, 0.032353])
BEST_EPOCH = 5

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
a.text(BEST_EPOCH, a.get_ylim()[1], f' best epoch {BEST_EPOCH}\n f_i_mult={F_I_MULT[BEST_EPOCH]:.3f}',
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
    'Fitted: f_i_mult=1.3265  k_wl=0.7532  k_wg=1.3173',
    f'  (converged epoch {BEST_EPOCH} of 20; run stopped manually at epoch 19',
    '   after 13 epochs with no improvement over epoch 5)',
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
