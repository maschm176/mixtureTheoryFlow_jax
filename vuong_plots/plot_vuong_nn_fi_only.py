"""
Figure: Vuong (2016) fit quality for the "Option B" hybrid -- a 13-weight
NN(log Re_SL, log Re_SG) -> f_i_mult (single output), with k_wl/k_wg as
ordinary flat scalars, all trained JOINTLY (same structure as the power
law fit, just swapping its 2-parameter form for a small network). Mirrors
plot_vuong_powerlaw.py / plot_vuong_nn.py's layout for direct comparison
across all four calibration stages (flat -> power-law -> this hybrid ->
full 27-weight NN).

Data is not re-run here -- it's the already-verified real output of
_vuong_eval_nn_fi_only_detail.py (retrains _vuong_fit_nn_fi_only.py's
exact architecture/warm-start/seed, then runs the DYNAMIC scan per
condition re-checking kh_total -- all 36/36 stable). Train loss=0.020799
(hld=0.008100, dpdz=0.012698), test loss=0.025095 (hld=0.009782,
dpdz=0.015313) -- beats both the flat fit AND the power law on both
splits, with train/test ratio ~1.2x (vs. the full NN's ~20x) -- i.e. no
sign of the degeneracy-exploit overfitting the full NN showed, since
k_wl/k_wg here are plain scalars with no way to vary per-condition.

One nuance worth seeing visually: despite the better aggregate loss, this
fit's correlation(U_sg, holdup_err%)=0.379 is WORSE than the power law's
0.220 -- the power law's smooth, monotonic functional form tracks the
systematic holdup-vs-gas-rate trend more cleanly than this network's
less-constrained (and partly saturating-tanh-driven, piecewise-looking)
shape does, even though it fits dPdz magnitude better in aggregate. The
f_i_mult-vs-Re_SG panel below makes this visible directly: unlike the
power law's clean curve, this fit's points don't collapse onto one smooth
line, and several near-identical f_i_mult values repeat at visibly
different Re_SG.

Usage: python3 plot_vuong_nn_fi_only.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, FuncFormatter, NullFormatter

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, S3 = '#2a78d6', '#eb6834', '#2f9e68'
seq = LinearSegmentedColormap.from_list('seq', ['#cfe1f7', '#2a78d6', '#0d2f5e'])

def _log_tick_fmt(val, pos=None):
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

# fields: split, U_sl, U_sg, Re_SG, f_i_mult, psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh_total
ROWS = [
    ("train", 0.0500, 1.620, 334216.5, 4.1911, 0.1501, 0.1533, 6.69, 5.99, 0),
    ("train", 0.0200, 1.623, 336475.9, 4.0978, 0.0833, 0.0730, 5.20, 4.30, 0),
    ("train", 0.0401, 1.625, 334644.7, 4.1911, 0.1360, 0.1287, 6.21, 5.47, 0),
    ("train", 0.0301, 1.626, 334869.9, 4.1911, 0.1101, 0.1020, 5.48, 4.92, 0),
    ("train", 0.0402, 2.379, 503018.1, 4.4388, 0.0882, 0.0919, 9.38, 9.81, 0),
    ("train", 0.0303, 2.382, 504477.5, 4.2306, 0.0672, 0.0735, 8.20, 8.88, 0),
    ("train", 0.0501, 2.396, 501681.7, 4.7778, 0.1045, 0.1070, 10.57, 10.90, 0),
    ("train", 0.0202, 2.400, 505872.8, 3.1474, 0.0552, 0.0578, 7.28, 7.39, 0),
    ("train", 0.0100, 2.417, 508457.5, 3.1395, 0.0314, 0.0308, 7.14, 6.53, 0),
    ("train", 0.0200, 2.476, 666671.0, 5.4450, 0.0500, 0.0413, 12.38, 10.89, 0),
    ("train", 0.0403, 2.479, 662840.4, 6.4615, 0.0791, 0.0720, 16.45, 13.76, 0),
    ("train", 0.0302, 2.481, 664701.8, 6.4689, 0.0657, 0.0558, 14.13, 12.74, 0),
    ("train", 0.0499, 2.482, 662066.4, 6.4200, 0.0907, 0.0865, 17.87, 14.69, 0),
    ("train", 0.0401, 2.823, 428896.0, 4.1913, 0.0805, 0.0907, 8.10, 9.99, 0),
    ("train", 0.0199, 2.828, 431225.1, 3.2131, 0.0503, 0.0552, 6.62, 7.60, 0),
    ("train", 0.0099, 2.829, 436593.7, 3.1395, 0.0294, 0.0299, 6.31, 6.67, 0),
    ("train", 0.0505, 2.844, 428728.3, 4.1925, 0.0960, 0.1092, 8.98, 10.89, 0),
    ("train", 0.0501, 3.131, 826080.5, 6.5407, 0.0728, 0.0708, 21.60, 20.86, 0),
    ("train", 0.0200, 3.132, 827311.3, 5.4629, 0.0361, 0.0335, 17.30, 15.77, 0),
    ("train", 0.0300, 3.136, 827697.7, 5.7633, 0.0497, 0.0472, 18.66, 17.51, 0),
    ("train", 0.0100, 3.138, 828318.3, 5.4629, 0.0217, 0.0177, 15.88, 14.12, 0),
    ("train", 0.0201, 3.262, 672486.1, 5.4528, 0.0330, 0.0355, 14.40, 14.14, 0),
    ("train", 0.0302, 3.264, 671981.7, 6.4825, 0.0458, 0.0477, 16.17, 16.46, 0),
    ("train", 0.0401, 3.273, 671213.1, 6.5067, 0.0603, 0.0611, 17.23, 17.79, 0),
    ("train", 0.0102, 3.288, 675998.4, 5.3159, 0.0195, 0.0192, 13.02, 12.57, 0),
    ("train", 0.0100, 4.111, 838159.6, 5.4629, 0.0169, 0.0151, 19.84, 18.32, 0),
    ("train", 0.0400, 4.387, 627874.7, 5.5083, 0.0509, 0.0568, 17.57, 20.74, 0),
    ("train", 0.0200, 4.419, 623214.5, 4.6355, 0.0294, 0.0327, 15.35, 16.80, 0),
    ("train", 0.0100, 4.439, 624813.0, 4.3316, 0.0175, 0.0178, 14.11, 14.79, 0),
    ("test", 0.0100, 1.628, 336777.2, 3.1395, 0.0559, 0.0442, 3.52, 3.40, 0),
    ("test", 0.0104, 2.457, 669201.2, 5.2727, 0.0256, 0.0232, 10.35, 9.49, 0),
    ("test", 0.0300, 2.834, 430714.3, 4.1911, 0.0654, 0.0707, 7.35, 9.20, 0),
    ("test", 0.0401, 3.134, 826810.4, 6.5406, 0.0616, 0.0582, 20.11, 19.63, 0),
    ("test", 0.0501, 3.277, 668477.3, 6.4757, 0.0740, 0.0742, 18.29, 18.95, 0),
    ("test", 0.0200, 4.114, 836019.9, 5.4629, 0.0275, 0.0286, 21.48, 20.48, 0),
    ("test", 0.0300, 4.406, 623619.8, 5.5056, 0.0412, 0.0439, 16.43, 19.35, 0),
]
split = np.array([r[0] for r in ROWS])
U_sl = np.array([r[1] for r in ROWS])
usg = np.array([r[2] for r in ROWS])
re_sg = np.array([r[3] for r in ROWS])
fi_mult = np.array([r[4] for r in ROWS])
psi_meas = np.array([r[5] for r in ROWS])
psi_pred = np.array([r[6] for r in ROWS])
dpdz_meas = np.array([r[7] for r in ROWS])
dpdz_pred = np.array([r[8] for r in ROWS])
herr = (psi_pred - psi_meas) / psi_meas * 100
derr = (dpdz_pred - dpdz_meas) / dpdz_meas * 100
trm, tem = split == 'train', split == 'test'

K_WL, K_WG = 1.8613, 0.7085

# power-law curve, for overlay comparison (already-converged result)
F0, ALPHA_FI = 5.5650, 0.6083
MEAN_LOG_RE_SG = 13.2389

# per-epoch trajectory (_vuong_eval_nn_fi_only_detail.py, logged every 30
# epochs; thinned here to every 150 to match the power-law plot's density)
_EPOCHS_FULL = list(range(0, 3000, 30)) + [2999]
_LOSS_FULL = [0.033699, 0.023709, 0.02252, 0.021943, 0.021703, 0.021602, 0.02155, 0.021518, 0.021496, 0.02148, 0.021466, 0.021451, 0.021435, 0.021419, 0.021403, 0.021388, 0.021375, 0.021363, 0.021352, 0.021342, 0.021332, 0.021323, 0.021314, 0.021304, 0.021295, 0.021286, 0.021277, 0.021269, 0.021261, 0.021253, 0.021245, 0.021238, 0.021232, 0.021226, 0.02122, 0.021214, 0.021209, 0.021204, 0.0212, 0.021195, 0.021191, 0.021187, 0.021183, 0.02118, 0.021176, 0.021172, 0.021168, 0.021156, 0.021104, 0.021064, 0.021044, 0.021028, 0.021014, 0.021001, 0.020989, 0.020978, 0.020968, 0.020959, 0.02095, 0.020942, 0.020934, 0.020927, 0.020921, 0.020914, 0.020909, 0.020903, 0.020898, 0.020893, 0.020888, 0.020883, 0.020879, 0.020875, 0.020871, 0.020867, 0.020865, 0.02086, 0.020856, 0.020853, 0.02085, 0.020847, 0.020844, 0.020841, 0.020838, 0.020835, 0.020833, 0.02083, 0.020828, 0.020825, 0.020823, 0.020822, 0.020819, 0.020816, 0.020814, 0.020812, 0.02081, 0.020808, 0.020806, 0.020804, 0.020803, 0.020801, 0.020799]
_LOSS_HLD_FULL = [0.010319, 0.005785, 0.006199, 0.006566, 0.006646, 0.006703, 0.006714, 0.006709, 0.006702, 0.006697, 0.006697, 0.006704, 0.00672, 0.006742, 0.006759, 0.006771, 0.00678, 0.006788, 0.006796, 0.006805, 0.006814, 0.006826, 0.006838, 0.006853, 0.006867, 0.006883, 0.006898, 0.006913, 0.006928, 0.006942, 0.006955, 0.006967, 0.006978, 0.006988, 0.006998, 0.007006, 0.007015, 0.007022, 0.00703, 0.007036, 0.007043, 0.007049, 0.007055, 0.007061, 0.007067, 0.007075, 0.007088, 0.007136, 0.007349, 0.007563, 0.007639, 0.00769, 0.007724, 0.007751, 0.007774, 0.007796, 0.007815, 0.007833, 0.007849, 0.007865, 0.007879, 0.007892, 0.007905, 0.007916, 0.007927, 0.007937, 0.007947, 0.007956, 0.007964, 0.007972, 0.00798, 0.007987, 0.007994, 0.008, 0.00801, 0.008013, 0.008018, 0.008023, 0.008028, 0.008033, 0.008038, 0.008042, 0.008047, 0.008051, 0.008055, 0.008058, 0.008062, 0.008066, 0.008069, 0.008072, 0.008075, 0.008078, 0.008081, 0.008084, 0.008086, 0.008089, 0.008092, 0.008094, 0.008096, 0.008098, 0.0081]
_LOSS_DPDZ_FULL = [0.02338, 0.017924, 0.016321, 0.015377, 0.015057, 0.014899, 0.014835, 0.014809, 0.014795, 0.014783, 0.014769, 0.014747, 0.014715, 0.014677, 0.014644, 0.014617, 0.014595, 0.014575, 0.014556, 0.014537, 0.014518, 0.014497, 0.014475, 0.014452, 0.014428, 0.014403, 0.014379, 0.014355, 0.014333, 0.014311, 0.014291, 0.014272, 0.014254, 0.014238, 0.014222, 0.014208, 0.014194, 0.014182, 0.01417, 0.014159, 0.014148, 0.014138, 0.014129, 0.014119, 0.014109, 0.014098, 0.014079, 0.01402, 0.013754, 0.013501, 0.013404, 0.013338, 0.013289, 0.01325, 0.013215, 0.013182, 0.013153, 0.013126, 0.013101, 0.013077, 0.013055, 0.013035, 0.013016, 0.012998, 0.012981, 0.012966, 0.012951, 0.012937, 0.012923, 0.012911, 0.012899, 0.012888, 0.012877, 0.012866, 0.012855, 0.012847, 0.012838, 0.01283, 0.012821, 0.012813, 0.012806, 0.012798, 0.012791, 0.012785, 0.012778, 0.012772, 0.012766, 0.01276, 0.012754, 0.012751, 0.012744, 0.012738, 0.012733, 0.012728, 0.012724, 0.012719, 0.012715, 0.01271, 0.012706, 0.012702, 0.012699]
EPOCHS = np.array(_EPOCHS_FULL[::5] + [_EPOCHS_FULL[-1]])
LOSS = np.array(_LOSS_FULL[::5] + [_LOSS_FULL[-1]])
LOSS_HLD = np.array(_LOSS_HLD_FULL[::5] + [_LOSS_HLD_FULL[-1]])
LOSS_DPDZ = np.array(_LOSS_DPDZ_FULL[::5] + [_LOSS_DPDZ_FULL[-1]])
BEST_EPOCH = 2999

# four-way model comparison
MODELS = ['flat\n3-scalar', 'power-law\n(f_i_mult)', 'NN f_i_mult\nonly (this)', '27-weight\nNN (all 3)']
TRAIN_LOSS = [0.033699, 0.027961, 0.020799, 0.005273]
TEST_LOSS = [0.042799, 0.026653, 0.025095, 0.104200]
USG_CORR = [0.610, 0.220, 0.379, 0.283]

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

# --- figure 1: holdup parity ---
fig1, a = new_ax()
sc = parity(a, psi_meas, psi_pred, 'liquid holdup')
a.set_title('Holdup (dashed ±10%, dotted ±20%)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, loc='upper left')
cb = fig1.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig1.savefig('figures/nn_fi_only/vuong_nn_fi_only_holdup.png', dpi=160, bbox_inches='tight')

# --- figure 2: pressure gradient parity ---
fig2, a = new_ax()
sc = parity(a, dpdz_meas, dpdz_pred, '|dP/dz| [Pa/m]')
a.set_title('Pressure gradient', color=INK, loc='left', fontsize=11)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.legend(frameon=False, loc='upper left')
cb = fig2.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig2.savefig('figures/nn_fi_only/vuong_nn_fi_only_dpdz.png', dpi=160, bbox_inches='tight')

# --- figure 3: convergence ---
fig3, a = new_ax()
a.plot(EPOCHS, LOSS, '-o', color=INK, lw=2, ms=4, label='total (holdup + dPdz), train')
a.plot(EPOCHS, LOSS_HLD, '-o', color=S1, lw=1.5, ms=3, label='holdup component')
a.plot(EPOCHS, LOSS_DPDZ, '-o', color=S2, lw=1.5, ms=3, label='dPdz component')
a.axhline(TEST_LOSS[2], color=S3, lw=1.5, ls='--', label=f'held-out TEST loss (final, ={TEST_LOSS[2]:.3f})')
a.axvline(BEST_EPOCH, color=INK2, lw=1, ls=':')
a.set_xlabel('epoch', color=INK); a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('NN f_i_mult-only fit: train and test stay close (ratio ~1.2x)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8)
fig3.savefig('figures/nn_fi_only/vuong_nn_fi_only_convergence.png', dpi=160, bbox_inches='tight')

# --- figure 4: holdup residual vs U_sg ---
fig4, a = new_ax()
a.axhline(0, color=INK, lw=1)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    a.scatter(usg[k], herr[k], marker=mk, s=40, color=S1, edgecolor='white', linewidth=1.2, label=sp, zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate', color=INK, loc='left', fontsize=11); a.legend(frameon=False)
fig4.savefig('figures/nn_fi_only/vuong_nn_fi_only_residual.png', dpi=160, bbox_inches='tight')

# --- figure 5: f_i_mult vs Re_SG -- NN output (scattered, no imposed
# functional form) vs. the power law's smooth fitted curve, for direct
# visual comparison of why the power law's U_sg-correlation is lower ---
fig5, a = new_ax()
re_grid = np.logspace(np.log10(re_sg.min()) - 0.05, np.log10(re_sg.max()) + 0.05, 200)
powerlaw_curve = F0 * np.exp(ALPHA_FI * (np.log(re_grid) - MEAN_LOG_RE_SG))
a.plot(re_grid, powerlaw_curve, color=INK2, lw=1.5, ls='--',
       label=f'power-law fit (for reference): f0={F0:.2f}, alpha_fi={ALPHA_FI:+.3f}', zorder=2)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    sc5 = a.scatter(re_sg[k], fi_mult[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=46,
                     edgecolor='white', linewidth=1.2, label=f'NN f_i_mult, {sp} (n={k.sum()})', zorder=3)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.set_xlabel('Re_SG [-]', color=INK); a.set_ylabel('f_i_mult [-]', color=INK)
a.set_title('NN-learned f_i_mult vs. Re_SG -- no smooth curve, unlike the power law',
            color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8, loc='upper left')
cb = fig5.colorbar(sc5, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig5.savefig('figures/nn_fi_only/vuong_nn_fi_only_vs_resg.png', dpi=160, bbox_inches='tight')

# --- figure 6: four-way model comparison bar chart ---
fig6, a = new_ax(figsize=(7.4, 5.0))
x = np.arange(len(MODELS))
w = 0.32
a.bar(x - w/2, TRAIN_LOSS, width=w, color=S1, label='train loss', zorder=3)
a.bar(x + w/2, TEST_LOSS, width=w, color=S2, label='test loss', zorder=3)
for xi, (tr, te) in enumerate(zip(TRAIN_LOSS, TEST_LOSS)):
    a.text(xi - w/2, tr + 0.002, f'{tr:.3f}', ha='center', fontsize=8.5, color=INK)
    a.text(xi + w/2, te + 0.002, f'{te:.3f}', ha='center', fontsize=8.5, color=INK)
a.set_xticks(x); a.set_xticklabels(MODELS, color=INK, fontsize=9)
a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('Model comparison: flat vs. power-law vs. NN(f_i_mult only) vs. full NN',
            color=INK, loc='left', fontsize=10.5)
a.legend(frameon=False)
fig6.savefig('figures/nn_fi_only/vuong_nn_fi_only_model_comparison.png', dpi=160, bbox_inches='tight')

# --- figure 7: terminal-style summary panel ---
fig7 = plt.figure(figsize=(6.8, 5.6))
ax_text = fig7.add_axes([0.06, 0.06, 0.88, 0.88])
ax_text.set_facecolor('black')
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    'Final evaluation -- Vuong (2016) "Option B" hybrid',
    '',
    'f_i_mult = NN(log Re_SL, log Re_SG): 2 -> 3 (tanh) -> 1',
    '(softplus), 13 weights. k_wl, k_wg stay flat scalars --',
    'NOT network outputs -- all trained jointly (15 params),',
    f'warm-started to the converged flat fit. Fitted: k_wl={K_WL:.4f},',
    f'k_wg={K_WG:.4f} (flat fit was 1.5419 / 0.8413).',
    '',
    f'Dataset: 36 conditions ({trm.sum()} train / {tem.sum()} test)',
    '         0 kh-excluded, 36/36 stable',
    '',
    '                 holdup      dP/dz',
    f'  train ({trm.sum():>2})     {np.mean(np.abs(herr[trm])):5.2f}%     {np.mean(np.abs(derr[trm])):5.2f}%',
    f'  test  ({tem.sum():>2})     {np.mean(np.abs(herr[tem])):5.2f}%     {np.mean(np.abs(derr[tem])):5.2f}%',
    '',
    f'corr(U_sg, holdup_err%) = {np.corrcoef(usg, herr)[0,1]:.3f}  '
    f'(flat 0.610, power-law 0.220, full NN 0.283)',
    '',
    'Beats BOTH flat and power-law on train AND test loss',
    '(train 0.0337->0.0280->0.0208; test 0.0428->0.0267->0.0251),',
    'with train/test ratio ~1.2x -- no sign of the full NN\'s',
    'degeneracy-exploit overfitting (k_wl/k_wg cannot vary',
    'per-condition here, unlike when they were network outputs).',
    '',
    'But corr(U_sg,...) is WORSE than the power law\'s -- the power',
    'law\'s smooth monotonic form tracks the systematic gas-rate',
    'trend more cleanly than this less-constrained NN shape does,',
    'despite its better aggregate (mostly dPdz-driven) loss.',
]
ax_text.text(0.04, 0.96, '\n'.join(lines), transform=ax_text.transAxes,
             color='white', fontsize=9.0, fontfamily='monospace',
             va='top', ha='left', linespacing=1.5, clip_on=False)
fig7.savefig('figures/nn_fi_only/vuong_nn_fi_only_summary.png', dpi=160, bbox_inches='tight')

print('saved: vuong_nn_fi_only_{holdup,dpdz,convergence,residual,vs_resg,model_comparison,summary}.png')
