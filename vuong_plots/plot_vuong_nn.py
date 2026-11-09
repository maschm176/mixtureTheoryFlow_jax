"""
Figure: Vuong (2016) fit quality for the 27-weight NN closure
(f_i_mult, k_wl, k_wg) = NN(log Re_SL, log Re_SG) -- 2 -> 4 (tanh) -> 3
(softplus), warm-started to the converged flat fit. Mirrors
plot_vuong_powerlaw.py's layout for direct visual comparison across
calibration stages (flat scalar -> power-law -> this NN).

Data is not re-run here -- it's the already-verified real output of
_vuong_eval_nn_detail.py (retrains _vuong_fit_nn_closure.py's exact
architecture/warm-start/hyperparameters, deterministic PRNGKey(0), then
runs the DYNAMIC scan per condition re-checking kh_total -- all 36/36
stable). Train loss=0.005273 (hld=0.002973, dpdz=0.002300), test
loss=0.104200 (hld=0.005847, dpdz=0.098352) -- badly overfit relative to
train, confirming CLAUDE.md "NN closure overfits on 29 points...".

One extra panel beyond the holdup/dPdz/convergence/residual convention,
since this closure has THREE outputs (not one curve like the power law):
  - (f_i_mult, k_wl, k_wg) vs. Re_SG, three side-by-side panels -- shows
    the network has no smooth functional structure (unlike the power
    law's clean curve) and, specifically, that k_wg collapses toward
    near-zero for many conditions while f_i_mult/k_wl compensate with
    inflated values -- the per-condition mechanism CLAUDE.md attributes
    the overfitting to (a local version of the f_i_mult/k_wall ratio
    degeneracy, exploited condition-by-condition with 27 free weights).

Usage: python3 plot_vuong_nn.py
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

# fields: split, U_sl, U_sg, Re_SL, Re_SG, f_i_mult, k_wl, k_wg,
#         psi_meas, psi_pred, dpdz_meas, dpdz_pred, kh_total
ROWS = [
    ("train", 0.0500, 1.620, 4385.2, 334216.5, 7.5247, 3.1684, 0.0033, 0.1501, 0.1557, 6.69, 6.62, 0),
    ("train", 0.0200, 1.623, 1753.4, 336475.9, 7.5344, 4.5289, 0.1568, 0.0833, 0.0836, 5.20, 5.20, 0),
    ("train", 0.0401, 1.625, 3515.1, 334644.7, 7.6939, 3.4593, 0.0048, 0.1360, 0.1335, 6.21, 6.17, 0),
    ("train", 0.0301, 1.626, 2637.9, 334869.9, 7.7766, 3.7820, 0.0098, 0.1101, 0.1087, 5.48, 5.55, 0),
    ("train", 0.0402, 2.379, 3525.9, 503018.1, 7.6985, 2.7262, 0.0012, 0.0882, 0.0884, 9.38, 9.28, 0),
    ("train", 0.0303, 2.382, 2654.6, 504477.5, 7.6176, 2.8127, 0.0020, 0.0672, 0.0702, 8.20, 8.24, 0),
    ("train", 0.0501, 2.396, 4389.8, 501681.7, 7.7193, 2.7159, 0.0011, 0.1045, 0.1057, 10.57, 10.36, 0),
    ("train", 0.0202, 2.400, 1772.6, 505872.8, 7.2480, 3.3769, 0.0280, 0.0552, 0.0538, 7.28, 7.42, 0),
    ("train", 0.0100, 2.417, 875.1, 508457.5, 6.8808, 3.8493, 0.2550, 0.0314, 0.0305, 7.14, 7.06, 0),
    ("train", 0.0200, 2.476, 1751.5, 666671.0, 10.4232, 3.9012, 0.0149, 0.0500, 0.0434, 12.38, 11.61, 0),
    ("train", 0.0403, 2.479, 3533.5, 662840.4, 10.8421, 3.1742, 0.0006, 0.0791, 0.0737, 16.45, 14.42, 0),
    ("train", 0.0302, 2.481, 2646.8, 664701.8, 10.7718, 3.2766, 0.0009, 0.0657, 0.0579, 14.13, 12.93, 0),
    ("train", 0.0499, 2.482, 4372.0, 662066.4, 10.8721, 3.1707, 0.0005, 0.0907, 0.0884, 17.87, 15.89, 0),
    ("train", 0.0401, 2.823, 3517.3, 428896.0, 6.7414, 2.0485, 0.0009, 0.0805, 0.0801, 8.10, 8.09, 0),
    ("train", 0.0199, 2.828, 1746.8, 431225.1, 6.3751, 2.7624, 0.0239, 0.0503, 0.0501, 6.62, 6.61, 0),
    ("train", 0.0099, 2.829, 868.3, 436593.7, 6.1423, 3.3047, 0.2131, 0.0294, 0.0291, 6.31, 6.36, 0),
    ("train", 0.0505, 2.844, 4425.7, 428728.3, 6.7522, 2.0313, 0.0008, 0.0960, 0.0967, 8.98, 9.04, 0),
    ("train", 0.0501, 3.131, 4391.6, 826080.5, 11.3039, 3.4958, 0.0006, 0.0728, 0.0743, 21.60, 23.17, 0),
    ("train", 0.0200, 3.132, 1752.7, 827311.3, 10.9236, 4.2896, 0.0177, 0.0361, 0.0359, 17.30, 17.27, 0),
    ("train", 0.0300, 3.136, 2629.0, 827697.7, 11.2355, 3.6370, 0.0011, 0.0497, 0.0479, 18.66, 19.08, 0),
    ("train", 0.0100, 3.138, 878.0, 828318.3, 10.6736, 4.8164, 0.1521, 0.0217, 0.0201, 15.88, 15.47, 0),
    ("train", 0.0201, 3.262, 1764.1, 672486.1, 10.5290, 3.9632, 0.0145, 0.0330, 0.0374, 14.40, 15.10, 0),
    ("train", 0.0302, 3.264, 2645.7, 671981.7, 10.8837, 3.3613, 0.0010, 0.0458, 0.0497, 16.17, 16.83, 0),
    ("train", 0.0401, 3.273, 3517.8, 671213.1, 10.9642, 3.2667, 0.0006, 0.0603, 0.0631, 17.23, 18.79, 0),
    ("train", 0.0102, 3.288, 893.1, 675998.4, 10.2252, 4.4676, 0.1325, 0.0195, 0.0213, 13.02, 13.39, 0),
    ("train", 0.0100, 4.111, 872.5, 838159.6, 10.6740, 4.8175, 0.1524, 0.0169, 0.0171, 19.84, 19.83, 0),
    ("train", 0.0400, 4.387, 3510.3, 627874.7, 9.6397, 2.3063, 0.0004, 0.0509, 0.0502, 17.57, 18.95, 0),
    ("train", 0.0200, 4.419, 1754.7, 623214.5, 8.8137, 2.7338, 0.0092, 0.0294, 0.0297, 15.35, 14.85, 0),
    ("train", 0.0100, 4.439, 872.6, 624813.0, 8.9734, 3.6489, 0.1037, 0.0175, 0.0175, 14.11, 14.20, 0),
    ("test", 0.0100, 1.628, 877.9, 336777.2, 7.2989, 5.1114, 1.0150, 0.0559, 0.0479, 3.52, 6.39, 0),
    ("test", 0.0104, 2.457, 910.6, 669201.2, 10.1025, 4.3708, 0.1269, 0.0256, 0.0255, 10.35, 10.11, 0),
    ("test", 0.0300, 2.834, 2627.4, 430714.3, 6.6855, 2.1547, 0.0015, 0.0654, 0.0635, 7.35, 7.26, 0),
    ("test", 0.0401, 3.134, 3512.1, 826810.4, 11.2938, 3.5164, 0.0007, 0.0616, 0.0612, 20.11, 21.16, 0),
    ("test", 0.0501, 3.277, 4390.6, 668477.3, 10.9661, 3.2409, 0.0005, 0.0740, 0.0763, 18.29, 20.59, 0),
    ("test", 0.0200, 4.114, 1754.7, 836019.9, 10.9250, 4.2880, 0.0175, 0.0275, 0.0307, 21.48, 22.17, 0),
    ("test", 0.0300, 4.406, 2625.8, 623619.8, 9.2171, 2.1751, 0.0006, 0.0412, 0.0384, 16.43, 16.19, 0),
]
split = np.array([r[0] for r in ROWS])
U_sl = np.array([r[1] for r in ROWS])
usg = np.array([r[2] for r in ROWS])
re_sl = np.array([r[3] for r in ROWS])
re_sg = np.array([r[4] for r in ROWS])
fi_mult = np.array([r[5] for r in ROWS])
k_wl = np.array([r[6] for r in ROWS])
k_wg = np.array([r[7] for r in ROWS])
psi_meas = np.array([r[8] for r in ROWS])
psi_pred = np.array([r[9] for r in ROWS])
dpdz_meas = np.array([r[10] for r in ROWS])
dpdz_pred = np.array([r[11] for r in ROWS])
herr = (psi_pred - psi_meas) / psi_meas * 100
derr = (dpdz_pred - dpdz_meas) / dpdz_meas * 100
trm, tem = split == 'train', split == 'test'

# per-epoch trajectory (_vuong_eval_nn_detail.py, logged every 50 epochs,
# 5000 epochs total; thinned here to every 250 for a readable plot --
# the full run never plateaus the way the power law did, loss keeps
# creeping down the entire time, which is itself part of the story)
_EPOCHS_FULL = list(range(0, 5000, 50)) + [4999]
_LOSS_FULL = [0.033699, 0.016524, 0.012533, 0.011159, 0.010508, 0.010209, 0.010005, 0.009824, 0.009596, 0.009378, 0.009221, 0.009073, 0.0089, 0.008695, 0.00761, 0.007156, 0.00694, 0.006765, 0.006607, 0.006473, 0.006363, 0.006275, 0.006202, 0.006142, 0.006091, 0.006047, 0.006008, 0.005973, 0.005941, 0.005912, 0.005885, 0.00586, 0.005836, 0.005813, 0.005791, 0.00577, 0.005749, 0.005728, 0.005708, 0.005688, 0.005669, 0.00565, 0.005631, 0.005614, 0.005598, 0.005583, 0.005569, 0.005557, 0.005546, 0.005533, 0.005522, 0.005512, 0.005502, 0.005493, 0.005484, 0.005478, 0.005468, 0.00546, 0.005453, 0.005446, 0.005439, 0.005433, 0.00543, 0.005421, 0.005415, 0.005409, 0.005404, 0.005398, 0.005399, 0.005388, 0.005384, 0.005379, 0.005374, 0.00537, 0.005366, 0.005361, 0.005357, 0.005353, 0.005348, 0.005345, 0.005341, 0.005337, 0.005333, 0.005329, 0.005326, 0.005322, 0.005318, 0.005315, 0.005311, 0.005308, 0.005305, 0.005301, 0.005298, 0.005295, 0.005301, 0.005289, 0.005285, 0.005282, 0.005279, 0.00528, 0.005273]
_LOSS_HLD_FULL = [0.010319, 0.004248, 0.004672, 0.004447, 0.004344, 0.004294, 0.004263, 0.004232, 0.004183, 0.004183, 0.004183, 0.004171, 0.00414, 0.003947, 0.003084, 0.002912, 0.002899, 0.002868, 0.002863, 0.002864, 0.002867, 0.002871, 0.002876, 0.00288, 0.002884, 0.002887, 0.002891, 0.002894, 0.002897, 0.0029, 0.002902, 0.002904, 0.002907, 0.002908, 0.00291, 0.002911, 0.002913, 0.002914, 0.002916, 0.002917, 0.002918, 0.00292, 0.002922, 0.002924, 0.002926, 0.002928, 0.002931, 0.002933, 0.002936, 0.002937, 0.002939, 0.002941, 0.002943, 0.002945, 0.002947, 0.002948, 0.00295, 0.002951, 0.002953, 0.002954, 0.002955, 0.002957, 0.002959, 0.002959, 0.00296, 0.002961, 0.002962, 0.002962, 0.002965, 0.002964, 0.002964, 0.002965, 0.002966, 0.002966, 0.002967, 0.002967, 0.002968, 0.002968, 0.002969, 0.002969, 0.002969, 0.002969, 0.00297, 0.00297, 0.002971, 0.00297, 0.002971, 0.002971, 0.002971, 0.002972, 0.002972, 0.002972, 0.002972, 0.002972, 0.002975, 0.002972, 0.002972, 0.002972, 0.002972, 0.002973, 0.002973]
_LOSS_DPDZ_FULL = [0.02338, 0.012276, 0.007861, 0.006713, 0.006164, 0.005915, 0.005741, 0.005592, 0.005413, 0.005195, 0.005038, 0.004902, 0.004761, 0.004748, 0.004526, 0.004243, 0.004042, 0.003897, 0.003744, 0.003609, 0.003496, 0.003403, 0.003326, 0.003262, 0.003207, 0.00316, 0.003116, 0.003078, 0.003044, 0.003012, 0.002983, 0.002956, 0.00293, 0.002905, 0.002881, 0.002859, 0.002836, 0.002814, 0.002793, 0.002771, 0.00275, 0.00273, 0.00271, 0.00269, 0.002672, 0.002655, 0.002639, 0.002623, 0.00261, 0.002596, 0.002583, 0.00257, 0.002559, 0.002548, 0.002537, 0.00253, 0.002518, 0.002509, 0.0025, 0.002492, 0.002484, 0.002476, 0.002471, 0.002462, 0.002455, 0.002449, 0.002442, 0.002436, 0.002434, 0.002425, 0.002419, 0.002414, 0.002408, 0.002403, 0.002399, 0.002394, 0.002389, 0.002384, 0.00238, 0.002376, 0.002371, 0.002367, 0.002363, 0.002359, 0.002355, 0.002352, 0.002348, 0.002344, 0.00234, 0.002337, 0.002333, 0.00233, 0.002326, 0.002323, 0.002326, 0.002316, 0.002313, 0.00231, 0.002307, 0.002307, 0.002301]
EPOCHS = np.array(_EPOCHS_FULL[::5] + [_EPOCHS_FULL[-1]])
LOSS = np.array(_LOSS_FULL[::5] + [_LOSS_FULL[-1]])
LOSS_HLD = np.array(_LOSS_HLD_FULL[::5] + [_LOSS_HLD_FULL[-1]])
LOSS_DPDZ = np.array(_LOSS_DPDZ_FULL[::5] + [_LOSS_DPDZ_FULL[-1]])
BEST_EPOCH = 4999

# three-way model comparison (same reference numbers as plot_vuong_powerlaw.py)
MODELS = ['flat\n3-scalar', 'power-law\n(f_i_mult)', '27-weight\nNN (this fit)']
TRAIN_LOSS = [0.033699, 0.027961, 0.005273]
TEST_LOSS = [0.042799, 0.026653, 0.104200]
USG_CORR = [0.610, 0.220, 0.283]

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
fig1.savefig('figures/nn_closure/vuong_nn_holdup.png', dpi=160, bbox_inches='tight')

# --- figure 2: pressure gradient parity ---
fig2, a = new_ax()
sc = parity(a, dpdz_meas, dpdz_pred, '|dP/dz| [Pa/m]')
a.set_title('Pressure gradient', color=INK, loc='left', fontsize=11)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.legend(frameon=False, loc='upper left')
cb = fig2.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig2.savefig('figures/nn_closure/vuong_nn_dpdz.png', dpi=160, bbox_inches='tight')

# --- figure 3: convergence ---
fig3, a = new_ax()
a.plot(EPOCHS, LOSS, '-', color=INK, lw=2, label='total (holdup + dPdz), train')
a.plot(EPOCHS, LOSS_HLD, '-', color=S1, lw=1.5, label='holdup component')
a.plot(EPOCHS, LOSS_DPDZ, '-', color=S2, lw=1.5, label='dPdz component')
a.axhline(TEST_LOSS[2], color=S3, lw=1.5, ls='--', label=f'held-out TEST loss (final, ={TEST_LOSS[2]:.3f})')
a.text(50, TEST_LOSS[2] - 0.004, 'train never catches up to this -- never plateaus,\nunlike the power law (which flattened by epoch 150)',
       color=INK2, va='top', ha='left', fontsize=8.5)
a.set_ylim(top=TEST_LOSS[2] * 1.12)
a.set_xlabel('epoch', color=INK); a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('27-weight NN closure: train keeps improving, test does not', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8, loc='center right')
fig3.savefig('figures/nn_closure/vuong_nn_convergence.png', dpi=160, bbox_inches='tight')

# --- figure 4: holdup residual vs U_sg ---
fig4, a = new_ax()
a.axhline(0, color=INK, lw=1)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    a.scatter(usg[k], herr[k], marker=mk, s=40, color=S1, edgecolor='white', linewidth=1.2, label=sp, zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate', color=INK, loc='left', fontsize=11); a.legend(frameon=False)
fig4.savefig('figures/nn_closure/vuong_nn_residual.png', dpi=160, bbox_inches='tight')

# --- figure 5: the three learned multipliers vs. Re_SG, side by side --
# unlike the power law's single smooth curve, these are per-condition
# network outputs with no imposed functional form; k_wg specifically
# collapses toward ~0 for many conditions while f_i_mult/k_wl inflate to
# compensate (same ratio-degeneracy mechanism as the f_i_mult/k_wall
# grid-scan finding, exploited per-condition here) ---
fig5, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor='white')
panels = [(fi_mult, 'f_i_mult [-]', False), (k_wl, 'k_wl [-]', False), (k_wg, 'k_wg [-]', True)]
for a, (vals, ylabel, logy) in zip(axes, panels):
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    for sp, mk in (('train', 'o'), ('test', 's')):
        k = split == sp
        sc5 = a.scatter(re_sg[k], vals[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=44,
                         edgecolor='white', linewidth=1.1, zorder=3)
    a.set_xscale('log')
    if logy:
        a.set_yscale('log')
    apply_log_ticks(a)
    a.set_xlabel('Re_SG [-]', color=INK); a.set_ylabel(ylabel, color=INK)
fig5.suptitle('NN-learned multipliers vs. gas Reynolds number -- no smooth structure, '
              'k_wg collapses toward 0 for many conditions', color=INK, fontsize=11, x=0.02, ha='left')
cb = fig5.colorbar(sc5, ax=axes, shrink=0.85, pad=0.02); cb.set_label('U_sg [m/s]', color=INK2)
fig5.savefig('figures/nn_closure/vuong_nn_params_vs_resg.png', dpi=160, bbox_inches='tight')

# --- figure 6: model comparison bar chart (same 3-way comparison as
# plot_vuong_powerlaw.py's, included here too so this folder is self-
# contained) ---
fig6, a = new_ax(figsize=(6.6, 5.0))
x = np.arange(len(MODELS))
w = 0.32
a.bar(x - w/2, TRAIN_LOSS, width=w, color=S1, label='train loss', zorder=3)
a.bar(x + w/2, TEST_LOSS, width=w, color=S2, label='test loss', zorder=3)
for xi, (tr, te) in enumerate(zip(TRAIN_LOSS, TEST_LOSS)):
    a.text(xi - w/2, tr + 0.002, f'{tr:.3f}', ha='center', fontsize=8.5, color=INK)
    a.text(xi + w/2, te + 0.002, f'{te:.3f}', ha='center', fontsize=8.5, color=INK)
a.set_xticks(x); a.set_xticklabels(MODELS, color=INK)
a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('Model comparison: flat vs. power-law vs. NN', color=INK, loc='left', fontsize=11)
a.legend(frameon=False)
fig6.savefig('figures/nn_closure/vuong_nn_model_comparison.png', dpi=160, bbox_inches='tight')

# --- figure 7: terminal-style summary panel ---
fig7 = plt.figure(figsize=(6.6, 5.4))
ax_text = fig7.add_axes([0.06, 0.06, 0.88, 0.88])
ax_text.set_facecolor('black')
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    'Final evaluation -- Vuong (2016) 27-weight NN closure',
    '',
    '(f_i_mult, k_wl, k_wg) = NN(log Re_SL, log Re_SG)',
    '2 -> 4 (tanh) -> 3 (softplus), 27 weights, warm-started to',
    'the converged flat fit (f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413)',
    '',
    f'Dataset: 36 conditions ({trm.sum()} train / {tem.sum()} test)',
    '         0 kh-excluded, 36/36 stable',
    '',
    '                 holdup      dP/dz',
    f'  train ({trm.sum():>2})     {np.mean(np.abs(herr[trm])):5.2f}%     {np.mean(np.abs(derr[trm])):5.2f}%',
    f'  test  ({tem.sum():>2})     {np.mean(np.abs(herr[tem])):5.2f}%     {np.mean(np.abs(derr[tem])):5.2f}%',
    '',
    f'corr(U_sg, holdup_err%) = {np.corrcoef(usg, herr)[0,1]:.3f}  (flat fit: 0.610)',
    '',
    'Train loss 0.0337->0.0053 (6.4x better); test loss',
    '0.0428->0.1042 (2.4x WORSE) -- severe overfitting.',
    'One test point (U_sl=0.0100, U_sg=1.628) dominates the',
    'test dPdz loss at +81.7% error -- 27 weights found enough',
    'freedom to nearly independently fit each of 29 training',
    'points (k_wg collapses toward 0, f_i_mult/k_wl compensate)',
    'rather than learning a generalizable Re-dependence.',
    'See CLAUDE.md "NN closure overfits on 29 points...".',
]
ax_text.text(0.04, 0.96, '\n'.join(lines), transform=ax_text.transAxes,
             color='white', fontsize=9.3, fontfamily='monospace',
             va='top', ha='left', linespacing=1.5, clip_on=False)
fig7.savefig('figures/nn_closure/vuong_nn_summary.png', dpi=160, bbox_inches='tight')

print('saved: vuong_nn_{holdup,dpdz,convergence,residual,params_vs_resg,model_comparison,summary}.png')
