"""
Figure: Vuong (2016) fit quality for the power-law f_i_mult(Re_SG)
extension -- f_i_mult = f0 * exp(alpha_fi*(log(Re_SG) - mean)), k_wl/k_wg
stay flat scalars (4 free parameters total vs. the joint fit's 3, vs. the
27-weight NN's, which badly overfit -- see CLAUDE.md). Mirrors
plot_vuong_joint.py's layout for direct visual comparison across
calibration stages.

Data is not re-run here -- it's the already-verified real output of:
  - Per-condition values: _vuong_eval_powerlaw_detail.py (the DYNAMIC
    scan, re-checking kh_total), all 36 conditions, kh_total=0
    throughout, at f0=5.5650, alpha_fi=+0.6083, k_wl=2.1475, k_wg=0.5659
    (best epoch 2725 of 3000, _vuong_fit_fi_powerlaw.py).
  - Per-epoch trajectory: that same run's stdout (printed every 150
    epochs -- the fit converges by ~epoch 150 and stays dead flat
    afterward, so this granularity already captures the whole story).

Two extra panels beyond the joint-fit convention, since this fit's whole
point is a Re_SG-dependent f_i_mult and a 3-way model comparison:
  - f_i_mult vs. Re_SG: the fitted curve plus all 36 points, with a
    thin reference line showing Carraretto et al. (2020)'s own
    Re_SG^-0.53 SHAPE (not magnitude -- anchored through this fit's own
    midpoint purely to compare slope/sign, a different fluid pair) --
    makes the sign flip (alpha_fi=+0.61 here vs. their -0.53) visually
    obvious rather than just a number in a table.
  - model comparison bar chart: train/test loss for the flat 3-scalar
    fit, this power-law fit, and the (overfit) 27-weight NN closure,
    side by side.

Usage: python3 plot_vuong_powerlaw.py
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
    ("train", 0.0500, 1.620, 334216.5, 4.0575, 0.1501, 0.1641, 6.69, 5.77, 0),
    ("train", 0.0200, 1.623, 336475.9, 4.0742, 0.0833, 0.0779, 5.20, 4.03, 0),
    ("train", 0.0401, 1.625, 334644.7, 4.0607, 0.1360, 0.1380, 6.21, 5.22, 0),
    ("train", 0.0301, 1.626, 334869.9, 4.0623, 0.1101, 0.1096, 5.48, 4.64, 0),
    ("train", 0.0402, 2.379, 503018.1, 5.2031, 0.0882, 0.0921, 9.38, 9.93, 0),
    ("train", 0.0303, 2.382, 504477.5, 5.2123, 0.0672, 0.0722, 8.20, 9.05, 0),
    ("train", 0.0501, 2.396, 501681.7, 5.1947, 0.1045, 0.1101, 10.57, 10.83, 0),
    ("train", 0.0202, 2.400, 505872.8, 5.2211, 0.0552, 0.0504, 7.28, 8.15, 0),
    ("train", 0.0100, 2.417, 508457.5, 5.2373, 0.0314, 0.0266, 7.14, 7.02, 0),
    ("train", 0.0200, 2.476, 666671.0, 6.1756, 0.0500, 0.0417, 12.38, 10.69, 0),
    ("train", 0.0403, 2.479, 662840.4, 6.1540, 0.0791, 0.0776, 16.45, 12.99, 0),
    ("train", 0.0302, 2.481, 664701.8, 6.1645, 0.0657, 0.0602, 14.13, 11.93, 0),
    ("train", 0.0499, 2.482, 662066.4, 6.1496, 0.0907, 0.0930, 17.87, 13.99, 0),
    ("train", 0.0401, 2.823, 428896.0, 4.7223, 0.0805, 0.0923, 8.10, 9.95, 0),
    ("train", 0.0199, 2.828, 431225.1, 4.7379, 0.0503, 0.0503, 6.62, 8.07, 0),
    ("train", 0.0099, 2.829, 436593.7, 4.7737, 0.0294, 0.0267, 6.31, 6.95, 0),
    ("train", 0.0505, 2.844, 428728.3, 4.7212, 0.0960, 0.1111, 8.98, 10.93, 0),
    ("train", 0.0501, 3.131, 826080.5, 7.0359, 0.0728, 0.0730, 21.60, 20.67, 0),
    ("train", 0.0200, 3.132, 827311.3, 7.0422, 0.0361, 0.0322, 17.30, 16.17, 0),
    ("train", 0.0300, 3.136, 827697.7, 7.0442, 0.0497, 0.0464, 18.66, 17.86, 0),
    ("train", 0.0100, 3.138, 828318.3, 7.0475, 0.0217, 0.0170, 15.88, 14.14, 0),
    ("train", 0.0201, 3.262, 672486.1, 6.2083, 0.0330, 0.0358, 14.40, 13.88, 0),
    ("train", 0.0302, 3.264, 671981.7, 6.2055, 0.0458, 0.0515, 16.17, 15.38, 0),
    ("train", 0.0401, 3.273, 671213.1, 6.2011, 0.0603, 0.0661, 17.23, 16.73, 0),
    ("train", 0.0102, 3.288, 675998.4, 6.2280, 0.0195, 0.0191, 13.02, 12.20, 0),
    ("train", 0.0100, 4.111, 838159.6, 7.0983, 0.0169, 0.0144, 19.84, 18.38, 0),
    ("train", 0.0400, 4.387, 627874.7, 5.9544, 0.0509, 0.0586, 17.57, 20.34, 0),
    ("train", 0.0200, 4.419, 623214.5, 5.9275, 0.0294, 0.0315, 15.35, 17.11, 0),
    ("train", 0.0100, 4.439, 624813.0, 5.9367, 0.0175, 0.0165, 14.11, 15.02, 0),
    ("test", 0.0100, 1.628, 336777.2, 4.0764, 0.0559, 0.0424, 3.52, 3.38, 0),
    ("test", 0.0104, 2.457, 669201.2, 6.1898, 0.0256, 0.0231, 10.35, 9.23, 0),
    ("test", 0.0300, 2.834, 430714.3, 4.7345, 0.0654, 0.0718, 7.35, 9.08, 0),
    ("test", 0.0401, 3.134, 826810.4, 7.0396, 0.0616, 0.0600, 20.11, 19.33, 0),
    ("test", 0.0501, 3.277, 668477.3, 6.1858, 0.0740, 0.0800, 18.29, 17.95, 0),
    ("test", 0.0200, 4.114, 836019.9, 7.0872, 0.0275, 0.0274, 21.48, 21.04, 0),
    ("test", 0.0300, 4.406, 623619.8, 5.9298, 0.0412, 0.0454, 16.43, 18.78, 0),
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

F0, ALPHA_FI, K_WL, K_WG = 5.5650, 0.6083, 2.1475, 0.5659
MEAN_LOG_RE_SG = 13.2389

# per-epoch trajectory (_vuong_fit_fi_powerlaw.py, printed every 150
# epochs -- already flat by epoch 150, confirmed flat through epoch 2999)
EPOCHS = np.array([0, 150, 300, 450, 600, 750, 900, 1050, 1200, 1350,
                    1500, 1650, 1800, 1950, 2100, 2250, 2400, 2550, 2700, 2850, 2999])
F_I0 = np.array([3.9676, 5.5675, 5.5650, 5.5650, 5.5650, 5.5650, 5.5650, 5.5650,
                  5.5650, 5.5650, 5.5655, 5.5650, 5.5650, 5.5670, 5.5650, 5.5650,
                  5.5630, 5.5650, 5.5650, 5.5625, 5.5650])
ALPHA_FI_T = np.array([0.0200, 0.6084, 0.6083, 0.6083, 0.6083, 0.6083, 0.6083, 0.6083,
                        0.6083, 0.6083, 0.6083, 0.6083, 0.6083, 0.6083, 0.6083, 0.6083,
                        0.6083, 0.6083, 0.6083, 0.6083, 0.6083])
LOSS = np.array([0.033699, 0.027961, 0.027961, 0.027961, 0.027961, 0.027961, 0.027961, 0.027961,
                  0.027961, 0.027961, 0.027961, 0.027961, 0.027961, 0.027961, 0.027961, 0.027961,
                  0.027961, 0.027961, 0.027961, 0.027961, 0.027961])
LOSS_HLD = np.array([0.010319, 0.010174, 0.010173, 0.010173, 0.010173, 0.010173, 0.010173, 0.010173,
                      0.010173, 0.010173, 0.010172, 0.010173, 0.010173, 0.010173, 0.010173, 0.010173,
                      0.010175, 0.010173, 0.010173, 0.010173, 0.010173])
LOSS_DPDZ = np.array([0.023380, 0.017787, 0.017788, 0.017788, 0.017788, 0.017788, 0.017788, 0.017788,
                       0.017788, 0.017788, 0.017789, 0.017788, 0.017788, 0.017788, 0.017788, 0.017788,
                       0.017786, 0.017788, 0.017788, 0.017788, 0.017788])
BEST_EPOCH = 2725

# three-way model comparison (CLAUDE.md "Nondimensionalization + ..." and
# the power-law fit's own stdout)
MODELS = ['flat\n3-scalar', 'power-law\n(this fit)', '27-weight\nNN (overfit)']
TRAIN_LOSS = [0.033699, 0.027961, 0.005263]
TEST_LOSS = [0.042799, 0.026653, 0.101338]
USG_CORR = [0.610, 0.220, 0.281]

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
fig1.savefig('figures/powerlaw_fit/vuong_powerlaw_holdup.png', dpi=160, bbox_inches='tight')

# --- figure 2: pressure gradient parity ---
fig2, a = new_ax()
sc = parity(a, dpdz_meas, dpdz_pred, '|dP/dz| [Pa/m]')
a.set_title('Pressure gradient', color=INK, loc='left', fontsize=11)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.legend(frameon=False, loc='upper left')
cb = fig2.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig2.savefig('figures/powerlaw_fit/vuong_powerlaw_dpdz.png', dpi=160, bbox_inches='tight')

# --- figure 3: convergence ---
fig3, a = new_ax()
a.plot(EPOCHS, LOSS, '-o', color=INK, lw=2, ms=5, label='total (holdup + dPdz)')
a.plot(EPOCHS, LOSS_HLD, '-o', color=S1, lw=1.5, ms=4, label='holdup component')
a.plot(EPOCHS, LOSS_DPDZ, '-o', color=S2, lw=1.5, ms=4, label='dPdz component')
a.axvline(BEST_EPOCH, color=INK2, lw=1, ls='--')
a.text(BEST_EPOCH, a.get_ylim()[1], f' best epoch {BEST_EPOCH}\n alpha_fi={ALPHA_FI:+.3f}',
       color=INK2, va='top', fontsize=9)
a.set_xlabel('epoch', color=INK); a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('Power-law fit convergence (f0, alpha_fi, k_wl, k_wg)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8)
fig3.savefig('figures/powerlaw_fit/vuong_powerlaw_convergence.png', dpi=160, bbox_inches='tight')

# --- figure 4: holdup residual vs U_sg ---
fig4, a = new_ax()
a.axhline(0, color=INK, lw=1)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    a.scatter(usg[k], herr[k], marker=mk, s=40, color=S1, edgecolor='white', linewidth=1.2, label=sp, zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate', color=INK, loc='left', fontsize=11); a.legend(frameon=False)
fig4.savefig('figures/powerlaw_fit/vuong_powerlaw_residual.png', dpi=160, bbox_inches='tight')

# --- figure 5: f_i_mult vs Re_SG -- the fitted power law itself, plus a
# shape-only (not magnitude) reference line at Carraretto's exponent,
# anchored through this fit's own geometric-mean point so only the SLOPE
# (sign/steepness) is being compared, not the absolute f_i_mult value ---
fig5, a = new_ax()
re_grid = np.logspace(np.log10(re_sg.min()) - 0.05, np.log10(re_sg.max()) + 0.05, 200)
fit_curve = F0 * np.exp(ALPHA_FI * (np.log(re_grid) - MEAN_LOG_RE_SG))
a.plot(re_grid, fit_curve, color=INK, lw=2, label=f'fitted: f0={F0:.2f}, alpha_fi={ALPHA_FI:+.3f}', zorder=2)
anchor_re, anchor_f = np.exp(MEAN_LOG_RE_SG), F0
carraretto_curve = anchor_f * (re_grid / anchor_re) ** (-0.53)
a.plot(re_grid, carraretto_curve, color=S2, lw=1.5, ls='--',
       label='Carraretto et al. 2020 exponent (shape only, alpha=-0.53;\nwater-air, not magnitude-comparable)', zorder=2)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    sc5 = a.scatter(re_sg[k], fi_mult[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=46,
                     edgecolor='white', linewidth=1.2, label=f'{sp} (n={k.sum()})', zorder=3)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.set_xlabel('Re_SG [-]', color=INK); a.set_ylabel('f_i_mult [-]', color=INK)
a.set_title('Interfacial friction multiplier vs. gas Reynolds number', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8, loc='upper left')
cb = fig5.colorbar(sc5, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig5.savefig('figures/powerlaw_fit/vuong_powerlaw_fi_vs_resg.png', dpi=160, bbox_inches='tight')

# --- figure 6: model comparison bar chart ---
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
fig6.savefig('figures/powerlaw_fit/vuong_powerlaw_model_comparison.png', dpi=160, bbox_inches='tight')

# --- figure 7: terminal-style summary panel ---
fig7 = plt.figure(figsize=(6.6, 5.2))
ax_text = fig7.add_axes([0.06, 0.06, 0.88, 0.88])
ax_text.set_facecolor('black')
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    'Final evaluation -- Vuong (2016) power-law f_i_mult(Re_SG) fit',
    '',
    'f_i_mult = f0*exp(alpha_fi*(log(Re_SG)-mean))',
    f'Fitted: f0={F0:.4f}  alpha_fi={ALPHA_FI:+.4f}  k_wl={K_WL:.4f}  k_wg={K_WG:.4f}',
    '  (true jax.grad through the equilibrium path; converged by',
    f'   ~epoch 150, dead flat through epoch 2999; best epoch {BEST_EPOCH})',
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
    'vs. flat 3-scalar fit: train 0.0337->0.0280, test 0.0428->0.0267',
    'Sign surprise: alpha_fi is POSITIVE here; Carraretto et al.',
    '(water-air) found -0.53 -- opposite direction, not just a',
    'different magnitude. See CLAUDE.md for discussion.',
]
ax_text.text(0.04, 0.96, '\n'.join(lines), transform=ax_text.transAxes,
             color='white', fontsize=9.5, fontfamily='monospace',
             va='top', ha='left', linespacing=1.5, clip_on=False)
fig7.savefig('figures/powerlaw_fit/vuong_powerlaw_summary.png', dpi=160, bbox_inches='tight')

print('saved: vuong_powerlaw_{holdup,dpdz,convergence,residual,fi_vs_resg,model_comparison,summary}.png')
