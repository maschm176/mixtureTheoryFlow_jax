"""
Figure set for the Fan (2005) "Option B" NN-for-f_i_mult-only fit
(_fan_fit_nn_fi_only.py) -- the Fan-specific counterpart of
vuong_plots/plot_vuong_nn_fi_only.py, same architecture (13-weight
NN(log Re_SL, log Re_SG) -> f_i_mult, k_wl/k_wg flat scalars, trained
jointly through the closed-form Taitel-Dukler equilibrium path).

REWRITTEN (2026-11-14) after two changes superseded the original version
of this script:
  1. Dataset expanded from 11 kh-clean HORIZONTAL-only conditions to 49
     kh-clean conditions (11 horizontal + 38 inclined, see _fan_data.py)
     -- large enough for a real fixed 39/10 train/test split
     (_fan_data.build_split), so the old leave-one-out-CV workaround
     (needed only because 11 points couldn't support a fixed split) is
     gone, replaced by an ordinary train/test comparison, matching
     Vuong's methodology directly.
  2. A genuine bug was found and fixed in the SOLVER itself (not this
     fit): `_bisect_ht`'s plain bisection silently assumed
     taitel_dukler_residual(ht) has exactly one root, which is false at
     low f_i_mult + uphill beta + very low U_sl -- the force balance can
     be S-shaped with THREE roots (a saddle-node/fold bifurcation).
     Before the fix, this caused ONE held-out test condition
     (U_sl=0.00308, U_sg=20.04, angle=+2) to blow up to 1610% holdup
     error and two unmeasured training conditions to predict
     nonsensical holdup (~0.095-0.196 vs. neighbors' ~0.005-0.01) -- not
     random noise, but the optimizer's own f_i_mult(Re_SG) surface
     dipping (between training points) into the region where the
     thin-film branch had already vanished. The fix (coarse pre-scan +
     always track the LOWEST root = the thin-film branch, see
     mixtureTheoryFlow_jax_solver_water_gas.py's _bisect_ht docstring)
     is a verified no-op everywhere previously validated (Fan/MDPI/Vuong
     horizontal, Vuong's adopted calibration -- bit-identical, ~1e-15
     diffs) and changed results ONLY in this newly-discovered corner.
     Refitting after the fix: the catastrophic outlier disappeared
     entirely (test holdup error 223% -> 21%, no outlier; the optimizer
     settled into a different, better basin once gradients near that
     corner stopped being corrupted by the old solver's inconsistent
     branch selection) -- not just a diagnostic nicety, a real
     improvement to the fitted result.

DOES NOT READ FROM, WRITE TO, OR OVERWRITE ANYTHING in vuong_plots/ or
figures/nn_fi_only/ -- entirely separate script, entirely separate output
directory (figures/fan_nn_fi_only/).

NOT YET RUN as of this rewrite -- data below is the verified real stdout
of _fan_fit_nn_fi_only.py's post-solver-fix run (saved at
/tmp/fan_nn_fi_only_run2.log), transcribed here exactly as printed, same
as every other *_fit_*.py -> plot_*.py handoff in this project. Run
`python3 fan_plots/plot_fan_nn_fi_only.py` (from repo root) whenever
ready to actually generate the PNGs.

Usage: python3 fan_plots/plot_fan_nn_fi_only.py (run from repo root)
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


def new_ax(figsize=(6.2, 5.6)):
    fig, a = plt.subplots(figsize=figsize, facecolor='white')
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    return fig, a


# fields: (split, U_sl, U_sg, Re_SG, f_i_mult, psi_meas, psi_pred, dpdz_meas, dpdz_pred, sigma)
# from _fan_fit_nn_fi_only.py's post-solver-fix run (/tmp/fan_nn_fi_only_run2.log):
# best_epoch=2999 train_loss=0.029812 (hld=0.017348 dpdz=0.012464)
# test_loss=0.081156 (holdup=0.070974 dpdz=0.010182); k_wl=2.3110 k_wg=1.2617
ROWS = [
    ("train", 0.013733, 15.080, 93204.5, 10.3862, 0.0100, 0.0127, 221.87, -208.35, 0.072),
    ("train", 0.004602, 15.170, 94188.8, 1.6319, 0.0100, 0.0091, 160.97, -122.02, 0.072),
    ("train", 0.002296, 15.010, 92348.2, 0.7127, None, 0.0086, 124.84, -110.23, 0.072),
    ("train", 0.000746, 14.990, 92648.2, 0.4846, None, 0.0054, 125.19, -108.40, 0.072),
    ("train", 0.014248, 20.070, 124612.4, 5.9995, 0.0100, 0.0130, 281.32, -281.05, 0.072),
    ("train", 0.003010, 19.830, 123122.3, 0.2163, None, 0.0114, 152.16, -175.48, 0.072),
    ("train", 0.001027, 19.750, 122625.6, 0.5279, None, 0.0051, 143.73, -179.84, 0.072),
    ("train", 0.001289, 24.720, 153483.7, 0.0016, None, 0.0092, 244.15, -256.37, 0.072),
    ("train", 0.018710, 15.110, 102345.1, 6.7166, 0.0200, 0.0199, 203.70, -200.36, 0.072),
    ("train", 0.014352, 15.240, 93763.3, 9.8630, 0.0200, 0.0135, 196.83, -209.47, 0.072),
    ("train", 0.010759, 15.080, 93204.5, 8.6163, 0.0100, 0.0111, 171.88, -188.55, 0.072),
    ("train", 0.004571, 15.090, 92840.4, 1.5934, 0.0100, 0.0097, 139.05, -120.28, 0.072),
    ("train", 0.002287, 15.060, 92230.8, 0.7100, 0.0100, 0.0093, 108.91, -110.91, 0.072),
    ("train", 0.006221, 19.810, 134179.7, 2.0333, 0.0100, 0.0099, 238.06, -220.12, 0.072),
    ("train", 0.003118, 20.030, 135669.9, 0.3941, 0.0100, 0.0105, 183.14, -195.41, 0.072),
    ("train", 0.001059, 20.050, 135805.3, 0.4308, None, 0.0056, 161.66, -196.95, 0.072),
    ("train", 0.001289, 24.750, 195580.0, 0.1402, None, 0.0073, 280.93, -317.12, 0.072),
    ("train", 0.018551, 15.110, 109594.5, 6.7389, 0.0200, 0.0196, 202.08, -211.64, 0.072),
    ("train", 0.013639, 15.090, 109449.4, 5.4612, 0.0200, 0.0163, 191.78, -189.83, 0.072),
    ("train", 0.010954, 15.120, 109667.0, 4.5605, 0.0200, 0.0145, 173.70, -176.98, 0.072),
    ("train", 0.007766, 14.930, 107867.6, 4.8635, 0.0100, 0.0104, 183.17, -169.85, 0.072),
    ("train", 0.004530, 14.970, 107734.1, 1.5763, 0.0100, 0.0096, 165.56, -134.58, 0.072),
    ("train", 0.002289, 14.850, 101841.3, 0.7499, 0.0100, 0.0097, 118.79, -119.26, 0.072),
    ("train", 0.006065, 19.860, 148530.7, 2.2388, 0.0100, 0.0091, 269.21, -242.54, 0.072),
    ("train", 0.001058, 19.960, 143082.2, 0.1727, None, 0.0087, 201.96, -199.67, 0.072),
    ("train", 0.001297, 24.960, 185264.2, 0.1350, None, 0.0083, 294.20, -306.07, 0.072),
    ("train", 0.018268, 14.900, 102184.2, 6.6458, 0.0200, 0.0205, 216.98, -198.56, 0.072),
    ("train", 0.007542, 14.900, 101763.7, 5.8393, 0.0100, 0.0096, 158.98, -170.52, 0.072),
    ("train", 0.004575, 14.940, 102458.5, 1.6886, 0.0100, 0.0100, 154.00, -130.88, 0.072),
    ("train", 0.002342, 15.150, 104326.3, 0.7739, 0.0100, 0.0105, 140.13, -124.87, 0.072),
    ("train", 0.006200, 19.890, 139773.7, 2.1413, 0.0100, 0.0099, 267.26, -231.15, 0.072),
    ("train", 0.001031, 20.100, 142383.9, 0.2270, None, 0.0090, 203.79, -201.67, 0.072),
    ("train", 0.001306, 24.820, 184225.1, 0.1352, None, 0.0097, 329.49, -302.94, 0.072),
    ("train", 0.007576, 15.070, 103775.4, 6.1137, 0.0100, 0.0094, 173.67, -177.94, 0.072),
    ("train", 0.004615, 14.990, 102801.4, 1.7082, 0.0100, 0.0106, 152.23, -132.63, 0.072),
    ("train", 0.000781, 15.210, 103880.9, 0.4922, None, 0.0090, 110.72, -122.13, 0.072),
    ("train", 0.006240, 20.150, 143875.5, 2.2667, 0.0100, 0.0097, 294.91, -242.14, 0.072),
    ("train", 0.001072, 20.050, 142595.6, 0.1800, None, 0.0196, 182.35, -200.66, 0.072),
    ("train", 0.001323, 24.920, 184264.0, 0.1370, None, 0.0122, 304.02, -304.13, 0.072),
    ("test", 0.018111, 14.940, 91917.5, 9.5048, 0.0200, 0.0171, 226.59, -206.35, 0.072),
    ("test", 0.006331, 20.090, 124736.6, 1.9847, 0.0100, 0.0101, 219.00, -209.67, 0.072),
    ("test", 0.007687, 15.070, 92717.3, 3.6226, 0.0100, 0.0122, 154.91, -141.92, 0.072),
    ("test", 0.010054, 19.890, 134721.6, 4.5383, 0.0100, 0.0106, 272.69, -265.21, 0.072),
    ("test", 0.000790, 14.860, 101909.9, 0.4921, None, 0.0065, 100.38, -116.75, 0.072),
    ("test", 0.003120, 20.260, 144089.1, 0.4147, 0.0100, 0.0110, 247.31, -208.28, 0.072),
    ("test", 0.000809, 15.160, 103967.3, 0.4947, None, 0.0073, 117.66, -121.43, 0.072),
    ("test", 0.003058, 19.960, 140828.9, 0.3904, 0.0100, 0.0128, 235.56, -201.53, 0.072),
    ("test", 0.010794, 15.140, 103830.1, 5.6671, 0.0100, 0.0136, 192.73, -182.07, 0.072),
    ("test", 0.003079, 20.040, 141958.9, 0.3987, 0.0100, 0.0152, 224.31, -204.52, 0.072),
]
split = np.array([r[0] for r in ROWS])
U_sl = np.array([r[1] for r in ROWS])
usg = np.array([r[2] for r in ROWS])
re_sg = np.array([r[3] for r in ROWS])
fi_mult = np.array([r[4] for r in ROWS])
psi_meas = np.array([r[5] if r[5] is not None else np.nan for r in ROWS])
psi_pred = np.array([r[6] for r in ROWS])
dpdz_meas = np.array([r[7] for r in ROWS])
dpdz_pred = np.abs(np.array([r[8] for r in ROWS]))  # solver sign convention
has_psi = ~np.isnan(psi_meas)
trm, tem = split == 'train', split == 'test'
herr = (psi_pred - psi_meas) / psi_meas * 100  # NaN where no measured holdup
derr = (dpdz_pred - dpdz_meas) / dpdz_meas * 100

K_WL, K_WG = 2.3110, 1.2617
WARM_FI, WARM_KWL, WARM_KWG = 0.8939, 0.5158, 1.3607  # from the (solver-fix-unaffected) flat warm-start fit

# per-epoch trajectory, printed every 500 epochs (_fan_fit_nn_fi_only.py stdout, post-fix run)
EPOCHS = np.array([0, 500, 1000, 1500, 2000, 2500, 2999])
LOSS = np.array([0.192877, 0.049138, 0.061761, 0.034476, 0.033916, 0.032232, 0.029812])
LOSS_HLD = np.array([0.138665, 0.028506, 0.027013, 0.021033, 0.020337, 0.019154, 0.017348])
LOSS_DPDZ = np.array([0.054213, 0.020632, 0.034749, 0.013443, 0.013580, 0.013078, 0.012464])
BEST_EPOCH = 2999
TRAIN_LOSS, TEST_LOSS = 0.029812, 0.081156
TEST_HLD_LOSS, TEST_DPDZ_LOSS = 0.070974, 0.010182

# Vuong power-law reference (CLAUDE.md "Power-law closure"), for trend
# comparison -- NOTE: absolute f_i_mult scale differs between datasets
# (different baseline closures / warm starts), so only the SIGN/SLOPE is
# comparable directly, not the absolute curve.
VUONG_F0, VUONG_ALPHA, VUONG_MEAN_LOG_RESG = 5.5650, 0.6083, 13.2389
CARRARETTO_ALPHA = -0.53  # f_i = 2.25 * Re_SG^-0.53, Carraretto et al. (2020)

# Weber number at predicted equilibrium relative velocity (u_rel = U_sg/(1-psi_pred)
# - U_sl/psi_pred), same definition/methodology as the Vuong/MDPI check
# (CLAUDE.md "Weber number check..."), recomputed for the expanded 49-row
# dataset at the post-solver-fix fitted parameters. D=0.0508 for every
# Fan condition here; sigma=0.072 N/m (air-water).
FAN_WE = np.array([
    311.2, 340.2, 340.3, 344.6, 574.5, 608.2, 599.3, 955.4, 354.9, 318.3,
    315.1, 335.4, 342.4, 636.0, 673.7, 675.6, 1210.7, 379.4, 381.4, 385.8,
    371.4, 385.8, 373.5, 701.9, 717.8, 1160.9, 351.6, 347.1, 367.1, 391.9,
    665.5, 720.3, 1153.1, 357.3, 371.3, 397.7, 693.0, 739.7, 1166.4,
    307.6, 600.5, 329.0, 621.3, 377.4, 725.6, 394.1, 704.1, 363.2, 718.8,
])
VUONG_WE = dict(median=958, lo=336, hi=1909)
MDPI_WE = dict(median=3.7, lo=0.8, hi=12.4)

norm = plt.Normalize(usg.min(), usg.max())


def parity(a, meas, pred, name, mask=None):
    if mask is None:
        mask = np.ones_like(meas, dtype=bool)
    m, p = meas[mask], pred[mask]
    lo, hi = min(m.min(), p.min()) * 0.8, max(m.max(), p.max()) * 1.2
    xs = np.array([lo, hi])
    a.plot(xs, xs, color=INK, lw=1.2)
    for pct, ls in ((10, '--'), (20, ':')):
        a.plot(xs, xs * (1 + pct / 100), color=INK2, lw=1, ls=ls)
        a.plot(xs, xs * (1 - pct / 100), color=INK2, lw=1, ls=ls)
    for sp, mk in (('train', 'o'), ('test', 's')):
        k = mask & (split == sp)
        if not k.any():
            continue
        sc = a.scatter(meas[k], pred[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=46,
                        edgecolor='white', linewidth=1.2, label=f'{sp} (n={k.sum()})', zorder=3)
    a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_aspect('equal')
    a.set_xlabel(f'measured {name}', color=INK); a.set_ylabel(f'predicted {name}', color=INK)
    return sc


# --- figure 1: holdup parity (n=33 with measured HL, train+test) ---
fig1, a = new_ax()
sc = parity(a, psi_meas, psi_pred, 'liquid holdup', mask=has_psi)
a.set_title('Holdup, train/test split (dashed +-10%, dotted +-20%)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, loc='upper left')
cb = fig1.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig1.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_holdup.png', dpi=160, bbox_inches='tight')

# --- figure 2: pressure gradient parity (n=49) ---
fig2, a = new_ax()
sc = parity(a, dpdz_meas, dpdz_pred, '|dP/dz| [Pa/m]')
a.set_title('Pressure gradient, train/test split', color=INK, loc='left', fontsize=11)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.legend(frameon=False, loc='upper left')
cb = fig2.colorbar(sc, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig2.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_dpdz.png', dpi=160, bbox_inches='tight')

# --- figure 3: convergence ---
fig3, a = new_ax()
a.plot(EPOCHS, LOSS, '-o', color=INK, lw=2, ms=5, label='total (holdup + dPdz), train')
a.plot(EPOCHS, LOSS_HLD, '-o', color=S1, lw=1.5, ms=4, label='holdup component')
a.plot(EPOCHS, LOSS_DPDZ, '-o', color=S2, lw=1.5, ms=4, label='dPdz component')
a.axhline(TEST_LOSS, color=S3, lw=1.5, ls='--', label=f'held-out TEST loss (final, ={TEST_LOSS:.3f})')
a.axvline(BEST_EPOCH, color=INK2, lw=1, ls=':')
a.set_xlabel('epoch', color=INK); a.set_ylabel('loss (mean squared rel. error)', color=INK)
a.set_title('Post-solver-fix fit converges cleanly, no outlier-driven blowup', color=INK, loc='left', fontsize=10.5)
a.legend(frameon=False, fontsize=8.5)
fig3.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_convergence.png', dpi=160, bbox_inches='tight')

# --- figure 4: f_i_mult vs Re_SG, Fan (train+test) vs. Vuong's power-law (absolute scale) ---
fig4, a = new_ax()
re_grid = np.logspace(np.log10(re_sg.min()) - 0.1, np.log10(re_sg.max()) + 0.1, 200)
vuong_curve = VUONG_F0 * np.exp(VUONG_ALPHA * (np.log(re_grid) - VUONG_MEAN_LOG_RESG))
a.plot(re_grid, vuong_curve, color=S3, lw=1.5, ls='--',
       label=f'Vuong power-law fit (ref.): alpha_fi={VUONG_ALPHA:+.3f}', zorder=2)
for sp, mk in (('train', 'o'), ('test', 's')):
    k = split == sp
    sc4 = a.scatter(re_sg[k], fi_mult[k], c=usg[k], cmap=seq, norm=norm, marker=mk, s=46,
                     edgecolor='white', linewidth=1.2, label=f'Fan NN f_i_mult, {sp} (n={k.sum()})', zorder=3)
a.set_xscale('log'); a.set_yscale('log')
apply_log_ticks(a)
a.set_xlabel('Re_SG [-]', color=INK); a.set_ylabel('f_i_mult [-]', color=INK)
a.set_title('Fan f_i_mult vs. Re_SG (post-solver-fix)', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, fontsize=8, loc='upper right')
cb = fig4.colorbar(sc4, ax=a, shrink=0.85, pad=0.03); cb.set_label('U_sg [m/s]', color=INK2)
fig4.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_vs_resg.png', dpi=160, bbox_inches='tight')

# --- figure 5: Weber-number regime comparison, Fan vs Vuong vs MDPI ---
fig5, a = new_ax(figsize=(7.0, 5.0))
datasets = ['MDPI\n(water-air)', 'Fan (2005)\n(water-air)', 'Vuong (2016)\n(oil-nitrogen)']
medians = [MDPI_WE['median'], np.median(FAN_WE), VUONG_WE['median']]
los = [MDPI_WE['lo'], FAN_WE.min(), VUONG_WE['lo']]
his = [MDPI_WE['hi'], FAN_WE.max(), VUONG_WE['hi']]
colors = [S2, S1, S3]
y = np.arange(len(datasets))
for yi, (lo, hi, med, c) in enumerate(zip(los, his, medians, colors)):
    a.plot([lo, hi], [yi, yi], color=c, lw=6, alpha=0.35, solid_capstyle='round')
    a.plot(med, yi, 'o', color=c, ms=11, markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    a.text(hi * 1.15, yi, f'median={med:.0f}', va='center', color=INK, fontsize=9)
a.axvline(1.0, color=INK2, lw=1, ls=':')
a.text(1.0, -0.55, 'We=1 (surface tension <-> inertia)',
       color=INK2, fontsize=8, ha='center', va='top')
a.set_xscale('log')
a.set_xlim(0.3, 3000)
a.xaxis.set_major_locator(LogLocator(base=10.0))
a.set_yticks(y); a.set_yticklabels(datasets, color=INK)
a.set_ylim(-0.85, len(datasets) - 0.4)
a.set_xlabel('Weber number We = rho_g * u_rel^2 * D / sigma  [-]', color=INK)
a.set_title('Fan sits in the SAME high-We regime as Vuong, not MDPI',
            color=INK, loc='left', fontsize=10.5)
fig5.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_weber_regime.png', dpi=160, bbox_inches='tight')

# --- figure 6: train/test holdup residual vs U_sg (replaces the old LOO-CV
# figure now that a real held-out test split exists) ---
fig6, a = new_ax(figsize=(7.2, 5.2))
a.axhline(0, color=INK, lw=1)
for sp, mk, c in (('train', 'o', S1), ('test', 's', S2)):
    k = has_psi & (split == sp)
    a.scatter(usg[k], herr[k], marker=mk, s=46, color=c, edgecolor='white', linewidth=1.2,
              label=f'{sp} (n={k.sum()}, mean|err%|={np.mean(np.abs(herr[k])):.1f})', zorder=3)
a.set_xlabel('U_sg [m/s]', color=INK); a.set_ylabel('holdup error [%]', color=INK)
a.set_title('Holdup residual vs. gas rate -- train/test, no outlier',
            color=INK, loc='left', fontsize=10.5)
a.legend(frameon=False, fontsize=9)
fig6.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_residual.png', dpi=160, bbox_inches='tight')

# --- figure 7: terminal-style summary panel ---
fig7 = plt.figure(figsize=(7.0, 6.4))
ax_text = fig7.add_axes([0.06, 0.06, 0.88, 0.88])
ax_text.set_facecolor('black')
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    'Final evaluation -- Fan (2005) "Option B" hybrid',
    '(same architecture as Vuong: 13-wt NN -> f_i_mult, flat k_wl/k_wg)',
    '',
    f'Dataset: 49 kh-clean conditions (11 horizontal + 38 inclined),',
    f'{trm.sum()} train / {tem.sum()} test ({has_psi[trm].sum()}/{has_psi[tem].sum()} with holdup).',
    f'Warm start (flat fit): f_i_mult={WARM_FI}, k_wl={WARM_KWL}, k_wg={WARM_KWG}',
    f'Fitted: k_wl={K_WL:.4f}  k_wg={K_WG:.4f}',
    '',
    'SOLVER FIX (mixtureTheoryFlow_jax_solver_water_gas.py, _bisect_ht):',
    'the equilibrium force balance can have 3 roots (fold bifurcation)',
    'at low f_i_mult + uphill + very low U_sl. Fixed by always tracking',
    'the lowest (thin-film) root. Verified a no-op everywhere previously',
    'validated (beta=0 all datasets, Vuong adopted calib); this fit is',
    'the first real-data case it changed the OUTCOME of, not just a',
    'diagnostic number -- the catastrophic test outlier is gone because',
    'training itself converged to a different, better basin once the',
    'solver stopped jumping branches inconsistently near that corner.',
    '',
    '           n    holdup err%    dPdz err%',
    f'  train  ({trm.sum():>2})    {np.mean(np.abs(herr[trm & has_psi])):5.2f}%        {np.mean(np.abs(derr[trm])):5.2f}%',
    f'  test   ({tem.sum():>2})    {np.mean(np.abs(herr[tem & has_psi])):5.2f}%        {np.mean(np.abs(derr[tem])):5.2f}%',
    '',
    'dPdz generalizes about as well as it fits in-sample; holdup shows',
    'a real but modest train/test gap -- no catastrophic outlier.',
    '',
    'Weber number (at predicted equilibrium u_rel):',
    f'  Fan median={np.median(FAN_WE):.0f} (range {FAN_WE.min():.0f}-{FAN_WE.max():.0f})',
    f'  Vuong median={VUONG_WE["median"]} (range {VUONG_WE["lo"]}-{VUONG_WE["hi"]})  -- SAME regime as Fan',
    f'  MDPI  median={MDPI_WE["median"]} (range {MDPI_WE["lo"]}-{MDPI_WE["hi"]})  -- different, near We~1',
]
ax_text.text(0.03, 0.97, '\n'.join(lines), transform=ax_text.transAxes,
             color='white', fontsize=8.3, fontfamily='monospace',
             va='top', ha='left', linespacing=1.4, clip_on=False)
fig7.savefig('figures/fan_nn_fi_only/fan_nn_fi_only_summary.png', dpi=160, bbox_inches='tight')

print('saved: figures/fan_nn_fi_only/fan_nn_fi_only_{holdup,dpdz,convergence,vs_resg,weber_regime,residual,summary}.png')
