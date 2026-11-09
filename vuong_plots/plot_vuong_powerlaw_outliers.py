"""
Figure: Vuong (2016) experimental operating map (U_sl vs. U_sg), with the
power-law fit's +-20% outliers highlighted -- the 2 conditions with
|holdup error|>20% in one color, the 7 conditions with |dPdz error|>20%
in a second color, everything else (within +-20% on both) in muted gray.
Zero overlap between the two outlier sets (see CLAUDE.md "Power-law
+-20% outliers do not overlap between holdup and dPdz").

Data is the same 36-row ROWS table as plot_vuong_powerlaw.py (no rerun).

Usage: python3 plot_vuong_powerlaw_outliers.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, MUTED = '#2a78d6', '#eb6834', '#b7b6b2'

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
psi_meas = np.array([r[5] for r in ROWS]); psi_pred = np.array([r[6] for r in ROWS])
dpdz_meas = np.array([r[7] for r in ROWS]); dpdz_pred = np.array([r[8] for r in ROWS])
herr = (psi_pred - psi_meas) / psi_meas * 100
derr = (dpdz_pred - dpdz_meas) / dpdz_meas * 100

hld_out = np.abs(herr) > 20
dpdz_out = np.abs(derr) > 20
both_out = hld_out & dpdz_out
neither = ~hld_out & ~dpdz_out
assert both_out.sum() == 0, "expected zero overlap between the two outlier sets"

fig, a = plt.subplots(figsize=(6.8, 5.6), facecolor='white')
a.set_facecolor('white'); a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
for s in ('top', 'right'): a.spines[s].set_visible(False)
for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
a.tick_params(colors=INK2)

groups = [
    (neither, MUTED, 'within ±20% (both)'),
    (hld_out, S1, 'holdup outlier (>20%)'),
    (dpdz_out, S2, 'dPdz outlier (>20%)'),
]
for mask, color, label in groups:
    for sp, mk in (('train', 'o'), ('test', 's')):
        k = mask & (split == sp)
        if not k.any():
            continue
        a.scatter(U_sl[k], usg[k], marker=mk, s=70, color=color, edgecolor='white',
                  linewidth=1.3, zorder=3, label=f'{label}, {sp}' if color != MUTED else None)

# single combined legend entry for the "neither" group (train+test same color/meaning)
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=MUTED, markeredgecolor='white',
           markersize=9, label=f'within ±20% (both), n={int(neither.sum())}'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=S1, markeredgecolor='white',
           markersize=9, label=f'holdup outlier (>20%), n={int(hld_out.sum())}'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=S2, markeredgecolor='white',
           markersize=9, label=f'dPdz outlier (>20%), n={int(dpdz_out.sum())}'),
    Line2D([0], [0], marker='', color='w', label='(circle=train, square=test)'),
]
a.set_xscale('log')
a.set_xlabel('U_sl [m/s]', color=INK)
a.set_ylabel('U_sg [m/s]', color=INK)
a.set_title('Vuong (2016) operating map -- power-law fit ±20% outliers', color=INK, loc='left', fontsize=11)
a.legend(handles=legend_handles, frameon=False, fontsize=8.5, loc='upper left')
fig.savefig('figures/powerlaw_fit/vuong_powerlaw_outliers_map.png', dpi=160, bbox_inches='tight')
print('saved: vuong_powerlaw_outliers_map.png')
