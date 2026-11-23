"""
Summary table (saved as a figure): the closure values (f_i_mult, k_wl, k_wg) from calibrating on SW only, on ST only, and from the
pooled SW+ST NN fit. Values are transcribed from the fit outputs (all equilibrium-path fits):
  SW only  : flat 3-scalar production fit (36 Vuong SW rows, 29 train / 7 test, horizontal) -- 3.8893 / 1.5419 / 0.8413,
             train loss 0.0337 / test 0.0428 (CLAUDE.md, equilibrium-path true-grad fit)
  ST only  : flat 3-scalar fit WITH beta=+2 deg threaded in (21 ST rows, 15/6) -- st_fit_beta_result.pkl:
             9.2809 / 1.6477 / 0.7736, train 0.023027 / test 0.062817
  Pooled   : NN for f_i_mult (2 inputs: log Re_SL, log Re_SG), flat k_wl/k_wg, beta-aware (57 rows, 44/13) --
             pooled_beta_result_base.pkl: k_wl 1.5357, k_wg 0.8567, train 0.019228 / test 0.045510; f_i_mult is per-row:
             SW rows median 4.00 (2.42-5.47), ST rows median 9.22 (4.01-9.22)
Output: figures/closure_values_table.png.  Usage (repo root): python3 vuong_plots/plot_closure_values_table.py
"""
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID, HDR = '#0b0b0b', '#52514e', '#e4e3df', '#f3f2ee'
os.makedirs('figures', exist_ok=True)

cols = ['Calibration', 'Data (rows; train / test)', 'f_i_mult', 'k_wl', 'k_wg', 'Train loss', 'Test loss']
rows = [
    ['SW only\nflat 3 scalars', '36 horizontal SW rows\n(29 / 7)', '3.889', '1.542', '0.841', '0.0337', '0.0428'],
    ['ST only\nflat 3 scalars, beta = +2 deg', '21 inclined ST rows\n(15 / 6)', '9.281', '1.648', '0.774', '0.0230', '0.0628'],
    ['SW + ST combined\npooled NN for f_i_mult\n(beta-aware, 2 inputs)', '57 rows: 36 SW + 21 ST\n(44 / 13)',
     'varies per row\nSW median 4.00 (2.42-5.47)\nST median 9.22 (4.01-9.22)', '1.536', '0.857', '0.0192', '0.0455'],
]
xpos = [0.02, 0.265, 0.49, 0.74, 0.815, 0.905, 0.99]       # left edge for the first three columns, right edge for the numeric ones
ha = ['left', 'left', 'left', 'right', 'right', 'right', 'right']

fig, ax = plt.subplots(figsize=(13.5, 4.6), facecolor='white'); ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.text(0.02, 0.97, 'Interfacial and wall friction multipliers by calibration data', fontsize=13, color=INK, fontweight='bold', va='top')
y_hdr, row_h = 0.80, [0.19, 0.19, 0.26]
ax.add_patch(plt.Rectangle((0.01, y_hdr - 0.06), 0.98, 0.10, color=HDR, lw=0))
for c, x, h in zip(cols, xpos, ha): ax.text(x, y_hdr - 0.01, c, fontsize=10.5, color=INK, fontweight='bold', ha=h, va='center')
ax.plot([0.01, 0.99], [y_hdr - 0.06] * 2, color=INK2, lw=1.0)
y = y_hdr - 0.06
for r, hgt in zip(rows, row_h):
    mid = y - hgt / 2
    for j, (txt, x, h) in enumerate(zip(r, xpos, ha)):
        multi = j == 2 and txt.startswith('varies')
        ax.text(x, mid, txt, fontsize=10 if multi else 10.5, color=INK, ha=h, va='center',
                fontweight='bold' if (j in (2, 3, 4) and not multi) else 'normal', linespacing=1.4)
    y -= hgt
    ax.plot([0.01, 0.99], [y] * 2, color=GRID, lw=1.0)
ax.text(0.02, 0.085, "Losses are mean squared relative error (holdup + dP/dz) on each fit's own rows, so they are not comparable across rows. All values are from equilibrium-path fits.\n"
        'Pooled NN: f_i_mult comes from a 13-weight network of log Re_SL and log Re_SG; k_wl and k_wg are single fitted scalars.\n'
        'Positive inclination = uphill (assumed, not verified against the source paper).',
        fontsize=8.5, color=INK2, va='top', linespacing=1.45)
fig.savefig('figures/closure_values_table.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('wrote figures/closure_values_table.png')
