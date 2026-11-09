"""
Figure: terminal-style summary table of how many of the 36 Vuong (2016)
conditions land within +-10%/+-20% of the measured value, for holdup and
dPdz separately, split by train/test -- the exact counts discussed
alongside the power-law fit's parity plots (plot_vuong_powerlaw.py).
Matches vuong_powerlaw_summary.png / vuong_joint_fit_summary.png's panel
styling for visual consistency across this project's figures.

Usage: python3 plot_vuong_powerlaw_bands.py
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (label, n, within10, within20)
HOLDUP = [('all   (36)', 36, 25, 34), ('train (29)', 29, 20, 28), ('test  ( 7)', 7, 5, 6)]
DPDZ = [('all   (36)', 36, 16, 29), ('train (29)', 29, 12, 23), ('test  ( 7)', 7, 4, 6)]

def fmt_rows(rows):
    lines = []
    for label, n, w10, w20 in rows:
        pct10 = w10 / n * 100
        pct20 = w20 / n * 100
        lines.append(f'  {label}    {w10:>2}/{n:<2} ({pct10:4.0f}%)     {w20:>2}/{n:<2} ({pct20:4.0f}%)')
    return lines

fig = plt.figure(figsize=(6.6, 4.6))
ax_text = fig.add_axes([0.06, 0.06, 0.88, 0.88])
ax_text.set_facecolor('black')
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    'Vuong (2016) power-law fit -- fraction of points within',
    '+-10% / +-20% of measured value',
    '',
    'Holdup',
    '                   within +-10%    within +-20%',
] + fmt_rows(HOLDUP) + [
    '',
    'dP/dz',
    '                   within +-10%    within +-20%',
] + fmt_rows(DPDZ)

ax_text.text(0.04, 0.96, '\n'.join(lines), transform=ax_text.transAxes,
             color='white', fontsize=10, fontfamily='monospace',
             va='top', ha='left', linespacing=1.55, clip_on=False)
fig.savefig('figures/powerlaw_fit/vuong_powerlaw_band_table.png', dpi=160, bbox_inches='tight')
print('saved: vuong_powerlaw_band_table.png')
