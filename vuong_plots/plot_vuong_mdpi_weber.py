"""
Figure: gas-phase Weber number comparison, Vuong (2016) oil-nitrogen vs.
MDPI/Carraretto (2020) water-air -- the check behind the power-law fit's
sign-flipped exponent (alpha_fi=+0.61 here vs. Carraretto's own -0.53).
See CLAUDE.md "Weber number check confirms Vuong and MDPI sit in
different interfacial-wave regimes" and _vuong_mdpi_weber_check.py (the
script that produced the numbers hardcoded below).

Two panels:
  1. We distribution by dataset (log y, jittered strip plot) -- shows
     MDPI sitting right at the We~O(1) surface-tension/inertia
     transition, Vuong sitting two to three orders of magnitude higher,
     deep in the inertia-dominated regime, across its ENTIRE range.
  2. Decomposition of the ~259-290x gap in We's median into its four
     multiplying factors (rho_g, u_rel^2, D, 1/sigma) -- the point being
     that gas density (pointing at Vuong's facility running at elevated
     pressure), not surface tension, is the dominant driver.

Usage: python3 plot_vuong_mdpi_weber.py
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
S1, S2 = '#2a78d6', '#eb6834'

VUONG_WE = np.array([
    419.84, 355.34, 399.53, 377.88, 769.27, 743.96, 801.95, 724.67, 706.81,
    918.38, 970.17, 945.88, 998.03, 813.06, 755.13, 732.30, 851.30, 1483.66,
    1399.26, 1427.92, 1386.28, 1249.22, 1280.29, 1315.20, 1239.51, 1889.22,
    1701.99, 1640.02, 1624.28, 336.47, 890.42, 787.49, 1455.50, 1347.54,
    1909.44, 1667.71,
])
MDPI_WE = np.array([
    9.02, 6.13, 4.74, 3.71, 2.22, 1.76, 1.16, 0.82,
    10.11, 6.99, 5.52, 4.36, 2.68, 2.18, 1.47, 1.06,
    11.28, 7.93, 6.29, 5.05, 3.16, 2.60, 1.78, 1.32,
    12.38, 8.92, 7.11, 5.76, 3.68, 3.04, 2.12, 1.58,
])

FACTORS = ['gas density\n(rho_g)', 'relative\nvelocity^2', 'pipe\ndiameter', '1/surface\ntension']
FACTOR_VALS = [20.1, 1.85, 2.60, 3.00]

def new_ax(figsize=(6.2, 5.6)):
    fig, a = plt.subplots(figsize=figsize, facecolor='white')
    a.set_facecolor('white'); a.grid(color=GRID, lw=0.8, axis='y'); a.set_axisbelow(True)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
    for s in ('left', 'bottom'): a.spines[s].set_color(INK2)
    a.tick_params(colors=INK2)
    return fig, a

# --- figure 1: We distribution by dataset ---
fig1, a = new_ax()
rng = np.random.default_rng(0)
x_vuong = 1 + rng.uniform(-0.12, 0.12, size=len(VUONG_WE))
x_mdpi = 2 + rng.uniform(-0.12, 0.12, size=len(MDPI_WE))

a.axhspan(1e-2, 1, color=S2, alpha=0.08, zorder=1)
a.axhline(1.0, color=INK2, lw=1.2, ls='--', zorder=2)
a.text(2.45, 1.0, 'We=1 (inertia = surface tension)', color=INK2, fontsize=8.5,
       va='bottom', ha='right')

a.scatter(x_vuong, VUONG_WE, s=42, color=S1, edgecolor='white', linewidth=1.0,
          alpha=0.85, zorder=3, label=f'Vuong, oil-N2 (n={len(VUONG_WE)})')
a.scatter(x_mdpi, MDPI_WE, s=42, color=S2, edgecolor='white', linewidth=1.0,
          alpha=0.85, zorder=3, label=f'MDPI, water-air (n={len(MDPI_WE)})')

for xc, data, color in ((1, VUONG_WE, S1), (2, MDPI_WE, S2)):
    med = np.median(data)
    a.plot([xc - 0.18, xc + 0.18], [med, med], color=color, lw=2.6, zorder=4)
    a.text(xc, med * 1.4, f'median\n{med:.1f}', color=color, fontsize=8.5,
           ha='center', va='bottom', fontweight='bold')

a.set_yscale('log')
a.set_xlim(0.5, 2.5)
a.set_xticks([1, 2]); a.set_xticklabels(['Vuong\n(oil-N2)', 'MDPI\n(water-air)'], color=INK)
a.set_ylabel('We = rho_g * u_rel^2 * D / sigma  [-]', color=INK)
a.set_title('Gas-phase Weber number: Vuong vs. MDPI', color=INK, loc='left', fontsize=11)
a.legend(frameon=False, loc='lower right', fontsize=8.5)
fig1.savefig('figures/weber_check/vuong_mdpi_weber_comparison.png', dpi=160, bbox_inches='tight')

# --- figure 2: decomposition of the We gap ---
fig2, a = new_ax(figsize=(6.6, 5.0))
x = np.arange(len(FACTORS))
bars = a.bar(x, FACTOR_VALS, width=0.55, color=S1, zorder=3)
a.axhline(1.0, color=INK2, lw=1, ls='--', zorder=2)
a.text(len(FACTORS) - 0.3, 1.05, 'no effect (1x)', color=INK2, fontsize=8.5, ha='right', va='bottom')
for xi, v in zip(x, FACTOR_VALS):
    a.text(xi, v * 1.05, f'{v:.2f}x', ha='center', fontsize=10, color=INK, fontweight='bold')
a.set_yscale('log')
a.set_xticks(x); a.set_xticklabels(FACTORS, color=INK, fontsize=9)
a.set_ylabel('multiplying factor (Vuong / MDPI)', color=INK)
a.set_title('What drives the We gap: density, not surface tension', color=INK, loc='left', fontsize=11)
fig2.savefig('figures/weber_check/vuong_mdpi_weber_decomposition.png', dpi=160, bbox_inches='tight')

print('saved: vuong_mdpi_weber_{comparison,decomposition}.png')
