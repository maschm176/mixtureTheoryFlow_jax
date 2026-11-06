"""
Stage 1 figure -- water-gas solver forward-run stability check.

Styled to match the oil-water solver's Phase 1 figure (terminal-style
"Final evaluation" panel + a second diagnostic plot + a parity-style
plot), for visual consistency with figures/stage2_synthetic_recovery.py.

Data below is not synthesized -- it is pulled directly from the
already-committed zero-refit result pickles (fan2005_real_data_results.pkl,
mdpi_real_data_results.pkl, both produced by run_real_data_condition at
the uncalibrated default f_i_mult=1.0, beta=0) and the source CSVs
(fan2005_sw_subset.csv filtered to Authors=="Fan, Y." & Flow
Pattern=="SW" & Inclination angle==0 deg, to match the 13-row horizontal
subset the pickle was built from; mdpi_airwater_phase2g_dataset.csv)
for each condition's (U_sl, U_sg). Row order in both CSV filters was
verified to line up with the pickles' own alpha_l_meas values before
use.

This is a *stability* check, not an accuracy check: f_i_mult is still at
its uncalibrated default (1.0), so holdup error is expected to be large
(the ~46-47% mean documented informally in CLAUDE.md) -- Stage 1 only
claims the forward solver runs to completion without NaN or a
hyperbolicity breakdown (kh_total>0) across these real conditions, not
that it is accurate yet (that's Step 3, calibration).

Vuong (2016) was added to the holdup parity panel on request, to show
the *calibrated* fit's accuracy on the dataset the project has leaned on
most (both holdup and dPdz measured on every condition, oil-nitrogen --
a different fluid pair than Fan/MDPI's water-air). This is a deliberate
mix of two different things on one axis, called out explicitly rather
than left implicit: Fan/MDPI points are zero-refit (f_i_mult=1.0,
k_wl=k_wg=1.0, uncalibrated); Vuong points are from the converged joint
fit (f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413 -- true jax.grad through
the closed-form equilibrium path, not finite differences through the
dynamic scan; see "Nondimensionalization + equilibrium-path true-gradient
training supersedes the finite-difference joint fit" in CLAUDE.md). Vuong
is NOT a zero-refit check -- these are the same 36 conditions (29 train /
7 held-out test) the parameters were fit against. Per-condition values
were pulled directly from a fresh run of _vuong_eval_converged_detail.py
(the DYNAMIC scan, all 36 conditions, kh_total=0 throughout -- re-checked
at the new parameters, not assumed), not reconstructed.
"""
import matplotlib.pyplot as plt
import numpy as np

# fields: i, U_sl, U_sg, alpha_l_pred, alpha_l_meas (None if missing), kh_total
FAN = [
    (0, 0.018551222, 15.11, 0.030719262842577110, 0.02, 0),
    (1, 0.014319654, 19.71, 0.019094427153694925, 0.02, 15051316),
    (2, 0.013638957, 15.09, 0.023498808166629907, 0.02, 0),
    (3, 0.010954350, 15.12, 0.019325589403634845, 0.02, 0),
    (4, 0.010300073, 19.88, 0.014141467601196094, 0.01, 14890685),
    (5, 0.007766032, 14.93, 0.014402051637007338, 0.01, 0),
    (6, 0.006064823, 19.86, 0.008789146773426460, 0.01, 0),
    (7, 0.004529977, 14.97, 0.008453264824096883, 0.01, 0),
    (8, 0.003119551, 20.26, 0.004488029194528018, 0.01, 0),
    (9, 0.002288800, 14.85, 0.005244023559500990, 0.01, 0),
    (10, 0.001296915, 24.96, 0.002068314690479113, None, 0),
    (11, 0.001058438, 19.96, 0.002359241599399545, None, 0),
    (12, 0.000789980, 14.86, 0.002764707401717292, None, 0),
]
MDPI = [
    (0, 0.03, 2.371082, 0.2406002745534188, 0.189162, 0),
    (1, 0.03, 1.858606, 0.2793376699445702, 0.192871, 0),
    (2, 0.03, 1.572003, 0.3078819837177368, 0.203227, 0),
    (3, 0.03, 1.332869, 0.3373518569617668, 0.212411, 0),
    (4, 0.03, 0.927075, 0.4059014095000127, 0.216421, 0),
    (5, 0.03, 0.780628, 0.4395786624211944, 0.225541, 0),
    (6, 0.03, 0.560077, 0.4108929935387989, 0.250026, 0),
    (7, 0.03, 0.420003, 0.3312566391233025, 0.274190, 0),
    (8, 0.04, 2.361156, 0.2874325417563273, 0.229800, 0),
    (9, 0.04, 1.848497, 0.3302151458085137, 0.245002, 0),
    (10, 0.04, 1.572355, 0.3599718425381269, 0.250057, 0),
    (11, 0.04, 1.330607, 0.3916615387288239, 0.259533, 0),
    (12, 0.04, 0.924562, 0.4630155601643355, 0.258222, 0),
    (13, 0.04, 0.782995, 0.4959943692323773, 0.269009, 0),
    (14, 0.04, 0.561410, 0.4613985144907619, 0.297659, 0),
    (15, 0.04, 0.420526, 0.3817732734766055, 0.321915, 0),
    (16, 0.05, 2.362214, 0.3262389316568273, 0.245403, 0),
    (17, 0.05, 1.853398, 0.3710122713786171, 0.257644, 0),
    (18, 0.05, 1.569972, 0.4028076645279831, 0.273864, 0),
    (19, 0.05, 1.332538, 0.4348655655711814, 0.275339, 0),
    (20, 0.05, 0.922622, 0.5076925171468454, 0.301673, 0),
    (21, 0.05, 0.782499, 0.5401253810872110, 0.307879, 0),
    (22, 0.05, 0.559856, 0.5041499930618184, 0.347789, 0),
    (23, 0.05, 0.419925, 0.4253886885278724, 0.381990, 0),
    (24, 0.06, 2.353211, 0.3603948608886750, 0.272685, 0),
    (25, 0.06, 1.862531, 0.4050271101105075, 0.287348, 0),
    (26, 0.06, 1.571856, 0.4382448478928998, 0.292539, 0),
    (27, 0.06, 1.335800, 0.4704545683250455, 0.302192, 0),
    (28, 0.06, 0.926358, 0.5427750886612884, 0.324994, 0),
    (29, 0.06, 0.784272, 0.5749954428982801, 0.343485, 0),
    (30, 0.06, 0.561491, 0.5440692031262976, 0.377302, 0),
    (31, 0.06, 0.420612, 0.4696094057619014, 0.413486, 0),
]

fan_alpha_pred = np.array([r[3] for r in FAN])
fan_U_sg = np.array([r[2] for r in FAN])
fan_kh = np.array([r[5] for r in FAN])
fan_stable = fan_kh == 0

mdpi_alpha_pred = np.array([r[3] for r in MDPI])
mdpi_U_sg = np.array([r[2] for r in MDPI])
mdpi_kh = np.array([r[5] for r in MDPI])
mdpi_stable = mdpi_kh == 0  # all True -- 32/32 clean

n_fan_clean = int(fan_stable.sum())
n_mdpi_clean = int(mdpi_stable.sum())
n_total = len(FAN) + len(MDPI)
n_clean = n_fan_clean + n_mdpi_clean

# Vuong (2016), calibrated fit (f_i_mult=3.8893, k_wl=1.5419, k_wg=0.8413) --
# fields: split, U_sl, U_sg, psi_meas, psi_pred. All kh_total=0 (clean).
VUONG = [
    ("train", 0.0500, 1.620, 0.1501, 0.1460),
    ("train", 0.0200, 1.623, 0.0833, 0.0688),
    ("train", 0.0401, 1.625, 0.1360, 0.1224),
    ("train", 0.0301, 1.626, 0.1101, 0.0969),
    ("train", 0.0402, 2.379, 0.0882, 0.0892),
    ("train", 0.0303, 2.382, 0.0672, 0.0700),
    ("train", 0.0501, 2.396, 0.1045, 0.1065),
    ("train", 0.0202, 2.400, 0.0552, 0.0491),
    ("train", 0.0100, 2.417, 0.0314, 0.0259),
    ("train", 0.0200, 2.476, 0.0500, 0.0434),
    ("train", 0.0403, 2.479, 0.0791, 0.0799),
    ("train", 0.0302, 2.481, 0.0657, 0.0623),
    ("train", 0.0499, 2.482, 0.0907, 0.0955),
    ("train", 0.0401, 2.823, 0.0805, 0.0861),
    ("train", 0.0199, 2.828, 0.0503, 0.0471),
    ("train", 0.0099, 2.829, 0.0294, 0.0251),
    ("train", 0.0505, 2.844, 0.0960, 0.1037),
    ("train", 0.0501, 3.131, 0.0728, 0.0790),
    ("train", 0.0200, 3.132, 0.0361, 0.0353),
    ("train", 0.0300, 3.136, 0.0497, 0.0506),
    ("train", 0.0100, 3.138, 0.0217, 0.0188),
    ("train", 0.0201, 3.262, 0.0330, 0.0374),
    ("train", 0.0302, 3.264, 0.0458, 0.0536),
    ("train", 0.0401, 3.273, 0.0603, 0.0685),
    ("train", 0.0102, 3.288, 0.0195, 0.0201),
    ("train", 0.0100, 4.111, 0.0169, 0.0160),
    ("train", 0.0400, 4.387, 0.0509, 0.0600),
    ("train", 0.0200, 4.419, 0.0294, 0.0324),
    ("train", 0.0100, 4.439, 0.0175, 0.0171),
    ("test", 0.0100, 1.628, 0.0559, 0.0373),
    ("test", 0.0104, 2.457, 0.0256, 0.0242),
    ("test", 0.0300, 2.834, 0.0654, 0.0670),
    ("test", 0.0401, 3.134, 0.0616, 0.0652),
    ("test", 0.0501, 3.277, 0.0740, 0.0827),
    ("test", 0.0200, 4.114, 0.0275, 0.0303),
    ("test", 0.0300, 4.406, 0.0412, 0.0465),
]
vuong_train_meas = np.array([r[3] for r in VUONG if r[0] == "train"])
vuong_train_pred = np.array([r[4] for r in VUONG if r[0] == "train"])
vuong_test_meas = np.array([r[3] for r in VUONG if r[0] == "test"])
vuong_test_pred = np.array([r[4] for r in VUONG if r[0] == "test"])
vuong_train_Usg = np.array([r[2] for r in VUONG if r[0] == "train"])
vuong_test_Usg = np.array([r[2] for r in VUONG if r[0] == "test"])
vuong_hld_err = np.array([(p - m) / m * 100 for _, _, _, m, p in VUONG])
vuong_train_mae = np.mean(np.abs(vuong_hld_err[:29]))
vuong_test_mae = np.mean(np.abs(vuong_hld_err[29:]))

COLOR_FAN, COLOR_MDPI, COLOR_UNSTABLE, COLOR_VUONG = "tab:blue", "tab:orange", "tab:red", "tab:green"

fig = plt.figure(figsize=(11, 6.4))
ax_text = fig.add_axes([0.085, 0.53, 0.40, 0.38])
ax_stab = fig.add_axes([0.085, 0.10, 0.40, 0.37])
ax_par  = fig.add_axes([0.58, 0.10, 0.34, 0.80])

# --- top-left: terminal-style "Final evaluation" text panel ---
ax_text.set_facecolor("black")
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

lines = [
    "Final evaluation",
    "Forward-solver stability + Vuong calibration check",
    "",
    f"Fan (2005), SW horiz., zero-refit:  {n_fan_clean}/{len(FAN)} clean",
    "  unstable: i=1,4 (U_sg~19.7-19.9, kh~1.5e7)",
    "",
    f"MDPI (Carraretto 2020), zero-refit: {n_mdpi_clean}/{len(MDPI)} clean",
    f"Overall zero-refit: {n_clean}/{n_total} ({100*n_clean/n_total:.1f}%) stable",
    "(unstable excluded, same policy as PHI_COLLAPSE)",
    "",
    "Vuong (2016), oil-N2, CALIBRATED fit: 36/36 clean",
    "  f_i_mult=3.8893  k_wl=1.5419  k_wg=0.8413",
    f"  train hld err={vuong_train_mae:.1f}%   test hld err={vuong_test_mae:.1f}%",
]
ax_text.text(0.03, 0.95, "\n".join(lines), transform=ax_text.transAxes,
             color="white", fontsize=9.3, fontfamily="monospace",
             va="top", ha="left", linespacing=1.55, clip_on=False)

# --- bottom-left: stability map (predicted holdup vs. U_sg) ---
ax_stab.set_xscale("log")
ax_stab.set_yscale("log")
ax_stab.scatter(fan_alpha_pred[fan_stable], fan_U_sg[fan_stable],
                 color=COLOR_FAN, marker="o", s=55, edgecolor="black", linewidth=0.4,
                 label="Fan (2005), stable", zorder=3)
ax_stab.scatter(mdpi_alpha_pred[mdpi_stable], mdpi_U_sg[mdpi_stable],
                 color=COLOR_MDPI, marker="s", s=45, edgecolor="black", linewidth=0.4,
                 label="MDPI, stable", zorder=3)
ax_stab.scatter(fan_alpha_pred[~fan_stable], fan_U_sg[~fan_stable],
                 facecolors="none", edgecolors=COLOR_UNSTABLE, marker="o", s=160,
                 linewidth=2.2, label="unstable (kh_total>0)", zorder=4)
ax_stab.scatter(vuong_train_pred, vuong_train_Usg, color=COLOR_VUONG, marker="^", s=55,
                 edgecolor="black", linewidth=0.4, label="Vuong (2016), train", zorder=3)
ax_stab.scatter(vuong_test_pred, vuong_test_Usg, facecolors="none", edgecolors=COLOR_VUONG,
                 marker="^", s=90, linewidth=1.8, label="Vuong (2016), test", zorder=4)
ax_stab.set_xlabel("alpha_l_pred (predicted holdup, log scale)")
ax_stab.set_ylabel("U_sg [m/s] (log scale)")
ax_stab.set_title("Stability map: holdup vs. gas velocity")
ax_stab.grid(alpha=0.3, which="both")
ax_stab.legend(fontsize=7.5, loc="upper right")

# --- right: holdup parity, zero-refit (uncalibrated f_i_mult=1.0) ---
fan_has_meas = np.array([r[4] is not None for r in FAN])
fan_meas = np.array([r[4] if r[4] is not None else np.nan for r in FAN])

ax_par.set_xscale("log")
ax_par.set_yscale("log")
lims = (0.0015, 0.7)
band = 0.20
xs = np.logspace(np.log10(lims[0]), np.log10(lims[1]), 50)
ax_par.fill_between(xs, xs * (1 - band), xs * (1 + band), color="gray", alpha=0.18, label="±20%")
ax_par.plot(lims, lims, color="black", ls="--", lw=1.2, label="y = x")

mask = fan_has_meas & fan_stable
ax_par.scatter(fan_meas[mask], fan_alpha_pred[mask], color=COLOR_FAN, marker="o", s=55,
               edgecolor="black", linewidth=0.4, label="Fan (2005)", zorder=3)
mask_u = fan_has_meas & ~fan_stable
ax_par.scatter(fan_meas[mask_u], fan_alpha_pred[mask_u], facecolors="none",
               edgecolors=COLOR_UNSTABLE, marker="o", s=160, linewidth=2.2,
               label="excluded (unstable)", zorder=4)

mdpi_meas = np.array([r[4] for r in MDPI])
ax_par.scatter(mdpi_meas, mdpi_alpha_pred, color=COLOR_MDPI, marker="s", s=45,
               edgecolor="black", linewidth=0.4, label="MDPI", zorder=3)

# Vuong (2016) -- CALIBRATED fit, not zero-refit (see docstring). Train filled,
# test (held-out) hollow-ringed, same visual language as the "excluded" markers
# above, so the calibrated/zero-refit split stays visually distinct at a glance.
ax_par.scatter(vuong_train_meas, vuong_train_pred, color=COLOR_VUONG, marker="^", s=55,
               edgecolor="black", linewidth=0.4, label="Vuong (2016), train (calibrated)", zorder=3)
ax_par.scatter(vuong_test_meas, vuong_test_pred, facecolors="none", edgecolors=COLOR_VUONG,
               marker="^", s=90, linewidth=1.8, label="Vuong (2016), test (held out)", zorder=4)

ax_par.set_xlim(*lims)
ax_par.set_ylim(*lims)
ax_par.set_xlabel("alpha_l_meas (measured holdup, log scale)")
ax_par.set_ylabel("alpha_l_pred (predicted holdup, log scale)")
ax_par.set_title("Holdup parity: Fan/MDPI zero-refit vs. Vuong calibrated fit")
ax_par.grid(alpha=0.3, which="both")
ax_par.legend(fontsize=7.2, loc="upper left")
ax_par.set_aspect("equal")

fig.suptitle("Stage 1 -- forward-solver stability + Vuong calibration check (water-gas solver)",
             fontsize=12, y=0.99)
fig.savefig("figures/stage1/stage1_forward_stability.png", dpi=150)
print("wrote figures/stage1_forward_stability.png")
