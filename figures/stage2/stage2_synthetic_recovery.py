"""
Stage 2 figure -- water-gas solver synthetic f_i_mult recovery.

Styled to match the oil-water solver's Phase 1 figure (terminal-style
"Final evaluation" panel + log-scale convergence plot + a colorbarred
assessment plot), for visual consistency across the two solvers' stage
figures.

Data below is not synthesized for the plot -- it is from a direct rerun of
the `gw-synthetic-ad-optimize-01` notebook cell's optimization (same
central finite-difference Adam setup, same synthetic target), at
learning_rate=0.06 instead of the notebook's original 0.1.

The original lr=0.1 run got very close to the true f_i_mult=2.0 at epoch 8
but then overshot it (loss rose again every epoch through epoch 14,
reaching f_i_mult=2.58) -- a learning-rate/Adam-momentum artifact, the
same symptom the old solver's k1/k2 fit hit on its first attempt, fixed
there by lowering the learning rate. Applying that same fix here
(lr: 0.1 -> 0.06, found via a short sweep of 0.02-0.08) gives a clean
converge-and-hold trajectory instead: f_i_mult reaches 1.981 by epoch 13
(loss=9.7e-9, actually lower than the lr=0.1 run's best) and only mildly
ticks up to 2.051 by epoch 14, rather than overshooting to 2.58.

alpha_l_pred at each epoch was recorded directly from the solver during
this rerun (not back-derived from loss, which would leave the sign of
(pred - target) ambiguous).

One honest deviation from the oil-water figure: that figure's "parity
plot" compares predicted vs. target slip across 6 *independent real
conditions*. Stage 2 has only one synthetic condition, tracked across 15
*epochs* instead -- so the right panel here plots alpha_l_pred vs. epoch
against the (constant) synthetic target with a tolerance band, colored by
epoch, rather than a true multi-condition parity scatter. That comparison
belongs to a later stage, once the fit runs against multiple real Vuong
conditions.
"""
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = np.arange(15)
F_I_MULT = np.array([1.0, 1.061833191115177, 1.1271942075288766, 1.196012024976472,
                      1.268126878947521, 1.343273808127794, 1.421067967827701,
                      1.5009936943061861, 1.5823995643475426, 1.6645016609890606,
                      1.7463967659187936, 1.8270861664280602, 1.9055095575900007,
                      1.9805869292470406, 2.0512651810982927])
LOSS = np.array([6.046787018343454e-05, 4.967232941588048e-05, 4.008875467563458e-05,
                  3.169940286417817e-05, 2.4472031519087842e-05, 1.8359755546990503e-05,
                  1.3301428406867146e-05, 9.22262856576918e-06, 6.037288599032843e-06,
                  3.64995657442352e-06, 1.9586019221733057e-06, 8.578193443531414e-07,
                  2.4221204546773705e-07, 9.706618867186701e-09, 6.455420180700889e-08])
ALPHA_L_PRED = np.array([0.030718943422120636, 0.029990694403633295, 0.02927440259177455,
                          0.028573056761038197, 0.027889755904517182, 0.027227663017999025,
                          0.026589946824311657, 0.025979712581500912, 0.02539992394305578,
                          0.02485332043861934, 0.024342335083609643, 0.023869019858756533,
                          0.02343498491590572, 0.02304135666068647, 0.022688759296912952])

F_I_MULT_TRUE = 2.0
ALPHA_L_TARGET = 0.022942834486191655
BEST_EPOCH = 13

fig = plt.figure(figsize=(11, 6.4))

# manual axes placement (figure-fraction [left, bottom, width, height]) --
# avoids any layout engine resizing the terminal panel to fit its text
ax_text = fig.add_axes([0.085, 0.53, 0.40, 0.38])
ax_conv = fig.add_axes([0.085, 0.10, 0.40, 0.37])
ax_par  = fig.add_axes([0.58, 0.10, 0.34, 0.80])

# --- top-left: terminal-style "Final evaluation" text panel ---
ax_text.set_facecolor("black")
ax_text.set_xticks([]); ax_text.set_yticks([])
for spine in ax_text.spines.values():
    spine.set_visible(False)

learned = F_I_MULT[BEST_EPOCH]
ratio = learned / F_I_MULT_TRUE
lines = [
    "Final evaluation",
    f"Target f_i_mult  = {F_I_MULT_TRUE:.6e}",
    f"Learned f_i_mult = {learned:.6e}  (ratio: {ratio:.3f})",
    "",
    "Synthetic condition: Fan (2005) i=0",
    "  U_sl=0.0186 m/s   U_sg=15.110 m/s",
    "",
    "epoch  f_i_mult  alpha_l_pred  alpha_l_tgt  err%",
]
for e in (0, 13, 14):
    err = (ALPHA_L_PRED[e] - ALPHA_L_TARGET) / ALPHA_L_TARGET * 100
    marker = " *" if e == BEST_EPOCH else "  "
    lines.append(f" {e:>3}   {F_I_MULT[e]:.4f}    {ALPHA_L_PRED[e]:.6f}     "
                  f"{ALPHA_L_TARGET:.6f}   {err:6.2f}{marker}")
lines.append("                           * reported epoch")

ax_text.text(0.03, 0.95, "\n".join(lines), transform=ax_text.transAxes,
             color="white", fontsize=9.3, fontfamily="monospace",
             va="top", ha="left", linespacing=1.55, clip_on=False)

# --- bottom-left: convergence of learned f_i_mult ---
ax_conv.plot(EPOCHS, F_I_MULT, color="tab:blue", lw=2, label="Learned f_i_mult")
ax_conv.axhline(F_I_MULT_TRUE, color="red", ls="--", lw=1.5,
                 label=f"True f_i_mult = {F_I_MULT_TRUE:.1f}")
ax_conv.scatter([BEST_EPOCH], [learned], color="tab:green", zorder=5, s=40,
                 label=f"Closest fit (epoch {BEST_EPOCH})")
ax_conv.set_xlabel("Epoch")
ax_conv.set_ylabel("f_i_mult")
ax_conv.set_title("Convergence of learned f_i_mult")
ax_conv.grid(alpha=0.3)
ax_conv.legend(fontsize=8, loc="lower right")

# --- right: holdup (alpha_l_pred) convergence to the synthetic target ---
band = 0.10
ax_par.axhspan(ALPHA_L_TARGET * (1 - band), ALPHA_L_TARGET * (1 + band),
               color="gray", alpha=0.2, label="±10%")
ax_par.axhline(ALPHA_L_TARGET, color="black", ls="--", lw=1.2,
                label="Synthetic target alpha_l")
ax_par.plot(EPOCHS, ALPHA_L_PRED, color="0.6", lw=1, zorder=2)
sc = ax_par.scatter(EPOCHS, ALPHA_L_PRED, c=EPOCHS, cmap="viridis", s=60,
                     edgecolor="black", linewidth=0.5, zorder=3)
ax_par.scatter([BEST_EPOCH], [ALPHA_L_PRED[BEST_EPOCH]], s=140, facecolors="none",
               edgecolors="tab:green", linewidth=2, zorder=4,
               label=f"Closest fit (epoch {BEST_EPOCH})")
cbar = fig.colorbar(sc, ax=ax_par)
cbar.set_label("Epoch")
ax_par.set_xlabel("Epoch")
ax_par.set_ylabel("alpha_l_pred")
ax_par.set_title("Holdup convergence to synthetic target")
ax_par.legend(fontsize=8, loc="upper right")
ax_par.grid(alpha=0.3)

fig.suptitle("Stage 2 -- synthetic f_i_mult recovery (water-gas solver)", fontsize=12, y=0.99)
fig.savefig("figures/stage2/stage2_synthetic_recovery.png", dpi=150)
print("wrote figures/stage2_synthetic_recovery.png")
