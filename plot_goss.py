import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Data
# -----------------------------
sampling_ratios = np.array([0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40])

sgb = np.array([
    0.502008, 0.506328, 0.509194, 0.511373, 0.514013, 0.514034, 0.513941
])

sgb_2 = np.array([
    0.506695, 0.509797, 0.511215, 0.511373, 0.513964, 0.51507, 0.514341
])

lgb_2 = np.array([
    0.512076, 0.514949, 0.515439, 0.515577, 0.516222, 0.5157, 0.516747
])

lightgbm = np.array([
    0.478259, 0.478259, 0.484079, 0.494402, 0.498009, 0.502355, 0.510709
])

lgb = np.array([
    0.513188, 0.513188, 0.519008, 0.529331, 0.532938, 0.537284, 0.545638, 
])

# -----------------------------
# Colors (consistent)
# -----------------------------
color_map = {
    "LightGBM": "#4C72B0",        # LightGBM family
    "SGB": "#DD8452",   # contrast color
}

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(
    sampling_ratios,
    sgb_2,
    marker="o",
    linewidth=2.5,
    markersize=7,
    label="SGB",
    color=color_map["SGB"]
)

ax.plot(
    sampling_ratios,
    lgb_2,
    marker="s",
    linewidth=2.5,
    markersize=7,
    label="LightGBM (GOSS)",
    color=color_map["LightGBM"]
)

# -----------------------------
# Axis formatting
# -----------------------------
ax.set_title(
    "NDCG@10 vs Sampling Ratio (MSLR)",
    fontsize=16,
    weight="bold"
)
ax.set_xlabel("Sampling Ratio", fontsize=12)
ax.set_ylabel("NDCG@10", fontsize=12)

# Zoom Y-axis to emphasize differences
ymin = min(sgb_2.min(), lgb_2.min()) - 0.002
ymax = max(sgb_2.max(), lgb_2.max()) + 0.001
ax.set_ylim(ymin, ymax)

ax.set_xticks(sampling_ratios)
ax.set_xticklabels([f"{r:.2f}" for r in sampling_ratios])

ax.grid(axis="y", linestyle="-", linewidth=1, alpha=0.3)

# -----------------------------
# Legend
# -----------------------------
ax.legend(loc="lower right", fontsize=11)

plt.tight_layout()
plt.show()
