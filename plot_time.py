import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Placeholder data dictionary
# -----------------------------

# paper allstate : 10.85 2.63 6.07 0.71 0.28
# paper mslr : 5.55 0.63 0.49 0.46 0.31
# our allstate : 80.35 2.78 7.86 2.42 1.54
# our mslr : 63.16 2.66 3.75 3.77 1.61

data = {
    "LightGBM": {
        "LightGBM": 1.61,     # baseline
        "EFB_only": 3.77,
        "LGB_base": 3.75,
    },
    "XGBoost": {
        "XGB_exa": 63.16,
        "XGB_his": 2.66,
    }
}

# -----------------------------
# Flatten + sort data
# -----------------------------
entries = []
for model, items in data.items():
    for name, value in items.items():
        entries.append((name, value, model))

# sort by increasing time
entries.sort(key=lambda x: x[1])

labels = [e[0] for e in entries]
values = [e[1] for e in entries]
models = [e[2] for e in entries]

# -----------------------------
# Automatic color mapping
# -----------------------------
color_map = {
    "LightGBM": "#4C72B0",
    "XGBoost": "#DD8452",
}
colors = [color_map[m] for m in models]

baseline = data["LightGBM"]["LightGBM"]

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(labels))
bars = ax.bar(x, values, color=colors, width=0.65, edgecolor="black")

ax.set_yscale("log")
ax.set_title("Training Time Cost Comparison (MSLR)", fontsize=16, weight="bold")
ax.set_ylabel("Iteration Time (seconds, log scale)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)

ax.grid(axis="y", which="major", linestyle="-", linewidth=1, alpha=0.3)

# -----------------------------
# Legend (auto-consistent)
# -----------------------------
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=color_map[k], label=k)
    for k in color_map
]
ax.legend(handles=legend_handles, loc="upper left")

# -----------------------------
# Ratio annotations (safe placement)
# -----------------------------
ymax = ax.get_ylim()[1]

for bar, val in zip(bars, values):
    ratio = val / baseline
    text_y = min(val * 1.15, ymax / 1.2)

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        text_y,
        f"{ratio:.1f}×",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
        clip_on=True,
    )

plt.tight_layout()
plt.show()
