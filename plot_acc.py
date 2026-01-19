import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# -----------------------------
# Placeholder data dictionary
# -----------------------------

# paper allstate : 0.6070 0.6089 0.6093 0.6064 0.6093
# paper mslr : 0.4977 0.4982 0.5277 0.5239 0.5275
# our allstate : 0.60078 0.60309 0.59344 0.59771 0.59442
# our mslr : 0.5063 0.5041 0.515392 0.507162 0.513188

data = {
    "LightGBM": {
        "LightGBM": 0.513188,     # baseline
        "SGB": 0.507162,
        "LGB_base": 0.515392,
    },
    "XGBoost": {
        "XGB_exa": 0.5063,
        "XGB_his": 0.5041,
    }
}

# -----------------------------
# Flatten + sort (best first)
# -----------------------------
entries = []
for model, items in data.items():
    for name, value in items.items():
        entries.append((name, value, model))

entries.sort(key=lambda x: -x[1])

labels = [e[0] for e in entries]
values = [e[1] for e in entries]
models = [e[2] for e in entries]

# -----------------------------
# Colors
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
bars = ax.bar(
    x,
    values,
    color=colors,
    width=0.65,
    edgecolor="black"
)

# ---- Zoomed Y-axis ----
ymin = min(values) - 0.002
ymax = max(values) + 0.001
ax.set_ylim(ymin, ymax)

ax.set_title("Accuracy Comparison (MSLR)", fontsize=16, weight="bold")
ax.set_ylabel("NDCG@10", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)

ax.grid(axis="y", linestyle="-", linewidth=1, alpha=0.3)

# -----------------------------
# Legend
# -----------------------------
ax.legend(
    handles=[Patch(facecolor=color_map[k], label=k) for k in color_map],
    loc="upper right"
)

# -----------------------------
# Delta annotations
# -----------------------------
for bar, val in zip(bars, values):
    delta = val - baseline
    sign = "+" if delta >= 0 else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        val + 0.0002,
        f"{sign}{delta:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        weight="bold",
    )

plt.tight_layout()
plt.show()
