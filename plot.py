import matplotlib.pyplot as plt
import glob
import numpy as np

from pathlib import Path
from utils import parse_lightgbm_log, parse_xgboost_log


# -----------------------------
# Configuration
# -----------------------------
folders = [
    ('results/allstate', 'auc'),
    ('results/mslr', 'ndcg@10'),
    ('results/efb', 'ndcg@10'),
    ('results/sampling', 'ndcg@10')
]


# -----------------------------
# Data collection
# -----------------------------
def collect_stats(folder: str, metric_name: str):
    files = glob.glob(f'{folder}/*.log')
    xgb = glob.glob(f'{folder}/xgb*.log')

    results = []

    # ---- LightGBM ----
    for file in list(set(files) - set(xgb)):
        t, m = parse_lightgbm_log(file, metric_name)
        if not t or not m:
            continue

        results.append({
            "name": Path(file).stem,
            "times": np.array(np.diff(t)),
            "metrics": np.array(m),
        })

    # ---- XGBoost ----
    for file in xgb:
        metric = metric_name
        if metric_name == 'ndcg@10':
            metric = 'ndcg'

        t, m = parse_xgboost_log(file, metric)
        if not t or not m:
            continue

        results.append({
            "name": Path(file).stem,
            "times": np.array(t),
            "metrics": np.array(m),
        })

    return results


# -----------------------------
# Console output
# -----------------------------
def print_stats(results, metric_name, folder):
    print(f"\n{folder}")
    print("-" * 90)
    print(
        f"{'Name':<25}"
        f"{'Median Time (s)':>18}"
        f"{'Mean '+metric_name:>18}"
        f"{'Std':>12}"
        f"{'Max '+metric_name:>17}"
    )
    print("-" * 90)

    for r in results:
        median_time = np.median(r["times"])
        mean_metric = np.mean(r["metrics"])
        std_metric = np.std(r["metrics"])
        max_metric = np.max(r["metrics"])

        print(
            f"{r['name']:<25}"
            f"{median_time:>18.3f}"
            f"{mean_metric:>18.5f}"
            f"{std_metric:>12.5f}"
            f"{max_metric:>17.5f}"
        )


# -----------------------------
# Plotting
# -----------------------------
def plot_time(results, title):
    plt.figure(figsize=(10, 6))

    for r in results:
        plt.bar(
            r['name'],
            np.median(r['times'])
        )

    plt.ylabel("Median Iteration Time (seconds)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_metric(results, metric_name, title):
    names = [r["name"] for r in results]
    means = [np.mean(r["metrics"]) for r in results]
    stds = [np.std(r["metrics"]) for r in results]

    x = np.arange(len(names))

    plt.figure(figsize=(12, 6))
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5)

    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    for folder, metric in folders:
        results = collect_stats(folder, metric)

        if not results:
            print(f"\n{folder}: No valid results found.")
            continue

        print_stats(results, metric, folder)

        plot_time(
            results,
            title=f"{folder} – Median Iteration Time"
        )

        plot_metric(
            results,
            metric,
            title=f"{folder} – {metric} (mean ± std)"
        )


if __name__ == '__main__':
    main()
