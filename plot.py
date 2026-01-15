import matplotlib.pyplot as plt
import glob
import numpy as np

from pathlib import Path
from utils import parse_lightgbm_log, parse_xgboost_log


folders = [
    ('results/allstate', 'rmse'),
    ('results/mslr', 'ndcg@10'),
    ('results/efb', 'ndcg@10')
]


def collect_stats(folder: str, metric_name: str):
    files = glob.glob(f'{folder}/*.log')
    xgb = glob.glob(f'{folder}/xgb*.log')

    results = []

    for file in list(set(files) - set(xgb)):
        t, m = parse_lightgbm_log(file, metric_name)
        if not t or not m:
            continue
        results.append({
            "name": Path(file).stem,
            "times": np.array(np.diff(t)),
            "metrics": np.array(m),
        })

    for file in xgb:
        if metric_name == 'ndcg@10':
            metric_name = 'ndcg'
        t, m = parse_xgboost_log(file, metric_name)
        if not t or not m:
            continue
        results.append({
            "name": Path(file).stem,
            "times": np.array(t),
            "metrics": np.array(m),
        })

    return results


def plot_time(results, title):
    plt.figure(figsize=(10, 6))

    for r in results:
        plt.bar(
            r['name'],
            np.median(r['times'])
        )
        
    plt.ylabel("Median Iteration Time (seconds)")
    plt.title(title)
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


def main():
    for folder, metric in folders:
        results = collect_stats(folder, metric)

        plot_time(
            results,
            title=f"{folder} – Iteration Time Distribution"
        )

        plot_metric(
            results,
            metric,
            title=f"{folder} – {metric} (mean ± std)"
        )


if __name__ == '__main__':
  main()
