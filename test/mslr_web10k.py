import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import time
from pathlib import Path
from configs import mslr_web10k
from engines import *
from callbacks import *


def load_letor(path: str):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    df = pd.read_parquet(path)

    # Mandatory columns
    required = {"qid", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    qid = df["qid"]
    y = df["label"]
    X = df.drop(columns=["qid", "label"])

    return X, y, qid


def task(fold: str):
    start = time.time()
    
    # Load fold once
    X_train, y_train, qid_train = load_letor(f"data/MSLR-WEB10K_PARQUET/{fold}/train.parquet")
    X_val, y_val, qid_val = load_letor(f"data/MSLR-WEB10K_PARQUET/{fold}/vali.parquet")
    X_test, y_test, qid_test = load_letor(f"data/MSLR-WEB10K_PARQUET/{fold}/test.parquet")

    engine_params = [X_train, y_train, qid_train, X_val, y_val, qid_val]
    cb_params = [X_test, y_test, qid_test]

    # Init xgb engine once
    xgb_engine = XGB_Engine(*engine_params)

    fold_results = {}
    
    print(f"Dataset loaded, elapsed {(time.time() - start):.3f} seconds")

    for model in mslr_web10k:
        name = model["name"]
        print(f"Experiment {name}")

        engine_type = model["engine"]

        if engine_type == "xgb":
            freq = 1 if (name == "xgb_exact") else 10
            engine = xgb_engine
            cb = XGB_NDCG_TIME(*cb_params, freq=freq, max_time=300, verbose=True)

        elif engine_type == "lgbm":
            engine = LGBM_Engine(*engine_params)  # new engine each time (Dataset handle needs this)
            cb = LGBM_NDCG_TIME(*cb_params, freq=5, max_time=60, verbose=True)

        else:
            raise RuntimeError(f"Unknown engine {engine_type}")

        engine.train(
            params=model["model"],
            boost_rounds=1000,
            callbacks=[cb],
            early_stopping_rounds=20
        )

        fold_results[name] = cb.to_dataframe()


    return fold_results


TASKS = 5


def main():
    all_results = {}

    for i in range(1, TASKS + 1):
        print(f"===== TASK {i} =====")
        fold_results = task(f"Fold{i}")

        for model_name, df in fold_results.items():
            all_results.setdefault(model_name, []).append(df)

    # Aggregate per model
    for model_name, dfs in all_results.items():
        print(f"Averaging model: {model_name}")

        min_len = min(len(df) for df in dfs)
        dfs_trunc = [
            df.iloc[:min_len].reset_index(drop=True)
            for df in dfs
        ]

        avg_df = (
            pd.concat(dfs_trunc)
            .groupby(level=0)
            .mean()
            .reset_index(drop=True)
        )

        filename = f"web10k_{model_name}_ndcgs.csv"
        avg_df.to_csv(filename, index=False)

        print(f"Saved averaged learning curve to {filename}")
  
  
if __name__ == '__main__':
    main()
    