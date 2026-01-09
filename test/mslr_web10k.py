import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import time
from configs import mslr_web10k
from engines import *
from callbacks import *


def load_letor(path):
    labels = []
    qids = []
    features = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split("#")[0].split()

            label = int(parts[0])
            qid = int(parts[1].split(":")[1])

            feats = {}
            for item in parts[2:]:
                k, v = item.split(":")
                feats[int(k)] = float(v)

            labels.append(label)
            qids.append(qid)
            features.append(feats)

    X = pd.DataFrame(features).fillna(0.0)
    y = pd.Series(labels)
    qid = pd.Series(qids)

    return X, y, qid


def task(fold: str):
    start = time.time()

    # Load fold once
    X_train, y_train, qid_train = load_letor(f"data/MSLR-WEB10K/{fold}/train.txt")
    X_val, y_val, qid_val = load_letor(f"data/MSLR-WEB10K/{fold}/vali.txt")
    X_test, y_test, qid_test = load_letor(f"data/MSLR-WEB10K/{fold}/test.txt")

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
            engine = xgb_engine
            cb = XGB_NDCG_TIME(*cb_params, freq=1, max_time=300, verbose=True)

        elif engine_type == "lgbm":
            engine = LGBM_Engine(
                *engine_params
            )  # new engine each time (Dataset handle needs this)
            cb = LGBM_NDCG_TIME(*cb_params, freq=1, max_time=300, verbose=True)

        else:
            raise RuntimeError(f"Unknown engine {engine_type}")

        engine.train(
            params=model["model"],
            boost_rounds=1000,
            callbacks=[cb],
        )

        fold_results[name] = cb.to_dataframe()

    return fold_results


TASKS = 5


def main():
    for i in range(1, TASKS + 1):
        print(f"===== TASK {i} =====")
        fold_results = task(f"Fold{i}")

        for model_name, df in fold_results.items():
            filename = f"web10k_{model_name}_fold{i}_ndcgs.csv"
            df.to_csv(filename, index=False)
            print(f"Saved {filename}")


if __name__ == "__main__":
    main()
