import argparse
import time
import sys
from typing import Optional
import xgboost as xgb
import numpy as np

def load_xgb_conf(path):
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)

            if v.lower() in ("true", "false"):
                v = v.lower() == "true"
            else:
                try:
                    if "." in v:
                        v = float(v)
                    else:
                        v = int(v)
                except ValueError:
                    pass

            params[k] = v
    return params

dts = []

class IterTimer(xgb.callback.TrainingCallback):
    def before_training(self, model):
        print("[XGBoost] training start")
        return model
    
    def before_iteration(self, model, epoch, evals_log):
        self.start = time.time()
        return False

    def after_iteration(self, model, epoch, evals_log):
        dt = time.time() - self.start
        dts.append(dt)
        print(f"[XGBoost] iteration {epoch + 1} took {dt:.6f} seconds")
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--train", required=True, type=str)
    p.add_argument("--val", required=True, type=str)
    p.add_argument("--test", required=False, type=str, default=None)
    p.add_argument("--objective", default="reg:squarederror")
    p.add_argument("--num_threads", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--num_round", type=int, default=100)
    p.add_argument("--eval", type=str, default="rmse")

    args = p.parse_args()

    config = load_xgb_conf(args.config)

    dtrain = xgb.DMatrix(f'{args.train}')
    dval = xgb.DMatrix(f'{args.val}')
    if args.test:
        dtest  = xgb.DMatrix(f'{args.test}')

    params = {
        "objective": args.objective,
        "learning_rate": args.learning_rate,
        "min_child_weight": args.min_child_weight,
        "nthread": args.num_threads,
        "eval_metric": args.eval,
        # from external config
        "tree_method": config.pop('tree_method', 'hist'),
        "max_bin": config.pop('max_bin', 255),
        "grow_policy": config.pop('grow_policy', 'depthwise'),
        "max_depth": config.pop('max_depth'),
        "max_leaves": config.pop('max_leaves', 0)
    }

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dval, "val")],
        callbacks=[IterTimer()],
    )
    
    if args.test:
        ndcg = booster.eval(
            dtest, "test"
        )
        
        print(f"[XGBoost] ndcg@10 on test set: {ndcg}")
    
if __name__ == "__main__":
    try:
        main()
        print(f"[XGBoost] Average iteration time: {np.mean(dts)}")
        print(f"[XGBoost] Std iteration time: {np.std(dts)}")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
