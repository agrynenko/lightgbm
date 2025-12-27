import time
import xgboost as xgb
import pandas as pd
import numpy as np

from utils import mean_ndcg_at_k


"""
    Evaluate NDCG@10 over time on a test dataset
"""
class XGB_NDCG_TIME(xgb.callback.TrainingCallback):
    def __init__(
        self,
        X_test,
        y_test,
        qid_test,
        freq: int = 10,
        max_time: int = -1,
        verbose: bool = False,
    ):
        self.y_test = y_test
        self.qid_test = qid_test
        self.freq = freq
        self.max_time = max_time
        self.verbose = verbose

        self.d_test = xgb.DMatrix(X_test, label=y_test, qid=qid_test)

        self.start_time = None
        self.times = []
        self.ndcgs = []

    def before_training(self, model):
        self.start_time = time.time()
        return model

    def after_iteration(self, model: xgb.Booster, epoch, evals_log):
        if epoch == 0 or epoch % self.freq != 0:
            return False

        elapsed = time.time() - self.start_time

        y_pred = model.predict(self.d_test)

        ndcg = mean_ndcg_at_k(
            self.y_test,
            y_pred,
            self.qid_test,
            k=10,
        )

        self.times.append(elapsed)
        self.ndcgs.append(ndcg)

        if self.verbose:
            print(
                f"[{epoch}] elapsed={elapsed:.3f}s "
                f"NDCG@10={ndcg:.5f}"
            )

        if self.max_time >= 0 and elapsed > self.max_time:
            return True  # stop training

        return False

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "time": self.times,
                "ndcg@10": self.ndcgs,
            }
        )

