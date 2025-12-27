import time
import lightgbm as lgbm
import pandas as pd

from utils import mean_ndcg_at_k


"""
    Evaluate NDCG@10 over time on a test dataset
"""
class LGBM_NDCG_TIME:
    def __init__(
        self,
        X_test,
        y_test,
        qid_test,
        freq: int = 10,
        max_time: int = -1,
        verbose: bool = False,
    ):
        import numpy as np

        self.X_test = X_test
        self.y_test = y_test
        self.qid_test = qid_test

        self.freq = freq
        self.max_time = max_time
        self.verbose = verbose

        self.start_time = None
        self.times = []
        self.ndcgs = []

    def __call__(self, env):
        epoch = env.iteration

        if self.start_time is None:
            self.start_time = time.time()
            
        if epoch == 0 or epoch % self.freq != 0:
            return

        elapsed = time.time() - self.start_time

        y_pred = env.model.predict(self.X_test)

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
            raise lgbm.callback.EarlyStopException(
                best_iteration=env.iteration,
                best_score=env.evaluation_result_list
            )

    def to_dataframe(self):
        return pd.DataFrame(
            {
                "time": self.times,
                "ndcg@10": self.ndcgs,
            }
        )

