from typing import Any, Dict
import xgboost as xgb

from utils import make_group



class XGB_Engine():
    def __init__(
        self,
        X_train,
        y_train,
        qid_train,
        X_val,
        y_val,
        qid_val,
    ):
        self.d_train = xgb.DMatrix(X_train, label=y_train, group=make_group(qid_train))
        self.d_val = xgb.DMatrix(X_val, label=y_val, group=make_group(qid_val))
    
    def train(
        self,
        params: Dict[str, Any],
        boost_rounds: int,
        callbacks: list,
        early_stopping_rounds: int = 10
    ):
        return xgb.train(
            params=params,
            dtrain=self.d_train,
            num_boost_round=boost_rounds,
            evals=[(self.d_val, "val")],
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks,
        )
