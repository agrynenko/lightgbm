import xgboost as xgb


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
        self.d_train = xgb.DMatrix(X_train, label=y_train, qid=qid_train)
        self.d_val = xgb.DMatrix(X_val, label=y_val, qid=qid_val)
    
    def train(
        self,
        params,
        boost_rounds,
        callbacks,
        early_stopping_rounds = 10
    ):
        return xgb.train(
            params=params,
            dtrain=self.d_train,
            num_boost_round=boost_rounds,
            evals=[self.d_val],
            early_stopping_rounds=early_stopping_rounds,
            callbacks=callbacks
        )
