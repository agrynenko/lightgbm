import lightgbm as lgbm

from utils import make_group


class LGBM_Engine():
    def __init__(
        self,
        X_train,
        y_train,
        qid_train,
        X_val,
        y_val,
        qid_val,
    ):
        self.d_train = lgbm.Dataset(X_train, label=y_train, group=make_group(qid_train))
        self.d_val = lgbm.Dataset(X_val, label=y_val, group=make_group(qid_val))
    
    def train(
        self,
        params,
        boost_rounds,
        callbacks: list,
        early_stopping_rounds = 10
    ):
        callbacks.append(lgbm.early_stopping(stopping_rounds=early_stopping_rounds))

        lgbm.train(
            params=params,
            train_set=self.d_train,
            num_boost_round=boost_rounds,
            valid_sets=[self.d_val],
            callbacks=callbacks,
        )
