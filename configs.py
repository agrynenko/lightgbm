# xgboost parameter list: https://xgboost.readthedocs.io/en/latest/parameter.html
# lightgbm parameter list: https://lightgbm.readthedocs.io/en/latest/Parameters.html

mslr_web10k = [
    # XGB (baseline)
    {
        "name":"xgb_exact",
        "engine": "xgb",
        "model": {
            "learning_rate": 0.05,
            "min_child_weight": 100,
            "tree_method": "exact",
            "grow_policy": "lossguide", # split at nodes with highest loss change => leaf-wise split
            "objective": "rank:ndcg", # LambdaMART for pair-wise ranking
            "nthread": 1
        }
    },
    {
        "name":"xgb_hist",
        "engine": "xgb",
        "model": {
            "learning_rate": 0.05,
            "min_child_weight": 100,
            "tree_method": "hist",
            "grow_policy": "lossguide", # split at nodes with highest loss change => leaf-wise split
            "max_leaves": 255,
            "objective": "rank:ndcg", # LambdaMART for pair-wise ranking
            "nthread": 1
        }
    },
    
    # LGBM
    {
        "name":"lgbm_base",
        "engine": "lgbm",
        "model": {
            "objective": "lambdarank",
            "boosting_type": "gbdt", # GOSS disabled
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_child_weight": 100,
            "num_threads": 1,

            # execution control
            "force_row_wise": True,
            "enable_bundle": False, # EFB disabled
            "data_sample_strategy": "bagging",
            
            # other
            "verbosity": -1
        }
    },
    {
        "name":"lgbm_efb",
        "engine": "lgbm",
        "model": {
            "objective": "lambdarank",
            "boosting_type": "gbdt", # GOSS disabled
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_child_weight": 100,
            "num_threads": 1,

            # execution control
            "force_row_wise": True,
            "enable_bundle": True,
            "data_sample_strategy": "bagging",
            
            #other
            "verbosity": -1
        }
    }
]
