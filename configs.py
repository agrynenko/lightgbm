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
    },

    # LGBM with GOSS only (Algorithm 2 from paper)
    {
        "name":"lgbm_goss",
        "engine": "lgbm",
        "model": {
            "objective": "lambdarank",
            "boosting_type": "goss",  # GOSS enabled
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_child_weight": 100,
            "num_threads": 1,

            # GOSS parameters (a=0.1, b=0.1 for LETOR as per paper)
            "top_rate": 0.1,      # a: ratio of large gradient instances to keep
            "other_rate": 0.1,    # b: ratio of small gradient instances to sample

            # execution control
            "force_row_wise": True,
            "enable_bundle": False,  # EFB disabled

            # other
            "verbosity": -1
        }
    },

    # Full LightGBM with GOSS + EFB (the complete algorithm from paper)
    {
        "name":"lgbm_full",
        "engine": "lgbm",
        "model": {
            "objective": "lambdarank",
            "boosting_type": "goss",  # GOSS enabled
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_child_weight": 100,
            "num_threads": 1,

            # GOSS parameters
            "top_rate": 0.1,
            "other_rate": 0.1,

            # execution control
            "force_row_wise": True,
            "enable_bundle": True,   # EFB enabled

            # other
            "verbosity": -1
        }
    },

    # Stochastic Gradient Boosting (SGB) - baseline comparison from paper
    {
        "name":"lgbm_sgb",
        "engine": "lgbm",
        "model": {
            "objective": "lambdarank",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 255,
            "min_child_weight": 100,
            "num_threads": 1,

            # SGB: random uniform sampling (same ratio as GOSS for fair comparison)
            "data_sample_strategy": "bagging",
            "bagging_fraction": 0.2,  # sample 20% of data (equivalent to a+b in GOSS)
            "bagging_freq": 1,        # resample every iteration

            # execution control
            "force_row_wise": True,
            "enable_bundle": False,

            # other
            "verbosity": -1
        }
    }
]
