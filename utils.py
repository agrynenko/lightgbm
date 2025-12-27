import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score


def mean_ndcg_at_k(y_true, y_pred, qids, k=10):
    scores = []

    for qid in np.unique(qids):
        mask = qids == qid
        n_docs = np.sum(mask)

        # NDCG undefined for single-doc queries
        if n_docs < 2:
            continue

        score = ndcg_score(
            y_true[mask].to_numpy().reshape(1, -1),
            y_pred[mask].reshape(1, -1),
            k=min(k, n_docs),
        )
        scores.append(score)

    return float(np.mean(scores)) if scores else 0.0


def make_group(qids):
    return qids.value_counts().sort_index().values
