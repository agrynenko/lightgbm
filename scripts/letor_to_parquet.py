import pandas as pd
from pathlib import Path

def load_letor_raw(path):
    labels = []
    qids = []
    rows = []

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            parts = line.split("#")[0].split()
            labels.append(int(parts[0]))
            qids.append(int(parts[1].split(":")[1]))

            feats = {}
            for item in parts[2:]:
                k, v = item.split(":")
                feats[int(k)] = float(v)
            rows.append(feats)

    X = pd.DataFrame(rows).fillna(0.0).astype("float32")
    y = pd.Series(labels, name="label", dtype="int8")
    qid = pd.Series(qids, name="qid", dtype="int32")

    return X, y, qid


def convert_fold(fold):
    base = Path(f"data/MSLR-WEB10K/{fold}")
    out = Path(f"data/MSLR-WEB10K_PARQUET/{fold}")
    out.mkdir(parents=True, exist_ok=True)

    for split in ["train", "vali", "test"]:
        X, y, qid = load_letor_raw(base / f"{split}.txt")

        df = pd.concat([qid, y, X], axis=1)
        df.to_parquet(out / f"{split}.parquet", compression="zstd")

        print(f"{fold}/{split}: saved {df.shape}")


if __name__ == "__main__":
    for i in range(1, 6):
        convert_fold(f"Fold{i}")
