from collections import defaultdict

CHUNK_SIZE = 500_000

# Columns that should NEVER be categorical or features
EXCLUDE_COLS = {
    "Row_ID",
    "Household_ID",
    "Claim_Amount",  # label source
}


def detect_categorical_columns(csv_path):
    """
    Detect categorical columns and collect global categories.
    Returns: dict of {col_name: sorted_list_of_categories}, header
    """
    categories = defaultdict(set)

    print(f"[pass 1] Scanning categories in {csv_path}")

    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')

        line_count = 0
        for line in f:
            line_count += 1
            if line_count % CHUNK_SIZE == 0:
                print(f"  Scanned {line_count} rows...")

            values = line.strip().split(',')

            for col_name, val in zip(header, values):
                if col_name in EXCLUDE_COLS:
                    continue

                try:
                    float(val)
                except (ValueError, TypeError):
                    if val and val.strip():
                        categories[col_name].add(val.strip())

    cat_mapping = {
        col: sorted(vals)
        for col, vals in categories.items()
    }

    return cat_mapping, header


def write_libsvm_row(f, label, features):
    """
    Write a single LIBSVM row.
    features: dict of {feature_index: value}
    """
    f.write(str(label))
    for idx in sorted(features.keys()):
        val = features[idx]
        if val != 0:
            f.write(f" {idx}:{val}")
    f.write('\n')


def csv_to_libsvm(
    csv_path,
    train_out,
    test_out,
    test_size=1_000_000,
    label_col="Claim_Amount",
):
    # -------------------------
    # Pass 1: detect categories
    # -------------------------
    cat_mapping, header = detect_categorical_columns(csv_path)

    print(f"\nDetected {len(cat_mapping)} categorical columns:")
    print(sorted(cat_mapping.keys()))

    label_idx = header.index(label_col)

    num_cols = []
    cat_cols = []

    for col_name in header:
        if col_name == label_col:
            continue
        if col_name in cat_mapping:
            cat_cols.append(col_name)
        else:
            num_cols.append(col_name)

    print(f"Numerical features: {len(num_cols)}")
    print(f"Categorical features: {len(cat_cols)}")

    # -------------------------
    # Feature index mapping
    # -------------------------
    feature_idx = 0
    num_feature_map = {}

    for col_name in num_cols:
        num_feature_map[col_name] = feature_idx
        feature_idx += 1

    cat_feature_map = {}
    for col_name in cat_cols:
        categories = cat_mapping[col_name]
        cat_feature_map[col_name] = {
            cat: feature_idx + i for i, cat in enumerate(categories)
        }
        feature_idx += len(categories)

    total_features = feature_idx
    print(f"Total one-hot features: {total_features}")

    # -------------------------
    # Pass 2: write LIBSVM
    # -------------------------
    print(f"\n[pass 2] Converting to LIBSVM format")

    total_rows = 0
    with open(csv_path, 'r') as f:
        next(f)
        for _ in f:
            total_rows += 1

    train_rows = total_rows - test_size if test_size > 0 else total_rows

    print(f"Total rows: {total_rows}")
    print(f"Train rows: {train_rows}")
    print(f"Test rows: {total_rows - train_rows}")

    with open(csv_path, 'r') as f_in:
        next(f_in)

        with open(train_out, 'w') as f_train:
            f_test = open(test_out, 'w') if test_size > 0 else None

            row_num = 0
            for line in f_in:
                row_num += 1
                if row_num % CHUNK_SIZE == 0:
                    print(f"  Processed {row_num}/{total_rows} rows...")

                values = line.strip().split(',')

                # -------------------------
                # Binary label
                # -------------------------
                try:
                    claim_amount = float(values[label_idx])
                    label = 1 if claim_amount > 0 else 0
                except ValueError:
                    label = 0

                features = {}

                for col_idx, col_name in enumerate(header):
                    if col_name == label_col:
                        continue

                    val = values[col_idx].strip()

                    if col_name in num_feature_map:
                        try:
                            num_val = float(val)
                            if num_val != 0:
                                features[num_feature_map[col_name]] = num_val
                        except ValueError:
                            pass

                    elif col_name in cat_feature_map:
                        if val and val in cat_feature_map[col_name]:
                            features[cat_feature_map[col_name][val]] = 1.0

                f_out = f_train if row_num <= train_rows else f_test
                write_libsvm_row(f_out, label, features)

            if f_test:
                f_test.close()

    print("\nDone.")


# -------------------------
# Entry point
# -------------------------
in_folder = "data"
out_folder = "data"

if __name__ == "__main__":
    csv_to_libsvm(
        csv_path=f"{in_folder}/train_set.csv",
        train_out=f"{out_folder}/allstate.train",
        test_out=f"{out_folder}/allstate.test",
        test_size=1_000_000,
    )
