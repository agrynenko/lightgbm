from collections import defaultdict

CHUNK_SIZE = 500_000

# Columns that should NEVER be categorical
EXCLUDE_COLS = {
    "Row_ID",
    "Household_ID",
    "Claim_Amount",
}


def detect_categorical_columns(csv_path):
    """
    Detect categorical columns and collect global categories.
    Returns: dict of {col_name: sorted_list_of_categories}
    """
    categories = defaultdict(set)
    header = None
    
    print(f"[pass 1] Scanning categories in {csv_path}")
    
    with open(csv_path, 'r') as f:
        header = f.readline().strip().split(',')
        
        line_count = 0
        for line in f:
            line_count += 1
            if line_count % CHUNK_SIZE == 0:
                print(f"  Scanned {line_count} rows...")
            
            values = line.strip().split(',')
            for col_idx, (col_name, val) in enumerate(zip(header, values)):
                if col_name in EXCLUDE_COLS:
                    continue
                
                # Try to detect if it's numeric
                try:
                    float(val)
                except (ValueError, AttributeError):
                    # Non-numeric or empty → categorical candidate
                    if val and val.strip():
                        categories[col_name].add(val.strip())
    
    # Convert sets to sorted lists for consistent ordering
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
        if val != 0:  # Only write non-zero values
            f.write(f" {idx}:{val}")
    f.write('\n')


def csv_to_libsvm(
    csv_path,
    train_out,
    test_out,
    test_size=1_000_000,
    label_col="Claim_Amount"
):
    # -------------------------
    # Pass 1: detect categories
    # -------------------------
    cat_mapping, header = detect_categorical_columns(csv_path)
    
    print(f"\nDetected {len(cat_mapping)} categorical columns:")
    print(sorted(cat_mapping.keys()))
    
    # Build feature index mapping
    # Numerical features come first, then one-hot encoded categoricals
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
    
    # Create mapping: col_name -> (start_idx, category_to_idx)
    feature_idx = 0
    num_feature_map = {}
    
    # Numerical features get sequential indices
    for col_name in num_cols:
        num_feature_map[col_name] = feature_idx
        feature_idx += 1
    
    # Categorical features get one-hot encoded indices
    cat_feature_map = {}
    for col_name in cat_cols:
        categories = cat_mapping[col_name]
        cat_to_idx = {cat: feature_idx + i for i, cat in enumerate(categories)}
        cat_feature_map[col_name] = cat_to_idx
        feature_idx += len(categories)
    
    total_features = feature_idx
    print(f"Total one-hot features: {total_features}")
    
    # -------------------------
    # Pass 2: write LIBSVM files
    # -------------------------
    print(f"\n[pass 2] Converting to LIBSVM format")
    
    # Count total rows first to determine split
    total_rows = 0
    with open(csv_path, 'r') as f:
        next(f)  # skip header
        for _ in f:
            total_rows += 1
    
    train_rows = total_rows - test_size if test_size > 0 else total_rows
    
    print(f"Total rows: {total_rows}")
    print(f"Train rows: {train_rows}")
    print(f"Test rows: {total_rows - train_rows}")
    
    with open(csv_path, 'r') as f_in:
        next(f_in)  # skip header
        
        with open(train_out, 'w') as f_train:
            f_test = open(test_out, 'w') if test_size > 0 else None
            
            row_num = 0
            for line in f_in:
                row_num += 1
                if row_num % CHUNK_SIZE == 0:
                    print(f"  Processed {row_num}/{total_rows} rows...")
                
                values = line.strip().split(',')
                
                # Get label
                label = float(values[label_idx])
                
                # Build feature dict
                features = {}
                
                # Add numerical features
                for col_idx, col_name in enumerate(header):
                    if col_name == label_col:
                        continue
                    
                    val = values[col_idx].strip()
                    
                    if col_name in num_feature_map:
                        # Numerical feature
                        try:
                            num_val = float(val)
                            if num_val != 0:
                                features[num_feature_map[col_name]] = num_val
                        except ValueError:
                            pass  # Missing or invalid → treat as 0
                    
                    elif col_name in cat_feature_map:
                        # Categorical feature → one-hot
                        if val and val in cat_feature_map[col_name]:
                            one_hot_idx = cat_feature_map[col_name][val]
                            features[one_hot_idx] = 1.0
                
                # Write to appropriate file
                f_out = f_train if row_num <= train_rows else f_test
                write_libsvm_row(f_out, label, features)
            
            if f_test:
                f_test.close()
    
    print("\nDone.")


in_folder = 'data'
out_folder = 'data'


if __name__ == "__main__":
    csv_to_libsvm(
        csv_path=f"{in_folder}/train_set.csv",
        train_out=f"{out_folder}/allstate.train",
        test_out=f"{out_folder}/allstate.test",
        test_size=1_000_000
    )
