# LightGBM Simplified Implementation

This is a simplified implementation of LightGBM based on the paper:
**"LightGBM: A Highly Efficient Gradient Boosting Decision Tree"** (NIPS 2017)

## Features Implemented

### ✅ Gradient-based One-Side Sampling (GOSS)

- Keeps all instances with large gradients
- Randomly samples instances with small gradients
- Applies appropriate weights to maintain unbiased estimation

### ✅ Leaf-wise Tree Growth

- Grows trees by selecting the leaf with maximum gain
- More efficient than level-wise growth
- Better accuracy with same number of leaves

### ✅ Histogram-based Algorithm

- Uses histograms to find optimal splits
- Reduces computation from O(data) to O(bins)
- Configurable number of bins (default: 255)

### ❌ Exclusive Feature Bundling (EFB)

- This feature bundles mutually exclusive features to reduce dimensionality

## Key Components

### `Histogram`

Efficiently aggregates gradients and hessians into bins for fast split finding.

### `DecisionTree`

Implements a decision tree with:

- Leaf-wise growth strategy
- Histogram-based split finding
- Regularization (L2)

### `LightGBM`

Main gradient boosting framework that:

- Uses GOSS for sample selection
- Builds trees iteratively
- Supports MSE and logistic loss

## Usage

```python
import numpy as np
from lightgbm_simple import LightGBM

# Generate data
X = np.random.randn(1000, 10)
y = X[:, 0] + 2 * X[:, 1] + 0.5 * np.random.randn(1000)

# Create and train model
model = LightGBM(
    n_estimators=50,
    learning_rate=0.1,
    max_depth=5,
    min_child_samples=10,
    top_rate=0.1,      # GOSS: keep top 10% of samples
    other_rate=0.5,    # GOSS: sample 50% of remaining
    random_state=42
)

model.fit(X, y, loss_type='mse')

# Make predictions
predictions = model.predict(X)
```

## Parameters

- `n_estimators`: Number of boosting rounds
- `learning_rate`: Shrinkage factor (default: 0.1)
- `max_depth`: Maximum tree depth (-1 for unlimited)
- `min_child_samples`: Minimum samples in a leaf
- `lambda_l2`: L2 regularization parameter
- `min_gain_to_split`: Minimum gain to perform a split
- `top_rate`: GOSS parameter - proportion of top samples to keep
- `other_rate`: GOSS parameter - proportion of other samples to sample
- `num_bins`: Number of histogram bins (default: 255)

## Algorithm Details

### GOSS (Gradient-based One-Side Sampling)

1. Sort instances by absolute gradient value
2. Keep top `top_rate * n` instances (all of them)
3. Randomly sample `other_rate * n` instances from the rest
4. Apply weights: top samples get weight 1.0, others get `(1 - top_rate) / other_rate`

This allows focusing computation on instances that contribute more to the gradient while maintaining unbiased estimation.

### Leaf-wise Growth

Instead of growing level by level (like XGBoost), LightGBM:

1. Finds the leaf with maximum gain
2. Splits that leaf
3. Repeats until stopping criteria met

This is more efficient and often achieves better accuracy.

### Histogram-based Split Finding

1. Discretize feature values into bins
2. Aggregate gradients and hessians per bin
3. Find best split by iterating over bins (O(bins) instead of O(data))

## Limitations

This is a simplified implementation for educational purposes. The full LightGBM includes:

- More sophisticated loss functions
- Categorical feature handling
- Parallel and distributed training
- GPU acceleration
- More optimizations

## References

- Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NIPS 2017.
- [Official LightGBM GitHub](https://github.com/microsoft/LightGBM)
