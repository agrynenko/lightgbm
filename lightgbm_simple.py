"""
LightGBM Implementation (Simplified)
Based on: "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (NIPS 2017)

This implementation includes:
- Gradient-based One-Side Sampling (GOSS)
- Leaf-wise tree growth
- Histogram-based algorithm for finding splits

Note: Exclusive Feature Bundling (EFB) is not implemented in this version.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TreeNode:
    """Represents a node in the decision tree."""

    feature_idx: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None
    value: float = 0.0  # Prediction value for leaf nodes
    is_leaf: bool = False


class Histogram:
    """Histogram for efficient split finding."""

    def __init__(self, num_bins: int = 255):
        self.num_bins = num_bins
        self.bins = np.zeros(num_bins)
        self.bin_counts = np.zeros(num_bins)
        self.bin_edges = None

    def build(self, values: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """Build histogram from values, gradients, and hessians."""
        if len(values) == 0:
            return

        # Create bins
        min_val = values.min()
        max_val = values.max()

        if min_val == max_val:
            self.bin_edges = np.array([min_val, max_val + 1e-7])
            self.bins[0] = np.sum(gradients)
            self.bin_counts[0] = np.sum(hessians)  # Use sum of hessians, not count
            return

        # Create bin edges
        self.bin_edges = np.linspace(min_val, max_val, self.num_bins + 1)

        # Assign values to bins
        bin_indices = np.digitize(values, self.bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)

        # Aggregate gradients and hessians per bin
        for i in range(self.num_bins):
            mask = bin_indices == i
            if np.any(mask):
                self.bins[i] = np.sum(gradients[mask])
                self.bin_counts[i] = np.sum(hessians[mask])  # Sum of hessians

    def find_best_split(
        self, min_child_samples: int, lambda_l2: float, min_gain_to_split: float
    ) -> Tuple[float, float]:
        """
        Find best split point using histogram.
        Returns: (best_gain, best_threshold)
        """
        if self.bin_edges is None or len(self.bin_edges) < 2:
            return (-np.inf, 0.0)

        total_grad = np.sum(self.bins)
        total_hess = np.sum(self.bin_counts)

        # Check minimum samples constraint (using hessian sum as proxy)
        if total_hess < 2 * min_child_samples:
            return (-np.inf, 0.0)

        best_gain = -np.inf
        best_threshold = self.bin_edges[0]

        left_grad = 0.0
        left_hess = 0.0

        # Try each bin as a split point
        for i in range(self.num_bins - 1):
            left_grad += self.bins[i]
            left_hess += self.bin_counts[i]

            if left_hess < min_child_samples:
                continue

            right_grad = total_grad - left_grad
            right_hess = total_hess - left_hess

            if right_hess < min_child_samples:
                break

            # Calculate gain using variance reduction formula
            # Gain = left_score + right_score - parent_score
            left_score = (left_grad**2) / (left_hess + lambda_l2)
            right_score = (right_grad**2) / (right_hess + lambda_l2)
            parent_score = (total_grad**2) / (total_hess + lambda_l2)

            gain = left_score + right_score - parent_score

            if gain > best_gain:
                best_gain = gain
                best_threshold = self.bin_edges[i + 1]

        if best_gain < min_gain_to_split:
            return (-np.inf, 0.0)

        return (best_gain, best_threshold)


class DecisionTree:
    """Decision tree with leaf-wise growth."""

    def __init__(
        self,
        max_depth: int = -1,
        min_child_samples: int = 20,
        lambda_l2: float = 0.1,
        min_gain_to_split: float = 0.0,
        num_bins: int = 255,
    ):
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.num_bins = num_bins
        self.root = None

    def _calculate_leaf_value(
        self, gradients: np.ndarray, hessians: np.ndarray
    ) -> float:
        """Calculate optimal leaf value using Newton's method."""
        if len(gradients) == 0:
            return 0.0
        # Optimal leaf value: -sum(gradients) / (sum(hessians) + lambda_l2)
        sum_grad = np.sum(gradients)
        sum_hess = np.sum(hessians)
        if sum_hess + self.lambda_l2 == 0:
            return 0.0
        return -sum_grad / (sum_hess + self.lambda_l2)

    def _find_best_split(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[int, float, float]:
        """Find the best split across all features."""
        best_gain = -np.inf
        best_feature = -1
        best_threshold = 0.0

        n_features = X.shape[1]

        for feature_idx in range(n_features):
            feature_values = X[indices, feature_idx]

            # Build histogram
            hist = Histogram(self.num_bins)
            hist.build(feature_values, gradients[indices], hessians[indices])

            # Find best split for this feature
            gain, threshold = hist.find_best_split(
                self.min_child_samples, self.lambda_l2, self.min_gain_to_split
            )

            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

        return (best_feature, best_threshold, best_gain)

    def _build_tree_leafwise(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        indices: np.ndarray,
        depth: int = 0,
    ) -> TreeNode:
        """Build tree using leaf-wise growth strategy."""
        node = TreeNode()

        # Check stopping conditions
        if len(indices) < 2 * self.min_child_samples:
            node.is_leaf = True
            node.value = self._calculate_leaf_value(
                gradients[indices], hessians[indices]
            )
            return node

        if self.max_depth > 0 and depth >= self.max_depth:
            node.is_leaf = True
            node.value = self._calculate_leaf_value(
                gradients[indices], hessians[indices]
            )
            return node

        # Find best split
        best_feature, best_threshold, best_gain = self._find_best_split(
            X, gradients, hessians, indices
        )

        if best_gain <= self.min_gain_to_split:
            node.is_leaf = True
            node.value = self._calculate_leaf_value(
                gradients[indices], hessians[indices]
            )
            return node

        # Split data
        feature_values = X[indices, best_feature]
        left_mask = feature_values <= best_threshold
        right_mask = ~left_mask

        left_indices = indices[left_mask]
        right_indices = indices[right_mask]

        if len(left_indices) == 0 or len(right_indices) == 0:
            node.is_leaf = True
            node.value = self._calculate_leaf_value(
                gradients[indices], hessians[indices]
            )
            return node

        # Create internal node
        node.feature_idx = best_feature
        node.threshold = best_threshold

        # Recursively build children (leaf-wise: always split the leaf with max gain)
        # For simplicity, we build both children here
        node.left = self._build_tree_leafwise(
            X, gradients, hessians, left_indices, depth + 1
        )
        node.right = self._build_tree_leafwise(
            X, gradients, hessians, right_indices, depth + 1
        )

        return node

    def fit(self, X: np.ndarray, gradients: np.ndarray, hessians: np.ndarray):
        """Fit the decision tree."""
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        self.root = self._build_tree_leafwise(X, gradients, hessians, indices)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the tree."""
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            node = self.root
            while not node.is_leaf:
                if X[i, node.feature_idx] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions[i] = node.value

        return predictions


def goss_sampling(
    gradients: np.ndarray, top_rate: float = 0.1, other_rate: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gradient-based One-Side Sampling (GOSS).

    According to the paper, GOSS keeps all instances with large gradients
    and randomly samples instances with small gradients.

    Args:
        gradients: Gradient values for each sample
        top_rate: Proportion of samples with large gradients to keep
        other_rate: Proportion of samples with small gradients to keep

    Returns:
        (selected_indices, weights) - indices of selected samples and their weights
    """
    n_samples = len(gradients)
    n_top = max(1, int(n_samples * top_rate))
    n_other = max(1, int(n_samples * other_rate))

    # Sort by absolute gradient (descending)
    abs_gradients = np.abs(gradients)
    sorted_indices = np.argsort(abs_gradients)[::-1]

    # Select top samples (all of them)
    top_indices = sorted_indices[:n_top]

    # Randomly sample from the rest
    remaining_indices = sorted_indices[n_top:]
    if len(remaining_indices) > 0:
        n_other_actual = min(n_other, len(remaining_indices))
        other_indices = np.random.choice(
            remaining_indices, size=n_other_actual, replace=False
        )
    else:
        other_indices = np.array([], dtype=int)

    # Combine indices
    selected_indices = np.concatenate([top_indices, other_indices])

    # Calculate weights for GOSS
    # Top samples get weight 1.0, others get weight (1 - top_rate) / other_rate
    weights = np.ones(len(selected_indices))
    other_mask = np.isin(selected_indices, other_indices)
    if np.any(other_mask) and other_rate > 0:
        weights[other_mask] = (1.0 - top_rate) / other_rate

    return selected_indices, weights


class LightGBM:
    """
    LightGBM implementation with GOSS and leaf-wise growth.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = -1,
        min_child_samples: int = 20,
        lambda_l2: float = 0.1,
        min_gain_to_split: float = 0.0,
        top_rate: float = 0.1,
        other_rate: float = 0.5,
        num_bins: int = 255,
        random_state: Optional[int] = None,
    ):
        """
        Initialize LightGBM.

        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate (shrinkage)
            max_depth: Maximum tree depth (-1 for unlimited)
            min_child_samples: Minimum samples in a leaf
            lambda_l2: L2 regularization
            min_gain_to_split: Minimum gain to perform split
            top_rate: GOSS top rate
            other_rate: GOSS other rate
            num_bins: Number of histogram bins
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_samples = min_child_samples
        self.lambda_l2 = lambda_l2
        self.min_gain_to_split = min_gain_to_split
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.num_bins = num_bins
        self.trees = []
        # TODO: Add EFB (Exclusive Feature Bundling) support
        # This would include:
        # - Feature bundling mapping (which features are bundled together)
        # - Methods to bundle mutually exclusive features
        # - Update split finding to work with bundled features

        if random_state is not None:
            np.random.seed(random_state)

    def _loss_gradient(
        self, y_true: np.ndarray, y_pred: np.ndarray, loss_type: str = "mse"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradients and hessians for a given loss function.

        Returns:
            (gradients, hessians)
        """
        if loss_type == "mse":
            # For MSE: gradient = 2 * (y_pred - y_true), hessian = 2
            gradients = 2 * (y_pred - y_true)
            hessians = np.full_like(gradients, 2.0)
        elif loss_type == "logistic":
            # For logistic: gradient = y_pred - y_true, hessian = y_pred * (1 - y_pred)
            # Assuming y_pred is probability
            gradients = y_pred - y_true
            hessians = y_pred * (1 - y_pred) + 1e-8
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        return gradients, hessians

    def _apply_efb(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        TODO: Implement Exclusive Feature Bundling (EFB).

        EFB bundles mutually exclusive features (features that rarely take
        nonzero values simultaneously) to reduce the number of features.

        According to the paper:
        1. Build a graph where edges represent feature conflicts (non-zero overlap)
        2. Use graph coloring to find bundles (features that can be bundled together)
        3. Merge features in each bundle by adding offsets to feature values

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            (X_bundled, feature_mapping) - Bundled features and mapping from
            original features to bundled features
        """
        # Placeholder: return original features for now
        feature_mapping = {i: i for i in range(X.shape[1])}
        return X, feature_mapping

    def fit(self, X: np.ndarray, y: np.ndarray, loss_type: str = "mse"):
        """
        Fit the LightGBM model.

        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
            loss_type: Loss function type ('mse' or 'logistic')
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)

        self.trees = []

        for i in range(self.n_estimators):
            # Calculate gradients and hessians
            gradients, hessians = self._loss_gradient(y, y_pred, loss_type)

            # Apply GOSS sampling
            selected_indices, weights = goss_sampling(
                gradients, self.top_rate, self.other_rate
            )

            # Get sampled data
            X_sampled = X[selected_indices]
            # Weight the gradients and hessians according to GOSS
            gradients_sampled = gradients[selected_indices] * weights
            hessians_sampled = hessians[selected_indices] * weights

            # TODO: Apply Exclusive Feature Bundling (EFB) here
            # X_sampled, feature_mapping = self._apply_efb(X_sampled)
            # This would reduce the number of features by bundling mutually
            # exclusive features, improving training efficiency for high-dimensional
            # sparse data

            # Build tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_child_samples=self.min_child_samples,
                lambda_l2=self.lambda_l2,
                min_gain_to_split=self.min_gain_to_split,
                num_bins=self.num_bins,
            )
            tree.fit(X_sampled, gradients_sampled, hessians_sampled)

            # Update predictions
            tree_pred = tree.predict(X)
            y_pred += self.learning_rate * tree_pred

            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                mse = np.mean((y - y_pred) ** 2)
                print(f"Iteration {i + 1}/{self.n_estimators}, MSE: {mse:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        y_pred = np.zeros(X.shape[0])

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] + 2 * X[:, 1] + 0.5 * np.random.randn(n_samples)

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train model
    model = LightGBM(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        min_child_samples=10,
        top_rate=0.1,
        other_rate=0.5,
        random_state=42,
    )

    print("Training LightGBM...")
    model.fit(X_train, y_train, loss_type="mse")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test RMSE: {np.sqrt(mse):.4f}")
