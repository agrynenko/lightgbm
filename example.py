"""
Example usage of the simplified LightGBM implementation.
"""

import numpy as np
from lightgbm_simple import LightGBM


def example_regression():
    """Example: Regression task"""
    print("=" * 60)
    print("Example 1: Regression")
    print("=" * 60)

    # Generate synthetic regression data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Create a simple linear relationship with noise
    X = np.random.randn(n_samples, n_features)
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 0.5 * np.random.randn(n_samples)

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Create and train model
    print("\nTraining LightGBM model...")
    model = LightGBM(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        min_child_samples=10,
        top_rate=0.1,  # GOSS: keep top 10% of samples
        other_rate=0.5,  # GOSS: sample 50% of remaining
        random_state=42,
    )

    model.fit(X_train, y_train, loss_type="mse")

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_mse = np.mean((y_train - y_pred_train) ** 2)
    test_mse = np.mean((y_test - y_pred_test) ** 2)

    print(f"\nResults:")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Train RMSE: {np.sqrt(train_mse):.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test RMSE: {np.sqrt(test_mse):.4f}")


def example_with_different_parameters():
    """Example: Using different parameters"""
    print("\n" + "=" * 60)
    print("Example 2: Different Parameters")
    print("=" * 60)

    # Generate data
    np.random.seed(123)
    n_samples = 500
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] ** 2 + np.sin(X[:, 1]) + 0.3 * np.random.randn(n_samples)

    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Model with different parameters
    print("\nTraining with different GOSS parameters...")
    model = LightGBM(
        n_estimators=30,
        learning_rate=0.15,
        max_depth=3,
        min_child_samples=5,
        top_rate=0.2,  # Keep more top samples
        other_rate=0.3,  # Sample fewer other samples
        num_bins=128,  # Fewer bins for faster training
        random_state=123,
    )

    model.fit(X_train, y_train, loss_type="mse")

    y_pred = model.predict(X_test)
    test_mse = np.mean((y_test - y_pred) ** 2)

    print(f"\nResults:")
    print(f"  Test MSE: {test_mse:.4f}")
    print(f"  Test RMSE: {np.sqrt(test_mse):.4f}")


if __name__ == "__main__":
    example_regression()
    example_with_different_parameters()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
