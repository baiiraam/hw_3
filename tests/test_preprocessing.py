import numpy as np
import pandas as pd
from src.imputers import SimpleImputer, KNNImputer
from src.encoders import OneHotEncoder, TargetEncoder
from src.scalers import StandardScaler, MinMaxScaler, RobustScaler
from src.feature_creation import PolynomialFeatures, KBinsDiscretizer
from scipy.sparse import issparse

np.random.seed(42)


def test_simple_imputer_mean():
    """Test SimpleImputer with mean strategy."""
    X = np.array([[1, 2], [np.nan, 3], [3, np.nan]])
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    expected = np.array([[1, 2], [2, 3], [3, 2.5]])
    np.testing.assert_array_almost_equal(X_imputed, expected)


def test_simple_imputer_median():
    """Test SimpleImputer with median strategy."""
    X = np.array([[1, 2], [np.nan, 3], [3, np.nan], [5, 6]])
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    expected = np.array([[1, 2], [3, 3], [3, 3], [5, 6]])
    np.testing.assert_array_almost_equal(X_imputed, expected)


def test_knn_imputer_basic():
    """Test KNNImputer on simple data."""
    X = np.array([[1, 2], [1, np.nan], [1, 4], [10, 20]])
    imputer = KNNImputer(k=2)
    X_imputed = imputer.fit_transform(X)

    assert abs(X_imputed[1, 1] - 3) < 1.5


def test_one_hot_encoder():
    """Test OneHotEncoder output dimensions."""
    X = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    encoder = OneHotEncoder(drop="first")
    X_encoded = encoder.fit_transform(X)

    assert X_encoded.shape[1] == 2
    assert X_encoded.shape[0] == 4


def test_target_encoder():
    """Test TargetEncoder with CV-safe encoding."""
    # Categories with different means: A=1.0, B=0.0, C=0.5
    X = np.array(["A", "A", "A", "B", "B", "B", "C", "C", "C"])
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1])  # C mean = 0.5

    encoder = TargetEncoder(smooth=0.0, cv=3)
    X_encoded = encoder.fit_transform(X, y)

    # Check shape
    assert X_encoded.shape == (9, 1)

    # Check ordering: A (1.0) > C (0.5) > B (0.0)
    a_vals = X_encoded[X == "A"]
    c_vals = X_encoded[X == "C"]
    b_vals = X_encoded[X == "B"]

    assert np.mean(a_vals) > np.mean(c_vals)
    assert np.mean(c_vals) > np.mean(b_vals)

    # Check transform on new data
    new_X = np.array(["A", "B", "C"])
    X_transformed = encoder.transform(new_X)
    assert X_transformed.shape == (3, 1)

    # Category ordering should be preserved
    assert X_transformed[0, 0] > X_transformed[1, 0]  # A > B
    assert X_transformed[0, 0] > X_transformed[2, 0]  # A > C
    assert X_transformed[2, 0] > X_transformed[1, 0]  # C > B


def test_standard_scaler():
    """Test StandardScaler mean and variance."""
    X = np.array([[1, 2], [2, 4], [3, 6]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert abs(np.mean(X_scaled[:, 0])) < 1e-10
    assert abs(np.mean(X_scaled[:, 1])) < 1e-10
    assert abs(np.std(X_scaled[:, 0]) - 1) < 1e-10


def test_minmax_scaler():
    """Test MinMaxScaler range."""
    X = np.array([[1, 10], [2, 20], [3, 30]])
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    assert np.min(X_scaled[:, 0]) >= 0
    assert np.max(X_scaled[:, 0]) <= 1
    assert X_scaled[0, 0] == 0
    assert X_scaled[2, 0] == 1


def test_robust_scaler():
    """Test RobustScaler uses median and IQR."""
    X = np.array([[1, 100], [2, 102], [3, 101], [1000, 200]])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    assert abs(np.median(X_scaled[:, 0])) < 1e-10


def test_polynomial_features():
    """Test PolynomialFeatures output shape."""
    X = np.array([[1, 2], [3, 4]])
    poly = PolynomialFeatures(degree=2, include_bias=True)
    X_poly = poly.fit_transform(X)

    assert X_poly.shape[1] == 6


def test_polynomial_features_no_bias():
    """Test PolynomialFeatures without bias."""
    X = np.array([[1, 2], [3, 4]])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    assert X_poly.shape[1] == 5


def test_discretizer_uniform():
    """Test KBinsDiscretizer uniform strategy."""
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    discretizer = KBinsDiscretizer(n_bins=5, strategy="uniform", encode="onehot-dense")
    X_binned = discretizer.fit_transform(X)

    assert X_binned.shape[1] == 5
    assert X_binned.shape[0] == 10
    assert np.all(np.sum(X_binned, axis=1) == 1)


def test_discretizer_quantile():
    """Test KBinsDiscretizer quantile strategy."""
    X = np.array([1, 1, 1, 1, 100, 100, 100, 100])
    discretizer = KBinsDiscretizer(n_bins=2, strategy="quantile", encode="onehot-dense")
    X_binned = discretizer.fit_transform(X)

    assert np.sum(X_binned[:, 0]) == 4
    assert np.sum(X_binned[:, 1]) == 4


def test_one_hot_encoder_sparse():
    """Test OneHotEncoder with sparse output."""
    X = pd.DataFrame({"color": ["red", "blue", "red", "green"]})
    encoder = OneHotEncoder(drop="first", sparse=True)
    X_encoded = encoder.fit_transform(X)

    # Should be a sparse matrix
    assert issparse(X_encoded)
    assert X_encoded.shape[1] == 2
    assert X_encoded.shape[0] == 4


def test_minmax_scaler_clip():
    """Test MinMaxScaler with clipping."""
    X = np.array([[1, 10], [2, 20], [3, 30]])
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler.fit(X)

    # Test with value outside training range
    X_test = np.array([[0, 5], [4, 40]])
    X_scaled = scaler.transform(X_test)

    # Clipped values should be within [0, 1]
    assert np.min(X_scaled[:, 0]) >= 0
    assert np.max(X_scaled[:, 0]) <= 1
