import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict, Any
from sklearn.model_selection import KFold
from scipy.sparse import csr_matrix, hstack


class OneHotEncoder:
    """
    Custom vectorized one-hot encoder with drop='first' option.
    Optional sparse matrix output for memory efficiency.
    """

    def __init__(self, drop: str = "first", sparse: bool = False):
        self.drop = drop
        self.sparse = sparse
        self.categories_: List[List[Any]] = []
        self.n_features_in_: Optional[int] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.n_features_in_ = X.shape[1]
        self.categories_ = []

        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            unique_cats = sorted([cat for cat in np.unique(col) if pd.notna(cat)])
            self.categories_.append(unique_cats)

        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, csr_matrix]:
        if isinstance(X, pd.DataFrame):
            X = X.values

        rows = X.shape[0]
        encoded_blocks = []

        for col_idx, categories in enumerate(self.categories_):
            cats_to_encode = (
                categories[1:]
                if (self.drop == "first" and len(categories) > 0)
                else categories
            )

            if not cats_to_encode:
                continue

            col_data = X[:, col_idx]
            cats_arr = np.array(cats_to_encode)
            block = (col_data[:, np.newaxis] == cats_arr).astype(np.float64)

            if self.sparse:
                block = csr_matrix(block)
            encoded_blocks.append(block)

        if not encoded_blocks:
            return np.zeros((rows, 0), dtype=np.float64)

        if self.sparse:
            return hstack(encoded_blocks)
        return np.hstack(encoded_blocks)

    def fit_transform(
        self, X: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, csr_matrix]:
        return self.fit(X).transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        names = []
        for col_idx, categories in enumerate(self.categories_):
            cats = (
                categories[1:]
                if (self.drop == "first" and len(categories) > 0)
                else categories
            )
            for cat in cats:
                prefix = input_features[col_idx] if input_features else f"col{col_idx}"
                names.append(f"{prefix}_{cat}")
        return names


class TargetEncoder:
    """
    Custom Target Encoder with Laplace smoothing and cross-validation.
    Optimized with numpy operations instead of pandas groupby.
    """

    def __init__(self, smooth: float = 1.0, cv: int = 5):
        self.smooth = smooth
        self.cv = cv
        self.global_mean_: Optional[float] = None
        self.encoding_map_: Dict[Any, float] = {}

    def _to_1d_array(self, arr: Union[np.ndarray, pd.Series]) -> np.ndarray:
        if isinstance(arr, (pd.Series, pd.DataFrame)):
            return arr.values.ravel()
        return np.asarray(arr).ravel()

    def _compute_smoothed_values(
        self, X: np.ndarray, y: np.ndarray, global_mean: float
    ) -> Dict[Any, float]:
        """
        Optimized version using numpy unique instead of pandas groupby.
        2-5x faster for large datasets.
        """
        # Handle NaN values
        nan_mask = pd.isna(X)
        if np.any(nan_mask):
            X = X[~nan_mask]
            y = y[~nan_mask]

        if len(X) == 0:
            return {}

        # Vectorized computation using numpy
        unique_cats, inverse = np.unique(X, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=y.astype(np.float64))

        # Compute means, avoiding division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.divide(
                sums,
                counts,
                out=np.zeros_like(sums, dtype=np.float64),
                where=counts > 0,
            )

        # Apply smoothing: (n * mean + smooth * global_mean) / (n + smooth)
        smoothed = (counts * means + self.smooth * global_mean) / (counts + self.smooth)

        # Where count is 0, use global mean
        smoothed[counts == 0] = global_mean

        return dict(zip(unique_cats, smoothed))

    def fit(self, X: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]):
        X_arr = self._to_1d_array(X)
        y_arr = self._to_1d_array(y)

        non_nan_mask = ~pd.isna(X_arr)
        self.global_mean_ = float(np.mean(y_arr))

        self.encoding_map_ = self._compute_smoothed_values(
            X_arr[non_nan_mask], y_arr[non_nan_mask], self.global_mean_
        )
        return self

    def transform(self, X: Union[np.ndarray, pd.Series]) -> np.ndarray:
        if self.global_mean_ is None:
            raise RuntimeError("Must call fit() before transform()")

        X_arr = self._to_1d_array(X)
        encoded = (
            pd.Series(X_arr).map(self.encoding_map_).fillna(self.global_mean_).values
        )
        return encoded.reshape(-1, 1)

    def fit_transform(
        self, X: Union[np.ndarray, pd.Series], y: Union[np.ndarray, pd.Series]
    ) -> np.ndarray:
        X_arr = self._to_1d_array(X)
        y_arr = self._to_1d_array(y)

        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        encoded_train = np.zeros(len(X_arr))

        # Out-of-fold calculations WITHOUT calling self.fit() beforehand
        for train_idx, val_idx in kf.split(X_arr):
            X_train, y_train = X_arr[train_idx], y_arr[train_idx]
            X_val = X_arr[val_idx]

            fold_global_mean = float(np.mean(y_train))
            train_mask = ~pd.isna(X_train)

            fold_map = self._compute_smoothed_values(
                X_train[train_mask], y_train[train_mask], fold_global_mean
            )

            encoded_train[val_idx] = (
                pd.Series(X_val).map(fold_map).fillna(fold_global_mean).values
            )

        # Fit on full data after to store true state for test data transforms
        self.fit(X_arr, y_arr)
        return encoded_train.reshape(-1, 1)
