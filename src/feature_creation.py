import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Optional
from itertools import combinations_with_replacement


class PolynomialFeatures:
    """Generate polynomial and interaction features with cached combinations."""

    def __init__(
        self, degree: int = 2, include_bias: bool = True, interaction_only: bool = False
    ):
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.n_features_in_: int = 0
        self.combinations_: List[Tuple[int, ...]] = []
        self._combos_cache: dict = {}

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # Check cache
        cache_key = (n_features, self.degree, self.interaction_only)
        if cache_key in self._combos_cache:
            self.combinations_ = self._combos_cache[cache_key]
            return self

        combo_list = []
        for d in range(1, self.degree + 1):
            for combos in combinations_with_replacement(range(n_features), d):
                if (
                    self.interaction_only
                    and len(combos) > 1
                    and len(set(combos)) != len(combos)
                ):
                    continue
                combo_list.append(combos)

        self.combinations_ = combo_list
        # Cache for future use
        self._combos_cache[cache_key] = combo_list
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples = X.shape[0]
        output_features = []

        if self.include_bias:
            output_features.append(np.ones((n_samples, 1)))

        for combo in self.combinations_:
            prod = np.prod(X[:, combo], axis=1, keepdims=True)
            output_features.append(prod)

        return np.hstack(output_features)

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_feature_names_out(
        self, input_features: Optional[List[str]] = None
    ) -> List[str]:
        if not self.combinations_:
            return []

        if input_features is None:
            if self.n_features_in_ == 0:
                return []
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        names = []
        if self.include_bias:
            names.append("bias")

        for combo in self.combinations_:
            if len(combo) == 1:
                names.append(input_features[combo[0]])
            else:
                terms = [input_features[idx] for idx in combo]
                names.append(" * ".join(terms))

        return names


class KBinsDiscretizer:
    """
    Vectorized KBinsDiscretizer avoiding loop steps.
    Handles exact-match maximum boundaries by clipping index buckets.
    """

    def __init__(
        self, n_bins: int = 5, strategy: str = "uniform", encode: str = "onehot-dense"
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.encode = encode
        self.bin_edges_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values.ravel()
        else:
            X_arr = X.ravel()

        non_nan_X = X_arr[~np.isnan(X_arr)]
        min_val, max_val = np.min(non_nan_X), np.max(non_nan_X)

        if self.strategy == "uniform":
            self.bin_edges_ = np.linspace(min_val, max_val, self.n_bins + 1)
        elif self.strategy == "quantile":
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges_ = np.percentile(non_nan_X, percentiles)
        else:
            raise ValueError("strategy must be 'uniform' or 'quantile'")

        # Ensure first and last edges handle boundaries
        self.bin_edges_[0] = (
            -np.inf if not np.isinf(self.bin_edges_[0]) else self.bin_edges_[0]
        )
        self.bin_edges_[-1] = (
            np.inf if not np.isinf(self.bin_edges_[-1]) else self.bin_edges_[-1]
        )

        return self

    def transform(self, X: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("Must call fit() before transform()")

        if isinstance(X, (pd.DataFrame, pd.Series)):
            X_arr = X.values.ravel()
        else:
            X_arr = X.ravel()

        bin_indices = np.digitize(X_arr, self.bin_edges_[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)
        bin_indices[np.isnan(X_arr)] = -1

        if self.encode == "onehot-dense":
            n_samples = len(X_arr)
            onehot = np.zeros((n_samples, self.n_bins), dtype=np.float64)
            valid_mask = bin_indices >= 0
            onehot[valid_mask, bin_indices[valid_mask]] = 1.0
            return onehot

        return bin_indices.reshape(-1, 1)

    def fit_transform(
        self, X: Union[np.ndarray, pd.Series, pd.DataFrame]
    ) -> np.ndarray:
        return self.fit(X).transform(X)

    def get_bin_edges(self) -> np.ndarray:
        if self.bin_edges_ is None:
            raise RuntimeError("Must call fit() before get_bin_edges()")
        return self.bin_edges_
