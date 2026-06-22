import numpy as np
import pandas as pd
from typing import Union, Optional
from sklearn.neighbors import BallTree


class SimpleImputer:
    """
    Custom simple imputer for mean/median imputation.
    Ensures safe broadcasting and fallback for entirely NaN columns.
    """

    def __init__(self, strategy: str = "mean"):
        valid_strategies = ["mean", "median"]
        if strategy not in valid_strategies:
            raise ValueError(f"Strategy must be one of {valid_strategies}")
        self.strategy = strategy
        self.statistics_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.strategy == "mean":
            self.statistics_ = np.nanmean(X, axis=0)
        else:
            self.statistics_ = np.nanmedian(X, axis=0)

        self.statistics_ = np.nan_to_num(self.statistics_, nan=0.0)
        return self

    def transform(
        self, X: Union[np.ndarray, pd.DataFrame], copy: bool = True
    ) -> np.ndarray:
        if self.statistics_ is None:
            raise RuntimeError("Must call fit() before transform()")

        if copy:
            X_out = (
                np.array(X, copy=True, dtype=np.float64)
                if isinstance(X, np.ndarray)
                else X.values.copy()
            )
        else:
            X_out = X if isinstance(X, np.ndarray) else X.values

        nan_indices = np.where(np.isnan(X_out))
        X_out[nan_indices] = np.take(self.statistics_, nan_indices[1])
        return X_out

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


class KNNImputer:
    """
    Fast NaN-aware Vectorized KNN Imputer with optional KD-Tree optimization.
    """

    def __init__(self, k: int = 5, use_tree: bool = True):
        self.k = k
        self.use_tree = use_tree
        self.reference_data_: Optional[np.ndarray] = None
        self.global_means_: Optional[np.ndarray] = None
        self.tree: Optional[BallTree] = None
        self._tree_imputer_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.reference_data_ = np.array(X, copy=True, dtype=np.float64)
        self.global_means_ = np.nanmean(X, axis=0)
        self.global_means_ = np.nan_to_num(self.global_means_, nan=0.0)

        # Build KD-Tree for fast neighbor search
        if self.use_tree:
            # Fill NaNs with 0 for tree construction (only for distance computation)
            X_filled = np.nan_to_num(X, nan=0.0)
            self.tree = BallTree(X_filled, metric="euclidean")

        return self

    def _nan_euclidean_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Euclidean distance ignoring NaN values."""
        mask = ~np.isnan(x) & ~np.isnan(y)
        if not np.any(mask):
            return np.inf
        weight = len(x) / np.sum(mask)
        return np.sqrt(weight * np.sum((x[mask] - y[mask]) ** 2))

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.reference_data_ is None or self.global_means_ is None:
            raise RuntimeError("Must call fit() before transform()")

        X_out = (
            np.array(X, copy=True, dtype=np.float64)
            if isinstance(X, np.ndarray)
            else X.values.copy()
        )
        n_samples, n_features = X_out.shape

        # Locate rows containing NaNs
        nan_rows = np.where(np.isnan(X_out).any(axis=1))[0]

        if self.use_tree and self.tree is not None:
            # Use KD-Tree for fast neighbor search
            for i in nan_rows:
                row = X_out[i]
                missing_mask = np.isnan(row)

                # Fill NaN with 0 for tree query (just for distance computation)
                row_filled = np.nan_to_num(row, nan=0.0)
                distances, indices = self.tree.query(
                    row_filled.reshape(1, -1), k=self.k
                )

                # Get neighbor values
                neighbors = self.reference_data_[indices[0]]

                # Calculate nanmean horizontally along the neighbor matrix
                with np.errstate(all="ignore"):
                    neighbor_means = np.nanmean(neighbors, axis=0)

                # Fill missing entries
                fallback_mask = np.isnan(neighbor_means) & missing_mask
                replace_mask = missing_mask & ~fallback_mask

                X_out[i, replace_mask] = neighbor_means[replace_mask]
                X_out[i, fallback_mask] = self.global_means_[fallback_mask]

        else:
            # Fallback to brute force (original vectorized method)
            for i in nan_rows:
                row = X_out[i]
                missing_mask = np.isnan(row)

                # Vectorized distance computation
                diff = self.reference_data_ - row
                coord_mask = ~np.isnan(diff)
                diff[~coord_mask] = 0.0
                sq_distances = np.sum(diff**2, axis=1)

                present_counts = np.sum(coord_mask, axis=1)
                with np.errstate(divide="ignore", invalid="ignore"):
                    weights = n_features / present_counts
                    weights[present_counts == 0] = np.inf
                    distances = np.sqrt(weights * sq_distances)

                if np.all(np.isinf(distances)):
                    X_out[i, missing_mask] = self.global_means_[missing_mask]
                    continue

                k_adjusted = min(self.k, len(distances) - 1)
                if k_adjusted < 1:
                    k_indices = np.array([0])
                else:
                    k_indices = np.argpartition(distances, k_adjusted)[: self.k]

                neighbors = self.reference_data_[k_indices]

                with np.errstate(all="ignore"):
                    neighbor_means = np.nanmean(neighbors, axis=0)

                fallback_mask = np.isnan(neighbor_means) & missing_mask
                replace_mask = missing_mask & ~fallback_mask

                X_out[i, replace_mask] = neighbor_means[replace_mask]
                X_out[i, fallback_mask] = self.global_means_[fallback_mask]

        return X_out

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)
