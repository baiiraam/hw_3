# import numpy as np
# import pandas as pd
# from typing import Union, Optional, Tuple

# class StandardScaler:
#     """Standardize features by removing mean and scaling to unit variance."""

#     def __init__(self):
#         self.mean_: Optional[np.ndarray] = None
#         self.scale_: Optional[np.ndarray] = None

#     def fit(self, X: Union[np.ndarray, pd.DataFrame]):
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         self.mean_ = np.nanmean(X, axis=0)
#         self.scale_ = np.nanstd(X, axis=0)
#         self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
#         return self

#     def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         if self.mean_ is None or self.scale_ is None:
#             raise RuntimeError("Must call fit() before transform()")
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         return (X - self.mean_) / self.scale_

#     def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         if self.mean_ is None or self.scale_ is None:
#             raise RuntimeError("Must call fit() before inverse_transform()")
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         return (X * self.scale_) + self.mean_

#     # ✅ ADDED: fit_transform method
#     def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         return self.fit(X).transform(X)


# class MinMaxScaler:
#     """Transform features by scaling each feature to a given range [0, 1]."""

#     # ✅ FIXED: Added feature_range parameter
#     def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
#         self.feature_range = feature_range
#         self.range_min_ = feature_range[0]
#         self.range_max_ = feature_range[1]
#         self.data_min_: Optional[np.ndarray] = None
#         self.data_max_: Optional[np.ndarray] = None
#         self.scale_: Optional[np.ndarray] = None

#     def fit(self, X: Union[np.ndarray, pd.DataFrame]):
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         self.data_min_ = np.nanmin(X, axis=0)
#         self.data_max_ = np.nanmax(X, axis=0)
#         self.scale_ = self.data_max_ - self.data_min_
#         self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
#         return self

#     def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         if self.data_min_ is None or self.scale_ is None:
#             raise RuntimeError("Must call fit() before transform()")
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         return (X - self.data_min_) / self.scale_

#     def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         if self.data_min_ is None or self.scale_ is None:
#             raise RuntimeError("Must call fit() before inverse_transform()")
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         return (X * self.scale_) + self.data_min_

#     # ✅ ADDED: fit_transform method
#     def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         return self.fit(X).transform(X)


# class RobustScaler:
#     """Scale features using statistics that are robust to outliers."""

#     def __init__(self):
#         self.center_: Optional[np.ndarray] = None
#         self.scale_: Optional[np.ndarray] = None

#     def fit(self, X: Union[np.ndarray, pd.DataFrame]):
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         self.center_ = np.nanmedian(X, axis=0)
#         q75 = np.nanpercentile(X, 75, axis=0)
#         q25 = np.nanpercentile(X, 25, axis=0)
#         self.scale_ = q75 - q25
#         self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
#         return self

#     def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         if self.center_ is None or self.scale_ is None:
#             raise RuntimeError("Must call fit() before transform()")
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         return (X - self.center_) / self.scale_

#     def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         if self.center_ is None or self.scale_ is None:
#             raise RuntimeError("Must call fit() before inverse_transform()")
#         if isinstance(X, pd.DataFrame):
#             X = X.values

#         return (X * self.scale_) + self.center_

#     # ✅ ADDED: fit_transform method
#     def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
#         return self.fit(X).transform(X)


import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple


class StandardScaler:
    """Standardize features by removing mean and scaling to unit variance."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Must call fit() before transform()")
        if isinstance(X, pd.DataFrame):
            X = X.values

        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        if isinstance(X, pd.DataFrame):
            X = X.values

        return (X * self.scale_) + self.mean_

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Transform features by scaling each feature to a given range [0, 1]."""

    def __init__(self, feature_range: Tuple[float, float] = (0, 1), clip: bool = False):
        self.feature_range = feature_range
        self.range_min_ = feature_range[0]
        self.range_max_ = feature_range[1]
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.clip = clip  # If True, clip values to training min/max

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.scale_ = self.data_max_ - self.data_min_
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.data_min_ is None or self.scale_ is None:
            raise RuntimeError("Must call fit() before transform()")
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.clip:
            # Clip extreme values that exceed training min/max
            X = np.clip(X, self.data_min_, self.data_max_)

        return (X - self.data_min_) / self.scale_

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.data_min_ is None or self.scale_ is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        if isinstance(X, pd.DataFrame):
            X = X.values

        return (X * self.scale_) + self.data_min_

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)


class RobustScaler:
    """Scale features using statistics that are robust to outliers."""

    def __init__(self):
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        if isinstance(X, pd.DataFrame):
            X = X.values

        self.center_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.scale_ = q75 - q25
        self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Must call fit() before transform()")
        if isinstance(X, pd.DataFrame):
            X = X.values

        return (X - self.center_) / self.scale_

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Must call fit() before inverse_transform()")
        if isinstance(X, pd.DataFrame):
            X = X.values

        return (X * self.scale_) + self.center_

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        return self.fit(X).transform(X)
