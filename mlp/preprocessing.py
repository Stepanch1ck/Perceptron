import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any


class DatasetPreprocessor:
    FEATURE_NAMES = [
        'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
        'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
        'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    def __init__(self, normalization: str = 'standard'):
        self.normalization = normalization
        self.class_means: Dict[int, np.ndarray] = {}
        self.global_means: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.min_val: Optional[np.ndarray] = None
        self.max_val: Optional[np.ndarray] = None
        self.is_fitted = False

    @staticmethod
    def load_dataframe(csv_path: str) -> pd.DataFrame:
        preview = pd.read_csv(csv_path, nrows=5)
        if preview.shape[1] == 32:
            if 'diagnosis' in [str(c).lower() for c in preview.columns]:
                df = pd.read_csv(csv_path)
            else:
                cols = ['id', 'diagnosis'] + DatasetPreprocessor.FEATURE_NAMES
                df = pd.read_csv(csv_path, header=None, names=cols)
        elif preview.shape[1] == 31:
            cols = ['diagnosis'] + DatasetPreprocessor.FEATURE_NAMES
            df = pd.read_csv(csv_path, header=None, names=cols)
        else:
            df = pd.read_csv(csv_path)
        return df

    def parse_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        df_clean = df.copy()
        target_col = None
        for col in df_clean.columns:
            if str(col).lower() in ('diagnosis', 'label', 'target', '1'):
                target_col = col
                break

        if target_col is None and df_clean.shape[1] >= 2:
            unique_vals = set(df_clean.iloc[:, 1].dropna().astype(str).str.strip().str.upper())
            if unique_vals.issubset({'M', 'B', '0', '1'}):
                target_col = df_clean.columns[1]

        y_labels = None
        y_one_hot = None

        if target_col is not None:
            raw_labels = df_clean[target_col].astype(str).str.strip().str.upper()
            label_map = {'B': 0, 'BENIGN': 0, '0': 0, 'M': 1, 'MALIGNANT': 1, '1': 1}
            y_labels = raw_labels.map(label_map).to_numpy()
            df_clean = df_clean.drop(columns=[target_col])

        for col in list(df_clean.columns):
            if str(col).lower() in ('id', '0') or 'id' in str(col).lower():
                df_clean = df_clean.drop(columns=[col])
                break

        X = df_clean.to_numpy(dtype=np.float64)

        if y_labels is not None:
            num_samples = len(y_labels)
            y_one_hot = np.zeros((num_samples, 2), dtype=np.float64)
            for i, val in enumerate(y_labels):
                if not np.isnan(val):
                    y_one_hot[i, int(val)] = 1.0

        return X, y_labels, y_one_hot

    def impute_missing_values(self, X: np.ndarray, y_labels: Optional[np.ndarray] = None, fit: bool = False) -> np.ndarray:
        X_imputed = X.copy()
        num_features = X.shape[1]

        if fit:
            self.global_means = np.nanmean(X_imputed, axis=0)
            self.global_means = np.nan_to_num(self.global_means, nan=0.0)

            if y_labels is not None:
                for cls in (0, 1):
                    mask = (y_labels == cls)
                    if np.any(mask):
                        c_mean = np.nanmean(X_imputed[mask], axis=0)
                        c_mean = np.where(np.isnan(c_mean), self.global_means, c_mean)
                        self.class_means[cls] = c_mean
                    else:
                        self.class_means[cls] = self.global_means.copy()

        has_nans = np.isnan(X_imputed)
        if np.any(has_nans):
            for i in range(X_imputed.shape[0]):
                for j in range(num_features):
                    if np.isnan(X_imputed[i, j]):
                        if y_labels is not None and not np.isnan(y_labels[i]) and int(y_labels[i]) in self.class_means:
                            cls = int(y_labels[i])
                            X_imputed[i, j] = self.class_means[cls][j]
                        else:
                            X_imputed[i, j] = self.global_means[j]

        return X_imputed

    def fit(self, X: np.ndarray, y_labels: Optional[np.ndarray] = None):
        X_clean = self.impute_missing_values(X, y_labels, fit=True)
        self.mean = np.mean(X_clean, axis=0)
        self.std = np.std(X_clean, axis=0)
        self.std = np.where(self.std == 0.0, 1.0, self.std)

        self.min_val = np.min(X_clean, axis=0)
        self.max_val = np.max(X_clean, axis=0)
        range_val = self.max_val - self.min_val
        self.range_val = np.where(range_val == 0.0, 1.0, range_val)
        self.is_fitted = True

    def transform(self, X: np.ndarray, y_labels: Optional[np.ndarray] = None) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor is not fitted.")

        X_clean = self.impute_missing_values(X, y_labels, fit=False)
        if self.normalization == 'standard':
            return (X_clean - self.mean) / self.std
        elif self.normalization == 'minmax':
            return (X_clean - self.min_val) / self.range_val
        elif self.normalization == 'none':
            return X_clean
        raise ValueError(f"Unknown normalization: {self.normalization}")

    def fit_transform(self, X: np.ndarray, y_labels: Optional[np.ndarray] = None) -> np.ndarray:
        self.fit(X, y_labels)
        return self.transform(X, y_labels)

    def get_state(self) -> Dict[str, Any]:
        return {
            'normalization': self.normalization,
            'class_means_0': self.class_means.get(0, None),
            'class_means_1': self.class_means.get(1, None),
            'global_means': self.global_means,
            'mean': self.mean,
            'std': self.std,
            'min_val': self.min_val,
            'max_val': self.max_val,
            'is_fitted': self.is_fitted
        }

    def set_state(self, state: Dict[str, Any]):
        self.normalization = state.get('normalization', 'standard')
        self.global_means = state.get('global_means', None)
        self.class_means = {}
        if state.get('class_means_0') is not None:
            self.class_means[0] = state['class_means_0']
        if state.get('class_means_1') is not None:
            self.class_means[1] = state['class_means_1']
        self.mean = state.get('mean', None)
        self.std = state.get('std', None)
        self.min_val = state.get('min_val', None)
        self.max_val = state.get('max_val', None)
        if self.min_val is not None and self.max_val is not None:
            range_val = self.max_val - self.min_val
            self.range_val = np.where(range_val == 0.0, 1.0, range_val)
        self.is_fitted = state.get('is_fitted', True)
