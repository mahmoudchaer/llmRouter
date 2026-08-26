from __future__ import annotations

import numpy as np


class TrainOnlyStructuralScaler:
    """Scales only the final structural block and is fit on training rows only."""
    def __init__(self, semantic_dim: int):
        from sklearn.preprocessing import StandardScaler
        self.semantic_dim = semantic_dim
        self.scaler = StandardScaler()

    def fit(self, X_train: np.ndarray):
        self.scaler.fit(X_train[:, self.semantic_dim:])
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        out = np.asarray(X, dtype=np.float32).copy()
        out[:, self.semantic_dim:] = self.scaler.transform(out[:, self.semantic_dim:])
        return out

