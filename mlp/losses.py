import numpy as np


class Loss:
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class CategoricalCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        clipped_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        return float(-np.sum(y_true * np.log(clipped_pred)) / y_pred.shape[0])

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return y_pred - y_true


class BinaryCrossEntropy(Loss):
    def __init__(self, eps: float = 1e-15):
        self.eps = eps

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        clipped_pred = np.clip(y_pred, self.eps, 1.0 - self.eps)
        if y_pred.ndim == 2 and y_pred.shape[1] == 2:
            return float(-np.sum(y_true * np.log(clipped_pred)) / y_pred.shape[0])
        return float(-np.mean(y_true * np.log(clipped_pred) + (1.0 - y_true) * np.log(1.0 - clipped_pred)))

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return y_pred - y_true


def get_loss(name: str) -> Loss:
    name = name.lower().strip()
    if name in ('binarycrossentropy', 'bce', 'binary_cross_entropy'):
        return BinaryCrossEntropy()
    elif name in ('categoricalcrossentropy', 'cce', 'crossentropy', 'cross_entropy'):
        return CategoricalCrossEntropy()
    raise ValueError(f"Unknown loss: {name}")
