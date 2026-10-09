import numpy as np


class Activation:
    def forward(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        z_clipped = np.clip(z, -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return a * (1.0 - a)


class ReLU(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        return (z > 0.0).astype(np.float64)


class LeakyReLU(Activation):
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.where(z > 0.0, z, self.alpha * z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        return np.where(z > 0.0, 1.0, self.alpha)


class Tanh(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return 1.0 - np.square(a)


class Softmax(Activation):
    def forward(self, z: np.ndarray) -> np.ndarray:
        shift_z = z - np.max(z, axis=-1, keepdims=True)
        exps = np.exp(shift_z)
        return exps / np.sum(exps, axis=-1, keepdims=True)

    def derivative(self, z: np.ndarray, a: np.ndarray = None) -> np.ndarray:
        if a is None:
            a = self.forward(z)
        return a * (1.0 - a)


def get_activation(name: str) -> Activation:
    name = name.lower().strip()
    if name in ('sigmoid', 'sigm'):
        return Sigmoid()
    elif name in ('relu',):
        return ReLU()
    elif name in ('leaky_relu', 'leakyrelu'):
        return LeakyReLU()
    elif name in ('tanh',):
        return Tanh()
    elif name in ('softmax',):
        return Softmax()
    raise ValueError(f"Unknown activation: {name}")
