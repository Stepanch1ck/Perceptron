import numpy as np


class Optimizer:
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.step_count = 0

    def update(self, layer_idx: int, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray):
        raise NotImplementedError

    def step(self):
        self.step_count += 1


class SGD(Optimizer):
    def update(self, layer_idx: int, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray):
        new_w = w - self.learning_rate * dw
        new_b = b - self.learning_rate * db
        return new_w, new_b


class Momentum(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v_w = {}
        self.v_b = {}

    def update(self, layer_idx: int, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray):
        if layer_idx not in self.v_w:
            self.v_w[layer_idx] = np.zeros_like(w)
            self.v_b[layer_idx] = np.zeros_like(b)

        self.v_w[layer_idx] = self.momentum * self.v_w[layer_idx] + self.learning_rate * dw
        self.v_b[layer_idx] = self.momentum * self.v_b[layer_idx] + self.learning_rate * db
        return w - self.v_w[layer_idx], b - self.v_b[layer_idx]


class Adam(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m_w = {}
        self.m_b = {}
        self.v_w = {}
        self.v_b = {}

    def update(self, layer_idx: int, w: np.ndarray, b: np.ndarray, dw: np.ndarray, db: np.ndarray):
        if layer_idx not in self.m_w:
            self.m_w[layer_idx] = np.zeros_like(w)
            self.m_b[layer_idx] = np.zeros_like(b)
            self.v_w[layer_idx] = np.zeros_like(w)
            self.v_b[layer_idx] = np.zeros_like(b)

        t = max(1, self.step_count + 1)

        self.m_w[layer_idx] = self.beta1 * self.m_w[layer_idx] + (1.0 - self.beta1) * dw
        self.m_b[layer_idx] = self.beta1 * self.m_b[layer_idx] + (1.0 - self.beta1) * db

        self.v_w[layer_idx] = self.beta2 * self.v_w[layer_idx] + (1.0 - self.beta2) * np.square(dw)
        self.v_b[layer_idx] = self.beta2 * self.v_b[layer_idx] + (1.0 - self.beta2) * np.square(db)

        m_w_hat = self.m_w[layer_idx] / (1.0 - (self.beta1 ** t))
        m_b_hat = self.m_b[layer_idx] / (1.0 - (self.beta1 ** t))

        v_w_hat = self.v_w[layer_idx] / (1.0 - (self.beta2 ** t))
        v_b_hat = self.v_b[layer_idx] / (1.0 - (self.beta2 ** t))

        new_w = w - self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
        new_b = b - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
        return new_w, new_b


def get_optimizer(name: str, learning_rate: float = 0.01, **kwargs) -> Optimizer:
    name = name.lower().strip()
    if name in ('sgd', 'gradient_descent', 'gd'):
        return SGD(learning_rate=learning_rate)
    elif name in ('momentum', 'sgd_momentum'):
        beta = kwargs.get('momentum', 0.9)
        return Momentum(learning_rate=learning_rate, momentum=beta)
    elif name in ('adam',):
        return Adam(learning_rate=learning_rate)
    raise ValueError(f"Unknown optimizer: {name}")
