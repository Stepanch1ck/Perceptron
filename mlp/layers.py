import numpy as np
from typing import Optional, Dict, Any
from .activations import get_activation, Activation


class DenseLayer:
    def __init__(
        self,
        output_dim: int,
        input_dim: Optional[int] = None,
        activation: str = 'sigmoid',
        weights_initializer: str = 'heUniform'
    ):
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation_name = activation
        self.activation: Activation = get_activation(activation)
        self.weights_initializer = weights_initializer

        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None

        self.input_cache: Optional[np.ndarray] = None
        self.z_cache: Optional[np.ndarray] = None
        self.a_cache: Optional[np.ndarray] = None

        self.dw: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

        if self.input_dim is not None:
            self.initialize(self.input_dim)

    def initialize(self, input_dim: int, rng: Optional[np.random.RandomState] = None):
        self.input_dim = input_dim
        if rng is None:
            rng = np.random.RandomState()

        init = self.weights_initializer.lower().strip()
        fan_in = self.input_dim
        fan_out = self.output_dim

        if init in ('heuniform', 'he_uniform', 'kaiming_uniform'):
            limit = np.sqrt(6.0 / fan_in)
            self.weights = rng.uniform(-limit, limit, size=(fan_in, fan_out))
        elif init in ('henormal', 'he_normal', 'kaiming_normal'):
            std = np.sqrt(2.0 / fan_in)
            self.weights = rng.normal(0.0, std, size=(fan_in, fan_out))
        elif init in ('xavieruniform', 'glorotuniform', 'xavier_uniform', 'glorot_uniform'):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            self.weights = rng.uniform(-limit, limit, size=(fan_in, fan_out))
        elif init in ('xaviernormal', 'glorotnormal', 'xavier_normal', 'glorot_normal'):
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.weights = rng.normal(0.0, std, size=(fan_in, fan_out))
        elif init == 'normal':
            self.weights = rng.normal(0.0, 0.05, size=(fan_in, fan_out))
        else:
            limit = 1.0 / np.sqrt(fan_in)
            self.weights = rng.uniform(-limit, limit, size=(fan_in, fan_out))

        self.biases = np.zeros((1, fan_out), dtype=np.float64)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.input_cache = inputs
        self.z_cache = np.dot(inputs, self.weights) + self.biases
        self.a_cache = self.activation.forward(self.z_cache)
        return self.a_cache

    def backward(self, da: np.ndarray) -> np.ndarray:
        m = self.input_cache.shape[0]
        dz = da * self.activation.derivative(self.z_cache, self.a_cache)
        self.dw = np.dot(self.input_cache.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        return np.dot(dz, self.weights.T)

    def backward_from_dz(self, dz: np.ndarray) -> np.ndarray:
        m = self.input_cache.shape[0]
        self.dw = np.dot(self.input_cache.T, dz) / m
        self.db = np.sum(dz, axis=0, keepdims=True) / m
        return np.dot(dz, self.weights.T)

    def get_config(self) -> Dict[str, Any]:
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
            'weights_initializer': self.weights_initializer
        }
